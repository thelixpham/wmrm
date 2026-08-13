"""Running one job: spawn `wmrm run`, watch it, report what happened.

Three things here are load-bearing and none are obvious.

**The child gets its own process group.** `wmrm run` spawns two ffmpeg children -- a
decoder reading the source and an encoder writing the output. Signalling only the Python
process leaves both alive, holding file handles and, for ProPainter, writing into the
parts directory that the next attempt will try to reuse. `start_new_session=True` plus
`killpg` is what makes "stop" mean stop.

**Cancellation is decided here, not in the child.** A process cannot tell who sent it
SIGTERM: an operator cancelling and a pod being restarted look identical from the inside.
The original design passed `WMRM_CANCEL_REQUESTED` in the environment, which cannot work
-- the environment is fixed when the process starts, and a cancellation arrives hours
later. So the child always reports `interrupted`, and this module rewrites it to
`canceled` when it was the one that asked. It is the only party that knows.

**Progress is counted, not parsed.** For ProPainter the finished parts are files on disk,
which is what the README tells an operator to watch. Every other engine reports no
progress at all: parsing the log for it would tie the wire format to log text that exists
for humans. `null` is honest; a number scraped out of a banner is not.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .config import Config
from .hooks import Notifier
from .models import JobSpec
from .store import JobRecord, JobStore
from .transfer import (NotEnoughSpace, TransferError, abort_r2, download, pull_r2,
                       push_r2, require_space, stat_r2)

#: Grace between asking the process group to stop and insisting.
TERM_GRACE = 30.0


def _say(job_id: str, message: str) -> None:
    """One line to the server's own log, so the console is not silent for hours.

    The run's own output goes to a file rather than here -- a nine-hour job produces far
    more than is sensible to keep in memory or scroll past. But that left the console
    showing the R2 download (which happens in this process) and then nothing at all, so
    there was no way to tell a working job from a wedged one without going to look for a
    file. These lines are the lifecycle only: what state it moved to, and how it ended.
    """
    print(f"[job {job_id}] {message}", file=sys.stderr, flush=True)

#: Report outcome -> the state this pod publishes.
#
#: `upload_failed` maps to `failed` but is worth its own outcome: the run produced a file
#: that passed verification and only the delivery went wrong, so retrying costs one
#: upload rather than the hours of GPU time that made the file.
_OUTCOME_STATE = {
    "ok": "succeeded",
    "coverage_inconclusive": "needs_review",
    "interrupted": "interrupted",
    "canceled": "canceled",
}


def _state_for(outcome: str) -> str:
    return _OUTCOME_STATE.get(outcome, "failed")


class JobRunner:
    """Owns the lifecycle of the jobs on this pod."""

    def __init__(self, cfg: Config, store: JobStore):
        self.cfg = cfg
        self.store = store
        self._tasks: dict[str, asyncio.Task] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        # One JobRecord instance per in-flight job, shared by whoever touches it.
        #
        # Without this, `cancel` and the run loop each hold their own object loaded from
        # the same file, and the last one to write wins: cancel sets `cancelRequested`,
        # the run loop then saves its stale copy over it, and the job reports
        # `interrupted` -- so the control plane retries a job somebody deliberately
        # stopped. Reloading everywhere would work too, but it makes every read a place
        # to forget.
        self._records: dict[str, JobRecord] = {}

    def _record(self, job_id: str) -> JobRecord | None:
        """The live instance if there is one, otherwise whatever is on disk."""
        rec = self._records.get(job_id)
        if rec is not None:
            return rec
        return self.store.get(job_id)

    # -- capacity -------------------------------------------------------------- #

    def running_ids(self) -> list[str]:
        return [r.job_id for r in self.store.live()]

    def at_capacity(self) -> bool:
        return len(self.store.live()) >= self.cfg.max_concurrent

    # -- submission ------------------------------------------------------------ #

    def resolve_input(self, spec: JobSpec) -> Path | None:
        """The source's path when it is already on disk, or None when it has to be fetched.

        A caller-supplied path is resolved and then required to be inside the configured
        root. Resolving first is the point: checking the string before following symlinks
        accepts `root/../../etc/passwd` and a symlink out of the tree equally.

        Written as "only `local` has a path" rather than "everything except `url` has a
        path", which is how it started and how it broke: adding the `r2` kind left this
        function unchanged, so an r2 job fell into the local branch and was rejected for
        not having WMRM_LOCAL_INPUT_ROOT set -- an error about a field it had not sent.
        This way a kind added later returns None and fails somewhere that names it, instead
        of being silently treated as a local path.
        """
        if spec.input.kind != "local":
            return None
        root = self.cfg.local_input_root
        if root is None:
            # Says what to do, because this is exactly the moment someone needs it -- and
            # the first suggestion is the one that is almost always right, since a pod with
            # R2 credentials has no reason to read from its own disk.
            raise ValueError(
                "input.kind='local' needs WMRM_LOCAL_INPUT_ROOT set on this pod, and it "
                "is not. Either use input.kind='r2' with the object key "
                + ("(this pod has R2 credentials, so that is the normal path), "
                   if self.cfg.r2_configured
                   else "(needs R2_* on this pod), ")
                + "or restart with WMRM_LOCAL_INPUT_ROOT=/workspace/wmrm to allow local "
                "files from there. The root exists so a path cannot escape it: without "
                "one, anyone holding the pod token could read any file the pod can.")
        candidate = Path(spec.input.path or "").expanduser()
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            raise ValueError(f"input path does not exist: {candidate}")
        if root not in resolved.parents and resolved != root:
            raise ValueError(
                f"input path is outside WMRM_LOCAL_INPUT_ROOT ({root}): {resolved}")
        if not resolved.is_file():
            raise ValueError(f"input path is not a file: {resolved}")
        return resolved

    def submit(self, spec: JobSpec, notifier: Notifier) -> JobRecord:
        job_dir = self.cfg.job_dir(spec.jobId)
        job_dir.mkdir(parents=True, exist_ok=True)
        rec = self.store.create(
            job_id=spec.jobId,
            spec=json.loads(spec.model_dump_json(by_alias=True)),
            work_dir=str(job_dir),
        )
        self._records[spec.jobId] = rec
        task = asyncio.create_task(self._run(spec, rec, notifier))
        self._tasks[spec.jobId] = task
        task.add_done_callback(lambda _t, jid=spec.jobId: self._tasks.pop(jid, None))
        return rec

    # -- cancellation ---------------------------------------------------------- #

    async def cancel(self, job_id: str, *, purge: bool = False) -> JobRecord | None:
        rec = self._record(job_id)
        if rec is None:
            return None
        # Recorded before signalling. If this pod dies between the two, the flag is what
        # tells the next process that the job was cancelled rather than interrupted.
        rec.set(cancelRequested=True)
        if rec.state in ("succeeded", "failed", "canceled", "needs_review", "interrupted"):
            return rec
        rec.set_state("canceling" if rec.is_live else rec.state, phase="canceling")
        await self._kill(job_id)
        if purge:
            self._purge(rec)
        return self._record(job_id) or rec

    async def _kill(self, job_id: str) -> None:
        proc = self._procs.get(job_id)
        if proc is None or proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            return
        # Give it the chance to write its report and release its lock; insist afterwards.
        deadline = time.monotonic() + TERM_GRACE
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            await asyncio.sleep(0.5)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass

    def _purge(self, rec: JobRecord) -> None:
        import shutil
        try:
            shutil.rmtree(rec.data.get("workDir") or "", ignore_errors=True)
        except OSError:                                # pragma: no cover
            pass

    # -- the run --------------------------------------------------------------- #

    async def _run(self, spec: JobSpec, rec: JobRecord, notifier: Notifier) -> None:
        job_dir = Path(rec.data["workDir"])
        report_path = job_dir / "report.json"
        log_path = job_dir / "run.log"
        hb: asyncio.Task | None = None
        try:
            _say(spec.jobId, f"accepted: engine={spec.engine} input={spec.input.kind}"
                             f"{'' if spec.box else ' (no box -- the pod will detect one)'}")
            src = await self._stage_input(spec, rec, job_dir, notifier)
            dst, output_key_plan = self._plan_output(spec, job_dir, src)
            _say(spec.jobId,
                 f"source ready: {src.name} -> {dst.name}"
                 + (f", publishing to {output_key_plan}" if output_key_plan
                    else " (staying on disk -- no R2 output)"))
            # Recorded now so `GET /jobs/{id}` can say where the result is going before it
            # gets there, and so a cleanup knows what to abort if this run dies.
            rec.set(plannedOutputKey=output_key_plan, outputPath=str(dst))

            rec.set_state("running", phase="running")
            hb = asyncio.create_task(self._heartbeat(spec, rec, dst, notifier))
            started = time.time()
            code = await self._spawn(spec, rec, src, dst, report_path, log_path)
            _say(spec.jobId, f"wmrm run exited {code} after {time.time() - started:.0f}s")

            report = self._read_report(report_path)
            outcome = self._decide_outcome(rec, report, code)

            output_key = None
            if outcome == "ok":
                try:
                    output_key = await self._deliver_output(
                        spec, rec, dst, output_key_plan)
                except (TransferError, Exception) as exc:      # noqa: BLE001
                    # The pixels are fine, the delivery is not. Reported as its own
                    # outcome so the control plane retries the upload rather than the
                    # nine hours of GPU work that produced the file.
                    outcome = "upload_failed"
                    report = report or {}
                    report["error"] = {"code": outcome, "message": str(exc)}
                    await self._abort_upload(spec)

            state = _state_for(outcome)
            rec.set_state(
                state,
                outcome=outcome,
                report=report,
                outputKey=output_key,
                box=(report or {}).get("box"),
                error=(report or {}).get("error") or (
                    None if outcome == "ok"
                    else {"code": outcome, "message": f"wmrm run exited {code}"}),
                pid=None,
            )
            if hb is not None:
                hb.cancel()

            detail = (rec.data.get("error") or {}).get("message") or ""
            _say(spec.jobId,
                 f"{state} (outcome={outcome})"
                 + (f" -> {output_key}" if output_key else "")
                 + (f" -- {detail[:160]}" if detail else ""))

            sent = await notifier.terminal(
                job_id=spec.jobId, dispatch_token=spec.dispatchToken,
                state=state, outcome=outcome, report=report,
                error=rec.data.get("error"), output_key=output_key,
            )
            if not sent and notifier.enabled:
                # Worth saying out loud: the work is finished and correct, but whoever
                # dispatched it does not know, so they will see a stalled job until they poll.
                _say(spec.jobId, "WARNING: could not report the result back. "
                                 "The state above is on disk; the caller has not been told.")

        except asyncio.CancelledError:                  # pragma: no cover
            raise
        except (NotEnoughSpace, TransferError, ValueError) as exc:
            await self._fail(spec, rec, notifier, hb,
                             code="input_error" if isinstance(exc, (ValueError, TransferError))
                             else "disk_full",
                             message=str(exc))
        except Exception as exc:                        # noqa: BLE001
            await self._fail(spec, rec, notifier, hb, code="internal",
                             message=f"{type(exc).__name__}: {exc}")
        finally:
            if hb is not None and not hb.done():
                hb.cancel()
            self._procs.pop(spec.jobId, None)
            # Dropped only now the job is terminal: until then `cancel` and `/jobs/{id}`
            # must see the same object this loop is writing.
            self._records.pop(spec.jobId, None)

    async def _abort_upload(self, spec: JobSpec) -> None:
        """Drop a half-finished multipart upload for this job's output key.

        Best effort. The parts of an incomplete upload are stored and billed while
        appearing in no listing, so if this is not done nothing else will notice them --
        but failing the job because the cleanup failed would be worse.
        """
        rec = self._record(spec.jobId)
        key = (rec.data.get("plannedOutputKey") if rec else None) or (
            spec.output.key if spec.output is not None and spec.output.kind == "r2" else None
        )
        if not key:
            return
        bucket = spec.output.bucket if spec.output is not None else None
        try:
            await abort_r2(key, bucket=bucket)
        except Exception:                                # noqa: BLE001
            pass

    async def _fail(self, spec: JobSpec, rec: JobRecord, notifier: Notifier,
                    hb: asyncio.Task | None, *, code: str, message: str) -> None:
        if hb is not None and not hb.done():
            hb.cancel()
        outcome = "canceled" if rec.cancel_requested else code
        state = _state_for(outcome)
        rec.set_state(state, outcome=outcome,
                      error={"code": outcome, "message": message}, pid=None)
        await notifier.terminal(job_id=spec.jobId, dispatch_token=spec.dispatchToken,
                                state=state, outcome=outcome, report=None,
                                error={"code": outcome, "message": message})

    async def _stage_input(self, spec: JobSpec, rec: JobRecord, job_dir: Path,
                           notifier: Notifier) -> Path:
        local = self.resolve_input(spec)
        if local is not None:
            # Already on the volume. Not copied: these are tens of gigabytes, and a copy
            # buys nothing when the run only reads it.
            return local

        if spec.input.kind == "r2":
            # Ask R2 for the real size rather than trusting `sizeBytes` from the
            # dispatcher: it is what the space check has to be right about, and being
            # wrong means running out of disk at hour seven.
            size, key = await stat_r2(spec.input.key or "", bucket=spec.input.bucket)
            require_space(job_dir, size)
            rec.set_state("downloading", phase="downloading",
                          progress={"stage": "download", "bytesTotal": size})
            dest = job_dir / Path(key).name
            return await pull_r2(key, dest, bucket=spec.input.bucket,
                                 workers=self.cfg.r2_workers, progress=False)

        size = spec.input.sizeBytes or 0
        require_space(job_dir, size)
        rec.set_state("downloading", phase="downloading")
        name = spec.input.filename or "input.mp4"
        dest = job_dir / Path(name).name

        last = [0.0]

        def _tick(done: int, total: int | None) -> None:
            now = time.time()
            if now - last[0] < 5.0:
                return
            last[0] = now
            rec.set(progress={"bytesDone": done, "bytesTotal": total,
                              "stage": "download"})

        return await download(spec.input.url or "", dest,
                              expected_size=spec.input.sizeBytes or None,
                              on_progress=_tick)

    async def _deliver_output(self, spec: JobSpec, rec: JobRecord, dst: Path,
                              key: str | None) -> str | None:
        """Publish the finished file, if there is anywhere to publish it. Returns the key.

        Only reached when the run succeeded. An upload attempted after a failure would put
        out a file that did not pass verification, which is the one thing the acceptance
        checks exist to prevent.
        """
        if key is None:
            return None
        if not dst.is_file():
            raise TransferError(f"the run reported success but {dst} is not there")
        rec.set_state("uploading", phase="uploading")
        bucket = spec.output.bucket if spec.output is not None else None
        return await push_r2(dst, key, bucket=bucket,
                             workers=self.cfg.r2_workers, progress=False)

    def _plan_output(self, spec: JobSpec, job_dir: Path, src: Path) -> tuple[Path, str | None]:
        """Where to write it, and the R2 key to publish it under (or None).

        Both derived here, in one place. An absent `output` is the normal case for a queue:
        the caller has nothing to add, since this side already knows the job id and the
        source's name. Requiring it would mean the same rule implemented twice, and the day
        the two disagree the result lands somewhere nobody is looking.
        """
        from .models import OUTPUT_SUFFIX

        if spec.output is not None and spec.output.kind == "local":
            path = Path(spec.output.path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            return path, None

        if spec.output is not None:                      # kind == "r2", key pinned
            key = spec.output.key
            # Named after the key, so the file on disk and the object it becomes are
            # obviously the same thing to anyone looking at both.
            return job_dir / Path(key).name, key

        # Nothing asked for. Derive it.
        name = f"{src.stem}{OUTPUT_SUFFIX}{src.suffix or '.mp4'}"
        local = job_dir / name
        if not self.cfg.r2_configured:
            # No credentials, so there is nowhere to publish to. The file stays here and the
            # path is reported -- which is the useful answer, not an error.
            return local, None
        return local, f"output/{spec.jobId}/{name}"

    async def _spawn(self, spec: JobSpec, rec: JobRecord, src: Path, dst: Path,
                     report_path: Path, log_path: Path) -> int:
        argv = spec.argv(input_path=str(src), output_path=str(dst),
                         report_path=str(report_path))
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")

        # The run's output goes to a file **and** to this server's log, one line at a time.
        #
        # It used to go only to the file, on the reasoning that a nine-hour run produces far
        # more log than is sensible to hold in memory. The first half of that is right; the
        # conclusion was not. Nothing has to be *held* to be shown -- a line is read,
        # written, echoed and dropped, so the memory cost is one line no matter how long the
        # run is. What the file-only version actually bought was a console that went silent
        # for hours, where a working job and a wedged one look identical.
        #
        # **The per-job file is off by default**, and the console echo is on.
        #
        # It was the other way round, and the argument for the file was `GET /jobs/{id}/log`
        # -- reading a job's log without a shell on the pod. That argument does not survive
        # how this is actually operated: logs get read by going into the pod, where the
        # server's own output is right there, so a second copy per job was two places to
        # look and one of them nobody opens.
        #
        # What is genuinely lost is durability across a restart, and the answer to that is
        # one log for the pod rather than one per job:
        #
        #     wmrm serve 2>&1 | tee -a /workspace/wmrm-serve.log
        #
        # WMRM_RUN_LOG=1 brings the per-job files back; WMRM_ECHO_RUN=0 silences the console.
        #
        # The pipe is read by a dedicated thread. That matters: a pipe nobody drains fills
        # up and blocks the child, which is the failure the file-only version was avoiding.
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
            cwd=str(dst.parent),
            start_new_session=True,          # its own process group: see module docstring
        )
        self._procs[spec.jobId] = proc
        rec.set(pid=proc.pid)

        def flag(name: str, default: str) -> bool:
            return os.environ.get(name, default).lower() not in ("0", "off", "false", "no")

        echo = flag("WMRM_ECHO_RUN", "1")     # on: the console is where logs get read
        to_file = flag("WMRM_RUN_LOG", "0")   # off: a second copy nobody opens
        # Short tag rather than the whole job id: with more than one job at a time the
        # console interleaves, and 40 characters of prefix on every line is its own problem.
        tag = f"[{spec.jobId[-6:]}] "

        def pump() -> int:
            # nullcontext rather than a branch around the loop, so the reading half exists
            # in exactly one place -- draining the pipe is the part that must not be
            # accidentally skipped, since a full pipe blocks the child.
            from contextlib import nullcontext

            opened = open(log_path, "ab") if to_file else nullcontext(None)
            with opened as log:
                if log is not None:
                    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                              f"{' '.join(argv)}\n".encode())
                    log.flush()
                assert proc.stdout is not None
                for raw in proc.stdout:
                    if log is not None:
                        log.write(raw)
                        log.flush()         # so `tail -f` keeps up with a long run
                    if echo:
                        sys.stderr.write(tag + raw.decode("utf-8", "replace"))
                        sys.stderr.flush()
            return proc.wait()

        # Off the event loop, so a running job never blocks the API.
        return await asyncio.to_thread(pump)

    def _read_report(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def _decide_outcome(self, rec: JobRecord, report: dict[str, Any] | None,
                        code: int) -> str:
        """What happened, in this pod's judgement.

        The report is trusted for everything except cancellation. `wmrm run` cannot know
        whether the SIGTERM it received was a person cancelling or its pod being
        restarted, so it says `interrupted`; this pod knows which, and the difference
        decides whether the control plane retries or stops.
        """
        outcome = (report or {}).get("outcome")
        if rec.cancel_requested and outcome in (None, "interrupted", "internal"):
            return "canceled"
        if outcome:
            return str(outcome)
        # No report at all. The process died before it could write one -- killed, or
        # gone before it started.
        if rec.cancel_requested:
            return "canceled"
        if code and code < 0:
            return "interrupted"          # negative == died on a signal
        return "internal"

    # -- heartbeat ------------------------------------------------------------- #

    async def _heartbeat(self, spec: JobSpec, rec: JobRecord, dst: Path,
                         notifier: Notifier) -> None:
        """Say we are alive, on a fixed interval, for every engine.

        This is the only signal the control plane has that a long job is progressing at
        all -- and it has to exist for engines with no countable progress, or they look
        indistinguishable from a dead pod.
        """
        interval = max(5, spec.heartbeatEverySeconds)
        # Progress is only echoed to the console when it moves, and at most once a minute.
        # A line every 30 seconds saying the same thing is how a log stops being read.
        last_echo, last_done = 0.0, -1
        while True:
            try:
                await asyncio.sleep(interval)
                progress = self._progress(spec, dst)
                if progress is not None:
                    rec.set(progress=progress)
                    done = int(progress.get("partsDone") or 0)
                    if done != last_done and time.time() - last_echo > 60:
                        total, eta = progress.get("partsTotal"), progress.get("etaSeconds")
                        _say(spec.jobId,
                             f"{done}/{total if total else '?'} parts"
                             + (f", ~{eta // 60} min left" if eta else ""))
                        last_echo, last_done = time.time(), done
                fresh = self._record(spec.jobId) or rec
                await notifier.heartbeat(
                    job_id=spec.jobId, dispatch_token=spec.dispatchToken,
                    state=fresh.state, phase=str(fresh.data.get("phase") or fresh.state),
                    progress=progress,
                )
            except asyncio.CancelledError:
                return
            except Exception:                          # noqa: BLE001  pragma: no cover
                # Never let reporting kill the job it reports on.
                continue

    def _progress(self, spec: JobSpec, dst: Path) -> dict[str, Any] | None:
        """Finished parts, for ProPainter only.

        `<output>.parts/` is created solely by the ProPainter path, and the README already
        tells operators to count it. Every other engine gets `null` rather than a number
        scraped from a log line.
        """
        if spec.engine != "video":
            return None
        parts_dir = dst.with_name(dst.name + ".parts")
        if not parts_dir.is_dir():
            return None
        done = len(list(parts_dir.glob("part-*.mp4")))
        total = None
        # The manifest is written by video.py and knows how many parts the plan has.
        try:
            manifest = json.loads((parts_dir / "manifest.json").read_text())
            frames = manifest.get("frames") or manifest.get("nframes")
            per_part = manifest.get("part_frames") or spec.options.get("ppPart") or 3600
            if frames and per_part:
                total = int(math.ceil(int(frames) / int(per_part)))
        except (OSError, ValueError, TypeError, ZeroDivisionError):
            pass

        started = (self._record(spec.jobId) or JobRecord(Path(), {})).data.get("startedAt")
        eta = None
        if started and done >= 2 and total:
            rate = done / max(1e-6, time.time() - float(started))
            if rate > 0:
                eta = int((total - done) / rate)
        return {"stage": "propainter", "partsDone": done, "partsTotal": total,
                "etaSeconds": eta}
