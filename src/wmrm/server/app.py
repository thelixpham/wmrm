"""The HTTP surface.

Two rules the routes follow:

- **Nothing long-running happens inside a request.** `POST /jobs` records the job and
  returns; the work is a background task. A request that waited would hit the RunPod
  proxy's 100-second ceiling long before a job finished, and there is nothing useful to
  say at second 99 that cannot be said at second one.
- **`/live` is the only unauthenticated route, and it says nothing.** A platform health
  check needs to reach something without credentials; it does not need to learn what this
  machine is or what it is working on.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from .. import __version__
from . import reclaim
from .auth import require_token
from .config import GPU_ENGINES, Config
from .hooks import MezonNotifier, Notifier
from .models import (CancelAccepted, Health, JobList, JobSpec, JobStatus,
                     SubmitAccepted)
from .probe import free_gb, probe_machine, vram_free_mb
from .runner import JobRunner, _say
from .store import JobStore

STARTED_AT = time.time()


def _round_or_none(value: float | None) -> float | None:
    """Round for display, and let `None` stay `None` all the way to the JSON."""
    return None if value is None else round(value, 2)


def _describe(spec: JobSpec) -> str:
    """A job request in one line, safe to print.

    Deliberately not `spec.model_dump()`. A `kind: "url"` input carries a presigned URL,
    which is a credential with hours of life on it -- the status endpoint already leaves
    `spec` out for that reason, and a log file is no better a place for it than an HTTP
    response. So the URL is described, never shown.
    """
    src = spec.input
    if src.kind == "r2":
        where = f"r2:{src.key}"
    elif src.kind == "local":
        where = f"local:{src.path}"
    else:
        # Host only. Enough to tell "the presigned URL points somewhere unexpected" from
        # "the download failed", without putting the signature in the log.
        host = str(src.url).split("/")[2] if "//" in str(src.url) else "?"
        where = f"url:{src.filename or '?'} via {host}"

    out = "derived"
    if spec.output is not None:
        out = f"r2:{spec.output.key}" if spec.output.kind == "r2" else f"local:{spec.output.path}"

    bits = [f"engine={spec.engine}", f"in={where}", f"out={out}"]
    if spec.box is not None:
        bits.append(f"box={spec.box.x},{spec.box.y},{spec.box.w},{spec.box.h}")
    else:
        bits.append("box=detect")
    if spec.options:
        bits.append("opts=" + ",".join(f"{k}={v}" for k, v in sorted(spec.options.items())))
    bits.append("callback=" + ("yes" if spec.callbackBaseUrl else "NONE -- nothing will be reported"))
    return "  ".join(bits)


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config.from_env()
    cfg.ensure_dirs()

    # The docs are open; the API is not.
    #
    # They were behind the token at first, and that was a mistake with an obvious symptom:
    # a browser cannot attach an `Authorization` header to a URL you type, so /docs
    # answered `{"detail":"bad or missing bearer token"}` and the interactive docs could
    # not be reached by the only tool that renders them. Gating them was not buying much
    # either -- the routes are `/jobs` and `/health`, guessable in one attempt, and knowing
    # the shape of an API gets nobody past the token on the calls that matter.
    #
    # `WMRM_DOCS=off` removes them for a deployment that would rather not serve them.
    docs_off = os.environ.get("WMRM_DOCS", "").lower() in ("0", "off", "false", "no")
    app = FastAPI(title="wmrm pod", version=__version__,
                  docs_url=None if docs_off else "/docs",
                  redoc_url=None if docs_off else "/redoc",
                  openapi_url=None if docs_off else "/openapi.json",
                  description=(
                      "HTTP wrapper around `wmrm run` for a GPU pod.\n\n"
                      "Every route except `/live` needs `Authorization: Bearer "
                      "<WMRM_POD_TOKEN>`. Jobs are accepted with 202 and run in the "
                      "background for minutes to days; poll `GET /jobs/{jobId}` or "
                      "wait for the webhook."))
    app.state.cfg = cfg
    app.state.store = JobStore(cfg.state_dir)
    app.state.runner = JobRunner(cfg, app.state.store)
    app.state.machine = None            # filled on startup: see below
    app.state.sweeper = None            # the disk sweep, started below

    async def _sweep_forever() -> None:
        """Free what finished jobs no longer need, for as long as this process lives.

        The runner already reclaims a job's files the moment its output is in R2, so this
        loop is the backstop rather than the mechanism. It is what covers the cases the
        runner structurally cannot: a pod killed between the upload and the reclaim, a job
        that failed and has now waited out its retention window, and directories left by
        a job whose state file is gone.

        A restart runs it once immediately -- a pod comes back precisely after the kind of
        event that leaves rubbish behind, and waiting a quarter of an hour to notice means
        the first dispatch after a restart is the one refused for space.
        """
        while True:
            try:
                summary = await asyncio.to_thread(
                    reclaim.sweep, cfg, app.state.store,
                    say=lambda m: print(m, file=sys.stderr, flush=True))
                line = reclaim.describe(summary)
                if line:
                    print(line, file=sys.stderr, flush=True)
            except asyncio.CancelledError:
                return
            except Exception as exc:                       # noqa: BLE001
                # Housekeeping must never take the server with it.
                print(f"[reclaim] sweep failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            try:
                await asyncio.sleep(reclaim.SWEEP_EVERY)
            except asyncio.CancelledError:
                return

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        task = app.state.sweeper
        if task is not None and not task.done():
            task.cancel()

    @app.on_event("startup")
    async def _startup() -> None:
        # Probing here rather than lazily in /health: the answer is static, the probe
        # costs a subprocess, and a machine whose GPU is unusable should be visible as
        # such from the first health check rather than from the first failed job.
        app.state.machine = probe_machine()

        # A job still marked running belongs to a process that no longer exists -- this
        # server just started. Left alone it would hold a slot forever and never be
        # retried, so it is published as `interrupted` and the control plane is told.
        orphans = app.state.store.adopt_orphans()
        for rec in orphans:
            spec = rec.data.get("spec") or {}
            notifier = Notifier(
                base_url=spec.get("callbackBaseUrl"),
                pod_token=cfg.token,
                mezon=MezonNotifier(cfg.mezon_webhook_url, pod_id=cfg.pod_id),
            )
            await notifier.terminal(
                job_id=rec.job_id,
                dispatch_token=str(spec.get("dispatchToken") or ""),
                state="interrupted", outcome="interrupted", report=None,
                error=rec.data.get("error"),
            )

        # Started after adoption, not before: the jobs just marked `interrupted` are the
        # ones the sweep has to reason about, and a sweep that ran while they still said
        # `running` would skip every one of them.
        if reclaim.enabled():
            app.state.sweeper = asyncio.create_task(_sweep_forever())
        else:
            print("[reclaim] WMRM_RECLAIM is off: finished jobs keep their files, and "
                  "space comes back only from DELETE /jobs/{id}.",
                  file=sys.stderr, flush=True)

    # ------------------------------------------------------- validation errors --

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request,
                               exc: RequestValidationError) -> JSONResponse:
        """Flatten pydantic's errors by hand.

        In pydantic v2 an error raised from a model validator carries the original
        exception object in `ctx`, which is not JSON-serialisable -- and FastAPI's default
        handler then fails while trying to report the 422, turning a clear "you left out
        input.key" into an opaque 500.
        """
        return JSONResponse(status_code=400, content={"detail": _flatten(exc.errors())})

    # ---------------------------------------------------------------- liveness --

    @app.get("/live", include_in_schema=False)
    async def live() -> Response:
        return Response(status_code=200)

    # ------------------------------------------------------------------ health --

    @app.get("/health", dependencies=[Depends(require_token)],
             response_model=Health, summary="What this machine can do")
    async def health() -> dict[str, Any]:
        import anyio

        machine = app.state.machine or probe_machine()
        gpu = dict(machine["gpu"])
        # Fresh VRAM comes from nvidia-smi in a worker thread. With a single uvicorn
        # worker, a blocking subprocess on the event loop would stall every other request
        # on a machine that is already busy with a run.
        gpu["vramFreeMb"] = await anyio.to_thread.run_sync(vram_free_mb)

        store: JobStore = app.state.store
        runner: JobRunner = app.state.runner
        return {
            "ok": bool(machine["ffmpeg"]),
            "schema": 1,
            "wmrmVersion": __version__,
            "podId": cfg.pod_id,
            "gpu": gpu,
            "ffmpeg": machine["ffmpeg"],
            "ffprobe": machine["ffprobe"],
            "nvdec": machine["nvdec"],
            "engines": list(_usable_engines(machine)),
            "capacity": {"maxConcurrent": cfg.max_concurrent,
                         "running": len(store.live())},
            # `workDirFreeGb` is `null` when the filesystem will not answer, never a
            # stand-in number: the control plane shows this figure to a person deciding
            # whether to queue work, and a plausible-looking 0.0 is the one answer that
            # sends them looking for space they already have.
            # `heldGb` is what this pod's own jobs are sitting on. Reported because "the
            # volume is filling up" and "the volume is full of my own leftovers" send an
            # operator somewhere completely different, and until this existed telling them
            # apart needed a shell on the pod.
            "disk": {"workDirPath": str(cfg.work_dir),
                     "workDirFreeGb": _round_or_none(free_gb(cfg.work_dir)),
                     "minFreeGb": cfg.min_free_gb,
                     "heldGb": _round_or_none(
                         await anyio.to_thread.run_sync(
                             lambda: reclaim.held_bytes(cfg.work_dir) / (1024 ** 3))),
                     "retentionHours": cfg.retention_hours,
                     "reclaim": reclaim.enabled()},
            # Whether this pod can fetch and publish objects itself. The control plane
            # reads this to decide which input/output kinds it may ask for, instead of
            # finding out from a failed job.
            "r2": {"configured": cfg.r2_configured,
                   "bucket": cfg.r2_bucket,
                   "workers": cfg.r2_workers,
                   "reason": cfg.r2_reason()},
            "currentJobIds": runner.running_ids(),
            "uptimeSeconds": int(time.time() - STARTED_AT),
        }

    # -------------------------------------------------------------------- jobs --

    @app.post("/jobs", status_code=202, dependencies=[Depends(require_token)],
              summary="Submit a job", response_model=None,
              responses={
                  202: {"description": "Accepted; the run happens in the background"},
                  200: {"description": "This jobId is already known -- not started twice"},
                  400: {"description": "Bad payload, unusable path, or an engine/"
                                       "credential this pod does not have"},
                  409: {"description": "Pod is already at its concurrency limit"},
                  507: {"description": "Not enough free disk for this job"},
              })
    async def submit(spec: JobSpec) -> Any:
        """Accept a job and return immediately.

        Typed rather than parsed from a raw `Request`, so `JobSpec` and its nested models
        appear in the schema -- the reason for generating OpenAPI at all is that the
        client cannot drift from the server, and a hand-parsed body gives up exactly that.
        """
        store: JobStore = app.state.store
        runner: JobRunner = app.state.runner

        # What was asked for, on the pod's own console, before anything can refuse it.
        #
        # A refusal used to leave nothing behind but a status code in the access log -- and
        # `507` on its own reads like a problem with the request, when it is a fact about
        # this machine's disk. Both halves are needed to tell those apart: what came in, and
        # why it was turned away.
        _say(spec.jobId, f"submit: {_describe(spec)}")

        def refuse(status: int, detail: str) -> HTTPException:
            _say(spec.jobId, f"REFUSED {status}: {detail}")
            return HTTPException(status_code=status, detail=detail)

        existing = store.get(spec.jobId)
        if existing is not None:
            # Idempotent: the control plane may retry a dispatch whose response was lost,
            # and starting the same job twice on one machine means two processes writing
            # one output.
            return JSONResponse(
                status_code=200,
                content={"accepted": False, "jobId": spec.jobId,
                         "state": existing.state,
                         "reason": "job already known to this pod",
                         "workDir": existing.data.get("workDir")},
            )

        machine = app.state.machine or {}
        if spec.engine in GPU_ENGINES and not (machine.get("gpu") or {}).get("cuda"):
            # Refused rather than accepted and run at ~400x the cost. Measured: ProPainter
            # on six CPU cores is 0.27 fps, about 1.8 hours per minute of 1080p.
            raise refuse(400,
                         f"engine '{spec.engine}' needs CUDA and this pod reports none")

        if runner.at_capacity():
            raise refuse(409, f"pod is busy ({cfg.max_concurrent} concurrent job(s) max)")

        # `mkdir` first, for the same reason `require_space` does it: measuring a directory
        # that does not exist yet raises, and "not created yet" is not "out of space".
        cfg.work_dir.mkdir(parents=True, exist_ok=True)
        free = free_gb(cfg.work_dir)
        if free is None:
            # Said out loud rather than passed over. A guard that silently stops guarding is
            # worse than one that is off, because the log still reads as though it ran.
            _say(spec.jobId, f"disk: {cfg.work_dir} does not report free space "
                             f"-- the {cfg.min_free_gb:g} GiB floor is not applied. "
                             f"A full disk will surface as a failed job instead.")
        elif free < cfg.min_free_gb:
            # Sweep before refusing, not on a timer alone. This is the one moment the
            # answer matters: the space held by finished jobs is worthless and the request
            # in hand is real, so turning it away while sitting on 26 GB of delivered
            # output -- because the next tick is eleven minutes off -- is a refusal nobody
            # can explain. The sweep is bounded and idempotent; the cost of one here is a
            # few unlinks.
            summary = await asyncio.to_thread(
                reclaim.sweep, cfg, store,
                say=lambda m: print(m, file=sys.stderr, flush=True))
            line = reclaim.describe(summary)
            if line:
                _say(spec.jobId, line + " -- looking again before refusing")
            # Explicitly against None: 0.0 GiB free is a real answer, and `or` would
            # discard it in favour of the stale reading.
            again = free_gb(cfg.work_dir)
            if again is not None:
                free = again
        if free is not None and free < cfg.min_free_gb:
            raise refuse(507, f"only {free:.1f} GiB free in {cfg.work_dir}, "
                              f"need {cfg.min_free_gb:.0f} GiB "
                              + ("-- and finished jobs have already been swept, so this is "
                                 "space something else is using"
                                 if reclaim.enabled()
                                 else "-- WMRM_RECLAIM is off, so finished jobs are still "
                                      "holding their files")
                              + " (raise or lower the floor with WMRM_MIN_FREE_GB)")

        # Refused here rather than discovered by the job. A pod without credentials
        # cannot fetch a key or publish a result, and finding that out after the job is
        # accepted turns a clear 400 into a mysterious failure on someone else's machine.
        #
        # An absent `output` is not a request for R2: the pod derives one, and falls back to
        # leaving the file in the work directory when there are no credentials. Only an
        # explicit `kind: "r2"` is a demand.
        wants_r2 = spec.input.kind == "r2" or (
            spec.output is not None and spec.output.kind == "r2"
        )
        if wants_r2 and not cfg.r2_configured:
            raise HTTPException(
                status_code=400,
                detail=f"this pod has no R2 credentials, so it cannot use "
                       f"kind='r2' ({cfg.r2_reason()})")

        try:
            runner.resolve_input(spec)        # validates a local path before accepting
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        notifier = Notifier(
            base_url=spec.callbackBaseUrl,
            pod_token=cfg.token,
            mezon=MezonNotifier(cfg.mezon_webhook_url, pod_id=cfg.pod_id),
        )
        rec = runner.submit(spec, notifier)
        return SubmitAccepted(jobId=rec.job_id, state=rec.state,
                              workDir=str(rec.data["workDir"]))

    @app.get("/jobs", dependencies=[Depends(require_token)],
             response_model=JobList, summary="Jobs this pod knows about")
    async def list_jobs(live_only: bool = Query(default=False, alias="live")) -> dict:
        """Used by the control plane to reconcile: a job running here that it did not
        dispatch is burning GPU time for nobody."""
        store: JobStore = app.state.store
        records = store.live() if live_only else list(store.all())
        return {"jobs": [r.public() for r in records], "podId": cfg.pod_id}

    @app.get("/jobs/{job_id}", dependencies=[Depends(require_token)],
             response_model=JobStatus, summary="One job's state",
             responses={404: {"description": "No such job on this pod"}})
    async def get_job(job_id: str) -> dict:
        rec = app.state.store.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="no such job on this pod")
        return rec.public()

    @app.post("/jobs/{job_id}/cancel", status_code=202,
              dependencies=[Depends(require_token)],
              response_model=CancelAccepted, summary="Cancel a job",
              responses={404: {"description": "No such job on this pod"}})
    async def cancel_job(job_id: str, request: Request) -> Any:
        purge = False
        try:
            body = await request.json()
            purge = bool((body or {}).get("purge"))
        except Exception:                       # noqa: BLE001 -- an empty body is fine
            pass
        rec = await app.state.runner.cancel(job_id, purge=purge)
        if rec is None:
            raise HTTPException(status_code=404, detail="no such job on this pod")
        return CancelAccepted(jobId=job_id, state=rec.state, purge=purge)

    @app.get("/jobs/{job_id}/log", response_class=PlainTextResponse,
             dependencies=[Depends(require_token)])
    async def job_log(job_id: str, tail: int = Query(default=200, ge=1, le=5000)) -> str:
        """The last N lines of the run's stderr. For a human, deliberately not parsed.

        Nothing in the protocol reads this. Outcomes travel in the report file, and
        progress is counted from the parts directory -- so the log is free to stay a log.
        """
        rec = app.state.store.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="no such job on this pod")
        path = Path(rec.data.get("workDir") or "") / "run.log"
        if not path.is_file():
            # Says which of the two it is, because they lead somewhere different: a job that
            # has not written anything yet, or a pod that keeps no per-job file at all.
            # Returning "" for both sends whoever is debugging to look for a file that was
            # never going to exist.
            if os.environ.get("WMRM_RUN_LOG", "0").lower() in ("0", "off", "false", "no"):
                return ("No per-job log file on this pod: WMRM_RUN_LOG is off, which is the "
                        "default. The run's output goes to the server's own output instead, "
                        "so read it there -- or start the server with "
                        "`wmrm serve 2>&1 | tee -a /workspace/wmrm-serve.log` to keep it. "
                        "WMRM_RUN_LOG=1 restores a run.log per job.")
            return ""
        with open(path, "rb") as fh:
            # Read from the end: these files reach hundreds of megabytes on a long run.
            fh.seek(0, 2)
            size = fh.tell()
            window = min(size, 256 * 1024)
            fh.seek(size - window)
            data = fh.read().decode("utf-8", "replace")
        return "\n".join(data.splitlines()[-tail:])

    @app.delete("/jobs/{job_id}", status_code=204,
                dependencies=[Depends(require_token)])
    async def delete_job(job_id: str) -> Response:
        store: JobStore = app.state.store
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="no such job on this pod")
        if rec.is_live:
            raise HTTPException(status_code=409,
                                detail=f"job is {rec.state}; cancel it first")
        import shutil
        shutil.rmtree(rec.data.get("workDir") or "", ignore_errors=True)
        store.delete(job_id)
        return Response(status_code=204)

    return app


def _flatten(errors: list[dict[str, Any]], limit: int = 8) -> list[dict[str, str]]:
    """Pydantic errors as plain strings, safe to serialise."""
    return [
        {"loc": ".".join(str(p) for p in e.get("loc", ())),
         "msg": str(e.get("msg", "")),
         "type": str(e.get("type", ""))}
        for e in errors[:limit]
    ]


def _usable_engines(machine: dict[str, Any]) -> tuple[str, ...]:
    """Engines this machine can actually run, not the ones it knows the names of.

    Reporting `video` on a CPU-only pod would invite the control plane to dispatch a job
    that technically starts and then takes days.
    """
    from .config import ENGINES

    cuda = bool((machine.get("gpu") or {}).get("cuda"))
    return tuple(e for e in ENGINES if cuda or e not in GPU_ENGINES)
