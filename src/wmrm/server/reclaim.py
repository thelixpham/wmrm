"""Getting the disk back when a job is over.

A job leaves two very large files in its work directory: the source it downloaded and the
output it produced. Nothing used to remove either. That is survivable for one job and
fatal for a queue -- a 4K feature is ~18 GB in and ~9 GB out, so three of them hold 80 GB
of the volume for work nobody will look at again, and the next dispatch is refused with
507 for want of space that is entirely rubbish. The pod this was written for had 26 GB
sitting in one job directory whose output had been in R2 for half an hour.

Four rules, and each one is a thing that would otherwise go wrong.

**The output goes only once R2 has it.** `outputKey` is set by the runner after `push_r2`
returns, so it is the record that the bytes exist somewhere else. Reclaiming on
`state == "succeeded"` alone would delete the only copy of a file for a job whose output
was deliberately left on disk (`output.kind: "local"`, or a pod with no credentials).

**Only what is inside the job's own directory.** A `local` input and a `local` output both
live outside it, and neither is this pod's to delete -- an operator pointed `input.path` at
a file they still want. Containment is checked against `work_dir` after resolving, so a
hand-edited state file cannot aim this at `/`.

**The small records stay.** `report.json`, the detect preview and the preset are a few
megabytes together and they are the answer to "what box was this actually made with",
asked days later when an output looks wrong. Freeing them buys nothing and costs the only
evidence. Anything over `KEEP_MAX_BYTES` is not a record, whatever its extension.

**An interrupted job is the one worth waiting on.** Its `<output>.parts/` directory is what
`--resume` picks up, so deleting it converts a retry that costs minutes into one that costs
the whole nine hours. Those wait out `WMRM_RETENTION_HOURS`, and under the free-space floor
they are still the last thing to go.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from .config import Config
from .probe import free_gb
from .store import JobRecord, JobStore

#: How often the background sweep runs. Not tunable: the sweep is cheap and idempotent,
#: and the thing it protects against -- a volume filling up between two long jobs -- is
#: not sensitive to whether it noticed within a minute or within fifteen.
SWEEP_EVERY = 900.0

#: Extensions kept when a job's bytes are reclaimed. Records, not results.
KEEP_SUFFIXES = (".json", ".png", ".jpg", ".txt", ".log", ".csv")

#: Above this, a file is not a record no matter what it is called. `run.log` from a
#: nine-hour ProPainter run reaches hundreds of megabytes, which is not evidence anyone
#: reads -- and the console copy is the one that actually gets read anyway.
KEEP_MAX_BYTES = 64 * 1024 * 1024

#: A directory in the work dir that no job record claims has to be at least this old
#: before it is removed. `submit` creates the directory and then the record, so a brand
#: new job looks exactly like an abandoned one for the microsecond in between.
ORPHAN_DIR_MIN_AGE = 3600.0

#: Order the free-space emergency deletes in. Earlier is cheaper to lose: a delivered
#: output is already in R2, while an interrupted job's parts are hours of GPU time that
#: `--resume` would otherwise not have to spend again.
EMERGENCY_ORDER = ("succeeded", "canceled", "failed", "needs_review", "interrupted")


def enabled() -> bool:
    """Whether this pod reclaims anything on its own.

    `WMRM_RECLAIM=off` turns off both halves -- the reclaim after a delivery and the
    sweep -- and leaves `DELETE /jobs/{id}` as the only way space comes back. One switch
    rather than two, because "clean up after jobs but not really" is not a position anyone
    holds, and a pod where only half of it is off is a pod that fills up more slowly.
    """
    return os.environ.get("WMRM_RECLAIM", "1").lower() not in ("0", "off", "false", "no")


def _du(path: Path) -> int:
    """Bytes held by a file or a directory tree. Symlinks counted, never followed."""
    try:
        st = path.lstat()
    except OSError:
        return 0
    if not path.is_dir() or path.is_symlink():
        return st.st_size
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += (Path(root) / name).lstat().st_size
            except OSError:                                # pragma: no cover -- raced
                continue
    return total


def held_bytes(work_dir: Path) -> int:
    """Everything under the work directory, for `/health` to report.

    Worth a number rather than a boolean: "the disk is filling" and "the disk is full of
    my own leftovers" lead somewhere different, and until this existed the only way to
    tell them apart was a shell on the pod.
    """
    return _du(work_dir)


def _job_dir(cfg: Config, rec: JobRecord) -> Path | None:
    """The job's directory, if it is one this pod is allowed to delete.

    Resolved before the containment check, not after. Checking the string first accepts
    `work_dir/../../etc` and a symlink out of the tree equally -- the same reasoning as
    `resolve_input`, and the same reason it is not written as a string comparison.
    """
    raw = rec.data.get("workDir")
    if not raw:
        return None
    try:
        path = Path(raw).resolve(strict=True)
        root = cfg.work_dir.resolve()
    except (OSError, RuntimeError):
        return None
    if not path.is_dir() or (root not in path.parents and path != root):
        return None
    if path == root:
        # The pod's whole work directory as one job's directory is a bug somewhere else,
        # and acting on it would take every other job with it.
        return None
    return path


def _is_record(entry: Path) -> bool:
    if not entry.is_file() or entry.is_symlink():
        return False
    if entry.suffix.lower() not in KEEP_SUFFIXES:
        return False
    try:
        return entry.lstat().st_size <= KEEP_MAX_BYTES
    except OSError:                                        # pragma: no cover -- raced
        return False


def _remove(entry: Path) -> int:
    """Delete one entry, returning what it freed. Best effort, by design.

    A file that will not go is not a reason to fail anything: the job it belonged to is
    already over, and the sweep will be back in fifteen minutes.
    """
    size = _du(entry)
    try:
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    except OSError:                                        # pragma: no cover
        return 0
    return 0 if entry.exists() else size


def reclaim(cfg: Config, rec: JobRecord, *, why: str) -> int:
    """Drop the heavy contents of one finished job's directory. Returns bytes freed.

    The directory itself and its records stay, so `GET /jobs/{id}` keeps working and the
    report is still readable. `reclaimedAt` is what tells a later reader that the absence
    of a 9 GB file is a decision rather than a mystery.
    """
    job_dir = _job_dir(cfg, rec)
    if job_dir is None:
        return 0

    freed, removed = 0, []
    for entry in sorted(job_dir.iterdir()):
        if _is_record(entry):
            continue
        size = _remove(entry)
        if size:
            freed += size
            removed.append(entry.name)

    if removed:
        rec.set(reclaimedAt=time.time(),
                reclaimedBytes=int(rec.data.get("reclaimedBytes") or 0) + freed,
                reclaimReason=why)
    return freed


def delivered(rec: JobRecord) -> bool:
    """Whether this job's output is in R2, which is the only licence to delete it."""
    return bool(rec.data.get("outputKey"))


def after_delivery(cfg: Config, rec: JobRecord) -> int:
    """Reclaim immediately after a successful upload, which is the common case.

    Doing it here rather than leaving it to the sweep matters for a queue that is never
    idle: the next job is dispatched within a minute of this one's webhook, and it is
    checked against `min_free_gb` at that moment. Space that comes back a quarter of an
    hour later comes back after the 507.
    """
    if not enabled() or not delivered(rec):
        return 0
    return reclaim(cfg, rec, why="delivered")


def _age(rec: JobRecord, now: float) -> float:
    stamp = rec.data.get("finishedAt") or rec.data.get("updatedAt") or 0.0
    try:
        return max(0.0, now - float(stamp))
    except (TypeError, ValueError):                         # pragma: no cover
        return 0.0


def _reclaimable(cfg: Config, store: JobStore) -> list[tuple[JobRecord, Path]]:
    """Terminal jobs whose directory still holds something worth freeing."""
    out = []
    for rec in store.all():
        if rec.is_live or rec.state == "canceling":
            continue
        job_dir = _job_dir(cfg, rec)
        if job_dir is None:
            continue
        try:
            heavy = any(not _is_record(e) for e in job_dir.iterdir())
        except OSError:                                     # pragma: no cover -- raced
            continue
        if heavy:
            out.append((rec, job_dir))
    return out


def _orphan_dirs(cfg: Config, store: JobStore, now: float) -> list[Path]:
    """Directories in the work dir that no job record claims.

    They come from an ordinary sequence: `DELETE /jobs/{id}` that failed halfway, a state
    directory wiped while the volume kept its files, a pod id changed so the old
    directories belong to nobody. Nothing else in the system will ever mention them
    again, which is exactly why they are worth a pass here.
    """
    known = {r.data.get("workDir") for r in store.all()}
    known = {str(Path(k).resolve()) for k in known if k}
    orphans = []
    try:
        entries = list(cfg.work_dir.iterdir())
    except OSError:
        return []
    for entry in sorted(entries):
        if not entry.is_dir() or entry.is_symlink():
            continue
        if str(entry.resolve()) in known:
            continue
        try:
            if now - entry.stat().st_mtime < ORPHAN_DIR_MIN_AGE:
                continue
        except OSError:                                     # pragma: no cover -- raced
            continue
        orphans.append(entry)
    return orphans


def _emergency(cfg: Config, candidates: Iterable[tuple[JobRecord, Path]],
               say=None) -> tuple[int, list[str]]:
    """Ignore the retention window, because the alternative is refusing every job.

    Only reached below `min_free_gb`, where the pod is already turning work away. Oldest
    first within each state, and the states in the order that loses the least: a delivered
    output is a duplicate of an R2 object, an interrupted job's parts are GPU hours.
    """
    free = free_gb(cfg.work_dir)
    if free is None or free >= cfg.min_free_gb:
        return 0, []

    def rank(item: tuple[JobRecord, Path]) -> tuple[int, float]:
        rec = item[0]
        state = rec.state
        order = (EMERGENCY_ORDER.index(state) if state in EMERGENCY_ORDER
                 else len(EMERGENCY_ORDER))
        return order, float(rec.data.get("finishedAt") or rec.data.get("updatedAt") or 0.0)

    freed, ids = 0, []
    # A margin over the floor, so a pod that just cleared it by a megabyte is not back
    # here on the next tick -- and so the job it is clearing space for actually fits.
    target = cfg.min_free_gb * 1.25
    for rec, _dir in sorted(candidates, key=rank):
        if rec.state == "interrupted" and say is not None:
            say(f"[reclaim] {rec.job_id} was interrupted and its parts are being dropped "
                f"to get under the {cfg.min_free_gb:g} GiB floor. A retry now starts from "
                f"the beginning rather than resuming.")
        got = reclaim(cfg, rec, why="low_space")
        if got:
            freed += got
            ids.append(rec.job_id)
        now_free = free_gb(cfg.work_dir)
        if now_free is None or now_free >= target:
            break
    return freed, ids


def sweep(cfg: Config, store: JobStore, *, now: float | None = None,
          say=None) -> dict[str, Any]:
    """One pass. Safe to call at any time; does nothing when there is nothing to do.

    Three passes in one, in the order that makes the third rarely necessary: deliver-then-
    delete for anything the runner missed (a pod that died between the upload and the
    reclaim), the retention window for everything else terminal, and only then the
    free-space emergency.
    """
    now = time.time() if now is None else now
    window = cfg.retention_hours * 3600.0
    summary: dict[str, Any] = {"freedBytes": 0, "jobs": [], "orphanDirs": [],
                               "kept": 0, "emergency": []}
    if not enabled():
        return summary

    candidates = _reclaimable(cfg, store)
    leftover: list[tuple[JobRecord, Path]] = []
    for rec, job_dir in candidates:
        # Delivered means the bytes are in R2, so the window has nothing to protect.
        if delivered(rec) or _age(rec, now) >= window:
            freed = reclaim(cfg, rec, why="delivered" if delivered(rec) else "retention")
            if freed:
                summary["freedBytes"] += freed
                summary["jobs"].append(rec.job_id)
                continue
        leftover.append((rec, job_dir))
    summary["kept"] = len(leftover)

    for path in _orphan_dirs(cfg, store, now):
        freed = _remove(path)
        if freed:
            summary["freedBytes"] += freed
            summary["orphanDirs"].append(path.name)

    freed, ids = _emergency(cfg, leftover, say=say)
    summary["freedBytes"] += freed
    summary["emergency"] = ids

    return summary


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"                              # pragma: no cover


def describe(summary: dict[str, Any]) -> str | None:
    """One line for the console, or None when there was nothing to say.

    A sweep that freed nothing is the normal outcome and printing it every fifteen minutes
    is how a log stops being read.
    """
    if not summary.get("freedBytes"):
        return None
    bits = [f"freed {human(int(summary['freedBytes']))}"]
    if summary.get("jobs"):
        bits.append(f"{len(summary['jobs'])} finished job(s)")
    if summary.get("orphanDirs"):
        bits.append(f"{len(summary['orphanDirs'])} directory(ies) no job claims")
    if summary.get("emergency"):
        bits.append(f"{len(summary['emergency'])} dropped early for space")
    if summary.get("kept"):
        bits.append(f"{summary['kept']} kept for now")
    return "[reclaim] " + ", ".join(bits)
