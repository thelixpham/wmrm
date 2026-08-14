"""Job state on disk, one file per job.

On disk rather than in memory because uvicorn does not get to choose when it stops. A
pod is restarted, a deploy replaces the process, the OOM killer picks it -- and in every
one of those cases the control plane is waiting to hear what happened to a job that may
be eight hours in. Memory-only state turns all of them into silence.

`adopt_orphans()` is the other half: on startup, any job still in a running state is one
whose process died with the last server. It becomes `interrupted`, which is distinct from
`canceled` on purpose -- `interrupted` means "nothing wrong with the work, run it again"
and `--resume` can pick up the finished parts, while `canceled` means a person said stop.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

#: States from which the job may still move. Anything else is final.
LIVE_STATES = ("preparing", "downloading", "detecting", "running", "uploading")
TERMINAL_STATES = ("succeeded", "failed", "needs_review", "canceled", "interrupted")


class JobRecord:
    """A job's state, mirrored to `<state_dir>/jobs/<id>.json` on every change."""

    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = path
        self.data = data

    # -- construction ---------------------------------------------------------- #

    @classmethod
    def create(cls, path: Path, *, job_id: str, spec: dict[str, Any],
               work_dir: str) -> "JobRecord":
        now = time.time()
        return cls(path, {
            "schema": 1,
            "jobId": job_id,
            "state": "preparing",
            "phase": "preparing",
            "spec": spec,
            "workDir": work_dir,
            "createdAt": now,
            "updatedAt": now,
            "startedAt": None,
            "finishedAt": None,
            "pid": None,
            "progress": None,
            "box": None,
            "outcome": None,
            "report": None,
            "error": None,
            "cancelRequested": False,
        })

    @classmethod
    def load(cls, path: Path) -> "JobRecord | None":
        try:
            return cls(path, json.loads(path.read_text()))
        except (OSError, ValueError):
            return None

    # -- accessors ------------------------------------------------------------- #

    @property
    def job_id(self) -> str:
        return str(self.data["jobId"])

    @property
    def state(self) -> str:
        return str(self.data["state"])

    @property
    def is_live(self) -> bool:
        return self.state in LIVE_STATES

    @property
    def cancel_requested(self) -> bool:
        return bool(self.data.get("cancelRequested"))

    # -- mutation -------------------------------------------------------------- #

    def set(self, **fields: Any) -> "JobRecord":
        self.data.update(fields)
        self.data["updatedAt"] = time.time()
        self.save()
        return self

    def set_state(self, state: str, *, phase: str | None = None, **fields: Any) -> "JobRecord":
        if state == "running" and not self.data.get("startedAt"):
            fields["startedAt"] = time.time()
        if state in TERMINAL_STATES:
            fields["finishedAt"] = time.time()
        return self.set(state=state, phase=phase or state, **fields)

    def save(self) -> None:
        # Atomic. The control plane polls this through the API while a run is writing it,
        # and a half-written document read as JSON is a crash in the reader.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.partial")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.replace(tmp, self.path)

    # -- wire form ------------------------------------------------------------- #

    def public(self) -> dict[str, Any]:
        """What `GET /jobs/{id}` returns.

        `spec` is deliberately not included: it carries the presigned input URL, and a
        status endpoint is not the place to hand that back out.
        """
        d = self.data
        return {
            "jobId": d["jobId"],
            "state": d["state"],
            "phase": d.get("phase"),
            "engine": (d.get("spec") or {}).get("engine"),
            "outputKey": d.get("outputKey"),
            "box": d.get("box"),
            "progress": d.get("progress"),
            "outcome": d.get("outcome"),
            "error": d.get("error"),
            "report": d.get("report"),
            "createdAt": d.get("createdAt"),
            "updatedAt": d.get("updatedAt"),
            "startedAt": d.get("startedAt"),
            "finishedAt": d.get("finishedAt"),
        }


class JobStore:
    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir) / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        return self.dir / f"{job_id}.json"

    def get(self, job_id: str) -> JobRecord | None:
        return JobRecord.load(self._path(job_id))

    def create(self, *, job_id: str, spec: dict[str, Any], work_dir: str) -> JobRecord:
        rec = JobRecord.create(self._path(job_id), job_id=job_id, spec=spec,
                               work_dir=work_dir)
        rec.save()
        return rec

    def all(self) -> Iterator[JobRecord]:
        for path in sorted(self.dir.glob("*.json")):
            rec = JobRecord.load(path)
            if rec is not None:
                yield rec

    def live(self) -> list[JobRecord]:
        return [r for r in self.all() if r.is_live]

    def delete(self, job_id: str) -> bool:
        try:
            self._path(job_id).unlink()
            return True
        except OSError:
            return False

    def adopt_orphans(self) -> list[JobRecord]:
        """Mark jobs left running by a previous process as `interrupted`.

        Called once at startup. Returns them so the caller can tell the control plane --
        which is the whole point: a job that silently stays `running` forever holds a
        slot on this machine and is never retried.
        """
        orphans = []
        for rec in self.all():
            if rec.is_live:
                rec.set_state(
                    "interrupted",
                    outcome="interrupted",
                    error={"code": "interrupted",
                           "message": "the server restarted while this job was running"},
                    pid=None,
                )
                orphans.append(rec)
        return orphans
