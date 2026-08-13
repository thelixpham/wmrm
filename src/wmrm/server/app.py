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

import os
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from .. import __version__
from .auth import require_token
from .config import GPU_ENGINES, Config
from .hooks import Notifier
from .models import (CancelAccepted, Health, JobList, JobSpec, JobStatus,
                     SubmitAccepted)
from .probe import free_gb, probe_machine, vram_free_mb
from .runner import JobRunner
from .store import JobStore

STARTED_AT = time.time()


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
                secret=cfg.webhook_secret,
                access_client_id=cfg.access_client_id,
                access_client_secret=cfg.access_client_secret,
            )
            await notifier.terminal(
                job_id=rec.job_id,
                dispatch_token=str(spec.get("dispatchToken") or ""),
                state="interrupted", outcome="interrupted", report=None,
                error=rec.data.get("error"),
            )

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
            "disk": {"workDirPath": str(cfg.work_dir),
                     "workDirFreeGb": round(free_gb(cfg.work_dir), 2),
                     "minFreeGb": cfg.min_free_gb},
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
            raise HTTPException(
                status_code=400,
                detail=f"engine '{spec.engine}' needs CUDA and this pod reports none")

        if runner.at_capacity():
            raise HTTPException(
                status_code=409,
                detail=f"pod is busy ({cfg.max_concurrent} concurrent job(s) max)")

        free = free_gb(cfg.work_dir)
        if free < cfg.min_free_gb:
            raise HTTPException(
                status_code=507,
                detail=f"only {free:.1f} GiB free in {cfg.work_dir}, "
                       f"need {cfg.min_free_gb:.0f} GiB")

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
            secret=cfg.webhook_secret,
            access_client_id=cfg.access_client_id,
            access_client_secret=cfg.access_client_secret,
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
