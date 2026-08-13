"""Configuration, all from the environment.

Nothing is read from a file in the repo. On a RunPod Pod the container filesystem is
discarded on stop/restart -- only `/workspace` (the volume) survives -- so anything
that must outlive a restart has to be under `/workspace`, and the defaults here say so
rather than leaving it to whoever writes the entrypoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Engines `wmrm run --quality` accepts. Kept in step with cli.py deliberately: the
#: control plane validates against what /health reports, so a stale list here would let
#: it dispatch a job this machine cannot run.
ENGINES = ("unblend", "video", "high", "fast", "draft")

#: Engines that will not do useful work without a GPU. `video` on CPU is ~400x slower
#: (measured: 0.27 fps on six cores), which is not a slow option, it is a broken one.
GPU_ENGINES = ("video", "high")


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name) or default).expanduser()


@dataclass(frozen=True)
class Config:
    pod_id: str
    token: str | None
    work_dir: Path
    state_dir: Path
    local_input_root: Path | None
    webhook_secret: str | None
    access_client_id: str | None
    access_client_secret: str | None
    max_concurrent: int
    min_free_gb: float
    r2_bucket: str | None
    r2_workers: int

    @classmethod
    def from_env(cls) -> "Config":
        # RUNPOD_POD_ID is set by the platform, so the common case needs no configuration
        # at all -- and getting this wrong matters: the work directory is namespaced by
        # it, which is what keeps two pods off one path if they ever share a volume.
        pod_id = (os.environ.get("WMRM_POD_ID")
                  or os.environ.get("RUNPOD_POD_ID")
                  or "local")
        root = _env_path("WMRM_LOCAL_INPUT_ROOT", "") if os.environ.get(
            "WMRM_LOCAL_INPUT_ROOT") else None
        return cls(
            pod_id=pod_id,
            token=os.environ.get("WMRM_POD_TOKEN") or None,
            work_dir=_env_path("WMRM_WORK_DIR", "/workspace/wmrm-work") / pod_id,
            state_dir=_env_path("WMRM_STATE", "/workspace/wmrm-state") / pod_id,
            local_input_root=root.resolve() if root else None,
            webhook_secret=os.environ.get("WMRM_WEBHOOK_SECRET") or None,
            access_client_id=os.environ.get("CF_ACCESS_CLIENT_ID") or None,
            access_client_secret=os.environ.get("CF_ACCESS_CLIENT_SECRET") or None,
            # One job at a time. ProPainter materialises a whole segment as a tensor, so
            # two runs on one card is the fastest way to turn a working machine into two
            # out-of-memory failures.
            max_concurrent=int(os.environ.get("WMRM_MAX_CONCURRENT") or "1"),
            # Refuse a job rather than discover at hour seven that the disk is full. A
            # 100 GB source needs input + parts + output, so roughly 3x itself.
            min_free_gb=float(os.environ.get("WMRM_MIN_FREE_GB") or "50"),
            r2_bucket=os.environ.get("R2_BUCKET") or os.environ.get("S3_BUCKET") or None,
            # 8 is where `wmrm pull` settled: past that the link or the disk saturates
            # first, so more workers only add connections.
            r2_workers=int(os.environ.get("WMRM_R2_WORKERS") or "8"),
        )

    @property
    def r2_configured(self) -> bool:
        """Whether this pod can reach R2 on its own.

        Checked so a job asking for `kind: "r2"` is refused at submit time with a clear
        reason, rather than accepted and then failing once it is already the pod's
        problem. Credential *validity* is not checked here -- that needs a request, and
        the first download is soon enough.
        """
        try:
            from ..r2 import Creds
            Creds.from_env(self.r2_bucket)
        except Exception:                              # noqa: BLE001 -- R2Error or ImportError
            return False
        return True

    def r2_reason(self) -> str | None:
        """Why R2 is unavailable, in the words the credential loader used."""
        try:
            from ..r2 import Creds
            Creds.from_env(self.r2_bucket)
        except Exception as exc:                       # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        return None

    def job_dir(self, job_id: str) -> Path:
        return self.work_dir / job_id

    def ensure_dirs(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "jobs").mkdir(parents=True, exist_ok=True)
