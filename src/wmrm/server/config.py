"""Configuration, all from the environment.

**Only `WMRM_POD_TOKEN` has to be set.** Everything else has a default that adapts to the
machine, because an API that needs five exports before it will start is an API nobody can
try out -- and the first version of this file did exactly that: it hardcoded RunPod's
`/workspace` layout as the default, so on any other machine the work and state directories
pointed somewhere that did not exist and had to be overridden by hand.

The layout is now detected. On a pod, `/workspace` is the volume that survives a
stop/restart (the container filesystem does not), so that is where everything belongs.
Off a pod there is no `/workspace`, so the user cache directory is used instead. Same
reasoning as `setup.sh` keeping its venv next to the repo: state goes somewhere that
exists on the machine you are actually on.

The resolved paths are printed at startup, so what got chosen is visible rather than
guessed at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: The RunPod volume. Its presence is what distinguishes a pod from a workstation --
#: on a pod this is the only directory that survives a restart.
POD_VOLUME = Path("/workspace")

#: Engines `wmrm run --quality` accepts. Kept in step with cli.py deliberately: the
#: control plane validates against what /health reports, so a stale list here would let
#: it dispatch a job this machine cannot run.
ENGINES = ("unblend", "video", "high", "fast", "draft")

#: Engines that will not do useful work without a GPU. `video` on CPU is ~400x slower
#: (measured: 0.27 fps on six cores), which is not a slow option, it is a broken one.
GPU_ENGINES = ("video", "high")


def on_pod() -> bool:
    """Whether this is really a RunPod Pod.

    The platform's own variable, not the presence of `/workspace`. That directory turned
    out to exist -- empty, root-owned -- on an ordinary workstation image, so testing for
    it decided "this is a pod" on a laptop and put the work directory somewhere
    unwritable. `RUNPOD_POD_ID` is set by the platform and by nothing else.

    It also has to be writable: an image can ship the mount point without the volume
    actually being attached, and a default that cannot be written to is not a default.
    """
    if not os.environ.get("RUNPOD_POD_ID"):
        return False
    return POD_VOLUME.is_dir() and os.access(POD_VOLUME, os.W_OK)


def default_root() -> Path:
    """Where work and state live when nothing says otherwise.

    `/workspace` on a pod, because it is the only thing there that survives a restart --
    the container filesystem is discarded, taking a venv, model weights and any job state
    with it. The user cache directory anywhere else.
    """
    if on_pod():
        return POD_VOLUME
    return Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "wmrm"


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default).expanduser()


@dataclass(frozen=True)
class Config:
    pod_id: str
    token: str | None
    work_dir: Path
    state_dir: Path
    local_input_root: Path | None
    max_concurrent: int
    min_free_gb: float
    r2_bucket: str | None
    r2_workers: int
    mezon_webhook_url: str | None

    @classmethod
    def from_env(cls) -> "Config":
        # RUNPOD_POD_ID is set by the platform, so the common case needs no configuration
        # at all -- and getting this wrong matters: the work directory is namespaced by
        # it, which is what keeps two pods off one path if they ever share a volume.
        pod_id = (os.environ.get("WMRM_POD_ID")
                  or os.environ.get("RUNPOD_POD_ID")
                  or "local")
        root = _env_path("WMRM_LOCAL_INPUT_ROOT", Path()) if os.environ.get(
            "WMRM_LOCAL_INPUT_ROOT") else None
        base = default_root()
        return cls(
            pod_id=pod_id,
            token=os.environ.get("WMRM_POD_TOKEN") or None,
            work_dir=_env_path("WMRM_WORK_DIR", base / "wmrm-work") / pod_id,
            state_dir=_env_path("WMRM_STATE", base / "wmrm-state") / pod_id,
            local_input_root=root.resolve() if root else None,
            # One job at a time. ProPainter materialises a whole segment as a tensor, so
            # two runs on one card is the fastest way to turn a working machine into two
            # out-of-memory failures.
            max_concurrent=int(os.environ.get("WMRM_MAX_CONCURRENT") or "1"),
            # A cheap gate before a transfer starts. The gate that actually matters is
            # per-job: `require_space` compares 3x the source's real size, because a
            # 100 GB input needs input + parts + output alive at once.
            #
            # So this floor only has to be big enough to be worth refusing over, and the
            # right size depends on the machine: 50 GB on a pod that handles feature-length
            # 4K, but on a workstation trying a 15-second fixture that same 50 GB is why
            # the first thing anyone had to do was override it.
            min_free_gb=float(os.environ.get("WMRM_MIN_FREE_GB")
                              or (50 if on_pod() else 2)),
            r2_bucket=os.environ.get("R2_BUCKET") or os.environ.get("S3_BUCKET") or None,
            # 8 is where `wmrm pull` settled: past that the link or the disk saturates
            # first, so more workers only add connections.
            r2_workers=int(os.environ.get("WMRM_R2_WORKERS") or "8"),
            # Optional, and unset means the pod simply does not announce anything -- the
            # control plane's webhook is unaffected either way. Kept out of the repo
            # because the URL *is* the credential: anyone holding it can post to the
            # channel, and there is nothing else to check.
            mezon_webhook_url=os.environ.get("WMRM_MEZON_WEBHOOK_URL") or None,
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
