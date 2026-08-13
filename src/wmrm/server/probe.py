"""What this machine can do, worked out once and cached.

**The GPU probe runs in a subprocess and then exits.** Importing torch and touching
`torch.cuda` inside the server process risks leaving a CUDA context resident for the
life of uvicorn, competing for VRAM with every ProPainter run on this machine -- and
caching the answer does not release it. A child process that reports and dies cannot.

`torch.cuda.get_arch_list()` is the interesting part and the reason this is not just
`nvidia-smi`. A cu124 wheel on a Blackwell card installs cleanly, reports `cuda (...)`,
loads the models, and then dies at the first kernel launch with "no kernel image is
available for execution on the device". The arch list is what distinguishes that machine
before a job is dispatched to it rather than eight hours in.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Runs in a child interpreter. Prints one JSON object and exits, so whatever CUDA state
# it created goes with it.
_TORCH_PROBE = r"""
import json
out = {"torch": None, "cuda": False, "name": None, "vram_total_mb": None,
       "arch_list": [], "error": None}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = bool(torch.cuda.is_available())
    try:
        out["arch_list"] = list(torch.cuda.get_arch_list())
    except Exception:
        pass
    if out["cuda"]:
        out["name"] = torch.cuda.get_device_name(0)
        out["vram_total_mb"] = int(
            torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""


def _run_torch_probe(timeout: float = 120.0) -> dict[str, Any]:
    try:
        res = subprocess.run([sys.executable, "-c", _TORCH_PROBE],
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"torch": None, "cuda": False, "name": None, "vram_total_mb": None,
                "arch_list": [], "error": "probe timed out"}
    if res.returncode != 0:
        return {"torch": None, "cuda": False, "name": None, "vram_total_mb": None,
                "arch_list": [], "error": (res.stderr or "").strip()[-400:]}
    try:
        return json.loads(res.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"torch": None, "cuda": False, "name": None, "vram_total_mb": None,
                "arch_list": [], "error": "probe produced no JSON"}


def _nvdec_available(ffmpeg: str | None) -> bool:
    """Whether ffmpeg can decode through the GPU.

    Worth reporting because scene detection is the one pass where decode is the entire
    cost: measured, NVDEC took it from 32.9s to 6.8s on 120s of 1080p, and on a
    feature-length film that is roughly 43 minutes against 9.
    """
    if not ffmpeg:
        return False
    try:
        res = subprocess.run([ffmpeg, "-hide_banner", "-hwaccels"],
                            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "cuda" in (res.stdout or "")


def probe_machine() -> dict[str, Any]:
    """Everything static about this machine. Call once, at startup."""
    ffmpeg = shutil.which("ffmpeg")
    torch_info = _run_torch_probe()
    return {
        "ffmpeg": bool(ffmpeg),
        "ffprobe": bool(shutil.which("ffprobe")),
        "nvdec": _nvdec_available(ffmpeg),
        "gpu": {
            "cuda": torch_info["cuda"],
            "name": torch_info["name"],
            "vramTotalMb": torch_info["vram_total_mb"],
            "archList": torch_info["arch_list"],
            "torch": torch_info["torch"],
            "error": torch_info["error"],
        },
    }


def vram_free_mb() -> int | None:
    """Free VRAM right now, without torch.

    Synchronous and therefore called from a threadpool by the route -- with a single
    uvicorn worker, a blocking subprocess in the event loop stalls every other request
    on a machine that is already saturated by a run.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        res = subprocess.run(
            [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    first = (res.stdout or "").strip().splitlines()
    try:
        return int(first[0].strip())
    except (ValueError, IndexError):
        return None


def free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return 0.0
