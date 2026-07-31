"""Inpainting backends. All of them operate on the *tile*, never the frame.

Measured on 6 CPU cores (KNOWLEDGE.md 7.1):

    LaMa, tile 320x192      452 ms/frame   ->  6.8 min for a 30s clip
    LaMa, full 1920x1080  16285 ms/frame   ->  244 min  (77x slower, same output)
    cv2.inpaint, tile         3 ms/frame   ->  2.8 s    (but smears on texture)
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod

import cv2
import numpy as np


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device. 'auto' takes CUDA when it is genuinely usable.

    A GPU is not present on the dev box but is available for real runs, so this
    must not hardcode CPU -- LaMa is roughly 20-50x faster on CUDA, which changes
    what quality settings are affordable.
    """
    if requested not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(f"unknown device {requested!r} (want auto/cpu/cuda/mps)")
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but torch reports no CUDA device.\n"
            "This build of torch may be the CPU-only wheel; reinstall with:\n"
            "  uv pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    return requested


class Backend(ABC):
    name: str

    @abstractmethod
    def inpaint(self, tile_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return an inpainted copy of `tile_bgr`. `mask` is binary uint8."""


class Cv2Backend(Backend):
    """Fast diffusion fill. Fine for flat backgrounds, visibly smears on
    texture -- keep it for previews and for the `fast` quality tier."""

    name = "cv2"

    def __init__(self, radius: int = 3, method: str = "telea") -> None:
        if method not in ("telea", "ns"):
            raise ValueError(f"method must be 'telea' or 'ns', got {method!r}")
        self.radius = radius
        self.flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

    def inpaint(self, tile_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.inpaint(tile_bgr, mask, self.radius, self.flag)


class LamaBackend(Backend):
    """LaMa (big-lama, ~196MB). Loaded once per process, not per video."""

    name = "lama"

    def __init__(self, device: str = "auto", threads: int | None = None) -> None:
        try:
            import torch
            from simple_lama_inpainting import SimpleLama
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "LaMa backend needs torch + simple-lama-inpainting.\n"
                "  uv pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "  uv pip install --no-deps simple-lama-inpainting"
            ) from exc

        device = resolve_device(device)
        if device == "cpu" and threads:
            torch.set_num_threads(threads)
        print(f"[wmrm] loading LaMa on {device} ...", file=sys.stderr, flush=True)
        self._lama = SimpleLama(device=torch.device(device))
        self._Image = __import__("PIL.Image", fromlist=["Image"])
        self.device = device

    def inpaint(self, tile_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        Image = self._Image
        rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        out = self._lama(Image.fromarray(rgb), Image.fromarray(mask))
        arr = np.asarray(out.convert("RGB"), dtype=np.uint8)
        # simple-lama pads to a multiple of 8 internally; crop back defensively
        # so a padded return can never desync the composite.
        arr = arr[: tile_bgr.shape[0], : tile_bgr.shape[1]]
        if arr.shape[:2] != tile_bgr.shape[:2]:
            arr = cv2.resize(arr, (tile_bgr.shape[1], tile_bgr.shape[0]),
                             interpolation=cv2.INTER_LANCZOS4)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


class CachingBackend(Backend):
    """Reuse the previous patch when the tile barely changed.

    Two wins at once: it skips model calls on slow-moving footage, and because
    consecutive frames then share an identical patch it also suppresses the
    per-frame flicker you get from inpainting each frame independently
    (KNOWLEDGE.md 3.2, 7.1).

    `tolerance` is mean absolute difference in 8-bit levels over the tile,
    measured outside the hole -- inside the hole the watermark is constant
    anyway, so including it would make everything look identical.

    `hold` forces reuse for N-1 frames after each fresh inpaint, regardless of
    how much the tile changed. This trades spatial accuracy for temporal
    stability and is the only lever here against boiling: independent per-frame
    inpainting produces variation of realistic amplitude but incoherent
    structure (measured: correlation ~0.01 against the real motion field), which
    is what reads as a boiling patch. Holding the patch removes that variation
    outright. On genuinely moving footage a held patch starts to lag the
    background, so keep N small.
    """

    def __init__(self, inner: Backend, tolerance: float = 1.0, hold: int = 1) -> None:
        self.inner = inner
        bits = [inner.name, "cache"]
        if hold > 1:
            bits.append(f"hold{hold}")
        self.name = "+".join(bits)
        self.tolerance = tolerance
        self.hold = max(1, hold)
        self._prev_ctx: np.ndarray | None = None
        self._prev_out: np.ndarray | None = None
        self._age = 0
        self.hits = 0
        self.misses = 0

    def inpaint(self, tile_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        outside = mask == 0
        ctx = tile_bgr[outside].astype(np.int16)
        reusable = (
            self._prev_ctx is not None
            and self._prev_out is not None
            and ctx.shape == self._prev_ctx.shape
        )
        if reusable and (
            self._age < self.hold
            or float(np.abs(ctx - self._prev_ctx).mean()) <= self.tolerance
        ):
            self._age += 1
            self.hits += 1
            return self._prev_out  # type: ignore[return-value]
        out = self.inner.inpaint(tile_bgr, mask)
        self._prev_ctx, self._prev_out, self._age = ctx, out, 1
        self.misses += 1
        return out

    def stats(self) -> str:
        total = self.hits + self.misses
        pct = (100.0 * self.hits / total) if total else 0.0
        return f"cache {self.hits}/{total} hits ({pct:.0f}%)"


def make_backend(quality: str, *, threads: int | None = None,
                 cache_tolerance: float = 1.0, patch_hold: int = 1,
                 device: str = "auto") -> Backend:
    if quality == "high":
        inner: Backend = LamaBackend(device=device, threads=threads)
    elif quality == "draft":
        inner = Cv2Backend()
    else:
        raise ValueError(f"unknown quality {quality!r} (want 'high' or 'draft')")
    if cache_tolerance > 0 or patch_hold > 1:
        return CachingBackend(inner, tolerance=cache_tolerance, hold=patch_hold)
    return inner
