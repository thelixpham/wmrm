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


def describe_device(device: str) -> str:
    """Human-readable note about where work will actually run.

    Worth printing on every run: installing the CPU-only torch wheel on a GPU
    machine is silent and costs 20-50x, and there is no way to notice it from the
    output video.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return f"{device} (torch not installed)"

    if device == "cuda":
        try:
            i = torch.cuda.current_device()
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            return f"cuda ({name}, {vram:.1f} GB, torch {torch.__version__})"
        except Exception:  # pragma: no cover
            return f"cuda (torch {torch.__version__})"
    if device == "cpu":
        why = "no CUDA device" if not torch.cuda.is_available() else "CUDA available but not selected"
        return (f"cpu ({torch.get_num_threads()} threads, {why}, "
                f"torch {torch.__version__})")
    return f"{device} (torch {torch.__version__})"


class Backend(ABC):
    name: str
    device_note: str = "cpu"

    @abstractmethod
    def inpaint(self, tile_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return an inpainted copy of `tile_bgr`. `mask` is binary uint8."""


class Cv2Backend(Backend):
    """Fast diffusion fill. Fine for flat backgrounds, visibly smears on
    texture -- keep it for previews and for the `fast` quality tier."""

    name = "cv2"
    device_note = "cpu (OpenCV, no model)"

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
        self.device_note = describe_device(device)

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


class UnblendBackend(Backend):
    """Solve the alpha blend for the real background, then optionally erase what is
    left on the glyph strokes alone.

    Only valid for semi-transparent marks, and only after `unblend.fit` has been
    given frames from the specific clip -- the fitted map is per-video, not a
    property of the logo alone (compression and grading change it).

    Two stages, and the second one is what makes complete removal possible:

    1. `fit.apply` divides the blend back out. This recovers real background and
       cannot flicker, but it can never reach zero residual: the leftover is
       proportional to the error in alpha, and alpha has to be estimated from
       background statistics that are by definition unobservable under the mark.
    2. Where alpha says there is a real stroke, hand *those pixels only* to an
       inpainter. Removal there is guaranteed because the pixels are replaced, not
       corrected.

    MEASURED RESULT: stage 2 with a stroke-shaped mask **does not work**, and is off
    by default (`stroke_alpha=0`). Residual locked-edge energy on two reference
    frames, un-blend alone scoring 32.4:

        stroke mask, alpha > 0.10  (40% of box)   29.7
        stroke mask, alpha > 0.15  (28% of box)   31.9
        stroke mask, alpha > 0.20  (14% of box)   32.6
        full-box mask                            20.5   <- and visually clean

    The trace survives every stroke threshold. Widening the mask until it goes is the
    same thing as masking the whole box, which is just `--quality high`. So there is
    no middle setting that removes the mark while keeping most original pixels: the
    parameter is kept because the negative result is worth recording, not because
    there is a value of it worth using.

    Stage 2 is fed the *stage 1 output* rather than the original frame, which also
    turned out not to matter: with a full-box mask the inpainter discards everything
    inside it anyway, and B/C of that experiment scored identically.
    """

    name = "unblend"

    def __init__(self, fitted, fallback: Backend | None = None, *,
                 stroke_alpha: float = 0.0, stroke_dilate: int = 2,
                 feather_px: int = 3) -> None:
        self.fit = fitted
        self.fallback = fallback

        hard = fitted.opaque.copy()
        if stroke_alpha > 0:
            hard |= fitted.stroke_mask(stroke_alpha, dilate_px=stroke_dilate)
        self.hard = hard
        self.hard_fraction = float(hard.sum() / max(fitted.mark_mask.sum(), 1))
        self._hard_u8 = (hard * 255).astype(np.uint8)

        # Feathered copy used only for compositing, so the repaired strokes do not
        # meet the recovered background along a hard 1px edge.
        k = max(1, feather_px) | 1
        self._alpha = (cv2.GaussianBlur(self._hard_u8, (k, k), 0)
                       .astype(np.float32) / 255.0)[:, :, None]

        self.device_note = "cpu (numpy only, no model)"
        if fallback is not None and hard.any():
            self.name = f"unblend+{fallback.name}"
            self.device_note = (f"cpu (numpy); stroke repair on "
                                f"{fallback.device_note}")

    def inpaint(self, tile_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = self.fit.apply(tile_bgr)
        if self.fallback is None or not self.hard.any():
            return out
        patched = self.fallback.inpaint(out, self._hard_u8)
        blended = patched.astype(np.float32) * self._alpha + \
            out.astype(np.float32) * (1.0 - self._alpha)
        return np.clip(blended, 0, 255).astype(np.uint8)


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
        self.device_note = inner.device_note
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
                 device: str = "auto", fitted=None,
                 stroke_alpha: float = 0.0, stroke_dilate: int = 2) -> Backend:
    if quality == "unblend":
        if fitted is None:
            raise ValueError("quality='unblend' needs a fitted map from unblend.fit")
        needs_repair = fitted.opaque_fraction > 0 or stroke_alpha > 0
        fallback: Backend | None = None
        if needs_repair:
            fallback = LamaBackend(device=device, threads=threads)
            # Only the repair stage is wrapped, and only when asked: the un-blend
            # transform itself is deterministic, but per-frame inpainting of the
            # strokes is not, so --patch-hold is the lever against boiling there.
            if patch_hold > 1:
                fallback = CachingBackend(fallback, tolerance=cache_tolerance,
                                          hold=patch_hold)
        return UnblendBackend(fitted, fallback=fallback,
                              stroke_alpha=stroke_alpha,
                              stroke_dilate=stroke_dilate)
    if quality == "high":
        inner: Backend = LamaBackend(device=device, threads=threads)
    elif quality == "draft":
        inner = Cv2Backend()
    else:
        raise ValueError(f"unknown quality {quality!r}")
    if cache_tolerance > 0 or patch_hold > 1:
        return CachingBackend(inner, tolerance=cache_tolerance, hold=patch_hold)
    return inner
