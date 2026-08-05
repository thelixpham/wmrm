"""Recover the background under a semi-transparent watermark instead of inventing it.

A fixed translucent mark is an alpha blend, per pixel and per channel:

    C_t = a*W + (1 - a)*B_t          a, W constant over time; B_t is the real scene

Write it as an affine map with m = 1 - a and k = a*W:

    C_t = m*B_t + k        =>        B_t = (C_t - k) / m

So if m and k can be estimated once, every frame is recovered by one subtraction
and one division. Two consequences that matter:

- It is **recovery, not generation.** Inpainting deletes the region and guesses a
  replacement; this reads the background that is still present in the signal.
  Measured on the footage the customer objected to, the mark suppressed nothing at
  all -- variance inside it was *higher* than the surrounding clean pixels -- so
  inpainting there was discarding an essentially intact background.
- It **cannot flicker.** m and k are fixed, so consecutive frames go through the
  same deterministic transform. The boiling that per-frame inpainting produces
  (measured: 0.01 correlation with real motion) has no mechanism here.

Estimating m and k is where the care goes. Differences between frames cancel k:

    C_t - C_s = m*(B_t - B_s)

so taking the standard deviation over time,  std(C) = m * std(B).  We know std(C)
everywhere, and std(B) wherever the mark is absent. The trick is that **std(B) and
mean(B) are smooth, low-frequency fields** -- unlike the image itself -- so
interpolating them across the mark from the surrounding ring is safe. Interpolating
image detail is exactly what we are trying to avoid; interpolating a variance map
is not the same problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Below this, the mark blocks too much for division to be stable and the pixel is
# handed to an inpainting fallback instead.
MIN_M = 0.12


@dataclass
class Unblend:
    """The fitted per-pixel, per-channel affine map, tile-sized."""

    m: np.ndarray            # float32 (h, w, 3) -- 1 - alpha
    k: np.ndarray            # float32 (h, w, 3) -- alpha * W
    opaque: np.ndarray       # bool (h, w) -- where m fell below MIN_M
    mark_colour: np.ndarray  # float32 (3,) -- estimated W, for reporting

    @property
    def alpha_median(self) -> float:
        return float(np.median(1.0 - self.m))

    @property
    def opaque_fraction(self) -> float:
        return float(self.opaque.mean())

    def describe(self) -> str:
        a = 1.0 - self.m
        b, g, r = self.mark_colour
        return (
            f"mark colour W : B={b:.0f} G={g:.0f} R={r:.0f}\n"
            f"alpha median  : {float(np.median(a)):.3f}\n"
            f"alpha 90th pct: {float(np.percentile(a, 90)):.3f}\n"
            f"opaque pixels : {100 * self.opaque_fraction:.1f}% "
            f"(m < {MIN_M}, handed to the inpainting fallback)"
        )

    def apply(self, tile_bgr: np.ndarray) -> np.ndarray:
        """B = (C - k) / m, clamped."""
        c = tile_bgr.astype(np.float32)
        out = (c - self.k) / np.maximum(self.m, MIN_M)
        return np.clip(out, 0, 255).astype(np.uint8)


def _fill_from_outside(field: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Interpolate a smooth scalar/vector field across `hole` from its surroundings.

    Only ever applied to temporal statistics (std, mean over many frames), which
    are low-frequency. Doing this to a single frame's pixels would be the very
    hallucination we are avoiding.
    """
    out = np.empty_like(field)
    hole_u8 = hole.astype(np.uint8) * 255
    for c in range(field.shape[2]):
        ch = field[:, :, c]
        lo, hi = float(ch.min()), float(ch.max())
        span = max(hi - lo, 1e-6)
        as_u8 = np.clip((ch - lo) / span * 255.0, 0, 255).astype(np.uint8)
        filled = cv2.inpaint(as_u8, hole_u8, 5, cv2.INPAINT_TELEA)
        out[:, :, c] = filled.astype(np.float32) / 255.0 * span + lo
    return out


def fit(
    frames_bgr: np.ndarray,
    mark_mask: np.ndarray,
    *,
    smooth_px: int = 3,
    min_frames: int = 12,
) -> Unblend:
    """Estimate the affine map from N sampled frames of the tile.

    `frames_bgr` is (N, h, w, 3) float32; `mark_mask` is bool (h, w), True where
    the watermark is. Frames should be spread across the whole clip: the estimate
    relies on the background actually changing behind the mark.
    """
    if frames_bgr.ndim != 4 or frames_bgr.shape[3] != 3:
        raise ValueError(f"expected (N,h,w,3), got {frames_bgr.shape}")
    n = frames_bgr.shape[0]
    if n < min_frames:
        raise ValueError(
            f"un-blend needs at least {min_frames} frames to separate the mark "
            f"from the background; got {n}"
        )
    if mark_mask.shape != frames_bgr.shape[1:3]:
        raise ValueError("mark_mask does not match the tile size")
    if not mark_mask.any():
        raise ValueError("mark_mask is empty")
    if mark_mask.all():
        raise ValueError(
            "the mark covers the whole tile, leaving no clean pixels to estimate "
            "the background statistics from -- increase --margin"
        )

    std_c = frames_bgr.std(axis=0)          # (h, w, 3)
    mean_c = frames_bgr.mean(axis=0)

    # Grow the hole a little: anti-aliased glyph edges are partially covered and
    # would otherwise poison the "clean" statistics we interpolate from.
    hole = cv2.dilate(mark_mask.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)

    std_b = _fill_from_outside(std_c, hole)
    mean_b = _fill_from_outside(mean_c, hole)

    # m = std(C) / std(B). Where the background barely moves the ratio is
    # meaningless, so fall back to "transparent" (m=1) rather than trusting noise.
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(std_b > 1.5, std_c / np.maximum(std_b, 1e-6), 1.0)
    m = np.clip(np.nan_to_num(m, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
    m = m.astype(np.float32)

    # REJECTED ALTERNATIVE, kept as a warning. The variance ratio is biased: std(B)
    # is interpolated from the ring, the mark tends to sit over the busiest content
    # while the ring includes calmer areas, so std(B) comes out low, the ratio
    # exceeds 1, m clips to 1 and nothing is removed exactly where it matters.
    # Measured on real footage: std inside the mark 39.2 vs ring 26.5.
    #
    # The obvious fix is to use the mean offset with a constant mark colour W:
    #     mean(C) = m*mean(B) + (1-m)*W  =>  m = (mean(C) - W) / (mean(B) - W)
    # It was implemented and it is WORSE. W cannot be estimated reliably when the
    # mark lies over saturated colour -- on maple leaves it came out pink,
    # (193,146,173) instead of near-white -- and any error in W is divided by a
    # small m, so it erupts into bright yellow/green blowouts.
    #
    # Keep the variance ratio because of *how* it fails: it under-removes, leaving a
    # faint ghost. The mean-offset version over-corrects into vivid artifacts. A
    # mild ghost is far less objectionable than a bright splash, so the estimator
    # that fails safe wins even though it is the less accurate one on paper.

    # The mark's own structure is at glyph scale, so only light smoothing -- enough
    # to calm the ratio noise without blurring stroke edges away.
    if smooth_px >= 3:
        ksize = smooth_px | 1
        m = cv2.medianBlur(m, ksize)
        m = cv2.GaussianBlur(m, (ksize, ksize), 0)

    # Outside the mark there is nothing to undo; pin it exactly so the recovery is
    # an identity there and the composite cannot introduce a seam.
    outside = ~mark_mask
    m[outside] = 1.0

    k = (mean_c - m * mean_b).astype(np.float32)
    k[outside] = 0.0

    opaque = (m.min(axis=2) < MIN_M) & mark_mask

    # Reported only: W implied by the fit, k = (1-m)*W, read off the most-covered
    # pixels. A wildly non-neutral value here is a hint the fit is struggling.
    strong = mark_mask & (m.min(axis=2) < 0.9)
    if strong.any():
        w_est = np.median(k[strong] / np.maximum(1.0 - m[strong], 1e-3), axis=0)
    else:
        w_est = np.zeros(3, np.float32)

    return Unblend(m=m, k=k, opaque=opaque,
                   mark_colour=np.clip(w_est, 0, 255).astype(np.float32))
