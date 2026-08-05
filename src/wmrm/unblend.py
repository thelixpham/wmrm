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

# Both marks in scope are white text, and every broadcast badge of this kind is
# near-white. Fixing W instead of estimating it is what makes the mean-offset
# estimator usable at all: see the note in `_fit_matte`.
WHITE = (255.0, 255.0, 255.0)

# Below this the mark colour and the background are too close to tell apart, so the
# blend equation is ill-conditioned and alpha is left at 0 rather than guessed.
MIN_CONTRAST = 6.0


@dataclass
class Unblend:
    """The fitted per-pixel, per-channel affine map, tile-sized."""

    m: np.ndarray            # float32 (h, w, 3) -- 1 - alpha, broadcast
    k: np.ndarray            # float32 (h, w, 3) -- alpha * W
    opaque: np.ndarray       # bool (h, w) -- where alpha exceeded 1 - MIN_M
    mark_colour: np.ndarray  # float32 (3,) -- the W that was used
    mark_mask: np.ndarray    # bool (h, w) -- where the mark is
    alpha_scale: float = 1.0       # global correction chosen by the refine sweep
    residual: float = 0.0          # leftover pixel-locked energy after removal
    residual_before: float = 0.0   # ... and before, for the same measure

    @property
    def alpha(self) -> np.ndarray:
        """Single-channel alpha (h, w)."""
        return 1.0 - self.m[:, :, 0]

    @property
    def alpha_median(self) -> float:
        """Median alpha **inside the mark**.

        Deliberately not over the whole tile. The tile is mostly clean background
        where alpha is pinned to 0, so a tile-wide median reads ~0.000 for any mark
        whatsoever -- a number that once led to the confident and wrong conclusion
        that a plainly visible mark was almost transparent.
        """
        return float(np.median(self.alpha[self.mark_mask]))

    @property
    def opaque_fraction(self) -> float:
        return float(self.opaque.mean())

    def describe(self) -> str:
        a = self.alpha[self.mark_mask]
        b, g, r = self.mark_colour
        pct = lambda p: float(np.percentile(a, p))       # noqa: E731
        return (
            f"mark colour W : B={b:.0f} G={g:.0f} R={r:.0f} (assumed, not fitted)\n"
            f"alpha in mark : median {float(np.median(a)):.3f}  "
            f"p90 {pct(90):.3f}  p99 {pct(99):.3f}  max {float(a.max()):.3f}\n"
            f"              : {100 * float((a > 0.3).mean()):.1f}% of the mark has "
            f"alpha > 0.3\n"
            f"opaque pixels : {100 * self.opaque_fraction:.2f}% "
            f"(alpha > {1 - MIN_M:.2f}, handed to the inpainting fallback)\n"
            f"alpha scale   : x{self.alpha_scale:.1f} (chosen by residual sweep)\n"
            f"mark residual : {self.residual_before:.2f} -> {self.residual:.2f} "
            f"locked-edge energy in the mark"
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


def _fit_matte(mean_c: np.ndarray, mean_b: np.ndarray,
               mark_colour: np.ndarray) -> np.ndarray:
    """Least-squares alpha per pixel from the mean offset, with W held fixed.

    Returns a single-channel alpha (h, w). This is the estimator that actually
    removes the mark; see `fit` for why the variance ratio does not.

    Two decisions carry the whole thing:

    **Alpha is one number per pixel, not three.** A matte is a coverage fraction --
    physically identical across channels. The old estimator solved for three
    independent values, so channel noise turned into colour fringes and the mark's
    own colour leaked into the estimate. Sharing one alpha across three channels is
    3x the data for the same unknown.

    **W is fixed, not estimated.** With `mean(C) = (1-a)*mean(B) + a*W` the only
    unknown left is `a`, and the least-squares solution over channels is

        a = sum_c (dC_c * D_c) / sum_c D_c^2,
        dC = mean(C) - mean(B),   D = W - mean(B)

    which weights each channel by how far the background sits from W -- exactly the
    conditioning of that channel. Estimating W as well is what made the earlier
    attempt blow up: on maple leaves it came out pink, (193,146,173), and the error
    was then divided by a small (1-a) into vivid yellow-green splashes. Assuming
    white is also the *safe* direction to be wrong in: if the real mark is light
    grey, the denominator is too big, alpha comes out low, and the failure is a
    residual ghost rather than a blowout.
    """
    d_c = mean_c - mean_b                        # (h, w, 3) observed lift toward W
    d = mark_colour[None, None, :] - mean_b      # (h, w, 3) available headroom
    num = (d_c * d).sum(axis=2)
    den = (d * d).sum(axis=2)

    # Where the background is already as bright as the mark there is no headroom to
    # measure against; alpha is unrecoverable there, and it is also invisible there,
    # so 0 is both the honest and the harmless answer.
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(den > MIN_CONTRAST ** 2 * 3, num / np.maximum(den, 1e-6), 0.0)
    return np.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _locked_edge_energy(stack: np.ndarray, mask: np.ndarray) -> float:
    """How much pixel-locked structure sits inside `mask`.

    Signed gradients are averaged over frames and only THEN made absolute, so edges
    that move with the scene cancel out and anything welded to the pixel grid
    survives. This is the same signal the detector uses to find a watermark, which
    makes it the right objective here: it measures "is a watermark still findable",
    not "is this blurry". PSNR would happily reward smearing the region flat.
    """
    gx, gy = [], []
    for frame in stack:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx.append(cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3))
        gy.append(cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3))
    locked = np.abs(np.mean(gx, axis=0)) + np.abs(np.mean(gy, axis=0))
    return float(locked[mask].mean())


def _refine_scale(frames_bgr: np.ndarray, alpha: np.ndarray, mark_mask: np.ndarray,
                  w: np.ndarray, bound: np.ndarray,
                  *, cap: float = 4.0) -> tuple[float, float, float]:
    """Find one global multiplier for alpha by minimising leftover mark structure.

    The mean-offset fit recovers the *shape* of the matte well -- it knows which
    pixels are glyph and which are gap -- but its *scale* is biased low, because
    mean(B) has to be interpolated across the mark from a ring that sits over
    different content. Measured: it removed only two thirds of the mark.

    Rather than model that bias, measure it. Sweep a single scalar and keep the value
    that leaves the least pixel-locked structure behind. One degree of freedom fitted
    against the objective the customer actually judges.

    The objective is U-shaped, which is what makes this safe: too little correction
    leaves the glyph, too much prints an inverted glyph, and both raise the energy.
    The minimum is a real optimum rather than the end of a ramp, so the sweep cannot
    silently run away into over-correction the way an unbounded solve could.
    """
    best_s, best_e = 1.0, float("inf")
    base_e = None
    for s in np.arange(1.0, cap + 1e-9, 0.1):
        a = np.minimum(np.clip(alpha * s, 0.0, 1.0 - MIN_M), bound)
        m = np.repeat((1.0 - a)[:, :, None], 3, axis=2).astype(np.float32)
        k = (a[:, :, None] * w[None, None, :]).astype(np.float32)
        out = np.clip((frames_bgr - k) / np.maximum(m, MIN_M), 0, 255).astype(np.uint8)
        e = _locked_edge_energy(out, mark_mask)
        if base_e is None:
            base_e = e
        if e < best_e:
            best_s, best_e = float(s), e
    return best_s, best_e, float(base_e)


def fit(
    frames_bgr: np.ndarray,
    mark_mask: np.ndarray,
    *,
    smooth_px: int = 0,
    min_frames: int = 12,
    mark_colour: tuple[float, float, float] = WHITE,
    estimator: str = "matte",
    refine: bool = True,
) -> Unblend:
    """Estimate the affine map from N sampled frames of the tile.

    `frames_bgr` is (N, h, w, 3) float32; `mark_mask` is bool (h, w), True where
    the watermark is. Frames should be spread across the whole clip: the estimate
    relies on the background actually changing behind the mark.

    `estimator` selects how alpha is measured:

    - `matte` (default) -- least squares on the mean offset with W fixed white.
    - `variance` -- the original `std(C)/std(B)` ratio. **Kept only for comparison.**
      It is biased low and by a lot: std(B) is interpolated from the ring, the mark
      sits over busier content than the ring average, so the ratio drifts above 1,
      clips, and removes nothing exactly where the mark is strongest. Measured on
      real footage: std inside the mark 39.2 vs 26.5 in the ring, fitted alpha
      median 0.074 -- and the mark stayed plainly legible in the output.
    """
    if estimator not in ("matte", "variance"):
        raise ValueError(f"estimator must be 'matte' or 'variance', got {estimator!r}")
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

    w = np.asarray(mark_colour, np.float32)

    if estimator == "matte":
        alpha = _fit_matte(mean_c, mean_b, w)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(std_b > 1.5, std_c / np.maximum(std_b, 1e-6), 1.0)
        ratio = np.clip(np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
        alpha = 1.0 - ratio.min(axis=2).astype(np.float32)

    # Smoothing defaults to OFF, and that is the single biggest quality decision in
    # this module. Glyph strokes are 2-3 px wide, so a 3x3 median plus a 3x3 gaussian
    # filters at exactly the scale of the signal it is meant to preserve: the matte
    # keeps the right shape but goes soft, the division then only partly cancels the
    # stroke, and the mark stays legible. Measured on real footage, residual
    # locked-edge energy inside the mark (background floor 10.05):
    #
    #     smooth_px=0   8.58   <- below the surrounding background
    #     smooth_px=1   8.58
    #     smooth_px=3  14.06   <- the old default; logo still readable
    #     smooth_px=5  19.45
    #
    # Averaging 40 frames already removes the per-pixel noise that smoothing was
    # there to fight, so there is nothing left for it to buy. Kept as a knob only for
    # the `variance` estimator, which is noisier by construction.
    if smooth_px >= 3:
        ksize = smooth_px | 1
        alpha = cv2.medianBlur(alpha, ksize)
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)

    # Outside the mark there is nothing to undo; pin it exactly so the recovery is an
    # identity there and the composite cannot introduce a seam.
    outside = ~mark_mask
    alpha[outside] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0)

    # Physical upper bound on alpha, per pixel. The background cannot be negative, so
    # C = (1-a)B + aW >= aW, which pins  a <= min over frames and channels of C/W.
    # Without this, a dark background plus a confident alpha drives C - aW below zero,
    # the clamp in `apply` swallows the difference, and the result is a dark blotch --
    # exactly what appeared on the saturated red pillar where the mark crossed it.
    #
    # Capping trades removal for correctness on dark pixels. That is the right way
    # round: a near-black smear reads as damage, while leftover mark on an already
    # dark area is where a white mark is least visible anyway.
    bound = np.clip(
        (frames_bgr.min(axis=0) / np.maximum(w[None, None, :], 1e-6)).min(axis=2),
        0.0, 1.0 - MIN_M).astype(np.float32)
    alpha = np.minimum(alpha, bound)

    scale, energy, energy_before = 1.0, 0.0, 0.0
    if refine:
        scale, energy, energy_before = _refine_scale(
            frames_bgr, alpha, mark_mask, w, bound)
        alpha = np.minimum(np.clip(alpha * scale, 0.0, 1.0 - MIN_M), bound)
        alpha[outside] = 0.0

    # Anything past this is more blocked than division can recover; hand it to the
    # inpainting fallback instead of dividing by a number close to zero.
    opaque = (alpha >= 1.0 - MIN_M) & mark_mask

    m = np.repeat((1.0 - alpha)[:, :, None], 3, axis=2).astype(np.float32)
    k = (alpha[:, :, None] * w[None, None, :]).astype(np.float32)
    k[outside] = 0.0

    return Unblend(m=m, k=k, opaque=opaque, mark_colour=w.copy(),
                   mark_mask=mark_mask.copy(), alpha_scale=scale,
                   residual=energy, residual_before=energy_before)
