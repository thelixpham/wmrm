"""Check that a box actually covers the whole watermark.

This exists because detection is not dependable across watermark types, and the
dangerous failure is always the same: a box slightly too small, leaving a sliver
of the mark in the output where nobody notices until the file has shipped.

Verifying coverage is a much easier problem than detection. We already know
roughly where the mark is; the only question is whether anything mark-like
survives in a ring just outside the proposed box. And it can use two signals at
once, which is what makes it work on marks that defeat the detector:

- **A: consistent signed gradient.** Catches semi-transparent marks, whose edges
  stay put while the scene behind them changes.
- **B: suppressed temporal variance.** Catches opaque marks. Pixels hidden under
  an opaque overlay never change, so their variance over time collapses relative
  to the surrounding picture. This is the signal that finds white-on-white text,
  which has no usable gradient at all and which signal A misses completely.

Neither signal alone covers both cases. Requiring only one to fire is deliberate.

Limits, measured on real footage. Starting from a box covering only the bold half
of a two-part mark (160 px short), iterating the suggestion converged to 16 px
short -- a large improvement but not a guarantee. The faintest edge of a very
low-contrast glyph stays below any threshold that does not also flag the whole
background. And when the background is itself static (fixed camera, plain wall),
no statistic can separate mark from wall at all; that case reports INCONCLUSIVE
rather than inventing a number. Treat this as a strong safety net, not a proof --
`wmrm grid` plus your eyes is the final authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detect import sample_frames
from .probe import probe
from .region import Box


@dataclass
class Coverage:
    box: Box
    ring_px: int
    residual_fraction: float          # of ring pixels that look mark-like
    reach: dict[str, int]             # px the mark extends past each side
    suggested: Box | None             # box grown to contain everything found
    signal_gradient: bool             # signal A fired
    signal_variance: bool             # signal B fired
    inconclusive: bool = False        # signals fired everywhere: cannot discriminate

    @property
    def ok(self) -> bool:
        return not self.inconclusive and not any(self.reach.values())

    @property
    def saturated(self) -> bool:
        """Whether a reach ran out of ring to measure in.

        Nothing outside the ring is ever sampled, so `reach` cannot exceed `ring_px` --
        see the arithmetic at the bottom of `check_coverage`. When it lands exactly
        there the mark was still going when the window ran out, which makes the number
        a floor rather than a distance, and `suggested` a box that is known to be at
        least this big rather than one known to be big enough. Measured on MOGI-108:
        at ring 48 this said `left +48` and the suggestion was still 52 px short; at
        ring 183 it said `left +94` and the suggestion covered the mark.

        A reach that merely reaches the frame edge is not saturated -- there is nothing
        beyond it for the mark to extend into.
        """
        return any(v >= self.ring_px for v in self.reach.values())

    def describe(self) -> str:
        x, y, w, h = self.box.as_tuple()
        lines = [f"box checked   : {x},{y},{w},{h}  (ring {self.ring_px}px)"]
        if self.inconclusive:
            lines.append(
                "result        : INCONCLUSIVE -- almost the whole ring looks "
                "mark-like,\n                so the background itself is static "
                "(fixed camera on a plain\n                surface) and no "
                "statistic can separate mark from wall here.\n"
                "                Judge this one from the grid image by eye."
            )
        elif self.ok:
            lines.append("result        : covered -- nothing mark-like outside the box")
        else:
            sides = ", ".join(f"{k} +{v}px" for k, v in self.reach.items() if v)
            at_least = " at least" if self.saturated else ""
            lines.append(f"result        : UNDER-COVERED -- mark extends{at_least} {sides}")
            if self.suggested:
                sx, sy, sw, sh = self.suggested.as_tuple()
                lines.append(f"suggested box : {sx},{sy},{sw},{sh}")
            if self.saturated:
                lines.append(
                    f"              : the mark was still going where the {self.ring_px}px "
                    f"ring ran out, so\n                that reach is a floor and this box "
                    f"may still be short. Re-check\n                with a wider --ring to "
                    f"measure how far it actually goes."
                )
        fired = [n for n, f in (("gradient", self.signal_gradient),
                                ("low-variance", self.signal_variance)) if f]
        lines.append(f"signals       : {', '.join(fired) if fired else 'none'}")
        lines.append(f"ring pixels flagged: {100 * self.residual_fraction:.2f}%")
        return "\n".join(lines)


def check_coverage(
    src: Path,
    box: Box,
    *,
    samples: int = 30,
    ring: int = 48,
    grad_threshold: float = 2.0,
    persistence: float = 0.85,
    variance_ratio: float = 0.45,
    min_fraction: float = 0.004,
    max_fraction: float = 0.40,
) -> Coverage:
    info = probe(src)
    box = box.clamp(info.width, info.height)

    x0 = max(0, box.x - ring)
    y0 = max(0, box.y - ring)
    x1 = min(info.width, box.x + box.w + ring)
    y1 = min(info.height, box.y + box.h + ring)
    outer = Box(x0, y0, x1 - x0, y1 - y0)

    stack = sample_frames(info, samples, outer)      # (N, h, w, 3)
    gray = stack.mean(axis=3)

    # Signal A -- a pixel-locked edge: mean the signed gradient before taking abs,
    # so scene edges cancel and only fixed ones survive.
    dy, dx = np.gradient(gray, axis=1), np.gradient(gray, axis=2)
    consistent = (np.abs(dy.mean(axis=0)) > grad_threshold) | \
                 (np.abs(dx.mean(axis=0)) > grad_threshold)
    persist = ((np.abs(dy) > grad_threshold) | (np.abs(dx) > grad_threshold)).mean(axis=0)
    sig_a = consistent & (persist >= persistence)

    # Signal B -- variance collapse. An opaque overlay freezes whatever is under
    # it, so its temporal std drops well below the rest of the neighbourhood.
    std = gray.std(axis=0)
    reference = float(np.median(std))
    sig_b = (std < variance_ratio * reference) if reference > 1.0 else np.zeros_like(sig_a)

    marklike = sig_a | sig_b

    # Only the ring counts: inside the box is supposed to be the mark.
    ring_only = np.ones(gray.shape[1:], bool)
    ring_only[box.y - outer.y: box.y - outer.y + box.h,
              box.x - outer.x: box.x - outer.x + box.w] = False
    residual = marklike & ring_only

    total_ring = int(ring_only.sum())
    fraction = float(residual.sum()) / total_ring if total_ring else 0.0

    reach = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    suggested: Box | None = None

    # If nearly the whole ring is flagged, the signals are not separating mark
    # from background -- which happens when the background is itself static (a
    # fixed camera on a plain wall). Then "the mark extends `ring` px in every
    # direction" is just the ring size, and reporting it as a measurement would
    # be worse than saying nothing.
    inconclusive = fraction > max_fraction

    # A handful of stray pixels is noise, not a missed glyph.
    if not inconclusive and fraction >= min_fraction and residual.any():
        ys, xs = np.nonzero(residual)
        bx, by = box.x - outer.x, box.y - outer.y
        reach["left"] = max(0, bx - int(xs.min()))
        reach["right"] = max(0, int(xs.max()) - (bx + box.w - 1))
        reach["top"] = max(0, by - int(ys.min()))
        reach["bottom"] = max(0, int(ys.max()) - (by + box.h - 1))
        if any(reach.values()):
            suggested = Box(
                box.x - reach["left"], box.y - reach["top"],
                box.w + reach["left"] + reach["right"],
                box.h + reach["top"] + reach["bottom"],
            ).clamp(info.width, info.height)

    return Coverage(
        box=box, ring_px=ring, residual_fraction=fraction, reach=reach,
        suggested=suggested,
        signal_gradient=bool((sig_a & ring_only).sum() > total_ring * min_fraction),
        signal_variance=bool((sig_b & ring_only).sum() > total_ring * min_fraction),
        inconclusive=inconclusive,
    )
