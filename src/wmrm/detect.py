"""One-time calibration: find the watermark box, then freeze it into a preset.

This is deliberately *not* part of the per-video run. The watermark is always
the same and always in the same place, so detection runs once, a human confirms
the preview, and every later video just reads the preset. That keeps the risk of
mis-detecting something else at zero for normal use.

Three guards stop burned-in subtitles or description text from being mistaken
for the logo (KNOWLEDGE.md 2.3.1):

1. Geometric  -- only a corner ROI is searched, so bottom/centre text is
   excluded by construction. Verified on real footage: a burned-in Japanese
   disclaimer along the bottom edge was never even looked at.
2. Persistence -- a candidate must be present in ~every sampled frame. The logo
   is; captions that appear for part of the clip are not.
3. Human     -- a preview PNG is written and the tool stops. Nothing runs
   automatically off a guess.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .probe import VideoInfo, probe, require_tools
from .region import Box, corner_search_roi


class DetectError(RuntimeError):
    pass


# Swept high to low when no threshold is given. A high threshold sees only the
# boldest mark; lowering it picks up faint ones too, and eventually scenery.
THRESHOLD_SWEEP = (10.0, 7.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5)


@dataclass
class Detection:
    box: Box                  # in full-frame coordinates
    roi: Box                  # the search window that was used
    area_percent: float       # of the whole frame
    inside_std: float         # temporal std inside the box
    background_std: float     # temporal std of nearby background
    opacity: str              # "opaque" | "semi" | "unclear"
    n_samples: int
    threshold: float = 0.0    # gradient threshold actually used
    warnings: tuple[str, ...] = ()

    @property
    def std_ratio(self) -> float:
        return self.inside_std / self.background_std if self.background_std > 1e-6 else 0.0

    def describe(self) -> str:
        x, y, w, h = self.box.as_tuple()
        text = (
            f"box       : x={x} y={y} w={w} h={h}\n"
            f"area      : {self.area_percent:.2f}% of frame\n"
            f"samples   : {self.n_samples} frames, gradient threshold {self.threshold:g}\n"
            f"temporal  : std inside={self.inside_std:.2f}  "
            f"background={self.background_std:.2f}  ratio={self.std_ratio:.3f}\n"
            f"opacity   : {self.opacity}"
        )
        return text + "".join(f"\nWARNING   : {w}" for w in self.warnings)


def sample_frames(info: VideoInfo, n: int, roi: Box | None = None) -> np.ndarray:
    """Grab `n` frames spread evenly across the whole duration.

    Spread matters: the persistence test only means anything if the samples come
    from all over the clip rather than one scene. `-ss` before `-i` makes each
    extraction a cheap keyframe seek.
    """
    ffmpeg, _ = require_tools()
    if info.duration <= 0:
        raise DetectError(f"{info.source}: unknown duration, cannot sample")

    # Avoid the very start/end: fades and black frames carry no useful gradient.
    stamps = np.linspace(info.duration * 0.04, info.duration * 0.96, num=max(2, n))

    frames: list[np.ndarray] = []
    nbytes = info.width * info.height * 3
    for t in stamps:
        res = subprocess.run(
            [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", str(info.source),
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True,
        )
        if res.returncode != 0 or len(res.stdout) < nbytes:
            continue
        frame = np.frombuffer(res.stdout[:nbytes], np.uint8).reshape(
            info.height, info.width, 3
        )
        if roi is not None:
            frame = frame[roi.y: roi.y + roi.h, roi.x: roi.x + roi.w]
        frames.append(frame.copy())

    if len(frames) < 2:
        raise DetectError(
            f"{info.source}: could only extract {len(frames)} frame(s); need at least 2"
        )
    return np.stack(frames).astype(np.float32)


def _box_at(dy: np.ndarray, dx: np.ndarray, thr: float, persistence: float,
            pad: int, shape: tuple[int, int]) -> Box | None:
    """Candidate box in ROI-local coordinates for one threshold, or None."""
    # Signed-gradient cancellation: take the mean *before* abs. Scene edges flip
    # sign between unrelated frames and average toward zero; a pixel-locked
    # watermark keeps the same signed edge every frame and survives.
    consistent = (np.abs(dy.mean(axis=0)) > thr) | (np.abs(dx.mean(axis=0)) > thr)
    # Guard 2: the edge must also be present in nearly every frame.
    persist = ((np.abs(dy) > thr) | (np.abs(dx) > thr)).mean(axis=0)

    score = (consistent & (persist >= persistence)).astype(np.float32)
    if score.max() <= 0:
        return None

    # Blur + re-threshold closes glyph outlines into solid strokes and reaches
    # slightly into the watermark's anti-aliased fringe.
    score = cv2.GaussianBlur(score, (0, 0), 3)
    score /= score.max()

    # Hysteresis, as in Canny. A single cut-off under-covers on textured
    # backgrounds: only the high-contrast parts of the mark register, so the box
    # comes out too small and leaves residue. Seed from confident pixels, then
    # grow through weaker ones connected to them.
    strong = score > 0.20
    weak = (score > 0.06).astype(np.uint8)
    n_weak, weak_labels = cv2.connectedComponents(weak, connectivity=8)
    if n_weak > 1:
        touching = np.unique(weak_labels[strong])
        touching = touching[touching != 0]
        binary = np.isin(weak_labels, touching).astype(np.uint8) * 255
    else:
        binary = strong.astype(np.uint8) * 255

    # A logo is several separate strokes, so bridge nearby pieces into one blob
    # before labelling. Without this the largest single component is one glyph.
    gap = max(9, int(round(min(binary.shape) * 0.06)) | 1)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap)),
    )

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    # Union every component that is a meaningful fraction of the biggest one.
    # Specks are noise; a second real chunk of the logo is not.
    keep = np.flatnonzero(areas >= max(4.0, 0.15 * float(areas.max()))) + 1
    x0 = int(stats[keep, cv2.CC_STAT_LEFT].min())
    y0 = int(stats[keep, cv2.CC_STAT_TOP].min())
    x1 = int((stats[keep, cv2.CC_STAT_LEFT] + stats[keep, cv2.CC_STAT_WIDTH]).max())
    y1 = int((stats[keep, cv2.CC_STAT_TOP] + stats[keep, cv2.CC_STAT_HEIGHT]).max())

    return Box(max(0, x0 - pad), max(0, y0 - pad),
               (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad).clamp(shape[1], shape[0])


def detect(
    src: Path,
    *,
    corner: str = "tr",
    samples: int = 40,
    roi_frac: float = 0.30,
    grad_threshold: float | None = None,   # None = sweep
    persistence: float = 0.90,
    max_area_percent: float = 10.0,
    pad: int = 2,
) -> Detection:
    info = probe(src)
    roi = corner_search_roi(info.width, info.height, corner, roi_frac)
    stack = sample_frames(info, samples, roi)      # (N, h, w, 3) float32

    gray = stack.mean(axis=3)                      # (N, h, w)
    dy = np.gradient(gray, axis=1)
    dx = np.gradient(gray, axis=2)
    shape = gray.shape[1:]
    frame_area = info.width * info.height
    limit = max_area_percent / 100.0 * frame_area

    def candidate(thr: float) -> Box | None:
        return _box_at(dy, dx, thr, persistence, pad, shape)

    chosen_thr: float | None = grad_threshold
    if grad_threshold is not None:
        local = candidate(grad_threshold)
    else:
        # Sweep the whole range, then pick the largest *stable* box still under the
        # area limit. Sampling frames is the expensive part and is already done, so
        # evaluating every threshold is nearly free.
        #
        # Why sweep: a threshold tuned for a bold mark silently misses a faint one
        # beside it, and the box then covers only half the watermark. Measured on
        # real footage: at 10 only the rating badge was found (116 px wide), at 2
        # the faint studio logo beside it was included too (283 px).
        #
        # Why "largest stable" and not "first plateau": the area curve has several
        # plateaus -- one per mark it has managed to pick up. Stopping at the first
        # one finds only the boldest mark. A plateau means two adjacent thresholds
        # agree to within 10%, which distinguishes a real mark from a one-off
        # blow-up at the bottom of the range; among those, the largest covers every
        # mark. Over-covering costs a little speed, under-covering leaves residue.
        # A corner mark occupies a small part of the search window. Anything
        # filling most of the ROI is the background itself, not a watermark --
        # and that really happens: a smooth static gradient *is* a pixel-locked
        # consistent edge, so at a low enough threshold the whole sky qualifies.
        # Measured: without this guard a soft-sky clip returned a 112x192 box
        # (7% of frame, the full ROI height) instead of the 84x36 badge.
        roi_area = roi.area()

        def plausible(box: Box) -> bool:
            return (box.area() <= limit
                    and box.area() <= 0.35 * roi_area
                    and box.w <= 0.75 * roi.w
                    and box.h <= 0.75 * roi.h)

        found = [(thr, box) for thr in THRESHOLD_SWEEP
                 if (box := candidate(thr)) is not None and plausible(box)]
        best: tuple[float, Box] | None = None
        for i, (thr, box) in enumerate(found):
            stable = any(
                abs(box.area() - other.area()) <= 0.10 * max(box.area(), other.area())
                for j, (_, other) in enumerate(found) if abs(i - j) == 1
            )
            if stable and (best is None or box.area() > best[1].area()):
                best = (thr, box)
        if best is None and found:
            best = found[0]               # nothing stable: take the most conservative
        if best is not None:
            chosen_thr, local = best[0], best[1]
        else:
            local = None

    if local is None:
        hint = (f"a lower --grad-threshold (swept {THRESHOLD_SWEEP[0]:g}"
                f"..{THRESHOLD_SWEEP[-1]:g})" if grad_threshold is None
                else f"a lower --grad-threshold (now {grad_threshold:g})")
        raise DetectError(
            f"no watermark found in the {corner} corner of {src.name}.\n"
            f"Try --corner (tl/tr/bl/br), a larger --roi-frac, {hint}, "
            f"or pass --box x,y,w,h by hand."
        )

    box = Box(roi.x + local.x, roi.y + local.y, local.w, local.h).clamp(
        info.width, info.height)
    area_percent = 100.0 * box.area() / frame_area

    inside_std = float(
        stack[:, local.y: local.y + local.h, local.x: local.x + local.w, :]
        .std(axis=0).mean()
    )
    outside = np.ones(shape, bool)
    outside[max(0, local.y - 8): local.y + local.h + 8,
            max(0, local.x - 8): local.x + local.w + 8] = False
    background_std = (float(stack[:, outside, :].std(axis=0).mean())
                      if outside.any() else 0.0)

    ratio = inside_std / background_std if background_std > 1e-6 else 0.0
    # A truly opaque badge hides the background completely, so its pixels barely
    # change over time. If they do change, the background is bleeding through and
    # the alpha-unblend route becomes available (KNOWLEDGE.md 3.1).
    opacity = "opaque" if ratio < 0.15 else ("semi" if ratio > 0.40 else "unclear")

    # Under-coverage is the dangerous failure: too small a box leaves residue,
    # and unlike over-coverage it is easy to miss in a preview.
    warnings: list[str] = []
    ratio_wh = box.w / box.h if box.h else 0.0
    if ratio_wh > 5 or ratio_wh < 0.2:
        warnings.append(
            f"box is very elongated ({box.w}x{box.h}); a partly-detected mark "
            "looks like this. Check the zoom preview closely."
        )
    # A background-busyness warning was tried here and dropped: measured
    # background_std did not discriminate (it fired on a clip where detection was
    # in fact good), and a warning that cries wolf gets ignored.

    return Detection(
        box=box, roi=roi, area_percent=area_percent,
        inside_std=inside_std, background_std=background_std,
        opacity=opacity, n_samples=stack.shape[0],
        threshold=float(chosen_thr or 0.0), warnings=tuple(warnings),
    )


def write_preview(src: Path, box: Box, out_png: Path, *, roi: Box | None = None,
                  at: float | None = None, zoom_png: Path | None = None) -> None:
    """Render one real frame with the box drawn on it, for human confirmation."""
    ffmpeg, _ = require_tools()
    info = probe(src)
    t = at if at is not None else max(0.0, info.duration / 2)
    res = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", str(src),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True,
    )
    nbytes = info.width * info.height * 3
    if res.returncode != 0 or len(res.stdout) < nbytes:
        raise DetectError(f"could not extract a preview frame from {src}")
    frame = np.frombuffer(res.stdout[:nbytes], np.uint8).reshape(
        info.height, info.width, 3
    ).copy()

    if roi is not None:
        cv2.rectangle(frame, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h),
                      (0, 200, 255), 1)
    cv2.rectangle(frame, (box.x, box.y), (box.x + box.w, box.y + box.h),
                  (0, 0, 255), 1)

    # Zoom is cropped from the annotated frame: its whole purpose is checking that
    # the box covers the entire watermark, so it needs the box drawn in it.
    if zoom_png is not None:
        m = 40
        crop = frame[max(0, box.y - m): box.y + box.h + m,
                     max(0, box.x - m): box.x + box.w + m]
        if crop.size:
            scale = max(1, min(8, 480 // max(1, max(crop.shape[:2]))))
            big = cv2.resize(crop, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_NEAREST)
            zoom_png.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(zoom_png), big)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_png), frame):
        raise DetectError(f"could not write preview to {out_png}")
    print(f"[wmrm] preview -> {out_png}", file=sys.stderr)
