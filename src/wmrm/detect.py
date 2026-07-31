"""One-time calibration: find the watermark box, then freeze it into a preset.

This is deliberately *not* part of the per-video run. The watermark is always
the same and always in the same place, so detection runs once, a human confirms
the preview, and every later video just reads the preset. That keeps the risk of
mis-detecting something else at zero for normal use.

Three guards stop burned-in subtitles or description text from being mistaken
for the logo (KNOWLEDGE.md 2.3.1):

1. Geometric  -- only a corner ROI is searched, so bottom/centre text is
   excluded by construction.
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


@dataclass
class Detection:
    box: Box                  # in full-frame coordinates
    roi: Box                  # the search window that was used
    area_percent: float       # of the whole frame
    inside_std: float         # temporal std inside the box
    background_std: float     # temporal std of nearby background
    opacity: str              # "opaque" | "semi"
    n_samples: int
    warnings: tuple[str, ...] = ()

    @property
    def std_ratio(self) -> float:
        return self.inside_std / self.background_std if self.background_std > 1e-6 else 0.0

    def describe(self) -> str:
        x, y, w, h = self.box.as_tuple()
        return (
            f"box       : x={x} y={y} w={w} h={h}\n"
            f"area      : {self.area_percent:.2f}% of frame\n"
            f"samples   : {self.n_samples} frames\n"
            f"temporal  : std inside={self.inside_std:.2f}  background={self.background_std:.2f}"
            f"  ratio={self.std_ratio:.3f}\n"
            f"opacity   : {self.opacity}"
            + ("".join(f"\nWARNING   : {w}" for w in self.warnings) if self.warnings else "")
        )


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
    lo, hi = info.duration * 0.04, info.duration * 0.96
    stamps = np.linspace(lo, hi, num=max(2, n))

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


def detect(
    src: Path,
    *,
    corner: str = "tr",
    samples: int = 40,
    roi_frac: float = 0.30,
    grad_threshold: float = 10.0,
    persistence: float = 0.90,
    max_area_percent: float = 10.0,
    pad: int = 2,
) -> Detection:
    info = probe(src)
    roi = corner_search_roi(info.width, info.height, corner, roi_frac)
    stack = sample_frames(info, samples, roi)     # (N, h, w, 3) float32
    n = stack.shape[0]

    gray = stack.mean(axis=3)                      # (N, h, w)
    dy = np.gradient(gray, axis=1)
    dx = np.gradient(gray, axis=2)

    # Signed-gradient cancellation: take the mean *before* abs. Scene edges flip
    # sign between unrelated frames and average toward zero; a pixel-locked
    # watermark keeps the same signed edge every frame and survives.
    consistent = (np.abs(dy.mean(axis=0)) > grad_threshold) | \
                 (np.abs(dx.mean(axis=0)) > grad_threshold)

    # Guard 2: the edge must also be present in nearly every frame.
    hit = (np.abs(dy) > grad_threshold) | (np.abs(dx) > grad_threshold)
    persist = hit.mean(axis=0)

    score = (consistent & (persist >= persistence)).astype(np.float32)
    if score.max() <= 0:
        raise DetectError(
            f"no watermark found in the {corner} corner of {src.name}.\n"
            f"Try --corner (tl/tr/bl/br), a larger --roi-frac, a lower "
            f"--grad-threshold (now {grad_threshold}), or pass --box x,y,w,h by hand."
        )

    # Blur + re-threshold closes glyph outlines into solid strokes and reaches
    # slightly into the watermark's anti-aliased fringe.
    score = cv2.GaussianBlur(score, (0, 0), 3)
    score /= score.max()

    # Hysteresis, as in Canny. A single threshold under-covers badly on textured
    # backgrounds: only the parts of the badge that contrast strongly register,
    # so the box comes out too small and leaves watermark residue behind. Seed
    # from confident pixels, then grow through weaker ones connected to them.
    strong = score > 0.20
    weak = (score > 0.06).astype(np.uint8)
    n_weak, weak_labels = cv2.connectedComponents(weak, connectivity=8)
    if n_weak > 1:
        touching = np.unique(weak_labels[strong])
        touching = touching[touching != 0]
        binary = np.isin(weak_labels, touching).astype(np.uint8) * 255
    else:
        binary = strong.astype(np.uint8) * 255

    # A logo is several separate strokes ("AI" + "gen" + a plate), so bridge
    # nearby pieces into one blob before labelling. Without this, the largest
    # single component is one glyph and the box under-covers the watermark --
    # which leaves part of it in the output.
    gap = max(9, int(round(min(binary.shape) * 0.06)) | 1)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap)),
    )

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        raise DetectError(f"no connected watermark region in the {corner} corner of {src.name}")

    areas = stats[1:, cv2.CC_STAT_AREA]
    # Union every component that is a meaningful fraction of the biggest one.
    # Specks are noise; a second real chunk of the logo is not.
    keep = np.flatnonzero(areas >= max(4.0, 0.15 * float(areas.max()))) + 1
    x0 = int(stats[keep, cv2.CC_STAT_LEFT].min())
    y0 = int(stats[keep, cv2.CC_STAT_TOP].min())
    x1 = int((stats[keep, cv2.CC_STAT_LEFT] + stats[keep, cv2.CC_STAT_WIDTH]).max())
    y1 = int((stats[keep, cv2.CC_STAT_TOP] + stats[keep, cv2.CC_STAT_HEIGHT]).max())
    bx, by, bw, bh = x0, y0, x1 - x0, y1 - y0

    # temporal statistics, measured before padding
    inside = stack[:, by: by + bh, bx: bx + bw, :]
    inside_std = float(inside.std(axis=0).mean())
    outside = np.ones(gray.shape[1:], bool)
    outside[max(0, by - 8): by + bh + 8, max(0, bx - 8): bx + bw + 8] = False
    background_std = float(stack[:, outside, :].std(axis=0).mean()) if outside.any() else 0.0

    bx, by = max(0, bx - pad), max(0, by - pad)
    bw, bh = bw + 2 * pad, bh + 2 * pad
    box = Box(roi.x + bx, roi.y + by, bw, bh).clamp(info.width, info.height)

    area_percent = 100.0 * box.area() / (info.width * info.height)
    # Guard: an oversized box means the detector latched onto scenery, not a badge.
    if area_percent > max_area_percent:
        raise DetectError(
            f"candidate covers {area_percent:.1f}% of the frame (limit "
            f"{max_area_percent}%) -- that is scenery or a caption, not a corner badge.\n"
            f"Tighten with --roi-frac / --grad-threshold, or pass --box by hand."
        )

    ratio = inside_std / background_std if background_std > 1e-6 else 0.0
    # A truly opaque badge hides the background completely, so its pixels barely
    # change over time. If they do change, the background is bleeding through and
    # the alpha-unblend route becomes available (KNOWLEDGE.md 3.1).
    opacity = "opaque" if ratio < 0.15 else ("semi" if ratio > 0.40 else "unclear")

    # Under-coverage is the dangerous failure: a box that is too small leaves
    # watermark residue, and unlike over-coverage it is easy to miss in a preview.
    # It happens on textured backgrounds, where only the high-contrast parts of
    # the badge register. Flag the shapes that suggest it rather than pretend the
    # detection is trustworthy everywhere.
    warnings: list[str] = []
    ratio_wh = box.w / box.h if box.h else 0.0
    if ratio_wh > 5 or ratio_wh < 0.2:
        warnings.append(
            f"box is very elongated ({box.w}x{box.h}); a partly-detected badge "
            "looks like this. Check the zoom preview closely."
        )
    # A background-busyness warning was tried here and dropped: measured
    # background_std does not discriminate (it fired on a clip where detection
    # was in fact good), and a warning that cries wolf gets ignored.

    return Detection(
        box=box, roi=roi, area_percent=area_percent,
        inside_std=inside_std, background_std=background_std,
        opacity=opacity, n_samples=n, warnings=tuple(warnings),
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

    # Zoom is cropped from the annotated frame: the point of it is to check that
    # the box actually covers the whole watermark, so it needs the box in it.
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
