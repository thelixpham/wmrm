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
2. Persistence -- a candidate must hold its edge in most sampled frames (see
   DEFAULT_PERSISTENCE). The logo does; captions that appear for part of the
   clip do not. Deliberately "most" and not "~every": a semi-transparent mark
   loses its edge over some backgrounds, so demanding ~every frame throws the
   mark away rather than the captions.
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

# Smallest connected component that counts as part of the mark rather than noise. Kept
# absolute on purpose -- see the long note at the `keep =` line in `_box_at`.
_MIN_COMPONENT_PX = 4.0

# Fraction of sampled frames a pixel must hold its edge in (guard 2 in the module
# docstring). This was 0.90, and 0.90 sits on a cliff.
#
# The trap is that the test is a fraction of *all* sampled frames, so raising --samples
# makes it strictly harder to pass: more frames drawn across a whole film cover more
# varied backgrounds, and a semi-transparent mark loses its edge over some of them. The
# same knob therefore decides both how much evidence there is and how much is demanded,
# and the second effect wins. `_grow_to_cover` in cli.py already carries this lesson for
# the coverage check; detection had not learned it.
#
# Measured across eight clips at samples=20/40/60 -- detected box width, correct value in
# brackets. `original` is the control: it carries no mark at all, so any box there is a
# false positive and the right answer is the DetectError below.
#
#     clip       [want]  p=0.90           p=0.80        p=0.70          p=0.60        p=0.50
#     input      [284]   285 283 284      284 284 284   285 285 285     284 284 284   284 284 284
#     test       [284]   218  22  53 <--  283 283 283   284 284 284     284 284 284   284 284 284
#     1sods      [284]   284 283 283      284 284 284   284 284 284     285 284 284   285 285 284
#     sone       [211]   211 211 211      211 211 211   211 211 211     211 211 211   211 211 211
#     snos       [211]    81  95  19 <--  208 210 210   211 211 211     211 211 211   211 211 211
#     fx-busy     [84]    99  99  99       99  99  99    99  99  99      99  99  99    99  99  99
#     fx-smooth   [81]   108 108 108       95  95  95   107 107  95 <--  95  95  95    95  95  95
#     original   [none]  none none none   none none n.   19 none none    19 none none  191  97  98 <--
#
# 0.90 is a cliff: two of five real clips collapse to a fragment, and *which* fragment
# depends on the sample count -- that is the "works on some videos, not others" report.
# The mechanism is that raising --samples raises the bar this test sets, because the bar
# is a fraction of however many frames were drawn.
#
# Everything from 0.80 down to 0.60 gets all five real clips right at every sample count.
# 0.50 is the far edge: it finds a 191x261 blob (2.4% of frame) on a clip with no mark.
#
# 0.60 is the default because it holds the widest box on the marks that were hardest to
# see -- test and snos come out 1-3 px wider than at 0.80 -- while measuring as stable as
# 0.80 on the fixtures. Its one cost is a 19x19 speck on the mark-free control at n=20,
# and _MIN_AREA_PERCENT below exists to reject exactly that.
#
# Verified against ground truth taken by differencing each fixture against its unmarked
# twin, which is exact rather than statistical:
#
#     fixture     true mark         detected at 0.60      margin
#     busy        384,12,84,36      377, 4, 99,51         7-8 px on every side
#     smooth      384,12,81,36      377, 6, 95,49         6-7 px on every side
#
# What the guard still buys: it rejects burned-in subtitles and text that is on screen for
# only part of the clip. 0.60 demands the mark in six sampled frames out of ten spread
# across the whole runtime; a caption living in one or two scenes lands near 0.05. The
# margin is an order of magnitude, so this is not the knob that lets captions in.
DEFAULT_PERSISTENCE = 0.60

# Smallest candidate that can be a watermark rather than noise, as a percentage of the
# frame. A pure floor, not a tuning knob -- it exists because the persistence default sits
# low enough to occasionally bridge a few static specks on footage with no mark at all,
# and a box built from specks is never right.
#
# Measured areas, same corpus: every real mark lands between 0.68% (snos/sone, 211x67 on
# 1080p) and 2.65% (fx-smooth), while the false positive at 0.60 is 19x19 = 0.017%. 0.05%
# sits 13x below the smallest true mark and 3x above the speck, so it separates them
# without being tuned to either. On 1080p that is a 32x32 box; a real mark smaller than
# that wants --box, not a lower floor.
_MIN_AREA_PERCENT = 0.05


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


# How much wider the search window gets on the one retry, and how far it may go. 0.30 ->
# 0.48 -> stop; a window past ~half the frame is no longer a corner search and starts
# competing with the scene for the largest consistent-edge blob.
_ROI_WIDEN = 1.6
_MAX_ROI_FRAC = 0.50
# Most the box may grow on that retry before the growth reads as background rather than
# as mark the old window was hiding. See the note at the call site for the two measured
# cases this sits between: 1.2x (a genuinely clipped mark) and 6.3x (wall).
_WIDEN_LIMIT = 3.0


def _roi_edges_touched(box: Box, roi: Box, info: VideoInfo, pad: int) -> tuple[str, ...]:
    """Names of the ROI edges the box is flush against, ignoring frame edges.

    `pad` is the slack `_box_at` already added around the component union, so a box that
    was clipped comes back sitting exactly `pad` px inside the window rather than on it.
    """
    slack = pad + 1
    touched = []
    if box.x - roi.x <= slack and roi.x > 0:
        touched.append("left")
    if (roi.x + roi.w) - (box.x + box.w) <= slack and roi.x + roi.w < info.width:
        touched.append("right")
    if box.y - roi.y <= slack and roi.y > 0:
        touched.append("top")
    if (roi.y + roi.h) - (box.y + box.h) <= slack and roi.y + roi.h < info.height:
        touched.append("bottom")
    return tuple(touched)


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
    # Union every component big enough not to be noise. The bar is absolute, and that
    # matters more than it looks: it used to be `max(4.0, 0.15 * areas.max())`, relative
    # to the biggest component, which made the box *shrink* as the threshold dropped.
    #
    # Mechanism, measured on a two-part mark (faint "SOFT DEMAND" logo beside a bold
    # rating badge). The MORPH_CLOSE above merges the badge's glyphs into one blob as
    # soon as they are all above threshold, so `areas.max()` jumps -- and the 15% bar
    # jumps with it, past the faint half:
    #
    #     thr 7 -> badge still 4 fragments, max 642  -> bar  96 -> logo (555 px) kept
    #     thr 5 -> badge merged into one blob, 4140  -> bar 621 -> logo (592 px) DROPPED
    #     thr 4/3/2.5 -> bar 668/722/741            -> logo dropped every time
    #
    # The logo was never lost by hysteresis: its post-normalisation peak score stays
    # >=0.72 throughout, far above the `strong > 0.20` seed cut. It was dropped here and
    # only here. That gave a box that was non-monotonic in threshold -- wide at 7, narrow
    # at 5..2.5 -- so the sweep's plateau rule below saw a 4-long narrow plateau beat a
    # 1-long wide one and picked the box covering the badge alone, leaving the logo in the
    # output. It surfaced as intermittent because whether the wide box happens to land on
    # two adjacent thresholds depends on the sampled frames: at samples=12/60 it did and
    # the box came out right, at 20/40 it did not. Over 120 random 40-frame subsets, 41%
    # returned the narrow box.
    #
    # With an absolute bar the wide box holds a 3-4 threshold plateau and wins on its own
    # merits: 284/285/283/284 px at samples=12/20/40/60, i.e. correct every time.
    #
    # Nothing here bounds how far the union may stretch -- `plausible()` in `detect()`
    # does that, and it has to, because the bound belongs with the sweep that can compare
    # thresholds rather than with one threshold in isolation.
    keep = np.flatnonzero(areas >= _MIN_COMPONENT_PX) + 1
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
    persistence: float = DEFAULT_PERSISTENCE,
    max_area_percent: float = 10.0,
    pad: int = 2,
    _widen: bool = True,          # internal: False on the one retry, so it cannot recurse
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
    floor = _MIN_AREA_PERCENT / 100.0 * frame_area

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
        # beside it, and the box then covers only half the watermark, so no single
        # threshold is safe to hardcode.
        #
        # Which direction helps is *not* fixed, and a note here used to claim it was
        # (that low thresholds include more of the mark). Measured on a two-part mark it
        # ran the other way -- thr 7 gave the full 277 px box, thr 5..2.5 gave the 116 px
        # badge alone, thr 2 blew up to 268x209 of background. Lowering the threshold
        # grows the bold half faster than the faint one, and used to lose the faint one
        # outright; see the `keep =` note in `_box_at`. Sweeping both ways and comparing
        # is the point -- do not shortcut it by assuming a direction.
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
            return (box.area() >= floor
                    and box.area() <= limit
                    and box.area() <= 0.35 * roi_area
                    and box.w <= 0.75 * roi.w
                    and box.h <= 0.75 * roi.h)

        # Adjacency is by position in THRESHOLD_SWEEP, not by position in this list.
        # `found` is filtered -- by `plausible`, and by `candidate` returning None -- so
        # indexing into it splices thresholds that are not neighbours at all into
        # "neighbours", and two boxes from either side of a gap then vouch for each
        # other. It is a plateau test; a plateau in a curve with points missing is not
        # one. Measured on the `detail` fixture: adding the area floor removed a speck
        # from the middle of the list, which silently re-paired the survivors and flipped
        # the selection to a box 22 px shorter.
        found = [(i, thr, box) for i, thr in enumerate(THRESHOLD_SWEEP)
                 if (box := candidate(thr)) is not None and plausible(box)]
        best: tuple[float, Box] | None = None
        for i, thr, box in found:
            stable = any(
                abs(box.area() - other.area()) <= 0.10 * max(box.area(), other.area())
                for j, _, other in found if abs(i - j) == 1
            )
            if stable and (best is None or box.area() > best[1].area()):
                best = (thr, box)
        if best is None and found:
            _, thr0, box0 = found[0]      # nothing stable: take the most conservative
            best = (thr0, box0)
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

    # A box flush against an inside edge of the search window was not measured, it was
    # cut off: nothing beyond the ROI is ever looked at, so the mark may well continue
    # there. Measured on the `detail` fixture, whose true mark is 384,430,84,36 -- the
    # br window at roi_frac 0.30 starts at y=448, 18 px below the top of the mark, and
    # detection returned y=448 at every threshold and every persistence value. The box
    # was short by exactly the amount the window had hidden, and nothing said so.
    #
    # Widening and re-running costs one more sampling pass and is the only way to tell
    # a mark that stops at the edge from one that is being clipped by it. Frame edges
    # are excluded: there is nothing beyond those to widen into.
    pinned = _roi_edges_touched(box, roi, info, pad)
    if pinned and _widen and roi_frac < _MAX_ROI_FRAC:
        wider = min(_MAX_ROI_FRAC, roi_frac * _ROI_WIDEN)
        print(f"[wmrm] the box sits on the {'/'.join(pinned)} edge of the "
              f"{roi_frac:g} search window, so it may be clipped -- re-running at "
              f"--roi-frac {wider:g}", file=sys.stderr)
        try:
            retry = detect(
                src, corner=corner, samples=samples, roi_frac=wider,
                grad_threshold=grad_threshold, persistence=persistence,
                max_area_percent=max_area_percent, pad=pad, _widen=False,
            )
        except DetectError:
            retry = None                  # the wider window found nothing; keep this one
        # Only if it actually grew, and only by an amount a clipped mark could account
        # for. A wider window admits more background as well as more mark, and at a low
        # persistence the bottom of the threshold sweep will find *something* in it --
        # so "bigger" alone is not evidence. Measured on the mark-free control clip: the
        # 0.30 window gave an 88x76 blob of wall pinned to its left edge, and widening to
        # 0.48 turned it into 245x173, a 6.3x jump that the retry then adopted. The mark
        # it was built for needs 1.2x (the `detail` fixture, 99x27 clipped -> 87x37 whole).
        # Same reasoning as _GROW_LIMIT in cli.py: a sudden multiple reads as background.
        if retry is not None and box.area() < retry.box.area() <= _WIDEN_LIMIT * box.area():
            return retry
        if retry is not None and retry.box.area() > box.area():
            print(f"[wmrm] the wider window gave {retry.box.w}x{retry.box.h}, "
                  f"{retry.box.area() / box.area():.1f}x the area -- that reads as "
                  f"background rather than a clipped mark, keeping "
                  f"{box.w}x{box.h}", file=sys.stderr)
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
    # Still pinned after the retry above (or with the retry disabled), which means the
    # window cannot be widened any further and the box may genuinely be cut off.
    if pinned:
        warnings.append(
            f"box sits on the {'/'.join(pinned)} edge of the search window, so it may "
            f"be clipped rather than complete -- nothing outside the window is looked "
            f"at. Re-run with a larger --roi-frac, or set --box by hand."
        )
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
