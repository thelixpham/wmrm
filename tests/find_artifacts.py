"""Find where a processed video's patch misbehaves, and say what it lines up with.

Written for a real failure: a brief wrong-coloured blob in the repaired region of one
clip, in a run whose own acceptance checks all passed. Verification measures whole-clip
aggregates (PSNR inside and outside the mask), and a few bad frames out of 1827
disappear into an average. Eyeballing a 61-second clip to find them again is not a
process that scales, and on an hour of footage it is not a process at all.

Two signals, because they fail differently:

- **fill** -- how far the output patch moved from the input patch, inside the mask.
  Always nonzero: that is the watermark being removed. Spikes mean the patch was
  filled with something unusually unlike what was there.
- **jump** -- how much the output patch changes frame to frame, *minus* how much the
  source changes frame to frame. Near zero when the patch tracks the content. A blob
  that appears and vanishes creates a jump the source does not have, so this catches
  the temporal artifact that `fill` can miss when the fill happens to be dark.

Both are reported as robust z-scores (median and MAD, not mean and sigma) because the
thing being looked for is a handful of outliers in a long tail, and a mean is dragged
by exactly those.

Every flagged frame is then annotated with the two explanations worth telling apart:
a **scene cut**, where flow-guided fill has no valid source and the patch can pull
colour from the wrong shot -- a model limitation -- and a **segment boundary**, where
this tool's own context trimming could be off by a frame. One is upstream's problem,
the other is ours, and the fix is different.

    python tests/find_artifacts.py orig.mp4 orig-clean.mp4 --box 1554,44,283,64
    python tests/find_artifacts.py orig.mp4 orig-clean.mp4 --preset outbox/.presets/x.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from wmrm.probe import probe, require_tools          # noqa: E402
from wmrm.region import Box, Preset, build_region    # noqa: E402


def tile_stream(ffmpeg: str, src: Path, tile: Box):
    """Yield the tile of every frame, as bgr24."""
    nbytes = tile.w * tile.h * 3
    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-nostdin", "-i", str(src),
         "-vf", f"crop={tile.w}:{tile.h}:{tile.x}:{tile.y}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=nbytes * 4)
    assert proc.stdout is not None
    try:
        while True:
            raw = proc.stdout.read(nbytes)
            if len(raw) < nbytes:
                return
            yield np.frombuffer(raw, np.uint8).reshape(tile.h, tile.w, 3)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def scene_cuts(ffmpeg: str, src: Path, threshold: float) -> list[float]:
    res = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(src),
         "-vf", f"select='gt(scene,{threshold})',metadata=print:file=-",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True)
    out = []
    for line in (res.stdout + res.stderr).splitlines():
        if "pts_time:" in line:
            try:
                out.append(float(line.split("pts_time:")[1].split()[0]))
            except (IndexError, ValueError):
                continue
    return sorted(out)


def signals(ffmpeg: str, orig: Path, proc: Path, tile: Box,
            mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two per-frame signals, for any caller that wants numbers not a report.

    fill : mean |output - input| inside the mask. Always nonzero -- that is the
           watermark being removed. Spikes mean an unusual fill.
    jump : how much the output patch moves frame to frame, minus how much the source
           does. Near zero when the patch tracks the content; a blob that appears and
           vanishes shows up here even when its absolute value looks plausible.
    """
    fill: list[float] = []
    jump: list[float] = []
    prev_o = prev_p = None
    for a, b in zip(tile_stream(ffmpeg, orig, tile), tile_stream(ffmpeg, proc, tile)):
        ao = a[mask].astype(np.int16)
        bo = b[mask].astype(np.int16)
        fill.append(float(np.abs(bo - ao).mean()))
        if prev_o is None:
            jump.append(0.0)
        else:
            jump.append(float(np.abs(bo - prev_p).mean())
                        - float(np.abs(ao - prev_o).mean()))
        prev_o, prev_p = ao, bo
    return np.array(fill), np.array(jump)


def robust_z(x: np.ndarray) -> np.ndarray:
    """Deviation in MAD units. Zero-variance input gives zeros, not NaN."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= 1e-9:
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)          # 1.4826: MAD -> sigma for normal data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("processed")
    ap.add_argument("--box", help="x,y,w,h")
    ap.add_argument("--preset", help="preset json written by wmrm detect")
    ap.add_argument("--segment", type=int, default=400,
                    help="--pp-segment the run used, to annotate boundaries (default 400)")
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--ref-stride", type=int, default=10,
                    help="--pp-ref-stride the run used; sets what counts as a shot "
                         "too short for the model to find references inside (default 10)")
    ap.add_argument("--scene-threshold", type=float, default=0.3)
    ap.add_argument("--z", type=float, default=6.0,
                    help="flag frames beyond this many MADs (default 6)")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    ffmpeg, _ = require_tools()
    orig, proc = Path(args.original), Path(args.processed)
    for p in (orig, proc):
        if not p.exists():
            raise SystemExit(f"error: {p} not found")

    info = probe(orig)
    if args.preset:
        # scaled_px the same way the CLI does it, or the dilate/feather/margin that
        # shaped the mask at run time are not the ones measured against here.
        preset = Preset.load(Path(args.preset)).scaled_px(info.width, info.height)
        box = preset.box_for(info.width, info.height)
        dilate, feather, margin = preset.dilate_px, preset.feather_px, preset.margin_px
    elif args.box:
        x, y, w, h = (int(v) for v in args.box.split(","))
        box = Box(x, y, w, h)
        dilate, feather, margin = 5, 12, 64
    else:
        raise SystemExit("error: need --box or --preset")

    region = build_region(box, info.width, info.height, dilate_px=dilate,
                          feather_px=feather, margin_px=margin)
    tile, mask = region.tile, region.inpaint_mask > 0
    fps = float(info.fps)
    print(f"tile {tile.w}x{tile.h} at {tile.x},{tile.y}  "
          f"mask {int(mask.sum())} px  fps {fps:.3f}")

    f, j = signals(ffmpeg, orig, proc, tile, mask)
    n = len(f)
    if n == 0:
        raise SystemExit("error: decoded 0 frames -- do the two videos match?")
    print(f"compared {n} frames (source reports {info.nframes})")
    if info.nframes and abs(n - info.nframes) > 1:
        print(f"WARNING  frame counts differ by {abs(n - info.nframes)}")

    zf, zj = robust_z(f), robust_z(j)
    print(f"fill  median {np.median(f):.2f}  max {f.max():.2f}")
    print(f"jump  median {np.median(j):.2f}  max {j.max():.2f}")

    cuts = scene_cuts(ffmpeg, orig, args.scene_threshold)
    bounds = [i for i in range(args.segment, n, args.segment)]
    print(f"{len(cuts)} scene cut(s), {len(bounds)} segment boundar(y/ies) at frames "
          f"{bounds}")

    # Count before truncating. The first version sliced to --top and then reported
    # len(flagged), so a clip with 90 bad frames printed "25 frame(s) beyond 6.0 MADs"
    # -- a number that looks like a finding and is really the value of --top. Anything
    # that truncates has to say so.
    all_flagged = sorted(
        (i for i in range(n) if zf[i] > args.z or zj[i] > args.z),
        key=lambda i: -max(zf[i], zj[i]))
    flagged = all_flagged[:args.top]
    if len(all_flagged) > len(flagged):
        print(f"\n{len(all_flagged)} frame(s) beyond {args.z} MADs "
              f"({100 * len(all_flagged) / n:.1f}% of the clip); showing the worst "
              f"{len(flagged)}. Raise --top to see the rest.")

    if not flagged:
        print(f"\nno frame beyond {args.z} MADs on either signal -- nothing anomalous "
              f"found. That is not proof the clip is clean: a defect that lasts long "
              f"enough to shift the median hides from this test.")
        return 0

    print(f"\n{len(all_flagged)} frame(s) beyond {args.z} MADs"
          + (f", worst {len(flagged)} shown:" if len(all_flagged) > len(flagged)
             else ", worst first:"))
    print(f"{'frame':>7} {'time':>8} {'fill':>7} {'z':>6} {'jump':>7} {'z':>6}  "
          f"{'nearest cut':>14}  {'nearest boundary':>18}")
    for i in sorted(flagged):
        t = i / fps
        cut_d = min((abs(t - c) for c in cuts), default=float("inf"))
        cut_s = (f"{cut_d * fps:.0f}f ({cut_d:.2f}s)" if cut_d < float("inf") else "-")
        b_d = min((abs(i - b) for b in bounds), default=None)
        b_s = f"{b_d}f" if b_d is not None else "-"
        print(f"{i:>7} {t:>7.2f}s {f[i]:>7.2f} {zf[i]:>6.1f} {j[i]:>7.2f} "
              f"{zj[i]:>6.1f}  {cut_s:>14}  {b_s:>18}")

    # Which explanation the flagged frames cluster on, not whether any single one is
    # near something -- with a cut every two seconds, "near a cut" is true of almost
    # any frame by chance.
    #
    # The shot test is the one that matters, and getting it wrong once is why it is
    # spelled out here. A first version asked only "is this frame close to a cut",
    # which missed the real case completely: the artifact filled a whole 30-frame shot
    # between two cuts, and its middle frames are half a second from either one, so
    # they failed a proximity test while being the clearest evidence in the run. The
    # defect is not located *at* the cut; it covers shots too short for the model to
    # find references inside, so the question to ask about a frame is which shot it is
    # in and how long that shot is.
    shot_starts = [0] + [int(round(c * fps)) for c in cuts] + [n]
    shot_of = np.zeros(n, int)
    shot_len = np.zeros(n, int)
    for a, b in zip(shot_starts, shot_starts[1:]):
        a, b = max(0, a), min(n, b)
        if b > a:
            shot_of[a:b] = a
            shot_len[a:b] = b - a

    near_bound = sum(1 for i in flagged
                     if min((abs(i - b) for b in bounds), default=9e9) <= args.overlap)
    # A shot is "short" relative to what the model needs to work inside it: local
    # neighbours plus at least a few global references, i.e. a few times ref_stride.
    short = 3 * args.ref_stride
    in_short = sum(1 for i in flagged if 0 < shot_len[i] <= short)
    hit_shots = sorted({int(shot_of[i]) for i in flagged})

    print(f"\nof {len(flagged)} flagged: {near_bound} within {args.overlap} frames of a "
          f"segment boundary, {in_short} inside a shot of <= {short} frames")
    print(f"shots containing flagged frames (start frame -> length): "
          + ", ".join(f"{s}->{int(shot_len[s])}f" for s in hit_shots))

    if near_bound and near_bound >= len(flagged) / 2:
        print("VERDICT  clustered on segment boundaries -- suspect the context "
              "trimming in video.py, i.e. ours.")
    elif in_short and in_short >= len(flagged) / 2:
        print("VERDICT  clustered inside short shots. The model fills from other "
              "frames, and in a shot this short its reference frames come from "
              "outside it -- a different scene -- while the flow across the cut is "
              "meaningless. Fix is to segment on cuts, so a segment never spans one "
              "and every reference is same-shot.")
    elif any(min((abs(i - s) for s in shot_starts), default=9e9) <= 5 for i in flagged):
        print("VERDICT  clustered at shot starts rather than through whole shots -- "
              "the first frames after a cut have one-sided context. Same fix, cheaper: "
              "do not carry overlap across a cut.")
    else:
        print("VERDICT  no clustering on either. Look at the frames listed above "
              "before changing anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
