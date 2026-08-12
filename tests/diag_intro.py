#!/usr/bin/env python3
"""Is anything glowing where the watermark used to be, over a black frame?

The intro defect this exists to measure: on a frame with no picture in it, the repaired
watermark region came back as a soft bright blob -- ProPainter's invention, because a
black static shot gives it nothing to propagate from, conditioned on bright reference
frames because a fade through black does not register as a scene cut. See BLACK_PIX_TH
and _dark_guard in src/wmrm/video.py.

Point it at a *finished* video and it answers whether the defect is in there:

    tests/diag_intro.py outbox/MOGI-146-clean.mp4
    tests/diag_intro.py out.mp4 --seconds 60          # look further in
    tests/diag_intro.py out.mp4 --corner tl
    tests/diag_intro.py head.mp4 --plan               # also print the segment plan
    tests/diag_intro.py 'https://...signed-r2-url...' # only the head is fetched

No box or preset needed. It finds the blank frames itself, then looks for the brightest
cluster in the corner on exactly those frames -- which is the artifact, since a frame
with no picture has nothing else to be bright. That also makes it a before/after
measurement: run it on the old output and the new one and compare the peak.

`--plan` additionally runs the shot detection over the *whole* file and prints the first
few segments, which is how you check that the intro gets a segment of its own. That is a
full decode, so give it a short clip:

    ffmpeg -v error -i source.mp4 -t 30 -c copy head.mp4
    tests/diag_intro.py head.mp4 --plan
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmrm.probe import probe, require_tools                           # noqa: E402
from wmrm.region import corner_search_roi                             # noqa: E402
from wmrm.video import (                                              # noqa: E402
    DARK_TILE_MAX,
    _segment_plan,
    _shot_starts,
)

# A pixel this dark is not picture. Same number the guard uses, so "clean" here means
# the same thing it means there.
BLANK = DARK_TILE_MAX
# Above this, a cluster on a blank frame is something a viewer sees.
VISIBLE = 24
# Below this it is indistinguishable from encoder noise.
NOISE = 4


def stream_gray(ffmpeg: str, src: str, seconds: float, w: int, h: int):
    """Yield luma planes, one frame at a time. Full resolution, nothing held."""
    proc = subprocess.Popen(
        [ffmpeg, "-v", "error", "-nostdin", "-i", str(src), "-t", f"{seconds:g}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    assert proc.stdout is not None
    nbytes = w * h
    buf = bytearray(nbytes)
    view = memoryview(buf)
    try:
        while True:
            off = 0
            while off < nbytes:
                got = proc.stdout.readinto(view[off:])
                if not got:
                    break
                off += got
            if off < nbytes:
                return
            yield np.frombuffer(buf, np.uint8).reshape(h, w)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        proc.stderr.read()
        proc.wait()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="how much of the head to scan (default 30)")
    ap.add_argument("--corner", default="tr", help="where the watermark sits (default tr)")
    ap.add_argument("--plan", action="store_true",
                    help="also run shot detection over the whole file and print the "
                         "first segments -- a full decode, so use a short clip")
    args = ap.parse_args()

    ffmpeg, _ = require_tools()
    # A path or a URL, because ffmpeg and ffprobe take either -- and a signed R2 link
    # means the head of a 60 GB output can be measured without pulling the whole file.
    # This reads only the first --seconds of it either way.
    src = args.video
    name = Path(src.split("?")[0]).name or src
    info = probe(src)
    roi = corner_search_roi(info.width, info.height, args.corner)
    print(f"{name}: {info.width}x{info.height} @ {info.fps} "
          f"({info.nframes or '?'} frames)")
    print(f"scanning the first {args.seconds:g}s, {args.corner} corner "
          f"{roi.w}x{roi.h} at {roi.x},{roi.y}\n")

    blank = 0
    hits: list[tuple[int, int, int, int, int]] = []   # frame, peak, outside, y, x
    worst = 0                                        # largest excess over the frame
    for i, frame in enumerate(stream_gray(ffmpeg, src, args.seconds,
                                          info.width, info.height)):
        # "No picture in this frame" -- the only frames where an invented fill is
        # visible, and the only ones this can say anything about.
        if int(np.percentile(frame[::4, ::4], 99)) > BLANK:
            continue
        blank += 1
        corner = frame[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]
        peak = int(corner.max())
        # The signature is not brightness, it is brightness *the rest of the frame does
        # not have*. Without this, the first frame of a fade reads as a hit: the picture
        # is legitimately coming up everywhere, corner included.
        slabs = (frame[:roi.y], frame[roi.y + roi.h:],
                 frame[roi.y:roi.y + roi.h, :roi.x],
                 frame[roi.y:roi.y + roi.h, roi.x + roi.w:])
        outside = max((int(s.max()) for s in slabs if s.size), default=0)
        if peak > NOISE and peak - outside > NOISE:
            y, x = np.unravel_index(int(corner.argmax()), corner.shape)
            hits.append((i, peak, outside, int(y) + roi.y, int(x) + roi.x))
            worst = max(worst, peak - outside)

    if not blank:
        print("no blank frames in that range -- nothing this can measure. The defect "
              "only shows over a frame with no picture in it; try --seconds larger, or "
              "the file simply does not open on black.")
        return 0

    print(f"blank frames found          : {blank}")
    print(f"of those, glowing in the box: {len(hits)}")
    print(f"worst excess over the frame : {worst}/255")
    if hits:
        span = (min(h[0] for h in hits), max(h[0] for h in hits))
        print(f"frames affected             : {span[0]}..{span[1]}")
        print("\n  frame   corner   rest of frame   excess   at (y,x)")
        for f, p, o, y, x in hits[:12]:
            print(f"  {f:5d}   {p:6d}   {o:13d}   {p - o:6d}   {y},{x}")
        if len(hits) > 12:
            print(f"  ... and {len(hits) - 12} more")

    print()
    if worst > VISIBLE:
        print(f"VERDICT  the defect is in this file: the watermark box is {worst}/255 "
              f"brighter than anything else on a frame that holds no picture.")
        print("         Re-run the file with the black-run boundaries and the "
              "dark-tile guard in place.")
    elif worst > NOISE:
        print(f"VERDICT  faint residue only (+{worst}/255) -- present but below what a "
              f"viewer sees. This is what the 1080p outputs measured before the fix.")
    else:
        print(f"VERDICT  clean: nothing above encoder noise on {blank} blank frame(s).")

    if args.plan:
        print("\n--- shot detection over the whole file ---")
        scene, black = _shot_starts(ffmpeg, src, 0.3, float(info.fps),
                                    black_min_frames=16, say=lambda m: print(m))
        starts = sorted(set(scene) | set(black))
        print(f"scene cuts       : {scene or 'none'}")
        print(f"black boundaries : {black or 'none'}")
        plan = _segment_plan(starts, info.nframes or 0, 400, 20, 16)
        print(f"{len(plan)} segment(s); the first few:")
        for s, e, lcap, rcap in plan[:6]:
            print(f"  frames {s}-{e - 1}  (+{lcap}/{rcap} context)")
        if black and plan:
            first_black_end = min((b for b in black if b > 0), default=None)
            ok = first_black_end is not None and plan[0][1] == first_black_end
            print(f"\nintro isolated in its own segment: {'YES' if ok else 'NO'}"
                  + ("" if ok else "  <-- the black run shares a segment with the shot "
                                   "after it"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
