"""The intro defect: a fade through black, and the fill invented over it.

What went wrong on real footage. A file opened on black, showed a rating card, then
faded into the first shot. `select='gt(scene,X)'` found nothing in the first 11.7s at
any threshold down to 0.1 -- a fade is gradual, so it never scores -- so the black run
and the bright shot after it were planned as one segment. The watermark region is
masked in every frame of a black run and nothing moves in it, so ProPainter had nothing
to propagate from and filled the hole from that segment's reference frames, which were
bright. The result was a glowing smear the size of the hole over an otherwise black
frame, starting 40 frames before the picture did -- exactly ref_stride * ref_num / 2,
the reach of its global references. Measured at 10/255 on a 1080p output, plainly
visible at 4K.

Two independent fixes, and this file pins both, because either one alone leaves a hole:

1. `_shot_starts` also reports black boundaries, so the black run becomes its own
   segment and bright frames can never be its references. Depends on the boundary
   being found.
2. `_dark_guard` replaces a fill that is brighter than a surround holding no picture
   at all. Depends on nothing, and catches the case where no boundary was found.

The fixture is generated: 1s of black, a 0.7s fade, then testsrc2 -- the transition the
scene score cannot see. Nothing here needs a GPU or the model.

    python tests/test_black_bounds.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmrm.probe import require_tools                                  # noqa: E402
from wmrm.region import Box                                           # noqa: E402
from wmrm.video import (                                              # noqa: E402
    DARK_TILE_MAX,
    ProPainterOpts,
    _dark_guard,
    _segment_plan,
    _shot_starts,
    find_repo,
    run_propainter,
)

FPS = 30.0
BLACK_UNTIL = 30          # frames 0..29 are black; the fade starts at 1.0s
FADE_FRAMES = 21          # 0.7s
TOTAL = 150               # 5s

failures: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}".rstrip())
    if not cond:
        failures.append(name)


def build(ffmpeg: str, out: Path) -> None:
    """One shot that fades in from black. No cut anywhere in it, by construction."""
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size=320x180:rate={FPS:g}", "-t", f"{TOTAL / FPS:g}",
         "-vf", f"fade=t=in:st={BLACK_UNTIL / FPS:g}:d={FADE_FRAMES / FPS:g}",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True)


def dark_tile(h: int = 48, w: int = 120) -> np.ndarray:
    """A tile with no picture in it: black, plus the sensor noise a real one has."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 6, (h, w, 3), dtype=np.uint8)


def masks(h: int = 48, w: int = 120) -> tuple[np.ndarray, np.ndarray]:
    hole = np.zeros((h, w), bool)
    hole[12:36, 30:90] = True
    return hole, ~hole


BOX = Box(220, 20, 60, 24)


class Smearer:
    """A stand-in for the model that always invents a bright fill.

    The defect being fixed is not in the weights, it is in what the pipeline does with
    an answer that cannot be right, so the test does not need the weights -- and must
    not need a GPU either. This is seeded into the worker cache, which is the same
    thing a resident model is to `run_propainter`.
    """

    def __init__(self) -> None:
        self.opts = None
        self.nonfinite = 0
        self.nonfinite_frames: set[int] = set()
        self.calls = 0

    def inpaint(self, frames, mask):
        self.calls += 1
        out = frames.copy()
        out[:, mask > 0] = 210
        return out


def tile_peak(ffmpeg: str, path: Path, frame: int) -> int:
    """Brightest pixel inside the watermark box, in one frame of a finished video."""
    res = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-i", str(path),
         "-vf", f"select='eq(n\\,{frame})',crop={BOX.w}:{BOX.h}:{BOX.x}:{BOX.y}",
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True)
    if res.returncode != 0 or not res.stdout:
        return -1
    return int(np.frombuffer(res.stdout, np.uint8).max())


def end_to_end(ffmpeg: str, td: Path) -> None:
    """The whole engine, with the model replaced by something that always smears."""
    from wmrm import video as V

    src = td / "fade.mp4"
    build(ffmpeg, src)
    dst = td / "out.mp4"

    opts = ProPainterOpts(repo=find_repo(), device="cpu", segment=64, overlap=4,
                          part_frames=64, fp16=False)
    fake = Smearer()
    # Same key _load_worker computes, so it takes this instead of loading 190 MB of
    # weights. If that key ever changes this test fails loudly rather than quietly
    # running the real model on the CPU.
    V._WORKER_CACHE[(str(find_repo()), "cpu", False)] = fake
    try:
        run_propainter(src, dst, box=BOX, dilate_px=4, feather_px=6, margin_px=24,
                       opts=opts, progress=False, resume=False)
    finally:
        V._WORKER_CACHE.clear()

    check("the fake model was the one that ran", fake.calls > 0,
          f"{fake.calls} segment(s)")
    check("every frame reached the output",
          probe_frames(dst) == TOTAL, f"{probe_frames(dst)} of {TOTAL}")

    black_peak = tile_peak(ffmpeg, dst, 5)
    lit_peak = tile_peak(ffmpeg, dst, TOTAL - 5)
    check("the smear does not reach the output over a black frame",
          0 <= black_peak <= 40, f"peak in the box on frame 5: {black_peak}")
    check("and the fill is untouched where there is a picture",
          lit_peak > 150, f"peak in the box on frame {TOTAL - 5}: {lit_peak}")


def probe_frames(path: Path) -> int:
    _, ffprobe = require_tools()
    res = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return int(res.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def main() -> int:
    ffmpeg, _ = require_tools()

    # --- 1. the boundary the scene score cannot see -------------------------------- #
    with tempfile.TemporaryDirectory(prefix="wmrm-black-") as td:
        fix = Path(td) / "fade.mp4"
        build(ffmpeg, fix)

        scene, black = _shot_starts(ffmpeg, fix, 0.3, FPS, black_min_frames=16)
        check("the fade is invisible to the scene score", not scene,
              f"scene cuts: {scene or 'none'}")
        # The boundary is where the picture comes back, i.e. where the fade begins to
        # lift the frame off black. blackdetect stops calling it black a frame or two
        # into the ramp, so this is a window, not a number.
        near = [b for b in black if abs(b - BLACK_UNTIL) <= 4]
        check("the black run is found, and its end lands on the fade",
              bool(near), f"black boundaries: {black}")

        low, _ = _shot_starts(ffmpeg, fix, 0.05, FPS, black_min_frames=0)
        check("lowering the scene threshold does not find it either",
              not any(abs(c - BLACK_UNTIL) <= 4 for c in low),
              f"cuts at threshold 0.05: {low or 'none'}")

        check("both signals off skips the pass entirely",
              _shot_starts(ffmpeg, fix, 0, FPS, black_min_frames=0) == ([], []))

        # --- 2. the black run gets a segment of its own --------------------------- #
        starts = sorted(set(scene) | set(black))
        plan = _segment_plan(starts, TOTAL, 400, 20, 16)
        first = plan[0]
        check("the black run is a segment, not the head of the bright one",
              len(plan) > 1 and abs(first[1] - BLACK_UNTIL) <= 4,
              f"segment 1 = frames {first[0]}-{first[1] - 1} of {TOTAL}")
        check("and it takes no context from across the boundary",
              first[2] == 0 and plan[1][2] == 0,
              f"left context: {[p[2] for p in plan[:2]]}")

    # --- 3. the guard, which needs no boundary at all ------------------------------ #
    hole, ring = masks()
    src = dark_tile()

    invented = src.copy()
    invented[hole] = 180                       # what the model produced: a bright smear
    fixed = _dark_guard(src, invented, hole, ring)
    check("a bright fill over a tile with no picture is replaced",
          fixed is not None and int(fixed[hole].max()) <= DARK_TILE_MAX,
          f"peak in the hole: {int(invented[hole].max())} -> "
          f"{int(fixed[hole].max()) if fixed is not None else 'kept'}")
    check("and nothing outside the hole is touched",
          fixed is not None and np.array_equal(fixed[ring], src[ring]))

    plausible = src.copy()
    plausible[hole] = 3                        # a fill that matches its surround
    check("a fill that already sits in the dark is left alone",
          _dark_guard(src, plausible, hole, ring) is None)

    # The narrowness is the point: over real footage this must never fire, or it
    # replaces the model's picture with a flat patch.
    rng = np.random.default_rng(1)
    picture = rng.integers(40, 210, (48, 120, 3), dtype=np.uint8)
    bright_fill = picture.copy()
    bright_fill[hole] = 255
    check("a bright fill inside real footage is the model's business, not ours",
          _dark_guard(picture, bright_fill, hole, ring) is None)

    # A dark surround with one bright element in it is still a picture: a night shot
    # with a street lamp must not be flattened.
    lamp = dark_tile()
    lamp[0:20, 0:30] = 200
    lamp_fill = lamp.copy()
    lamp_fill[hole] = 120
    check("a dark shot that still holds something bright is left to the model",
          _dark_guard(lamp, lamp_fill, hole, ring) is None)

    # --- 4. and all of it, through the real engine -------------------------------- #
    with tempfile.TemporaryDirectory(prefix="wmrm-black-e2e-") as td:
        end_to_end(ffmpeg, Path(td))

    print(f"\n{len(failures)} failed" if failures else "\nall pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
