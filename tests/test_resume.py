"""A resumed run must produce the same video as a run that was never interrupted.

Resuming is only worth having if it is exact. The cheap version of this feature --
keep whatever was encoded, start the model at the next frame -- is wrong in a way
nobody would notice until much later: the model's answer for a frame depends on the
block of frames it was handed, so restarting mid-segment gives that segment different
context and different pixels than a whole run would have produced. The seam is a few
frames wide, in the region the whole tool exists to repair, in a file that is hours
long. Nobody is going to find that by watching.

So the property under test is byte-identity, not plausibility:

1. **Exactness.** A run interrupted after two segments and then resumed produces a
   file byte-identical to one made in a single pass. That covers the segment restart
   (the resumed run recomputes the segment the crash landed in, from its start), the
   frame-exact seek both the decoder and each part's compositor rely on, and the
   concatenation of parts made by two different processes.
2. **It actually skips work.** The resumed run must call the model fewer times than a
   whole run. Without this the first check would also pass for a "resume" that
   quietly redid everything.
3. **Incomplete parts are not trusted.** The part the crash was in the middle of must
   be gone, not left on disk looking finished.

The model is stubbed, as in tests/test_video_order.py and for the same reasons -- it
is slow, it wants a GPU, and it is not what is under test. Everything else is real:
real ffmpeg, real parts, real seeking.

    python tests/test_resume.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(HERE))

from wmrm import video as V                                        # noqa: E402
from wmrm.probe import probe                                        # noqa: E402
from test_video_order import (BOX, FRAMES, H, StubWorker, W,        # noqa: E402
                              decode, write_source)

SEGMENT = 17          # -> a 4-segment plan over 60 frames
OVERLAP = 5
PART = 12             # -> 5 parts, so the concatenation is exercised too
CRASH_AFTER = 2       # model calls to allow before the interruption
# Frame markers step by this much, cycling rather than climbing. Check 4 asks whether
# a repaired tile landed on its own frame, and the answer has to survive an x264 pass:
# tests/test_video_order.py can afford a step of 4 because it averages a large flat
# corner and only needs monotonicity, but measured here, a 4-level step leaves 2.5
# levels of encode noise against a 4-level signal -- too close to call. Cycling means
# the step can be large without the values running past 255 over a long clip.
MARK = 24
CYCLE = 9


def marker_of(i: int) -> int:
    return MARK + (i % CYCLE) * MARK


def marked_frames() -> np.ndarray:
    """Like tests/test_video_order.py's clip, with a marker built to survive x264."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.empty((FRAMES, H, W, 3), np.uint8)
    for t in range(FRAMES):
        out[t, :, :, 0] = marker_of(t)
        out[t, :, :, 1] = np.clip(np.sin(xx * 0.05 + t * 0.3) * 100 + 128, 0, 255)
        out[t, :, :, 2] = np.clip(np.cos(yy * 0.07 - t * 0.2) * 100 + 128, 0, 255)
    return out


class CrashingStub(StubWorker):
    """A stub that dies mid-run, the way an OOM or a killed container would."""

    def __init__(self, after: int) -> None:
        super().__init__()
        self.after = after

    def inpaint(self, frames, mask, *, progress=None):
        if len(self.calls) >= self.after:
            raise RuntimeError("simulated crash")
        return super().inpaint(frames, mask, progress=progress)


def run(src: Path, dst: Path, *, worker, resume: bool) -> None:
    real = V._load_worker
    V._load_worker = lambda opts: (worker, True)
    try:
        V.run_propainter(
            src, dst,
            box=BOX, dilate_px=5, feather_px=12, margin_px=32,
            opts=V.ProPainterOpts(repo=Path("/unused"), segment=SEGMENT,
                                  overlap=OVERLAP, part_frames=PART),
            progress=False, resume=resume,
        )
    finally:
        V._load_worker = real


def main() -> int:
    if not shutil.which("ffmpeg"):
        print("error: ffmpeg not on PATH")
        return 2

    failures = 0
    with tempfile.TemporaryDirectory(prefix="wmrm-resume-") as td:
        work = Path(td)
        src = work / "src.mp4"
        write_source(marked_frames(), src)

        whole = work / "whole.mp4"
        whole_stub = StubWorker()
        run(src, whole, worker=whole_stub, resume=False)

        # 1. interrupt
        broken = work / "resumed.mp4"
        parts = broken.with_name(broken.name + ".parts")
        crashed = CrashingStub(CRASH_AFTER)
        try:
            run(src, broken, worker=crashed, resume=False)
        except RuntimeError:
            pass
        else:
            print("FAIL  the simulated crash did not stop the run")
            failures += 1

        kept = sorted(parts.glob("part-*.mp4"))
        done = CRASH_AFTER * SEGMENT // PART        # whole parts behind the crash
        if broken.exists():
            print(f"FAIL  a crashed run still produced {broken.name}")
            failures += 1
        if len(kept) == done:
            print(f"PASS  crash kept {len(kept)} finished part(s), dropped the "
                  f"one it was writing")
        else:
            print(f"FAIL  crash left {len(kept)} part(s), expected {done}: "
                  f"{[p.name for p in kept]}")
            failures += 1

        # 2. resume
        resumed_stub = StubWorker()
        run(src, broken, worker=resumed_stub, resume=True)

        if not broken.exists():
            print("FAIL  the resumed run produced no output")
            return 1

        if broken.read_bytes() == whole.read_bytes():
            print(f"PASS  resumed output byte-identical to an uninterrupted run "
                  f"({broken.stat().st_size} bytes)")
        else:
            print(f"FAIL  resumed output differs: {broken.stat().st_size} bytes vs "
                  f"{whole.stat().st_size}")
            failures += 1

        if len(resumed_stub.calls) < len(whole_stub.calls):
            print(f"PASS  work was skipped: {len(resumed_stub.calls)} model call(s) "
                  f"on resume vs {len(whole_stub.calls)} for the whole clip")
        else:
            print(f"FAIL  resume redid everything: {len(resumed_stub.calls)} model "
                  f"call(s) vs {len(whole_stub.calls)}")
            failures += 1

        # 4. every repaired tile landed on the frame it was made from. Byte-identity
        #    above cannot see this: a whole run seeks for every part after the first
        #    too, so a seek that is off by one is off by one in both runs and the two
        #    files still match. The stub fills the hole with the marker of the frame
        #    it was given, and the marker is also readable from a corner it never
        #    touches, so the two disagreeing means the tile stream and the base
        #    slipped against each other -- a watermark repaired with the wrong
        #    frame's pixels, which is the failure this whole seek is one step away
        #    from.
        #    Averaged over a block, not read off one pixel: crf 18 moves a single
        #    pixel by more than the 4-level marker step, which says nothing about
        #    alignment.
        got = decode(broken)
        cy, cx = BOX.y + BOX.h // 2, BOX.x + BOX.w // 2
        inside = got[:, cy - 14:cy + 14, cx - 28:cx + 28, 1].reshape(len(got), -1).mean(axis=1)
        expected = np.array([marker_of(i) for i in range(len(got))], float)
        slip = np.abs(inside - expected)
        if slip.max() < MARK / 2:
            print(f"PASS  tile aligned with its own frame in all {len(got)} frames "
                  f"(worst drift {slip.max():.2f}, one frame of slip would be "
                  f"{MARK})")
        else:
            off = [(i, round(float(a), 1), int(e))
                   for i, (a, e) in enumerate(zip(inside, expected))
                   if abs(a - e) >= MARK / 2]
            print(f"FAIL  repaired tile is on the wrong frame in {len(off)} of "
                  f"{len(got)}; first (frame, in-hole, expected): {off[:6]}")
            failures += 1

        # 5. the parts must not have rounded the clock on their way through. Cheap to
        #    get wrong -- a container whose timebase cannot express one frame exactly
        #    loses a fraction per part, and the loss is cumulative -- and it surfaces
        #    far from here, as audio drift and a `frame rate` failure from verify.
        a, b = probe(src), probe(broken)
        if (a.fps, a.nframes) == (b.fps, b.nframes) and abs(a.duration - b.duration) < 1e-6:
            print(f"PASS  timing preserved exactly: {b.fps} fps, {b.nframes} frames, "
                  f"{b.duration:.6f}s")
        else:
            print(f"FAIL  timing changed: {a.fps} fps/{a.nframes} frames/"
                  f"{a.duration:.6f}s in, {b.fps}/{b.nframes}/{b.duration:.6f}s out")
            failures += 1

        if parts.exists():
            print(f"FAIL  {parts.name} survived a successful run")
            failures += 1
        else:
            print("PASS  parts directory cleaned up after success")

    print("\nall checks passed" if not failures else f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
