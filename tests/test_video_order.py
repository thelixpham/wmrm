"""The streaming ProPainter path must not reorder, drop or duplicate a frame.

Frames used to reach the compositor as a numbered directory of PNGs, where frame j
could only ever come from the one segment that owned it and its position in the video
was its filename. They now reach it as a *stream*: position is send order, and the
bookkeeping that decides which frames of a segment are kept and which are discarded
as context is what holds the video together. Get the left/right context arithmetic
off by one and the output is still the right length, still plays, and is silently
wrong -- a frame repeated here, a frame missing there.

This is the same property the previous version of this file pinned, for the reason it
warned about: it said that switching to a streaming writer would let ordering break
silently. That switch has now happened, so the checks moved with it.

Three checks, each pinning a different way it can break:

1. **Length.** Frames out == frames in.
2. **Order.** Every frame carries a unique intensity marker, read back from the
   output and asserted strictly increasing. A swap, a repeat or a dropped frame
   breaks monotonicity.
3. **Segmentation invariance.** The same clip run at (segment=17, overlap=5) and at
   (segment=1000, overlap=0) -- one many small segments with context, the other a
   single segment with none -- must produce byte-identical files. This is the check
   that actually exercises the context trimming: the two runs discard completely
   different frames, so any error in what is kept shows up as a diff.

The model is replaced by a stub, deliberately: it is slow, it wants a GPU, and it is
not what is under test. The stub is *content-derived* rather than a constant, so it
stays segmentation-invariant while still proving the mask lands where it should -- a
constant fill would pass check 3 even with the mask in the wrong place.

    python tests/test_video_order.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "src"))

from wmrm import video as V           # noqa: E402
from wmrm.region import Box           # noqa: E402

FRAMES = 60
W, H = 320, 180
FPS = "30000/1001"
BOX = Box(40, 30, 120, 60)
MARKER_STEP = 4                       # frame t has blue == t * MARKER_STEP


def source_frames() -> np.ndarray:
    """A clip where every frame is identifiable on its own.

    The blue channel is a flat per-frame marker; green and red carry moving texture so
    the frames are not otherwise identical. Marker steps of 4 survive an x264
    re-encode when averaged over a region, which is how check 2 reads them back.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.empty((FRAMES, H, W, 3), np.uint8)
    for t in range(FRAMES):
        out[t, :, :, 0] = t * MARKER_STEP
        out[t, :, :, 1] = np.clip(np.sin(xx * 0.05 + t * 0.3) * 100 + 128, 0, 255)
        out[t, :, :, 2] = np.clip(np.cos(yy * 0.07 - t * 0.2) * 100 + 128, 0, 255)
    return out


def write_source(frames: np.ndarray, path: Path) -> None:
    """Encode losslessly, so what comes back out is exactly what went in."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", FPS, "-i", "-",
         "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(path)],
        stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).data)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("failed to encode the test source")


def decode(path: Path) -> np.ndarray:
    res = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"failed to decode {path}: {res.stderr.decode()[:400]}")
    n = len(res.stdout) // (W * H * 3)
    return np.frombuffer(res.stdout[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)


class StubWorker:
    """Identity outside the mask; inside it, a value derived from the same frame.

    Content-derived on purpose. A constant fill would survive any reordering of frames
    within a segment and would make check 3 vacuous. Deriving the fill from this
    frame's own marker means a frame that lands in the wrong output position carries
    the evidence with it.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []

    def inpaint(self, frames: np.ndarray, mask: np.ndarray, *, progress=None):
        self.calls.append(len(frames))
        out = frames.copy()
        hole = mask > 0
        for f in out:
            f[hole, 1] = f[0, 0, 0]        # green inside the hole := this frame's marker
        return out


def run(src: Path, dst: Path, *, segment: int, overlap: int) -> StubWorker:
    stub = StubWorker()
    real = V._load_worker
    # (worker, was_cached) -- _load_worker reports whether it had to load, so the run
    # can say so. True here: a stub is always "already resident".
    V._load_worker = lambda opts: (stub, True)
    try:
        V.run_propainter(
            src, dst,
            box=BOX, dilate_px=5, feather_px=12, margin_px=32,
            opts=V.ProPainterOpts(repo=Path("/unused"), segment=segment,
                                  overlap=overlap),
            progress=False,
        )
    finally:
        V._load_worker = real
    return stub


def main() -> int:
    if not shutil.which("ffmpeg"):
        print("error: ffmpeg not on PATH")
        return 2

    failures = 0
    with tempfile.TemporaryDirectory(prefix="wmrm-order-") as td:
        work = Path(td)
        src = work / "src.mp4"
        write_source(source_frames(), src)

        chunked = work / "chunked.mp4"
        stub_a = run(src, chunked, segment=17, overlap=5)
        single = work / "single.mp4"
        stub_b = run(src, single, segment=1000, overlap=0)

        print(f"chunked: {len(stub_a.calls)} model call(s), block sizes {stub_a.calls}")
        print(f"single : {len(stub_b.calls)} model call(s), block sizes {stub_b.calls}")
        if len(stub_a.calls) < 3:
            print("FAIL  the chunked run did not chunk -- nothing was proved")
            failures += 1

        got = decode(chunked)

        # 1. length
        if len(got) == FRAMES:
            print(f"PASS  frame count preserved: {len(got)}")
        else:
            print(f"FAIL  frame count: {len(got)} out, {FRAMES} in")
            failures += 1

        # 2. order, read from a corner the stub never touches
        markers = got[:, H - 20:, W - 40:, 0].reshape(len(got), -1).mean(axis=1)
        expected = np.arange(len(got)) * MARKER_STEP
        drift = np.abs(markers - expected)
        monotonic = bool(np.all(np.diff(markers) > 0))
        if monotonic and drift.max() < 3.0:
            print(f"PASS  order preserved: markers monotonic, max drift from "
                  f"expected {drift.max():.2f}")
        else:
            print(f"FAIL  order: monotonic={monotonic}, max drift {drift.max():.2f}")
            bad = [(i, round(float(m), 1), int(e)) for i, (m, e)
                   in enumerate(zip(markers, expected)) if abs(m - e) >= 3.0]
            print(f"      first offenders (frame, got, expected): {bad[:6]}")
            failures += 1

        # 3. segmentation invariance
        if chunked.read_bytes() == single.read_bytes():
            print(f"PASS  segmentation invariant: byte-identical across "
                  f"segment=17/overlap=5 and segment=1000/overlap=0")
        else:
            other = decode(single)
            if len(other) != len(got):
                print(f"FAIL  segmentation changed the length: {len(got)} vs {len(other)}")
            else:
                d = np.abs(got.astype(np.int16) - other.astype(np.int16))
                per_frame = [(i, int((f > 0).sum())) for i, f in enumerate(d)
                             if (f > 0).any()]
                print(f"FAIL  segmentation changed pixels: {len(per_frame)}/{len(got)} "
                      f"frames differ, max |diff| {int(d.max())}; "
                      f"first few {per_frame[:6]}")
            failures += 1

    print("\nall checks passed" if not failures else f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
