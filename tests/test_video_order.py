"""`--pp-workers` must change only the speed, never a single pixel.

Concurrency here is only safe because segments write disjoint frame numbers into the
assembly directory, so a segment finishing early cannot reorder the video. That is an
easy property to break later -- switch to appending, or to a streaming writer, and
out-of-order completion silently shuffles the output. This test pins it.

ProPainter itself is replaced by an identity stub, deliberately: the real model is
slow, needs a GPU, and is not what is under test. The stub also finishes segments in
*reverse* order, so a run that depends on completion order fails here rather than on
someone's footage.

    python tests/test_video_order.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "src"))

from wmrm import video as V           # noqa: E402
from wmrm.region import Box           # noqa: E402

FRAMES = 120
W, H = 320, 180


def make_source(path: Path) -> None:
    """A clip whose every frame is visibly different, so a swap cannot hide."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=s={W}x{H}:r=30000/1001:d={FRAMES / 30000 * 1001}",
         "-frames:v", str(FRAMES),
         "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", str(path)],
        check=True)


def stub_run_segment(order: list[int]):
    """Identity 'model' that returns segments in reverse order of submission.

    Later segments sleep less, so with workers > 1 they overtake the earlier ones.
    `order` records actual completion order so the test can assert the race really
    happened -- a passing test that never raced would prove nothing.
    """
    def run(opts, frames_dir: Path, mask_png: Path, out_dir: Path) -> Path:
        si = int(frames_dir.name.removeprefix("seg"))
        time.sleep(max(0.0, 0.60 - 0.12 * si))
        got = out_dir / frames_dir.name / "frames"
        got.mkdir(parents=True)
        for f in sorted(frames_dir.glob("*.png")):
            shutil.copyfile(f.resolve(), got / f.name)   # resolve: inputs are symlinks
        order.append(si)
        return got
    return run


def run_once(src: Path, dst: Path, workers: int, order: list[int]) -> None:
    real = V._run_segment
    V._run_segment = stub_run_segment(order)
    try:
        V.run_propainter(
            src, dst,
            box=Box(40, 30, 120, 60),
            dilate_px=5, feather_px=12, margin_px=32,
            opts=V.ProPainterOpts(repo=Path("/unused"), segment=25, overlap=5,
                                  workers=workers),
            progress=False,
        )
    finally:
        V._run_segment = real


def frames_differ(a: Path, b: Path) -> int:
    """Count frames where the two videos are not bit-identical."""
    res = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(a), "-i", str(b),
         "-lavfi", "psnr=stats_file=-", "-f", "null", "-"],
        capture_output=True, text=True)
    bad = 0
    for line in res.stdout.splitlines():
        for field in line.split():
            if field.startswith("psnr_avg:") and field.split(":", 1)[1] != "inf":
                bad += 1
    return bad


def main() -> int:
    if not shutil.which("ffmpeg"):
        print("error: ffmpeg not on PATH")
        return 2

    failures = 0
    with tempfile.TemporaryDirectory(prefix="wmrm-order-") as td:
        work = Path(td)
        src = work / "src.mp4"
        make_source(src)

        baseline = work / "w1.mp4"
        order1: list[int] = []
        run_once(src, baseline, 1, order1)
        print(f"workers=1  completion order: {order1}")
        if order1 != sorted(order1):
            print("FAIL  serial run did not complete in order -- test is broken")
            failures += 1

        for workers in (2, 3, 5):
            out = work / f"w{workers}.mp4"
            order: list[int] = []
            run_once(src, out, workers, order)
            raced = order != sorted(order)
            bad = frames_differ(baseline, out)
            status = "PASS" if bad == 0 and raced else "FAIL"
            why = ""
            if bad:
                why = f" -- {bad} frame(s) differ from the serial run"
            elif not raced:
                why = " -- segments never completed out of order, nothing was proved"
            print(f"{status}  workers={workers}  order={order}{why}")
            failures += status == "FAIL"

    print("\nall checks passed" if not failures else f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
