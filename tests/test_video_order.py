"""Frames must come out in order, whatever the segmentation and worker count.

This is the failure mode with no natural alarm. Reordering, dropping or duplicating a
frame leaves resolution, frame rate, duration, audio and both PSNR gates passing --
every acceptance check the tool has -- while the video visibly stutters. And with
--pp-workers > 1 the segments genuinely complete out of order, so nothing but
explicit sequencing keeps them straight.

The model is stubbed out with an identity pass. That is the point: the ordering,
overlap accounting and write sequence are this project's code, and testing them
through a 40-second GPU inference would be slow, unrepeatable, and would not
distinguish a windowing bug from a model artifact.

    python tests/test_video_order.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmrm import video as V  # noqa: E402
from wmrm.region import Box  # noqa: E402

W, H = 320, 240
BOX = Box(100, 60, 80, 40)
N = 25
STEP = 10          # grey level per frame; big enough to survive x264 at crf 18
WORK = Path(__file__).resolve().parent / "_order_tmp"


def _identity_segment(opts, frames_dir: Path, mask_png: Path, out_dir: Path) -> Path:
    """Stand-in for ProPainter that returns its input untouched.

    Mirrors the real function's contract exactly: output lands in
    <out_dir>/<input dir name>/frames, one PNG per input frame, sorted-name order
    matching input order.
    """
    got = out_dir / frames_dir.name / "frames"
    got.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(sorted(frames_dir.glob("*.png"))):
        shutil.copyfile(src, got / f"{i:04d}.png")
    return got


def make_source(path: Path) -> None:
    """A clip whose frame index is readable off any pixel: frame i is grey i*STEP."""
    raw = WORK / "src"
    raw.mkdir(parents=True, exist_ok=True)
    for i in range(N):
        img = np.full((H, W, 3), i * STEP, np.uint8)
        cv2.imwrite(str(raw / f"{i:04d}.png"), img)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", "25",
         "-i", str(raw / "%04d.png"), "-c:v", "ffv1", str(path)],
        check=True)


def read_levels(path: Path) -> list[int]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Sample inside the box, which is the only region the pipeline rewrites.
        out.append(int(frame[BOX.y + BOX.h // 2, BOX.x + BOX.w // 2, 1]))
    cap.release()
    return out


def check(label: str, levels: list[int]) -> bool:
    expected = [i * STEP for i in range(N)]
    if len(levels) != N:
        print(f"FAIL  {label}: {len(levels)} frames out, expected {N}")
        return False
    bad = [(i, e, g) for i, (e, g) in enumerate(zip(expected, levels))
           if abs(e - g) > 6]
    if bad:
        i, e, g = bad[0]
        print(f"FAIL  {label}: frame {i} carries level {g}, expected ~{e} "
              f"({len(bad)} of {N} wrong -- frames are out of order, "
              f"duplicated or dropped)")
        return False
    print(f"ok    {label}")
    return True


def main() -> int:
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    src = WORK / "src.mkv"
    make_source(src)

    real = V._run_segment
    V._run_segment = _identity_segment
    bad = 0
    try:
        # segment/worker combinations that between them cover: a clean division, a
        # short final segment, overlap larger than the step, and concurrency.
        cases = [(10, 4, 1), (10, 4, 3), (7, 3, 2), (N + 50, 4, 1), (4, 0, 4)]
        for seg, ov, workers in cases:
            dst = WORK / f"out_s{seg}_o{ov}_w{workers}.mp4"
            V.run_propainter(
                src, dst, box=BOX, dilate_px=2, feather_px=4, margin_px=16,
                opts=V.ProPainterOpts(repo=Path("/unused"), segment=seg,
                                      overlap=ov, workers=workers),
                progress=False,
            )
            bad += not check(f"segment {seg:>3}, overlap {ov}, workers {workers}",
                             read_levels(dst))
    finally:
        V._run_segment = real

    shutil.rmtree(WORK, ignore_errors=True)
    print()
    if bad:
        print(f"{bad} case(s) failed")
        return 1
    print("frame order is preserved across every segmentation and worker count")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
