#!/usr/bin/env python
"""Acceptance-check tests. Run directly: `python tests/test_verify.py`.

Fast and offline -- frames are synthesised and `_mid_frame` is replaced, because what is
being pinned is how the samples are aggregated into a verdict, not ffmpeg's decoding.

Why this file exists: `verify` used to compare exactly one frame, taken at `duration/2`
of each file. A `-ss` seek can land on different pictures in two files whose keyframes
differ -- the output is re-encoded, and a long clip is concatenated from many parts -- and
on moving content a single frame of misalignment reads as tens of dB of "damage". A
6h20m run was failed on that one draw, and its pod was reclaimed before anyone could
check whether the file was actually bad.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmrm import verify as V                                  # noqa: E402
from wmrm.region import Box                                   # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []

H, W = 200, 300
BOX = Box(200, 40, 60, 30)


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))


@dataclass
class FakeInfo:
    width: int = W
    height: int = H
    fps: str = "30000/1001"
    duration: float = 7200.0
    has_audio: bool = True


def frame(base: int = 100) -> np.ndarray:
    return np.full((H, W, 3), base, np.uint8)


def offset_psnr(d: int) -> float:
    """PSNR of two frames differing by a constant `d` -- 2 -> 42.1 dB, 8 -> 30.1 dB."""
    return 10.0 * float(np.log10(255.0 * 255.0 / (d * d)))


def run_verify(pairs, samples: int = 9):
    """`pairs` is a list of (original, processed) frames, one per sample time."""
    # Indexed by timestamp, not by call count: when a frame is unreadable `verify`
    # skips the second read of that pair, so counting calls desynchronises everything
    # after it -- which is exactly what this helper is here to avoid faking wrong.
    times = V._sample_times(FakeInfo().duration, samples)
    seen: dict[float, int] = {}

    def fake_mid_frame(path, at):
        i = min(range(len(times)), key=lambda k: abs(times[k] - at))
        if i >= len(pairs) or pairs[i] is None:
            raise RuntimeError("unreadable")
        which = seen.get(at, 0)
        seen[at] = which + 1
        return pairs[i][min(which, 1)]

    orig_probe, orig_mid = V.probe, V._mid_frame
    V.probe = lambda p: FakeInfo()
    V._mid_frame = fake_mid_frame
    try:
        return V.verify(Path("a.mp4"), Path("b.mp4"), BOX, samples=samples)
    finally:
        V.probe, V._mid_frame = orig_probe, orig_mid


def changed_inside(a: np.ndarray, d: int = 40) -> np.ndarray:
    """A processed frame whose box region really did change, as a real run's would."""
    b = a.copy()
    b[BOX.y: BOX.y + BOX.h, BOX.x: BOX.x + BOX.w] = np.uint8(int(a[0, 0, 0]) + d)
    return b


def pair(outside_delta: int) -> tuple[np.ndarray, np.ndarray]:
    a = frame()
    b = changed_inside(np.full((H, W, 3), 100 + outside_delta, np.uint8))
    return a, b


def test_sample_times() -> None:
    print("\n[sampling]")
    t = V._sample_times(1000.0, 9)
    check("it takes the number of samples asked for", len(t) == 9)
    check("it stays clear of the fades at both ends", t[0] >= 50 and t[-1] <= 950,
          f"{t[0]:.0f}..{t[-1]:.0f}")
    check("a zero-duration file still yields one time", V._sample_times(0.0, 9) == [0.0])


def test_one_misaligned_frame_does_not_fail_the_run() -> None:
    """The regression this file was written for."""
    print("\n[one bad frame]")
    good = [pair(2) for _ in range(8)]          # 42.1 dB
    bad = pair(26)                              # 19.8 dB -- a seek that missed
    res = run_verify(good[:4] + [bad] + good[4:])
    detail = dict((n, d) for n, _, d in res.checks)["rest of frame preserved"]
    check("eight good frames outvote one misread one", res.ok, detail)
    check("but the outlier is still visible in the report", "19.8" in detail, detail)
    check("and the median is reported, not the single mid-point", "median" in detail)


def test_real_degradation_still_fails() -> None:
    print("\n[genuine degradation]")
    res = run_verify([pair(8) for _ in range(9)])       # 30.1 dB everywhere
    check("a frame that is degraded throughout fails", not res.ok)
    check("'rest of frame preserved' is the check that fails",
          "rest of frame preserved" in res.to_dict()["failed"],
          str(res.to_dict()["failed"]))


def test_unreadable_frames() -> None:
    print("\n[unreadable frames]")
    frames = [pair(2) for _ in range(9)]
    frames[3] = None
    frames[7] = None
    res = run_verify(frames)
    detail = dict((n, d) for n, _, d in res.checks)["rest of frame preserved"]
    check("a couple of unreadable frames are skipped, not fatal", res.ok, detail)
    check("and the count reflects what was actually read", "over 7 frames" in detail,
          detail)

    res2 = run_verify([None] * 9)
    check("no readable frames at all is a failure", not res2.ok)
    check("and says so plainly",
          "could not read" in dict((n, d) for n, _, d in res2.checks)
          ["rest of frame preserved"])


def test_near_far_split_is_diagnostic() -> None:
    print("\n[near-band vs far-field]")
    a = frame()
    b = a.copy()
    # Damage confined to the ring just outside the mask: what a repaint leak looks like.
    b[max(0, BOX.y - 96): BOX.y + BOX.h + 96,
      max(0, BOX.x - 96): BOX.x + BOX.w + 96] = 108           # delta 8 -> ~30 dB
    b[max(0, BOX.y - 24): BOX.y + BOX.h + 24,
      max(0, BOX.x - 24): BOX.x + BOX.w + 24] = 100           # masked out, ignored
    b[BOX.y: BOX.y + BOX.h, BOX.x: BOX.x + BOX.w] = 140       # the repaint itself
    _out, _ins, near, far = V._measure(a, b, BOX)
    check("a leak shows up in the near band", near < 40, f"near {near:.1f} dB")
    check("and not in the far field", far > near, f"far {far:.1f} dB vs near {near:.1f}")

    # Uniform loss, which is what a re-encode looks like: near and far agree.
    b2 = changed_inside(np.full((H, W, 3), 102, np.uint8))
    _o2, _i2, near2, far2 = V._measure(a, b2, BOX)
    check("uniform encode loss reads the same near and far",
          abs(near2 - far2) < 0.5, f"near {near2:.1f} vs far {far2:.1f}")


def main() -> int:
    test_sample_times()
    test_one_misaligned_frame_does_not_fail_the_run()
    test_real_degradation_still_fails()
    test_unreadable_frames()
    test_near_far_split_is_diagnostic()
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    for name in FAILED:
        print(f"  FAILED: {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
