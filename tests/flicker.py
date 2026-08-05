"""Measure temporal inconsistency (flicker) in the patched region.

Per-frame PSNR cannot see flicker: a patch can score well on every individual
frame and still boil visibly, because each frame is inpainted independently.

The metric is frame-to-frame change measured against the ground truth's own:

    delta  = mean|x[t] - x[t-1]|  inside the badge region
    excess = delta(output) - delta(truth)

The subtraction matters -- moving footage changes between frames legitimately,
so the truth sets the baseline. Both directions are defects:

    excess >> 0   invented motion: the patch boils/flickers
    excess << 0   too static: a frozen smear sitting in moving footage, which
                  reads as a stuck rectangle even though no frame looks wrong

So what we want is |excess| near zero -- the patch should move exactly as much
as the content it replaced.

    python tests/flicker.py detail
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"
BADGE_Y = {"busy": 12, "smooth": 12, "detail": 430}


def consecutive(path: Path, start: float, count: int) -> np.ndarray:
    """Decode `count` *consecutive* frames starting at `start` seconds.

    Flicker only shows up between adjacent frames, so the spread-out sampling
    used for PSNR is useless here.
    """
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(info[0]), int(info[1])
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-i", str(path),
         "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True)
    n = len(r.stdout) // (w * h * 3)
    if n < 2:
        raise RuntimeError(f"got {n} frames from {path}")
    return np.frombuffer(r.stdout[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)


def _region(stack: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    return stack[:, y:y + h, x:x + w].astype(np.float32)


def temporal_delta(stack: np.ndarray, box) -> float:
    return float(np.abs(np.diff(_region(stack, box), axis=0)).mean())


def temporal_corr(stack: np.ndarray, truth: np.ndarray, box) -> float:
    """Correlation between the output's temporal change field and the truth's.

    Magnitude alone is not enough: a patch that boils randomly can happen to
    change by the right *amount* while changing in the wrong *places*. This asks
    whether it changes where the real content changed. ~1.0 means the motion is
    genuinely tracking the scene; ~0 means the variation is invented.
    """
    a = np.diff(_region(stack, box), axis=0).ravel()
    b = np.diff(_region(truth, box), axis=0).ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main(case: str) -> int:
    if case not in BADGE_Y:
        print(f"error: unknown case {case!r}", file=sys.stderr)
        return 1
    by = BADGE_Y[case]
    badge = (384, by, 84, 36)

    truth_p = FIX / f"{case}-truth.mp4"
    if not truth_p.exists():
        print(f"error: {truth_p} missing", file=sys.stderr)
        return 1

    START, COUNT = 5.0, 60
    truth = consecutive(truth_p, START, COUNT)
    base = temporal_delta(truth, badge)

    # Missing files are skipped rather than fatal: not every variant is rendered for
    # every case, and a partial table is more useful than a crash.
    candidates = {
        "marked (untouched)": FIX / f"{case}-marked.mp4",
        "fast  (delogo+feather)": FIX / f"{case}-fast.mp4",
        "draft (cv2.inpaint)": FIX / f"{case}-draft.mp4",
        "high  (LaMa)": FIX / f"{case}-lama.mp4",
        "unblend (default)": FIX / f"{case}-unblend.mp4",
    }
    candidates = {k: v for k, v in candidates.items() if v.exists()}

    print(f"case: {case}   {COUNT} consecutive frames from t={START}s")
    print(f"ground-truth frame-to-frame change in badge region: {base:.2f} levels\n")
    print(f"{'variant':<26}{'delta':>8}{'excess':>9}{'corr':>8}   verdict")
    print("-" * 74)
    for label, path in candidates.items():
        if not path.exists():
            continue
        got = consecutive(path, START, COUNT)
        n = min(len(truth), len(got))
        d = temporal_delta(got[:n], badge)
        corr = temporal_corr(got[:n], truth[:n], badge)
        excess = d - base
        mag = abs(excess)
        if mag < 0.5:
            verdict = "matches content"
        elif mag < 1.5:
            verdict = "slight" + (" boiling" if excess > 0 else " freeze")
        elif mag < 3.0:
            verdict = "visible" + (" boiling" if excess > 0 else " freeze")
        else:
            verdict = "bad " + ("flicker" if excess > 0 else "frozen patch")
        print(f"{label:<26}{d:>7.3f}{excess:>+9.3f}{corr:>8.2f}   {verdict}")

    print("\n|excess| near 0 is the goal: the patch should move as much as what it")
    print("replaced. Positive = boiling, negative = a frozen patch in moving footage.")
    print("corr = does it change in the same *places* as the real content (1.0 best).")
    print("Matching magnitude with low corr would mean invented, incoherent motion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "detail"))
