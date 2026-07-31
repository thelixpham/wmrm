"""Score recovery against the ground truth and render a visual comparison.

Because the fixtures were made by burning a badge onto a genuinely clean clip,
we can ask the only question that matters: how close to the original did each
backend get, inside the watermark region?

    python tests/score.py busy
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"

# Where the badge really is per case, and a wider window that also catches any
# halo or seam left just outside it.
BADGE_Y = {"busy": 12, "smooth": 12, "detail": 430}


def frames(path: Path, count: int = 12) -> np.ndarray:
    info = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=width,height,duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip().split(",")
    w, h, dur = int(info[0]), int(info[1]), float(info[2])
    out = []
    for t in np.linspace(dur * 0.05, dur * 0.95, count):
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", str(path),
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True)
        if len(r.stdout) >= w * h * 3:
            out.append(np.frombuffer(r.stdout[:w * h * 3], np.uint8).reshape(h, w, 3))
    return np.stack(out)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse <= 1e-9 else 10.0 * float(np.log10(255.0 ** 2 / mse))


def crop(stack: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    return stack[:, y:y + h, x:x + w]


def main(case: str) -> int:
    truth_p = FIX / f"{case}-truth.mp4"
    if not truth_p.exists():
        print(f"error: {truth_p} missing -- run tests/make_fixtures.py", file=sys.stderr)
        return 1
    if case not in BADGE_Y:
        print(f"error: unknown case {case!r}; known: {', '.join(BADGE_Y)}", file=sys.stderr)
        return 1
    by = BADGE_Y[case]
    badge = (384, by, 84, 36)
    halo = (368, max(0, by - 12), 112, 64)
    truth = frames(truth_p)

    candidates = {
        "marked (untouched)": FIX / f"{case}-marked.mp4",
        "fast  (delogo+feather)": FIX / f"{case}-fast.mp4",
        "draft (cv2.inpaint)": FIX / f"{case}-draft.mp4",
        "high  (LaMa)": FIX / f"{case}-lama.mp4",
    }

    print(f"case: {case}   PSNR vs clean original, higher is better\n")
    print(f"{'variant':<26}{'badge region':>14}{'badge+halo':>13}{'rest of frame':>15}")
    print("-" * 68)
    rows = {}
    for label, path in candidates.items():
        if not path.exists():
            print(f"{label:<26}{'(not generated)':>14}")
            continue
        got = frames(path)
        n = min(len(truth), len(got))
        t, g = truth[:n], got[:n]

        keep = np.ones(t.shape[1:3], bool)
        x, y, w, h = halo
        keep[y:y + h, x:x + w] = False

        rows[label] = (
            psnr(crop(t, badge), crop(g, badge)),
            psnr(crop(t, halo), crop(g, halo)),
            psnr(t[:, keep], g[:, keep]),
        )
        b, ha, rest = rows[label]
        print(f"{label:<26}{b:>11.1f} dB{ha:>10.1f} dB{rest:>12.1f} dB")

    base = rows.get("marked (untouched)")
    if base:
        print(f"\nimprovement over leaving the watermark in place (badge region):")
        for label, (b, _, _) in rows.items():
            if label != "marked (untouched)":
                print(f"  {label:<24} {b - base[0]:+.1f} dB")

    # visual strip: truth | marked | each result, cropped to the badge
    x, y, w, h = halo
    tiles, labels = [], []
    mid = len(truth) // 2
    tiles.append(truth[mid, y:y + h, x:x + w]); labels.append("ORIGINAL")
    for label, path in candidates.items():
        if path.exists():
            g = frames(path, 12)
            tiles.append(g[mid, y:y + h, x:x + w])
            labels.append(label.split("(")[0].strip().upper())

    scale = 4
    big = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
           for t in tiles]
    pad = 26
    canvas = np.full((big[0].shape[0] + pad, sum(b.shape[1] + 6 for b in big), 3),
                     255, np.uint8)
    cx = 0
    for img, label in zip(big, labels):
        canvas[pad:pad + img.shape[0], cx:cx + img.shape[1]] = img
        cv2.putText(canvas, label, (cx + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 1, cv2.LINE_AA)
        cx += img.shape[1] + 6
    out = FIX / f"{case}-comparison.png"
    cv2.imwrite(str(out), canvas)
    print(f"\nvisual comparison -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "busy"))
