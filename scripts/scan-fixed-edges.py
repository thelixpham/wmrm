"""Show every fixed overlay in the frame, not just the one in a corner.

This exists because of a gap between two commands that each look correct on their own.
`wmrm detect` searches one corner by design -- "only a corner ROI is searched, so
bottom/centre text is [missed]", detect.py's own first paragraph. `wmrm coverage` only
inspects a ring around a box you already have, so "covered" means "nothing mark-like
survives just outside THIS box", never "the frame is clean". Put those together and a
second watermark in the middle of the picture passes every check a run makes, and is
still there in the output.

The signal is the one coverage.py calls A: average the *signed* gradient over frames
sampled across the whole clip. Edges belonging to moving content point different ways
from frame to frame and cancel out; an overlay pinned to the same pixels keeps the
same edge in every frame and survives. Semi-transparent marks come through strongly,
which a variance test would miss entirely -- those pixels never stop changing.

Deliberately no list of boxes. An earlier version guessed them and was tuned over five
rounds against synthetic clips: a wide merge joined a corner badge and a centre caption
into one meaningless rectangle spanning the frame, a narrow one split a single word
into three, and every automatic threshold either lost the fainter of two marks or lit
up whatever background happened to hold still. The picture needs none of that machinery
and does not overstate its confidence. Read coordinates off it with `wmrm grid` or
`wmrm.pick.write_picker`, then let `wmrm coverage` check them.

    python scripts/scan-fixed-edges.py VIDEO.mp4 [samples]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wmrm.detect import sample_frames        # noqa: E402
from wmrm.probe import probe                 # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[-1].strip())
        return 2
    src = Path(sys.argv[1])
    if not src.exists():
        print(f"error: {src} not found")
        return 2
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    info = probe(src)
    stack = sample_frames(info, n)                    # (N, H, W, 3) float32
    gray = stack.mean(axis=3)
    dy, dx = np.gradient(gray, axis=1), np.gradient(gray, axis=2)
    fixed = np.abs(dy.mean(axis=0)) + np.abs(dx.mean(axis=0))

    # Stretch between the noise floor and a high percentile. Scaling by the maximum
    # instead lets the boldest mark set the bar for every other one, and a faint
    # caption beside a bold badge vanishes -- measured, exactly the mark this is for.
    lo = float(np.percentile(fixed, 50))
    hi = float(np.percentile(fixed, 99.8))
    img = np.clip((fixed - lo) / max(hi - lo, 1e-6), 0, 1)

    energy = src.with_name(f"{src.stem}-fixed-edges.png")
    cv2.imwrite(str(energy), (img * 255).astype(np.uint8))

    # The same thing over a real frame, because "there is a smudge at 430,330" is
    # much easier to act on when you can see what it is sitting on top of.
    frame = stack[len(stack) // 2].astype(np.float32)
    heat = np.zeros_like(frame)
    heat[:, :, 2] = img * 255                         # BGR -> red
    over = np.clip(frame * 0.55 + heat * 0.85, 0, 255).astype(np.uint8)
    overlay = src.with_name(f"{src.stem}-fixed-edges-overlay.png")
    cv2.imwrite(str(overlay), over)

    print(f"{info.width}x{info.height}, {stack.shape[0]} frames sampled across the clip\n")
    print(f"  {energy}\n      white = an edge that never moves\n")
    print(f"  {overlay}\n      the same, in red over a real frame\n")
    print("Every watermark in the video is in there. So is any genuinely static")
    print("scenery, which is why this is a picture and not a verdict. For each mark:")
    print(f"  wmrm coverage {src.name} --box X,Y,W,H")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
