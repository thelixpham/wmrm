"""Build ground-truth fixtures: burn a fake AI badge at top-right on clean clips.

Having the genuinely clean video means recovery quality can be scored with PSNR
against the truth instead of judged by eye.

    python tests/make_fixtures.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "fixtures"
EX = HERE.parent.parent / "dreamina-delogo" / "examples"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# The corner a clip happens to have is not the corner that matters. In both
# example clips the top-right is smooth sky, which is the *easy* case -- and the
# fast path wins there. The real target is a badge sitting on busy texture, so
# 'detail' burns it lower down over the grass/water instead. `y` is the only
# difference; the box stays at the right edge.
#
#   busy   : right edge, y=12   -- over soft sky (easy)
#   smooth : right edge, y=12   -- over soft sky (easy)
#   detail : right edge, y=430  -- over detailed ground (hard, matches the
#            floral-background case this tool is actually for)
CASES = {
    "busy": ("02-clean.mp4", 12),
    "smooth": ("01-clean.mp4", 12),
    "detail": ("02-clean.mp4", 430),
}

BADGE_VF = (
    "drawbox=x=iw-96:y={y}:w=84:h=36:color=black@0.30:t=fill,"
    "drawtext=fontfile={font}:text='AI':fontcolor=white@0.92:fontsize=28:x=W-88:y={ty},"
    "drawtext=fontfile={font}:text='gen':fontcolor=white@0.85:fontsize=15:x=W-46:y={gy}"
)


def main() -> int:
    if not FONT.exists():
        return err(f"font not found: {FONT}")
    if not EX.is_dir():
        return err(f"example clips not found: {EX}")
    OUT.mkdir(parents=True, exist_ok=True)

    for name, (filename, y) in CASES.items():
        src = EX / filename
        if not src.exists():
            return err(f"missing {src}")
        truth, marked = OUT / f"{name}-truth.mp4", OUT / f"{name}-marked.mp4"
        shutil.copy(src, truth)

        res = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(truth),
             "-vf", BADGE_VF.format(font=FONT, y=y, ty=y + 3, gy=y + 15),
             "-c:v", "libx264", "-crf", "16", "-preset", "medium",
             "-pix_fmt", "yuv420p", str(marked)],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            return err(f"ffmpeg failed for {name}:\n{res.stderr[:800]}")

        info = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
             "stream=width,height,nb_frames,r_frame_rate", "-of", "csv=p=0", str(marked)],
            capture_output=True, text=True).stdout.strip()
        print(f"{name:7s} {info}  ->  {marked.name}   badge at 384,{y},84,36")

    print(f"\nfixtures in {OUT}")
    return 0


def err(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
