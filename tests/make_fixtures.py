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
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def find_font() -> Path | None:
    return next((p for c in FONT_CANDIDATES if (p := Path(c)).exists()), None)

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

# Fallback sources, synthesized by ffmpeg when the reference clips are not
# present. The point of fixtures is a self-contained smoke test that works on any
# machine -- depending on sibling repos meant this script broke the moment only
# wmrm/ was copied somewhere.
#
# mandelbrot is deliberately chosen for the hard cases: it is dense, aperiodic
# detail with real motion, which is exactly what defeats delogo and cv2.inpaint.
SIZE, RATE, SECONDS = "480x640", 24, 15
SYNTH = {
    "busy": f"mandelbrot=size={SIZE}:rate={RATE}",
    "detail": f"mandelbrot=size={SIZE}:rate={RATE}",
    "smooth": (f"gradients=size={SIZE}:rate={RATE}"
               ":c0=0x241b4d:c1=0x8fa8d8:speed=0.015"),
}

# Semi-transparent plate plus bright glyphs, shaped like a real AI-generator
# badge. The plate matters: it makes the watermark semi-transparent, so the
# opacity diagnosis in `wmrm detect` has something real to detect.
BADGE_TEXT = (
    "drawbox=x=iw-96:y={y}:w=84:h=36:color=black@0.30:t=fill,"
    "drawtext=fontfile={font}:text='AI':fontcolor=white@0.92:fontsize=28:x=W-88:y={ty},"
    "drawtext=fontfile={font}:text='gen':fontcolor=white@0.85:fontsize=15:x=W-46:y={gy}"
)

# No font on the box: build the glyphs out of rectangles instead. Uglier, but it
# exercises the same code paths -- a fixed semi-transparent mark with hard strokes
# over the background.
BADGE_SHAPES = (
    "drawbox=x=iw-96:y={y}:w=84:h=36:color=black@0.30:t=fill,"
    "drawbox=x=iw-88:y={ty}:w=8:h=24:color=white@0.92:t=fill,"
    "drawbox=x=iw-72:y={ty}:w=8:h=24:color=white@0.92:t=fill,"
    "drawbox=x=iw-88:y={gy}:w=24:h=6:color=white@0.92:t=fill,"
    "drawbox=x=iw-44:y={ty}:w=28:h=7:color=white@0.85:t=fill,"
    "drawbox=x=iw-44:y={gy}:w=28:h=7:color=white@0.85:t=fill"
)


def synth_truth(name: str, dst: Path) -> str | None:
    """Render a synthetic source clip. Returns an error string on failure."""
    res = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", SYNTH[name], "-t", str(SECONDS),
         "-c:v", "libx264", "-crf", "14", "-preset", "medium",
         "-pix_fmt", "yuv420p", str(dst)],
        capture_output=True, text=True,
    )
    return None if res.returncode == 0 else res.stderr[:600]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    font = find_font()
    if font is None:
        print("note: no TrueType font found; drawing the badge from rectangles.\n"
              "      Install fonts-dejavu for a more realistic text badge.\n")
    OUT.mkdir(parents=True, exist_ok=True)

    have_clips = EX.is_dir()
    if not have_clips:
        print(f"note: reference clips not found at {EX}")
        print("      synthesizing sources with ffmpeg instead.\n"
              "      Absolute quality numbers will differ from those in the README,\n"
              "      which were measured on the reference clips. Comparisons between\n"
              "      backends on these fixtures are still valid.\n")

    for name, (filename, y) in CASES.items():
        truth, marked = OUT / f"{name}-truth.mp4", OUT / f"{name}-marked.mp4"

        src = EX / filename
        if have_clips and src.exists():
            shutil.copy(src, truth)
        else:
            if problem := synth_truth(name, truth):
                return err(f"could not synthesize {name}:\n{problem}")

        template = BADGE_TEXT if font else BADGE_SHAPES
        res = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(truth),
             "-vf", template.format(font=font, y=y, ty=y + 3, gy=y + 15),
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
