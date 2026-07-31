"""ffprobe/ffmpeg discovery and stream metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class ToolMissing(RuntimeError):
    pass


class ProbeError(RuntimeError):
    pass


def require_tools() -> tuple[str, str]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    missing = [n for n, p in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if not p]
    if missing:
        raise ToolMissing(
            f"{' and '.join(missing)} not found on PATH. Install with: apt install ffmpeg"
        )
    return ffmpeg, ffprobe  # type: ignore[return-value]


@dataclass(frozen=True)
class VideoInfo:
    source: Path
    width: int
    height: int
    fps: Fraction
    duration: float
    nframes: int
    has_audio: bool
    pix_fmt: str

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


def probe(path: Path) -> VideoInfo:
    _, ffprobe = require_tools()
    out = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise ProbeError(f"ffprobe failed on {path}:\n{out.stderr.strip()[:600]}")
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise ProbeError(f"ffprobe returned non-JSON for {path}") from exc

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ProbeError(f"{path} has no video stream")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    width, height = int(video["width"]), int(video["height"])

    # r_frame_rate is the exact rational rate; avg_frame_rate can be 0/0 on
    # some containers. Keep it a Fraction so the encoder gets the exact value
    # rather than a truncated float (that is what causes A/V drift).
    fps = Fraction(0)
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = video.get(key, "0/0")
        try:
            cand = Fraction(raw)
        except (ZeroDivisionError, ValueError):
            continue
        if cand > 0:
            fps = cand
            break
    if fps <= 0:
        raise ProbeError(f"could not determine frame rate of {path}")

    duration = 0.0
    for source in (video.get("duration"), data.get("format", {}).get("duration")):
        try:
            duration = float(source)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if duration > 0:
            break

    nframes = 0
    try:
        nframes = int(video.get("nb_frames") or 0)
    except (TypeError, ValueError):
        nframes = 0
    if nframes <= 0 and duration > 0:
        nframes = int(round(duration * float(fps)))

    return VideoInfo(
        source=Path(path),
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        nframes=nframes,
        has_audio=has_audio,
        pix_fmt=str(video.get("pix_fmt", "yuv420p")),
    )
