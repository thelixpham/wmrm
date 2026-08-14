"""Automated acceptance checks, so quality is not judged by eye alone.

Two things are asserted (KNOWLEDGE.md 6):

- Geometry/streams unchanged: same WxH, fps, duration, audio presence.
- The edit is *local*: PSNR outside the mask stays high (we did not touch the
  rest of the picture), while PSNR inside the mask is low (we did in fact change
  the watermark). A run that scores high inside the box did nothing.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .probe import probe, require_tools
from .region import Box

OUTSIDE_PSNR_FLOOR = 38.0   # below this, the whole frame was degraded
INSIDE_PSNR_CEIL = 40.0     # above this, nothing was actually changed


@dataclass
class VerifyResult:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def render(self) -> str:
        lines = [f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {d}" if d else "")
                 for name, ok, d in self.checks]
        lines.append(f"  => {'all checks passed' if self.ok else 'FAILURES PRESENT'}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """For `--report`.

        Checks become named objects rather than the positional triples used
        internally: a wire format where the reader has to know that index 1 is the
        pass flag breaks silently the moment a field is inserted.
        """
        return {
            "ok": self.ok,
            "checks": [{"name": name, "passed": bool(ok), "detail": detail}
                       for name, ok, detail in self.checks],
            "failed": [name for name, ok, _ in self.checks if not ok],
        }


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("inf")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-9:
        return float("inf")
    return 10.0 * float(np.log10(255.0 * 255.0 / mse))


def _mid_frame(path: Path, at: float) -> np.ndarray:
    ffmpeg, _ = require_tools()
    info = probe(path)
    res = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-ss", f"{at:.3f}", "-i", str(path),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True,
    )
    nbytes = info.width * info.height * 3
    if res.returncode != 0 or len(res.stdout) < nbytes:
        raise RuntimeError(f"could not read a frame from {path}")
    return np.frombuffer(res.stdout[:nbytes], np.uint8).reshape(
        info.height, info.width, 3
    ).copy()


def verify(original: Path, processed: Path, box: Box | None = None) -> VerifyResult:
    r = VerifyResult()
    a, b = probe(original), probe(processed)

    r.add("resolution", (a.width, a.height) == (b.width, b.height),
          f"{a.width}x{a.height} vs {b.width}x{b.height}")
    r.add("frame rate", a.fps == b.fps, f"{a.fps} vs {b.fps}")
    r.add("duration", abs(a.duration - b.duration) <= max(0.15, a.duration * 0.01),
          f"{a.duration:.2f}s vs {b.duration:.2f}s")
    r.add("audio stream", a.has_audio == b.has_audio,
          f"{'present' if a.has_audio else 'none'} vs "
          f"{'present' if b.has_audio else 'none'}")

    if box is not None and r.checks[0][1]:
        at = max(0.0, min(a.duration, b.duration) / 2)
        fa, fb = _mid_frame(original, at), _mid_frame(processed, at)

        inside = _psnr(fa[box.y: box.y + box.h, box.x: box.x + box.w],
                       fb[box.y: box.y + box.h, box.x: box.x + box.w])
        keep = np.ones(fa.shape[:2], bool)
        m = 24
        keep[max(0, box.y - m): box.y + box.h + m,
             max(0, box.x - m): box.x + box.w + m] = False
        outside = _psnr(fa[keep], fb[keep])

        r.add("rest of frame preserved", outside >= OUTSIDE_PSNR_FLOOR,
              f"PSNR outside mask {outside:.1f} dB (floor {OUTSIDE_PSNR_FLOOR})")
        r.add("watermark region changed", inside <= INSIDE_PSNR_CEIL,
              f"PSNR inside mask {inside:.1f} dB (ceiling {INSIDE_PSNR_CEIL})")

    return r
