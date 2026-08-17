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

# `OUTSIDE_PSNR_FLOOR` is not yet calibrated and is known to be tight. Nothing outside
# the mask is repainted, so what it actually measures is libx264 loss at the run's crf,
# which depends on the content: a 2h05m clip that passed scored 39.3 dB, 1.3 dB of
# margin, and a grainier one is expected to sit lower for no fault of the run. Raising
# or lowering it from one anecdote would be guessing, so the numbers needed to set it --
# the spread across frames, and near-band against far-field -- are recorded on every run
# instead. Set it from those.


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


def _sample_times(duration: float, n: int) -> list[float]:
    """Spread the samples, avoiding the very ends where fades and black frames live."""
    if duration <= 0:
        return [0.0]
    return list(np.linspace(duration * 0.05, duration * 0.95, num=max(1, n)))


def _measure(fa: np.ndarray, fb: np.ndarray, box: Box) -> tuple[float, float, float, float]:
    """PSNR inside the box, outside the mask, and outside split into near and far.

    The near/far split is diagnostic, not a verdict. Nothing repaints beyond the box
    plus dilate and feather (17 px against the 24 px this masks out), so PSNR outside
    the mask measures the **re-encode**, not damage -- which is why an absolute floor on
    it is really a floor on how expensive this content is to encode. If the repaint ever
    did leak, it would show up close to the box and not far from it. Uniform near and far
    means encode loss; near much worse than far means something escaped the mask.
    """
    inside = _psnr(fa[box.y: box.y + box.h, box.x: box.x + box.w],
                   fb[box.y: box.y + box.h, box.x: box.x + box.w])

    m = 24
    masked = np.zeros(fa.shape[:2], bool)
    masked[max(0, box.y - m): box.y + box.h + m,
           max(0, box.x - m): box.x + box.w + m] = True

    band = 96
    ring = np.zeros(fa.shape[:2], bool)
    ring[max(0, box.y - band): box.y + box.h + band,
         max(0, box.x - band): box.x + box.w + band] = True

    outside = _psnr(fa[~masked], fb[~masked])
    near = _psnr(fa[ring & ~masked], fb[ring & ~masked])
    far = _psnr(fa[~ring], fb[~ring])
    return outside, inside, near, far


def verify(original: Path, processed: Path, box: Box | None = None, *,
           samples: int = 9) -> VerifyResult:
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
        dur = min(a.duration, b.duration)
        outside, inside, near, far = [], [], [], []
        for at in _sample_times(dur, samples):
            try:
                fa, fb = _mid_frame(original, at), _mid_frame(processed, at)
            except RuntimeError:
                # One unreadable frame is not a verdict on the file. Only having none
                # of them is, and that is handled below.
                continue
            o, i, n, f = _measure(fa, fb, box)
            outside.append(o)
            inside.append(i)
            near.append(n)
            far.append(f)

        if not outside:
            r.add("rest of frame preserved", False,
                  f"could not read any of the {samples} sampled frames")
            return r

        med_out, med_in = float(np.median(outside)), float(np.median(inside))
        # The median, not the single mid-point frame this used to read. A `-ss` seek can
        # land on different pictures in two files whose keyframes differ -- the output is
        # re-encoded and, for a long clip, concatenated from many parts -- and on moving
        # content one frame of misalignment reads as tens of dB of "damage". Deciding a
        # six-hour run on that one draw was a coin flip; the spread below is what tells
        # a misread frame (one outlier) from a real one (the whole distribution moves).
        r.add("rest of frame preserved", med_out >= OUTSIDE_PSNR_FLOOR,
              f"PSNR outside mask median {med_out:.1f} dB over {len(outside)} frames "
              f"(min {min(outside):.1f}, max {max(outside):.1f}, "
              f"floor {OUTSIDE_PSNR_FLOOR})  |  "
              f"near-band {float(np.median(near)):.1f} vs far-field "
              f"{float(np.median(far)):.1f} dB")
        r.add("watermark region changed", med_in <= INSIDE_PSNR_CEIL,
              f"PSNR inside mask median {med_in:.1f} dB over {len(inside)} frames "
              f"(min {min(inside):.1f}, max {max(inside):.1f}, "
              f"ceiling {INSIDE_PSNR_CEIL})")

    return r
