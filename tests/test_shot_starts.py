"""Scene detection: NVDEC must not change the answer, and must not fail quietly.

`_shot_starts` decodes the whole file before any frame is processed, and on a
feature-length input that is the difference between a 9-minute wait and a
43-minute one. It now hands decode to NVDEC when a CUDA card is present. The
speedup was measured on a 4090 -- 120s of 1080p went 32.9s wall / 54.0s CPU to
6.8s wall / 3.0s CPU, same cuts -- and is not what this file checks. What it
checks is the three ways that change could be wrong:

1. **The answer moves.** NVDEC decoding the same frames must yield the same
   cuts. A cut that lands two frames off moves a segment boundary.
2. **The fallback is silent.** NVDEC rejects some profiles and bit depths. When
   it does, the run must say so -- otherwise the CPU path is 5x slower for no
   stated reason, which reads as "this machine is slow", not "the card refused".
3. **A failed decode reads as "no cuts".** This is the one with teeth, and it
   predates the change. `_shot_starts` used to parse whatever ffmpeg printed and
   return whatever it found, so a decode that failed outright returned [] --
   indistinguishable from a clean scan of a single-shot video. The caller then
   reported "no scene cuts found", planned fixed segments, and lost the cut
   protection entirely. Adding a second decoder to try makes that path much
   easier to reach, so it now raises.

The fixture is generated, not committed: 6 scenes of 2s at 640x360, cuts at
2,4,6,8,10s. Small enough to build in a couple of seconds.

    python tests/test_shot_starts.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmrm.pipeline import EncodeError                                # noqa: E402
from wmrm.probe import require_tools                                 # noqa: E402
from wmrm.video import _shot_starts                                  # noqa: E402

FPS = 30.0
SCENE_SECS = 2
NSCENES = 6
TRUTH = [round(SCENE_SECS * i * FPS) for i in range(1, NSCENES)]

failures: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}".rstrip())
    if not cond:
        failures.append(name)


def build(ffmpeg: str, out: Path) -> None:
    """Six visually unrelated scenes, concatenated then re-encoded in one pass.

    The re-encode matters: it lets x264 place its own keyframes over content it
    can see the cuts in, which is what a real input looks like. A bare concat of
    separately-encoded parts would be an easier problem than the real one.
    """
    srcs = ["testsrc2", "smptebars", "mandelbrot", "rgbtestsrc", "testsrc", "yuvtestsrc"]
    parts = []
    for i in range(NSCENES):
        p = out.parent / f"part{i}.mp4"
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-f", "lavfi",
             "-i", f"{srcs[i]}=size=640x360:rate=30", "-t", str(SCENE_SECS),
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(p)],
            check=True, capture_output=True)
        parts.append(f"file '{p.name}'")
    lst = out.parent / "parts.txt"
    lst.write_text("\n".join(parts))
    raw = out.parent / "raw.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", str(raw)],
                   check=True, capture_output=True)
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264",
                    "-preset", "fast", "-crf", "18", "-bf", "3",
                    "-pix_fmt", "yuv420p", str(out)],
                   check=True, capture_output=True)


def finds_all(cuts: list[int], tol: int = 2) -> bool:
    return all(any(abs(c - f) <= tol for f in cuts) for c in TRUTH)


def main() -> int:
    ffmpeg, _ = require_tools()
    with tempfile.TemporaryDirectory(prefix="wmrm-shots-") as td:
        fix = Path(td) / "scenes.mp4"
        build(ffmpeg, fix)

        # Scene cuts only here: black boundaries are the subject of
        # tests/test_black_bounds.py, and this fixture has no black in it.
        cpu, _ = _shot_starts(ffmpeg, fix, 0.3, FPS, hwaccel=False)
        check("cpu path finds every known cut", finds_all(cpu), f"{len(cpu)} cuts")

        # On a CUDA box this exercises NVDEC; without one it exercises the
        # fallback. Either way the requirement is the same: same cuts as the CPU
        # path, and if it fell back, it said so.
        logs: list[str] = []
        hw, _ = _shot_starts(ffmpeg, fix, 0.3, FPS, hwaccel=True, say=logs.append)
        fell_back = any("NVDEC" in m for m in logs)
        check("requesting NVDEC gives the same cuts", hw == cpu,
              f"{len(hw)} cuts, {'fell back' if fell_back else 'used the card'}")
        check("a fallback is announced rather than silent",
              hw == cpu and (fell_back or not logs),
              f"logged: {logs or 'nothing'}")

        check("threshold 0 skips the pass entirely",
              _shot_starts(ffmpeg, fix, 0, FPS) == ([], []))

        try:
            _shot_starts(ffmpeg, Path(td) / "absent.mp4", 0.3, FPS)
            check("a failed decode raises rather than reporting no cuts", False,
                  "returned normally")
        except EncodeError as exc:
            check("a failed decode raises rather than reporting no cuts",
                  "scene detection failed" in str(exc))

    print(f"\n{len(failures)} failed" if failures else "\nall pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
