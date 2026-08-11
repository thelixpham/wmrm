"""Where a joined video's timing went wrong, when it does.

Not a test -- a probe to run when tests/test_resume.py reports `timing changed`.
That check compares the finished video's frame rate and duration against the source,
and it fails for reasons that live outside this project: what a container rounds, and
what a particular ffmpeg build's concatenator believes about a file's length. The
numbers differ between builds, so the useful thing is not another opinion but a
measurement from the machine that failed.

It prints, for the source, every part, and the joined result: frame rate, timebase,
frame count, and duration at both the track and the file level -- the two places a
duration is stored and the two places it can be rounded. Then it re-joins those same
parts four ways and reports each, so the fix is picked from what the build in front of
you actually does rather than from what another one did.

    python tests/diag_join.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from wmrm import video as V                    # noqa: E402
import test_resume as T                        # noqa: E402

TOTAL = 60          # frames in the fixture, matching tests/test_resume.py
TIMESCALE = "30000"


def info(tag: str, path: Path) -> None:
    stream = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,nb_frames,duration,time_base,start_time",
         "-of", "default=nw=1", str(path)], capture_output=True, text=True).stdout
    file_dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    print(f"  {tag:18s} " + " ".join(stream.split()) + f" file_dur={file_dur}")


def quote(p: Path) -> str:
    return str(p.resolve()).replace("'", r"'\''")


def main() -> int:
    print(subprocess.run(["ffmpeg", "-version"], capture_output=True,
                         text=True).stdout.splitlines()[0])

    keep, V.shutil.rmtree = V.shutil.rmtree, lambda *a, **k: None
    try:
        with tempfile.TemporaryDirectory(prefix="wmrm-diag-") as td:
            work = Path(td)
            src = work / "src.mp4"
            T.write_source(T.marked_frames(), src)
            dst = work / "out.mp4"
            T.run(src, dst, worker=T.StubWorker(), resume=False)

            parts = sorted((work / "out.mp4.parts").glob("part-*.mp4"))
            info("source", src)
            for p in parts:
                info(p.name, p)
            info("joined", dst)

            fps, n = Fraction(30000, 1001), T.PART
            plain = work / "plain.txt"
            plain.write_text("".join(f"file '{quote(p)}'\n" for p in parts))
            exact = work / "exact.txt"
            exact.write_text("".join(
                f"file '{quote(p)}'\n"
                f"duration {float(Fraction(min(n, TOTAL - i * n)) / fps):.9f}\n"
                for i, p in enumerate(parts)))

            for tag, listing, extra in (
                ("plain", plain, []),
                ("plain+movie_ts", plain, ["-movie_timescale", TIMESCALE]),
                ("exact durations", exact, []),
                ("exact+movie_ts", exact, ["-movie_timescale", TIMESCALE]),
            ):
                out = work / "join.mp4"
                res = subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(listing), "-c", "copy",
                     "-video_track_timescale", TIMESCALE, *extra, str(out)],
                    capture_output=True, text=True)
                if res.returncode:
                    print(f"  {tag:18s} ERR {res.stderr.strip()[:100]}")
                else:
                    info(tag, out)
    finally:
        V.shutil.rmtree = keep

    print(f"  {'exact answer':18s} r_frame_rate=30000/1001 nb_frames={TOTAL} "
          f"duration=2.002000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
