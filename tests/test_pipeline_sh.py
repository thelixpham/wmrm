"""Exercise run.sh end to end: first run calibrates, second run reuses the preset.

The script is the thing most people will actually invoke, so it gets the same
treatment as the library: run it for real and check the files it claims to produce.

    python tests/test_pipeline_sh.py
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SCRIPT = PROJECT / "run.sh"
SAMPLE = HERE / "fixtures" / "busy-marked.mp4"


def run_sh(work: Path, *argv: str, **env_extra) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "INBOX": str(work / "inbox"),
        "OUTBOX": str(work / "outbox"),
        "PRESET": str(work / "preset.json"),
        "QUALITY": "draft",          # fast enough for a test, same code path
        **{k: str(v) for k, v in env_extra.items()},
    }
    env.pop("VIRTUAL_ENV", None)     # force the script's own venv lookup
    return subprocess.run(["bash", str(SCRIPT), *argv],
                          capture_output=True, text=True, env=env)


def main() -> int:
    if not SAMPLE.exists():
        print(f"error: {SAMPLE} missing -- run tests/make_fixtures.py first",
              file=sys.stderr)
        return 1

    # syntax first: a broken script would otherwise fail in confusing ways
    check = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    if check.returncode != 0:
        print(f"FAIL syntax:\n{check.stderr}", file=sys.stderr)
        return 1
    print("ok    bash -n")

    if not SCRIPT.stat().st_mode & stat.S_IXUSR:
        SCRIPT.chmod(SCRIPT.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print("ok    executable bit")

    work = HERE / "_pipeline_tmp"
    shutil.rmtree(work, ignore_errors=True)
    (work / "inbox").mkdir(parents=True)
    for name in ("a.mp4", "b.mp4"):
        shutil.copy(SAMPLE, work / "inbox" / name)

    bad = 0

    # --- empty inbox is not an error -------------------------------------------
    empty = HERE / "_pipeline_empty"
    shutil.rmtree(empty, ignore_errors=True)
    (empty / "inbox").mkdir(parents=True)
    res = run_sh(empty)
    ok = res.returncode == 0 and "nothing to do" in res.stdout
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'}  empty inbox exits 0 with a message")
    shutil.rmtree(empty, ignore_errors=True)

    # --- first run: no preset, should detect, process, and save the preset -----
    res = run_sh(work)
    if res.returncode != 0:
        print(f"FAIL first run exited {res.returncode}\n{res.stdout[-2000:]}\n"
              f"{res.stderr[-2000:]}", file=sys.stderr)
        return 1
    print("ok    first run exited 0")

    for label, cond in [
        ("detected rather than using a preset", "no preset yet" in res.stdout),
        ("preset saved for next time", (work / "preset.json").exists()),
        ("outputs moved to outbox", sorted(p.name for p in (work / "outbox").iterdir())
         == ["a-clean.mp4", "b-clean.mp4"]),
        ("inbox originals untouched", sorted(p.name for p in (work / "inbox").glob("*.mp4"))
         == ["a.mp4", "b.mp4"]),
        ("points at a preview to check", "-preview-zoom.png" in res.stdout),
    ]:
        bad += not cond
        print(f"{'ok  ' if cond else 'FAIL'}  {label}")

    # --- second run: preset exists, must reuse it and not detect again ---------
    for name in ("c.mp4",):
        shutil.copy(SAMPLE, work / "inbox" / name)
    res2 = run_sh(work)
    for label, cond in [
        ("second run exited 0", res2.returncode == 0),
        ("reused the saved preset", "using saved box" in res2.stdout),
        ("did not detect again", "no preset yet" not in res2.stdout),
        ("processed only the new file", (work / "outbox" / "c-clean.mp4").exists()),
        ("skipped the already-done ones", "skip (exists)" in res2.stdout),
    ]:
        bad += not cond
        print(f"{'ok  ' if cond else 'FAIL'}  {label}")

    # --- explicit file arguments ------------------------------------------------
    loose = work / "loose"
    loose.mkdir()
    for name in ("one.mp4", "two.mp4", "three.mp4"):
        shutil.copy(SAMPLE, loose / name)

    res3 = run_sh(work, str(loose / "one.mp4"))
    for label, cond in [
        ("single file argument exits 0", res3.returncode == 0),
        ("single file produced its output", (work / "outbox" / "one-clean.mp4").exists()),
        ("single file did not pull in its neighbours",
         not (work / "outbox" / "two-clean.mp4").exists()),
    ]:
        bad += not cond
        print(f"{'ok  ' if cond else 'FAIL'}  {label}")

    res4 = run_sh(work, str(loose / "two.mp4"), str(loose / "three.mp4"))
    ok = res4.returncode == 0 and all(
        (work / "outbox" / f"{n}-clean.mp4").exists() for n in ("two", "three"))
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'}  several file arguments")

    # --- an arbitrary directory argument ---------------------------------------
    other = work / "other"
    other.mkdir()
    shutil.copy(SAMPLE, other / "dir1.mp4")
    res5 = run_sh(work, str(other))
    ok = res5.returncode == 0 and (work / "outbox" / "dir1-clean.mp4").exists()
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'}  directory argument")

    # --- bad input is rejected, not silently ignored ---------------------------
    res6 = run_sh(work, str(work / "nope.mp4"))
    ok = res6.returncode != 0 and "no such file" in res6.stderr
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'}  missing path fails loudly")

    res7 = run_sh(work, str(work / "preset.json"))
    ok = res7.returncode != 0 and "not a video" in res7.stderr
    bad += not ok
    print(f"{'ok  ' if ok else 'FAIL'}  non-video file fails loudly")

    shutil.rmtree(work, ignore_errors=True)
    print()
    if bad:
        print(f"{bad} check(s) failed")
        return 1
    print("run.sh behaves as documented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
