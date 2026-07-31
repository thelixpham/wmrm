"""Assert the .gitignore actually does what it claims.

Written because a .gitignore is easy to get subtly wrong and the failure is
silent -- you notice when a 1 GB venv or someone else's source footage lands in a
commit. Uses `git check-ignore` against a throwaway repo so nothing here depends
on this directory being a repo yet.

    python tests/check_gitignore.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent          # wmrm/
PARENT = PROJECT.parent        # remove-watermark/

MUST_IGNORE = [
    ".venv/bin/wmrm",
    "src/wmrm/__pycache__/cli.cpython-312.pyc",
    "wmrm.egg-info/PKG-INFO",
    "tests/fixtures/busy-truth.mp4",
    "tests/fixtures/busy-comparison.png",
    "tests/fixtures/wm-preset.json",
    "video-clean.mp4",
    "sample-preview.png",
    "sample-preview-zoom.png",
    "sample-boxcheck-zoom.png",
    ".busy-marked.a1b2c3.mp4",   # atomic-write temp
    "big-lama.pt",
]

MUST_TRACK = [
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "src/wmrm/cli.py",
    "src/wmrm/backends.py",
    "tests/make_fixtures.py",
    "tests/score.py",
    # the calibration result is worth committing once confirmed
    "wm-preset.json",
]

PARENT_MUST_IGNORE = [
    "dreamina-delogo/delogo.sh",
    "WatermarkRemover-AI/remwm.py",
    "some-source-footage.mp4",
    "frame001.png",
]

PARENT_MUST_TRACK = [
    "KNOWLEDGE.md",
    "REPORT.md",
    "wmrm/src/wmrm/cli.py",
]


def check(gitignore: Path, must_ignore: list[str], must_track: list[str], label: str) -> int:
    if not gitignore.exists():
        print(f"error: {gitignore} not found", file=sys.stderr)
        return 1
    if not shutil.which("git"):
        print("error: git not on PATH", file=sys.stderr)
        return 1

    bad = 0
    with tempfile.TemporaryDirectory(prefix="wmrm-gitignore-") as td:
        repo = Path(td)
        subprocess.run(["git", "init", "-q", str(repo)], check=True,
                       capture_output=True)
        shutil.copy(gitignore, repo / ".gitignore")

        def ignored(path: str) -> bool:
            return subprocess.run(
                ["git", "-C", str(repo), "check-ignore", "-q", "--no-index", path]
            ).returncode == 0

        print(f"\n{label}  ({gitignore})")
        print("-" * 66)
        for p in must_ignore:
            ok = ignored(p)
            bad += not ok
            print(f"  {'ok  ' if ok else 'FAIL'}  ignored   {p}")
        for p in must_track:
            ok = not ignored(p)
            bad += not ok
            print(f"  {'ok  ' if ok else 'FAIL'}  tracked   {p}")
    return bad


def main() -> int:
    bad = check(PROJECT / ".gitignore", MUST_IGNORE, MUST_TRACK, "wmrm/")
    bad += check(PARENT / ".gitignore", PARENT_MUST_IGNORE, PARENT_MUST_TRACK,
                 "remove-watermark/")
    print()
    if bad:
        print(f"{bad} rule(s) wrong")
        return 1
    print("all .gitignore rules behave as intended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
