#!/usr/bin/env python
"""Output-lock tests. Run directly: `python tests/test_outputlock.py`.

The interesting property is the **granularity**, and it is the one a naive test misses.
Locking the output's *directory* would also make the "same output twice" case fail, so a
test that only checks that case passes either way. What separates them is the second
assertion: two runs writing **different** outputs into the **same** directory must not
block each other -- that is legal today, since the non-ProPainter engines stage through
`tempfile.mkstemp(dir=dst.parent, ...)` and rename, and outboxes routinely hold many
outputs at once.

Fast: this exercises the primitive in-process rather than paying ~78 seconds per real run.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))


def _try_lock(path_str: str, q) -> None:
    """Acquire in a child process, report whether it got it.

    A separate process, not a thread: `flock` is held per open file description, and two
    threads in one process would not contend the way two `wmrm run` invocations do.
    """
    from wmrm.errors import OutputLocked
    from wmrm.lock import output_lock

    try:
        with output_lock(Path(path_str)):
            q.put("acquired")
    except OutputLocked:
        q.put("blocked")
    except Exception as exc:                              # noqa: BLE001
        q.put(f"error:{type(exc).__name__}:{exc}")


def _child_result(path: Path) -> str:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_try_lock, args=(str(path), q))
    p.start()
    p.join(30)
    return q.get(timeout=5) if not q.empty() else "timeout"


def main() -> int:
    from wmrm.errors import OutputLocked
    from wmrm.lock import lock_path_for, output_lock

    print("\n[output lock]")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        a = d / "clipA-clean.mp4"
        b = d / "clipB-clean.mp4"

        with output_lock(a):
            check("the lock file is a sibling of the output, not the directory",
                  lock_path_for(a).parent == a.parent
                  and lock_path_for(a).name.startswith(".")
                  and a.name in lock_path_for(a).name,
                  str(lock_path_for(a).name))

            # Same output: must be refused.
            check("a second holder of the SAME output is blocked",
                  _child_result(a) == "blocked")

            # Different output, same directory: must be allowed. This is the assertion
            # that distinguishes a per-path lock from a per-directory one.
            res_b = _child_result(b)
            check("a holder of a DIFFERENT output in the same directory is allowed",
                  res_b == "acquired", res_b)

        # Released on exit, so the next run can have it.
        check("the lock is released when the block exits",
              _child_result(a) == "acquired")

        # Re-entering in the same process after release works too.
        try:
            with output_lock(a):
                pass
            check("the same process can re-acquire after releasing", True)
        except OutputLocked:
            check("the same process can re-acquire after releasing", False)

        # Opting out has to actually opt out -- the service relies on the default, but
        # `--no-lock` exists for a caller that knows better.
        with output_lock(a):
            try:
                with output_lock(a, enabled=False):
                    check("enabled=False skips the lock entirely", True)
            except OutputLocked:
                check("enabled=False skips the lock entirely", False)

        # A missing parent directory is created rather than being an error: the service
        # locks an output inside a per-job work directory that may not exist yet.
        nested = d / "deep" / "nested" / "out.mp4"
        try:
            with output_lock(nested):
                check("a missing parent directory is created", nested.parent.is_dir())
        except Exception as exc:                          # noqa: BLE001
            check("a missing parent directory is created", False, str(exc))

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
