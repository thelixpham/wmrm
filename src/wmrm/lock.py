"""One writer per output path.

`video.py::_usable_parts` deletes part files it does not consider usable:

    for stale in sorted(parts_dir.glob(PART_GLOB))[done:]:
        stale.unlink(missing_ok=True)

That is correct for one process resuming its own work and destructive when two
processes share a parts directory -- the second one deletes the first one's finished
parts and both then write `part-000042.mp4`. Until this module there was no lock
anywhere in the package, so nothing prevented it.

**The lock is per output path, not per directory.** Two runs writing different outputs
into the same directory are already safe: the non-ProPainter engines stage through
`tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.stem}.")` and `os.replace`, so their
temporaries cannot collide. Locking the directory would make those two runs exclude
each other for no reason -- and the common case is a directory like `/tmp` or an
outbox holding many outputs at once.

**`flock` is not dependable on a network filesystem.** If several machines mount the
same volume, this does not protect them from each other. Put the machine's identity in
the work directory instead -- the service uses `$WMRM_WORK_DIR/<podId>/<jobId>/` -- so
two machines never aim at one output path in the first place.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

from .errors import OutputLocked

SUFFIX = ".wmrm-lock"


def lock_path_for(dst: Path) -> Path:
    """Sibling of the output, hidden, one per output name."""
    return Path(dst).with_name(f".{Path(dst).name}{SUFFIX}")


@contextmanager
def output_lock(dst: Path, *, enabled: bool = True):
    """Hold an exclusive, non-blocking lock for as long as `dst` is being written.

    Non-blocking on purpose: waiting would turn "someone else is already making this
    file" into a run that appears hung, and the useful response is to say so and stop.
    """
    if not enabled:
        yield
        return

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    path = lock_path_for(dst)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OutputLocked(
                f"{dst} is already being written by another run (lock: {path}). "
                f"Wait for it to finish, or choose a different -o."
            ) from exc
        # Whose lock this is, for the person who finds it held and wants to know by
        # what. Best effort -- an unwritable directory is not a reason to refuse.
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except OSError:                                  # pragma: no cover
            pass
        yield
    finally:
        # Closing releases the lock. The file is left behind deliberately: unlinking it
        # races with another process that has just opened it and is about to flock, and
        # an empty hidden file costs nothing.
        os.close(fd)
