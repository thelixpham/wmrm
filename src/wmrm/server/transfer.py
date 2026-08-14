"""Moving bytes on and off this machine.

Two routes, and the R2 one is the normal one:

- **`pull_r2` / `push_r2`** use `wmrm.r2`, which is 8 parallel ranged GETs into a
  preallocated file for downloads and a worker pool of UploadPart calls for uploads,
  both resumable at chunk granularity. This is what the sizes here need: 20-100 GB over
  one connection is slower by the width of the parallelism, and resume driven by R2's own
  ListParts cannot drift out of sync with the server the way a sidecar file can.
- **`download`** is the single-stream presigned-URL path, for a pod that holds no
  credentials. Chunked and resumable too, but one connection.

Both are synchronous underneath -- boto3 is, and so is the thread pool it drives -- so
the callers run them off the event loop. With a single uvicorn worker, doing that work
inline would stall every other request on a machine that is already busy.

Free space is checked before the first byte rather than discovered at 90 GB. A run needs
the input, the parts directory and the output alive at once, so budget roughly three
times the source.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Awaitable, Callable

import httpx

CHUNK = 8 * 1024 * 1024
CONNECT_TIMEOUT = 30.0
READ_TIMEOUT = 300.0


class TransferError(RuntimeError):
    pass


class NotEnoughSpace(TransferError):
    pass


async def stat_r2(key: str, *, bucket: str | None = None) -> tuple[int, str]:
    """Size and key of an R2 object, without downloading it.

    Used before accepting the transfer so the space check has a real number instead of
    whatever the caller claimed in `sizeBytes`.
    """
    from ..r2 import stat

    info = await asyncio.to_thread(stat, key, bucket=bucket)
    return int(info.size), str(info.key)


async def pull_r2(key: str, dest: Path, *, bucket: str | None = None,
                  workers: int = 8, progress: bool = False) -> Path:
    """Fetch an R2 key with the parallel, resumable downloader.

    `overwrite=False` is deliberate: `wmrm.r2.download` then skips a file that is already
    present at the right size, which is what makes a re-dispatch of the same job cheap
    instead of re-fetching 100 GB.
    """
    from ..r2 import download as r2_download

    return await asyncio.to_thread(
        r2_download, key, dest, bucket=bucket, workers=workers,
        progress=progress, overwrite=False)


async def push_r2(path: Path, key: str, *, bucket: str | None = None,
                  workers: int = 8, progress: bool = False) -> str:
    """Send the finished output up, multipart and resumable. Returns the key."""
    from ..r2 import upload as r2_upload

    return await asyncio.to_thread(
        r2_upload, path, key, bucket=bucket, workers=workers, progress=progress)


async def abort_r2(key: str, *, bucket: str | None = None) -> int:
    """Drop any multipart upload left open for this key.

    Called when a job ends without a successful upload. The parts of an incomplete
    upload are billed while appearing in no listing, so nothing else would ever draw
    attention to them.
    """
    from ..r2 import abort_uploads

    return await asyncio.to_thread(abort_uploads, key, bucket=bucket)


def require_space(path: Path, needed_bytes: int, *, headroom: float = 3.0) -> None:
    """Refuse before starting rather than fail deep into a job.

    `headroom` defaults to 3x the source: input, parts and output coexist. It is a rule
    of thumb, not a measurement, which is why the number is visible here instead of
    buried in a comparison.
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        usage = None
    # Nothing to compare against is not a refusal. A network work dir that will not answer
    # `statvfs` would otherwise fail every job on a filesystem with room to spare -- and the
    # job that really does run out still fails, just later and with its own message.
    if usage is None or usage.total == 0:
        return
    needed = int(needed_bytes * headroom)
    if needed and usage.free < needed:
        raise NotEnoughSpace(
            f"{path} has {usage.free / 1024**3:.1f} GiB free; this job needs about "
            f"{needed / 1024**3:.1f} GiB (source {needed_bytes / 1024**3:.1f} GiB "
            f"x{headroom:g} for input + parts + output)"
        )


async def download(url: str, dest: Path, *,
                   expected_size: int | None = None,
                   refresh_url: Callable[[], Awaitable[str]] | None = None,
                   on_progress: Callable[[int, int | None], None] | None = None,
                   attempts: int = 6) -> Path:
    """Fetch `url` to `dest`, resuming into `dest.part` across retries.

    `refresh_url` exists because a presigned URL has a lifetime and these downloads do
    not fit inside a short one. When the server rejects the range, the caller is asked
    for a fresh URL rather than the transfer being abandoned -- the bytes on disk are
    still good.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        # Already here and the right length. Re-fetching 100 GB to confirm what the
        # length already tells us would be its own kind of failure.
        return dest

    delay = 2.0
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if expected_size and have == expected_size:
            os.replace(part, dest)
            return dest

        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            timeout = httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                async with client.stream("GET", url, headers=headers) as res:
                    if res.status_code in (403, 401, 410) and refresh_url is not None:
                        # Almost certainly an expired signature. Ask for a new one and
                        # keep the bytes.
                        url = await refresh_url()
                        raise TransferError(f"signature rejected ({res.status_code})")
                    if have and res.status_code == 200:
                        # Range ignored: this response is the whole object, so what is on
                        # disk is not a prefix of what is arriving. Start over rather than
                        # append and produce a corrupt mix.
                        have = 0
                        part.unlink(missing_ok=True)
                    elif have and res.status_code != 206:
                        raise TransferError(
                            f"expected 206 for a ranged request, got {res.status_code}")
                    elif not have and res.status_code >= 300:
                        raise TransferError(f"GET failed: {res.status_code}")

                    total = expected_size
                    if total is None:
                        length = res.headers.get("content-length")
                        if length and length.isdigit():
                            total = have + int(length)

                    mode = "ab" if have else "wb"
                    with open(part, mode) as fh:
                        async for chunk in res.aiter_bytes(CHUNK):
                            fh.write(chunk)
                            have += len(chunk)
                            if on_progress is not None:
                                on_progress(have, total)

            if expected_size and part.stat().st_size != expected_size:
                raise TransferError(
                    f"got {part.stat().st_size} bytes, expected {expected_size}")
            os.replace(part, dest)
            return dest

        except (httpx.HTTPError, TransferError, OSError) as exc:
            if attempt == attempts:
                raise TransferError(f"download failed after {attempts} attempts: {exc}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)

    raise TransferError("download failed")            # pragma: no cover
