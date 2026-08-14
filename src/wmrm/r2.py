"""Pull source video out of Cloudflare R2 before processing.

Written for the file sizes this project actually sees -- 22 to 100 GB per clip --
which rules out the obvious implementation. `boto3.download_file` parallelises
ranged GETs but keeps no record of what it finished, so a dropped connection at
the 80th gigabyte throws away the first 79 and starts again. At these sizes that
is not an edge case, it is the normal outcome of a long enough transfer.

So: fixed-size chunks, a worker pool of ranged GETs, and a sidecar JSON listing
the chunks that landed. Re-running the same command resumes. Chunk writes go
through os.pwrite into one preallocated file, which is positional and needs no
lock -- there is no seek+write race to lose.

Credentials come from the environment (see `Creds.from_env`); nothing is read
from or written to this repo.

    from wmrm.r2 import download

    path = download("uploads/3d80.../MOGI-125.mp4", Path("work/"))
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

__all__ = ["Creds", "R2Error", "abort_uploads", "download", "ls", "parse_uri",
           "stat", "upload"]

CHUNK = 64 * 1024 * 1024          # 64 MiB: ~1600 parts at 100 GB, ~350 at 22 GB
WORKERS = 8
ATTEMPTS = 5                      # per chunk, with backoff
STATE_SUFFIX = ".wmrm-dl.json"
PART_SUFFIX = ".part"


class R2Error(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Creds:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    endpoint: str

    @staticmethod
    def from_env(bucket: str | None = None) -> "Creds":
        """Read credentials from the environment.

        R2_* first, AWS_*/S3_* as a fallback so an existing S3 profile in the
        shell keeps working. R2_ENDPOINT overrides the derived URL, which is
        what a custom domain or an account-less jurisdiction endpoint needs.
        """
        env = os.environ.get
        account = env("R2_ACCOUNT_ID", "") or env("CLOUDFLARE_ACCOUNT_ID", "")
        key = env("R2_ACCESS_KEY_ID", "") or env("AWS_ACCESS_KEY_ID", "")
        secret = (env("R2_SECRET_ACCESS_KEY", "")
                  or env("AWS_SECRET_ACCESS_KEY", ""))
        bucket = bucket or env("R2_BUCKET", "") or env("S3_BUCKET", "")
        endpoint = env("R2_ENDPOINT", "") or env("R2_ENDPOINT_URL", "")

        if not endpoint:
            if not account:
                raise R2Error(
                    "no R2 endpoint: set R2_ACCOUNT_ID (or R2_ENDPOINT directly).\n"
                    "  export R2_ACCOUNT_ID=...\n"
                    "  export R2_ACCESS_KEY_ID=...\n"
                    "  export R2_SECRET_ACCESS_KEY=...\n"
                    "  export R2_BUCKET=remove-watermark"
                )
            endpoint = f"https://{account}.r2.cloudflarestorage.com"
        missing = [n for n, v in (("R2_ACCESS_KEY_ID", key),
                                  ("R2_SECRET_ACCESS_KEY", secret),
                                  ("R2_BUCKET", bucket)) if not v]
        if missing:
            raise R2Error(f"missing environment: {', '.join(missing)}")
        return Creds(account, key, secret, bucket, endpoint.rstrip("/"))


def parse_uri(uri: str, *, default_bucket: str | None = None) -> tuple[str | None, str]:
    """Split `r2://bucket/key`, `s3://bucket/key` or a bare key into (bucket, key).

    A bare key returns bucket=None, meaning "whatever the environment says".
    """
    if "://" in uri:
        u = urlparse(uri)
        if u.scheme not in ("r2", "s3"):
            raise R2Error(f"not an R2 URI: {uri} (expected r2://bucket/key)")
        key = u.path.lstrip("/")
        if not key:
            raise R2Error(f"no object key in {uri}")
        return (u.netloc or default_bucket), key
    return default_bucket, uri.lstrip("/")


def _client(creds: Creds, *, workers: int):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:                       # pragma: no cover
        raise R2Error(
            "boto3 is not installed. Install the R2 extra:\n"
            "  uv pip install -e '.[r2]'      # or: pip install boto3"
        ) from exc

    return boto3.client(
        "s3",
        endpoint_url=creds.endpoint,
        aws_access_key_id=creds.access_key_id,
        aws_secret_access_key=creds.secret_access_key,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            # Every worker holds a connection for the whole body of its chunk.
            # A pool smaller than the worker count serialises them silently --
            # the download still finishes, just at a fraction of the speed, with
            # nothing in the output to say why.
            max_pool_connections=max(workers + 4, 10),
            retries={"max_attempts": 3, "mode": "standard"},
            # Generous: a 64 MiB body on a slow link legitimately takes minutes,
            # and a timeout here means re-fetching the whole chunk.
            connect_timeout=20,
            read_timeout=300,
        ),
    )


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ObjectInfo:
    bucket: str
    key: str
    size: int
    etag: str
    content_type: str

    def describe(self) -> str:
        return (f"r2://{self.bucket}/{self.key}\n"
                f"  size : {_human(self.size)} ({self.size} bytes)\n"
                f"  etag : {self.etag}\n"
                f"  type : {self.content_type or 'unknown'}")


def stat(key: str, *, bucket: str | None = None, creds: Creds | None = None
         ) -> ObjectInfo:
    """HEAD the object. Cheap, and it is what sizes the transfer below."""
    bucket, key = parse_uri(key, default_bucket=bucket)
    creds = creds or Creds.from_env(bucket)
    bucket = bucket or creds.bucket
    cli = _client(creds, workers=1)
    try:
        head = cli.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        raise R2Error(_explain(exc, bucket, key)) from exc
    return ObjectInfo(
        bucket=bucket, key=key,
        size=int(head["ContentLength"]),
        etag=str(head.get("ETag", "")).strip('"'),
        content_type=head.get("ContentType", ""),
    )


def _explain(exc: Exception, bucket: str, key: str) -> str:
    """Turn botocore's error codes into something that names the likely cause."""
    code = ""
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = str(resp.get("Error", {}).get("Code", ""))
    if code in ("404", "NoSuchKey", "NotFound"):
        return (f"object not found: r2://{bucket}/{key}\n"
                f"  list what is there: wmrm pull --list {key.rsplit('/', 1)[0]}/")
    if code in ("NoSuchBucket",):
        return f"bucket {bucket!r} does not exist on this account"
    if code in ("403", "AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
        return (f"access denied for r2://{bucket}/{key} ({code or 'AccessDenied'})\n"
                "  check R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY, and that the API "
                "token covers this bucket")
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# resume state
# --------------------------------------------------------------------------- #

class _State:
    """Which chunks are already on disk, persisted next to the .part file.

    Flushed on a timer rather than per chunk: at 64 MiB a chunk lands every few
    seconds, and rewriting the list each time is pointless IO. Losing the last
    few seconds of it on a hard kill costs a re-fetch of those chunks, nothing
    worse -- the file itself is always consistent, because a chunk is only ever
    recorded after its pwrite returned.
    """

    def __init__(self, path: Path, info: ObjectInfo, chunk: int):
        self.path = path
        self.info = info
        self.chunk = chunk
        self.done: set[int] = set()
        self._lock = threading.Lock()
        self._dirty = False
        self._flushed = 0.0

    def load(self) -> bool:
        """Adopt an existing state file. False if it does not match this object."""
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return False
        if (raw.get("key") != self.info.key
                or raw.get("bucket") != self.info.bucket
                or raw.get("size") != self.info.size
                or raw.get("etag") != self.info.etag
                or raw.get("chunk") != self.chunk):
            # The object changed under us, or the chunking did. Either way the
            # bytes on disk are not a prefix of what we now want.
            return False
        self.done = set(raw.get("done", []))
        return True

    def mark(self, index: int) -> None:
        with self._lock:
            self.done.add(index)
            self._dirty = True
            if time.monotonic() - self._flushed > 5.0:
                self._write_locked()

    def flush(self) -> None:
        with self._lock:
            if self._dirty:
                self._write_locked()

    def _write_locked(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({
            "bucket": self.info.bucket, "key": self.info.key,
            "size": self.info.size, "etag": self.info.etag,
            "chunk": self.chunk, "done": sorted(self.done),
        }))
        tmp.replace(self.path)
        self._dirty = False
        self._flushed = time.monotonic()


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #

def download(key: str, dest: Path | str, *, bucket: str | None = None,
             creds: Creds | None = None, chunk: int = CHUNK,
             workers: int = WORKERS, progress: bool = True,
             overwrite: bool = False) -> Path:
    """Fetch an object to `dest` and return the local path.

    `dest` may be a directory, in which case the object's basename is used.
    An interrupted run leaves `dest.part` plus a state file; calling again with
    the same arguments picks up where it stopped.
    """
    info = stat(key, bucket=bucket, creds=creds)
    creds = creds or Creds.from_env(info.bucket)

    dest = Path(dest)
    if dest.is_dir() or str(dest).endswith(os.sep):
        dest = dest / Path(info.key).name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        if dest.stat().st_size == info.size:
            _log(f"[r2] already have {dest.name} ({_human(info.size)}), skipping")
            return dest
        raise R2Error(
            f"{dest} exists with the wrong size "
            f"({_human(dest.stat().st_size)} local vs {_human(info.size)} remote). "
            f"Delete it or pass overwrite=True."
        )

    part = dest.with_name(dest.name + PART_SUFFIX)
    state = _State(dest.with_name(dest.name + STATE_SUFFIX), info, chunk)
    resumed = part.exists() and state.load()
    if not resumed:
        # Either a fresh start or a stale part for a different object. Both mean
        # the existing bytes are unusable, and silently reusing them would
        # produce a corrupt file that only shows up as a decode error later.
        state.done.clear()
        if part.exists():
            part.unlink()

    nchunks = max(1, -(-info.size // chunk))
    todo = [i for i in range(nchunks) if i not in state.done]
    have = (nchunks - len(todo)) * chunk
    have = min(have, info.size)

    _require_space(dest.parent, info.size - have)

    if resumed and todo:
        _log(f"[r2] resuming: {len(state.done)}/{nchunks} chunks already local "
             f"({_human(have)} of {_human(info.size)})")
    else:
        _log(f"[r2] {info.key} -> {dest}")
        _log(f"[r2] {_human(info.size)} in {nchunks} chunks of {_human(chunk)}, "
             f"{workers} workers")

    fd = os.open(part, os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.ftruncate(fd, info.size)          # allocate once; keeps the file contiguous
        if todo:
            _run_chunks(fd, todo, info, creds, state, chunk=chunk,
                        workers=workers, progress=progress, already=have)
        state.flush()
    finally:
        os.close(fd)

    size = part.stat().st_size
    if size != info.size:
        raise R2Error(f"size mismatch after download: got {size}, expected {info.size}")

    part.replace(dest)
    state.path.unlink(missing_ok=True)
    _log(f"[r2] done -> {dest} ({_human(info.size)})")
    return dest


def _run_chunks(fd: int, todo: list[int], info: ObjectInfo, creds: Creds,
                state: _State, *, chunk: int, workers: int, progress: bool,
                already: int) -> None:
    cli = _client(creds, workers=workers)
    started = time.monotonic()
    counter = _Counter(already)

    def fetch(index: int) -> None:
        start = index * chunk
        end = min(start + chunk, info.size) - 1
        last: Exception | None = None
        for attempt in range(ATTEMPTS):
            offset = start
            try:
                body = cli.get_object(Bucket=info.bucket, Key=info.key,
                                      Range=f"bytes={start}-{end}")["Body"]
                # Stream into position rather than reading the chunk into memory
                # first: 8 workers x 64 MiB resident is 512 MiB of nothing, on a
                # machine that also has to hold video frames.
                try:
                    while True:
                        buf = body.read(1024 * 1024)
                        if not buf:
                            break
                        written = 0
                        while written < len(buf):
                            written += os.pwrite(fd, buf[written:], offset + written)
                        offset += len(buf)
                        counter.add(len(buf))
                finally:
                    body.close()
                if offset - start != end - start + 1:
                    raise R2Error(f"short read on chunk {index}: "
                                  f"{offset - start} of {end - start + 1} bytes")
                state.mark(index)
                return
            except Exception as exc:                 # network, 5xx, short read
                last = exc
                # Roll the progress counter back over the partial chunk. Those
                # bytes are on disk and correct, but the retry re-fetches the
                # whole range, so counting them twice would report a rate and an
                # ETA that the transfer is not achieving.
                counter.add(start - offset)
                if attempt + 1 < ATTEMPTS:
                    time.sleep(min(2 ** attempt, 15))
        raise R2Error(f"chunk {index} failed after {ATTEMPTS} attempts: "
                      f"{_explain(last, info.bucket, info.key)}")

    stop = threading.Event()
    reporter = None
    if progress:
        reporter = threading.Thread(
            target=_report, args=(counter, info.size, started, stop), daemon=True)
        reporter.start()

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, i): i for i in todo}
            try:
                for fut in as_completed(futures):
                    fut.result()                     # re-raise the first failure
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
    except KeyboardInterrupt:
        state.flush()
        _log("\n[r2] interrupted -- progress kept, run the same command to resume")
        raise
    finally:
        stop.set()
        if reporter is not None:
            reporter.join(timeout=1.0)
        if progress:
            _log("")


class _Counter:
    def __init__(self, initial: int = 0):
        self._n = initial
        self._lock = threading.Lock()

    def add(self, n: int) -> None:
        with self._lock:
            self._n += n

    def get(self) -> int:
        with self._lock:
            return self._n


def _report(counter: _Counter, total: int, started: float,
            stop: threading.Event) -> None:
    while not stop.wait(1.0):
        done = counter.get()
        secs = max(time.monotonic() - started, 1e-6)
        rate = done / secs
        eta = (total - done) / rate if rate > 0 else 0
        pct = 100.0 * done / total if total else 100.0
        print(f"\r[r2] {pct:5.1f}%  {_human(done)}/{_human(total)}  "
              f"{_human(rate)}/s  eta {_clock(eta)}   ",
              end="", file=sys.stderr, flush=True)


def _require_space(where: Path, need: int) -> None:
    """Refuse before starting rather than dying at 90 GB with ENOSPC.

    A 20% headroom on top, because the finished file and its .part briefly
    coexist only as a rename -- but the pipeline that follows writes an output
    of comparable size into the same place.
    """
    free = shutil.disk_usage(where).free
    if free < need:
        raise R2Error(
            f"not enough space in {where}: need {_human(need)}, "
            f"{_human(free)} free"
        )
    if free < need * 1.2:
        _log(f"[r2] WARNING: {_human(free)} free for a {_human(need)} download -- "
             f"the processed output will not fit next to it")


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"                            # pragma: no cover


def _clock(secs: float) -> str:
    secs = int(max(secs, 0))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# upload
#
# This lived in scripts/upload-r2.py, which reached into this module for
# `_client`, `_explain` and `_human`. That was fine while a person at a terminal
# was the only caller; it stopped being fine when the pod server needed to push
# its own output, because a second caller importing another module's privates is
# how a refactor here turns into a broken worker there. Same code, public door.
# --------------------------------------------------------------------------- #

MIN_PART = 5 * 1024 * 1024        # S3/R2 floor for every part but the last
UPLOAD_ATTEMPTS = 5

CONTENT_TYPE = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska",
    ".webm": "video/webm", ".avi": "video/x-msvideo", ".m4v": "video/x-m4v",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".json": "application/json", ".txt": "text/plain", ".log": "text/plain",
}


def _plan_parts(size: int, part: int) -> list[tuple[int, int, int]]:
    """(part_number, offset, length) for every part. Part numbers start at 1."""
    if part < MIN_PART:
        raise R2Error(f"part size must be at least {_human(MIN_PART)}")
    n = max(1, -(-size // part))
    return [(i + 1, i * part, min(part, size - i * part)) for i in range(n)]


def _find_open_upload(cli, bucket: str, key: str) -> str | None:
    """An earlier run's multipart upload for this exact key, if one is open.

    R2 keeps these until completed or aborted, which is what makes resume free.
    Newest last: if several are somehow open, the most recent attempt is the one
    whose parts match the file about to be sent.
    """
    found: list[tuple[object, str]] = []
    token: dict = {}
    while True:
        resp = cli.list_multipart_uploads(Bucket=bucket, Prefix=key, **token)
        for u in resp.get("Uploads", []):
            if u["Key"] == key:
                found.append((u["Initiated"], u["UploadId"]))
        if not resp.get("IsTruncated"):
            break
        token = {"KeyMarker": resp["NextKeyMarker"],
                 "UploadIdMarker": resp["NextUploadIdMarker"]}
    if not found:
        return None
    found.sort(key=lambda t: t[0])
    return found[-1][1]


def _reusable_parts(cli, bucket: str, key: str, upload_id: str,
                    plan: list[tuple[int, int, int]]) -> dict[int, str]:
    """Parts already on R2 worth keeping, as {part_number: etag}.

    Length is the check. A part whose length does not match what this run intends
    to send for that number came from a different file or a different part size,
    and completing an upload that mixes the two produces an object of exactly the
    right length that is silently corrupt -- the one outcome worse than
    re-uploading.
    """
    want = {n: length for n, _, length in plan}
    keep: dict[int, str] = {}
    marker: dict = {}
    while True:
        resp = cli.list_parts(Bucket=bucket, Key=key, UploadId=upload_id, **marker)
        for p in resp.get("Parts", []):
            n = int(p["PartNumber"])
            if want.get(n) == int(p["Size"]):
                keep[n] = p["ETag"]
        if not resp.get("IsTruncated"):
            break
        marker = {"PartNumberMarker": resp["NextPartNumberMarker"]}
    return keep


def _send_part(cli, bucket: str, key: str, upload_id: str, path: Path,
               number: int, offset: int, length: int,
               counter: _Counter) -> tuple[int, str]:
    """One UploadPart, retried. Read with pread so workers share no file offset."""
    last: Exception | None = None
    for attempt in range(UPLOAD_ATTEMPTS):
        try:
            with open(path, "rb") as f:
                body = os.pread(f.fileno(), length, offset)
            if len(body) != length:
                raise R2Error(
                    f"{path.name} shrank while uploading: wanted {length} bytes at "
                    f"{offset}, got {len(body)}")
            resp = cli.upload_part(Bucket=bucket, Key=key, UploadId=upload_id,
                                   PartNumber=number, Body=body)
            counter.add(length)
            return number, resp["ETag"]
        except Exception as exc:                        # noqa: BLE001 -- retry anything
            last = exc
            if attempt < UPLOAD_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
    raise R2Error(f"part {number} failed after {UPLOAD_ATTEMPTS} attempts: {last}")


def upload(path: Path | str, key: str, *, bucket: str | None = None,
           creds: Creds | None = None, part: int = CHUNK,
           workers: int = WORKERS, progress: bool = True) -> str:
    """Send a local file to R2 and return the key it landed on.

    Resumable in the same sense as `download`: an interrupted run leaves its parts
    on R2, and calling again with the same arguments reuses the ones whose lengths
    still match. That is why an interruption does **not** abort the upload -- the
    parts that landed are the only reason a retry is cheap.

    The cost of that choice is that abandoned parts are billed while appearing in
    no listing, so something has to clean up eventually: `abort_uploads`.
    """
    path = Path(path)
    if not path.is_file():
        raise R2Error(f"{path} is not a file")
    bucket, key = parse_uri(key, default_bucket=bucket) if "://" in key else (bucket, key)
    creds = creds or Creds.from_env(bucket)
    bucket = bucket or creds.bucket
    cli = _client(creds, workers=workers)

    size = path.stat().st_size
    ctype = CONTENT_TYPE.get(path.suffix.lower(), "application/octet-stream")

    # One request is the whole story below the part size. A multipart upload of a
    # single part adds two round trips and a resumable state nothing needs.
    if size <= part:
        try:
            with open(path, "rb") as f:
                cli.put_object(Bucket=bucket, Key=key, Body=f, ContentType=ctype)
        except Exception as exc:                        # noqa: BLE001
            raise R2Error(_explain(exc, bucket, key)) from exc
        # Not gated on `progress`. That flag is about the live percentage line, not about
        # whether anything is said at all -- `download` has always logged its start and
        # finish unconditionally, and the asymmetry meant a caller passing progress=False
        # (the server does) saw the fetch and heard nothing about the publish. "Did the
        # result actually land?" is not a progress detail.
        _log(f"[r2] uploaded {path.name} ({_human(size)}) in one request -> {key}")
        return key

    plan = _plan_parts(size, part)
    # Said before the work, symmetrically with `download`: a multipart upload of tens of
    # gigabytes is long enough that "it started" and "it finished" are different facts.
    _log(f"[r2] {path.name} ({_human(size)}) -> {key}")
    _log(f"[r2] {len(plan)} parts of {_human(part)}, {workers} workers")
    try:
        upload_id = _find_open_upload(cli, bucket, key)
        keep: dict[int, str] = {}
        if upload_id:
            keep = _reusable_parts(cli, bucket, key, upload_id, plan)
            if keep and progress:
                done = sum(length for n, _, length in plan if n in keep)
                _log(f"[r2] resuming {path.name}: {len(keep)}/{len(plan)} parts "
                     f"already on R2 ({_human(done)})")
        else:
            upload_id = cli.create_multipart_upload(
                Bucket=bucket, Key=key, ContentType=ctype)["UploadId"]
    except R2Error:
        raise
    except Exception as exc:                            # noqa: BLE001
        raise R2Error(_explain(exc, bucket, key)) from exc

    todo = [(n, off, length) for n, off, length in plan if n not in keep]
    total = sum(length for _, _, length in todo)
    counter = _Counter()
    etags = dict(keep)

    stop = threading.Event()
    reporter = None
    if progress and total:
        started = time.monotonic()
        reporter = threading.Thread(target=_report,
                                    args=(counter, total, started, stop), daemon=True)
        reporter.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_send_part, cli, bucket, key, upload_id, path,
                                   n, off, length, counter)
                       for n, off, length in todo]
            for fut in as_completed(futures):
                n, etag = fut.result()
                etags[n] = etag
    except BaseException:
        # Not aborted, including on Ctrl-C: see the docstring. Say where things
        # stand so the choice to resume or drop is an informed one.
        stop.set()
        if reporter is not None:
            reporter.join(timeout=1.0)
        if progress:
            _log("")
        _log(f"[r2] interrupted -- {len(etags)}/{len(plan)} parts of {path.name} are "
             f"on R2. Re-run to finish, or drop them with abort_uploads({key!r}).")
        raise
    finally:
        stop.set()
        if reporter is not None:
            reporter.join(timeout=1.0)
        if progress and total:
            _log("")

    try:
        cli.complete_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload_id,
            MultipartUpload={"Parts": [{"PartNumber": n, "ETag": etags[n]}
                                       for n in sorted(etags)]})
    except Exception as exc:                            # noqa: BLE001
        raise R2Error(_explain(exc, bucket, key)) from exc

    _log(f"[r2] uploaded {path.name} ({_human(size)}) in {len(plan)} parts -> {key}")
    return key


def abort_uploads(key: str, *, bucket: str | None = None,
                  creds: Creds | None = None) -> int:
    """Drop every open multipart upload for `key`. Returns how many were dropped.

    Worth running: the parts of an incomplete upload are stored and billed while
    appearing in no object listing, so nothing draws attention to them.
    """
    bucket, key = parse_uri(key, default_bucket=bucket) if "://" in key else (bucket, key)
    creds = creds or Creds.from_env(bucket)
    bucket = bucket or creds.bucket
    cli = _client(creds, workers=1)
    n = 0
    try:
        while (upload_id := _find_open_upload(cli, bucket, key)):
            cli.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
            n += 1
    except Exception as exc:                            # noqa: BLE001
        raise R2Error(_explain(exc, bucket, key)) from exc
    return n


# --------------------------------------------------------------------------- #
# listing -- for finding a key when you only half remember it
# --------------------------------------------------------------------------- #

def ls(prefix: str = "", *, bucket: str | None = None, creds: Creds | None = None,
       limit: int = 200) -> list[tuple[str, int]]:
    bucket, prefix = parse_uri(prefix, default_bucket=bucket) if "://" in prefix \
        else (bucket, prefix)
    creds = creds or Creds.from_env(bucket)
    bucket = bucket or creds.bucket
    cli = _client(creds, workers=1)
    out: list[tuple[str, int]] = []
    try:
        for page in cli.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                out.append((obj["Key"], int(obj["Size"])))
                if len(out) >= limit:
                    return out
    except Exception as exc:
        raise R2Error(_explain(exc, bucket, prefix)) from exc
    return out
