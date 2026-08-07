#!/usr/bin/env python3
"""Push finished videos back up to Cloudflare R2.

The mirror image of `wmrm pull`, and it has the same problem for the same reason:
a 22-100 GB body over one HTTP request is a bet that nothing drops for an hour,
and `boto3.upload_file` loses that bet expensively -- it starts a multipart
upload, dies at the 80th gigabyte, and leaves nothing behind that a second
attempt can use. So the parts are ours to manage: a fixed part size, a worker
pool of UploadPart calls, and resume driven by R2's own ListParts rather than a
sidecar file. The server already knows which parts landed; asking it is both
simpler and impossible to get out of sync with.

Credentials come from the environment, exactly as `wmrm pull` reads them:

    export R2_ACCOUNT_ID=...          # or R2_ENDPOINT=https://...
    export R2_ACCESS_KEY_ID=...
    export R2_SECRET_ACCESS_KEY=...
    export R2_BUCKET=remove-watermark

Usage:

    scripts/upload-r2.py outbox/                        # every video in outbox/
    scripts/upload-r2.py outbox/MOGI-125-clean.mp4      # one file
    scripts/upload-r2.py out.mp4 --key final/out.mp4    # an exact key
    scripts/upload-r2.py outbox/ --prefix cleaned/2026-08/
    scripts/upload-r2.py outbox/ --dry-run              # show the keys, send nothing
    scripts/upload-r2.py out.mp4 --abort                # drop a stuck multipart upload

An interrupted run is resumed by re-running the same command: the parts already
on R2 are kept and only the rest go up. That is also why Ctrl-C leaves the
multipart upload open instead of cleaning it up -- open is what makes resume
possible. Incomplete uploads do cost storage, so `--abort` exists for the ones
you decide not to finish.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from wmrm.r2 import (  # noqa: E402  -- after the path fix above
    Creds,
    R2Error,
    _client,
    _explain,
    _human,
    parse_uri,
)

PART = 64 * 1024 * 1024           # 64 MiB -> ~1600 parts at 100 GB, well under 10k
WORKERS = 8
ATTEMPTS = 5                      # per part, with backoff
MIN_PART = 5 * 1024 * 1024        # S3/R2 floor for every part but the last

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
CONTENT_TYPE = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".mkv": "video/x-matroska", ".webm": "video/webm", ".avi": "video/x-msvideo",
    ".json": "application/json", ".png": "image/png", ".txt": "text/plain",
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"\033[33m{msg}\033[0m", flush=True)


# --------------------------------------------------------------------------- #
# what to send, and under what key
# --------------------------------------------------------------------------- #

def collect(paths: list[str], *, all_files: bool) -> list[Path]:
    """Expand the arguments into files. Directories contribute their videos.

    A file named explicitly is always sent -- you asked for it by name, so its
    extension is not this script's business. Inside a directory the filter does
    apply, because outbox/ also holds preview PNGs and per-file presets that
    nobody meant to publish.
    """
    out: list[Path] = []
    for arg in paths:
        p = Path(arg)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if not f.is_file() or f.name.startswith("."):
                    continue
                if all_files or f.suffix.lower() in VIDEO_EXT:
                    out.append(f)
        elif p.is_file():
            out.append(p)
        else:
            raise R2Error(f"no such file or directory: {arg}")
    return out


def key_for(path: Path, *, prefix: str, explicit: str | None) -> str:
    if explicit:
        return explicit.lstrip("/")
    prefix = prefix.strip("/")
    return f"{prefix}/{path.name}" if prefix else path.name


# --------------------------------------------------------------------------- #
# progress
# --------------------------------------------------------------------------- #

class Progress:
    """Bytes-done counter shared by the worker pool, printed on one line."""

    def __init__(self, total: int, *, enabled: bool) -> None:
        self.total = total
        self.enabled = enabled and sys.stdout.isatty()
        self.done = 0
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._last = 0.0

    def add(self, n: int) -> None:
        with self._lock:
            self.done += n
            now = time.monotonic()
            if self.enabled and now - self._last >= 0.5:
                self._last = now
                self._draw(now)

    def _draw(self, now: float) -> None:
        elapsed = max(now - self.started, 1e-6)
        rate = self.done / elapsed
        pct = 100.0 * self.done / self.total if self.total else 100.0
        eta = (self.total - self.done) / rate if rate > 0 else 0
        sys.stdout.write(
            f"\r    {pct:5.1f}%  {_human(self.done)} / {_human(self.total)}"
            f"  {_human(rate)}/s  eta {int(eta // 60)}m{int(eta % 60):02d}s   "
        )
        sys.stdout.flush()

    def finish(self) -> None:
        if self.enabled:
            sys.stdout.write("\r" + " " * 78 + "\r")
            sys.stdout.flush()


# --------------------------------------------------------------------------- #
# multipart
# --------------------------------------------------------------------------- #

def _plan(size: int, part: int) -> list[tuple[int, int, int]]:
    """(part_number, offset, length) for every part. Part numbers start at 1."""
    if part < MIN_PART:
        raise R2Error(f"part size must be at least {_human(MIN_PART)}")
    n = max(1, -(-size // part))
    return [(i + 1, i * part, min(part, size - i * part)) for i in range(n)]


def _find_upload(cli, bucket: str, key: str) -> str | None:
    """An earlier run's multipart upload for this exact key, if one is open.

    R2 keeps these until they are completed or aborted, which is what makes
    resume free. Newest first: if several are somehow open, the last attempt is
    the one whose parts match the file we are about to send.
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


def _existing_parts(cli, bucket: str, key: str, upload_id: str,
                    plan: list[tuple[int, int, int]]) -> dict[int, str]:
    """Parts already on R2 that we can keep, as {part_number: etag}.

    Size is the check. A part whose length does not match what this run intends
    to send for that number came from a different file or a different part size,
    and completing an upload that mixes the two produces an object of exactly
    the right length that is silently corrupt -- the one outcome worse than
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
               number: int, offset: int, length: int, prog: Progress) -> tuple[int, str]:
    """One UploadPart, retried. Read with pread so workers share no file offset."""
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            with open(path, "rb") as f:
                body = os.pread(f.fileno(), length, offset)
            if len(body) != length:
                raise R2Error(
                    f"{path.name} shrank while uploading: wanted {length} bytes at "
                    f"{offset}, got {len(body)}")
            resp = cli.upload_part(Bucket=bucket, Key=key, UploadId=upload_id,
                                   PartNumber=number, Body=body)
            prog.add(length)
            return number, resp["ETag"]
        except Exception as exc:                       # noqa: BLE001 -- retry anything
            last = exc
            if attempt < ATTEMPTS - 1:
                time.sleep(2 ** attempt)
    raise R2Error(f"part {number} failed after {ATTEMPTS} attempts: {last}")


def upload(cli, bucket: str, key: str, path: Path, *, part: int, workers: int,
           progress: bool) -> None:
    size = path.stat().st_size
    ctype = CONTENT_TYPE.get(path.suffix.lower(), "application/octet-stream")

    # One request is the whole story below the part size, and a multipart upload
    # of a single part would only add two round trips and a resumable state
    # nothing needs.
    if size <= part:
        prog = Progress(size, enabled=progress)
        with open(path, "rb") as f:
            cli.put_object(Bucket=bucket, Key=key, Body=f, ContentType=ctype)
        prog.add(size)
        prog.finish()
        log(f"    uploaded {_human(size)} in one request")
        return

    plan = _plan(size, part)
    upload_id = _find_upload(cli, bucket, key)
    keep: dict[int, str] = {}
    if upload_id:
        keep = _existing_parts(cli, bucket, key, upload_id, plan)
        if keep:
            log(f"    resuming: {len(keep)}/{len(plan)} parts already on R2 "
                f"({_human(sum(l for n, _, l in plan if n in keep))})")
        else:
            log("    found an open upload with no usable parts -- reusing its id")
    else:
        upload_id = cli.create_multipart_upload(
            Bucket=bucket, Key=key, ContentType=ctype)["UploadId"]

    todo = [(n, off, length) for n, off, length in plan if n not in keep]
    prog = Progress(sum(length for _, _, length in todo), enabled=progress)
    etags = dict(keep)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_send_part, cli, bucket, key, upload_id, path,
                                   n, off, length, prog)
                       for n, off, length in todo]
            for fut in as_completed(futures):
                n, etag = fut.result()
                etags[n] = etag
    except BaseException:
        # Deliberately not aborted, including on Ctrl-C: the parts that landed
        # are the only reason a retry is cheap. Abandoning them here would make
        # every interruption cost the whole transfer again.
        prog.finish()
        warn(f"    interrupted -- {len(etags)}/{len(plan)} parts are on R2.")
        warn("    Re-run the same command to finish, or drop it with:")
        warn(f"      {Path(sys.argv[0]).name} {path} --key {key} --abort")
        raise
    prog.finish()

    cli.complete_multipart_upload(
        Bucket=bucket, Key=key, UploadId=upload_id,
        MultipartUpload={"Parts": [{"PartNumber": n, "ETag": etags[n]}
                                   for n in sorted(etags)]})
    log(f"    uploaded {_human(size)} in {len(plan)} parts")


def abort(cli, bucket: str, key: str) -> int:
    n = 0
    while (upload_id := _find_upload(cli, bucket, key)):
        cli.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        n += 1
    return n


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Upload files to a Cloudflare R2 bucket (multipart, resumable).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Credentials: R2_ACCOUNT_ID (or R2_ENDPOINT), R2_ACCESS_KEY_ID,\n"
               "R2_SECRET_ACCESS_KEY, R2_BUCKET -- same as `wmrm pull`.")
    ap.add_argument("paths", nargs="+",
                    help="files, or directories whose videos should go up")
    ap.add_argument("--key", default=None,
                    help="exact object key; only valid for a single file")
    ap.add_argument("--prefix", default=os.environ.get("R2_UPLOAD_PREFIX", "cleaned"),
                    help="key prefix for the uploaded names (default: %(default)s)")
    ap.add_argument("--bucket", default=None,
                    help="override R2_BUCKET; also accepts r2://bucket in --key")
    ap.add_argument("--all", action="store_true", dest="all_files",
                    help="inside a directory, take every file, not just videos")
    ap.add_argument("--force", action="store_true",
                    help="re-upload even if a same-size object is already there")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help="parallel parts (default: %(default)s)")
    ap.add_argument("--part-mib", type=int, default=PART // 1024 // 1024,
                    help="part size in MiB (default: %(default)s)")
    ap.add_argument("--abort", action="store_true",
                    help="abort open multipart uploads for these keys, upload nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be sent, and where")
    ap.add_argument("--quiet", action="store_true", help="no progress line")
    args = ap.parse_args(argv)

    try:
        files = collect(args.paths, all_files=args.all_files)
        if not files:
            log("nothing to upload")
            return 0
        if args.key and len(files) > 1:
            raise R2Error(f"--key names one object but {len(files)} files matched")

        bucket_from_key, key_override = (
            parse_uri(args.key, default_bucket=args.bucket) if args.key
            else (args.bucket, None))
        creds = Creds.from_env(bucket_from_key)
        bucket = bucket_from_key or creds.bucket

        jobs = [(f, key_for(f, prefix=args.prefix, explicit=key_override))
                for f in files]

        if args.dry_run:
            total = sum(f.stat().st_size for f, _ in jobs)
            log(f"would upload {len(jobs)} file(s), {_human(total)} to "
                f"r2://{bucket}/")
            for f, key in jobs:
                log(f"    {f}  ->  r2://{bucket}/{key}  ({_human(f.stat().st_size)})")
            return 0

        cli = _client(creds, workers=args.workers)

        if args.abort:
            for _, key in jobs:
                n = abort(cli, bucket, key)
                log(f"{key}: aborted {n} open upload(s)")
            return 0

        total = sum(f.stat().st_size for f, _ in jobs)
        log(f"==> {len(jobs)} file(s), {_human(total)} -> r2://{bucket}/")

        sent = skipped = failed = 0
        failures: list[str] = []
        for f, key in jobs:
            size = f.stat().st_size
            log(f"\n{f.name} -> r2://{bucket}/{key}  ({_human(size)})")

            if not args.force:
                # Size only. R2's ETag for a multipart object is not the md5 of
                # the body, so there is no cheap content check to make here --
                # --force is the honest way to say "send it anyway".
                try:
                    head = cli.head_object(Bucket=bucket, Key=key)
                    if int(head["ContentLength"]) == size:
                        log("    skip: already there, same size (--force to redo)")
                        skipped += 1
                        continue
                    warn(f"    exists with a different size "
                         f"({_human(int(head['ContentLength']))}) -- overwriting")
                except Exception as exc:                # noqa: BLE001
                    resp = getattr(exc, "response", {})
                    code = str(resp.get("Error", {}).get("Code", "")) \
                        if isinstance(resp, dict) else ""
                    if code not in ("404", "NoSuchKey", "NotFound"):
                        raise R2Error(_explain(exc, bucket, key)) from exc

            try:
                upload(cli, bucket, key, f, part=args.part_mib * 1024 * 1024,
                       workers=args.workers, progress=not args.quiet)
                sent += 1
            except KeyboardInterrupt:
                raise
            except Exception as exc:                    # noqa: BLE001
                warn(f"    FAILED: {_explain(exc, bucket, key)}")
                failed += 1
                failures.append(f.name)

        log(f"\n==> done: {sent} uploaded, {skipped} skipped, {failed} failed")
        if failures:
            warn(f"failed: {', '.join(failures)}")
        return 1 if failed else 0

    except R2Error as exc:
        print(f"\n\033[31merror:\033[0m {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
