#!/usr/bin/env python3
"""Push finished videos back up to Cloudflare R2.

The mirror image of `wmrm pull`, and it had the same problem for the same reason:
a 22-100 GB body over one HTTP request is a bet that nothing drops for an hour,
and `boto3.upload_file` loses that bet expensively -- it starts a multipart
upload, dies at the 80th gigabyte, and leaves nothing behind that a second
attempt can use.

The multipart machinery that solves that now lives in `wmrm.r2.upload`, not here.
It moved because the pod server needs to push its own output, and two callers of
logic that reached into `wmrm.r2`'s private names (`_client`, `_explain`) is how a
tidy-up in one place quietly breaks the other. This file is now what it should
always have been: argument handling, key naming, and a summary.

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
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from wmrm.r2 import (  # noqa: E402  -- after the path fix above
    CHUNK,
    WORKERS,
    Creds,
    R2Error,
    abort_uploads,
    parse_uri,
    stat,
    upload,
)

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"\033[33m{msg}\033[0m", flush=True)


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"                                # pragma: no cover


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


def _already_there(key: str, size: int, *, bucket: str, creds: Creds) -> bool:
    """Whether an object of exactly this length is already at this key.

    Size only. R2's ETag for a multipart object is not the md5 of the body, so
    there is no cheap content check to make here -- `--force` is the honest way
    to say "send it anyway".
    """
    try:
        info = stat(key, bucket=bucket, creds=creds)
    except R2Error:
        return False              # missing, or unreadable; either way, try sending
    if info.size == size:
        return True
    warn(f"    exists with a different size ({_human(info.size)}) -- overwriting")
    return False


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
    ap.add_argument("--part-mib", type=int, default=CHUNK // 1024 // 1024,
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

        if args.abort:
            for _, key in jobs:
                n = abort_uploads(key, bucket=bucket, creds=creds)
                log(f"{key}: aborted {n} open upload(s)")
            return 0

        total = sum(f.stat().st_size for f, _ in jobs)
        log(f"==> {len(jobs)} file(s), {_human(total)} -> r2://{bucket}/")

        sent = skipped = failed = 0
        failures: list[str] = []
        for f, key in jobs:
            size = f.stat().st_size
            log(f"\n{f.name} -> r2://{bucket}/{key}  ({_human(size)})")

            if not args.force and _already_there(key, size, bucket=bucket, creds=creds):
                log("    skip: already there, same size (--force to redo)")
                skipped += 1
                continue

            try:
                upload(f, key, bucket=bucket, creds=creds,
                       part=args.part_mib * 1024 * 1024,
                       workers=args.workers, progress=not args.quiet)
                sent += 1
            except KeyboardInterrupt:
                raise
            except R2Error as exc:
                warn(f"    FAILED: {exc}")
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
