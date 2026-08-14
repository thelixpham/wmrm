#!/usr/bin/env python
"""`wmrm.r2.upload` must be resumable, and must never complete a mixed object.

Run against a real S3 server (moto, on a real socket) rather than a mocked boto3
client, for the same reason `test_r2.py` does: the things that break here --
UploadPart numbering, positional reads, ListParts-driven resume, the short final part --
are exactly the things a mock answers correctly by fiat.

Four properties:

1. **Correctness.** What lands on R2 is byte-identical to the local file, and the
   fixture is deliberately not a whole multiple of the part size so the short final
   part is exercised.
2. **Resume.** An upload interrupted after some parts landed sends only the rest, and
   the completed object is still byte-identical. This is the property that costs a
   100 GB re-upload when it breaks.
3. **A part of the wrong length is not reused.** If the local file changed, parts from
   the previous attempt are the wrong bytes. Completing an upload that mixes the two
   produces an object of exactly the right length that is silently corrupt -- the one
   outcome worse than re-uploading.
4. **`abort_uploads` actually drops them.** Incomplete uploads are billed while
   appearing in no listing, so nothing else would draw attention to them.

    uv pip install 'moto[server]' boto3
    python tests/test_r2_upload.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PORT = 5601
ENDPOINT = f"http://127.0.0.1:{PORT}"
BUCKET = "remove-watermark"
KEY = "output/job_test/MOGI-125-clean.mp4"
PART = 5 * 1024 * 1024                  # the S3 floor; smallest legal multipart

os.environ.update(
    R2_ENDPOINT=ENDPOINT,
    R2_ACCESS_KEY_ID="test",
    R2_SECRET_ACCESS_KEY="test",
    R2_BUCKET=BUCKET,
)

failures: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name} {extra}".rstrip())
    if not cond:
        failures.append(name)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    try:
        import boto3
        from moto.server import ThreadedMotoServer
    except ImportError as exc:
        print(f"SKIP: {exc}. Install with: uv pip install 'moto[server]' boto3")
        return 0

    from wmrm.r2 import R2Error, abort_uploads, download, stat, upload

    srv = ThreadedMotoServer(port=PORT, verbose=False)
    srv.start()
    try:
        cli = boto3.client("s3", endpoint_url=ENDPOINT,
                           aws_access_key_id="test", aws_secret_access_key="test",
                           # "auto" is what R2 wants and what wmrm.r2 sends; moto
                           # rejects it on CreateBucket only, so this fixture client
                           # differs from the client under test here alone.
                           region_name="us-east-1")
        cli.create_bucket(Bucket=BUCKET)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            # Not a whole multiple of PART: the last part is short on purpose.
            body = os.urandom(PART * 2 + 1234)
            src = tmp / "clean.mp4"
            src.write_bytes(body)

            # -- 1. correctness ------------------------------------------------ #
            key = upload(src, KEY, progress=False, part=PART, workers=4)
            check("upload returns the key", key == KEY, key)
            got = cli.get_object(Bucket=BUCKET, Key=KEY)["Body"].read()
            check("the uploaded object is byte-identical",
                  sha(got) == sha(body), f"{len(got)} vs {len(body)} bytes")
            check("stat reports the right size",
                  stat(KEY).size == len(body), str(stat(KEY).size))

            # A single-request upload below the part size must also work, and must not
            # leave a multipart upload behind.
            small_key = "output/job_test/small.mp4"
            small = tmp / "small.mp4"
            small.write_bytes(os.urandom(1024))
            upload(small, small_key, progress=False, part=PART)
            check("a file below the part size uploads in one request",
                  cli.get_object(Bucket=BUCKET, Key=small_key)["Body"].read()
                  == small.read_bytes())
            check("a single-request upload opens no multipart upload",
                  not cli.list_multipart_uploads(
                      Bucket=BUCKET, Prefix=small_key).get("Uploads"))

            # -- 2. resume ----------------------------------------------------- #
            # Simulate an interrupted attempt: open an upload and land part 1 only.
            rkey = "output/job_test/resume.mp4"
            rbody = os.urandom(PART * 2 + 999)
            rsrc = tmp / "resume.mp4"
            rsrc.write_bytes(rbody)

            uid = cli.create_multipart_upload(Bucket=BUCKET, Key=rkey)["UploadId"]
            cli.upload_part(Bucket=BUCKET, Key=rkey, UploadId=uid, PartNumber=1,
                            Body=rbody[:PART])
            before = cli.list_parts(Bucket=BUCKET, Key=rkey, UploadId=uid)["Parts"]
            check("the interrupted attempt left one part", len(before) == 1)

            upload(rsrc, rkey, progress=False, part=PART, workers=4)
            got = cli.get_object(Bucket=BUCKET, Key=rkey)["Body"].read()
            check("a resumed upload is byte-identical", sha(got) == sha(rbody),
                  f"{len(got)} vs {len(rbody)}")
            check("resume reused the open upload id rather than starting a new one",
                  not cli.list_multipart_uploads(
                      Bucket=BUCKET, Prefix=rkey).get("Uploads"))

            # -- 3. a wrong-length part is not reused -------------------------- #
            wkey = "output/job_test/changed.mp4"
            wbody = os.urandom(PART * 2 + 500)
            wsrc = tmp / "changed.mp4"
            wsrc.write_bytes(wbody)

            uid = cli.create_multipart_upload(Bucket=BUCKET, Key=wkey)["UploadId"]
            # Part 1 from a *different* file, and a different length: exactly the part
            # that must be discarded rather than kept.
            cli.upload_part(Bucket=BUCKET, Key=wkey, UploadId=uid, PartNumber=1,
                            Body=os.urandom(PART + 7))
            upload(wsrc, wkey, progress=False, part=PART, workers=4)
            got = cli.get_object(Bucket=BUCKET, Key=wkey)["Body"].read()
            check("a part of the wrong length is discarded, not spliced in",
                  sha(got) == sha(wbody), f"{len(got)} vs {len(wbody)}")

            # -- 4. abort ------------------------------------------------------ #
            akey = "output/job_test/abandoned.mp4"
            cli.create_multipart_upload(Bucket=BUCKET, Key=akey)
            cli.create_multipart_upload(Bucket=BUCKET, Key=akey)
            n = abort_uploads(akey)
            check("abort_uploads drops every open upload", n == 2, str(n))
            check("nothing is left open afterwards",
                  not cli.list_multipart_uploads(
                      Bucket=BUCKET, Prefix=akey).get("Uploads"))
            check("aborting when there is nothing open returns 0",
                  abort_uploads(akey) == 0)

            # -- round trip through the real downloader ------------------------ #
            back = download(KEY, tmp / "roundtrip.mp4", progress=False, workers=4)
            check("upload -> download round-trips byte-identically",
                  sha(back.read_bytes()) == sha(body))

            # -- error surface -------------------------------------------------- #
            try:
                upload(tmp / "nope.mp4", "output/x.mp4", progress=False)
                check("uploading a missing file raises R2Error", False, "no raise")
            except R2Error:
                check("uploading a missing file raises R2Error", True)

            try:
                upload(src, KEY, progress=False, part=1024)
                check("a part size below the S3 floor raises", False, "no raise")
            except R2Error as exc:
                check("a part size below the S3 floor raises", "at least" in str(exc))
    finally:
        srv.stop()

    print()
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        return 1
    print("all pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
