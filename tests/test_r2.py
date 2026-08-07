"""`wmrm pull` must be resumable, and must never resume onto the wrong bytes.

The whole reason this downloader exists instead of `boto3.download_file` is
resume, so resume is what these checks pin. They run against a real S3 server
(moto, on a real socket) rather than a mocked boto3 client, because the things
that break here -- ranged GETs, positional writes, short final chunks -- are
exactly the things a mock would answer correctly by fiat.

Four properties:

1. **Correctness.** A downloaded file is byte-identical to what was uploaded.
   The fixture is deliberately not a round number of chunks, so the short final
   range is exercised; a fixture sized to a whole multiple would hide an
   off-by-one there and still pass.
2. **Resume.** Given a part file and a state listing chunks 0-2, the run fetches
   only the rest and still ends byte-identical. This is the property that costs
   a 100 GB re-download when it breaks.
3. **Stale state is discarded.** If the object changed on R2, the local bytes
   are no longer a prefix of it. Trusting the state file there would splice old
   and new content into a file that is the right length, opens, and is wrong --
   the failure mode worth an explicit test.
4. **A missing key says so**, rather than surfacing botocore's bare "404".

    uv pip install 'moto[server]' boto3
    python tests/test_r2.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PORT = 5599
ENDPOINT = f"http://127.0.0.1:{PORT}"
BUCKET = "remove-watermark"
KEY = "uploads/3d809a59-3e5c-4977-9dd6-bbc15b4f58d6/MOGI-125.mp4"
CHUNK = 512 * 1024               # small, so the pool genuinely runs concurrently

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

    from wmrm.r2 import R2Error, download, ls, parse_uri, stat

    srv = ThreadedMotoServer(port=PORT, verbose=False)
    srv.start()
    try:
        cli = boto3.client("s3", endpoint_url=ENDPOINT,
                           aws_access_key_id="test", aws_secret_access_key="test",
                           # "auto" is what R2 wants and what wmrm.r2 sends; moto
                           # rejects it on CreateBucket only, so the fixture client
                           # differs from the client under test here alone.
                           region_name="us-east-1")
        cli.create_bucket(Bucket=BUCKET)
        blob = os.urandom(10 * CHUNK + 12345)          # 10 full chunks + a stub
        cli.put_object(Bucket=BUCKET, Key=KEY, Body=blob)
        want = sha(blob)

        check("parse_uri r2://", parse_uri("r2://b/k/x.mp4") == ("b", "k/x.mp4"))
        check("parse_uri bare",
              parse_uri("k/x.mp4", default_bucket="b") == ("b", "k/x.mp4"))

        info = stat(KEY)
        check("stat size", info.size == len(blob), str(info.size))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            p = download(KEY, out, chunk=CHUNK, workers=4, progress=False)
            check("names the file after the key", p.name == "MOGI-125.mp4", p.name)
            check("bytes match", sha(p.read_bytes()) == want)
            check("state file cleaned up",
                  not (out / (p.name + ".wmrm-dl.json")).exists())
            check("part file cleaned up", not (out / (p.name + ".part")).exists())

            again = download(KEY, out, chunk=CHUNK, workers=4, progress=False)
            check("complete file is left alone", again.read_bytes() == blob)

            # --- resume, from the state an interrupted run would have left ---
            p.unlink()
            part = out / (p.name + ".part")
            with open(part, "wb") as f:
                f.truncate(len(blob))
                f.write(blob[: 3 * CHUNK])
            (out / (p.name + ".wmrm-dl.json")).write_text(json.dumps(
                {"bucket": BUCKET, "key": KEY, "size": len(blob),
                 "etag": info.etag, "chunk": CHUNK, "done": [0, 1, 2]}))
            check("resume completes the file",
                  sha(download(KEY, out, chunk=CHUNK, workers=4,
                               progress=False).read_bytes()) == want)

            # --- the object changed: the part file must not be trusted ---
            p.unlink()
            blob2 = os.urandom(2 * CHUNK + 77)
            cli.put_object(Bucket=BUCKET, Key=KEY, Body=blob2)
            with open(part, "wb") as f:
                f.write(b"\x00" * 1024)
            (out / (p.name + ".wmrm-dl.json")).write_text(json.dumps(
                {"bucket": BUCKET, "key": KEY, "size": len(blob),
                 "etag": info.etag, "chunk": CHUNK, "done": [0, 1, 2]}))
            check("stale part discarded, not spliced",
                  sha(download(KEY, out, chunk=CHUNK, workers=4,
                               progress=False).read_bytes()) == sha(blob2))

        check("ls returns the prefix", ls("uploads/") == [(KEY, len(blob2))])

        try:
            stat("uploads/nope.mp4")
            check("missing key raises R2Error", False)
        except R2Error as exc:
            check("missing key raises R2Error", "not found" in str(exc))
    finally:
        srv.stop()

    print(f"\n{len(failures)} failed" if failures else "\nall pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
