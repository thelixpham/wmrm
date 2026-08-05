"""Video in, video out.

Frames stream through ffmpeg pipes as raw bgr24 -- no PNG dumps, no temp frame
directories. Two of the reference repos write every frame to disk as a PNG and
reassemble; that is 10-100x slower and needs cleanup logic.

The output is always written to a temp file and moved into place on success, so
an interrupted run cannot leave a truncated file that a later batch run would
silently skip (KNOWLEDGE.md 4.7).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .backends import Backend, CachingBackend
from .probe import VideoInfo, probe, require_tools
from .region import Region, build_region


class EncodeError(RuntimeError):
    pass


@dataclass
class EncodeOpts:
    crf: int = 18
    x264_preset: str = "medium"
    faststart: bool = True


def _decoder_cmd(ffmpeg: str, src: Path) -> list[str]:
    return [ffmpeg, "-v", "error", "-nostdin", "-i", str(src),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]


def _encoder_cmd(ffmpeg: str, dst: Path, info: VideoInfo, opts: EncodeOpts) -> list[str]:
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-y",
        # input 0: our processed frames
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{info.width}x{info.height}",
        # exact rational rate -- a truncated float here is what causes drift
        "-r", str(info.fps),
        "-i", "-",
        # input 1: the original, only for its audio and metadata
        "-i", str(info.source),
        "-map", "0:v:0",
        # '?' makes audio optional: AI-generated clips are often silent, and a
        # hard -map 1:a:0 would abort on them.
        "-map", "1:a:0?",
        "-map_metadata", "1",
        "-c:v", "libx264", "-crf", str(opts.crf), "-preset", opts.x264_preset,
        "-pix_fmt", "yuv420p",
        # bit-exact audio passthrough; re-encoding to AAC every pass is pure loss
        "-c:a", "copy",
        # NO -shortest. It ends the output when the shortest stream ends, and real
        # files routinely have an audio track shorter than the video -- one source
        # here was 2.6s short, which silently truncated 78 frames off the end. The
        # rawvideo input on stdin terminates by itself, so it bought nothing.
        # Measured on a 151-frame clip: with -shortest 150 frames, without it 151.
    ]
    if opts.faststart:
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(dst))
    return cmd


def _fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def _read_exact(stream, view: memoryview) -> int:
    """Fill `view` from `stream`, looping over short reads.

    A pipe hands back whatever is buffered -- typically 64 KB, far less than one
    frame -- so a single readinto() returning less than requested means "more is
    coming", not EOF. Treating a short read as end-of-stream stops the loop on
    the very first frame.
    """
    need = len(view)
    off = 0
    while off < need:
        got = stream.readinto(view[off:])
        if not got:
            break
        off += got
    return off


def _drain(pipe) -> str:
    if pipe is None:
        return ""
    try:
        return pipe.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""


def run_inpaint(
    src: Path,
    dst: Path,
    backend: Backend,
    *,
    box,
    dilate_px: int,
    feather_px: int,
    margin_px: int,
    encode: EncodeOpts | None = None,
    progress: bool = True,
) -> Region:
    """Stream `src` -> `dst`, inpainting the watermark tile on every frame."""
    ffmpeg, _ = require_tools()
    encode = encode or EncodeOpts()

    if src.resolve() == dst.resolve():
        raise EncodeError(f"refusing to overwrite the input: {src}")

    info = probe(src)
    region = build_region(
        box, info.width, info.height,
        dilate_px=dilate_px, feather_px=feather_px, margin_px=margin_px,
    )
    ys, xs = region.tile_slice
    mask = region.inpaint_mask
    alpha = region.alpha
    inv_alpha = 1.0 - alpha

    frame_bytes = info.width * info.height * 3
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=dst.parent, prefix=f".{dst.stem}.", suffix=dst.suffix or ".mp4"
    )
    os.close(tmp_fd)
    tmp = Path(tmp_name)

    dec = subprocess.Popen(_decoder_cmd(ffmpeg, src), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, bufsize=0)
    enc = subprocess.Popen(_encoder_cmd(ffmpeg, tmp, info, encode), stdin=subprocess.PIPE,
                           stderr=subprocess.PIPE, bufsize=0)
    assert dec.stdout is not None and enc.stdin is not None

    n = 0
    broken = truncated = False
    dec_err = enc_err = ""
    started = time.monotonic()
    try:
        buf = bytearray(frame_bytes)
        view = memoryview(buf)
        while True:
            got = _read_exact(dec.stdout, view)
            if got == 0:
                break  # clean end of stream
            if got < frame_bytes:
                truncated = True
                break  # trailing partial frame: drop it rather than emit garbage

            frame = np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
            tile = frame[ys, xs]
            patch = backend.inpaint(tile, mask)

            # Binary mask went to the model; the blurred mask blends the result.
            # This is the step that removes the rectangular seam.
            blended = tile.astype(np.float32) * inv_alpha + patch.astype(np.float32) * alpha
            out = frame.copy()
            out[ys, xs] = np.clip(blended, 0, 255).astype(np.uint8)

            enc.stdin.write(out.tobytes())  # type: ignore[union-attr]
            n += 1
            if progress and (n % 15 == 0 or n == 1):
                elapsed = time.monotonic() - started
                rate = n / elapsed if elapsed else 0.0
                left = (info.nframes - n) / rate if rate and info.nframes > n else 0.0
                pct = f"{100.0 * n / info.nframes:.0f}%" if info.nframes else "?"
                print(
                    f"\r[wmrm] {n}/{info.nframes or '?'} ({pct})  "
                    f"{rate:.1f} fps  eta {_fmt_eta(left)}   ",
                    end="", file=sys.stderr, flush=True,
                )
    except BrokenPipeError:
        broken = True
    finally:
        if progress:
            print(file=sys.stderr)
        try:
            if enc.stdin and not enc.stdin.closed:
                enc.stdin.close()
        except BrokenPipeError:
            pass

        # Order matters. Reading the decoder's stderr waits for EOF, which only
        # arrives when it exits -- and it cannot exit while it is blocked writing
        # frames that nobody is draining any more. Stop it first, then read.
        if dec.poll() is None:
            dec.kill()
        dec_err = _drain(dec.stderr)
        dec.wait()
        enc_err = _drain(enc.stderr)
        enc.wait()

    if broken:
        tmp.unlink(missing_ok=True)
        raise EncodeError(
            "ffmpeg encoder exited while frames were still being written:\n"
            f"{enc_err.strip()[:800]}"
        )
    if n == 0:
        tmp.unlink(missing_ok=True)
        raise EncodeError(f"decoded 0 frames from {src}\n{dec_err.strip()[:600]}")
    if enc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise EncodeError(f"ffmpeg encode failed:\n{enc_err.strip()[:600]}")

    os.replace(tmp, dst)  # atomic

    if truncated:
        print(f"[wmrm] warning: {src.name} ended on a partial frame; it was dropped",
              file=sys.stderr)
    elapsed = time.monotonic() - started
    extra = f"  {backend.stats()}" if isinstance(backend, CachingBackend) else ""
    print(
        f"[wmrm] {dst.name}: {n} frames in {_fmt_eta(elapsed)} "
        f"({n / elapsed:.1f} fps){extra}",
        file=sys.stderr,
    )
    return region


def run_fast(
    src: Path,
    dst: Path,
    *,
    box,
    dilate_px: int,
    feather_px: int,
    margin_px: int,
    blur_sigma: float = 2.0,
    encode: EncodeOpts | None = None,
) -> Region:
    """ffmpeg-only path: `delogo` + radially feathered re-blur, one filter graph.

    Near-realtime, no model. `delogo` alone leaves a rectangular seam that reads
    as 'edited' louder than the badge did, so the blurred corner is composited
    back through the same alpha ramp the LaMa path uses -- which also means the
    feather anchor follows the box instead of being nailed to 0,0 like the
    reference implementation (KNOWLEDGE.md 2.1).

    Only appropriate on smooth backgrounds; on texture it smears.
    """
    ffmpeg, _ = require_tools()
    encode = encode or EncodeOpts()
    if src.resolve() == dst.resolve():
        raise EncodeError(f"refusing to overwrite the input: {src}")

    info = probe(src)
    region = build_region(
        box, info.width, info.height,
        dilate_px=dilate_px, feather_px=feather_px, margin_px=margin_px,
    )
    b, t = region.box, region.tile

    with tempfile.TemporaryDirectory(prefix="wmrm-") as td:
        # Reuse the exact same alpha as the model path, as a real PNG rather
        # than a geq expression evaluated per frame.
        mask_png = Path(td) / "alpha.png"
        cv2.imwrite(str(mask_png), (region.alpha[:, :, 0] * 255).astype(np.uint8))

        filt = (
            f"[0:v]delogo=x={b.x}:y={b.y}:w={b.w}:h={b.h}[clean];"
            f"[clean]split=2[base][src];"
            f"[src]crop={t.w}:{t.h}:{t.x}:{t.y},gblur=sigma={blur_sigma}[blur];"
            f"[1:v]format=gray,scale={t.w}:{t.h}[m];"
            f"[blur][m]alphamerge[ba];"
            f"[base][ba]overlay={t.x}:{t.y}:format=auto:shortest=1[out]"
        )
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=dst.parent, prefix=f".{dst.stem}.", suffix=dst.suffix or ".mp4"
        )
        os.close(tmp_fd)
        tmp = Path(tmp_name)
        cmd = [
            ffmpeg, "-v", "error", "-nostdin", "-y", "-stats",
            "-i", str(src),
            "-loop", "1", "-i", str(mask_png),
            "-filter_complex", filt,
            "-map", "[out]", "-map", "0:a:0?", "-map_metadata", "0",
            "-c:v", "libx264", "-crf", str(encode.crf), "-preset", encode.x264_preset,
            # No -shortest: it truncates the video to a shorter audio track. The
            # looped mask input is bounded by `shortest=1` on the overlay filter
            # instead, which ends the graph without touching the muxer.
            "-pix_fmt", "yuv420p", "-c:a", "copy",
        ]
        if encode.faststart:
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(tmp))

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise EncodeError(f"ffmpeg fast path failed:\n{res.stderr.strip()[:800]}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, dst)

    print(f"[wmrm] {dst.name}: written (fast path)", file=sys.stderr)
    return region
