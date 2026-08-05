"""ProPainter backend: fill the watermark from *other frames* instead of guessing.

Why this exists. Everything else in this tool works inside a single frame, and that
is a hard ceiling. Measured on the reference clip, tile 400x168, background
locked-edge floor 12.24:

    engine            residual   vs floor   corr   damage
    original             25.60     +13.36    1.00     0.00
    un-blend             13.02      +0.77    0.99     0.00
    LaMa, full box       12.42      +0.17    0.68     0.00
    ProPainter           11.36      -0.89    0.82     0.11

`residual` is pixel-locked structure left inside the mark, measured the way the
detector finds watermarks; below the surrounding floor means nothing findable is
left. `corr` is whether frame-to-frame change happens where the real content
changed -- low means the patch moves on its own, which is what reads as boiling.

Un-blend keeps the picture but cannot reach zero residual: the leftover is
proportional to the error in alpha, and alpha has to be estimated from background
statistics that are unobservable under the mark. LaMa reaches zero residual by
deleting the region, but it invents the replacement per frame and visibly wrecks
detail when moving content crosses the box. ProPainter is the only one that does
both, because it takes the pixels from neighbouring frames where the same content is
not covered -- real content, not invention, which is also why it stays coherent.

Integration notes, each one a thing that bit:

- **Run on the tile, not the frame.** 400x168 against 1920x1080 is 77x fewer pixels
  for identical output, and it is the difference between a long video fitting in
  memory and not.
- **Read its PNG frames, never its mp4.** Its writer pads to a multiple of 16 --
  "resizing from (400, 168) to (400, 176)" -- so the mp4 does not line up with the
  crop it came from. `--save_frames` and the PNGs are exact.
- **Segment the clip.** `--subvideo_length` only chunks inference; the script still
  loads every frame into memory first, so an unsegmented hour of video is tens of
  gigabytes. Segments overlap and the overlap is discarded, so the model always has
  temporal context either side of the frames we keep.
- **It picks its own device.** `model.misc.get_device()` takes MPS first, then CUDA
  if both cuda and cudnn report available, else CPU. There is no flag to override,
  so the log here reports what that same rule will choose rather than claiming.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .pipeline import EncodeOpts, EncodeError, _drain, _read_exact
from .probe import probe, require_tools
from .region import Region, build_region

# Frames per ProPainter invocation. Bounded by the fact that the script materialises
# every frame it is given as a tensor before it starts.
SEGMENT = 400
# Frames of context added to each side of a segment and then thrown away. Without it
# the first and last frames of a segment have one-sided temporal context and the
# joins show.
OVERLAP = 20


class ProPainterError(RuntimeError):
    pass


# ProPainter is published under the S-Lab License 1.0, which permits
# "redistribution and use for non-commercial purpose" and requires contacting the
# authors for commercial use. Every other engine in this tool is Apache-2.0 or
# BSD-family and carries no such restriction, so this warning exists to stop the one
# encumbered path from being used on paid work by accident -- it is the sort of thing
# that is invisible in the output and expensive to discover later.
LICENCE_WARNING = """\
[pp] LICENCE: ProPainter is S-Lab License 1.0 -- NON-COMMERCIAL use only.
[pp]          Commercial use requires written permission from the authors:
[pp]            Dr. Shangchen Zhou <shangchenzhou@gmail.com>
[pp]            Prof. Chen Change Loy <ccloy@ntu.edu.sg>
[pp]          The other engines (unblend / high / fast / draft) are Apache-2.0 or
[pp]          BSD and have no such restriction. Set WMRM_PROPAINTER_OK=1 to silence
[pp]          this once your own legal position is settled."""


def _hms(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{seconds:.1f}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _sizeof(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


@dataclass
class ProPainterOpts:
    repo: Path
    python: str = sys.executable
    segment: int = SEGMENT
    overlap: int = OVERLAP
    subvideo_length: int = 80
    neighbor_length: int = 10
    ref_stride: int = 10
    raft_iter: int = 20
    mask_dilation: int = 4
    fp16: bool = True
    # Segments run concurrently. Default 1 because the right number depends on VRAM
    # and on how much of the GPU one segment already uses -- both unmeasurable from
    # here. Raise it and watch the reported fps; stop when it stops improving or the
    # device runs out of memory.
    workers: int = 1


def find_repo(explicit: str | os.PathLike | None = None) -> Path:
    """Locate the ProPainter checkout.

    It is a research repo, not a package: there is nothing to import and no entry
    point, so the path has to come from somewhere. Checked in order: the argument,
    $PROPAINTER_HOME, then a sibling of this project, which is where `git clone`
    lands it if you follow the README.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("PROPAINTER_HOME"):
        candidates.append(Path(os.environ["PROPAINTER_HOME"]))
    here = Path(__file__).resolve()
    for up in (here.parents[3], here.parents[4]):
        candidates.append(up / "ProPainter")

    for c in candidates:
        if (c / "inference_propainter.py").is_file():
            return c.resolve()
    raise ProPainterError(
        "ProPainter checkout not found. Clone it and point at it:\n"
        "  git clone https://github.com/sczhou/ProPainter.git\n"
        "  export PROPAINTER_HOME=/path/to/ProPainter\n"
        f"tried: {', '.join(str(c) for c in candidates)}"
    )


def describe_device() -> str:
    """Report the device ProPainter will pick, using its own rule."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "unknown (torch not importable)"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps (ProPainter prefers MPS when present)"
    if torch.cuda.is_available() and torch.backends.cudnn.is_available():
        i = torch.cuda.current_device()
        vram = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        return f"cuda ({torch.cuda.get_device_name(i)}, {vram:.1f} GB)"
    why = ("cuda present but cudnn is not"
           if torch.cuda.is_available() else "no CUDA device")
    return f"cpu ({why}) -- expect minutes per hundred frames"


def _open_tile_reader(ffmpeg: str, src: Path, tile):
    """Sequential rawvideo stream of the cropped tile.

    Streaming rather than "extract every frame to PNG first" is what makes an
    hours-long source possible at all. Measured on the reference tile, PNG frames
    cost 83.5 KB each -- 35 GB per hour of footage, and the same again for the
    repaired copy. A 3-hour video would need ~105 GB of scratch before this changed.
    Reading sequentially also avoids seeking, so no assumptions about keyframe
    placement are needed.
    """
    return subprocess.Popen(
        [ffmpeg, "-v", "error", "-nostdin", "-i", str(src),
         "-vf", f"crop={tile.w}:{tile.h}:{tile.x}:{tile.y}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)


def _open_tile_writer(ffmpeg: str, tile, fps, out: Path):
    """Lossless sink for the repaired tile.

    FFV1 because the tile is composited back over untouched pixels afterwards; any
    lossy step here would show up as a seam against them.
    """
    return subprocess.Popen(
        [ffmpeg, "-v", "error", "-nostdin", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{tile.w}x{tile.h}",
         "-framerate", f"{fps.numerator}/{fps.denominator}",
         "-i", "-", "-c:v", "ffv1", "-level", "3", str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _run_segment(opts: ProPainterOpts, frames_dir: Path, mask_png: Path,
                 out_dir: Path) -> Path:
    """Invoke the upstream script once. Returns the directory of output PNGs."""
    cmd = [opts.python, "inference_propainter.py",
           "-i", str(frames_dir), "-m", str(mask_png), "-o", str(out_dir),
           "--save_frames",
           "--mask_dilation", str(opts.mask_dilation),
           "--subvideo_length", str(opts.subvideo_length),
           "--neighbor_length", str(opts.neighbor_length),
           "--ref_stride", str(opts.ref_stride),
           "--raft_iter", str(opts.raft_iter)]
    if opts.fp16:
        cmd.append("--fp16")          # ignored on CPU by the script itself
    res = subprocess.run(cmd, cwd=str(opts.repo), capture_output=True, text=True)
    if res.returncode != 0:
        raise ProPainterError(
            f"ProPainter failed on {frames_dir.name}:\n"
            f"{res.stdout[-1500:]}\n{res.stderr[-2500:]}")
    got = out_dir / frames_dir.name / "frames"
    if not got.is_dir():
        raise ProPainterError(
            f"ProPainter produced no frames for {frames_dir.name}; it writes to "
            f"<output>/<input dir name>/frames and that path is missing.\n"
            f"{res.stdout[-1500:]}")
    return got


def run_propainter(
    src: Path,
    dst: Path,
    *,
    box,
    dilate_px: int,
    feather_px: int,
    margin_px: int,
    opts: ProPainterOpts,
    encode: EncodeOpts | None = None,
    progress: bool = True,
) -> Region:
    ffmpeg, _ = require_tools()
    encode = encode or EncodeOpts()
    if src.resolve() == dst.resolve():
        raise EncodeError(f"refusing to overwrite the input: {src}")

    info = probe(src)
    region = build_region(box, info.width, info.height, dilate_px=dilate_px,
                          feather_px=feather_px, margin_px=margin_px)
    tile = region.tile
    say = (lambda m: print(m, file=sys.stderr, flush=True)) if progress else (lambda m: None)

    if not os.environ.get("WMRM_PROPAINTER_OK"):
        say(LICENCE_WARNING)

    t_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="wmrm-pp-") as td:
        work = Path(td)
        # Same binary mask the model paths use, so every engine is asked to remove
        # exactly the same pixels.
        mask_png = work / "mask.png"
        cv2.imwrite(str(mask_png), region.inpaint_mask)

        clean_tile = work / "clean_tile.mkv"
        frame_bytes = tile.w * tile.h * 3
        step = max(1, opts.segment)
        seg_times: list[float] = []
        peak_bytes = 0
        n = 0                      # frames written out
        t_io = 0.0
        expected = info.nframes or 0

        dec = _open_tile_reader(ffmpeg, src, tile)
        enc = _open_tile_writer(ffmpeg, tile, info.fps, clean_tile)

        def read_one():
            """Next tile frame from the decoder, or None at end of stream."""
            buf = bytearray(frame_bytes)
            if _read_exact(dec.stdout, memoryview(buf)) == 0:
                return None
            return np.frombuffer(bytes(buf), np.uint8).reshape(tile.h, tile.w, 3)

        try:
            # `window` holds frames not yet written out. The first `lead` of them are
            # context for the model only -- already written in a previous segment --
            # and everything after is new. Carrying frames in memory rather than
            # re-reading them is what lets the reader stay a single forward pass.
            window: list[np.ndarray] = []
            lead = 0
            si = 0
            eof = False
            wall0 = time.monotonic()

            while not eof or len(window) > lead:
                # --- build up to `workers` independent segments ----------------- #
                # A 412x172 tile is small for a modern GPU, so one ProPainter
                # process almost certainly leaves it idle. Running several segments
                # at once is the cheapest way to use it, and it also spreads the
                # fixed per-process startup cost over concurrent work instead of
                # paying it serially.
                batch = []          # (index, seg_in dir, out dir, keep_lo, keep_hi,
                                    #  lead, tail, n_frames)
                while len(batch) < max(1, opts.workers):
                    want = lead + step + opts.overlap
                    t0 = time.monotonic()
                    while len(window) < want:
                        f = read_one()
                        if f is None:
                            eof = True
                            break
                        window.append(f)
                    t_io += time.monotonic() - t0
                    if len(window) <= lead:
                        break

                    tail = min(opts.overlap, max(0, len(window) - lead - step))
                    keep_lo, keep_hi = lead, len(window) - tail

                    seg_in = work / f"seg{si:04d}"
                    seg_in.mkdir()
                    t0 = time.monotonic()
                    for j, frame in enumerate(window):
                        cv2.imwrite(str(seg_in / f"{j:06d}.png"), frame)
                    t_io += time.monotonic() - t0

                    batch.append((si, seg_in, work / f"out{si:04d}",
                                  keep_lo, keep_hi, lead, tail, len(window)))

                    if eof and tail == 0:
                        window = window[:0]
                        lead = 0
                        break
                    carry = max(0, keep_hi - opts.overlap)
                    window = window[carry:]
                    lead = keep_hi - carry
                    si += 1

                if not batch:
                    break

                # --- run them concurrently -------------------------------------- #
                t0 = time.monotonic()
                if len(batch) == 1:
                    results = [_run_segment(opts, batch[0][1], mask_png, batch[0][2])]
                else:
                    with cf.ThreadPoolExecutor(max_workers=len(batch)) as pool:
                        # Threads are fine: every task blocks on a subprocess, so the
                        # GIL is released for the whole of the work that matters.
                        results = list(pool.map(
                            lambda b: _run_segment(opts, b[1], mask_png, b[2]),
                            batch))
                dt = time.monotonic() - t0
                seg_times.append(dt)

                # --- write out strictly in order -------------------------------- #
                for (idx, seg_in, out_dir, keep_lo, keep_hi, ld, tl, nfr), got in \
                        zip(batch, results):
                    produced = sorted(got.glob("*.png"))
                    if len(produced) != nfr:
                        raise ProPainterError(
                            f"segment {idx} returned {len(produced)} frames, "
                            f"expected {nfr}; refusing to guess how they line up")
                    t0 = time.monotonic()
                    for j in range(keep_lo, keep_hi):
                        img = cv2.imread(str(produced[j]))
                        if img is None or img.shape[:2] != (tile.h, tile.w):
                            shape = "unreadable" if img is None else \
                                f"{img.shape[1]}x{img.shape[0]}"
                            raise ProPainterError(
                                f"repaired frame is {shape}, expected "
                                f"{tile.w}x{tile.h}. That is the macro_block_size "
                                "resize -- the PNGs should be exact, so something "
                                "read the mp4 instead.")
                        enc.stdin.write(img.tobytes())
                    t_io += time.monotonic() - t0
                    n += keep_hi - keep_lo
                    peak_bytes = max(peak_bytes,
                                     _dir_bytes(seg_in) + _dir_bytes(out_dir))
                    shutil.rmtree(seg_in, ignore_errors=True)
                    shutil.rmtree(out_dir, ignore_errors=True)

                kept = sum(b[4] - b[3] for b in batch)
                # Rate is measured on wall clock, not summed segment times: with
                # workers > 1 those overlap, so adding them would report throughput
                # the pipeline never achieved.
                rate = n / max(time.monotonic() - wall0, 1e-6)
                eta = ((expected - n) / rate) if expected > n else 0.0
                say(f"[pp] batch of {len(batch)}: frames {n - kept}-{n - 1}  "
                    f"{_hms(dt)}  {kept / dt:.2f} fps"
                    + (f"  |  {rate:.2f} fps avg, eta {_hms(eta)}"
                       if expected else f"  |  {rate:.2f} fps avg"))
        finally:
            if dec.poll() is None:
                dec.kill()
            dec_err = _drain(dec.stderr)
            dec.wait()
            try:
                enc.stdin.close()
            except (OSError, ValueError):
                pass
            enc_err = _drain(enc.stderr)
            if enc.wait() != 0:
                raise ProPainterError(f"writing the repaired tile failed:\n{enc_err}")

        if n == 0:
            raise ProPainterError(
                f"no frames were read from the tile.\n{dec_err[-1000:]}")
        if expected and n != expected:
            raise ProPainterError(
                f"repaired {n} frames but the source reports {expected}; "
                "refusing to write a video of the wrong length")

        t_extract = t_io
        t_model = sum(seg_times)
        tile_bytes = clean_tile.stat().st_size
        say(f"[pp] {n} frames repaired in {_hms(t_model)} "
            f"({n / max(t_model, 1e-6):.2f} fps); compositing")
        t0 = time.monotonic()

        alpha_png = work / "alpha.png"
        cv2.imwrite(str(alpha_png), (region.alpha[:, :, 0] * 255).astype(np.uint8))

        filt = (
            f"[2:v]format=gray,scale={tile.w}:{tile.h}[m];"
            f"[1:v][m]alphamerge[ba];"
            f"[0:v][ba]overlay={tile.x}:{tile.y}:format=auto:shortest=1[out]"
        )
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=dst.parent, prefix=f".{dst.stem}.", suffix=dst.suffix or ".mp4")
        os.close(tmp_fd)
        tmp = Path(tmp_name)
        cmd = [
            ffmpeg, "-v", "error", "-nostdin", "-y", "-stats",
            "-i", str(src),
            # A single lossless FFV1 file, not 324k PNGs. It already carries the
            # source's exact rational rate from the writer, so there is no
            # -framerate to get wrong here and no drift against the video it sits on.
            "-i", str(clean_tile),
            "-loop", "1", "-i", str(alpha_png),
            "-filter_complex", filt,
            "-map", "[out]", "-map", "0:a:0?", "-map_metadata", "0",
            "-c:v", "libx264", "-crf", str(encode.crf),
            "-preset", encode.x264_preset,
            # No -shortest: it ends the file when the shortest stream ends, and an
            # audio track 2.6s shorter than the video silently cost 78 frames off
            # the tail. The looped alpha input is bounded by `shortest=1` on the
            # overlay filter, so the graph still terminates.
            "-pix_fmt", "yuv420p", "-c:a", "copy",
        ]
        if encode.faststart:
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(tmp))

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise EncodeError(f"compositing failed:\n{res.stderr[-2000:]}")
        t_composite = time.monotonic() - t0
        os.replace(tmp, dst)

    total = time.monotonic() - t_start
    # The breakdown is the point, not the total. Model time is the only part a GPU
    # or --pp-segment changes; extract and composite are ffmpeg on CPU and will not
    # move. Seeing which one dominates tells you whether tuning the model settings
    # is worth anything on this footage.
    say(f"[pp] TIME  extract {_hms(t_extract)} "
        f"({100 * t_extract / max(total, 1e-6):.0f}%)  "
        f"model {_hms(t_model)} ({100 * t_model / max(total, 1e-6):.0f}%)  "
        f"composite {_hms(t_composite)} "
        f"({100 * t_composite / max(total, 1e-6):.0f}%)")
    say(f"[pp] TOTAL {_hms(total)} for {n} frames "
        f"({n / max(total, 1e-6):.2f} fps, "
        f"{total / max(info.duration, 1e-6):.1f}x realtime)  "
        f"temp disk: {_sizeof(peak_bytes)} per segment + "
        f"{_sizeof(tile_bytes)} lossless tile")
    # Extrapolate, because the interesting videos are hours long and a one-minute
    # test says nothing until it is scaled up. Disk is split deliberately: the
    # segment scratch does NOT grow with the video, only the lossless tile does, and
    # that distinction is the whole point of streaming instead of pre-extracting.
    fph = 3600.0 * float(info.fps)
    per_hour = fph / max(n / max(total, 1e-6), 1e-6)
    disk_hour = peak_bytes + int(tile_bytes * fph / max(n, 1))
    say(f"[pp] EXTRAPOLATED  1 hour of this footage -> ~{_hms(per_hour)}, "
        f"~{_sizeof(disk_hour)} temp disk "
        f"({_sizeof(peak_bytes)} fixed + {_sizeof(int(tile_bytes * fph / max(n, 1)))} "
        f"growing) at --pp-segment {opts.segment}")
    return region
