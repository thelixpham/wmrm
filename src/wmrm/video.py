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

from .pipeline import EncodeOpts, EncodeError
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


def _hms(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{seconds:.1f}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _sizeof(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} B" if unit == "B" else f"{n:.1f} {unit}"
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
    # and on how much of the GPU one segment already uses -- neither is knowable from
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


def _extract_tile(ffmpeg: str, src: Path, tile, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(src),
           "-vf", f"crop={tile.w}:{tile.h}:{tile.x}:{tile.y}",
           "-start_number", "0", str(out_dir / "%06d.png")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise ProPainterError(f"tile extraction failed:\n{res.stderr[-2000:]}")
    return len(list(out_dir.glob("*.png")))


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

    t_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="wmrm-pp-") as td:
        work = Path(td)
        raw = work / "tile"
        t0 = time.monotonic()
        n = _extract_tile(ffmpeg, src, tile, raw)
        t_extract = time.monotonic() - t0
        if n == 0:
            raise ProPainterError("no frames were extracted from the tile")
        say(f"[pp] extracted {n} tile frames ({tile.w}x{tile.h}) "
            f"in {_hms(t_extract)}, {_sizeof(_dir_bytes(raw))} on disk")

        # Same binary mask the model paths use, so every engine is asked to remove
        # exactly the same pixels.
        mask_png = work / "mask.png"
        cv2.imwrite(str(mask_png), region.inpaint_mask)

        clean = work / "clean"
        clean.mkdir()
        src_frames = sorted(raw.glob("*.png"))
        step = max(1, opts.segment)
        n_seg = (n + step - 1) // step

        # Stage every segment's input up front. These are directories of symlinks --
        # no pixels are copied -- so the whole set costs kilobytes even for an hour
        # of video, and having them ready means a worker never waits on setup.
        plan = []                   # (si, seg_in, out_dir, start, end, lo, hi)
        for si, start in enumerate(range(0, n, step)):
            end = min(n, start + step)
            lo = max(0, start - opts.overlap)
            hi = min(n, end + opts.overlap)

            seg_in = work / f"seg{si:04d}"
            seg_in.mkdir()
            for j in range(lo, hi):
                # Symlink rather than copy: the frames are already on disk and a
                # long video would otherwise be written twice.
                (seg_in / f"{j - lo:06d}.png").symlink_to(src_frames[j])
            plan.append((si, seg_in, work / f"out{si:04d}", start, end, lo, hi))

        def collect(item, got: Path) -> int:
            """Move one finished segment's kept frames into `clean`."""
            si, seg_in, out_dir, start, end, lo, hi = item
            produced = sorted(got.glob("*.png"))
            if len(produced) != hi - lo:
                raise ProPainterError(
                    f"segment {si} returned {len(produced)} frames, expected "
                    f"{hi - lo}; refusing to guess how they line up")
            for j in range(start, end):
                shutil.copyfile(produced[j - lo], clean / f"{j:06d}.png")
            shutil.rmtree(seg_in, ignore_errors=True)
            shutil.rmtree(out_dir, ignore_errors=True)
            return end - start

        workers = max(1, min(opts.workers, len(plan)))
        say(f"[pp] {n_seg} segment(s) of {step} frames, {workers} at a time")
        t_model0 = time.monotonic()
        done_frames = 0

        if workers == 1:
            for item in plan:
                si, seg_in, out_dir, start, end, lo, hi = item
                say(f"[pp] segment {si + 1}/{n_seg}: frames {start}-{end - 1} "
                    f"(+{start - lo}/{hi - end} context)")
                ts = time.monotonic()
                done_frames += collect(item, _run_segment(opts, seg_in, mask_png,
                                                          out_dir))
                say(f"[pp]   {_hms(time.monotonic() - ts)}  "
                    f"{done_frames / (time.monotonic() - t_model0):.2f} fps avg")
        else:
            # Threads, not processes: every task blocks in subprocess.run, which
            # releases the GIL for the whole of the work that matters.
            #
            # Results are consumed as they complete rather than in submission order.
            # That is safe *here* specifically because segments write disjoint
            # filenames into `clean` -- frame j only ever comes from the one segment
            # that keeps it -- so out-of-order completion cannot reorder the video.
            # Consuming eagerly also means at most `workers` output directories are
            # alive at once instead of all of them.
            # Announce each segment as it *starts*, not only when it finishes.
            # Reporting completions alone made a concurrent run read as a serial one
            # -- three segments all logging the same elapsed time looks like a stuck
            # clock unless you already know they were launched together.
            def launch(item):
                si, seg_in, out_dir, start, end, lo, hi = item
                say(f"[pp] segment {si + 1}/{n_seg} start: frames {start}-{end - 1} "
                    f"(+{start - lo}/{hi - end} context) "
                    f"at {_hms(time.monotonic() - t_model0)}")
                return _run_segment(opts, seg_in, mask_png, out_dir)

            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(launch, it): it for it in plan}
                for fut in cf.as_completed(futs):
                    item = futs[fut]
                    done_frames += collect(item, fut.result())
                    el = time.monotonic() - t_model0
                    say(f"[pp] segment {item[0] + 1}/{n_seg} done  "
                        f"({done_frames}/{n} frames, {_hms(el)}, "
                        f"{done_frames / el:.2f} fps avg)")

        t_model = time.monotonic() - t_model0
        # Frames the model actually ran on, not frames kept. Context frames at the
        # segment joins are computed and then thrown away, so `kept / t_model`
        # flatters the model on a many-segment run and makes two configurations with
        # different segment counts look faster or slower than they are.
        model_frames = sum(hi - lo for _, _, _, _, _, lo, hi in plan)
        waste = model_frames - n
        say(f"[pp] {n} frames repaired in {_hms(t_model)} "
            f"({n / max(t_model, 1e-6):.2f} fps, {workers} worker(s))"
            + (f"  |  {model_frames} frames computed, {waste} discarded as segment "
               f"context ({100 * waste / model_frames:.1f}% wasted)" if waste else
               "  |  no context discarded"))

        done = sorted(clean.glob("*.png"))
        if len(done) != n:
            raise ProPainterError(f"assembled {len(done)} frames, expected {n}")
        probe_img = cv2.imread(str(done[0]))
        if probe_img is None or probe_img.shape[:2] != (tile.h, tile.w):
            got = "unreadable" if probe_img is None else f"{probe_img.shape[1]}x{probe_img.shape[0]}"
            raise ProPainterError(
                f"cleaned tile is {got}, expected {tile.w}x{tile.h}. "
                "This is the macro_block_size resize -- the PNG frames should be "
                "exact, so something read the mp4 instead.")
        say("[pp] compositing")
        t_comp0 = time.monotonic()

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
            # The tile sequence is timed to the source's exact rational rate, not a
            # rounded decimal, or the overlay drifts against the video it sits on.
            "-framerate", f"{info.fps.numerator}/{info.fps.denominator}",
            "-start_number", "0", "-i", str(clean / "%06d.png"),
            "-loop", "1", "-i", str(alpha_png),
            "-filter_complex", filt,
            "-map", "[out]", "-map", "0:a:0?", "-map_metadata", "0",
            "-c:v", "libx264", "-crf", str(encode.crf),
            "-preset", encode.x264_preset,
            # No -shortest: it ends the file when the shortest stream ends, and an
            # audio track 2.6s shorter than the video silently cost 78 frames off
            # the tail -- the `duration 60.96s vs 58.36s` verify failure. The looped
            # alpha input is bounded by `shortest=1` on the overlay filter, so the
            # graph still terminates.
            "-pix_fmt", "yuv420p", "-c:a", "copy",
        ]
        if encode.faststart:
            cmd += ["-movflags", "+faststart"]
        cmd.append(str(tmp))

        peak_bytes = _dir_bytes(work)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise EncodeError(f"compositing failed:\n{res.stderr[-2000:]}")
        t_composite = time.monotonic() - t_comp0
        os.replace(tmp, dst)

    total = time.monotonic() - t_start
    # The breakdown is the point, not the total. Only `model` responds to the
    # ProPainter settings or to a faster card; extract and composite are ffmpeg on
    # the CPU and will not move no matter what is tuned here. Printing the split
    # stops effort going into the 82% that is already the model, or into the model
    # when it turns out ffmpeg is what costs.
    pct = lambda x: 100 * x / max(total, 1e-6)          # noqa: E731
    say(f"[pp] TIME  extract {_hms(t_extract)} ({pct(t_extract):.0f}%)  "
        f"model {_hms(t_model)} ({pct(t_model):.0f}%)  "
        f"composite {_hms(t_composite)} ({pct(t_composite):.0f}%)")
    say(f"[pp] TOTAL {_hms(total)} for {n} frames "
        f"({n / max(total, 1e-6):.2f} fps, "
        f"{total / max(info.duration, 1e-6):.1f}x realtime)  "
        f"peak temp disk {_sizeof(peak_bytes)}")
    # Extrapolate, because a one-minute test says nothing about the videos this is
    # actually for until it is scaled up.
    say(f"[pp] EXTRAPOLATED  1 hour of this footage -> "
        f"~{_hms(3600.0 * total / max(info.duration, 1e-6))}, "
        f"~{_sizeof(peak_bytes * 3600.0 / max(info.duration, 1e-6))} temp disk "
        f"at --pp-segment {opts.segment}")
    return region
