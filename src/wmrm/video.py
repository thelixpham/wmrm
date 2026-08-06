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
- **Nothing is materialised as PNG.** Frames stream: one ffmpeg decodes and crops the
  tile into a pipe, the model works on numpy in memory, and the results go straight
  into a second ffmpeg that composites them back over the source. The old shape of
  this file wrote every tile frame to disk as a PNG, handed the directory to
  upstream's script, read its result PNGs back and fed those to ffmpeg -- which meant
  a ten-hour video paid for 1.7M PNG encodes before the model saw frame one, and
  temp disk scaled with duration (measured 48.1 KB/frame, so ~83 GB). It is now flat
  at a few hundred megabytes of in-flight frames regardless of length.
- **Three stages run concurrently.** Decode, model and composite each own a thread
  with a bounded queue between them, so the model is not waiting on ffmpeg and
  ffmpeg is not waiting on the model. Bounded, not unbounded: the queues are the
  backpressure that keeps memory flat when one stage is slower than another, which
  on a GPU is usually the two ffmpegs and on CPU is always the model.
- **Segment the clip.** The model materialises every frame it is handed as a tensor,
  so it gets `opts.segment` frames at a time. Segments overlap and the overlap is
  discarded, so the model always has temporal context either side of the frames that
  are kept.
- **Segments are processed in order, by one resident model.** This is what removed
  the need for the old symlink staging, the input-purging bookkeeping and the
  out-of-order collection logic -- see `ProPainterOpts.workers`.
- **The models load once per run, not once per segment.** Upstream's inference lives
  entirely inside `if __name__ == '__main__':`, so the old subprocess-per-segment
  reloaded RAFT, the flow completion net and the inpaint generator every time: ~190 MB
  and a CUDA context, ~2700 times on a ten-hour video at the default segment size.
  `vendor/ProPainter/wmrm_worker.py` is the importable equivalent, and
  `tests/test_propainter_parity.py` asserts it produces what the script produced.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .pipeline import EncodeOpts, EncodeError, drain, read_exact
from .probe import probe, require_tools
from .region import Region, build_region

# Frames per model invocation. Bounded by the fact that the model materialises every
# frame it is given as a tensor before it starts. Now that no frames touch the disk
# this is purely a VRAM/RAM knob.
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


def _sizeof(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


@dataclass
class ProPainterOpts:
    repo: Path
    device: str | None = None
    # None means work it out from free VRAM, host RAM and the tile size. A number
    # pins it, which is what you want when reproducing a run or bisecting an OOM.
    segment: int | None = None
    overlap: int = OVERLAP
    subvideo_length: int = 80
    neighbor_length: int = 10
    ref_stride: int = 10
    raft_iter: int = 20
    mask_dilation: int = 4
    fp16: bool = True
    # ffmpeg scene score above which a frame starts a new shot. 0 disables cut
    # detection and falls back to cutting every `segment` frames regardless of
    # content, which is what produced the wrong-shot fill described in _segment_plan.
    scene_threshold: float = 0.3
    # Shots shorter than this are merged with the previous one rather than handed to
    # the model alone. ProPainter needs frames to propagate from; a 4-frame segment
    # has none.
    min_shot: int = 16
    # Accepted and ignored, kept so existing command lines and presets do not break.
    #
    # It used to mean "this many segments in flight", which was worth having when each
    # segment was a separate process that spent 10-20s loading models -- overlapping
    # that load with another segment's compute hid some of it. With one resident model
    # there is nothing left to hide: two segments submitted at once to the same device
    # serialise on it anyway, and the decode/composite work that genuinely can run
    # alongside the model now always does, in its own thread.
    #
    # Running N segments truly concurrently would mean N copies of the weights in
    # VRAM and bringing back the reorder buffer that kept output frames in sequence.
    # That is a real option on a 44 GB card and a deliberate piece of work, not a
    # flag that should quietly do it.
    workers: int = 1


def find_repo(explicit: str | os.PathLike | None = None) -> Path:
    """Locate the ProPainter checkout.

    It is a research repo, not a package: there is nothing to import and no entry
    point, so the path has to come from somewhere. Checked in order: the argument,
    $PROPAINTER_HOME, the copy vendored into this repo, then a sibling of the
    project, which is where `git clone` lands it if you follow the older README.

    The vendored copy is checked before the sibling but after the environment
    variable on purpose. It is the one that ships with a known commit and a parity
    test, so it should win by default; but someone debugging against a different
    upstream needs a way to override it that does not involve editing the tree.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("PROPAINTER_HOME"):
        candidates.append(Path(os.environ["PROPAINTER_HOME"]))
    here = Path(__file__).resolve()
    # src/wmrm/video.py -> parents[2] is the project root holding vendor/.
    candidates.append(here.parents[2] / "vendor" / "ProPainter")
    for up in (here.parents[3], here.parents[4]):
        candidates.append(up / "ProPainter")

    for c in candidates:
        if (c / "inference_propainter.py").is_file():
            return c.resolve()
    raise ProPainterError(
        "ProPainter checkout not found. It ships vendored in this repo; if that is "
        "missing, restore it with:\n"
        "  scripts/vendor-propainter.sh\n"
        "or point at your own checkout:\n"
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


def _lands_on_cpu(device: str | None) -> bool:
    """Would this device selection end up on the CPU?

    Answered before the models load, because loading them takes ~40s and the answer
    decides whether to load them at all.
    """
    try:
        import torch
    except ImportError:
        return True
    if device is None:                       # ProPainter's own rule
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return False
        return not (torch.cuda.is_available() and torch.backends.cudnn.is_available())
    if device.startswith("cuda"):
        return not torch.cuda.is_available()
    if device.startswith("mps"):
        mps = getattr(torch.backends, "mps", None)
        return not (mps is not None and mps.is_available())
    return True


# Keyed on what actually determines the loaded model: where it came from, the device
# it lives on, and its dtype. Everything else in ProPainterOpts is an inference knob
# read per call, so it is reassigned on a hit rather than forcing a reload.
#
# Scope worth being clear about: this is per *process*. `wmrm batch` processes a whole
# folder in one process and gains the whole saving; run.sh invokes `wmrm run` per file,
# so it does not. Closing that gap means teaching `batch` the per-file detect and
# coverage gate that run.sh does in bash.
_WORKER_CACHE: dict[tuple[str, str, bool], object] = {}


def release_worker() -> None:
    """Drop cached models. For a caller that needs the VRAM back mid-process."""
    _WORKER_CACHE.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _load_worker(opts: ProPainterOpts) -> tuple[object, bool]:
    """Get the resident worker, loading it only if this process has not already.

    Returns (worker, was_cached).

    The cache is why this returns a flag: loading is ~40-50s of the run, and a batch
    used to pay it per file. `wmrm batch` over fifty clips spent three quarters of an
    hour loading the same three files, which is the same defect this worker was written
    to fix at the segment level, left in place one level up.
    """
    repo = Path(opts.repo)
    if not (repo / "wmrm_worker.py").is_file():
        raise ProPainterError(
            f"{repo} has no wmrm_worker.py. That file is ours, not upstream's -- a "
            "plain `git clone` of ProPainter will not have it. Use the vendored copy "
            "in this repo (vendor/ProPainter), or copy wmrm_worker.py into your "
            "checkout."
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from wmrm_worker import ProPainterWorker, WorkerOpts
    except ImportError as exc:
        raise ProPainterError(
            f"could not import the ProPainter worker from {repo}: {exc}\n"
            "Its dependencies are torch and torchvision from the same index -- see "
            "the README's ProPainter section."
        ) from exc

    # 'auto' is the CLI's word for "use ProPainter's own rule", which is what the
    # worker does when handed None.
    device = None if opts.device in (None, "", "auto") else opts.device

    # This engine is the default, so a CPU fallback has to be loud. Measured at 0.27
    # fps on six cores -- roughly 1.8 hours per minute of 1080p, ~400x the un-blend
    # path. That is not a slower run, it is a run nobody waits for, and silently
    # starting one is worse than refusing. Naming --device cpu overrides this: then it
    # is a decision rather than an accident.
    if device != "cpu" and _lands_on_cpu(device):
        raise ProPainterError(
            "--quality video would run on the CPU, and that is ~0.27 fps: about 1.8 "
            "hours for one minute of 1080p.\n"
            f"  device asked for : {opts.device or 'auto'}\n"
            f"  what is available: {describe_device()}\n"
            "Either use the engine that is built for the CPU, which on a "
            "semi-transparent mark is also the least destructive one here:\n"
            "  --quality unblend            (34 fps at 1080p, cannot flicker)\n"
            "or say you mean it:\n"
            "  --quality video --device cpu\n"
            "If this box does have a card, torch cannot see it -- most often the "
            "CPU-only wheel. Check with:\n"
            "  python -c \"import torch; print(torch.__version__, "
            "torch.cuda.is_available())\""
        )

    knobs = WorkerOpts(
        subvideo_length=opts.subvideo_length,
        neighbor_length=opts.neighbor_length,
        ref_stride=opts.ref_stride,
        raft_iter=opts.raft_iter,
        mask_dilation=opts.mask_dilation,
        fp16=opts.fp16,
    )
    key = (str(repo), str(device), bool(opts.fp16))
    cached = _WORKER_CACHE.get(key)
    if cached is not None:
        # Inference knobs are read per call, so a change in them needs no reload.
        # fp16 is not among them -- it is part of the key, because it decides the
        # dtype the weights are held in.
        cached.opts = knobs           # type: ignore[attr-defined]
        return cached, True

    worker = ProPainterWorker(device=device, opts=knobs)
    _WORKER_CACHE[key] = worker
    return worker, False


def _decode_cmd(ffmpeg: str, src: Path, tile) -> list[str]:
    """Decode the source and emit only the tile, as raw frames on stdout.

    Cropping in ffmpeg rather than in numpy is what keeps this cheap: a 1080p frame
    is 6 MB and a 400x168 tile is 200 KB, so the pipe carries 3% of the data and the
    Python side never sees a full frame.
    """
    return [ffmpeg, "-v", "error", "-nostdin", "-i", str(src),
            "-vf", f"crop={tile.w}:{tile.h}:{tile.x}:{tile.y}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]


def _composite_cmd(ffmpeg: str, src: Path, dst: Path, info, tile,
                   alpha_png: Path, encode: EncodeOpts) -> list[str]:
    """Overlay the repaired tile stream back onto the untouched source.

    Input 0 is the original (video, audio, metadata), input 1 is our repaired tile
    arriving on stdin, input 2 is the feather alpha as a still.
    """
    filt = (
        f"[2:v]format=gray,scale={tile.w}:{tile.h}[m];"
        f"[1:v][m]alphamerge[ba];"
        f"[0:v][ba]overlay={tile.x}:{tile.y}:format=auto:shortest=1[out]"
    )
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-y",
        "-i", str(src),
        # The tile stream is timed to the source's exact rational rate, not a rounded
        # decimal, or the overlay drifts against the video it sits on.
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{tile.w}x{tile.h}", "-r", str(info.fps),
        "-i", "-",
        "-loop", "1", "-i", str(alpha_png),
        "-filter_complex", filt,
        "-map", "[out]", "-map", "0:a:0?", "-map_metadata", "0",
        "-c:v", "libx264", "-crf", str(encode.crf), "-preset", encode.x264_preset,
        # No -shortest: it ends the file when the shortest stream ends, and an audio
        # track 2.6s shorter than the video silently cost 78 frames off the tail --
        # the `duration 60.96s vs 58.36s` verify failure. The looped alpha input is
        # bounded by `shortest=1` on the overlay filter, so the graph still terminates.
        "-pix_fmt", "yuv420p", "-c:a", "copy",
    ]
    if encode.faststart:
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(dst))
    return cmd


class _Stage(threading.Thread):
    """A daemon thread that records the exception that killed it.

    A bare thread that dies takes its traceback with it and the main loop then blocks
    on a queue that will never be fed again. Keeping the exception lets the caller
    fail with the real cause instead of a hang or a timeout.
    """

    def __init__(self, fn, name: str) -> None:
        super().__init__(name=name, daemon=True)
        self._fn = fn
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self._fn()
        except BaseException as exc:            # noqa: BLE001 -- re-raised by caller
            self.error = exc


def _decode_frames(stream, tile):
    """Generate tile frames from a raw bgr24 pipe, one array each."""
    nbytes = tile.w * tile.h * 3
    buf = bytearray(nbytes)
    view = memoryview(buf)
    while True:
        got = read_exact(stream, view)
        if got == 0:
            return
        if got < nbytes:
            # Trailing partial frame: drop it rather than emit garbage. Same rule as
            # the streaming inpaint path.
            return
        # Copy: `buf` is reused for the next frame, and the array outlives this loop.
        yield np.frombuffer(buf, np.uint8).reshape(tile.h, tile.w, 3).copy()


def _hardware() -> dict:
    """What this machine has. Everything optional, nothing raises.

    No psutil: one more dependency for two numbers that /proc and torch already have,
    and this has to keep working in a container whose installs were pinned months ago.
    """
    hw: dict = {"cpu": os.cpu_count() or 1, "ram_total": 0, "ram_avail": 0,
                "gpu": None, "vram_total": 0, "vram_free": 0}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    hw["ram_total"] = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    hw["ram_avail"] = int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            hw["gpu"] = torch.cuda.get_device_name(i)
            hw["vram_total"] = torch.cuda.get_device_properties(i).total_memory
            try:
                free, _ = torch.cuda.mem_get_info(i)
                hw["vram_free"] = int(free)
            except Exception:                     # noqa: BLE001 -- older torch
                hw["vram_free"] = hw["vram_total"]
    except ImportError:
        pass
    return hw


# Bytes of device memory per tile pixel per frame, for a whole segment held at once.
#
# Calibrated from one measurement, which is the honest description of it: an A40 held a
# 1827-frame segment of a 416x176 tile in 10 GiB, so 10*1024^3 / (1827*73216) = 80. It
# covers the frames tensor, both flow fields, the completed flows and the propagated
# frames.
#
# One data point is not a model. That is exactly why the OOM fallback in run_propainter
# exists: this estimate is deliberately conservative, and when it is wrong anyway the
# run degrades instead of dying.
BYTES_PER_PIXEL_FRAME = 80

# Fraction of free device memory to plan for. The rest absorbs allocator fragmentation
# and anything else sharing the card.
VRAM_BUDGET = 0.7
# Same idea for host RAM, which holds the in-flight frame queues rather than tensors.
RAM_BUDGET = 0.25


def auto_segment(tile_area: int, hw: dict, overlap: int,
                 ceiling: int = 2000) -> tuple[int, str]:
    """Pick a segment length from the hardware. Returns (frames, why).

    Two limits, and the reason for each:

    - **Device memory.** The model materialises a whole segment as tensors before it
      starts, so this is the limit that OOMs.
    - **Host memory.** Decoded tile frames sit in two bounded queues either side of the
      model, so peak host use is about 2 * (segment + 2*overlap) * tile bytes.

    The ceiling is not a memory limit. Past a couple of thousand frames a longer segment
    buys nothing -- inference is already chunked internally by --pp-subvideo -- while a
    single failure costs more work to redo.
    """
    if not tile_area:
        return ceiling, "no tile area to reason about"

    # Only limits that were actually measured go in here. Defaulting an unmeasured
    # limit to the ceiling and then attributing the result to it is how a CPU-only box
    # ended up reporting "bound by VRAM".
    limits: dict[str, int] = {f"the {ceiling}-frame ceiling": ceiling}
    reasons = []

    if hw.get("vram_free"):
        by_vram = max(32, int(hw["vram_free"] * VRAM_BUDGET
                              // (BYTES_PER_PIXEL_FRAME * tile_area)))
        limits["VRAM"] = by_vram
        reasons.append(f"{_sizeof(hw['vram_free'])} free VRAM -> {by_vram}")

    avail = hw.get("ram_avail") or hw.get("ram_total") or 0
    if avail:
        per_frame = 2 * tile_area * 3          # two queues, bgr24
        by_ram = max(32, int(avail * RAM_BUDGET // per_frame) - 2 * overlap)
        limits["RAM"] = by_ram
        reasons.append(f"{_sizeof(avail)} available RAM -> {by_ram}")

    which = min(limits, key=lambda k: limits[k])
    chosen = max(32, limits[which])
    detail = f"{', '.join(reasons)}; " if reasons else ""
    return chosen, f"{detail}bound by {which}"


def _is_oom(exc: BaseException) -> bool:
    """Is this an out-of-memory failure, whatever torch version raised it?

    Matched on the message as well as the type: torch.cuda.OutOfMemoryError only exists
    from 1.13, and the CPU path raises a plain RuntimeError.
    """
    if exc.__class__.__name__ in ("OutOfMemoryError", "CudaOutOfMemoryError"):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in text or "cuda oom" in text)


def _inpaint_or_split(worker, block: list, mask, say, depth: int = 0):
    """Repair a block, halving it if the device runs out of memory.

    The automatic segment size is extrapolated from a single measurement on one card,
    so it will sometimes be too big. Without this, being wrong means the run dies after
    however long it had been going -- with it, being wrong costs one wasted attempt and
    a smaller segment.

    Splitting is not free and is not silent. The two halves each keep the other's
    adjacent frames as context, so the join has two-sided context like any other
    segment boundary, but their global reference frames are drawn from half the range,
    which is a small quality cost. It is logged every time so a run that is quietly
    doing this all the way through is visible as such -- the fix for that is a smaller
    --pp-segment, not this fallback.
    """
    import numpy as np                       # local: keeps the module import light

    try:
        return worker.inpaint(np.stack(block), mask)
    except BaseException as exc:             # noqa: BLE001 -- re-raised unless OOM
        if not _is_oom(exc) or len(block) < 32 or depth >= 3:
            raise
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        half = len(block) // 2
        ctx = min(16, half)
        say(f"[pp]   out of memory on {len(block)} frames -- retrying as "
            f"{half} + {len(block) - half} with {ctx} frames of context across the "
            f"join. Set --pp-segment lower to avoid the wasted attempt.")
        first = _inpaint_or_split(worker, block[:half + ctx], mask, say, depth + 1)
        second = _inpaint_or_split(worker, block[half - ctx:], mask, say, depth + 1)
        # Drop each half's context so the result lines up with `block` one for one.
        return np.concatenate([first[:half], second[ctx:]])


def _shot_starts(ffmpeg: str, src: Path, threshold: float, fps: float) -> list[int]:
    """Frame indices where a new shot begins, via ffmpeg's scene score.

    One extra decode of the video, which is affordable here only because decoding is a
    small slice of this engine's cost: measured 7% against 77% for the model. It is not
    affordable at all in the un-blend path, which is why this lives here.
    """
    if threshold <= 0:
        return []
    res = subprocess.run(
        [ffmpeg, "-v", "error", "-nostdin", "-i", str(src),
         "-vf", f"select='gt(scene,{threshold})',metadata=print:file=-",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True)
    out = set()
    for line in (res.stdout + res.stderr).splitlines():
        if "pts_time:" in line:
            try:
                out.add(int(round(float(line.split("pts_time:")[1].split()[0]) * fps)))
            except (IndexError, ValueError):
                continue
    return sorted(out)


def _segment_plan(shot_starts: list[int], total: int, segment: int,
                  overlap: int, min_shot: int) -> list[tuple[int, int, int, int]]:
    """Body ranges plus how much context each may take, as (start, end, left, right).

    Segments stop at shot boundaries. This is the fix for a measured defect: a 30-frame
    shot inside a 440-frame segment had almost every one of its global reference frames
    land in a different scene, and the optical flow across the cut bounding it is
    meaningless, so the model filled the watermark region with content from the wrong
    shot for the whole of that shot. Confining a segment to one shot makes every
    reference and every flow vector same-shot by construction.

    Context is clipped at the shot too, for the same reason: 20 frames of "context"
    from the previous scene is worse than none.

    Shots shorter than `min_shot` are merged forwards instead of being handed over
    alone, because below roughly a dozen frames the model has nothing to propagate
    from and would produce a worse result than a slightly impure segment. That merge
    is a compromise and it is logged rather than hidden.
    """
    bounds = [0]
    for s in shot_starts:
        if 0 < s < total and s - bounds[-1] >= min_shot:
            bounds.append(s)
    if bounds[-1] != total:
        # A trailing shot shorter than min_shot is absorbed by the one before it.
        if total - bounds[-1] < min_shot and len(bounds) > 1:
            bounds[-1] = total
        else:
            bounds.append(total)

    plan = []
    for a, b in zip(bounds, bounds[1:]):
        for s in range(a, b, segment):
            e = min(b, s + segment)
            plan.append((s, e, min(overlap, s - a), min(overlap, b - e)))
    return plan


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

    hw = _hardware()
    if opts.segment is None:
        segment, why = auto_segment(tile.w * tile.h, hw, opts.overlap)
        say(f"[pp] hardware: {hw['cpu']} cpu, {_sizeof(hw['ram_total'])} RAM"
            + (f", {hw['gpu']} {_sizeof(hw['vram_total'])}" if hw["gpu"] else ", no GPU"))
        say(f"[pp] segment {segment} frames, chosen automatically ({why}). "
            f"Pin it with --pp-segment.")
    else:
        segment = opts.segment
        say(f"[pp] segment {segment} frames (pinned via --pp-segment)")
    # Everything below reads opts.segment, so resolve it once rather than threading a
    # second variable through the plan and the tail loop.
    opts = replace(opts, segment=segment)

    if opts.workers > 1:
        say(f"[pp] note: --pp-workers {opts.workers} is ignored. One resident model "
            f"processes segments in order; decode and composite already run "
            f"alongside it. See ProPainterOpts.workers.")

    t_start = time.monotonic()
    t0 = time.monotonic()
    worker, reused = _load_worker(opts)
    t_load = time.monotonic() - t0
    if reused:
        say(f"[pp] models already resident, reused in {_hms(t_load)} "
            f"(loading is paid once per process, not once per file)")
    else:
        say(f"[pp] models loaded in {_hms(t_load)} (once for this process -- not per "
            f"segment, and not per file in a batch)")

    t0 = time.monotonic()
    shot_starts = _shot_starts(ffmpeg, src, opts.scene_threshold, float(info.fps))
    t_cuts = time.monotonic() - t0
    if opts.scene_threshold > 0:
        say(f"[pp] scene detection: {len(shot_starts)} cut(s) found in {_hms(t_cuts)} "
            f"(threshold {opts.scene_threshold})")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wmrm-pp-") as td:
        work = Path(td)
        # The only two files this path writes. Everything else stays in memory.
        alpha_png = work / "alpha.png"
        cv2.imwrite(str(alpha_png), (region.alpha[:, :, 0] * 255).astype(np.uint8))

        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=dst.parent, prefix=f".{dst.stem}.", suffix=dst.suffix or ".mp4")
        os.close(tmp_fd)
        tmp = Path(tmp_name)

        dec = subprocess.Popen(_decode_cmd(ffmpeg, src, tile),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               bufsize=0)
        enc = subprocess.Popen(_composite_cmd(ffmpeg, src, tmp, info, tile,
                                              alpha_png, encode),
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                               bufsize=0)
        assert dec.stdout is not None and enc.stdin is not None

        # Bounded queues are the backpressure. One segment of lookahead on each side
        # is enough to keep every stage busy; more would only raise peak memory.
        depth = opts.segment + 2 * opts.overlap
        q_in: queue.Queue = queue.Queue(maxsize=depth)
        q_out: queue.Queue = queue.Queue(maxsize=depth)
        DONE = object()

        def pump_decode() -> None:
            try:
                for frame in _decode_frames(dec.stdout, tile):
                    q_in.put(frame)
            finally:
                q_in.put(DONE)

        written = 0

        def pump_encode() -> None:
            nonlocal written
            while True:
                item = q_out.get()
                if item is DONE:
                    return
                enc.stdin.write(item.data)      # .data: no tobytes() copy
                written += 1

        t_decode = _Stage(pump_decode, "wmrm-decode")
        t_encode = _Stage(pump_encode, "wmrm-encode")
        t_decode.start()
        t_encode.start()

        def next_frame():
            """One frame from the decoder, or None at end of stream."""
            if t_encode.error is not None:
                raise t_encode.error
            item = q_in.get()
            return None if item is DONE else item

        def emit(frame: np.ndarray) -> None:
            """Hand one repaired frame to the compositor, in order.

            The timeout is not belt-and-braces. If the compositing ffmpeg dies, the
            encode thread stops draining this queue, and a plain blocking put() would
            wait on it forever -- a hang instead of the error that caused it. Waking
            up to re-check gives us the real exception.
            """
            while True:
                if t_encode.error is not None:
                    raise t_encode.error
                try:
                    q_out.put(frame, timeout=5.0)
                    return
                except queue.Full:
                    continue

        n = 0                  # frames read
        emitted = 0            # frames handed to the compositor
        nonfinite_total = 0
        nonfinite_at: set[int] = set()
        model_frames = 0       # frames the model actually ran on, context included
        n_seg = 0
        eof = False
        hist: list[np.ndarray] = []      # last `overlap` emitted frames, for left context
        buf: list[np.ndarray] = []       # decoded, not yet emitted; buf[0] is frame `emitted`
        t_model = 0.0
        est_total = info.nframes or 0
        broken = False

        plan = _segment_plan(shot_starts, est_total or 1 << 30, opts.segment,
                             opts.overlap, opts.min_shot)
        if shot_starts:
            lens = [e - s for s, e, _, _ in plan]
            say(f"[pp] {len(shot_starts)} scene cut(s) -> {len(plan)} segment(s), "
                f"{min(lens)}-{max(lens)} frames each; no segment spans a cut")
        else:
            say(f"[pp] no scene cuts found -> {len(plan)} fixed segment(s) of "
                f"{opts.segment}")

        try:
            for seg_i, (s_abs, e_abs, lcap, rcap) in enumerate(plan):
                if eof and not buf:
                    break
                # Frames still needed: this body, plus its right-hand context. Those
                # context frames are the next segment's body, so they stay in `buf`.
                want = (e_abs - s_abs) + rcap
                while not eof and len(buf) < want:
                    f = next_frame()
                    if f is None:
                        eof = True
                        break
                    buf.append(f)
                    n += 1
                if not buf:
                    break

                # min(): the plan is built from probe's frame count, which can be an
                # estimate. A short final segment is normal, not an error.
                body_len = min(e_abs - s_abs, len(buf))
                ahead_len = min(rcap, len(buf) - body_len)
                left = hist[-lcap:] if lcap else []
                body = buf[:body_len]
                ahead = buf[body_len:body_len + ahead_len]

                block = left + body + ahead
                n_seg += 1
                say(f"[pp] segment {n_seg}/{len(plan)}: frames {s_abs}-{s_abs + body_len - 1} "
                    f"(+{len(left)}/{len(ahead)} context)"
                    + (f", {n}/{est_total} decoded" if est_total else ""))

                ts = time.monotonic()
                out = _inpaint_or_split(worker, block, region.inpaint_mask, say)
                t_model += time.monotonic() - ts
                model_frames += len(block)

                # The model diverging is not a crash and does not change the frame
                # count, so without this it reaches the output as a wrong-coloured
                # patch in a run whose own checks all pass.
                bad = getattr(worker, "nonfinite", 0)
                if bad:
                    frames_hit = sorted(getattr(worker, "nonfinite_frames", ()))
                    # `emitted` is the absolute index of body[0]: it counts frames
                    # already sent, and body[0] is the next one to send. Block index i
                    # therefore sits at emitted - len(left) + i.
                    absolute = [emitted - len(left) + i for i in frames_hit]
                    nonfinite_total += bad
                    nonfinite_at.update(absolute)
                    say(f"[pp]   WARNING the model produced {bad} non-finite value(s) "
                        f"in this segment, affecting frames "
                        f"{absolute[:6]}{'...' if len(absolute) > 6 else ''}. Those "
                        f"pixels are arbitrary bytes. Try --no-fp16.")

                if out.shape[0] != len(block):
                    raise ProPainterError(
                        f"segment {n_seg} returned {out.shape[0]} frames, expected "
                        f"{len(block)}; refusing to guess how they line up")
                if out.shape[1:3] != (tile.h, tile.w):
                    raise ProPainterError(
                        f"repaired tile is {out.shape[2]}x{out.shape[1]}, expected "
                        f"{tile.w}x{tile.h}")

                # Emit only the body, in order. Frame j is produced by exactly one
                # segment, so ordering here is positional and not a race.
                for i in range(len(left), len(left) + len(body)):
                    emit(np.ascontiguousarray(out[i]))
                    emitted += 1

                el = time.monotonic() - t_start
                rate = emitted / max(t_model, 1e-6)
                eta = ((est_total - emitted) / rate) if rate and est_total > emitted else 0.0
                say(f"[pp]   {_hms(time.monotonic() - ts)}  "
                    f"{rate:.2f} fps model" + (f"  eta {_hms(eta)}" if eta else "")
                    + f"  elapsed {_hms(el)}")

                # Keep the tail of what was just emitted as the next segment's left
                # context. `lcap` on the next segment is what stops it reaching back
                # across a cut, so nothing here needs to know where the shots are.
                if opts.overlap:
                    hist = (hist + body)[-opts.overlap:]
                buf = buf[body_len:]
                if eof and not buf:
                    break

            # probe's frame count can undercount; anything left is processed rather
            # than dropped, on fixed segments, and said out loud because it means the
            # cut-aware plan did not cover the whole clip.
            while buf or not eof:
                while not eof and len(buf) < opts.segment + opts.overlap:
                    f = next_frame()
                    if f is None:
                        eof = True
                        break
                    buf.append(f)
                    n += 1
                if not buf:
                    break
                body_len = min(opts.segment, len(buf))
                ahead_len = min(opts.overlap, len(buf) - body_len)
                left = hist[-opts.overlap:] if opts.overlap else []
                body, ahead = buf[:body_len], buf[body_len:body_len + ahead_len]
                n_seg += 1
                say(f"[pp] segment {n_seg} (beyond the planned {len(plan)}): "
                    f"{body_len} frames -- the source reported {est_total} frames but "
                    f"has more, so these are cut on size, not on shots")
                ts = time.monotonic()
                out = _inpaint_or_split(worker, left + body + ahead,
                                        region.inpaint_mask, say)
                t_model += time.monotonic() - ts
                model_frames += len(left) + body_len + ahead_len
                for i in range(len(left), len(left) + body_len):
                    emit(np.ascontiguousarray(out[i]))
                    emitted += 1
                if opts.overlap:
                    hist = (hist + body)[-opts.overlap:]
                buf = buf[body_len:]
        except BrokenPipeError:
            broken = True
        finally:
            q_out.put(DONE)
            t_encode.join(timeout=60)
            try:
                if enc.stdin and not enc.stdin.closed:
                    enc.stdin.close()
            except BrokenPipeError:
                broken = True

            # Order matters. Reading the decoder's stderr waits for EOF, which only
            # arrives when it exits -- and it cannot exit while it is blocked writing
            # frames nobody is draining any more. Stop it first, then read.
            if dec.poll() is None:
                dec.kill()
            dec_err = drain(dec.stderr)
            dec.wait()
            enc_err = drain(enc.stderr)
            enc.wait()

        if t_decode.error is not None:
            tmp.unlink(missing_ok=True)
            raise ProPainterError(f"decoding the tile failed: {t_decode.error}\n"
                                  f"{dec_err.strip()[:800]}")
        if broken:
            tmp.unlink(missing_ok=True)
            raise EncodeError("the compositing ffmpeg exited while frames were still "
                              f"being written:\n{enc_err.strip()[:800]}")
        if n == 0:
            tmp.unlink(missing_ok=True)
            raise ProPainterError(f"decoded 0 tile frames from {src}\n"
                                  f"{dec_err.strip()[:600]}")
        if written != n:
            tmp.unlink(missing_ok=True)
            raise ProPainterError(
                f"wrote {written} frames but decoded {n}; refusing to ship a video "
                "with frames missing or duplicated")
        if enc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise EncodeError(f"compositing failed:\n{enc_err.strip()[:800]}")

        os.replace(tmp, dst)          # atomic

    total = time.monotonic() - t_start
    if nonfinite_total:
        say(f"[pp] WARNING {nonfinite_total} non-finite model output value(s) across "
            f"{len(nonfinite_at)} frame(s): {sorted(nonfinite_at)[:12]}"
            f"{'...' if len(nonfinite_at) > 12 else ''}")
        say(f"[pp]         Those became arbitrary bytes in the repaired region. This "
            f"is upstream behaviour, not silently corrected here. Re-run with "
            f"--no-fp16; if it persists in fp32 the model is diverging on this clip.")
    waste = model_frames - n
    pct = lambda x: 100 * x / max(total, 1e-6)          # noqa: E731
    say(f"[pp] {n} frames repaired in {_hms(t_model)} "
        f"({n / max(t_model, 1e-6):.2f} fps model-only)"
        + (f"  |  {model_frames} frames computed, {waste} discarded as segment "
           f"context ({100 * waste / model_frames:.1f}% wasted)" if waste else
           "  |  no context discarded"))
    # Only `model` responds to the ProPainter settings or to a faster card. Loading is
    # now a one-off rather than per segment, and it is printed separately so that
    # stops being invisible. Decode and composite no longer have a slice of their own:
    # they run in their own threads alongside the model, which is the point -- what is
    # left of them shows up as the gap between model time and total.
    rest = max(0.0, total - t_model - t_load - t_cuts)
    say(f"[pp] TIME  load {_hms(t_load)} ({pct(t_load):.0f}%)  "
        f"cuts {_hms(t_cuts)} ({pct(t_cuts):.0f}%)  "
        f"model {_hms(t_model)} ({pct(t_model):.0f}%)  "
        f"decode+composite, overlapped {_hms(rest)} ({pct(rest):.0f}%)")
    say(f"[pp] TOTAL {_hms(total)} for {n} frames "
        f"({n / max(total, 1e-6):.2f} fps, "
        f"{total / max(info.duration, 1e-6):.1f}x realtime)  "
        f"peak in-flight {_sizeof(2 * (opts.segment + 2 * opts.overlap) * tile.w * tile.h * 3)}"
        f" of frames, flat in video length")
    # Extrapolate, because a one-minute test says nothing about the videos this is
    # actually for until it is scaled up.
    say(f"[pp] EXTRAPOLATED  1 hour of this footage -> "
        f"~{_hms(t_load + 3600.0 * t_model / max(info.duration, 1e-6))} "
        f"(the load cost does not scale with length)")
    return region
