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

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .pipeline import EncodeOpts, EncodeError, drain, read_exact
from .probe import ProbeError, probe, require_tools
from .region import Region, build_region

# Frames per model invocation. Bounded by the fact that the model materialises every
# frame it is given as a tensor before it starts. Now that no frames touch the disk
# this is purely a VRAM/RAM knob.
SEGMENT = 400
# Frames of context added to each side of a segment and then thrown away. Without it
# the first and last frames of a segment have one-sided temporal context and the
# joins show.
OVERLAP = 20
# Frames per composited part, and so the granularity a killed run resumes at.
# Deliberately independent of the model plan: two runs with different --pp-segment
# have to produce identical parts (tests/test_video_order.py pins that invariance),
# and a resumed run has to produce the same parts as the run it continues. 3600 is
# two minutes of 30fps footage -- small enough that a crash costs little, large
# enough that per-part ffmpeg startup stays in the noise.
PART_FRAMES = 3600
PART_NAME = "part-{:06d}.mp4"
PART_GLOB = "part-*.mp4"
MANIFEST = "manifest.json"
MANIFEST_VERSION = 1
# How the final flush is watched. The compositor is given as long as it needs, and is
# declared stuck only when it has not accepted a single frame for STALL_LIMIT. This is
# a liveness check, not a budget: the thing it has to tolerate is one whole segment
# arriving at once with the model no longer running to hide the encode behind.
STALL_POLL = 5.0
STALL_LIMIT = 300.0

# Black runs are the other half of "no segment spans a cut", and the half the scene
# score cannot see. A fade through black is gradual by definition, so it never scores:
# measured on a real intro (a black frame, a rating card, then a fade into the first
# shot), `select='gt(scene,X)'` found nothing at all in the first 11.7s at any
# threshold down to 0.1, while the picture went from pure black to full brightness at
# frame 246. The black run and the shot after it therefore shared one segment, and the
# model filled the hole on the black frames from the bright shot's reference frames:
# a glowing smear over an otherwise black frame. It started at frame ~216, which is
# ref_stride * ref_num / 2 = 40 frames ahead of the picture -- exactly the reach of
# ProPainter's global references. blackdetect finds those boundaries for the price of
# one more filter in a decode that was already happening.
BLACK_PIX_TH = 0.10       # per-pixel luma, 0-1, below which a pixel counts as black
BLACK_PIC_TH = 0.98       # fraction of the frame that has to be that dark

# The dark-tile guard, which is the same defect caught one stage later and without
# depending on any boundary being found. See _dark_guard.
DARK_TILE_MAX = 24        # 99th-percentile luma around the hole, out of 255
DARK_FILL_SLACK = 8       # how far above that a fill may sit before it is distrusted


class ProPainterError(RuntimeError):
    pass


def _join_parts(ffmpeg: str, parts_dir: Path, work: Path, src: Path, dst: Path,
                info, encode: EncodeOpts, expect: int, part_frames: int, say) -> Path:
    """Concatenate the parts into one file and prove it holds every frame.

    This is the only step that is not resumable, and it does not need to be: it is a
    stream copy, minutes of I/O against hours of encoding.
    """
    parts = sorted(parts_dir.glob(PART_GLOB))
    if not parts:
        raise EncodeError(f"no parts to join in {parts_dir}")
    for i, p in enumerate(parts):
        # A gap here would produce a video that is complete-looking and missing a
        # couple of minutes out of the middle, which is the worst way for this to
        # fail: nothing downstream would notice.
        if p.name != PART_NAME.format(i):
            raise EncodeError(f"the parts in {parts_dir} are not a contiguous run: "
                              f"expected {PART_NAME.format(i)}, found {p.name}")

    # Each part's length is stated here rather than left for the concatenator to read
    # back out of the container, and this is not belt-and-braces. Every part is
    # exactly `part_frames` frames long except the last, so the exact duration is
    # arithmetic -- while the number in the file has been through a container's
    # timebase and may come back rounded. Measured: on ffmpeg 6.1.1 reading it back
    # was exact, on another build five 12-frame parts joined 2.4ms long, which
    # ffprobe then read as 359/12 fps instead of 30000/1001. On a feature-length file
    # that is audio drift and a `frame rate` failure from `wmrm verify`, from a
    # rounding error nobody can see in any single part.
    lines = []
    for i, p in enumerate(parts):
        lines.append("file '{}'\n".format(str(p.resolve()).replace("'", r"'\''")))
        held = min(part_frames, expect - i * part_frames) if expect else 0
        if held > 0:
            lines.append(f"duration {float(Fraction(held) / info.fps):.9f}\n")
    listing = work / "parts.txt"
    listing.write_text("".join(lines))
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=dst.parent, prefix=f".{dst.stem}.", suffix=dst.suffix or ".mp4")
    os.close(tmp_fd)
    tmp = Path(tmp_name)

    say(f"[pp] joining {len(parts)} part(s) and putting the audio back -- stream "
        f"copy, nothing is re-encoded")
    res = subprocess.run(_assemble_cmd(ffmpeg, listing, src, tmp, info, encode),
                         capture_output=True, text=True)
    if res.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise EncodeError(f"joining the parts failed:\n{_first_error(res.stderr)}\n"
                          f"The parts are still in {parts_dir} -- re-running with "
                          f"--resume retries just this step.")
    try:
        got = probe(tmp).nframes
    except ProbeError:
        got = 0
    if expect and got != expect:
        tmp.unlink(missing_ok=True)
        raise EncodeError(
            f"the joined video holds {got or 'an unreadable number of'} frames, "
            f"expected {expect}; refusing to ship a video with frames missing or "
            f"duplicated. The parts are still in {parts_dir}.")
    return tmp


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
    # Frames per composited part, which is the granularity a killed run resumes at.
    # Nothing about the output depends on it -- see PART_FRAMES -- so it is a
    # crash-cost knob, not a quality one.
    part_frames: int = PART_FRAMES
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
    # Also end a segment where the picture goes to black or comes back from it. This is
    # not a second opinion on the scene score, it is the transition the scene score
    # provably cannot see -- see BLACK_PIX_TH. It costs one extra filter in the scene
    # scan, no extra decode. False falls back to scene cuts alone.
    black_cuts: bool = True
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


def _decode_cmd(ffmpeg: str, src: Path, tile, start: int = 0,
                fps: Fraction | None = None) -> list[str]:
    """Decode the source from frame `start` and emit only the tile, as raw frames.

    Cropping in ffmpeg rather than in numpy is what keeps this cheap: a 1080p frame
    is 6 MB and a 400x168 tile is 200 KB, so the pipe carries 3% of the data and the
    Python side never sees a full frame.

    `start` is what stops a resumed run from decoding hours of video it already has
    parts for. Seeking costs a keyframe of pre-roll and nothing else.
    """
    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if start and fps is not None:
        cmd += ["-ss", _seek_arg(start, fps)]
    return cmd + ["-i", str(src),
                  "-vf", f"crop={tile.w}:{tile.h}:{tile.x}:{tile.y}",
                  "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]


def _seek_arg(start: int, fps: Fraction) -> str:
    """Where to seek so that frame `start` is the first one out.

    Half a frame early, deliberately. Input `-ss` is frame-accurate -- ffmpeg seeks
    to the keyframe before the target and decodes forward, discarding frames whose
    timestamp is below it -- so the only way to land on the wrong frame is for the
    target to fall on the boundary and rounding to push it across. Aiming at the
    midpoint between frame start-1 and frame start leaves half a frame of margin
    either way, against a value written to microsecond precision. tests/test_resume.py
    is what proves this lands where it claims: a one-frame error there shows up as a
    diff between a resumed run and a whole one.
    """
    return f"{max(0.0, float((Fraction(start) - Fraction(1, 2)) / fps)):.6f}"


def _timescale(fps: Fraction) -> int:
    """A timebase in which one frame is a whole number of ticks.

    This is not a detail. A part whose duration is stored rounded is a part the
    concatenator places the next one slightly wrong against, and the error is
    cumulative: measured, five 12-frame parts written at Matroska's default
    millisecond resolution came back as 30fps/2.000s from a 30000/1001/2.002s source,
    because each part lost 0.4ms. Over a feature-length file that is both audio drift
    and a `frame rate` failure from `wmrm verify`. Ticking at a multiple of the rate's
    own numerator makes every frame exactly `denominator` ticks, for any rational
    rate, so nothing is ever rounded.
    """
    scale = fps.numerator
    while scale < 10000:
        scale *= 10
    return scale


def _part_cmd(ffmpeg: str, src: Path, out: Path, info, tile, alpha_png: Path,
              encode: EncodeOpts, start: int, count: int) -> list[str]:
    """Overlay one part's worth of repaired tile onto the untouched source.

    Input 0 is the original, seeked to this part's first frame; input 1 is our
    repaired tile arriving on stdin; input 2 is the feather alpha as a still.

    Video only. Audio and metadata are attached once at assembly, from the source, so
    a part carries nothing that has to be reconciled with its neighbours.
    """
    filt = (
        f"[2:v]format=gray,scale={tile.w}:{tile.h}[m];"
        f"[1:v][m]alphamerge[ba];"
        f"[0:v][ba]overlay={tile.x}:{tile.y}:format=auto:shortest=1[out]"
    )
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-y"]
    if start:
        cmd += ["-ss", _seek_arg(start, info.fps)]
    cmd += [
        "-i", str(src),
        # The tile stream is timed to the source's exact rational rate, not a rounded
        # decimal, or the overlay drifts against the video it sits on.
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{tile.w}x{tile.h}", "-r", str(info.fps),
        "-i", "-",
        "-loop", "1", "-i", str(alpha_png),
        "-filter_complex", filt,
        "-map", "[out]", "-an",
        # Two independent bounds on the length, because they fail differently:
        # `shortest=1` ends the graph when our tile stream does, `-frames:v` stops it
        # even if the graph somehow outlives that.
        "-frames:v", str(count),
        "-c:v", "libx264", "-crf", str(encode.crf), "-preset", encode.x264_preset,
        "-pix_fmt", "yuv420p",
        "-video_track_timescale", str(_timescale(info.fps)),
        str(out),
    ]
    return cmd


def _assemble_cmd(ffmpeg: str, listing: Path, src: Path, dst: Path, info,
                  encode: EncodeOpts) -> list[str]:
    """Join the parts and put the audio and metadata back, by stream copy.

    Nothing is re-encoded here: the video is already exactly what it will ship as,
    and the audio has never been touched. On a feature-length 4K file this is minutes
    of I/O against hours of encoding, which is the whole reason the parts are encoded
    in their final form rather than as an intermediate.
    """
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-i", str(src),
        # No -shortest: it ends the file when the shortest stream ends, and an audio
        # track 2.6s shorter than the video silently cost 78 frames off the tail --
        # the `duration 60.96s vs 58.36s` verify failure.
        "-map", "0:v:0", "-map", "1:a:0?", "-map_metadata", "1",
        "-c", "copy",
        "-video_track_timescale", str(_timescale(info.fps)),
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


def _fingerprint(src: Path, info, region, opts, encode, part_frames: int) -> dict:
    """What has to match for parts from an earlier run to still be usable.

    Everything that decides an output pixel, and nothing that does not. The source is
    identified by size and mtime rather than by a content hash: hashing 15 GB to
    answer "is this the same file" costs minutes on every resume, and the only case it
    catches beyond this is someone substituting a different file of exactly the same
    size at exactly the same mtime.

    Note what is *in* here. `segment` and `overlap` change which frames the model sees
    together and therefore change pixels, so a resume with a different segment size is
    refused rather than silently producing a video whose halves were made differently.
    That is also why a resumed run reuses the recorded segment instead of asking
    `auto_segment` again -- free VRAM at startup is not a property of the video.
    """
    st = src.stat()
    b, t = region.box, region.tile
    return {
        "version": MANIFEST_VERSION,
        "source": {"name": src.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                   "nframes": info.nframes, "fps": str(info.fps),
                   "frame": [info.width, info.height]},
        "region": {"box": [b.x, b.y, b.w, b.h], "tile": [t.x, t.y, t.w, t.h]},
        "model": {"segment": opts.segment, "overlap": opts.overlap,
                  "subvideo_length": opts.subvideo_length,
                  "neighbor_length": opts.neighbor_length,
                  "ref_stride": opts.ref_stride, "raft_iter": opts.raft_iter,
                  "mask_dilation": opts.mask_dilation, "fp16": bool(opts.fp16),
                  "scene_threshold": opts.scene_threshold, "min_shot": opts.min_shot,
                  "black_cuts": bool(opts.black_cuts)},
        "encode": {"crf": encode.crf, "preset": encode.x264_preset},
        "part_frames": part_frames,
    }


def build_manifest(fp: dict, cuts: list, frames: int, part_frames: int) -> dict:
    """What gets written beside the parts.

    `fingerprint` and `cuts` are for resume, and were the whole of this file until the
    server started reporting progress. `frames` and `part_frames` are for that reader: the
    number of parts a run will produce is `ceil(frames / part_frames)` -- the same
    arithmetic `_usable_parts` walks -- and without both numbers written down, the only
    other way to know the total is to decode the source a second time.

    A function rather than a literal at the call site so the two sides of that contract can
    be tested against each other. They were not, and drifted: the reader looked for `frames`
    and `part_frames` while the writer stored neither, so every job reported `partsTotal:
    null` and could never show a fraction or an ETA.
    """
    return {"fingerprint": fp, "cuts": cuts, "frames": frames, "part_frames": part_frames}


def _read_manifest(parts_dir: Path) -> dict | None:
    """The record left by an earlier run, or None if there is nothing to trust.

    Unreadable, malformed and written-by-another-version all mean the same thing here
    -- start over -- because the alternative is guessing at what half a manifest meant.
    """
    try:
        data = json.loads((parts_dir / MANIFEST).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("fingerprint", {}).get("version") != MANIFEST_VERSION:
        return None
    return data


def _part_length(path: Path) -> int:
    """Frames in a finished part, counted from the container, not by decoding.

    `-count_packets` reads the packet headers only: for H.264 one packet is one
    frame, and a 3600-frame 4K part answers in well under a second. This is what
    makes a part trustworthy -- the frames are counted in the file that shipped them,
    not in the pipe that fed it.
    """
    _, ffprobe = require_tools()
    res = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    if res.returncode != 0:
        return -1
    try:
        return int(res.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def _usable_parts(parts_dir: Path, total: int, part_frames: int, say) -> int:
    """How many frames at the head of the video are already composited on disk.

    The longest *unbroken* prefix of complete parts, and only that. A part with the
    wrong frame count ends the prefix rather than being repaired: it is either the
    one the crash interrupted or evidence that something else is wrong, and in both
    cases the cheap, correct answer is to make it again. Parts after a gap are
    discarded for the same reason -- keeping them would mean trusting that the gap is
    the only thing missing.
    """
    done = 0
    for index in range(0, (total + part_frames - 1) // part_frames):
        path = parts_dir / PART_NAME.format(index)
        if not path.exists():
            break
        want = min(part_frames, total - index * part_frames)
        got = _part_length(path)
        if got != want:
            say(f"[pp] resume: {path.name} holds {got} frames, expected {want} -- "
                f"redoing it and everything after")
            break
        done += 1
    keep = done * part_frames
    for stale in sorted(parts_dir.glob(PART_GLOB))[done:]:
        stale.unlink(missing_ok=True)
    return min(keep, total)


class _PartSink:
    """Composites repaired tile frames into fixed-size, self-contained parts.

    One ffmpeg per part, seeked to that part's first frame. A part on disk with the
    right frame count is finished for good, so the next run starts at the first frame
    that has no part -- which is the whole of the resume mechanism.

    It also removes the failure this replaced. The compositor used to be a single
    ffmpeg fed for nine hours, and anything that killed it, or killed the pipe into
    it, discarded every frame it had encoded.
    """

    def __init__(self, *, ffmpeg: str, src: Path, parts_dir: Path, info, tile,
                 alpha_png: Path, encode: EncodeOpts, total: int, part_frames: int,
                 first: int, say) -> None:
        self.ffmpeg, self.src, self.parts_dir = ffmpeg, src, parts_dir
        self.info, self.tile, self.alpha_png = info, tile, alpha_png
        self.encode, self.total, self.part_frames = encode, total, part_frames
        self.say = say
        self.index = first          # absolute index of the next frame to write
        self.written = 0            # frames written by this run, for the stall watch
        self.parts: list[tuple[int, int]] = []   # (part index, frames written)
        self._proc: subprocess.Popen | None = None
        self._part = -1
        self._count = 0

    def write(self, frame) -> None:
        part = self.index // self.part_frames
        if part != self._part:
            self.close()
            self._open(part)
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(frame.data)       # .data: no tobytes() copy
        self.index += 1
        self.written += 1
        self._count += 1

    def _open(self, part: int) -> None:
        start = part * self.part_frames
        want = min(self.part_frames, max(0, self.total - start)) or self.part_frames
        out = self.parts_dir / PART_NAME.format(part)
        self._proc = subprocess.Popen(
            _part_cmd(self.ffmpeg, self.src, out, self.info, self.tile,
                      self.alpha_png, self.encode, start, want),
            stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self._part, self._count = part, 0

    def close(self) -> None:
        """Finish the open part and prove it holds what was fed into it."""
        if self._proc is None:
            return
        proc, part, count = self._proc, self._part, self._count
        self._proc, self._part, self._count = None, -1, 0
        path = self.parts_dir / PART_NAME.format(part)
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except BrokenPipeError:
            pass
        # Wait first, with a bound, then read stderr. An ffmpeg that will not exit is
        # a real possibility here and it must not become a hang: this is a subprocess
        # whose stdin has just been closed, so it has everything it is ever going to
        # get. Reading stderr first would block on the same wait with no timeout at
        # all. Safe in this order only because -v error keeps stderr far below the
        # pipe buffer -- a chattier ffmpeg would deadlock on the buffer instead.
        try:
            proc.wait(timeout=STALL_LIMIT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise EncodeError(
                f"the ffmpeg compositing {path.name} did not exit within "
                f"{_hms(STALL_LIMIT)} of being given its last frame; killed it\n"
                f"{drain(proc.stderr).strip()[:800]}")
        err = drain(proc.stderr)
        if proc.returncode != 0:
            path.unlink(missing_ok=True)
            raise EncodeError(f"compositing {path.name} failed (exit "
                              f"{proc.returncode}) after {count} frames:\n"
                              f"{err.strip()[:800]}")
        got = _part_length(path)
        if got != count:
            path.unlink(missing_ok=True)
            raise EncodeError(f"{path.name} holds {got} frames but {count} were "
                              f"written into it; refusing to build a video out of "
                              f"parts that do not hold what they were given\n"
                              f"{err.strip()[:400]}")
        self.parts.append((part, count))

    def abandon(self) -> None:
        """Drop the part being written. Called when the run is already failing.

        The open part is incomplete by definition, and an incomplete part that looks
        finished is the one thing a resume must never find.
        """
        if self._proc is None:
            return
        proc, part = self._proc, self._part
        self._proc, self._part, self._count = None, -1, 0
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        drain(proc.stderr)
        proc.wait()
        (self.parts_dir / PART_NAME.format(part)).unlink(missing_ok=True)


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


def _dark_guard(src_tile: np.ndarray, out_tile: np.ndarray, hole: np.ndarray,
                ring: np.ndarray) -> np.ndarray | None:
    """Refuse a fill that is brighter than the picture it sits in, where there is none.

    ProPainter fills the hole from *other frames*, and in a black shot there are none
    to fill from: the region is masked in every frame of it and nothing moves, so
    flow-guided propagation contributes nothing and the entire fill is the
    transformer's invention. An invention is not black. Against a black frame that is a
    glowing smear exactly the size of the hole -- measured at 10/255 on a 1080p intro,
    plainly visible at 4K -- and the same error over real footage is invisible, which is
    why it only ever shows up in an intro.

    Segmenting on black runs stops the worst of it by keeping bright reference frames
    out of the segment, but it depends on a boundary being found. This does not: the
    test is the one thing that is true whatever the model did, that a fill can be no
    brighter than its surroundings. Where the tile around the hole holds no picture at
    all, the hole gets the median of that surround instead of the invention -- black on
    a black frame, and the watermark is just as gone.

    Deliberately narrow. It fires only when there is provably nothing to inpaint from,
    never to second-guess a fill that has real content around it, because the median of
    a real surround would be a flat patch where the model had a picture.

    Returns the corrected tile, or None to keep the model's answer.
    """
    around = src_tile[ring]
    if around.size == 0:
        return None
    if float(np.percentile(around.max(axis=1), 99)) > DARK_TILE_MAX:
        return None                       # there is a picture here; the model wins
    fill = np.median(around, axis=0)
    if int(out_tile[hole].max()) <= max(DARK_TILE_MAX, int(fill.max()) + DARK_FILL_SLACK):
        return None                       # the fill already sits in the dark surround
    fixed = out_tile.copy()
    fixed[hole] = fill.astype(np.uint8)
    return fixed


def _shot_starts(ffmpeg: str, src: Path, threshold: float, fps: float, *,
                 black_min_frames: int = 0, hwaccel: bool = False,
                 say=lambda m: None) -> tuple[list[int], list[int]]:
    """Frame indices where a new shot begins: (scene cuts, black boundaries).

    Two signals, one decode. The scene score finds hard cuts; `blackdetect` finds the
    frame the picture goes black on and the frame it comes back on, which is the
    transition the scene score cannot see at any threshold -- see BLACK_PIX_TH for the
    measurement and for what it cost. Both filters publish to frame metadata, so the
    second question is answered by one more filter in the graph rather than by a second
    pass over the file. `blackdetect` goes first in the chain because it has to see
    every frame and `select` drops most of them.

    One extra decode of the video, which is affordable here only because decoding is a
    small slice of this engine's cost: measured 7% against 77% for the model. It is not
    affordable at all in the un-blend path, which is why this lives here.

    On a GPU box it decodes through NVDEC. This is the one step where decode is 100% of
    the cost rather than 7%, and `select='gt(scene,...)'` scores each frame against its
    predecessor, so the work is serial no matter how many cores are free -- measured on
    a 128-core EPYC, the CPU path pinned 1.8 cores and left 126 idle. Handing decode to
    the card fixes what the cores could not: measured on 120s of 1080p with a 4090,
    32.9s wall / 54.0s CPU became 6.8s wall / 3.0s CPU, same cuts found. That is 4.8x
    on the clock and 18x less CPU, which for a 2.6-hour film is ~43 minutes down to ~9.

    Two things were measured and rejected before this. Downscaling ahead of the filter
    (what PySceneDetect does) is 0.8x here -- ffmpeg still decodes at full resolution,
    so the scale is pure added work. Splitting the timeline across parallel ffmpegs
    finds identical cuts but only pays when cores are the constraint, and they are not:
    it was 1.2x on a box whose serial pass already used 4.3 of 6 cores.
    """
    if threshold <= 0 and black_min_frames <= 0:
        return [], []

    chain: list[str] = []
    if black_min_frames > 0:
        # A black run shorter than `min_shot` would be merged away by _segment_plan, so
        # asking for it would only produce boundaries nothing can use -- and in a night
        # scene, thousands of them.
        chain += [f"blackdetect=d={black_min_frames / max(fps, 1e-6):.4f}"
                  f":pic_th={BLACK_PIC_TH}:pix_th={BLACK_PIX_TH}",
                  "metadata=print:file=-"]
    if threshold > 0:
        chain += [f"select='gt(scene,{threshold})'", "metadata=print:file=-"]
    vf = ",".join(chain)

    def attempt(extra: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [ffmpeg, "-v", "error", "-nostdin", *extra, "-i", str(src),
             "-vf", vf, "-an", "-f", "null", "-"],
            capture_output=True, text=True)

    res = None
    if hwaccel:
        res = attempt(["-hwaccel", "cuda"])
        if res.returncode != 0:
            # NVDEC refuses some profiles and bit depths outright. Falling back is
            # correct, doing it silently is not: the CPU path is 5x slower and the
            # run would just look mysteriously slow.
            say(f"[pp] NVDEC declined this file, decoding scene cuts on the CPU "
                f"instead (~5x slower): {_first_error(res.stderr)}")
            res = None
    if res is None:
        res = attempt([])

    # A failed decode used to be indistinguishable from a clean scan that found
    # nothing: no pts_time lines either way, so the caller reported "no scene cuts
    # found" and planned fixed segments. That silently discards the cut protection
    # this function exists to provide, which is exactly the case where a segment
    # spans a cut and the watermark region gets filled from the wrong scene.
    if res.returncode != 0:
        raise EncodeError(
            f"scene detection failed to decode {src.name}: "
            f"{_first_error(res.stderr)}\n"
            f"  pass --pp-scene-threshold 0 to skip it, but read what that costs "
            f"in the --pp-scene-threshold help first"
        )

    # `metadata=print` emits a `frame:.. pts_time:..` header and then one line per key,
    # and only for frames that carry any -- so the header alone is not a cut. Reading
    # the key rather than the header is also what keeps the two signals apart now that
    # both are printing into the same pipe.
    scene: set[int] = set()
    black: set[int] = set()
    at: int | None = None
    for line in (res.stdout + res.stderr).splitlines():
        if "pts_time:" in line:
            try:
                at = int(round(float(line.split("pts_time:")[1].split()[0]) * fps))
            except (IndexError, ValueError):
                at = None
        elif "lavfi.scene_score" in line:
            if at is not None:
                scene.add(at)
        elif "lavfi.black_start=" in line or "lavfi.black_end=" in line:
            # The value is the boundary's own timestamp: `black_start` lands on the
            # first black frame, `black_end` on the first frame that is not. Both are
            # where a shot begins, which is what this returns.
            try:
                black.add(int(round(float(line.split("=", 1)[1]) * fps)))
            except (IndexError, ValueError):
                continue
    return sorted(scene), sorted(black)


def _first_error(stderr: str) -> str:
    for line in stderr.splitlines():
        if line.strip():
            return line.strip()[:200]
    return "no error message"


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

    `shot_starts` carries black-run boundaries as well as scene cuts, and they are the
    same thing to this function. They are not the same thing to the defect: a fade
    through black scores below any useful threshold, so before those boundaries existed
    the black intro shared a segment with the bright shot after it -- the worst possible
    version of filling from the wrong scene, because black is the content least like
    anything else. See BLACK_PIX_TH.

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
    resume: bool = False,
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

    parts_dir = dst.with_name(dst.name + ".parts")
    prior = _read_manifest(parts_dir) if resume else None
    if resume and prior is None:
        say(f"[pp] resume: nothing usable in {parts_dir.name}/ -- starting from the "
            f"beginning")

    hw = _hardware()
    if prior is not None and opts.segment is None:
        # Segment size decides which frames the model sees together, so it decides
        # pixels. Asking `auto_segment` again would answer from whatever VRAM happens
        # to be free now, and the second half of the video would be made differently
        # from the first. The recorded value is the one the parts on disk were made
        # with, so it is the only correct answer here.
        segment = int(prior["fingerprint"]["model"]["segment"])
        say(f"[pp] segment {segment} frames, taken from the run being resumed")
    elif opts.segment is None:
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
    # NVDEC whenever there is a card and the user has not asked for CPU. Nothing about
    # the result changes -- it is the same filter over the same frames -- so this is a
    # speed decision only, and _shot_starts falls back on its own if the card refuses.
    use_nvdec = bool(hw["gpu"]) and opts.device != "cpu"
    if prior is not None and prior.get("cuts") is not None:
        # Detection is a whole extra decode of the source -- ~9 minutes for a 2.6-hour
        # film on NVDEC, ~43 on the CPU -- to answer a question about a file that has
        # not changed since the last run answered it. The fingerprint covers the
        # source and the threshold, so a matching manifest means the same cuts.
        shot_starts = [int(c) for c in prior["cuts"]]
        scene_cuts = black_cuts = None
        say(f"[pp] resume: reusing {len(shot_starts)} shot boundary/ies from the "
            f"manifest instead of decoding the source again to find them")
    else:
        looking = ([f"scene cuts (threshold {opts.scene_threshold})"]
                   if opts.scene_threshold > 0 else [])
        if opts.black_cuts:
            looking.append("black runs")
        if looking:
            say(f"[pp] scanning for {' and '.join(looking)} ({info.nframes or '?'} "
                f"frames to decode, the one step here that is pure decode)"
                + (" -- via NVDEC" if use_nvdec else " -- on the CPU"))
        scene_cuts, black_cuts = _shot_starts(
            ffmpeg, src, opts.scene_threshold, float(info.fps),
            black_min_frames=opts.min_shot if opts.black_cuts else 0,
            hwaccel=use_nvdec, say=say)
        shot_starts = sorted(set(scene_cuts) | set(black_cuts))
    _plan_est_total = info.nframes or (1 << 30)
    plan = _segment_plan(shot_starts, _plan_est_total, opts.segment, opts.overlap,
                         opts.min_shot)
    t_cuts = time.monotonic() - t0
    if (scene_cuts is not None and black_cuts is not None
            and (opts.scene_threshold > 0 or opts.black_cuts)):
        say(f"[pp] shot detection: {len(scene_cuts)} scene cut(s) + "
            f"{len(black_cuts)} black boundary/ies -> {len(shot_starts)} boundary/ies "
            f"in {_hms(t_cuts)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    n_total = info.nframes or 0
    fp = _fingerprint(src, info, region, opts, encode, opts.part_frames)
    if prior is not None and prior.get("fingerprint") != fp:
        was = prior.get("fingerprint", {})
        changed = [k for k, v in fp.items() if was.get(k) != v]
        say(f"[pp] resume: the parts in {parts_dir.name} were made with different "
            f"settings ({', '.join(changed)}) -- starting from the beginning")
        prior = None

    if prior is None:
        for stale in parts_dir.glob(PART_GLOB):
            stale.unlink(missing_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / MANIFEST).write_text(
        json.dumps(build_manifest(fp, shot_starts, n_total, opts.part_frames), indent=1))

    # Where this run has to pick up: the first frame with no finished part behind it.
    resume_from = _usable_parts(parts_dir, n_total, opts.part_frames, say) if prior else 0
    # The model restarts at the beginning of the segment that frame belongs to, not at
    # the frame itself. A segment is a pure function of the source frames it spans --
    # `hist` carries decoded frames, never repaired ones -- so recomputing the part of
    # it that is already on disk reproduces it exactly, and starting mid-segment would
    # not: the model would be given a different block and would answer differently.
    seg0, dec_start, pos0 = 0, 0, 0
    if resume_from:
        for i, (s_abs, e_abs, lcap, _r) in enumerate(plan):
            if s_abs <= resume_from < e_abs:
                seg0, dec_start, pos0 = i, max(0, s_abs - lcap), s_abs
                break
        else:
            seg0, dec_start, pos0 = len(plan), resume_from, resume_from
        say(f"[pp] resume: {resume_from} of {n_total or '?'} frames already composited; "
            f"restarting the model at segment {seg0 + 1}/{len(plan)}, frame {pos0}"
            + (f" (redoing {resume_from - pos0} frame(s) that a part already holds, "
               f"because a segment is only reproducible whole)" if resume_from > pos0
               else ""))

    with tempfile.TemporaryDirectory(prefix="wmrm-pp-") as td:
        work = Path(td)
        # The only file this path writes outside the parts directory. Everything else
        # stays in memory.
        alpha_png = work / "alpha.png"
        cv2.imwrite(str(alpha_png), (region.alpha[:, :, 0] * 255).astype(np.uint8))

        dec = subprocess.Popen(_decode_cmd(ffmpeg, src, tile, dec_start, info.fps),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               bufsize=0)
        assert dec.stdout is not None
        sink = _PartSink(ffmpeg=ffmpeg, src=src, parts_dir=parts_dir, info=info,
                         tile=tile, alpha_png=alpha_png, encode=encode, total=n_total,
                         part_frames=opts.part_frames, first=resume_from, say=say)

        # Bounded queues are the backpressure. One segment of lookahead on each side
        # is enough to keep every stage busy; more would only raise peak memory.
        # Sized from the largest segment the plan actually contains, not from
        # --pp-segment. Those are the same number only when no scene cut shortens
        # anything; with cuts they are far apart, and sizing on the ceiling cost 855 MB
        # of queues to carry segments of at most 232 frames. The queues are the
        # backpressure, so they need to hold one segment plus its context and no more.
        longest = max((e - s for s, e, _, _ in plan), default=opts.segment)
        depth = queue_depth = longest + 2 * opts.overlap
        q_in: queue.Queue = queue.Queue(maxsize=depth)
        q_out: queue.Queue = queue.Queue(maxsize=depth)
        DONE = object()

        def pump_decode() -> None:
            try:
                for frame in _decode_frames(dec.stdout, tile):
                    q_in.put(frame)
            finally:
                q_in.put(DONE)

        def pump_encode() -> None:
            while True:
                item = q_out.get()
                if item is DONE:
                    return
                sink.write(item)

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

        def emit_body(out, block: list[np.ndarray], n_left: int, n_body: int) -> None:
            """Emit this segment's body, in order, past the dark-tile guard.

            Frame j is produced by exactly one segment, so ordering here is positional
            and not a race. On a resumed run the head of this segment is already in a
            finished part; it was recomputed to make the rest of the segment
            reproducible, and it is dropped here rather than written twice.
            """
            nonlocal emitted, guarded
            for i in range(n_left, n_left + n_body):
                if pos + (i - n_left) < resume_from:
                    continue
                tile_out = out[i]
                fixed = _dark_guard(block[i], tile_out, hole_px, ring_px)
                if fixed is not None:
                    tile_out = fixed
                    guarded += 1
                emit(np.ascontiguousarray(tile_out))
                emitted += 1

        n = 0                  # frames read by this run
        emitted = 0            # frames handed to the compositor by this run
        pos = pos0             # absolute index of buf[0] -- not `emitted`, which on a
                               # resumed run starts counting again from zero
        hole_px = region.inpaint_mask > 0     # what the model replaced
        ring_px = ~hole_px                    # the picture it had to match
        guarded = 0            # frames the dark-tile guard corrected
        nonfinite_total = 0
        nonfinite_at: set[int] = set()
        model_frames = 0       # frames the model actually ran on, context included
        n_seg = 0
        eof = False
        hist: list[np.ndarray] = []      # last `overlap` decoded frames, for left context
        buf: list[np.ndarray] = []       # decoded, not yet emitted; buf[0] is frame `pos`
        t_model = 0.0
        est_total = info.nframes or 0
        broken = False
        crash: BaseException | None = None

        # A resumed run decodes from `dec_start` rather than from zero, and the frames
        # between there and the first segment body are exactly that segment's left
        # context -- the ones a whole run would have had in `hist` by now.
        while len(hist) < pos0 - dec_start:
            f = next_frame()
            if f is None:
                eof = True
                break
            hist.append(f)
            n += 1

        if shot_starts:
            lens = [e - s for s, e, _, _ in plan]
            say(f"[pp] {len(shot_starts)} shot boundary/ies -> {len(plan)} segment(s), "
                f"{min(lens)}-{max(lens)} frames each; no segment spans a cut or a "
                f"black run")
        else:
            say(f"[pp] no shot boundaries found -> {len(plan)} fixed segment(s) of "
                f"{opts.segment}")

        try:
            for seg_i, (s_abs, e_abs, lcap, rcap) in enumerate(plan[seg0:], seg0):
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
                say(f"[pp] segment {seg_i + 1}/{len(plan)}: frames {s_abs}-{s_abs + body_len - 1} "
                    f"(+{len(left)}/{len(ahead)} context)"
                    + (f", {dec_start + n}/{est_total} decoded" if est_total else ""))

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
                    # `pos` is the absolute index of body[0], so block index i sits at
                    # pos - len(left) + i.
                    absolute = [pos - len(left) + i for i in frames_hit]
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

                emit_body(out, block, len(left), len(body))
                pos += body_len

                el = time.monotonic() - t_start
                rate = emitted / max(t_model, 1e-6)
                eta = ((est_total - pos) / rate) if rate and est_total > pos else 0.0
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
                block = left + body + ahead
                out = _inpaint_or_split(worker, block, region.inpaint_mask, say)
                t_model += time.monotonic() - ts
                model_frames += len(block)
                emit_body(out, block, len(left), body_len)
                pos += body_len
                if opts.overlap:
                    hist = (hist + body)[-opts.overlap:]
                buf = buf[body_len:]
        except BrokenPipeError:
            broken = True
        except BaseException as exc:            # noqa: BLE001 -- re-raised below
            # Held rather than propagated, so that the compositor is still shut down
            # and the open part still abandoned. Letting it fly from here would skip
            # both: an OOM in the model would leave a half-written part on disk with
            # nothing to mark it unfinished, and the next --resume would read it as
            # done and build a video with a gap in it. tests/test_resume.py pins this.
            crash = exc
        finally:
            # The sentinel must never be able to block. If the compositor has stopped
            # draining -- its ffmpeg dead, or wedged -- a plain put() on a full queue
            # waits forever and the whole run hangs silently, which is strictly worse
            # than the failure that caused it: no message, no traceback, no clue. Give
            # up on delivering it instead, and let the stall watch below say what is
            # actually wrong. Frames left in the queue are not lost work -- their part
            # is being abandoned either way.
            sentinel = time.monotonic() + STALL_LIMIT
            while True:
                try:
                    q_out.put(DONE, timeout=1.0)
                    break
                except queue.Full:
                    if not t_encode.is_alive() or time.monotonic() > sentinel:
                        break
            # Wait on *progress*, not on a fixed deadline. The last segment hands its
            # whole body over at once -- up to `segment` frames -- and the model is
            # idle from that moment on, so the flush is minutes of encoding with
            # nothing left to overlap it against. A flat 60s deadline expired
            # mid-flush: main closed the pipe underneath the writer thread and the
            # frames still queued were counted as missing, which failed the run after
            # nine hours of model time. Measured twice at 4K, where the compositor
            # runs around 22-23 fps: a 1417-frame final segment came up 4 frames
            # short, a 1387-frame one 50 short -- both exactly `body - 60s * rate`.
            # Shorter final segments cleared the deadline, which is why this looked
            # like it depended on the file rather than on where the last cut fell.
            # A thread that is still moving frames is not a stuck thread.
            stalled = 0.0
            while t_encode.is_alive():
                mark = sink.written
                t_encode.join(timeout=STALL_POLL)
                if sink.written != mark:
                    stalled = 0.0
                    continue
                stalled += STALL_POLL
                if stalled >= STALL_LIMIT:
                    break
            encode_stuck = t_encode.is_alive()

            # Order matters. Reading the decoder's stderr waits for EOF, which only
            # arrives when it exits -- and it cannot exit while it is blocked writing
            # frames nobody is draining any more. Stop it first, then read.
            if dec.poll() is None:
                dec.kill()
            dec_err = drain(dec.stderr)
            dec.wait()

        # Anything raised from here abandons the part still being written: it is
        # incomplete by definition, and an incomplete part that looks finished is the
        # one thing a resume must never find.
        try:
            if crash is not None:
                raise crash
            if t_decode.error is not None:
                raise ProPainterError(f"decoding the tile failed: {t_decode.error}\n"
                                      f"{dec_err.strip()[:800]}")
            if broken:
                raise EncodeError("a compositing ffmpeg exited while frames were "
                                  "still being written into it")
            # Causes before symptoms. A frame-count mismatch is what you notice, but
            # it is also what a stuck compositor and a dead writer thread both look
            # like from here, and reporting the count first left ffmpeg's own message
            # unread while the run died of something else.
            if encode_stuck:
                raise EncodeError(
                    f"the compositing ffmpeg accepted no frames for "
                    f"{_hms(STALL_LIMIT)}, with {emitted - sink.written} still "
                    f"queued; giving up")
            if t_encode.error is not None:
                raise t_encode.error
            if n == 0 and not resume_from:
                raise ProPainterError(f"decoded 0 tile frames from {src}\n"
                                      f"{dec_err.strip()[:600]}")
            if sink.written != emitted:
                raise EncodeError(f"{emitted} frames were handed to the compositor "
                                  f"but {sink.written} reached it")
            sink.close()
            # The invariant the old `written != n` check was reaching for, stated
            # where it is actually true: every frame decoded from the source has one
            # frame in the parts, counted in absolute positions so that it means the
            # same thing on a resumed run as on a whole one.
            if sink.index != dec_start + n:
                raise ProPainterError(
                    f"the parts hold {sink.index} frames but {dec_start + n} were "
                    f"decoded; refusing to ship a video with frames missing or "
                    f"duplicated")
        except BaseException:
            sink.abandon()
            say(f"[pp] {resume_from + emitted} frame(s) are composited and kept in "
                f"{parts_dir.name}/. Re-run the same command with --resume to carry "
                f"on from there instead of from the beginning.")
            raise

        tmp = _join_parts(ffmpeg, parts_dir, work, src, dst, info, encode,
                          sink.index, opts.part_frames, say)
        os.replace(tmp, dst)          # atomic
        shutil.rmtree(parts_dir, ignore_errors=True)

    total = time.monotonic() - t_start
    if guarded:
        # Said out loud rather than counted silently: a handful of frames is the intro
        # doing what intros do, and a number anywhere near the frame count means the
        # footage itself is dark and the fill is being replaced by a flat patch all
        # the way through -- which is a reason to look at the output, not to trust it.
        say(f"[pp] the dark-tile guard replaced the model's fill on {guarded} frame(s) "
            f"whose surroundings held no picture ({100.0 * guarded / max(n, 1):.1f}% "
            f"of this run) -- see _dark_guard")
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
        f"peak in-flight {_sizeof(2 * queue_depth * tile.w * tile.h * 3)}"
        f" of frames, flat in video length")
    # Extrapolate, because a one-minute test says nothing about the videos this is
    # actually for until it is scaled up.
    say(f"[pp] EXTRAPOLATED  1 hour of this footage -> "
        f"~{_hms(t_load + 3600.0 * t_model / max(info.duration, 1e-6))} "
        f"(the load cost does not scale with length)")
    return region
