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
from dataclasses import dataclass
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
    segment: int = SEGMENT
    overlap: int = OVERLAP
    subvideo_length: int = 80
    neighbor_length: int = 10
    ref_stride: int = 10
    raft_iter: int = 20
    mask_dilation: int = 4
    fp16: bool = True
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


def _load_worker(opts: ProPainterOpts):
    """Import the resident worker out of the checkout and construct it.

    Model loading happens here, once, and it is slow enough to be worth logging
    around by the caller.
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
    return ProPainterWorker(device=device, opts=WorkerOpts(
        subvideo_length=opts.subvideo_length,
        neighbor_length=opts.neighbor_length,
        ref_stride=opts.ref_stride,
        raft_iter=opts.raft_iter,
        mask_dilation=opts.mask_dilation,
        fp16=opts.fp16,
    ))


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

    if opts.workers > 1:
        say(f"[pp] note: --pp-workers {opts.workers} is ignored. One resident model "
            f"processes segments in order; decode and composite already run "
            f"alongside it. See ProPainterOpts.workers.")

    t_start = time.monotonic()
    t0 = time.monotonic()
    worker = _load_worker(opts)
    t_load = time.monotonic() - t0
    say(f"[pp] models loaded in {_hms(t_load)} (once for the whole video, not once "
        f"per segment)")

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
        model_frames = 0       # frames the model actually ran on, context included
        n_seg = 0
        eof = False
        left: list[np.ndarray] = []      # already-emitted frames kept as left context
        pending: list[np.ndarray] = []   # decoded, not yet emitted
        t_model = 0.0
        est_total = info.nframes or 0
        broken = False

        try:
            while True:
                # Fill the body of this segment.
                while not eof and len(pending) < opts.segment:
                    f = next_frame()
                    if f is None:
                        eof = True
                        break
                    pending.append(f)
                    n += 1
                if not pending:
                    break

                body = pending
                # Read the right-hand context. These frames belong to the next
                # segment's body too -- they are context here and kept there.
                ahead: list[np.ndarray] = []
                while not eof and len(ahead) < opts.overlap:
                    f = next_frame()
                    if f is None:
                        eof = True
                        break
                    ahead.append(f)
                    n += 1

                block = left + body + ahead
                n_seg += 1
                say(f"[pp] segment {n_seg}: {len(body)} frames "
                    f"(+{len(left)}/{len(ahead)} context)"
                    + (f", {n}/{est_total} decoded" if est_total else ""))

                ts = time.monotonic()
                out = worker.inpaint(np.stack(block), region.inpaint_mask)
                t_model += time.monotonic() - ts
                model_frames += len(block)

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

                left = body[-opts.overlap:] if opts.overlap else []
                pending = ahead
                if eof and not pending:
                    break
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
    say(f"[pp] TIME  load {_hms(t_load)} ({pct(t_load):.0f}%)  "
        f"model {_hms(t_model)} ({pct(t_model):.0f}%)  "
        f"decode+composite, overlapped {_hms(max(0.0, total - t_model - t_load))} "
        f"({pct(max(0.0, total - t_model - t_load)):.0f}%)")
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
