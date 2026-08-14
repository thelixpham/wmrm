"""Command line interface.

    wmrm detect  IN.mp4 [--corner tr] [--preset wm.json]   # calibrate once
    wmrm run     IN.mp4 [-o OUT.mp4] --preset wm.json      # or --box x,y,w,h
    wmrm batch   DIR --preset wm.json [-j 2]
    wmrm verify  ORIG.mp4 OUT.mp4 --preset wm.json
"""

from __future__ import annotations

import argparse
import signal
import sys
import traceback
from dataclasses import replace
from pathlib import Path

from . import __version__
from .detect import DetectError, detect, write_preview
from .errors import (CoverageInconclusive, CoverageUnder, InputMissing, UsageError,
                     WmrmError)
from .lock import output_lock
from .pipeline import EncodeError, EncodeOpts, run_fast, run_inpaint
from .probe import ProbeError, ToolMissing, probe
from .region import CORNERS, Box, Preset
from .report import ReportWriter, RunContext
from .verify import verify as run_verify

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
CLEAN_SUFFIX = "-clean"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _resolve_region(args, width: int, height: int, *, src: Path | None = None
                    ) -> tuple[Box, Preset]:
    """Return the box to use plus the knobs, from --box, --preset or --detect."""
    if args.box and args.preset:
        raise UsageError("pass either --box or --preset, not both")
    # No box, no preset -> detect. This used to be an error telling you to pass one of
    # three flags, which is the right default only if an accidental unattended guess is
    # the worst outcome. It no longer is: a detected box is now checked by the coverage
    # test before anything is processed, a preview is always written, and a file whose
    # box is provably too small is stopped rather than shipped. With those in place,
    # refusing to do the obvious thing was just a flag to memorise.
    #
    # A named --preset or --box still wins, and still skips detection entirely.
    if not args.box and not args.preset and not getattr(args, "detect", False):
        if src is None:
            raise UsageError(
                "need --box x,y,w,h or --preset wm.json here (this command "
                "cannot detect, it has no video to look at)"
            )
        args.detect = True
        print("[wmrm] no --box or --preset given, so detecting the watermark",
              file=sys.stderr)

    if args.box:
        box = Box.parse(args.box).clamp(width, height)
        preset = Preset.from_box(box, width, height)
    elif not args.preset and getattr(args, "detect", False):
        if src is None:
            raise UsageError("--detect needs an input file")
        # Detection is a guess with measured failure modes, so a preview is
        # always written and every processed file still goes through the usual
        # verification. If detection finds nothing it raises, which aborts the
        # run rather than quietly processing with a bogus box.
        # Every knob `wmrm detect` exposes, reachable from `run` and `batch` too.
        # Without this, tuning detection meant running `detect` by hand to write a
        # preset and then passing it back in -- two commands and a file, purely
        # because one call site hardcoded the defaults. Anything that decides the
        # box has to be reachable from the command that uses the box.
        det = detect(
            src,
            corner=args.corner,
            samples=getattr(args, "samples", 40),
            roi_frac=getattr(args, "roi_frac", 0.30),
            grad_threshold=getattr(args, "grad_threshold", None),
            persistence=getattr(args, "persistence", 0.90),
            max_area_percent=getattr(args, "max_area", 10.0),
        )
        print(det.describe())
        preview = src.with_name(f"{src.stem}-preview.png")
        write_preview(src, det.box, preview, roi=det.roi,
                      zoom_png=preview.with_name(f"{preview.stem}-zoom.png"))
        print(f"\nCHECK {preview.with_name(preview.stem + '-zoom.png')} afterwards. "
              "Detection is a guess, not a guarantee.\n")
        box = det.box
        preset = Preset.from_box(box, width, height, opacity=det.opacity)
        # Write the box down, next to the preview and for the same reason: this run is
        # about to spend hours on a guess, and "what box was it actually made with" is
        # the first question anyone asks when an output looks wrong. It also saves
        # running `detect` by hand purely to get a file to pass to --preset.
        #
        # A record, not an input. Nothing picks this up on its own, and that asymmetry
        # is deliberate -- a preset that merely happens to exist quietly overriding
        # what the command asked for is the exact failure run.sh already names.
        saved = src.with_name(f"{src.stem}-preset.json")
        try:
            preset.save(saved)
            print(f"box written -> {saved}\n"
                  f"  reuse it with --preset {saved.name}; left alone it is only a "
                  f"record, detection still runs by default", file=sys.stderr)
        except OSError as exc:
            # A read-only mount next to the source is not a reason to refuse the work.
            print(f"[wmrm] note: could not write {saved}: {exc}", file=sys.stderr)
    elif args.preset:
        path = Path(args.preset)
        if not path.exists():
            raise UsageError(
                f"preset {path} not found. Create it with:\n"
                f"  wmrm detect YOUR.mp4 --preset {path}"
            )
        preset = Preset.load(path).scaled_px(width, height)
        box = preset.box_for(width, height)
    else:                                    # pragma: no cover -- unreachable now
        raise UsageError("could not work out which box to use")

    overrides = {
        f"{name}_px": getattr(args, name)
        for name in ("dilate", "feather", "margin")
        if getattr(args, name, None) is not None
    }
    if overrides:
        preset = replace(preset, **overrides)
    return box, preset


def _box_source(args) -> str:
    """Where the box came from, for the report. Derived, so it cannot disagree."""
    if getattr(args, "box", None):
        return "given"
    if getattr(args, "preset", None):
        return "preset"
    return "detect"


def _log_config(src: Path, dst: Path, info, box: Box, preset: Preset, args,
                backend) -> RunContext:
    """Print the effective configuration once, before any frames are touched.

    Mostly so a wrong device is impossible to miss: the CPU-only torch wheel on a
    GPU machine runs 20-50x slower and produces byte-identical output, so nothing
    downstream reveals the mistake.

    Returns what it worked out as well as printing it. The engine label, the resolved
    device and the tile size used to exist only inside these f-strings, so anything
    that needed them -- the report, and through it the service -- had to recompute them
    and could end up disagreeing with the banner the operator read.
    """
    from .region import build_region

    region = build_region(box, info.width, info.height,
                          dilate_px=preset.dilate_px, feather_px=preset.feather_px,
                          margin_px=preset.margin_px)
    if backend is not None:
        engine, where = backend.name, backend.device_note
    elif args.quality == "video":
        from .video import describe_device, find_repo

        engine = "propainter (flow propagation across frames)"
        try:
            engine += f" @ {find_repo(args.propainter)}"
        except Exception as exc:                     # reported, not fatal here
            engine += f" -- NOT FOUND: {exc}"
        where = describe_device()
    else:
        engine, where = "ffmpeg delogo+feather", "cpu (ffmpeg filters, no model)"

    p = lambda s: print(s, file=sys.stderr)      # noqa: E731
    p(f"[cfg] input    : {src.name} -> {dst.name}")
    p(f"[cfg] video    : {info.width}x{info.height} @ {info.fps} fps, "
      f"{info.nframes or '?'} frames, {info.duration:.2f}s, "
      f"audio {'yes' if info.has_audio else 'none'}")
    p(f"[cfg] box      : {box.x},{box.y},{box.w},{box.h}   "
      f"dilate {preset.dilate_px}  feather {preset.feather_px}  "
      f"margin {preset.margin_px}")
    p(f"[cfg] tile     : {region.tile.w}x{region.tile.h} "
      f"({100 * region.tile.area() / (info.width * info.height):.2f}% of frame) "
      f"-- cost scales with this")
    p(f"[cfg] engine   : {engine}  (--quality {args.quality})")
    p(f"[cfg] device   : {where}")
    # Only report patch reuse when the backend genuinely does it. un-blend does
    # not: its transform is already deterministic, so echoing the cache flags there
    # would describe behaviour that is not happening.
    from .backends import CachingBackend, UnblendBackend

    if isinstance(backend, CachingBackend):
        p(f"[cfg] reuse    : patch-hold {backend.hold}, "
          f"tolerance {backend.tolerance}")
    if isinstance(backend, UnblendBackend) and backend.fallback is None:
        p("[cfg] note     : recovery only -- nothing is synthesized, so a faint "
          "trace of the mark can remain")
    p(f"[cfg] encode   : libx264 crf {args.crf} preset {args.x264_preset}, "
      f"audio copy")

    return RunContext(
        info=info, box=box, tile=region.tile,
        engine=args.quality, engine_label=engine,
        device=getattr(args, "device", "auto"), device_label=where,
        box_source=_box_source(args),
    )


def _default_output(src: Path) -> Path:
    return src.with_name(f"{src.stem}{CLEAN_SUFFIX}{src.suffix or '.mp4'}")


def _process_one(src: Path, dst: Path, box: Box, preset: Preset, args, backend):
    encode = EncodeOpts(crf=args.crf, x264_preset=args.x264_preset)
    if args.quality == "video":
        from .video import ProPainterOpts, find_repo, run_propainter

        return run_propainter(
            src, dst, box=box,
            dilate_px=preset.dilate_px, feather_px=preset.feather_px,
            margin_px=preset.margin_px, encode=encode,
            progress=not args.quiet,
            resume=args.resume,
            # --device now reaches this path. It used to be accepted and silently
            # ignored here: upstream's script has no device flag, so a CUDA box that
            # failed the cudnn check fell back to CPU without saying so, at ~400x the
            # cost. The worker takes it, and 'auto' still means its own rule.
            opts=ProPainterOpts(repo=find_repo(args.propainter),
                                device=args.device,
                                segment=args.pp_segment,
                                part_frames=args.pp_part,
                                subvideo_length=args.pp_subvideo,
                                raft_iter=args.raft_iter,
                                fp16=not args.no_fp16,
                                scene_threshold=args.pp_scene_threshold,
                                min_shot=args.pp_min_shot,
                                black_cuts=args.pp_black_cuts,
                                workers=args.pp_workers),
        )
    if args.quality == "fast":
        return run_fast(
            src, dst, box=box,
            dilate_px=preset.dilate_px, feather_px=preset.feather_px,
            margin_px=preset.margin_px, encode=encode,
        )
    return run_inpaint(
        src, dst, backend, box=box,
        dilate_px=preset.dilate_px, feather_px=preset.feather_px,
        margin_px=preset.margin_px, encode=encode,
        progress=not args.quiet,
    )


def _make_backend(args, *, src: Path | None = None, box: Box | None = None,
                  preset: Preset | None = None, report=None):
    # Both of these own their whole clip: `fast` is one ffmpeg graph, `video` is a
    # sequence model that needs many frames at once. Neither fits the per-frame
    # Backend interface, so there is nothing to build here.
    if args.quality in ("fast", "video"):
        return None
    from .backends import make_backend

    fitted = None
    if args.quality == "unblend":
        if src is None or box is None or preset is None:
            raise UsageError("--quality unblend needs an input file and a box")
        fitted = _fit_unblend(src, box, preset, samples=args.unblend_samples)
        # The fit is the only place these numbers exist, and `residual` is the line that
        # actually says whether removal worked. Hand them to the report on the way past
        # rather than recomputing or, worse, leaving the caller to parse the banner.
        if report is not None:
            report.set_unblend(fitted)

    return make_backend(args.quality, threads=args.threads,
                        cache_tolerance=args.cache_tolerance,
                        patch_hold=args.patch_hold, device=args.device,
                        fitted=fitted)


def _fit_unblend(src: Path, box: Box, preset: Preset, *, samples: int = 40):
    """Sample the clip and solve for the mark's alpha map.

    Fitted per video, not per logo: compression and colour grading change the
    numbers even for the same watermark.
    """
    import numpy as np

    from .detect import sample_frames
    from .region import build_region
    from .unblend import fit

    info = probe(src)
    region = build_region(box, info.width, info.height,
                          dilate_px=preset.dilate_px, feather_px=preset.feather_px,
                          margin_px=preset.margin_px)
    frames = sample_frames(info, samples, region.tile)

    mark = np.zeros(frames.shape[1:3], bool)
    mark[box.y - region.tile.y: box.y - region.tile.y + box.h,
         box.x - region.tile.x: box.x - region.tile.x + box.w] = True

    fitted = fit(frames, mark)
    print(f"[wmrm] un-blend fitted from {frames.shape[0]} frames", file=sys.stderr)
    print(fitted.describe(), file=sys.stderr)
    if fitted.alpha_median > 0.85:
        print("\nWARNING: this mark is nearly opaque -- un-blend has little to "
              "recover here.\n"
              "         Fully blocked pixels are handed to LaMa automatically, but "
              "the result\n"
              "         will be closer to inpainting than to recovery. Consider "
              "--quality high.", file=sys.stderr)
    return fitted


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_detect(args) -> int:
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"error: {src} not found")
    info = probe(src)

    # `--box` turns this into "freeze these coordinates into a preset". Detection
    # is unreliable on busy backgrounds, so hand-measured coordinates need a way
    # to become a reusable preset too -- otherwise you are retyping them forever.
    det = None
    if args.box:
        box = Box.parse(args.box).clamp(info.width, info.height)
        roi = None
        opacity = "unknown"
        print(f"using the box you supplied: x={box.x} y={box.y} w={box.w} h={box.h}")
    else:
        det = detect(
            src,
            corner=args.corner,
            samples=args.samples,
            roi_frac=args.roi_frac,
            grad_threshold=args.grad_threshold,
            persistence=args.persistence,
            max_area_percent=args.max_area,
        )
        print(det.describe())
        box, roi, opacity = det.box, det.roi, det.opacity

    preview = Path(args.preview) if args.preview else src.with_name(f"{src.stem}-preview.png")
    zoom = preview.with_name(f"{preview.stem}-zoom{preview.suffix}")
    write_preview(src, box, preview, roi=roi, zoom_png=zoom)

    preset_path = Path(args.preset) if args.preset else src.with_name("wm-preset.json")
    preset = Preset.from_box(
        box, info.width, info.height,
        opacity=opacity,
        dilate_px=args.dilate if args.dilate is not None else 5,
        feather_px=args.feather if args.feather is not None else 12,
        margin_px=args.margin if args.margin is not None else 64,
    )
    preset.save(preset_path)

    print(f"\npreset written -> {preset_path}")
    print(f"preview        -> {preview}"
          + ("  (red = watermark box, orange = search area)" if roi else "  (red = your box)"))
    print(f"zoomed preview -> {zoom}")
    if det and det.opacity == "semi":
        print(
            "\nNOTE: the watermark looks semi-transparent (background bleeds through),\n"
            "      so --quality unblend can recover the original background by dividing\n"
            "      the blend back out instead of repainting the region. It is the fast\n"
            "      CPU option and it cannot flicker; --quality video (the default) is\n"
            "      the one that removes the mark completely, and wants a GPU."
        )
    print(
        "\nLOOK AT THE PREVIEW. Detection is a guess with measured failure modes.\n"
        f"  {zoom}\n"
        "This command only wrote the box; to process with it:\n"
        f"  wmrm run {src} --preset {preset_path}\n"
        "If the box is wrong, measure it yourself and pass it directly:\n"
        f"  wmrm run {src} --box x,y,w,h"
    )
    return 0


def _install_signal_handlers() -> None:
    """Turn SIGTERM into an exception so the normal unwinding runs.

    Without this, a `kill` leaves no report, no released lock and orphaned ffmpeg
    children -- the default disposition tears the process down without running any
    `finally`. Raising KeyboardInterrupt instead means the same path that handles Ctrl-C
    handles being stopped by a supervisor, which is the case that actually happens in
    production when a pod is restarted mid-run.

    SIGINT already raises; it is re-registered only so both signals report identically.
    """
    def _raise(signum, _frame):                      # noqa: ANN001
        raise KeyboardInterrupt(f"received signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _raise)
        except (ValueError, OSError):                 # pragma: no cover
            # Not the main thread, or a platform without it. Not worth failing over.
            pass


def _run_coverage_gate(src: Path, box: Box, mode: str, report) -> None:
    """Check the box covers the whole mark, before a single frame is processed.

    `run` did not do this before -- only `batch` did, and only when it had detected the
    box itself. That left the most common invocation unguarded against the one failure
    that matters: a box a few pixels too small ships a video with watermark fringe still
    in it, and nothing downstream notices.

    Only `strict` stops, and it is not the default -- see the flag's definition for the
    measurement behind that. `warn` reports the same verdict and carries on, which keeps
    every existing invocation behaving as it did while still putting the verdict in the
    report where a caller can act on it.
    """
    if mode == "off":
        return

    from .coverage import check_coverage

    cov = check_coverage(src, box)
    print(cov.describe(), file=sys.stderr)
    if report is not None:
        report.set_coverage(cov)

    if mode == "warn":
        # Asked for explicitly, which reads as "I have looked and I accept it", so
        # neither verdict blocks and neither becomes a review item.
        if not cov.ok:
            print("[wmrm] --coverage-gate warn: continuing despite the verdict above",
                  file=sys.stderr)
        return

    if cov.inconclusive:
        raise CoverageInconclusive(
            "coverage INCONCLUSIVE -- the background is static, so no statistic can "
            "separate mark from wall here. Confirm this one by eye "
            "(wmrm run --preview-only), then re-run with --coverage-gate warn."
        )
    if not cov.ok:
        s = cov.suggested
        hint = f" Try --box {s.x},{s.y},{s.w},{s.h}" if s else ""
        raise CoverageUnder(
            f"UNDER-COVERED -- the watermark extends outside this box, so processing "
            f"it would ship a video with fringe left in.{hint}"
        )


def cmd_run(args) -> int:
    report = ReportWriter(Path(args.report)) if getattr(args, "report", None) else None
    _install_signal_handlers()
    try:
        return _cmd_run_inner(args, report)
    except BaseException as exc:
        # BaseException, not Exception: SystemExit and KeyboardInterrupt are the two
        # that matter most here. A run stopped by a supervisor is exactly the case the
        # caller needs a report for, and it arrives as KeyboardInterrupt.
        if report is not None:
            report.fail(exc)
        raise                       # re-raised untouched, so exit codes do not change
    finally:
        if report is not None:
            report.flush()          # no-op if the run already said how it ended


def _cmd_run_inner(args, report) -> int:
    src = Path(args.input)
    if not src.exists():
        raise InputMissing(f"{src} not found")
    info = probe(src)
    box, preset = _resolve_region(args, info.width, info.height, src=src)
    dst = Path(args.output) if args.output else _default_output(src)
    if report is not None:
        report.set_paths(src, dst)

    if args.preview_only:
        out = dst.with_name(f"{src.stem}-boxcheck.png")
        write_preview(src, box, out, zoom_png=out.with_name(f"{src.stem}-boxcheck-zoom.png"))
        print(f"box {box.as_tuple()} drawn on {out}; nothing processed")
        if report is not None:
            report.ok(dst=None)
        return 0

    # Before the lock and before the fit: both cost real time, and a box that fails the
    # gate is not going to be processed with either of them.
    _run_coverage_gate(src, box, getattr(args, "coverage_gate", "strict"), report)

    with output_lock(dst, enabled=not getattr(args, "no_lock", False)):
        backend = _make_backend(args, src=src, box=box, preset=preset, report=report)
        ctx = _log_config(src, dst, info, box, preset, args, backend)
        if report is not None:
            report.set_context(ctx)
        _process_one(src, dst, box, preset, args, backend)

        if not args.no_verify:
            result = run_verify(src, dst, box)
            print("\n[wmrm] verification:")
            print(result.render())
            if report is not None:
                report.set_verify(result)
            if not result.ok:
                # A verdict, not a crash, so there is no exception to map -- but the
                # caller still has to be able to tell this apart from a missing input.
                if report is not None:
                    report.fail_outcome(
                        "verify_failed",
                        "acceptance checks failed: "
                        + ", ".join(n for n, ok, _ in result.checks if not ok),
                    )
                return 1
    if report is not None:
        report.ok(dst=dst)
    return 0


def cmd_serve(args) -> int:
    """Run the pod HTTP API.

    Here rather than telling people a uvicorn command line, because two of the arguments
    are not optional and a forgotten one fails in an expensive way: `--workers 1`,
    because job state is per-process and on disk, and `--host 0.0.0.0`, because a RunPod
    proxy cannot reach a server bound to localhost.
    """
    try:
        import uvicorn
    except ImportError:
        raise UsageError(
            "wmrm serve needs the 'serve' extra:\n"
            "  uv pip install -e '.[serve]'"
        )
    from .server.app import create_app
    from .server.config import Config
    from .server.hooks import MezonNotifier

    from .server.config import on_pod

    cfg = Config.from_env()
    if not cfg.token:
        # The one thing that has to be set. Everything else is derived, so this is the
        # only reason a start can be "successful" and still serve nothing.
        print("[wmrm] WMRM_POD_TOKEN is not set, so every route except /live will "
              "answer 503.\n"
              "       It is the ONLY required variable. Pick any secret and use the "
              "same value\n"
              "       when you register this pod:  export WMRM_POD_TOKEN=...",
              file=sys.stderr)

    # Built as plain strings, not nested inside the f-strings below: quoting a quote of
    # the same kind inside an f-string is only legal from Python 3.12, and this package
    # declares requires-python >= 3.10.
    where = ("RunPod pod, state on /workspace" if on_pod()
             else "not a pod (no RUNPOD_POD_ID), state in the cache dir")
    input_note = str(cfg.local_input_root) if cfg.local_input_root else (
        "(r2 and url only -- set WMRM_LOCAL_INPUT_ROOT to allow kind=local)")
    r2_note = cfg.r2_bucket or "(not configured)"
    if not cfg.r2_configured:
        r2_note += "  <- kind=r2 jobs will be refused"
    mezon_note = MezonNotifier(cfg.mezon_webhook_url).describe()
    # Printed because the alternative was discovering it from a 507: a pod that keeps every
    # job's files is one job away from refusing work, and that is a property of the process
    # you started, not something to go and read the source for.
    from .server import reclaim as _reclaim
    clean_note = (f"delivered jobs at once, everything else after "
                  f"{cfg.retention_hours:g}h" if _reclaim.enabled()
                  else "off (WMRM_RECLAIM) -- files stay until DELETE /jobs/{id}")

    print(f"[wmrm] pod id : {cfg.pod_id}\n"
          f"[wmrm] machine: {where}\n"
          f"[wmrm]   work  : {cfg.work_dir}\n"
          f"[wmrm]   state : {cfg.state_dir}\n"
          f"[wmrm]   input : {input_note}\n"
          f"[wmrm]   r2    : {r2_note}\n"
          f"[wmrm]   mezon : {mezon_note}\n"
          f"[wmrm]   disk  : refuse below {cfg.min_free_gb:g} GiB free\n"
          f"[wmrm]   clean : {clean_note}\n"
          f"[wmrm]   jobs  : {cfg.max_concurrent} at a time\n"
          f"[wmrm] serving on {args.host}:{args.port}  (docs at /docs)",
          file=sys.stderr)

    uvicorn.run(create_app(cfg), host=args.host, port=args.port,
                log_level=args.log_level, workers=1)
    return 0


def cmd_batch(args) -> int:
    root = Path(args.directory)
    if not root.is_dir():
        raise SystemExit(f"error: {root} is not a directory")
    videos = sorted(
        p for p in root.iterdir()
        if p.suffix.lower() in VIDEO_SUFFIXES and CLEAN_SUFFIX not in p.stem
    )
    if not videos:
        raise SystemExit(f"error: no videos in {root} (looked for {', '.join(VIDEO_SUFFIXES)})")

    outbox = Path(args.outbox) if getattr(args, "outbox", None) else None
    if outbox is not None:
        outbox.mkdir(parents=True, exist_ok=True)

    def destination(src: Path) -> Path:
        if outbox is None:
            return _default_output(src)
        return outbox / f"{src.stem}{CLEAN_SUFFIX}{src.suffix or '.mp4'}"

    todo = []
    for src in videos:
        dst = destination(src)
        if dst.exists() and not args.force:
            print(f"[wmrm] skip (exists): {dst.name}", file=sys.stderr)
            continue
        todo.append((src, dst))
    if not todo:
        print("[wmrm] nothing to do")
        return 0

    print(f"[wmrm] {len(todo)} video(s) to process", file=sys.stderr)

    # Two ways to get a box, and the choice is about what the batch is.
    #
    # --detect (one box for the folder): repeatable, and there is one preview and one
    # decision for a human to make. Right when every file is the same watermark from
    # the same pipeline.
    #
    # --detect-each (a box per file): a better fit when the batch is mixed, since a box
    # measured on one source says nothing about a different crop, logo or placement and
    # normalized coordinates only rescue a change of resolution. What it costs is the
    # single place someone confirmed the box, which is why the coverage gate below is on
    # by default for it.
    # Per file is the default here, not --detect-each. A folder is the case where the
    # files are most likely to differ from each other, so reusing one file's box across
    # all of them is the assumption that needs asking for, and --detect is how you ask.
    detect_each = (not args.box and not args.preset
                   and not getattr(args, "detect", False)) or bool(
                       getattr(args, "detect_each", False))
    detect_each = detect_each and not args.box and not args.preset
    if detect_each:
        args.detect = True          # _resolve_region reads this per file
        print("[wmrm] detecting a box per video (--detect uses one box for the folder)",
              file=sys.stderr)

    shared: Preset | None = None
    if (not detect_each and getattr(args, "detect", False)
            and not args.box and not args.preset):
        first = todo[0][0]
        print(f"[wmrm] detecting on {first.name}, then applying to all "
              f"{len(todo)} files", file=sys.stderr)
        info = probe(first)
        _, shared = _resolve_region(args, info.width, info.height, src=first)

    first_src, _ = todo[0]
    first_info = probe(first_src)
    if shared is not None:
        first_box = shared.box_for(first_info.width, first_info.height)
        first_preset = shared
    else:
        first_box, first_preset = _resolve_region(
            args, first_info.width, first_info.height, src=first_src)
    # One fitted map for the folder, same reasoning as one box for the folder. With a
    # box per file that reasoning no longer holds: an un-blend map fitted on the first
    # file's box would be applied through a different box on the next one, so it is
    # refitted per file below instead.
    backend = None if detect_each else _make_backend(
        args, src=first_src, box=first_box, preset=first_preset)

    # The gate no longer depends on who chose the box. It used to run only for
    # --detect-each, on the reasoning that a box a human passed in had already been
    # confirmed -- but a box measured on one file and applied to fifty is exactly the
    # case where one of the fifty is cropped differently, and checking it costs one
    # sampling pass against hours of processing.
    gate_mode = getattr(args, "coverage_gate", "strict")
    if getattr(args, "no_coverage_gate", False):
        print("[wmrm] --no-coverage-gate is deprecated; use --coverage-gate off",
              file=sys.stderr)
        gate_mode = "off"
    gate = gate_mode != "off"
    failures: list[str] = []
    blocked: list[str] = []
    unverified: list[str] = []

    for i, (src, dst) in enumerate(todo, 1):
        print(f"\n[wmrm] ({i}/{len(todo)}) {src.name}", file=sys.stderr)
        try:
            info = probe(src)
            if shared is not None:
                # Normalized coordinates, so a folder of mixed resolutions still
                # gets the right box for each file.
                preset = shared.scaled_px(info.width, info.height)
                box = preset.box_for(info.width, info.height)
            else:
                box, preset = _resolve_region(args, info.width, info.height, src=src)

            if gate:
                from .coverage import check_coverage
                cov = check_coverage(src, box)
                print(cov.describe(), file=sys.stderr)
                if cov.inconclusive:
                    # The background is static, so no statistic separates mark from
                    # wall. Not a reason to refuse the work -- a reason to say so and
                    # list it again at the end.
                    #
                    # `run` stops here instead, and the difference is intended: it
                    # processes one file for a caller that can wait for a human, while
                    # a folder of fixed-camera clips would otherwise refuse every file
                    # and produce nothing. Same signal, different cost of being wrong.
                    print(f"[wmrm] {src.name}: coverage INCONCLUSIVE -- processing, "
                          f"confirm this one by eye", file=sys.stderr)
                    unverified.append(src.name)
                elif not cov.ok:
                    s = cov.suggested
                    hint = (f" Try --box {s.x},{s.y},{s.w},{s.h}" if s else "")
                    if gate_mode == "warn":
                        print(f"[wmrm] {src.name}: box is too small but "
                              f"--coverage-gate warn was asked for, processing "
                              f"anyway.{hint}", file=sys.stderr)
                        unverified.append(src.name)
                    else:
                        print(f"[wmrm] SKIPPED {src.name}: the detected box is too "
                              f"small, the watermark extends outside it.{hint}",
                              file=sys.stderr)
                        blocked.append(src.name)
                        continue

            per_file_backend = backend
            if detect_each:
                per_file_backend = _make_backend(args, src=src, box=box, preset=preset)
            _log_config(src, dst, info, box, preset, args, per_file_backend)
            _process_one(src, dst, box, preset, args, per_file_backend)
        except (EncodeError, ProbeError, DetectError) as exc:
            print(f"[wmrm] FAILED {src.name}: {exc}", file=sys.stderr)
            failures.append(src.name)

    # Repeated at the end because nobody scrolls back through fifty files of log, and
    # a blocked file that is not shouted about reads as a file that was fine.
    if blocked:
        print(f"\n[wmrm] NOT PROCESSED -- detected box too small ({len(blocked)}): "
              f"{', '.join(blocked)}", file=sys.stderr)
        print("[wmrm]   measure each by hand: wmrm grid FILE --corner tr", file=sys.stderr)
    if unverified:
        print(f"\n[wmrm] PROCESSED BUT UNVERIFIED -- coverage could not answer "
              f"({len(unverified)}): {', '.join(unverified)}", file=sys.stderr)
        print("[wmrm]   static background there; check these by eye", file=sys.stderr)
    if failures:
        print(f"\n[wmrm] {len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
    done = len(todo) - len(failures) - len(blocked)
    print(f"\n[wmrm] {done} processed, {len(blocked)} blocked, {len(failures)} failed",
          file=sys.stderr)
    return 1 if (failures or blocked) else 0


def cmd_grid(args) -> int:
    """Dump a frame with a labelled coordinate grid, for measuring by hand.

    Detection cannot be trusted across watermark types -- gradient-based search
    finds a semi-transparent mark and completely misses opaque white-on-white
    text. Measuring once by hand always works, so make that the easy path.
    """
    import subprocess

    from .probe import require_tools

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"error: {src} not found")
    ffmpeg, _ = require_tools()
    info = probe(src)
    out = Path(args.output) if args.output else src.with_name(f"{src.stem}-grid.png")

    step, scale = args.step, args.scale
    if args.corner:
        cw, ch = int(info.width * args.frac), int(info.height * args.frac)
        cx = 0 if args.corner in ("tl", "bl") else info.width - cw
        cy = 0 if args.corner in ("tl", "tr") else info.height - ch
    else:
        cx, cy, cw, ch = 0, 0, info.width, info.height

    at = args.at if args.at is not None else max(0.0, info.duration / 2)
    # Grid spacing is multiplied by `scale` so lines land on multiples of `step`
    # in *source* pixels after the zoom, which is what makes them readable.
    vf = (f"crop={cw}:{ch}:{cx}:{cy},scale=iw*{scale}:ih*{scale}:flags=neighbor,"
          f"drawgrid=w={step * scale}:h={step * scale}:t=1:c=red@0.7")
    res = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-nostdin", "-ss", f"{at:.3f}", "-i", str(src),
         "-vf", vf, "-frames:v", "1", str(out)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise SystemExit(f"error: ffmpeg failed:\n{res.stderr[:600]}")

    print(f"grid written -> {out}")
    print(f"video        : {info.width}x{info.height}")
    print(f"crop origin  : x={cx} y={cy}  (add these to what you read off the grid)")
    print(f"grid spacing : {step} source px, image zoomed {scale}x")
    print("\nRead the watermark's left/top/right/bottom off the gridlines, then:")
    print(f"  wmrm coverage {src} --box X,Y,W,H     # confirms nothing is left outside")
    return 0


def cmd_coverage(args) -> int:
    from .coverage import check_coverage

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"error: {src} not found")
    info = probe(src)
    box, _ = _resolve_region(args, info.width, info.height)
    result = check_coverage(src, box, samples=args.samples, ring=args.ring,
                            grad_threshold=args.grad_threshold)
    print(result.describe())
    if not result.ok and result.suggested:
        s = result.suggested
        print(f"\nre-check with:\n  wmrm coverage {src} "
              f"--box {s.x},{s.y},{s.w},{s.h}")

    # Three verdicts, three exit codes, because a caller automating this needs to
    # tell them apart. `ok` folds INCONCLUSIVE in with UNDER-COVERED, which is right
    # for a human reading the text and wrong for a gate: under-covered means the box
    # is provably too small, while inconclusive means the background is itself static
    # and no statistic can answer -- blocking on that would reject clips that are
    # fine, and there is nothing the caller could do about it except look.
    #
    #   0  covered
    #   1  UNDER-COVERED -- the box is too small, a suggestion was printed
    #   2  INCONCLUSIVE  -- unanswerable here, fall back to the preview and your eyes
    if result.inconclusive:
        return 2
    return 0 if result.ok else 1


def cmd_pull(args) -> int:
    """Fetch a source video out of R2 so the normal commands can work on it.

    Deliberately a separate step rather than a URI accepted by `run`: the
    download is hours at these sizes, it is resumable on its own, and the
    processing that follows is not. Keeping them apart means a failed run does
    not re-download 80 GB, and the file stays around for a second attempt with a
    different box.
    """
    from .r2 import Creds, R2Error, download, ls, stat

    try:
        if args.list:
            creds = Creds.from_env(args.bucket)
            objects = ls(args.key, bucket=args.bucket, creds=creds, limit=args.limit)
            if not objects:
                print(f"nothing under {args.key!r} in "
                      f"{args.bucket or creds.bucket}", file=sys.stderr)
                return 1
            for key, size in objects:
                print(f"{size:>14}  {key}")
            return 0

        if args.stat:
            print(stat(args.key, bucket=args.bucket).describe())
            return 0

        dest = Path(args.output) if args.output else Path.cwd()
        path = download(args.key, dest, bucket=args.bucket, chunk=args.chunk_mib << 20,
                        workers=args.workers, progress=not args.quiet,
                        overwrite=args.force)
    except R2Error as exc:
        raise SystemExit(f"error: {exc}")

    info = probe(path)
    print(f"\n{path}\n  {info.width}x{info.height} @ {info.fps} fps, "
          f"{info.duration:.1f}s, audio {'yes' if info.has_audio else 'none'}")
    print(f"\nnext:\n  wmrm detect {path} --corner tr\n  wmrm run {path} --preset "
          f"{path.with_name('wm-preset.json')}")
    return 0


def cmd_verify(args) -> int:
    orig, proc = Path(args.original), Path(args.processed)
    info = probe(orig)
    box = None
    if args.box or args.preset:
        box, _ = _resolve_region(args, info.width, info.height)
    result = run_verify(orig, proc, box)
    print(result.render())
    return 0 if result.ok else 1


# --------------------------------------------------------------------------- #
# argument wiring
# --------------------------------------------------------------------------- #

def _add_region_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", help="preset JSON from 'wmrm detect'")
    p.add_argument("--box", help="watermark box as x,y,w,h in pixels (instead of --preset)")
    p.add_argument("--dilate", type=int, default=None,
                   help="grow the mask by N px to swallow the soft edge (default 5)")
    p.add_argument("--feather", type=int, default=None,
                   help="blend width in px; hides the seam (default 12)")
    p.add_argument("--margin", type=int, default=None,
                   help="context px around the box given to the inpainter (default 64). "
                        "Bigger is slower: cost scales with tile area")


def _add_detect_args(p: argparse.ArgumentParser) -> None:
    """Knobs that decide the box, shared by `detect` and by the commands that use it.

    `run` and `batch` detect on their own when given neither --box nor --preset, so
    every one of these has to be reachable from them too. It was not: the call site
    inside `run` passed only --corner and hardcoded the rest, which meant tuning any
    of them cost a separate `detect` invocation, a preset file, and a second command
    -- for a value the run was about to compute anyway.
    """
    p.add_argument("--corner", choices=CORNERS, default="tr",
                   help="which corner to search (default tr = top-right)")
    p.add_argument("--samples", type=int, default=40,
                   help="frames sampled across the clip (default 40)")
    p.add_argument("--roi-frac", type=float, default=0.30,
                   help="corner search window as a fraction of the frame (default 0.30)")
    p.add_argument("--grad-threshold", type=float, default=None,
                   help="edge strength threshold. Default: swept automatically from "
                        "10 down to 1.5, stopping where the box stops growing -- that "
                        "finds faint marks a fixed threshold misses. Pass a number to "
                        "override")
    p.add_argument("--persistence", type=float, default=0.90,
                   help="fraction of sampled frames a pixel must appear in (default "
                        "0.90). This is what rejects subtitles and temporary text -- "
                        "and it also rejects a studio logo that is not on screen for "
                        "the whole film, so lower it when a second mark in the same "
                        "corner is being missed, and check the box you get")
    p.add_argument("--max-area", type=float, default=10.0,
                   help="reject a candidate larger than this %% of the frame (default 10)")


def _add_run_args(p: argparse.ArgumentParser) -> None:
    # On both `run` and `batch`, because both are commands that spend hours on a box.
    #
    # `warn` is the default, not `strict`, and that is a measurement rather than a
    # preference. `check_coverage` false-positives on this project's own reference
    # fixture: measured on tests/fixtures/detail-marked.mp4, both the box the README
    # uses for its smoke test (379,427,91,43) and the fixture's ground-truth badge
    # (384,430,84,36) come back UNDER-COVERED with only 1.29% and 1.74% of ring pixels
    # flagged -- the `detail` fixture puts the badge on dense static texture, which is
    # the inflating case coverage.py's own docstring warns about. A `strict` default
    # would turn a documented, working invocation into a hard failure.
    #
    # So the CLI reports and continues; the caller that needs a hard stop asks for one.
    # The service does, because there a blocked job is reviewable by a human and an
    # unnoticed watermark fringe is not.
    p.add_argument("--coverage-gate", choices=("strict", "warn", "off"),
                   default="warn",
                   help="check the box covers the whole mark before processing. "
                        "warn (default) = report the verdict and process anyway; "
                        "strict = stop if the mark reaches outside the box, or if the "
                        "check cannot tell (use this for unattended runs); off = skip "
                        "the check. The check is one sampling pass, so it is not a "
                        "speed lever")
    p.add_argument("--detect", action="store_true",
                   help="find the watermark and process in one go, no preset needed. "
                        "For 'batch' it detects once on the first file and applies "
                        "that box to all of them. A preview PNG is written either "
                        "way -- check it, detection is a guess")
    _add_detect_args(p)
    # Default is ProPainter. It is the only engine that reaches zero residual without
    # inventing content per frame -- measured on the reference clip, residual 11.36
    # against a background floor of 12.24, so nothing findable is left, at temporal
    # correlation 0.82. Un-blend gets 13.02 at correlation 0.99: it cannot flicker and
    # it damages nothing, but its leftover is proportional to the error in an alpha it
    # has to estimate from statistics that are unobservable under the mark, so it can
    # never reach zero and the mark stays faintly detectable.
    #
    # The cost of this default is a hard dependency on a GPU: 0.27 fps on six CPU
    # cores, ~1.8 hours per minute of 1080p. That is not a slow default, it is a
    # non-functional one, so the video path refuses to run on CPU unless --device cpu
    # says to. Un-blend remains the right answer without a card, and for a
    # semi-transparent mark over detail it is still the least destructive engine here.
    p.add_argument("--quality", choices=("unblend", "video", "high", "fast", "draft"),
                   default="video",
                   help="video = ProPainter, fills from neighbouring frames (default) "
                        "-- the only one that removes the mark COMPLETELY without "
                        "inventing per frame. Needs a GPU; "
                        "unblend = solve the alpha blend and RECOVER the real "
                        "background -- keeps every pixel and cannot flicker, runs at "
                        "34 fps on CPU, but leaves a faint trace; "
                        "high = LaMa inpainting on a crop tile, per frame; "
                        "fast = ffmpeg delogo+feather, smears on texture; "
                        "draft = cv2.inpaint, lowest quality")
    p.add_argument("--propainter", default=None,
                   help="path to the ProPainter checkout (else $PROPAINTER_HOME, "
                        "else a sibling directory of this project)")
    p.add_argument("--pp-segment", type=int, default=None,
                   help="MAXIMUM frames per ProPainter invocation. Default: worked out "
                        "from free VRAM, available RAM and the tile size, and printed "
                        "with its reasoning. Segments also stop at scene cuts, so most "
                        "are shorter. Pin a number to reproduce a run or to bisect an "
                        "out-of-memory failure")
    # On by default, and the asymmetry is the point. Off by default, the failure mode
    # is "you forgot a flag, so nine hours of finished parts were deleted before you
    # could read the message" -- silent, unrecoverable, and only visible afterwards.
    # On by default, the failure mode is "it reused work it should not have", and
    # that one cannot happen: the manifest fingerprints the source and every setting
    # that decides a pixel, and a mismatch starts over on its own and says so.
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="(the default) carry on from where a killed run stopped. The "
                        "video is composited in parts of --pp-part frames next to the "
                        "output; a run that dies leaves the finished ones behind, and "
                        "this reuses every part whose frame count checks out. Also "
                        "reuses the recorded scene cuts and segment size, so the "
                        "second half of a video is never made with different settings "
                        "from the first. Starts over on its own if the source or any "
                        "setting that decides a pixel has changed")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="delete any finished parts and start from frame zero. Only "
                        "needed to discard work the fingerprint would have accepted -- "
                        "a changed input or setting is already detected on its own")
    p.add_argument("--pp-part", type=int, default=3600,
                   help="frames per composited part (default 3600, two minutes at "
                        "30fps). This is what a crash costs you: at most one part of "
                        "model time and encoding. Output is identical whatever it is "
                        "set to -- parts are cut on a fixed grid, not on the model's "
                        "segments -- so it trades restart granularity against the "
                        "per-part ffmpeg startup")
    p.add_argument("--pp-scene-threshold", type=float, default=0.3,
                   help="ffmpeg scene score above which a frame starts a new shot "
                        "(default 0.3). Segments never span a cut, because the model "
                        "fills from other frames and across a cut those belong to a "
                        "different scene: measured, a 30-frame shot inside a 440-frame "
                        "segment had the watermark region filled with the wrong "
                        "scene's content for its whole duration. 0 disables detection "
                        "and cuts every --pp-segment frames regardless of content. "
                        "Note what it cannot do: a fade through black is gradual and "
                        "never scores at any threshold, which is what --pp-black-cuts "
                        "is for")
    p.add_argument("--pp-no-black-cuts", action="store_false", dest="pp_black_cuts",
                   help="do not end a segment where the picture goes to or comes back "
                        "from black. On by default, and it is not a second opinion on "
                        "the scene score: a fade through black scored below 0.1 on a "
                        "real intro while the picture went from black to full "
                        "brightness, so the black run shared a segment with the shot "
                        "after it and the model filled the watermark hole on the black "
                        "frames from the bright shot -- a glowing smear over an "
                        "otherwise black frame, 40 frames wide, which is exactly the "
                        "reach of its global references. Costs one filter in the scene "
                        "scan, no extra decode")
    p.add_argument("--pp-min-shot", type=int, default=16,
                   help="shots shorter than this are merged with the previous one "
                        "(default 16). Below roughly a dozen frames the model has "
                        "nothing to propagate from, so an impure segment beats an "
                        "empty one")
    p.add_argument("--pp-subvideo", type=int, default=80,
                   help="frames per inference chunk inside one segment (default 80). "
                        "This, not --pp-segment, is what caps VRAM during inference: "
                        "measured on an A40, a 1827-frame segment held only 10 GiB of "
                        "45 because the model still worked 80 frames at a time. "
                        "Raising it buys nothing measurable though -- 80 vs 200 on "
                        "that A40 came out 284s vs 283s, while two runs of the same "
                        "setting varied by 19s, so the knob is inside the noise. "
                        "Left exposed for other hardware, not as a recommendation")
    p.add_argument("--pp-workers", type=int, default=1,
                   help="segments to run concurrently (default 1). Output is "
                        "identical either way, so this is purely a speed knob. How "
                        "much it buys depends on the card: measured on an A40 a "
                        "single segment already sat at 99%% utilisation and 282W of "
                        "a 300W limit, so the gain there comes from overlapping the "
                        "per-segment startup, not from filling an idle GPU. Plenty "
                        "of free VRAM is not evidence it will help -- compare the "
                        "reported fps")
    p.add_argument("--raft-iter", type=int, default=20,
                   help="ProPainter optical-flow iterations (default 20). Lower is "
                        "faster and less accurate")
    p.add_argument("--no-fp16", action="store_true",
                   help="disable half precision in ProPainter (ignored on CPU)")
    p.add_argument("--unblend-samples", type=int, default=40,
                   help="frames used to fit the un-blend map (default 40)")
    # No --stroke-alpha flag. Repairing only the glyph strokes was implemented and
    # measured, and it does not work -- see UnblendBackend for the numbers. Exposing
    # a knob that cannot deliver just adds a decision nobody can make correctly.
    p.add_argument("--crf", type=int, default=18, help="x264 CRF (default 18)")
    p.add_argument("--x264-preset", default="medium", help="x264 preset (default medium)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
                   help="where to run LaMa (default auto: CUDA if available). "
                        "CUDA is roughly 20-50x faster than CPU")
    p.add_argument("--threads", type=int, default=None,
                   help="torch threads, CPU only (default: all cores)")
    p.add_argument("--cache-tolerance", type=float, default=1.0,
                   help="reuse the previous patch when the tile changed less than this "
                        "(mean abs 8-bit levels). 0 disables")
    p.add_argument("--patch-hold", type=int, default=1,
                   help="reuse each inpainted patch for N frames (default 1 = every "
                        "frame fresh). Raising it suppresses boiling and speeds the run "
                        "up N-fold, at the cost of the patch lagging a moving background")
    p.add_argument("--quiet", action="store_true", help="no progress line")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wmrm",
        description="Remove a fixed-position watermark from video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="version", version=f"wmrm {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("detect", help="find the watermark once and write a preset")
    d.add_argument("input")
    _add_detect_args(d)
    d.add_argument("--preview", help="where to write the preview PNG")
    _add_region_args(d)
    d.set_defaults(func=cmd_detect)

    r = sub.add_parser("run", help="process one video")
    r.add_argument("input")
    r.add_argument("-o", "--output", help=f"default: NAME{CLEAN_SUFFIX}.EXT next to input")
    r.add_argument("--preview-only", action="store_true",
                   help="draw the box on a frame and exit without processing")
    r.add_argument("--no-verify", action="store_true", help="skip the acceptance checks")
    r.add_argument("--report", default=None, metavar="FILE",
                   help="write a JSON result to FILE -- written on every exit path, "
                        "including failures. This is how a caller learns WHICH failure "
                        "happened: the exit status cannot say, because 1 today means "
                        "any of a missing input, a usage mistake, an ffmpeg error and "
                        "failed verification")
    r.add_argument("--no-lock", action="store_true",
                   help="skip the per-output lock. Only for a case where two runs must "
                        "share an output path, which is not a case that exists yet")
    _add_region_args(r)
    _add_run_args(r)
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="process every video in a directory")
    b.add_argument("directory")
    b.add_argument("--force", action="store_true", help="reprocess even if output exists")
    b.add_argument("--no-verify", action="store_true")
    b.add_argument("--outbox", default=None,
                   help="write results here instead of next to each input")
    b.add_argument("--detect-each", action="store_true",
                   help="detect a box per video instead of once for the folder. Right "
                        "when the batch is mixed -- a box measured on one source says "
                        "nothing about a different crop, logo or placement. It costs "
                        "the one place a human confirmed the box, so every detected "
                        "box is checked with the coverage test before anything is "
                        "processed")
    b.add_argument("--no-coverage-gate", action="store_true",
                   help="process even when the coverage check says the detected box is "
                        "too small. Not a speed lever: it turns a stopped file into a "
                        "silently bad one")
    _add_region_args(b)
    _add_run_args(b)
    b.set_defaults(func=cmd_batch, preview_only=False)

    # `pick` (browser box-drawing) is intentionally not wired up: it belongs with
    # the UI work, not the CLI. See pick.py.

    g = sub.add_parser("grid", help="dump a frame with a coordinate grid, to measure by hand")
    g.add_argument("input")
    g.add_argument("-o", "--output", help="default: NAME-grid.png")
    g.add_argument("--corner", choices=CORNERS, default="tr",
                   help="zoom into this corner (default tr). Pass --frac 1 for the "
                        "whole frame")
    g.add_argument("--frac", type=float, default=0.30,
                   help="corner size as a fraction of the frame (default 0.30)")
    g.add_argument("--step", type=int, default=25,
                   help="gridline spacing in source pixels (default 25)")
    g.add_argument("--scale", type=int, default=3, help="zoom factor (default 3)")
    g.add_argument("--at", type=float, default=None,
                   help="timestamp in seconds (default: middle of the clip)")
    g.set_defaults(func=cmd_grid)

    c = sub.add_parser("coverage",
                       help="check a box covers the whole watermark (any type)")
    c.add_argument("input")
    c.add_argument("--samples", type=int, default=30,
                   help="frames sampled across the clip (default 30)")
    c.add_argument("--ring", type=int, default=48,
                   help="how far outside the box to look, in px (default 48)")
    c.add_argument("--grad-threshold", type=float, default=2.0,
                   help="edge threshold for the gradient signal (default 2)")
    _add_region_args(c)
    c.set_defaults(func=cmd_coverage)

    pull = sub.add_parser("pull", help="download a source video from Cloudflare R2")
    pull.add_argument("key",
                      help="object key, or r2://bucket/key. "
                           "e.g. uploads/3d80.../MOGI-125.mp4")
    pull.add_argument("-o", "--output", default=None,
                      help="file or directory to write to (default: cwd)")
    pull.add_argument("--bucket", default=None,
                      help="bucket name (default: $R2_BUCKET, or the one in the URI)")
    pull.add_argument("--workers", type=int, default=8,
                      help="parallel ranged GETs (default 8). More only helps until "
                           "the link or the disk saturates -- watch the reported rate")
    pull.add_argument("--chunk-mib", type=int, default=64,
                      help="chunk size in MiB (default 64). This is also the resume "
                           "granularity: an interrupted chunk is re-fetched whole")
    pull.add_argument("--force", action="store_true",
                      help="re-download even if the file is already there")
    pull.add_argument("--stat", action="store_true",
                      help="print the object's size and etag, download nothing")
    pull.add_argument("--list", action="store_true",
                      help="list keys under KEY as a prefix, download nothing")
    pull.add_argument("--limit", type=int, default=200, help="--list cap (default 200)")
    pull.add_argument("--quiet", action="store_true", help="no progress line")
    pull.set_defaults(func=cmd_pull)

    s = sub.add_parser("serve", help="run the pod HTTP API (needs the 'serve' extra)")
    s.add_argument("--host", default="0.0.0.0",
                   help="default 0.0.0.0 -- a RunPod proxy cannot reach localhost")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--log-level", default="info",
                   choices=("critical", "error", "warning", "info", "debug", "trace"))
    s.set_defaults(func=cmd_serve)

    v = sub.add_parser("verify", help="compare an original and a processed file")
    v.add_argument("original")
    v.add_argument("processed")
    _add_region_args(v)
    v.set_defaults(func=cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (WmrmError, ToolMissing, ProbeError, DetectError, EncodeError,
            ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception:  # pragma: no cover
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
