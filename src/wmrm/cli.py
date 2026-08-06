"""Command line interface.

    wmrm detect  IN.mp4 [--corner tr] [--preset wm.json]   # calibrate once
    wmrm run     IN.mp4 [-o OUT.mp4] --preset wm.json      # or --box x,y,w,h
    wmrm batch   DIR --preset wm.json [-j 2]
    wmrm verify  ORIG.mp4 OUT.mp4 --preset wm.json
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import replace
from pathlib import Path

from . import __version__
from .detect import DetectError, detect, write_preview
from .pipeline import EncodeError, EncodeOpts, run_fast, run_inpaint
from .probe import ProbeError, ToolMissing, probe
from .region import CORNERS, Box, Preset
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
        raise SystemExit("error: pass either --box or --preset, not both")
    if args.box:
        box = Box.parse(args.box).clamp(width, height)
        preset = Preset.from_box(box, width, height)
    elif not args.preset and getattr(args, "detect", False):
        if src is None:
            raise SystemExit("error: --detect needs an input file")
        # Detection is a guess with measured failure modes, so a preview is
        # always written and every processed file still goes through the usual
        # verification. If detection finds nothing it raises, which aborts the
        # run rather than quietly processing with a bogus box.
        det = detect(src, corner=args.corner)
        print(det.describe())
        preview = src.with_name(f"{src.stem}-preview.png")
        write_preview(src, det.box, preview, roi=det.roi,
                      zoom_png=preview.with_name(f"{preview.stem}-zoom.png"))
        print(f"\nCHECK {preview.with_name(preview.stem + '-zoom.png')} afterwards. "
              "Detection is a guess, not a guarantee.\n")
        box = det.box
        preset = Preset.from_box(box, width, height, opacity=det.opacity)
    elif args.preset:
        path = Path(args.preset)
        if not path.exists():
            raise SystemExit(
                f"error: preset {path} not found. Create it with:\n"
                f"  wmrm detect YOUR.mp4 --preset {path}"
            )
        preset = Preset.load(path).scaled_px(width, height)
        box = preset.box_for(width, height)
    else:
        raise SystemExit(
            "error: need one of --preset wm.json, --box x,y,w,h, or --detect\n"
            "  --detect          find the watermark and process in one go\n"
            "  --box x,y,w,h     coordinates you measured (see 'wmrm grid')\n"
            "  --preset wm.json  saved coordinates (see 'wmrm detect')"
        )

    overrides = {
        f"{name}_px": getattr(args, name)
        for name in ("dilate", "feather", "margin")
        if getattr(args, name, None) is not None
    }
    if overrides:
        preset = replace(preset, **overrides)
    return box, preset


def _log_config(src: Path, dst: Path, info, box: Box, preset: Preset, args,
                backend) -> None:
    """Print the effective configuration once, before any frames are touched.

    Mostly so a wrong device is impossible to miss: the CPU-only torch wheel on a
    GPU machine runs 20-50x slower and produces byte-identical output, so nothing
    downstream reveals the mistake.
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
            opts=ProPainterOpts(repo=find_repo(args.propainter),
                                segment=args.pp_segment,
                                raft_iter=args.raft_iter,
                                fp16=not args.no_fp16,
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
                  preset: Preset | None = None):
    # Both of these own their whole clip: `fast` is one ffmpeg graph, `video` is a
    # sequence model that needs many frames at once. Neither fits the per-frame
    # Backend interface, so there is nothing to build here.
    if args.quality in ("fast", "video"):
        return None
    from .backends import make_backend

    fitted = None
    if args.quality == "unblend":
        if src is None or box is None or preset is None:
            raise SystemExit("error: --quality unblend needs an input file and a box")
        fitted = _fit_unblend(src, box, preset, samples=args.unblend_samples)

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
            "\nNOTE: the watermark looks semi-transparent (background bleeds through).\n"
            "      Alpha un-blend could recover the original almost exactly; that path\n"
            "      is not implemented yet -- inpainting is used for now."
        )
    print(
        "\nLOOK AT THE PREVIEW before running. Nothing was processed.\n"
        "If the box is right:\n"
        f"  wmrm run {src} --preset {preset_path}\n"
        "If it is wrong, measure it yourself and pass it directly:\n"
        f"  wmrm run {src} --box x,y,w,h"
    )
    return 0


def cmd_run(args) -> int:
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"error: {src} not found")
    info = probe(src)
    box, preset = _resolve_region(args, info.width, info.height, src=src)
    dst = Path(args.output) if args.output else _default_output(src)

    if args.preview_only:
        out = dst.with_name(f"{src.stem}-boxcheck.png")
        write_preview(src, box, out, zoom_png=out.with_name(f"{src.stem}-boxcheck-zoom.png"))
        print(f"box {box.as_tuple()} drawn on {out}; nothing processed")
        return 0

    backend = _make_backend(args, src=src, box=box, preset=preset)
    _log_config(src, dst, info, box, preset, args, backend)
    _process_one(src, dst, box, preset, args, backend)

    if not args.no_verify:
        result = run_verify(src, dst, box)
        print("\n[wmrm] verification:")
        print(result.render())
        if not result.ok:
            return 1
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

    todo = []
    for src in videos:
        dst = _default_output(src)
        if dst.exists() and not args.force:
            print(f"[wmrm] skip (exists): {dst.name}", file=sys.stderr)
            continue
        todo.append((src, dst))
    if not todo:
        print("[wmrm] nothing to do")
        return 0

    print(f"[wmrm] {len(todo)} video(s) to process", file=sys.stderr)

    # With --detect, detect ONCE on the first video and reuse it for the whole
    # folder. Detecting per file is what makes unattended batch dangerous: every
    # file gets a different box, so one bad guess corrupts one file and you have
    # no single thing to eyeball. One box, one preview, one decision.
    shared: Preset | None = None
    if getattr(args, "detect", False) and not args.box and not args.preset:
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
    # One fitted map for the folder, same reasoning as one box for the folder.
    backend = _make_backend(args, src=first_src, box=first_box, preset=first_preset)
    failures = []
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
            _log_config(src, dst, info, box, preset, args, backend)
            _process_one(src, dst, box, preset, args, backend)
        except (EncodeError, ProbeError, DetectError) as exc:
            print(f"[wmrm] FAILED {src.name}: {exc}", file=sys.stderr)
            failures.append(src.name)

    if failures:
        print(f"\n[wmrm] {len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\n[wmrm] all {len(todo)} done", file=sys.stderr)
    return 0


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
    return 0 if result.ok else 1


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


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--detect", action="store_true",
                   help="find the watermark and process in one go, no preset needed. "
                        "For 'batch' it detects once on the first file and applies "
                        "that box to all of them. A preview PNG is written either "
                        "way -- check it, detection is a guess")
    p.add_argument("--corner", choices=CORNERS, default="tr",
                   help="corner to search with --detect (default tr)")
    # Default is un-blend, not LaMa. On the footage this tool was built for the
    # mark is semi-transparent, so the background is still present in the signal
    # and recovering it beats generating a replacement on every axis that was
    # measured: temporal correlation 0.98 vs 0.30, 34 fps vs 1.4, and surrounding
    # detail left intact instead of repainted. LaMa stays available for genuinely
    # opaque marks, where there is nothing to recover.
    p.add_argument("--quality", choices=("unblend", "video", "high", "fast", "draft"),
                   default="unblend",
                   help="unblend = solve the alpha blend and RECOVER the real "
                        "background (default) -- keeps every pixel, but leaves a "
                        "faint trace; "
                        "video = ProPainter, fills from neighbouring frames -- the "
                        "only one that removes the mark COMPLETELY without "
                        "inventing per frame. Needs a GPU and a ProPainter "
                        "checkout; "
                        "high = LaMa inpainting on a crop tile, per frame; "
                        "fast = ffmpeg delogo+feather, smears on texture; "
                        "draft = cv2.inpaint, lowest quality")
    p.add_argument("--propainter", default=None,
                   help="path to the ProPainter checkout (else $PROPAINTER_HOME, "
                        "else a sibling directory of this project)")
    p.add_argument("--pp-segment", type=int, default=400,
                   help="frames per ProPainter invocation (default 400). It loads a "
                        "whole segment into memory, so lower this if you run out")
    p.add_argument("--pp-workers", type=int, default=1,
                   help="segments to run concurrently (default 1). The tile is small "
                        "for a modern GPU, so one segment leaves it mostly idle and "
                        "2-4 often scales nearly linearly. Output is identical either "
                        "way. Watch the reported fps and back off on out-of-memory")
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
    d.add_argument("--corner", choices=CORNERS, default="tr",
                   help="which corner to search (default tr = top-right)")
    d.add_argument("--samples", type=int, default=40,
                   help="frames sampled across the clip (default 40)")
    d.add_argument("--roi-frac", type=float, default=0.30,
                   help="corner search window as a fraction of the frame (default 0.30)")
    d.add_argument("--grad-threshold", type=float, default=None,
                   help="edge strength threshold. Default: swept automatically from "
                        "10 down to 1.5, stopping where the box stops growing -- that "
                        "finds faint marks a fixed threshold misses. Pass a number to "
                        "override")
    d.add_argument("--persistence", type=float, default=0.90,
                   help="fraction of sampled frames a pixel must appear in (default 0.90). "
                        "This is what rejects subtitles and temporary text")
    d.add_argument("--max-area", type=float, default=10.0,
                   help="reject a candidate larger than this %% of the frame (default 10)")
    d.add_argument("--preview", help="where to write the preview PNG")
    _add_region_args(d)
    d.set_defaults(func=cmd_detect)

    r = sub.add_parser("run", help="process one video")
    r.add_argument("input")
    r.add_argument("-o", "--output", help=f"default: NAME{CLEAN_SUFFIX}.EXT next to input")
    r.add_argument("--preview-only", action="store_true",
                   help="draw the box on a frame and exit without processing")
    r.add_argument("--no-verify", action="store_true", help="skip the acceptance checks")
    _add_region_args(r)
    _add_run_args(r)
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="process every video in a directory")
    b.add_argument("directory")
    b.add_argument("--force", action="store_true", help="reprocess even if output exists")
    b.add_argument("--no-verify", action="store_true")
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
    except (ToolMissing, ProbeError, DetectError, EncodeError,
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
