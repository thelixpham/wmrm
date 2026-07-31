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

def _resolve_region(args, width: int, height: int) -> tuple[Box, Preset]:
    """Return the box to use plus the knobs, from --box or --preset."""
    if args.box and args.preset:
        raise SystemExit("error: pass either --box or --preset, not both")
    if args.box:
        box = Box.parse(args.box).clamp(width, height)
        preset = Preset.from_box(box, width, height)
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
            "error: need --preset wm.json or --box x,y,w,h\n"
            "  run 'wmrm detect YOUR.mp4' first to create a preset"
        )

    overrides = {
        f"{name}_px": getattr(args, name)
        for name in ("dilate", "feather", "margin")
        if getattr(args, name, None) is not None
    }
    if overrides:
        preset = replace(preset, **overrides)
    return box, preset


def _default_output(src: Path) -> Path:
    return src.with_name(f"{src.stem}{CLEAN_SUFFIX}{src.suffix or '.mp4'}")


def _process_one(src: Path, dst: Path, box: Box, preset: Preset, args, backend):
    encode = EncodeOpts(crf=args.crf, x264_preset=args.x264_preset)
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


def _make_backend(args):
    if args.quality == "fast":
        return None
    from .backends import make_backend
    quality = "draft" if args.quality == "draft" else "high"
    return make_backend(quality, threads=args.threads,
                        cache_tolerance=args.cache_tolerance,
                        patch_hold=args.patch_hold, device=args.device)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_detect(args) -> int:
    src = Path(args.input)
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

    preview = Path(args.preview) if args.preview else src.with_name(f"{src.stem}-preview.png")
    zoom = preview.with_name(f"{preview.stem}-zoom{preview.suffix}")
    write_preview(src, det.box, preview, roi=det.roi, zoom_png=zoom)

    preset_path = Path(args.preset) if args.preset else src.with_name("wm-preset.json")
    info = probe(src)
    preset = Preset.from_box(
        det.box, info.width, info.height,
        opacity=det.opacity,
        dilate_px=args.dilate if args.dilate is not None else 5,
        feather_px=args.feather if args.feather is not None else 12,
        margin_px=args.margin if args.margin is not None else 64,
    )
    preset.save(preset_path)

    print(f"\npreset written -> {preset_path}")
    print(f"preview        -> {preview}  (red = watermark box, orange = search area)")
    print(f"zoomed preview -> {zoom}")
    if det.opacity == "semi":
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
    box, preset = _resolve_region(args, info.width, info.height)
    dst = Path(args.output) if args.output else _default_output(src)

    if args.preview_only:
        out = dst.with_name(f"{src.stem}-boxcheck.png")
        write_preview(src, box, out, zoom_png=out.with_name(f"{src.stem}-boxcheck-zoom.png"))
        print(f"box {box.as_tuple()} drawn on {out}; nothing processed")
        return 0

    backend = _make_backend(args)
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
    backend = _make_backend(args)
    failures = []
    for i, (src, dst) in enumerate(todo, 1):
        print(f"\n[wmrm] ({i}/{len(todo)}) {src.name}", file=sys.stderr)
        try:
            info = probe(src)
            box, preset = _resolve_region(args, info.width, info.height)
            _process_one(src, dst, box, preset, args, backend)
        except (EncodeError, ProbeError, DetectError) as exc:
            print(f"[wmrm] FAILED {src.name}: {exc}", file=sys.stderr)
            failures.append(src.name)

    if failures:
        print(f"\n[wmrm] {len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\n[wmrm] all {len(todo)} done", file=sys.stderr)
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


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--quality", choices=("high", "fast", "draft"), default="high",
                   help="high = LaMa on a crop tile (default, best); "
                        "fast = ffmpeg delogo+feather, near-realtime, smears on texture; "
                        "draft = cv2.inpaint, seconds, lowest quality")
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
    d.add_argument("--grad-threshold", type=float, default=10.0,
                   help="edge strength threshold (default 10); lower finds fainter logos")
    d.add_argument("--persistence", type=float, default=0.90,
                   help="fraction of sampled frames a pixel must appear in (default 0.90). "
                        "This is what rejects subtitles and temporary text")
    d.add_argument("--max-area", type=float, default=10.0,
                   help="reject a candidate larger than this %% of the frame (default 10)")
    d.add_argument("--preview", help="where to write the preview PNG")
    _add_region_args(d)
    d.set_defaults(func=cmd_detect, preset=None, box=None)

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
