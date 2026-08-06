"""Measure what the ProPainter speed knobs actually cost in quality.

`--margin` and `--raft-iter` are the only two levers that touch the 72% of the run
that is the model. Both trade quality for time and neither trade is guessable:
`--margin` shrinks the tile, so cost falls with its area, but the tile is also the
context the model fills from; `--raft-iter` makes the optical flow rougher, and rough
flow is what pulls the wrong pixels in.

So this runs the real pipeline at each setting and reports time *and* the artifact
signals side by side. It exists because "9.11 fps" is not a result you can act on
without knowing what the frame looks like afterwards.

Every config runs in this one process, so the models load once for the whole sweep --
which is also a live test of the worker cache in video.py.

    python tests/sweep_propainter.py clip.mp4 --box 1554,44,283,64
    python tests/sweep_propainter.py clip.mp4 --preset p.json --margins 64,32 --rafts 20,12

Read the output as a comparison between rows, not as an absolute verdict. It measures
one clip, and `fill` is a statistic, not an opinion about how the frame looks: a config
with fewer flagged frames that visibly smears would score well here. Look at the best
one or two before shipping a setting.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from find_artifacts import robust_z, signals                     # noqa: E402
from wmrm.pipeline import EncodeOpts                             # noqa: E402
from wmrm.probe import probe, require_tools                      # noqa: E402
from wmrm.region import Box, Preset, build_region                # noqa: E402
from wmrm.video import ProPainterOpts, find_repo, run_propainter  # noqa: E402


def ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--box", help="x,y,w,h")
    ap.add_argument("--preset")
    ap.add_argument("--outdir", default="/tmp/wmrm-sweep")
    ap.add_argument("--margins", default="64,48,32")
    ap.add_argument("--rafts", default="20,12")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--segment", type=int, default=400)
    ap.add_argument("--z", type=float, default=6.0)
    ap.add_argument("--keep", action="store_true",
                    help="keep the rendered videos (default: they are left in outdir "
                         "anyway; this only suppresses the reminder)")
    args = ap.parse_args()

    ffmpeg, _ = require_tools()
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"error: {src} not found")
    info = probe(src)

    if args.preset:
        preset = Preset.load(Path(args.preset)).scaled_px(info.width, info.height)
        box = preset.box_for(info.width, info.height)
        dilate, feather = preset.dilate_px, preset.feather_px
    elif args.box:
        x, y, w, h = ints(args.box)
        box = Box(x, y, w, h)
        dilate, feather = 5, 12
    else:
        raise SystemExit("error: need --box or --preset")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    repo = find_repo()

    configs = [(m, r) for m in ints(args.margins) for r in ints(args.rafts)]
    print(f"{src.name}: {info.width}x{info.height}, {info.nframes} frames, "
          f"{info.duration:.1f}s")
    print(f"box {box.as_tuple()}  dilate {dilate}  feather {feather}")
    print(f"{len(configs)} config(s), models load once for the whole sweep\n")

    rows = []
    for margin, raft in configs:
        out = outdir / f"{src.stem}-m{margin}-r{raft}.mp4"
        region = build_region(box, info.width, info.height, dilate_px=dilate,
                              feather_px=feather, margin_px=margin)
        tile = region.tile
        label = f"margin {margin:>3}  raft {raft:>3}"
        print(f"--- {label}  tile {tile.w}x{tile.h} ({tile.w * tile.h} px)")

        t0 = time.monotonic()
        try:
            run_propainter(
                src, out, box=box, dilate_px=dilate, feather_px=feather,
                margin_px=margin,
                opts=ProPainterOpts(repo=repo, device=args.device, segment=args.segment,
                                    raft_iter=raft),
                encode=EncodeOpts(), progress=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"    FAILED: {exc}")
            rows.append((label, tile, None, None, None, None, None))
            continue
        elapsed = time.monotonic() - t0

        f, j = signals(ffmpeg, src, out, tile, region.inpaint_mask > 0)
        if len(f) == 0:
            print("    FAILED: measured 0 frames")
            rows.append((label, tile, elapsed, None, None, None, None))
            continue
        zf, zj = robust_z(f), robust_z(j)
        flagged = int(((zf > args.z) | (zj > args.z)).sum())
        rows.append((label, tile, elapsed, float(np.median(f)), float(f.max()),
                     flagged, float(max(zf.max(), zj.max()))))
        print(f"    {elapsed:.1f}s   fill median {np.median(f):.2f}  max {f.max():.2f}"
              f"  flagged {flagged}/{len(f)}\n")

    print("\n" + "=" * 88)
    print(f"{'config':<22} {'tile':>11} {'time':>8} {'vs base':>8} "
          f"{'fill med':>9} {'fill max':>9} {'flagged':>8} {'worst z':>8}")
    print("-" * 88)
    base = next((r[2] for r in rows if r[2]), None)
    for label, tile, elapsed, fmed, fmax, flagged, worst in rows:
        if elapsed is None:
            print(f"{label:<22} {f'{tile.w}x{tile.h}':>11} {'failed':>8}")
            continue
        rel = f"{100 * elapsed / base:.0f}%" if base else "-"
        cells = (f"{fmed:.2f}" if fmed is not None else "-",
                 f"{fmax:.2f}" if fmax is not None else "-",
                 f"{flagged}" if flagged is not None else "-",
                 f"{worst:.1f}" if worst is not None else "-")
        print(f"{label:<22} {f'{tile.w}x{tile.h}':>11} {elapsed:>7.1f}s {rel:>8} "
              f"{cells[0]:>9} {cells[1]:>9} {cells[2]:>8} {cells[3]:>8}")
    print("=" * 88)
    print("`vs base` is against the first row that completed, not against main.")
    print("`flagged` counts frames beyond the z threshold on either signal -- lower is")
    print("better, but a config that smears everything uniformly also flags nothing.")
    print(f"\nRendered videos are in {outdir}. Watch the best two before choosing.")
    if not args.keep:
        print("They are not deleted: judging this by eye is the point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
