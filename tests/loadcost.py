"""What does one ProPainter invocation waste before it processes anything?

`--quality video` spawns a fresh process per segment, and each one pays for Python
imports, CUDA context creation, and three checkpoints. This measures that fixed cost
so the decision to keep spawning (or to refactor into a single long-lived process) is
made on a number rather than a guess.

    python tests/loadcost.py                 # uses $PROPAINTER_HOME or a sibling clone
    python tests/loadcost.py /path/to/ProPainter

Measured on 6 CPU cores: 2.74s, of which 2.17s was Python imports rather than
weights. On a CUDA box expect meaningfully more -- CUDA context creation and the
first-conv autotune are paid here too, and they do not exist on CPU. That gap is the
whole question, which is why this must be run on the machine that will do the work.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmrm.video import find_repo  # noqa: E402

# Frame counts to scale to, mirroring the segment sizes that matter in practice.
SEGMENTS = (400, 1500, 4000)


def main(argv: list[str]) -> int:
    repo = find_repo(argv[0] if argv else None)
    weights = repo / "weights"
    missing = [n for n in ("raft-things.pth", "recurrent_flow_completion.pth",
                           "ProPainter.pth") if not (weights / n).is_file()]
    if missing:
        print(f"error: weights not downloaded yet: {', '.join(missing)}\n"
              f"Run `wmrm run ... --quality video` once; they fetch on first use.",
              file=sys.stderr)
        return 1

    sys.path.insert(0, str(repo))
    t0 = time.monotonic()
    import torch
    from model.misc import get_device
    from model.modules.flow_comp_raft import RAFT_bi
    from model.propainter import InpaintGenerator
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    t_import = time.monotonic() - t0

    device = get_device()

    # Touch the device before timing the models, so CUDA context creation is
    # attributed to its own line instead of silently inflating the first model.
    t0 = time.monotonic()
    if device.type == "cuda":
        torch.zeros(1, device=device)
        torch.cuda.synchronize()
    t_ctx = time.monotonic() - t0

    t0 = time.monotonic()
    RAFT_bi(str(weights / "raft-things.pth"), device)
    t_raft = time.monotonic() - t0

    t0 = time.monotonic()
    flow = RecurrentFlowCompleteNet(str(weights / "recurrent_flow_completion.pth"))
    for p in flow.parameters():
        p.requires_grad = False
    flow.to(device).eval()
    t_flow = time.monotonic() - t0

    t0 = time.monotonic()
    InpaintGenerator(model_path=str(weights / "ProPainter.pth")).to(device).eval()
    t_model = time.monotonic() - t0

    total = t_import + t_ctx + t_raft + t_flow + t_model
    print(f"\ndevice           : {device}")
    print(f"python imports   : {t_import:6.2f}s")
    print(f"cuda context     : {t_ctx:6.2f}s")
    print(f"RAFT             : {t_raft:6.2f}s")
    print(f"flow completion  : {t_flow:6.2f}s")
    print(f"ProPainter       : {t_model:6.2f}s")
    print("-" * 25)
    print(f"per invocation   : {total:6.2f}s   <- paid once per segment today\n")

    for hours in (1, 3):
        frames = int(hours * 3600 * 30)
        print(f"{hours}h of 30 fps video ({frames} frames):")
        for seg in SEGMENTS:
            n = -(-frames // seg)
            waste = n * total
            print(f"  --pp-segment {seg:>4}: {n:>5} invocations = "
                  f"{waste / 3600:5.2f} h of pure loading")
        print()

    print("Rule of thumb: refactoring to load once is worth it if the largest number")
    print("above is hours rather than minutes. If not, raise --pp-segment and stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
