"""`wmrm_worker` must produce what `inference_propainter.py` produces.

The worker duplicates upstream's inference body so the models can stay loaded across
segments. That is a fork of logic we do not own, and the failure mode is quiet: an
upstream fix, or a transcription slip in a stage boundary, changes pixels without
changing shapes or raising anything. This runs both paths on identical input and
compares.

Two checks, testing different things:

1. **Parity** on a tile whose size is a multiple of 8, where upstream's internal
   crop-and-resize is a no-op and the two paths are therefore comparable at all.
2. **No resampling** on a tile whose size is *not* a multiple of 8. Upstream crops
   to the multiple of 8 and cubic-resizes back, so pixels far from the mask -- which
   nothing should touch -- come back altered. The worker reflect-pads instead, so
   they must come back bit-identical. This is the check that pins the bug fix; it is
   asserted against the input, not against upstream, because upstream fails it.

Runs on CPU in a couple of minutes: 24 frames at 64x64 is enough to exercise every
stage, and parity does not care about clip size.

    python tests/test_propainter_parity.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
VENDOR = PROJECT / "vendor" / "ProPainter"
sys.path.insert(0, str(PROJECT / "src"))

T = 24
MASK_BOX = (24, 24, 16, 16)          # x, y, w, h inside the tile

# None = let ProPainter's own get_device() choose, which is what the reference script
# does. Override from the command line only to reproduce a specific device:
#   python tests/test_propainter_parity.py cpu
DEVICE: str | None = None


def synth_frames(h: int, w: int) -> np.ndarray:
    """A moving textured clip, BGR uint8, deterministic.

    Content has to *move*, or there is nothing for flow-guided propagation to find
    and every engine would score the same. Synthesised rather than decoded so the
    test needs no fixtures and no ffmpeg.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = np.empty((T, h, w, 3), np.uint8)
    for t in range(T):
        # Two gratings at different angles and speeds: locally detailed, globally
        # non-repeating, so a frame copied from the wrong index is visible.
        a = np.sin((xx * 0.35 + yy * 0.12 + t * 1.7)) * 110 + 128
        b = np.sin((xx * 0.08 - yy * 0.41 - t * 1.1)) * 90 + 128
        c = np.sin(((xx + yy) * 0.2 + t * 2.3)) * 70 + 128
        out[t] = np.clip(np.stack([a, b, c], -1), 0, 255).astype(np.uint8)
    return out


def make_mask(h: int, w: int) -> np.ndarray:
    x, y, bw, bh = MASK_BOX
    m = np.zeros((h, w), np.uint8)
    m[y:y + bh, x:x + bw] = 255
    return m


def run_upstream(frames: np.ndarray, mask: np.ndarray, work: Path,
                 *, mask_dilation: int, raft_iter: int) -> np.ndarray | None:
    """Invoke the unmodified script the way wmrm used to, and read its PNG frames."""
    fdir = work / "frames"
    fdir.mkdir(parents=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(fdir / f"{i:06d}.png"), f)
    mask_png = work / "mask.png"
    cv2.imwrite(str(mask_png), mask)

    out = work / "out"
    cmd = [sys.executable, "inference_propainter.py",
           "-i", str(fdir), "-m", str(mask_png), "-o", str(out),
           "--save_frames",
           "--mask_dilation", str(mask_dilation),
           "--subvideo_length", "80",
           "--neighbor_length", "10",
           "--ref_stride", "10",
           "--raft_iter", str(raft_iter)]
    res = subprocess.run(cmd, cwd=str(VENDOR), capture_output=True, text=True)
    got = out / fdir.name / "frames"
    produced = sorted(got.glob("*.png")) if got.is_dir() else []

    if len(produced) != len(frames):
        # The script writes the frames before it writes its two mp4s, so a failure
        # in imageio (missing imageio-ffmpeg is the common one) still leaves usable
        # frames. Distinguish that from a real failure rather than reporting a
        # parity failure for a broken reference.
        print(f"  upstream: rc={res.returncode}, {len(produced)}/{len(frames)} frames")
        print("  stdout:", res.stdout.strip()[-600:])
        print("  stderr:", res.stderr.strip()[-900:])
        return None
    if res.returncode != 0:
        print(f"  note: upstream exited {res.returncode} but wrote all frames "
              f"(likely its mp4 writer); using the frames")
    return np.stack([cv2.imread(str(p)) for p in produced])


def run_worker(frames: np.ndarray, mask: np.ndarray, *, mask_dilation: int,
               raft_iter: int) -> np.ndarray:
    sys.path.insert(0, str(VENDOR))
    from wmrm_worker import ProPainterWorker, WorkerOpts, describe_device  # noqa: E402

    # DEVICE is None unless overridden, so the worker falls through to upstream's own
    # get_device(). That matters more than it looks: upstream's script has no device
    # flag, so on a CUDA box it always picks cuda. Pinning the worker to cpu here
    # would compare two different devices and call the difference a parity failure.
    #
    # fp16 off on purpose. The reference is run without --fp16, and half precision is
    # not reproducible enough to assert bit-equality against an fp32 run.
    worker = ProPainterWorker(
        device=DEVICE,
        opts=WorkerOpts(mask_dilation=mask_dilation, raft_iter=raft_iter, fp16=False),
    )
    print(f"      worker device: {describe_device(worker.device)}")
    return worker.inpaint(frames, mask)


def report(name: str, a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        print(f"FAIL  {name}: shapes differ, {a.shape} vs {b.shape}")
        return False
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    n_bad = int((diff > 0).sum())
    if n_bad == 0:
        print(f"PASS  {name}: bit-identical ({a.size} values)")
        return True
    print(f"FAIL  {name}: {n_bad}/{a.size} values differ "
          f"({100 * n_bad / a.size:.4f}%), max |diff| = {int(diff.max())}, "
          f"mean |diff| = {diff.mean():.4f}")
    per_frame = [(i, int((d > 0).sum())) for i, d in enumerate(diff) if (d > 0).any()]
    print(f"      frames affected: {len(per_frame)}/{len(diff)}; "
          f"first few: {per_frame[:5]}")
    return False


def check_parity() -> bool:
    """Aligned tile: the two implementations must agree exactly."""
    h = w = 64                      # 64 % 8 == 0, so upstream's resize is a no-op
    frames, mask = synth_frames(h, w), make_mask(h, w)
    print(f"[1] parity on an aligned tile {w}x{h}, {T} frames")

    with tempfile.TemporaryDirectory(prefix="wmrm-parity-") as td:
        up = run_upstream(frames, mask, Path(td), mask_dilation=4, raft_iter=20)
    if up is None:
        print("FAIL  parity: could not run the upstream reference")
        return False
    mine = run_worker(frames, mask, mask_dilation=4, raft_iter=20)
    return report("worker vs inference_propainter.py", up, mine)


def check_no_resampling() -> bool:
    """Unaligned tile: pixels the mask never touches must survive untouched.

    Upstream cannot pass this -- 60 is not a multiple of 8, so it processes at 56
    and cubic-resizes back, which perturbs the whole frame including pixels nowhere
    near the watermark. That is the silent bug the worker's padding removes.
    """
    h = w = 60
    frames, mask = synth_frames(h, w), make_mask(h, w)
    print(f"\n[2] no resampling on an unaligned tile {w}x{h} "
          f"({w} % 8 = {w % 8}), {T} frames")

    mine = run_worker(frames, mask, mask_dilation=4, raft_iter=20)
    if mine.shape != frames.shape:
        print(f"FAIL  shape changed: {frames.shape} -> {mine.shape}")
        return False

    # Outside the mask plus its dilation, output must equal input exactly. The
    # transform blends with the binary dilated mask, so anything beyond it is
    # arithmetically the original pixel -- unless something resampled.
    x, y, bw, bh = MASK_BOX
    pad = 4 + 2                     # mask_dilation, plus slack
    untouched = np.ones((h, w), bool)
    untouched[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad] = False
    print(f"      comparing {int(untouched.sum())} px/frame outside the mask")
    return report("untouched pixels vs input", frames[:, untouched], mine[:, untouched])


def main() -> int:
    global DEVICE
    if len(sys.argv) > 1:
        DEVICE = sys.argv[1]
        print(f"device forced to {DEVICE!r} -- note the reference script has no "
              f"device flag and always uses its own get_device(), so a parity "
              f"comparison is only meaningful if these agree")
    if not (VENDOR / "inference_propainter.py").is_file():
        print(f"error: vendored ProPainter not found at {VENDOR}")
        print("       run scripts/vendor-propainter.sh first")
        return 2

    ok = check_parity()
    ok = check_no_resampling() and ok
    print("\nall checks passed" if ok else "\nFAILURES -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
