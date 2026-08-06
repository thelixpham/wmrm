"""Resident, in-process ProPainter. Ours, not upstream's -- see ../README.md.

Upstream ships its inference as a script whose entire body is inside
`if __name__ == '__main__':`, so the only way to reach it is a subprocess. That cost
is not the subprocess, it is what a fresh process implies: RAFT, the recurrent flow
completion net and the inpaint generator are loaded again, ~190 MB of weights and a
CUDA context, before any frame is touched. wmrm calls it once per segment. On a
ten-hour video at `--pp-segment 400` that is ~2700 reloads -- hours of wall clock
spent loading the same three files, on the one engine that is already the slow one.

This module holds the models and exposes `ProPainterWorker.inpaint(frames, mask)`
over numpy arrays. Three things follow from that, in order of how much they matter:

- **The models load once per run instead of once per segment.**
- **The PNG round-trip disappears.** wmrm used to encode every tile frame to PNG,
  upstream decoded them back, wrote result PNGs, and ffmpeg read those. A tile is
  ~200 KB in memory; none of that was ever needed. It is also what made peak temp
  disk scale with video length.
- **Padding replaces resizing, which fixes a silent quality bug.** Upstream crops
  the processing size down to a multiple of 8 (`resize_frames`) and cubic-resizes
  the result back to the original size on the way out (`--save_frames`). For a tile
  whose size is not a multiple of 8 -- which was every tile before wmrm's region
  code aligned them -- the repaired region came back resampled: softened over
  exactly the pixels being repaired, and off by a fraction of a pixel against the
  frame it gets composited onto. The output kept the requested dimensions, so
  wmrm's shape assertion could not see it. Reflect-padding up to the multiple of 8
  and slicing the pad back off resamples nothing.

Everything else here is upstream's inference body, deliberately kept line-for-line
comparable to `inference_propainter.py` so the two can be diffed after an upgrade.
`tests/test_propainter_parity.py` asserts the outputs match. What is *left out* of
that body, all of it unused by wmrm:

- `masked_in.mp4` -- a green-overlay debug render of every input frame, built in
  float64. Encoded per segment and thrown away.
- `inpaint_out.mp4` -- the mp4 wmrm explicitly must not read, because upstream's
  writer pads to a multiple of 16.
- the outpainting mode, `--resize_ratio`, `--width/--height`.

One upstream call was moved rather than dropped: `torch.cuda.empty_cache()` inside
the per-window loop ran ~80 times per 400-frame segment. It synchronises and hands
cached blocks back to the driver for the allocator to re-request. It is kept at the
stage boundaries, where it is what lets a long segment fit, and gone from the hot
loop, where it only costs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.ndimage
import torch

# Upstream's modules import each other by top-level name (`from model...`,
# `from RAFT import RAFT`), so its directory has to be importable. Doing it here
# means callers only need this file's path, not knowledge of the layout.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from model.misc import get_device                              # noqa: E402
from model.modules.flow_comp_raft import RAFT_bi               # noqa: E402
from model.propainter import InpaintGenerator                  # noqa: E402
from model.recurrent_flow_completion import RecurrentFlowCompleteNet  # noqa: E402
from utils.download_util import load_file_from_url             # noqa: E402

PRETRAIN_URL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"

# The model stack downsamples by 8, which is why upstream crops to a multiple of it.
ALIGN = 8

# Upstream's own weights location, but absolute. It calls load_file_from_url with
# model_dir='weights', which resolves against the *current working directory* -- so
# the old subprocess call had to pass cwd=<repo> or silently download 190 MB into
# wherever it was launched from. Pinning it to this file's directory removes that.
WEIGHTS_DIR = _HERE / "weights"


@dataclass(frozen=True)
class WorkerOpts:
    """Upstream's inference knobs, same names and same defaults."""

    subvideo_length: int = 80
    neighbor_length: int = 10
    ref_stride: int = 10
    raft_iter: int = 20
    mask_dilation: int = 4
    fp16: bool = True


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        i = device.index if device.index is not None else torch.cuda.current_device()
        vram = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        return f"cuda ({torch.cuda.get_device_name(i)}, {vram:.1f} GB)"
    if device.type == "mps":
        return "mps"
    return "cpu -- expect minutes per hundred frames"


class ProPainterWorker:
    """Load the models once, then repair any number of frame batches."""

    def __init__(self, device: str | torch.device | None = None,
                 opts: WorkerOpts | None = None) -> None:
        self.opts = opts or WorkerOpts()
        # Default to upstream's own selection rule so behaviour matches when no
        # device is named, but allow naming one -- upstream has no flag for it, and
        # a silent fall back to CPU on this model is a 20-50x surprise.
        self.device = torch.device(device) if device is not None else get_device()

        # fp16 on CPU is not supported by these ops; upstream disables it the same way.
        self.use_half = bool(self.opts.fp16) and self.device.type != "cpu"

        raft_ckpt = load_file_from_url(url=PRETRAIN_URL + "raft-things.pth",
                                       model_dir=str(WEIGHTS_DIR), progress=True)
        self.fix_raft = RAFT_bi(raft_ckpt, self.device)

        flow_ckpt = load_file_from_url(url=PRETRAIN_URL + "recurrent_flow_completion.pth",
                                       model_dir=str(WEIGHTS_DIR), progress=True)
        self.fix_flow_complete = RecurrentFlowCompleteNet(flow_ckpt)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(self.device).eval()

        pp_ckpt = load_file_from_url(url=PRETRAIN_URL + "ProPainter.pth",
                                     model_dir=str(WEIGHTS_DIR), progress=True)
        self.model = InpaintGenerator(model_path=pp_ckpt).to(self.device).eval()

        # Upstream halves these two after computing flows, and keeps RAFT in fp32
        # ("use fp32 for RAFT"). Doing it once here is the same thing: neither model
        # runs before that point.
        if self.use_half:
            self.fix_flow_complete = self.fix_flow_complete.half()
            self.model = self.model.half()

        self._mask_cache: tuple[bytes, tuple[int, int], np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ masks

    def _dilated_masks(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Binary dilation of the mask, cached across calls.

        wmrm hands the same fixed-position mask to every segment of a video, so the
        two scipy dilations are the same work repeated once per segment. Upstream
        re-read and re-dilated the PNG every invocation because it had no way not to.
        """
        key = (mask.tobytes(), mask.shape)
        if self._mask_cache is not None and self._mask_cache[:2] == key:
            return self._mask_cache[2], self._mask_cache[3]

        it = self.opts.mask_dilation
        # Upstream passes --mask_dilation as *both* flow_mask_dilates and
        # mask_dilates, so the two differ only if that changes upstream.
        if it > 0:
            flow_mask = scipy.ndimage.binary_dilation(mask, iterations=it).astype(np.uint8)
            dilated = scipy.ndimage.binary_dilation(mask, iterations=it).astype(np.uint8)
        else:
            flow_mask = (mask > 0).astype(np.uint8)
            dilated = (mask > 0).astype(np.uint8)

        self._mask_cache = (key[0], key[1], flow_mask, dilated)
        return flow_mask, dilated

    # ------------------------------------------------------------------ public

    def inpaint(self, frames_bgr: np.ndarray, mask: np.ndarray, *,
                progress=None) -> np.ndarray:
        """Repair `mask` across `frames_bgr`, using the other frames as the source.

        frames_bgr : (T, H, W, 3) uint8, BGR -- wmrm's convention throughout, and
                     what ffmpeg's bgr24 hands over.
        mask       : (H, W) uint8, nonzero where pixels must be replaced.
        progress   : optional callable(done, total) for the per-window loop.

        Returns (T, H, W, 3) uint8 BGR, exactly the input shape. No resampling
        happens anywhere in here.
        """
        if frames_bgr.ndim != 4 or frames_bgr.shape[3] != 3:
            raise ValueError(f"expected (T, H, W, 3) frames, got {frames_bgr.shape}")
        if frames_bgr.dtype != np.uint8:
            raise ValueError(f"expected uint8 frames, got {frames_bgr.dtype}")
        t_len, h0, w0 = frames_bgr.shape[:3]
        if t_len < 1:
            raise ValueError("no frames given")
        if mask.shape[:2] != (h0, w0):
            raise ValueError(f"mask is {mask.shape[:2]}, frames are {(h0, w0)}")

        pad_h, pad_w = -h0 % ALIGN, -w0 % ALIGN
        frames_rgb = self._pad_frames(frames_bgr[..., ::-1], pad_h, pad_w)
        mask_p = np.pad(mask, ((0, pad_h), (0, pad_w)))   # zeros: nothing to fill there
        h, w = frames_rgb.shape[1:3]

        flow_mask_2d, dilated_2d = self._dilated_masks(mask_p)

        with torch.no_grad():
            frames = self._to_tensor(frames_rgb) * 2 - 1
            flow_masks = self._mask_tensor(flow_mask_2d, t_len)
            masks_dilated = self._mask_tensor(dilated_2d, t_len)

            gt_flows_bi = self._compute_flows(frames, t_len)

            if self.use_half:
                frames = frames.half()
                flow_masks = flow_masks.half()
                masks_dilated = masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())

            pred_flows_bi = self._complete_flows(gt_flows_bi, flow_masks)
            del gt_flows_bi
            self._empty_cache()

            updated_frames, updated_masks = self._propagate(
                frames, pred_flows_bi, masks_dilated, t_len, h, w)
            self._empty_cache()

            out_rgb = self._transform(frames_rgb, frames, updated_frames, updated_masks,
                                      masks_dilated, pred_flows_bi, t_len, h, w,
                                      progress)

        self._empty_cache()
        # Slice the pad off and return BGR. Both are views/copies, not resamples.
        return np.ascontiguousarray(out_rgb[:, :h0, :w0, ::-1])

    # ------------------------------------------------------------------ stages

    @staticmethod
    def _pad_frames(rgb: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
        if not pad_h and not pad_w:
            return np.ascontiguousarray(rgb)
        # Reflect gives the model a plausible continuation of the border instead of
        # a black edge, which would read as content that has to be inpainted around.
        # 'reflect' needs at least 2 px in the axis it mirrors; 'edge' is the
        # degenerate fallback and only reachable on a 1 px tile.
        mode = "reflect" if min(rgb.shape[1], rgb.shape[2]) > 1 else "edge"
        return np.ascontiguousarray(
            np.pad(rgb, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode=mode))

    def _to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        """(T,H,W,3) uint8 -> (1,T,3,H,W) float in [0,1], on the device.

        Same arithmetic as upstream's `to_tensors()`: uint8 -> float -> /255. That
        chain goes through PIL images and a transforms.Compose there; done directly
        it is bit-identical and skips T PIL objects per segment.
        """
        t = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous().float().div(255)
        return t.unsqueeze(0).to(self.device)

    def _mask_tensor(self, mask_2d: np.ndarray, t_len: int) -> torch.Tensor:
        """(H,W) uint8 {0,1} -> (1,T,1,H,W) float, one identical plane per frame.

        The repeat is upstream's behaviour and is kept for it: at 400 frames of
        400x168 it is ~107 MB per mask tensor, of which all but one plane is
        duplicate. `expand` would make it a view, but the tensor is handed into the
        model where a `view`/`reshape` on a non-contiguous input would throw, so
        that is a change to make with the parity test in hand, not in passing.
        """
        t = torch.from_numpy(mask_2d.astype(np.float32))[None, None, None]
        return t.repeat(1, t_len, 1, 1, 1).to(self.device)

    def _compute_flows(self, frames: torch.Tensor, t_len: int) -> tuple:
        """Bidirectional RAFT flow, chunked by width the way upstream chunks it."""
        width = frames.size(-1)
        if width <= 640:
            short_clip_len = 12
        elif width <= 720:
            short_clip_len = 8
        elif width <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        if t_len <= short_clip_len:
            flows = self.fix_raft(frames, iters=self.opts.raft_iter)
            self._empty_cache()
            return flows

        fwd, bwd = [], []
        for f in range(0, t_len, short_clip_len):
            end_f = min(t_len, f + short_clip_len)
            # From the second chunk on, one frame of overlap: flow is defined
            # between pairs, so starting at f would lose the f-1 -> f pair.
            lo = f if f == 0 else f - 1
            flows_f, flows_b = self.fix_raft(frames[:, lo:end_f], iters=self.opts.raft_iter)
            fwd.append(flows_f)
            bwd.append(flows_b)
            self._empty_cache()
        return torch.cat(fwd, dim=1), torch.cat(bwd, dim=1)

    def _complete_flows(self, gt_flows_bi: tuple, flow_masks: torch.Tensor) -> tuple:
        flow_length = gt_flows_bi[0].size(1)
        sub = self.opts.subvideo_length
        if flow_length <= sub:
            pred, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred = self.fix_flow_complete.combine_flow(gt_flows_bi, pred, flow_masks)
            self._empty_cache()
            return pred

        pred_f, pred_b = [], []
        pad_len = 5
        for f in range(0, flow_length, sub):
            s_f = max(0, f - pad_len)
            e_f = min(flow_length, f + sub + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(flow_length, f + sub)
            chunk = (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f])
            sub_pred, _ = self.fix_flow_complete.forward_bidirect_flow(
                chunk, flow_masks[:, s_f:e_f + 1])
            sub_pred = self.fix_flow_complete.combine_flow(
                chunk, sub_pred, flow_masks[:, s_f:e_f + 1])
            pred_f.append(sub_pred[0][:, pad_len_s:e_f - s_f - pad_len_e])
            pred_b.append(sub_pred[1][:, pad_len_s:e_f - s_f - pad_len_e])
            self._empty_cache()
        return torch.cat(pred_f, dim=1), torch.cat(pred_b, dim=1)

    def _propagate(self, frames: torch.Tensor, pred_flows_bi: tuple,
                   masks_dilated: torch.Tensor, t_len: int, h: int, w: int) -> tuple:
        masked_frames = frames * (1 - masks_dilated)
        # Upstream's comment: ensure a minimum of 100 frames for image propagation.
        sub = min(100, self.opts.subvideo_length)
        if t_len <= sub:
            b, t, _, _, _ = masks_dilated.size()
            prop, upd_masks = self.model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated, "nearest")
            updated = frames * (1 - masks_dilated) + prop.view(b, t, 3, h, w) * masks_dilated
            self._empty_cache()
            return updated, upd_masks.view(b, t, 1, h, w)

        frames_out, masks_out = [], []
        pad_len = 10
        for f in range(0, t_len, sub):
            s_f = max(0, f - pad_len)
            e_f = min(t_len, f + sub + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(t_len, f + sub)
            b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
            sub_flows = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
            prop, upd_local = self.model.img_propagation(
                masked_frames[:, s_f:e_f], sub_flows, masks_dilated[:, s_f:e_f], "nearest")
            upd = (frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f])
                   + prop.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f])
            frames_out.append(upd[:, pad_len_s:e_f - s_f - pad_len_e])
            masks_out.append(upd_local.view(b, t, 1, h, w)[:, pad_len_s:e_f - s_f - pad_len_e])
            self._empty_cache()
        return torch.cat(frames_out, dim=1), torch.cat(masks_out, dim=1)

    def _transform(self, ori_rgb: np.ndarray, frames: torch.Tensor,
                   updated_frames: torch.Tensor, updated_masks: torch.Tensor,
                   masks_dilated: torch.Tensor, pred_flows_bi: tuple,
                   t_len: int, h: int, w: int, progress) -> np.ndarray:
        """Feature propagation + transformer, window by window."""
        comp: list[np.ndarray | None] = [None] * t_len
        neighbor_stride = self.opts.neighbor_length // 2
        ref_num = self.opts.subvideo_length // self.opts.ref_stride if t_len > self.opts.subvideo_length else -1

        starts = list(range(0, t_len, neighbor_stride))
        for done, f in enumerate(starts, 1):
            neighbor_ids = list(range(max(0, f - neighbor_stride),
                                      min(t_len, f + neighbor_stride + 1)))
            ref_ids = self._ref_index(f, neighbor_ids, t_len, ref_num)
            picked = neighbor_ids + ref_ids

            pred_img = self.model(
                updated_frames[:, picked],
                (pred_flows_bi[0][:, neighbor_ids[:-1]], pred_flows_bi[1][:, neighbor_ids[:-1]]),
                masks_dilated[:, picked],
                updated_masks[:, picked],
                len(neighbor_ids),
            )
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = ((pred_img + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255
            binary = masks_dilated[0, neighbor_ids].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

            for i, idx in enumerate(neighbor_ids):
                img = (pred_img[i].astype(np.uint8) * binary[i]
                       + ori_rgb[idx] * (1 - binary[i]))
                if comp[idx] is None:
                    comp[idx] = img
                else:
                    # Windows overlap by design; a frame seen twice is averaged.
                    comp[idx] = (comp[idx].astype(np.float32) * 0.5
                                 + img.astype(np.float32) * 0.5).astype(np.uint8)
            if progress is not None:
                progress(done, len(starts))
        # No empty_cache() in this loop -- see the module docstring.

        if any(c is None for c in comp):
            missing = [i for i, c in enumerate(comp) if c is None]
            raise RuntimeError(f"frames never produced: {missing[:8]}")
        return np.stack(comp)  # type: ignore[arg-type]

    def _ref_index(self, mid: int, neighbor_ids: list[int], length: int,
                   ref_num: int) -> list[int]:
        stride = self.opts.ref_stride
        out: list[int] = []
        if ref_num == -1:
            for i in range(0, length, stride):
                if i not in neighbor_ids:
                    out.append(i)
        else:
            start = max(0, mid - stride * (ref_num // 2))
            end = min(length, mid + stride * (ref_num // 2))
            for i in range(start, end, stride):
                if i not in neighbor_ids:
                    if len(out) > ref_num:
                        break
                    out.append(i)
        return out

    def _empty_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
