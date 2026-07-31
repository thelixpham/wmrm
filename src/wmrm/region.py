"""Watermark region: the box, the tile around it, and the masks.

Two masks are derived from one box, and the distinction matters:

- ``inpaint_mask`` is **binary**. It is what the inpainter sees. Feeding a
  blurred mask to cv2.inpaint (or LaMa) does not blend anything -- it merely
  dilates the hole and destroys a ring of good pixels. That bug is present in
  several of the reference repos; see KNOWLEDGE.md 4.1.
- ``alpha`` is the blurred mask, used *afterwards* to composite the inpainted
  result over the original. This is what actually hides the seam.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

CORNERS = ("tl", "tr", "bl", "br")

# Defaults tuned for a small corner badge. dilate must be generous enough to
# swallow the watermark's anti-aliased fringe, or a halo ring survives.
DEFAULT_DILATE_PX = 5
DEFAULT_FEATHER_PX = 12
DEFAULT_MARGIN_PX = 64


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, width: int, height: int) -> Box:
        x = max(0, min(self.x, width - 1))
        y = max(0, min(self.y, height - 1))
        return Box(x, y, max(1, min(self.w, width - x)), max(1, min(self.h, height - y)))

    def area(self) -> int:
        return self.w * self.h

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @classmethod
    def parse(cls, spec: str) -> Box:
        parts = [p.strip() for p in spec.replace(" ", ",").split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError(f"--box wants 'x,y,w,h', got {spec!r}")
        try:
            x, y, w, h = (int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"--box values must be integers, got {spec!r}") from exc
        if w <= 0 or h <= 0:
            raise ValueError(f"--box width/height must be positive, got {spec!r}")
        if x < 0 or y < 0:
            raise ValueError(f"--box x/y must be non-negative, got {spec!r}")
        return cls(x, y, w, h)


@dataclass(frozen=True)
class Preset:
    """Calibration frozen for reuse. Coordinates are stored normalized so the
    same preset survives a resolution change (KNOWLEDGE.md 4.5)."""

    box_norm: tuple[float, float, float, float]
    reference_size: tuple[int, int]
    opacity: str = "opaque"  # "opaque" | "semi"
    dilate_px: int = DEFAULT_DILATE_PX
    feather_px: int = DEFAULT_FEATHER_PX
    margin_px: int = DEFAULT_MARGIN_PX
    version: int = 1

    @classmethod
    def from_box(cls, box: Box, width: int, height: int, **kw) -> Preset:
        return cls(
            box_norm=(box.x / width, box.y / height, box.w / width, box.h / height),
            reference_size=(width, height),
            **kw,
        )

    def box_for(self, width: int, height: int) -> Box:
        nx, ny, nw, nh = self.box_norm
        return Box(
            round(nx * width), round(ny * height),
            max(1, round(nw * width)), max(1, round(nh * height)),
        ).clamp(width, height)

    def scaled_px(self, width: int, height: int) -> Preset:
        """Scale pixel-denominated knobs when the target resolution differs."""
        rw, rh = self.reference_size
        factor = ((width / rw) + (height / rh)) / 2 if rw and rh else 1.0
        if abs(factor - 1.0) < 0.02:
            return self
        return replace(
            self,
            dilate_px=max(1, round(self.dilate_px * factor)),
            feather_px=max(1, round(self.feather_px * factor)),
            margin_px=max(8, round(self.margin_px * factor)),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "box_norm": list(self.box_norm),
            "reference_size": list(self.reference_size),
            "opacity": self.opacity,
            "dilate_px": self.dilate_px,
            "feather_px": self.feather_px,
            "margin_px": self.margin_px,
        }

    @classmethod
    def load(cls, path: Path) -> Preset:
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            raise ValueError(f"{path}: unsupported preset version {data.get('version')!r}")
        for key in ("box_norm", "reference_size"):
            if key not in data:
                raise ValueError(f"{path}: preset is missing {key!r}")
        box_norm = tuple(float(v) for v in data["box_norm"])
        if len(box_norm) != 4:
            raise ValueError(f"{path}: box_norm must have 4 values")
        ref = tuple(int(v) for v in data["reference_size"])
        if len(ref) != 2:
            raise ValueError(f"{path}: reference_size must have 2 values")
        return cls(
            box_norm=box_norm,  # type: ignore[arg-type]
            reference_size=ref,  # type: ignore[arg-type]
            opacity=str(data.get("opacity", "opaque")),
            dilate_px=int(data.get("dilate_px", DEFAULT_DILATE_PX)),
            feather_px=int(data.get("feather_px", DEFAULT_FEATHER_PX)),
            margin_px=int(data.get("margin_px", DEFAULT_MARGIN_PX)),
        )


@dataclass(frozen=True)
class Region:
    """Everything the per-frame loop needs, precomputed once."""

    box: Box
    tile: Box              # crop window: box + margin, clamped to the frame
    inpaint_mask: np.ndarray  # uint8 {0,255}, tile-sized -- binary, for the model
    alpha: np.ndarray         # float32 [0,1], tile-sized, HxWx1 -- for compositing

    @property
    def tile_slice(self) -> tuple[slice, slice]:
        return (slice(self.tile.y, self.tile.y + self.tile.h),
                slice(self.tile.x, self.tile.x + self.tile.w))


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def build_region(
    box: Box,
    width: int,
    height: int,
    *,
    dilate_px: int = DEFAULT_DILATE_PX,
    feather_px: int = DEFAULT_FEATHER_PX,
    margin_px: int = DEFAULT_MARGIN_PX,
) -> Region:
    box = box.clamp(width, height)

    # Tile = box + margin. Keeping this small is the single biggest speed lever:
    # inpaint cost scales with tile pixels (KNOWLEDGE.md 7.1).
    #
    # Offsets and extents are snapped to even numbers: ffmpeg's crop filter on
    # yuv420p silently rounds odd values down to match chroma subsampling, and a
    # tile that disagrees with the mask by one pixel fails to composite.
    x0 = max(0, box.x - margin_px) & ~1
    y0 = max(0, box.y - margin_px) & ~1
    x1 = min(width, (min(width, box.x + box.w + margin_px) + 1) & ~1)
    y1 = min(height, (min(height, box.y + box.h + margin_px) + 1) & ~1)
    tile = Box(x0, y0, x1 - x0, y1 - y0)

    # `core` must fully cover the watermark including its soft edge.
    core = np.zeros((tile.h, tile.w), np.uint8)
    core[box.y - tile.y: box.y - tile.y + box.h,
         box.x - tile.x: box.x - tile.x + box.w] = 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(dilate_px * 2), _odd(dilate_px * 2)))
        core = cv2.dilate(core, k)

    # The binary hole handed to the inpainter extends `feather_px` past `core`,
    # so that after blurring, alpha is still ~1 across all of `core` and only
    # ramps down outside it. Blur `core` directly and alpha would fall to 0.5
    # right where the watermark still is, leaving a visible ghost.
    inpaint_mask = core
    if feather_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd(feather_px * 2), _odd(feather_px * 2))
        )
        inpaint_mask = cv2.dilate(core, k)

    if feather_px > 0:
        blur = _odd(max(3, feather_px * 2 + 1))
        alpha = cv2.GaussianBlur(inpaint_mask, (blur, blur), 0)
    else:
        alpha = inpaint_mask
    alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]

    return Region(box=box, tile=tile, inpaint_mask=inpaint_mask, alpha=alpha_f)


def corner_search_roi(width: int, height: int, corner: str, frac: float = 0.30) -> Box:
    """Restrict auto-detection to one corner.

    This is calibration guard layer 1: burned-in subtitles and description text
    live at the bottom/centre, so a corner ROI excludes them geometrically and
    they can never be mistaken for the logo (KNOWLEDGE.md 2.3.1).
    """
    if corner not in CORNERS:
        raise ValueError(f"corner must be one of {CORNERS}, got {corner!r}")
    rw, rh = max(16, int(width * frac)), max(16, int(height * frac))
    x = 0 if corner in ("tl", "bl") else width - rw
    y = 0 if corner in ("tl", "tr") else height - rh
    return Box(x, y, rw, rh)
