"""Remove a fixed-position watermark from video.

Design notes and the reasoning behind the algorithm choices live in
`../KNOWLEDGE.md`. The short version:

- The watermark is fixed in place, so localization is a one-time calibration
  that gets frozen into a preset -- it is not part of the per-video run.
- Quality comes from inpainting a *small tile* around the watermark with LaMa.
  Running LaMa on the full frame is 77x slower for identical output.
- The patch is alpha-composited back with a blurred mask so there is no
  rectangular seam. The mask fed to the inpainter is always binary.
"""

__version__ = "0.1.0"
