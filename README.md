# wmrm — fixed-position watermark removal

Internal CLI. Removes a watermark that is always the same and always in the same
place from a video, keeps the audio bit-exact, and checks its own output.

Offline only — nothing is uploaded anywhere.

## Install

Prerequisites: `ffmpeg` and `ffprobe` on PATH, Python 3.10+, and
[uv](https://docs.astral.sh/uv/). Check with `ffmpeg -version`.

```bash
cd wmrm
uv venv .venv --python 3.12

# torch: pick ONE line depending on the machine
VIRTUAL_ENV=$PWD/.venv uv pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU-only box
# VIRTUAL_ENV=$PWD/.venv uv pip install --index-url https://download.pytorch.org/whl/cu124 torch  # GPU box

VIRTUAL_ENV=$PWD/.venv uv pip install --no-deps simple-lama-inpainting
VIRTUAL_ENV=$PWD/.venv uv pip install "pillow>=10" opencv-python-headless numpy
VIRTUAL_ENV=$PWD/.venv uv pip install --no-deps -e .
```

Then **activate the venv**, or the `wmrm` command will not be on your PATH:

```bash
source .venv/bin/activate
wmrm --version
```

Prefer not to activate? Call it by path instead: `./.venv/bin/wmrm --version`.
Every `wmrm ...` example below assumes one or the other.

Two install quirks, both deliberate:

- `simple-lama-inpainting` goes in with `--no-deps` because it pins
  `pillow==9.5`, which does not build on Python 3.12.
- torch comes from an explicit index. The default PyPI wheel is large and may not
  match your CUDA version; the CPU wheel is the right choice on a machine without
  a GPU, and `--device cuda` will tell you clearly if you installed the CPU wheel
  on a GPU box.

LaMa weights (`big-lama.pt`, 196 MB) download themselves on first use into
`~/.cache/torch/hub/checkpoints/`. First run is therefore slower; after that the
model loads in ~11 s on CPU.

### Verify the install

```bash
python tests/make_fixtures.py                    # builds test clips with a known badge
wmrm run tests/fixtures/busy-marked.mp4 -o /tmp/smoke.mp4 --box 377,8,95,47 --quality draft
```

That should end with `=> all checks passed`.

## Use

Calibrate once per watermark, then reuse the preset forever.

```bash
# 1. find the watermark and write a preset. Processes nothing.
wmrm detect sample.mp4 --corner tr --preset wm-preset.json

# 2. LOOK AT sample-preview-zoom.png. Confirm the red box covers the whole badge.

# 3. process
wmrm run video.mp4 --preset wm-preset.json          # -> video-clean.mp4
wmrm batch ./inbox --preset wm-preset.json          # whole folder
```

On a GPU machine add nothing — `--device auto` already picks CUDA. Force it with
`--device cuda` if you want the run to fail loudly rather than fall back to CPU.

**Calibrate on a clip with a calm background.** Detection is reliable when the
corner under the badge is smooth, and **under-covers on busy texture** — it finds
only the high-contrast part of the badge, and an under-sized box leaves watermark
residue. Measured against a known badge at `384,y,84,36`:

| background | detected | verdict |
| --- | --- | --- |
| soft sky | `377,8,95,47` | good, covers with margin |
| busy texture | `385,448,88,17` | **height 17 vs 36 — under-covers** |

`wmrm detect` warns when the box comes out suspiciously elongated, which is what
a partly-detected badge looks like. This matters less than it sounds: the
watermark is fixed, so you calibrate **once** on a favourable clip and reuse the
preset forever.

If detection picks the wrong thing, skip it entirely and give the box yourself:

```bash
wmrm run video.mp4 --box 379,10,91,43 --preview-only   # check the box first
wmrm run video.mp4 --box 379,10,91,43
```

### Quality tiers

| `--quality` | Method | Speed (480x640) | Use when |
| --- | --- | --- | --- |
| `high` *(default)* | LaMa on a crop tile | 5.5–7 fps | anything with texture under the badge |
| `fast` | ffmpeg `delogo` + radial feather | ~realtime | background is smooth sky/gradient |
| `draft` | `cv2.inpaint` | ~95 fps | checking coordinates, not for delivery |

### Knobs worth knowing

- `--dilate` (5) — grow the mask to swallow the watermark's soft edge. If any
  fringe survives, raise this before anything else.
- `--feather` (12) — width of the blend that hides the seam.
- `--margin` (64) — context given to the inpainter. **Cost scales with tile
  area**, so do not raise it casually.
- `--patch-hold` (1) — reuse each patch for N frames. See the honest caveat below.
- `--crf` (18), `--x264-preset` (medium).

## What it does under the hood

```
decode (ffmpeg rawvideo pipe)
  -> crop a tile around the box (+margin)
  -> inpaint the tile only
  -> alpha-composite the patch back over the original frame
  -> encode (ffmpeg, libx264 CRF 18, -c:a copy)
```

Three decisions that matter:

**Inpaint a tile, never the frame.** Measured: LaMa on a 320x192 tile is 452
ms/frame; on a full 1920x1080 frame it is 16.3 s/frame for identical output —
**77x slower**. Cost tracks tile pixels.

**Binary mask into the model, blurred mask for compositing.** Blurring the mask
and handing *that* to an inpainter does not blend anything — it just enlarges the
hole and destroys a ring of good pixels. The blur belongs in the composite step.
Several published tools get this backwards.

**Stay in one ffmpeg pipeline.** No PNG frame dumps, no temp frame directories.
Output is written to a temp file and `os.replace`d into position, so an
interrupted run cannot leave a truncated file that a later batch run skips.

Audio is `-c:a copy` with `-map 0:a:0?` — bit-exact, and the `?` means silent
inputs do not abort.

## Verification

`wmrm run` checks its own output unless you pass `--no-verify`:

- resolution, frame rate, duration and audio presence match the input
- PSNR **outside** the mask stays high (the rest of the picture was not damaged)
- PSNR **inside** the mask is low (something actually changed)

These are sanity gates, not a quality score. **Do not use PSNR to choose a
backend** — it rewards blur. Measured on a detailed background, `fast` scored
20.6 dB against LaMa's 18.4 dB while looking clearly worse: a smear that stays
near the local mean beats a sharp reconstruction that differs in exact pixel
placement. Judge with your eyes; use `tests/score.py` for the numbers and
`tests/fixtures/*-comparison.png` for the picture.

## Known limitation: temporal coherence

**No backend here is temporally coherent, and this is the tool's real ceiling.**

Every frame is inpainted independently, so the patch does not know what it looked
like a frame ago. Measured on handheld footage over detailed ground, inside the
badge region:

| variant | excess motion | correlation with real motion |
| --- | --- | --- |
| LaMa, `--patch-hold 1` | −0.003 | **0.01** |
| LaMa, `--patch-hold 4` | −4.33 | 0.01 |
| `fast` (delogo+feather) | −2.54 | 0.12 |
| `draft` (cv2.inpaint) | −2.11 | 0.00 |

LaMa's frame-to-frame variation has almost exactly the right *amplitude*
(excess ≈ 0) but a correlation of 0.01 with where the real content actually
changed — the variation is invented. That is what a boiling patch is. Looking
only at amplitude would have said "stable" and been wrong.

`--patch-hold N` does **not** fix this. It converts boiling into freezing: at
N=4 the patch becomes far too static (excess −4.33) for a 3.1x speedup, with
correlation unchanged. Which artifact is less objectionable depends on the
footage — freezing is worse the more the camera moves. Treat the flag as a speed
lever, not a quality fix.

A real fix needs a video inpainter that propagates across frames (ProPainter,
E2FGVI, STTN). Those need a GPU to be usable; there is none in this environment.

## Fixtures and measurement

```bash
python tests/make_fixtures.py      # burn a fake badge onto clean clips
python tests/score.py detail       # PSNR vs the clean original + comparison PNG
python tests/flicker.py detail     # temporal coherence
```

Fixtures burn a badge onto a genuinely clean clip, so recovery can be scored
against ground truth instead of guessed at. Three cases: `busy` and `smooth`
put the badge on soft sky (the easy case — the corner most clips happen to
have), `detail` puts it on busy texture (the hard case, and the one that matters).

## Not handled

- watermarks that move or animate
- watermarks larger than ~10% of the frame
- semi-transparent watermarks are inpainted, not un-blended. `wmrm detect`
  reports `opacity: semi` when it sees the background bleeding through; alpha
  un-blend would recover those almost exactly and is not implemented yet.
