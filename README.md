# wmrm — fixed-position watermark removal

Internal CLI. Removes a watermark that is always the same and always in the same
place, keeps the audio bit-exact, and checks its own output. Runs fully offline.

Findings, measurements and the quality trade-offs live in
[../REPORT.md](../REPORT.md). This file is only how to run it.

---

## Quickstart

Three commands. Replace `your.mp4` with a real clip.

```bash
source .venv/bin/activate        # every session

# 1. calibrate ONCE. Writes wm-preset.json. Processes no video.
wmrm detect your.mp4 --corner tr --preset wm-preset.json

# 2. open your-preview-zoom.png and check the red box covers the whole watermark

# 3. process
wmrm run your.mp4 --preset wm-preset.json     # -> your-clean.mp4
```

After step 1 you never need `detect` again — reuse `wm-preset.json` forever.

Whole folder:

```bash
wmrm batch ./inbox --preset wm-preset.json    # skips files already done
```

**If the box in step 2 is wrong**, measure it yourself and freeze that instead.
Detection still under-covers on dense texture, so check the preview every time:

```bash
# find the coordinates: pull a frame, crop the corner, overlay a 50px grid
ffmpeg -ss 30 -i your.mp4 -frames:v 1 \
    -vf "crop=520:130:1400:15,scale=iw*3:ih*3:flags=neighbor,drawgrid=w=150:h=390:c=red" \
    -y grid.png

wmrm run your.mp4 --box 1554,44,284,62 --preview-only   # check, processes nothing
wmrm detect your.mp4 --box 1554,44,284,62 --preset wm-preset.json   # freeze it
wmrm run your.mp4 --preset wm-preset.json
```

`detect --box` skips detection entirely and just writes the preset, so
hand-measured coordinates are reusable like any other.

**More than one watermark?** If they sit next to each other, use one box covering
both — that is the normal case for a studio logo beside a rating mark. Boxes far
apart are not supported; run the tool twice.

---

## Install

Prerequisites: `ffmpeg` + `ffprobe` on PATH, Python 3.10+, and
[uv](https://docs.astral.sh/uv/).

**Already in a venv that has torch?** Check first — if this prints a `+cu___`
build with `cuda: True`, skip step 1 and install into the venv you are already
in, or you will re-download ~2.5 GB of CUDA torch for nothing:

```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

```bash
cd wmrm
uv venv .venv --python 3.12
export VIRTUAL_ENV=$PWD/.venv
```

**Step 1 — torch.** The only line that differs per machine. Run `nvidia-smi`
first: a table means you have a GPU, `command not found` means you do not.

```bash
# no GPU
uv pip install --index-url https://download.pytorch.org/whl/cpu torch

# GPU — match CUDA to your driver (nvidia-smi, top right).
# cu124 needs driver >= 550; use cu121 for older.
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

**Step 2 — the rest.** Identical on both:

```bash
uv pip install --no-deps simple-lama-inpainting
uv pip install "pillow>=10" opencv-python-headless numpy
uv pip install --no-deps -e .
source .venv/bin/activate
```

Then confirm, **in the shell you will actually use**:

```bash
wmrm --version
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

On a GPU box that must say `cuda: True`. If it says `False` while `nvidia-smi`
works, you installed the CPU wheel — redo step 1 with the `cu124` index.

Optional end-to-end smoke test, using a generated clip with a known watermark:

```bash
python tests/make_fixtures.py
wmrm run tests/fixtures/detail-marked.mp4 -o /tmp/smoke.mp4 \
    --box 379,427,91,43 --patch-hold 8
```

Must end with `=> all checks passed`, and the log must say `loading LaMa on cuda`
on a GPU box (`cpu` otherwise). `--patch-hold 8` is only here to keep the check
quick — leave it off for real work.

Install notes, all deliberate:

- `simple-lama-inpainting` needs `--no-deps`: it pins `pillow==9.5`, which does
  not build on Python 3.12.
- Installing into an environment that already had torch? Drop `numpy` from step 2
  — pulling in numpy 2.x next to a torch built against 1.x breaks the ABI.
- LaMa weights (`big-lama.pt`, 196 MB) download on first use to
  `~/.cache/torch/hub/checkpoints/`. First run is slower.

---

## Commands

| | |
| --- | --- |
| `wmrm detect IN` | find the watermark, write a preset + preview PNGs. Processes nothing. |
| `wmrm run IN [-o OUT]` | process one video. Default output `IN-clean.EXT`. |
| `wmrm batch DIR` | process every video in a directory, skipping finished ones. |
| `wmrm verify ORIG OUT` | re-run the acceptance checks on an existing pair. |

`run` and `batch` need either `--preset` or `--box`.

### Options you may actually need

| flag | default | what it does |
| --- | --- | --- |
| `--quality` | `high` | `high` = LaMa (best), `fast` = ffmpeg delogo (near-realtime, smears on texture), `draft` = cv2.inpaint (coordinate checks only) |
| `--grad-threshold` | swept | `detect` only. Swept 10→1.5 automatically; pass a number to pin it. |
| `--device` | `auto` | `auto` takes CUDA when present. Use `cuda` to fail loudly instead of silently falling back to CPU. |
| `--dilate` | 5 | grow the mask. **Raise this first if any watermark fringe survives.** |
| `--feather` | 12 | blend width that hides the seam |
| `--margin` | 64 | context given to the inpainter. Cost scales with tile area — raising it is expensive. |
| `--patch-hold` | 1 | reuse each patch for N frames. Speed lever, not a quality fix — see limits below. |
| `--crf` | 18 | x264 quality |
| `--preview-only` | | draw the box on a frame and exit |
| `--no-verify` | | skip the acceptance checks |

`wmrm run --help` lists everything.

### Which `--quality`, and how long it takes

Measured on CPU (6 cores). The 1080p column is a real clip with a 284x62 mark:

| | 480x640 | 1080p | 1 minute of 1080p |
| --- | --- | --- | --- |
| `high` (LaMa) | 5.5–7 fps | 1.4 fps | **~22 min** |
| `fast` (delogo) | ~realtime | ~realtime | ~1 min |
| `draft` (cv2) | ~95 fps | fast | seconds |

**On GPU:** use `high` and stop thinking about it — 20-50x faster than the CPU
numbers above.

**On CPU: try `fast` first.** If the watermark sits on a plain wall, sky or
gradient it is indistinguishable from `high` and finishes in seconds rather than
tens of minutes. Measured on a real 1080p clip with the mark on a plain wall,
`fast` and `high` both removed it completely with no visible difference. Switch
to `high` only when the background under the mark has real detail — that is where
`fast` smears.

`--patch-hold N` cuts `high` runtime roughly N-fold. Read the caveat below first.

### Detect options

`--corner tr|tl|br|bl` (default `tr`), `--samples 40`, `--roi-frac 0.30`,
`--grad-threshold 10`, `--persistence 0.90`, `--max-area 10`.

Defaults are fine for a corner badge. Lower `--grad-threshold` for a fainter
watermark.

---

## Limits worth knowing before you rely on it

Full measurements in [../REPORT.md](../REPORT.md) §4.

**Temporal coherence is not solved.** Every frame is inpainted independently, so
the patch either boils (LaMa) or sits frozen while the background moves
(`fast`/`draft`). Measured: LaMa's frame-to-frame variation has the right
amplitude but only **0.01 correlation** with where the real content changed — the
variation is invented. `--patch-hold N` converts boiling into freezing rather
than fixing it; treat it as a speed lever. A real fix needs ProPainter or E2FGVI
on a GPU (REPORT.md §5, phase 2).

**Detection is good but not perfect — always look at the preview.** The gradient
threshold is swept automatically (10 down to 1.5) and the largest stable box under
the size limits wins, which is what lets it catch a faint mark next to a bold one.
Measured against known ground truth:

| case | true box | detected | verdict |
| --- | --- | --- | --- |
| real 1080p, two marks side by side | `1554,44,284,62` (measured by hand) | `1554,44,283,64` | correct |
| badge on soft sky | `384,12,84,36` | `364,0,108,68` | over-covers — safe |
| badge on sky, handheld | `384,12,84,36` | `377,4,99,51` | over-covers — safe |
| **badge on dense texture** | `384,430,84,36` | `377,448,96,27` | **under-covers vertically** |

Over-covering is harmless (a little slower, a few clean pixels repainted).
Under-covering leaves residue, and it still happens on dense texture, where only
the high-contrast part of the mark registers. `detect` warns when the box comes
out oddly elongated, but that check does not catch every case — **the preview is
the real safety net.** Calibrate on a clip with a calmer background, or measure
the box by hand.

**Do not pick a backend by PSNR.** It rewards blur — a smear scored *higher* than
a visibly better reconstruction. Judge with your eyes.

**Not handled:** watermarks that move or animate; watermarks over ~10% of frame;
semi-transparent watermarks are inpainted rather than un-blended (`detect`
reports `opacity: semi` when it sees the background bleeding through).

---

## For development

```bash
python tests/make_fixtures.py       # test clips with a badge at a known position
python tests/score.py detail        # PSNR vs the clean original + comparison PNG
python tests/flicker.py detail      # temporal coherence
python tests/check_gitignore.py     # assert the ignore rules still behave
```

Fixtures burn a fake badge onto a clean clip, so recovery is scored against
ground truth rather than guessed at. Cases: `smooth` and `busy` put the badge on
soft sky (easy), `detail` puts it on dense texture (hard — the case that
matters). Source clips come from `../dreamina-delogo/examples` if present, and
are otherwise synthesized with ffmpeg so the tests run anywhere.
