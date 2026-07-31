# wmrm — fixed-position watermark removal

Internal CLI. Removes a watermark that is always the same and always in the same
place, keeps the audio bit-exact, and checks its own output. Runs fully offline.

Findings, measurements and the quality trade-offs live in
[../REPORT.md](../REPORT.md). This file is only how to run it.

---

## If you only read one section

**Every day — one command.** Drop videos in a folder, point at the preset for
that watermark:

```bash
wmrm batch ./inbox --preset fanza.json
```

That already uses the best backend (LaMa). On a GPU box, never pass `--quality` —
there is nothing to trade away. `--quality fast` exists only for CPU-bound
machines, where LaMa costs about 22 minutes per minute of 1080p video.

**A watermark you have never handled before — three commands, once.** After this
you are back to the one-liner above, forever.

```bash
wmrm grid new.mp4 --corner tr                          # read X,Y,W,H off the grid
wmrm run new.mp4 --box X,Y,W,H --preview-only          # confirm the box by eye
wmrm detect new.mp4 --box X,Y,W,H --preset newmark.json    # save it
```

**Everything else in this file is for when something looks wrong.** Skip it until
it does.

Two rules that cover most mistakes:

- One preset per watermark **design**, named after it. A preset measured on one
  logo is meaningless for a different logo.
- Round the box **outward**. Too large costs a little speed; too small leaves a
  sliver of watermark in the output.
- Confirm the log says `loading LaMa on cuda` on a GPU box. If it says `cpu`, you
  have the CPU-only torch wheel and are running 20-50x slower than you should be.

---

## Calibrating a new watermark — the long version

Steps 1-4 happen **once per watermark design**, not once per video. Step 5 is the
only one you run day to day, and `wmrm batch` in the section above is step 5.

Read this the first time you meet a new logo, then never again. Keep one preset
per mark — `fanza.json`, `ippa.json`.

```bash
source .venv/bin/activate                                  # every session

wmrm grid your.mp4 --corner tr                             # 1. read coordinates
wmrm coverage your.mp4 --box X,Y,W,H                       # 2. check objectively
wmrm run your.mp4 --box X,Y,W,H --preview-only             # 3. check by eye
wmrm detect your.mp4 --box X,Y,W,H --preset fanza.json     # 4. freeze it
wmrm run your.mp4 --preset fanza.json                      # 5. process
```

### 1. Read the coordinates — `wmrm grid`

Writes `your-grid.png`: the corner, zoomed 3x, with red gridlines every 25 source
pixels. It also prints the crop origin, which you **add** to what you read off the
grid. Read the mark's left, top, right and bottom edges, then compute
`X,Y,W,H = left, top, right-left, bottom-top`.

Why by hand rather than `wmrm detect`: detection looks for pixel-locked *edges*.
That works on a semi-transparent mark and fails completely on opaque white text
over a white wall — measured, it found nothing at all on such a clip, and only one
glyph of a five-letter logo on another. Measuring takes two minutes and never
breaks. `wmrm detect` is still there as a first guess if you want one; treat its
answer as a proposal to verify, never as the answer.

### 2. Check it objectively — `wmrm coverage`

Answers "is any of the mark still outside this box?" It looks in a ring around the
box for two signals at once: pixel-locked edges (finds semi-transparent marks) and
collapsed temporal variance (finds opaque ones, because an overlay freezes whatever
is under it). That combination is what makes it work where detection does not.

Three possible answers:

- `covered` — nothing mark-like outside. Good.
- `UNDER-COVERED — mark extends left +48px` plus a suggested box. Re-run with the
  suggestion; it is capped by the ring size, so two or three rounds may be needed.
- `INCONCLUSIVE` — almost the whole ring looks mark-like, which means the
  background is itself static (fixed camera, plain wall) and no statistic can tell
  mark from wall. Fall back to step 3 and your eyes.

Not a proof: measured, from a box 160 px too small, iterating converged to 16 px
too small. It shrinks the error a lot; it does not eliminate it. Erring a few
pixels large is cheap, so round outward.

### 3. Check by eye — `wmrm run --preview-only`

Writes `your-boxcheck-zoom.png` with the box drawn on a real frame, and processes
nothing. **This is the final authority.** Both automated checks above have measured
failure modes; this one catches what they miss. Confirm every edge of the mark is
inside the red box.

### 4. Freeze it — `wmrm detect --box`

`--box` makes `detect` skip detection entirely and just write the preset, so
hand-measured coordinates become reusable like any other. Coordinates are stored
normalized, so the preset survives a resolution change.

### 5. Process — `wmrm run` / `wmrm batch`

```bash
wmrm run your.mp4 --preset fanza.json         # -> your-clean.mp4
wmrm batch ./inbox --preset fanza.json        # whole folder, skips finished files
```

`run` verifies its own output afterwards (resolution, fps, duration, audio, and
that the edit stayed local). On a GPU machine leave `--quality` alone; on CPU see
[which quality to use](#which---quality-and-how-long-it-takes) — `fast` is often
enough and ~20x quicker.

### More than one watermark?

If they sit next to each other — a studio logo beside a rating mark is the normal
case — use one box covering both. Marks far apart are not supported in a single
pass; run the tool twice, feeding the second run the first run's output.

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
| `wmrm grid IN` | frame + coordinate grid, to measure the box by hand. Always works. |
| `wmrm coverage IN --box ...` | check a box covers the whole mark. Watermark-agnostic. |
| `wmrm detect IN` | guess the box and write a preset + preview PNGs. Processes nothing. `--box` freezes coordinates you measured yourself. |
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

**Auto-detect breaks when the watermark changes.** It searches for pixel-locked
*edges*, which is the right signal for some marks and useless for others. Measured
against known ground truth:

| case | true box | detected | verdict |
| --- | --- | --- | --- |
| two semi-transparent marks side by side, 1080p | `1554,44,284,62` | `1554,44,283,64` | correct |
| badge on soft sky | `384,12,84,36` | `364,0,108,68` | over-covers — safe |
| badge on sky, handheld | `384,12,84,36` | `377,4,99,51` | over-covers — safe |
| badge on dense texture | `384,430,84,36` | `377,448,96,27` | under-covers |
| **opaque white text on a white wall** | — | **nothing found** | **total failure** |

Over-covering is harmless (slightly slower, a few clean pixels repainted).
Under-covering leaves residue. The last row is the one that matters: a different
watermark style can defeat the detector completely, so **treat `detect` as a first
guess and confirm with `wmrm coverage` + the preview**, or skip it and use
`wmrm grid`.

`wmrm coverage` is the objective guard, and it is watermark-agnostic because it
uses two signals: pixel-locked edges *and* collapsed temporal variance (an opaque
overlay freezes what is under it). Measured: from a box 160 px too small,
iterating its suggestion converged to 16 px too small — a big improvement, not a
guarantee. It reports `INCONCLUSIVE` rather than guessing when the background is
itself static (fixed camera on a plain wall), because then no statistic can
separate mark from wall.

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

`src/wmrm/pick.py` is **not wired into the CLI**. It generates a self-contained
HTML page for drawing the box with a mouse, and is parked for the UI phase — the
tool runs in headless containers today, where opening a browser is not practical.
It is the piece to start from when that UI is built.

Fixtures burn a fake badge onto a clean clip, so recovery is scored against
ground truth rather than guessed at. Cases: `smooth` and `busy` put the badge on
soft sky (easy), `detail` puts it on dense texture (hard — the case that
matters). Source clips come from `../dreamina-delogo/examples` if present, and
are otherwise synthesized with ffmpeg so the tests run anywhere.
