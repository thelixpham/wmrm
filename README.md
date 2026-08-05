# wmrm — fixed-position watermark removal

Internal CLI. Removes a watermark that is always the same and always in the same
place, keeps the audio bit-exact, and checks its own output. Runs fully offline.

Findings, measurements and the quality trade-offs live in
[../REPORT.md](../REPORT.md). This file is only how to run it.

---

## If you only read one section

**Just run the script.** It finds the watermark, processes everything, and puts
the results in `outbox/`.

```bash
./run.sh                        # every video in inbox/
./run.sh clip.mp4               # one file
./run.sh a.mp4 b.mp4 c.mov      # several files
./run.sh /data/videos           # another folder
./run.sh /data/videos x.mp4     # folders and files mixed
```

The first run detects the watermark and **saves the box to `preset.json`**. Every
run after that reuses it, so results are repeatable instead of re-guessed. Delete
`preset.json` to recalibrate.

Tune it with environment variables, no flags to remember:

```bash
CORNER=tl ./run.sh                  # watermark is top-left
QUALITY=video ./run.sh              # mark must be COMPLETELY gone (needs a GPU)
FORCE=1 ./run.sh                    # redo files already in outbox
OUTBOX=/data/out ./run.sh clip.mp4
```

`FORCE=1` matters more than it looks: without it, files already in `outbox/` are
skipped, so after changing anything you will be looking at the **old** output.

<details>
<summary>Prefer the CLI directly?</summary>

```bash
wmrm batch ./inbox --detect            # detect once, apply to the folder
wmrm run clip.mp4 --detect             # one file
```

It detects **once** on the first file, applies that box to the whole folder, and
writes a preview PNG next to it. **Look at that preview afterwards** — detection is
a guess with real failure modes (see below), and this is the only thing standing
between a bad guess and a folder of bad output.

**Once you know the coordinates, pass them instead** — no detection, no surprises:

```bash
wmrm batch ./inbox --box 1640,20,205,62
```

For a watermark you handle regularly, save the coordinates once and reuse them:

```bash
wmrm detect one.mp4 --box 1640,20,205,62 --preset fanza.json   # save
wmrm batch ./inbox --preset fanza.json                          # reuse forever
```

</details>

### Which engine — this is the only real decision

**Must the mark be *completely* gone?** Then `QUALITY=video ./run.sh`. It is the only
option that gets there. Needs a GPU and a [ProPainter](#complete-removal--quality-video)
checkout.

**Is "almost gone, nothing damaged" enough?** Leave the default. No GPU, 34 fps,
never touches a pixel it was not asked to.

Measured on the reference clip, tile 400×168, background locked-edge floor **12.24**
(residual at or below that means no watermark is findable any more):

| | residual | corr | detail in the box | GPU |
|---|---|---|---|---|
| untouched | 25.60 | 1.00 | — | — |
| `unblend` (default) | 13.02 | **0.99** | **untouched** | no |
| `high` (LaMa) | 12.42 | 0.68 | **wrecked when leaves cross the box** | wants one |
| **`video` (ProPainter)** | **11.36** | 0.82 | preserved | **needs one** |

`corr` is whether the patch changes where the real content changed; low means it
moves on its own, which is what reads as boiling.

Why the difference. A translucent mark suppresses the background but does not delete
it, so `unblend` divides the blend back out — recovery, not invention, which is why it
cannot flicker. But its leftover is proportional to the error in alpha, and alpha
must be estimated from statistics that are unobservable under the mark, so it can
never reach zero. `high` reaches zero by deleting the region, then invents a
replacement from the same frame, per frame. `video` reaches zero by taking the pixels
from **neighbouring frames**, where the same content is not covered — real content,
which is also why it stays coherent.

**Everything else in this file is for when something looks wrong.** Skip it until
it does.

Rules that cover most mistakes:

- One preset per watermark **design**, named after it. A preset measured on one
  logo is meaningless for a different logo.
- Round the box **outward**. Too large costs a little speed; too small leaves a
  sliver of watermark in the output.
- Read the `[cfg]` banner each run — it prints the box, the tile size, the engine
  and the device before any frame is touched.
- Only if you use `QUALITY=high`: confirm the banner says `cuda (<gpu name>, ...)`.
  If it says `cpu`, you have the CPU-only torch wheel and are running 20–50×
  slower for byte-identical output. `EXTRA="--device cuda" QUALITY=high ./run.sh`
  makes that a hard failure instead of a silent one.

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
that the edit stayed local). Leave `--quality` alone unless the mark is opaque —
the default runs at 34 fps on CPU, so there is nothing to trade away.

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
sudo apt update && sudo apt install ffmpeg -y
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
wmrm run tests/fixtures/detail-marked.mp4 -o /tmp/smoke.mp4 --box 379,427,91,43
```

Must end with `=> all checks passed`. It exercises the default engine, which loads
no model — so to prove the GPU half of the install works, run it once more with
`--quality high --patch-hold 8` and check the banner says
`device : cuda (<gpu name>, ...)`. `--patch-hold 8` is only there to keep that
check quick; leave it off for real work.

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
| `--quality` | `unblend` | `unblend` = recover the background (**leave it here**), `high` = LaMa, for opaque marks, `fast` = ffmpeg delogo, `draft` = cv2.inpaint. See below. |
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

### Why `unblend` is the default

Run `wmrm detect` and look at the `opacity` line. If it says `semi` — the normal
case — the background is still visible through the mark and can be **recovered**
rather than invented. That is what the default does; nothing to pass.

A translucent mark is an alpha blend, `C = a*W + (1-a)*B`. Fit `a` and `W` once
from ~40 sampled frames and every frame is solved back to the real `B`. Inpainting
throws that away and guesses instead — which is fine over a blurred background and
obvious over detailed one.

Measured on a 1080p clip where the mark sat over maple leaves, and where the
customer rejected the LaMa output:

| | LaMa | un-blend |
| --- | --- | --- |
| leaf detail under the mark | destroyed, replaced by a pink smear | preserved |
| frame-to-frame motion, correlation with real content | **0.30** (invented — this is the flicker) | **0.98** |
| speed at 1080p | 1.4 fps | **34 fps** |

Un-blend also **cannot flicker** — the fitted map is constant, so there is no
per-frame guessing to vary. On the opaque test fixture it scores 0.87 correlation
with the real motion against LaMa's 0.01.

### Reading the fit report

`run` prints the fit before it touches a frame. What the lines mean:

```
alpha in mark : median 0.040  p90 0.186  p99 0.298  max 0.387
              : 1.0% of the mark has alpha > 0.3
alpha scale   : x1.0 (chosen by residual sweep)
mark residual : 8.76 -> 8.76 locked-edge energy in the mark
```

**`mark residual` is the line that matters.** It is how much pixel-locked structure
is left inside the box, measured the same way the detector finds watermarks. Compare
it to the surrounding background — on this clip the background floor was `10.05`, so
`8.76` means the region is now *quieter* than its surroundings.

Do not read `alpha median` as "how strong the mark is". The median is over the whole
mark box, which is mostly gaps between glyphs, so it reads near zero for any text
mark. Use `p99` and `max`, and read the residual for whether removal worked.

Two guards run automatically and are worth knowing about:

- **Alpha is capped by physics.** `B >= 0` forces `alpha <= min(C)/W` per pixel, so
  on dark backgrounds alpha is clamped rather than driving the recovery negative.
  Without it, the mark crossing a saturated red pillar left a dark blotch.
- **Alpha is not smoothed.** Strokes are 2–3 px wide, so filtering at that scale
  softens the matte and the division stops cancelling the stroke. Measured: residual
  8.58 unsmoothed vs 14.06 with the old 3 px default, where the logo stayed readable.

Limits: it needs a **semi-transparent** mark (opaque white text has nothing to
recover — use `high`), at least ~12 frames, and background that actually changes
behind the mark. Pixels the mark blocks almost totally are handed to LaMa
automatically.

### Complete removal — `--quality video`

The only engine that removes the mark outright without inventing content per frame.
It is [ProPainter](https://github.com/sczhou/ProPainter): optical flow between frames,
so the hole is filled with the same content seen uncovered a few frames earlier or
later. On this footage the leaves move, so that content genuinely exists.

One-time setup — it is a research repo, not a package, so there is nothing to pip
install:

```bash
git clone https://github.com/sczhou/ProPainter.git
export PROPAINTER_HOME=$PWD/ProPainter          # or clone it beside this project
uv pip install av addict einops future scipy scikit-image imageio-ffmpeg \
               pyyaml requests timm matplotlib
uv pip install --index-url https://download.pytorch.org/whl/cu124 torchvision
```

The `torchvision` line is not optional and must come from the **same index as your
torch**. A PyPI torchvision next to a `+cu124` torch fails with
`operator torchvision::nms does not exist`, which does not mention either package.

Weights (~190 MB total) download themselves on first run.

```bash
QUALITY=video ./run.sh                    # or:
wmrm run clip.mp4 --preset fanza.json --quality video
```

Speed: **9m14s for 151 frames of 1080p on 6 CPU cores** — do not run it on CPU for
real work. Expect 20–50× that on CUDA. The log's `[cfg] device` line tells you which
you got, using ProPainter's own selection rule.

Knobs, in the order worth touching: `--pp-segment` (frames per invocation, default
400 — lower it if you run out of memory, since it loads a whole segment at once),
`--raft-iter` (flow iterations, default 20; lower is faster and rougher), `--no-fp16`.

Two things this wrapper does that matter:

- **It reads ProPainter's PNG frames, never its mp4.** Its writer pads to a multiple
  of 16 — `resizing from (400, 168) to (400, 176)` — so the mp4 no longer lines up
  with the crop it came from. There is an assertion on the tile size for this.
- **It segments the clip with discarded overlap.** `--subvideo_length` only chunks
  inference; the script still loads every frame it is handed into memory first, so an
  unsegmented hour of video does not fit. Overlap frames give the model context on
  both sides of every frame that is kept, so the joins do not show.

### Which `--quality`, and how long it takes

Measured on CPU (6 cores). The 1080p column is a real clip with a 284x62 mark:

| | 480x640 | 1080p | 1 minute of 1080p |
| --- | --- | --- | --- |
| `unblend` (default) | fast | **34 fps** | **~30 s** |
| `high` (LaMa) | 5.5–7 fps | 1.4 fps | **~22 min** |
| `video` (ProPainter) | — | **0.27 fps** | **~1.8 h** ← CPU is not an option |
| `fast` (delogo) | ~realtime | ~realtime | ~1 min |
| `draft` (cv2) | ~95 fps | fast | seconds |

The default is the fastest **and** the least destructive, so on a semi-transparent
mark there is no speed/quality trade to make — which is why the older advice in this
file to "try `fast` first on CPU" is gone. `fast` and `draft` remain only as
fallbacks for when un-blend cannot apply: too few frames, or a background that never
changes behind the mark.

The real trade is elsewhere: `video` is the only engine that removes the mark
completely, and the price is a GPU. On CPU it is ~400× slower than the default.

**Opaque marks are the exception.** There un-blend has nothing to recover and
`high` is the right answer — 1.4 fps on CPU, 20–50× that on CUDA. `--patch-hold N`
cuts its runtime roughly N-fold; read the caveat below first.

### Detect options

`--corner tr|tl|br|bl` (default `tr`), `--samples 40`, `--roi-frac 0.30`,
`--grad-threshold 10`, `--persistence 0.90`, `--max-area 10`.

Defaults are fine for a corner badge. Lower `--grad-threshold` for a fainter
watermark.

---

## Limits worth knowing before you rely on it

Full measurements in [../REPORT.md](../REPORT.md) §4.

**Temporal coherence is unsolved for the *inpainting* engines.** They fill each
frame independently, so the patch either boils (`high`) or sits frozen while the
background moves (`fast`/`draft`). Measured: LaMa's frame-to-frame variation has
the right amplitude but only **0.01 correlation** with where the real content
changed — the variation is invented. `--patch-hold N` converts boiling into
freezing rather than fixing it; treat it as a speed lever. A real fix needs
ProPainter or E2FGVI on a GPU (REPORT.md §5, phase 2).

This is why `unblend` is the default and why an opaque mark is the harder case: the
default's transform is fitted once and constant, so it has no per-frame guess to
vary and **cannot** flicker (measured 0.98 correlation with true motion). The
paragraph above applies only when you leave the default.

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
two marks far apart in one pass; opaque marks over moving detail (that is the
`high` path, and it flickers — see above).

**The default can leave a faint trace.** Not zero: on the reference clip the mark
went from plainly legible to barely detectable, not to nothing. Check the
`mark residual` line against the background around the box, and look at the output
before shipping a batch — the metric saturates before the eye does, so once residual
drops below the surrounding floor it can no longer tell you whether a trace remains.

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
