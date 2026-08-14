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

**It detects the watermark on every video, separately**, then checks that box
before touching a pixel. A box measured on one source says nothing about a
different crop, a different logo or a different placement, so a shared box is only
right when you know the batch is uniform.

Every detected box is gated by `wmrm coverage`. A box that is provably too small
**stops that file** rather than shipping a video with watermark fringe still in it,
and the ones it could not answer for are listed again at the end for you to eyeball.

It cleans up after itself. A file whose box came back `covered` has nothing left to
check, so its preview PNGs (~465 KB a file) and its saved box are deleted. Files that
were stopped or came back `INCONCLUSIVE` keep theirs — those are the evidence, and
`outbox/.presets/` is the record of what box each of them was made with. Your inputs
and outputs are never touched.

If your batch *is* uniform — same watermark, same pipeline — one box is better:
it is repeatable, and there is one preview to check instead of fifty.

```bash
DETECT=once ./run.sh                # detect on the first file, reuse for all
```

That saves the box to `preset.json`, and an existing `preset.json` always wins over
any detection. Delete it to recalibrate.

Tune it with environment variables, no flags to remember:

```bash
CORNER=tl ./run.sh                  # watermark is top-left
QUALITY=video ./run.sh              # mark must be COMPLETELY gone (needs a GPU)
DETECT=once ./run.sh                # one box for the whole run
COVERAGE=0 ./run.sh                 # process even boxes that fail the coverage check
CLEAN=0 ./run.sh                    # keep every preview, not just the ones worth looking at
FORCE=1 ./run.sh                    # redo files already in outbox
OUTBOX=/data/out ./run.sh clip.mp4
```

`FORCE=1` matters more than it looks: without it, files already in `outbox/` are
skipped, so after changing anything you will be looking at the **old** output.

`COVERAGE=0` is for when you have looked and disagree with the check — it is not a
speed lever, and it turns a stopped file into a silently bad one.

<details>
<summary>Prefer the CLI directly?</summary>

```bash
wmrm batch ./inbox --detect            # detect once, apply to the folder
wmrm run clip.mp4 --detect             # one file
```

```bash
export R2_ACCOUNT_ID=...  R2_ACCESS_KEY_ID=...  R2_SECRET_ACCESS_KEY=...
export R2_BUCKET=remove-watermark

wmrm pull --stat uploads/3d809a59-3e5c-4977-9dd6-bbc15b4f58d6/MOGI-125.mp4
wmrm pull uploads/dca49130-7a17-4cb5-9cde-9390efd6d590/MOGI-119.mp4
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

**Must the mark be *completely* gone?** That is `--quality video`, and it is the
**default**. It is the only option that gets there. Needs a GPU — on CPU it refuses to
run rather than take 400× longer.

**Is "almost gone, nothing damaged" enough, or is there no GPU?** Then
`--quality unblend` (`QUALITY=unblend ./run.sh`). It never touches a pixel it was not
asked to, and it cannot flicker.

Measured on the reference clip, tile 400×168, background locked-edge floor **12.24**
(residual at or below that means no watermark is findable any more):

| | residual | corr | detail in the box | GPU |
|---|---|---|---|---|
| untouched | 25.60 | 1.00 | — | — |
| `unblend` | 13.02 | **0.99** | **untouched** | no |
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
that the edit stayed local). `--quality` defaults to `video`, which needs a GPU; on a
machine without one, pass `--quality unblend`.

### More than one watermark?

If they sit next to each other — a studio logo beside a rating mark is the normal
case — use one box covering both. Marks far apart are not supported in a single
pass; run the tool twice, feeding the second run the first run's output.

**Nothing in the normal flow tells you a second mark exists.** `wmrm detect`
searches one corner by design, and `wmrm coverage` only inspects a ring around the
box you gave it — "covered" means *nothing survives just outside this box*, never
*the frame is clean*. A studio logo in one corner and a caption in the other will
pass every check a run makes and still be in the output. Measured on a real file:
detection found the rating badge, coverage said covered, verify passed, and two
other marks were untouched.

So look at the whole frame first:

```bash
python scripts/scan-fixed-edges.py VIDEO.mp4
```

It averages the *signed* gradient over 60 frames spread across the clip. Moving
content points different ways from frame to frame and cancels; anything pinned to
the same pixels survives. Two PNGs land next to the video — the raw energy map, and
the same in red over a real frame. Every fixed overlay in the video is legible in
them, including semi-transparent ones that a variance test would miss entirely.
It reports no boxes on purpose: see the script's docstring for the five rounds of
threshold tuning that produced nothing trustworthy, and read the coordinates off
the picture instead.

**Detect can be made to pick up a second mark in the same corner.** It rejects
anything not present in ≥90% of sampled frames, which is what stops it boxing
subtitles — but a studio logo that is not on for the whole film looks identical to
that rule. Lower it a notch at a time and check each result:

```bash
wmrm run VIDEO.mp4 --persistence 0.5 --preview-only    # look, decide, no processing
wmrm run VIDEO.mp4 --persistence 0.5                   # detect and process, one command
```

Every detect knob — `--corner`, `--samples`, `--roi-frac`, `--grad-threshold`,
`--persistence`, `--max-area` — works on `run` and `batch`, not just on `detect`,
because they are the commands that use the box. Tuning detection does not cost you a
separate `detect` invocation and a preset file to pass back in. `run` writes the box
it settled on to `<input>-preset.json` either way, so once a source is dialled in you
can stop detecting and pass that instead.

Components that survive are unioned into one box automatically, so a second mark
that clears the threshold joins the first one on its own. Verify each step: a lower
threshold also lets *static scenery* qualify as a pixel-locked edge — a fixed camera
on a plain surface is exactly the case that inflates the box — and a box that has
crept into the background costs GPU time for nothing.

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

**On a Blackwell card (RTX 50-series, RTX PRO Blackwell) `cu124` is not enough.**
Those are compute capability `sm_120`, and a cu124 build stops at `sm_90`, so it
installs cleanly, reports `cuda (…)` in the banner, loads the models — and then
dies at the first kernel launch with `no kernel image is available for execution
on the device`. Use a newer index:

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

Verify before running anything, because every symptom above is identical either
way until the first kernel launch:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"   # must contain sm_120
```

If it does not, try `cu130`. `torch` and `torchvision` must come from the **same**
index — see the warning under `--quality video`.

**Step 2 — the rest.** Identical on both:

```bash
uv pip install --no-deps simple-lama-inpainting
uv pip install "pillow>=10" opencv-python-headless numpy
uv pip install --no-deps -e .
source .venv/bin/activate
```

Pulling sources from R2 (`wmrm pull`)? Add `boto3` — either `uv pip install
boto3`, or `uv pip install -e '.[r2]'` to go through the extra.

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

- `simple-lama-inpainting` needs `--no-deps`: it declares `pillow<10`, which does
  not build on Python 3.12. It also declares `numpy<2` and `opencv-python<5`, and
  none of the three bounds reflect what its code needs — measured, LaMa imports
  and runs here under pillow 12.2, numpy 2.4 and opencv-python-headless 5.0.
  `[tool.uv] override-dependencies` in `pyproject.toml` says so, which is what
  lets any command that resolves the project properly (`uv pip install -e '.[r2]'`,
  say) succeed instead of reporting an unsatisfiable conflict. The `--no-deps`
  above is still the shortest path, and the override is what makes dropping it
  possible.
- That override *drops* `opencv-python` rather than relaxing it, because it and
  `opencv-python-headless` both ship `cv2`: with both installed the import order
  decides which one you get.
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
| `wmrm pull KEY` | download a source video from Cloudflare R2. Resumable. |
| `wmrm serve` | run this machine as a job worker over HTTP. See [below](#running-as-a-worker--wmrm-serve). |

`run` and `batch` need either `--preset` or `--box`.

### Getting the file — `wmrm pull`

Needs `boto3`: `uv pip install -e '.[r2]'`. Credentials from the environment,
nothing is read from the repo:

```sh
export R2_ACCOUNT_ID=...            # or R2_ENDPOINT for a custom domain
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
export R2_BUCKET=remove-watermark

wmrm pull --stat uploads/3d809a59-.../MOGI-125.mp4         # size first
wmrm pull uploads/3d809a59-.../MOGI-125.mp4 -o work/       # then fetch
```

Then the normal commands, on a local file, unchanged:

```sh
wmrm grid work/MOGI-125.mp4 --corner tr
wmrm coverage work/MOGI-125.mp4 --box X,Y,W,H
wmrm run work/MOGI-125.mp4 --box X,Y,W,H
```

This is a separate step rather than a `r2://` URI that `run` accepts, and that
is deliberate at the sizes here. A 22–100 GB transfer is hours; the processing
after it is hours more and is *not* resumable. Fused into one command, a run
that dies on a bad box throws the download away with it.

**It resumes.** 64 MiB chunks are fetched by 8 parallel ranged GETs into a
preallocated `.part`, and the chunks that landed are recorded in a sidecar
JSON. Re-run the identical command after a dropped connection or a Ctrl-C and
it picks up — which matters, because at 100 GB a transfer that can only start
from zero is a transfer that may never finish. If the object changed on R2
(different size or etag) the local bytes are no longer a prefix of it, so the
part file is discarded rather than silently producing a corrupt mix.

`--workers` past 8 only helps until the link or the disk saturates; the
progress line reports the achieved rate, so compare rather than guess. Free
space is checked before the first byte, not discovered at 90 GB.

`wmrm pull --list uploads/` lists keys under a prefix when you only half
remember one.

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

### Why `unblend` is the one to reach for without a GPU

Run `wmrm detect` and look at the `opacity` line. If it says `semi` — the normal
case — the background is still visible through the mark and can be **recovered**
rather than invented. That is what `--quality unblend` does.

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

**No clone needed any more.** ProPainter is vendored at `vendor/ProPainter` (upstream
commit and what was cut: `vendor/README.md`), because it is a research repo with
nothing to pip install and the pipeline imports it rather than shelling out to it.
Only the extra Python packages are left to install:

```bash
uv pip install av addict einops future scipy scikit-image imageio-ffmpeg \
               pyyaml requests timm matplotlib
uv pip install --index-url https://download.pytorch.org/whl/cu124 torchvision
```

The `torchvision` line is not optional and must come from the **same index as your
torch**. A PyPI torchvision next to a `+cu124` torch fails with
`operator torchvision::nms does not exist`, which does not mention either package.

Weights (~190 MB total) download themselves on first run, into
`vendor/ProPainter/weights/`. They are gitignored — GitHub rejects blobs over 100 MB,
and a binary that size cannot be taken back out of history cleanly.

`PROPAINTER_HOME` still overrides the vendored copy if you need to test against a
different upstream. Point it at a checkout that has `wmrm_worker.py` copied into it,
or the run fails with an explanation — that file is ours, not upstream's.

```bash
QUALITY=video ./run.sh                    # or:
wmrm run clip.mp4 --preset fanza.json --quality video
```

Speed: **9m14s for 151 frames of 1080p on 6 CPU cores** — do not run it on CPU for
real work. Expect 20–50× that on CUDA. The log's `[cfg] device` line tells you which
you got, using ProPainter's own selection rule.

Knobs, in the order worth touching: `--pp-segment` (frames per model call, default
400 — lower it if you run out of memory, since the model materialises a whole segment
as a tensor), `--raft-iter` (flow iterations, default 20; lower is faster and
rougher), `--no-fp16`, `--device cuda` to make a CPU fallback a hard failure instead
of a silent 400× slowdown.

`--pp-workers` is **accepted and ignored**. It used to overlap the per-segment model
*load* with another segment's compute, which mattered when every segment was a fresh
process. There is one resident model now, so there is no load left to hide, and two
segments handed to the same device would serialise on it anyway.

Four things this wrapper does that matter:

- **Nothing is written to disk.** Frames stream: one ffmpeg decodes and crops the tile
  into a pipe, the model works in memory, a second ffmpeg composites the result back
  over the untouched source. The older shape of this wrote every tile frame as a PNG
  first, so an hours-long video paid for a million PNG encodes before the model saw
  frame one, and temp disk grew with duration (48.1 KB/frame — ~83 GB for ten hours).
  Peak is now a few hundred MB of in-flight frames, flat in video length.
- **Decode, model and composite run at the same time**, each in its own thread with a
  bounded queue between them. Bounded is the point: the queues are what keeps memory
  flat when one stage is slower than the others.
- **The models load once per run, not once per segment.** Upstream's inference sits
  entirely inside `if __name__ == '__main__':`, so a subprocess per segment reloaded
  RAFT, the flow-completion net and the inpaint generator every time — ~190 MB and a
  CUDA context, ~2700 times on a ten-hour video at the default segment size.
  `vendor/ProPainter/wmrm_worker.py` is the importable equivalent, and
  `tests/test_propainter_parity.py` asserts it produces what the script produced,
  bit for bit.
- **It pads to a multiple of 8 rather than resizing.** Upstream crops its processing
  size down to a multiple of 8 and cubic-resizes the result back, so a tile that was
  not a multiple of 8 came back resampled — softened over exactly the pixels being
  repaired, and off by a fraction of a pixel against the frame it is composited onto.
  The output kept its requested dimensions, which is why a shape assertion never saw
  it. Tiles are now aligned in `region.py` as well, so padding is usually a no-op.

Overlap between segments is still discarded on both sides, so the model always has
temporal context either side of every frame that is kept and the joins do not show.

#### Scene detection decodes the whole file first

Before any frame is processed, one ffmpeg pass scores every frame against its
predecessor to find the cuts, so no segment ever spans one. On a feature-length
input that pass is long and silent — the run looks hung after `models loaded`,
and the next line only appears when it finishes. It now says what it is doing
and which decoder it is using.

**It decodes through NVDEC when there is a CUDA card.** This is the only step
where decode is the entire cost rather than ~7% of it, and `select='gt(scene,…)'`
is serial by nature — each frame is scored against the one before it — so extra
cores do not help. Measured on a 128-core EPYC, the CPU path pinned **1.8 cores
and left 126 idle**. Measured on 120 s of 1080p with a 4090:

| | wall | CPU time | cuts found |
| --- | --- | --- | --- |
| CPU decode | 32.9 s | 54.0 s | 4 |
| NVDEC (`-hwaccel cuda`) | **6.8 s** | **3.0 s** | 4 |

4.8× on the clock, 18× less CPU, same answer — roughly **43 min → 9 min** on a
2.6-hour film. If the card refuses the file (NVDEC rejects some profiles and bit
depths) it falls back to the CPU and says so, rather than just running 5× slower
for no stated reason. `--device cpu` opts out.

**Everything else was measured and rejected.** The reason none of it worked is one
number: against a decode-only null sink, the scene filter accounts for **7% of the
pass**. The other 93% is decoding. Nothing that optimises the filter can matter,
and nothing that decodes every frame can win — which leaves moving the decode off
the CPU, i.e. NVDEC.

| approach | result | why it fails |
| --- | --- | --- |
| `scdet` filter instead of `select` | 1.00× | same cost to the frame; with `sc_pass=1` it also found 5 of 23 cuts |
| downscale before the filter (PySceneDetect's advice) | 0.8× | ffmpeg still decodes at full resolution, so the scale is pure added work |
| split the timeline across parallel ffmpegs | 1.2× | identical cuts (`-copyts` keeps timestamps absolute), but cores were never the constraint — the serial pass already used 4.3 of 6 |
| keyframe index for candidates, decode only to confirm | 1.05× at best | see below |

The last one deserves its epitaph, because it looks like it should work. Every real
cut *is* a keyframe (encoders force an IDR at a scenecut) and the container index
lists them without decoding anything — 0.07s against 4.80s. But keyframes are a
superset, so each candidate still needs a seek and a short decode to confirm, and
that cost scales with the file exactly as fast as full decode does. Measured on a
40-minute fixture across keyframe densities: 0.74× at one keyframe per 2s, and
1.05× at the theoretical floor where every keyframe already *is* a cut and there is
nothing left to reject. 5% is not worth a dependency on how the input was encoded.

All of these produced the correct cuts. They were rejected on speed alone.

A failed decode here now raises. It used to return an empty cut list, which is
indistinguishable from a clean scan that found nothing: the run would report "no
scene cuts found", plan fixed segments, and silently lose the protection this
pass exists to provide.

#### The scene score cannot see a fade through black — the same pass now looks for one

The transition that matters most to this model is the one its cut detection is blind
to. A fade is gradual by definition, so it never scores. Measured on a real intro — a
black frame, a rating card, then a fade into the first shot:

| | found |
| --- | --- |
| `select='gt(scene,0.3)'` over the first 20 s | cuts at 11.78 s and 15.65 s, **nothing at the fade** |
| same at threshold 0.1 | the same two cuts |
| what the picture actually did | pure black through frame 243, full brightness by frame 246 |

So the black run and the bright shot after it were planned as **one segment**, which is
exactly the case segmenting on cuts exists to prevent. The watermark region is masked in
every frame of a black run and nothing moves in it, so flow-guided propagation has
nothing to propagate: the whole fill becomes the transformer's invention, conditioned on
that segment's reference frames, which were the bright ones. The output was a glowing
smear the size of the hole over an otherwise black frame. It began at frame ~216 —
`ref_stride * ref_num / 2` = **40 frames** ahead of the picture, i.e. precisely the reach
of ProPainter's global references, which is what identified the mechanism. Measured at
10/255 on a 1080p output, where nobody noticed it, and plainly visible at 4K.

`blackdetect` answers what the scene score cannot, for one more filter in a pass that
was already decoding every frame — no second decode. Its boundaries (the first black
frame, and the first frame that is not) join the scene cuts, so a black run becomes its
own segment and bright frames can never be its references. Black runs shorter than
`--pp-min-shot` are not asked for: `_segment_plan` would merge them away, and in a night
scene there would be thousands. `--pp-no-black-cuts` opts out.

**A second, independent guard, because the first one depends on a boundary being
found.** `_dark_guard` checks the one thing that is true whatever the model did: a fill
can be no brighter than the picture it sits in. Where the tile *around* the hole holds no
picture at all — 99th-percentile luma ≤ 24/255 — the hole gets the median of that
surround instead of the invention, which is black on a black frame and removes the
watermark just the same. It is deliberately narrow: with real content around the hole,
or a dark shot that still holds something bright, the model wins, because a median over a
real surround would be a flat patch where there was a picture. The count of corrected
frames is reported at the end of the run, so footage dark enough for this to fire all the
way through is visible as such instead of silently flattened.

### A killed run resumes — `--resume` is the default

A feature-length 4K file is 4–9 hours of model time. It used to be all-or-nothing:
the output was one temp file, and any failure deleted it. Measured twice on the same
job, at the very last segment, after nine hours each time.

The video is now composited in **parts** of `--pp-part` frames (default 3600, two
minutes at 30fps), written to `<output>.parts/` beside the output and joined by
stream copy at the end. A part whose frame count checks out is finished for good, so
a crash costs at most one part — everything else is on disk and gets reused.

```bash
wmrm run big.mp4 --preset box.json     # dies at hour 8
wmrm run big.mp4 --preset box.json     # carries on from the last finished part
```

The same command, twice. **`--resume` is on by default**, and the asymmetry is
deliberate: off by default, forgetting the flag silently deletes hours of finished
work before you can read the message; on by default, the worst case is reusing work
it should not have — and that cannot happen, because the manifest fingerprints the
source and every setting that decides a pixel. Change the box, the segment size, the
crf, or the input, and it starts over on its own and says which setting changed.
`--no-resume` forces that.

Two details that make resumption exact rather than approximate:

- The model **restarts at the beginning of the segment** the crash landed in, not at
  the next frame. A segment's output depends on the whole block it was handed, so
  restarting mid-segment would give those frames different context and different
  pixels. `tests/test_resume.py` pins the result byte-identical to an uninterrupted
  run — that is the property, not "looks about right".
- Scene cuts and the auto-chosen segment size are read back from the manifest, so a
  resume neither re-decodes the whole source to find the same cuts nor asks
  `auto_segment` again — free VRAM at startup is not a property of the video, and
  half a film made with different settings from the other half is not acceptable.

Watch progress at any time. The parts are ordinary mp4 files:

```bash
ls -1 <output>.parts/part-*.mp4 | wc -l     # how many are done
ffplay <output>.parts/part-000042.mp4       # what the result looks like
```

Budget disk for **twice the output size**: the join needs the parts and the finished
file to exist at the same moment. The parts directory is deleted only after the
joined file has been verified and moved into place.

### Which `--quality`, and how long it takes

Measured on CPU (6 cores). The 1080p column is a real clip with a 284x62 mark:

| | 480x640 | 1080p | 1 minute of 1080p |
| --- | --- | --- | --- |
| `unblend` | fast | **34 fps** | **~30 s** |
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

All of them are accepted by `run` and `batch` too, which detect on their own when
given neither `--box` nor `--preset`. There is no detect-then-run pipeline to
assemble: `wmrm run VIDEO.mp4 [detect knobs]` is that pipeline, and it writes the
box it used to `<input>-preset.json` on the way through.

Defaults are fine for a corner badge. Lower `--grad-threshold` for a fainter
watermark, and `--persistence` for a mark that is not on screen the whole film —
see [More than one watermark?](#more-than-one-watermark) for why that one rejects a
real logo and how to check the box it gives you instead.

`wmrm run` with neither `--box` nor `--preset` detects and processes in one command,
and writes the box it used to `<input>-preset.json`. That file is a record, not an
input: nothing picks it up on its own, because a preset that merely happens to exist
quietly overriding what the command asked for is a failure this project has already
had once. Pass it with `--preset` to reuse it — which is the right move for a batch
from one source, since the coordinates are normalized and a box measured on a 4K file
applies unchanged to the 1080p cut of the same layout.

---

---

## Running as a worker — `wmrm serve`

Everything above assumes a person at a terminal. `wmrm serve` is the other mode: this
machine sits waiting, and jobs arrive over HTTP instead of over ssh.

```bash
export WMRM_POD_TOKEN=$(head -c 24 /dev/urandom | base64)   # keep this, you paste it later
echo "token: $WMRM_POD_TOKEN"

wmrm serve --host 0.0.0.0 --port 8000
```

**That is the only variable you have to set.** Everything else is worked out from the
machine and printed at startup, so what got chosen is visible rather than remembered:

```
[wmrm] pod id : abc123
[wmrm] machine: RunPod pod, state on /workspace
[wmrm]   work  : /workspace/wmrm-work/abc123
[wmrm]   state : /workspace/wmrm-state/abc123
[wmrm]   input : (r2 and url only -- set WMRM_LOCAL_INPUT_ROOT to allow kind=local)
[wmrm]   r2    : remove-watermark
[wmrm]   disk  : refuse below 50 GiB free
[wmrm]   jobs  : 1 at a time
[wmrm] serving on 0.0.0.0:8000  (docs at /docs)
```

`--host 0.0.0.0` is the default and it matters: a RunPod proxy cannot reach a server bound
to localhost. One worker only — job state is per-process and on disk, so a second worker
would be a second opinion about what this machine is doing.

Needs the `serve` extra, which `./setup.sh` installs. By hand:

```bash
uv pip install fastapi 'uvicorn[standard]' httpx pydantic
```

### The token is not optional

The pod's URL is a public address on the internet — `https://<podid>-8000.proxy.runpod.net`
— and nothing authenticates in front of it. Without a token, anyone who has that URL can
start jobs on the card you are renting, cancel a nine-hour run, and read the logs.

So an unset token is a **refusal**, not an open door: every route answers 503 and says
why. A deploy missing its configuration must not look like a working one.

`/live` is the exception. It takes no token so a platform health check can use it, and it
returns an empty body — there is nothing in it to learn.

### Where to keep it

Any secret will do — it is compared, not parsed:

```bash
head -c 24 /dev/urandom | base64
```

`export` alone lasts as long as the shell, which is not long enough. **On a pod, the
container filesystem is discarded on stop or restart**, so `~/.bashrc`, `/root/.env` and
anything else outside `/workspace` goes with it — and the pod comes back serving 503 with
no obvious reason why.

Two places that survive:

- **RunPod's environment variables for the pod** (in its configuration, not in the shell).
  This is the one to use: the platform re-applies it on every start, so a restart needs
  nothing from you.
- **A file on the volume**, if you would rather keep it with the code:

  ```bash
  echo "export WMRM_POD_TOKEN=$(head -c 24 /dev/urandom | base64)" > /workspace/.wmrm-env
  chmod 600 /workspace/.wmrm-env
  # then, each time:
  source /workspace/.wmrm-env && wmrm serve
  ```

  Note the quoting: the value is expanded when the file is written, not when it is read.

Leading and trailing whitespace is stripped on both sides, so a stray space from copying
out of a terminal is not going to cost you an afternoon.

### When the pod and the registry disagree

They are two copies of one secret, so they can drift. The symptom is specific and the
Pods page shows it verbatim:

> **unhealthy** (HTTP 401) — the pod rejected the token. Check it matches
> `WMRM_POD_TOKEN` on the pod.

Jobs are not dispatched to a pod in that state, so the failure is visible rather than
silent. The fix is to re-save the pod with the token it is actually running with.

**Rotating: pod first, registry second.** The other order leaves a window where this app
calls with a token the pod has already stopped accepting.

```bash
# 1. on the pod
export WMRM_POD_TOKEN=<new>          # and update wherever it persists
wmrm serve                            # restart it

# 2. in the web app: Pods -> that pod -> paste the new token -> Save
```

Saving re-probes immediately, so you find out on that screen whether the two now agree.

### Environment

| variable | required | default |
| --- | --- | --- |
| `WMRM_POD_TOKEN` | **yes** | —. Unset means every route answers 503. |
| `WMRM_POD_ID` | no | `RUNPOD_POD_ID`, else `local`. Namespaces the work directory. |
| `WMRM_WORK_DIR` | no | `/workspace/wmrm-work` on a pod, else `~/.cache/wmrm/wmrm-work` |
| `WMRM_STATE` | no | `/workspace/wmrm-state` on a pod, else `~/.cache/wmrm/wmrm-state` |
| `WMRM_LOCAL_INPUT_ROOT` | no | unset. Required only for `input.kind: "local"`. |
| `WMRM_MIN_FREE_GB` | no | 50 on a pod, 2 elsewhere |
| `WMRM_MAX_CONCURRENT` | no | 1 |
| `R2_*` | for `kind: "r2"` | unset. See [`wmrm pull`](#getting-the-file--wmrm-pull) — the same four. |
| `WMRM_MEZON_WEBHOOK_URL` | no | unset. A Mezon channel webhook; see [Telling a person](#telling-a-person-mezon) |
| `WMRM_ECHO_RUN` / `WMRM_RUN_LOG` | no | see [Watching a run](#watching-a-run) |
| `WMRM_DOCS` | no | on. `off` removes `/docs`, `/redoc` and `/openapi.json` |

"On a pod" means `RUNPOD_POD_ID` is set **and** `/workspace` is writable. Both, because an
image can ship the mount point without the volume attached, and because `/workspace`
turned out to exist — empty and root-owned — on an ordinary workstation image, where
testing for the directory alone decided "this is a pod" and put the work directory
somewhere unwritable.

On a pod, everything that must survive a stop or restart has to be under `/workspace`: the
container filesystem is discarded, taking the venv, the model weights and any job state
with it.

### The API

Swagger UI at `/docs`, ReDoc at `/redoc`, the schema at `/openapi.json`. **Open — no token.**
Click **Authorize**, paste the token, and "Try it out" works against the live pod.

The docs are open and the API is not, which is the only arrangement that works: a browser
cannot attach an `Authorization` header to a URL you type, so putting the docs behind the
token means `/docs` answers `{"detail":"bad or missing bearer token"}` and the interactive
docs cannot be reached by the one tool that renders them. It buys little either — the
routes are `/jobs` and `/health`, guessable in one attempt, and knowing the shape of an API
gets nobody past the token on the calls that matter. `WMRM_DOCS=off` removes them if you
would rather not serve them at all.

| | |
| --- | --- |
| `GET /live` | 200, no auth, empty body. For a platform health check. |
| `GET /health` | what this machine is: GPU, `archList`, free disk, engines it can run, R2 configured, jobs in flight |
| `POST /jobs` | accept a job. **202 in well under a second** — the run itself is a background task |
| `GET /jobs/{id}` | state, phase, progress, outcome, the run's report |
| `GET /jobs` | every job this pod knows about. Used to reconcile against whoever dispatched them |
| `POST /jobs/{id}/cancel` | 202, idempotent |
| `GET /jobs/{id}/log?tail=N` | the last N lines, when per-job files are on (see below). For a human — nothing in the protocol parses it |
| `DELETE /jobs/{id}` | remove a finished job's state and work directory |

A submission, with the three shapes an input can take:

```jsonc
{
  "schema": 1,
  "jobId": "job_01JBQ7Z8K3M4N5P6Q7R8S9T0V1",
  "dispatchToken": "dt_9f3c...",
  "callbackBaseUrl": "https://wmrm.example.com",

  // "r2"    -> the pod fetches the key itself, 8 parallel ranged GETs, resumable
  // "url"   -> a presigned GET, for a pod holding no credentials
  // "local" -> already on the volume; must sit under WMRM_LOCAL_INPUT_ROOT
  "input":  { "kind": "r2", "key": "uploads/3d80.../4K_MOGI-130.mp4" },
  "output": { "kind": "r2", "key": "output/job_01J.../4K_MOGI-130-clean.mp4" },

  "engine": "video",
  "box": { "x": 1640, "y": 20, "w": 205, "h": 62 },   // omit and the pod detects one
  "options": { "device": "cuda" },                     // any run flag, camelCased
  "heartbeatEverySeconds": 30
}
```

`options` takes the run flags by camelCased name — `crf`, `x264Preset`, `ppSegment`,
`raftIter`, `dilate`, and the detect knobs. The four flags that only exist in the negative
are sent as positives and inverted on the way through: `"fp16": false` becomes
`--no-fp16`, and likewise `ppBlackCuts`, `resume`, `verify`.

`coverageGate` defaults to **`strict`** here, unlike the CLI. An unattended run is exactly
the case that must not ship a maybe: a box the coverage check calls too small fails the
job, and one it cannot judge sends the job for review rather than guessing.

### Watching a run

Everything a run prints goes to **the server's own output**, one line at a time, tagged
with the last six characters of the job id:

```
[job job_01J...] accepted: engine=video input=r2 (no box -- the pod will detect one)
[r2] uploads/3d80.../4K_MOGI-130.mp4 -> /workspace/wmrm-work/abc123/job_01J.../4K_MOGI-130.mp4
[r2] 37.3 MiB in 1 chunks of 64.0 MiB, 8 workers
[job job_01J...] source ready: 4K_MOGI-130.mp4 -> 4K_MOGI-130-clean.mp4, publishing to output/job_01J.../...
[T0V1] [cfg] engine   : propainter (flow propagation across frames)  (--quality video)
[T0V1] [pp] EXTRAPOLATED  1 hour of this footage -> ~1h23m
[T0V1] => all checks passed
[job job_01J...] wmrm run exited 0 after 181s
[job job_01J...] succeeded (outcome=ok) -> output/job_01J.../4K_MOGI-130-clean.mp4
```

The `[r2]` lines happen in the server process, so they appear directly. The `[T0V1]` lines
are the run's own output, read a line at a time from its pipe by a dedicated thread —
which is the part that matters, because a pipe nobody drains fills up and blocks the child.
Memory is one line regardless of how long the run is.

So keep the server's output somewhere if you want it after a restart. One file for the pod
beats one per job:

```bash
wmrm serve 2>&1 | tee -a /workspace/wmrm-serve.log
```

| | default | |
| --- | --- | --- |
| `WMRM_ECHO_RUN` | on | the echo above. `0` silences it |
| `WMRM_RUN_LOG` | **off** | a `run.log` per job in its work directory. `1` turns it on, and is what `GET /jobs/{id}/log` reads |

Per-job files are off because logs get read by going into the pod, where the server's
output is already in front of you — a second copy is two places to look and one of them
nobody opens.

**On a short clip the run looks idle, and it is not.** Progress is counted from finished
parts in `<output>.parts/`, and `ppPart` defaults to 3600 frames — so a one-minute clip is
a single part and there is nothing to count. It only becomes informative on long footage:
two hours at 30 fps is about 60 parts.

### What the pod tells you

Every job ends with a webhook to `{callbackBaseUrl}/api/pod/hooks`, signed with
a key derived from this pod's own `WMRM_POD_TOKEN`. Progress arrives the same way, and a **heartbeat
every 30 seconds regardless of engine** — only ProPainter has countable progress (its
parts directory), and without a heartbeat every other engine would look dead.

The outcome is not the exit code. Exit 1 already means six unrelated things here, so it
cannot say which failure happened; the answer travels in the report file (`--report`, and
`GET /jobs/{id}` returns it). The ones worth knowing apart:

| outcome | what to do |
| --- | --- |
| `ok` | nothing |
| `coverage_under` | the box is provably too small. Fix the box; do not retry as-is |
| `coverage_inconclusive` | the check could not tell. A person looks |
| `verify_failed` | the output did not pass its own acceptance checks |
| `interrupted` | stopped from outside — a pod restart, the OOM killer. **Run it again**; `--resume` picks up finished parts |
| `canceled` | a person stopped it. Terminal |
| `upload_failed` | the video is fine and only publishing it failed. Retry costs one upload, not the hours of GPU time |
| `oom` | the segment did not fit even after ProPainter halved it. A bigger card or a smaller `ppSegment` |
| `usage_error` | the caller is wrong. Retrying anywhere fails identically |

`interrupted` and `canceled` are deliberately separate. A process cannot tell who sent it
SIGTERM, so `wmrm run` reports `interrupted` either way and the pod — which knows whether
it was the one that asked — rewrites it. Collapsing them would mean a job somebody
cancelled gets retried, or a pod restart quietly loses nine hours of work.

### Telling a person (Mezon)

The webhook above is for the control plane. Set `WMRM_MEZON_WEBHOOK_URL` and the pod also
posts a line to a [Mezon channel webhook](https://mezon.ai/docs/vi/developer/webhooks/channel-webhook)
when a job ends:

```bash
export WMRM_MEZON_WEBHOOK_URL='https://webhook.mezon.ai/webhooks/<channelId>/<token>'
```

Set it in the pod's own environment, alongside the R2 credentials — `pod-entrypoint.sh`
does not supply a default, deliberately. See the note on the URL below.

```
✅ wmrm succeeded — ok
pod: pod-7f3a
job: 0f2c9b1e
output: clean/0f2c9b1e.mp4
```

**Terminal events only** — not heartbeats. Those arrive every 30 seconds for the life of
the job, and a channel that receives them is a channel everybody mutes.

It is a second destination for one event, not a second reporting path. The message is
unsigned, carries no dispatch token and no event id, and nothing reads it back; the report
the control plane acts on is still the signed one. So it is sent **after** that report, two
attempts and then silence, and every failure is swallowed — a job must not fail because a
chat channel is gone. When one does not land, the pod says so on its own console:

```
[job 0f2c9b1e] mezon: notification not delivered
```

**The URL is the entire credential.** There is no signature and nothing to verify against,
so anyone holding it can post to that channel — which is why it is not checked in as a
default, however convenient that would be, and why the startup banner prints
`mezon : channel 2081597…` and never the token: a pod's console is visible in the RunPod
dashboard.

### The secrets, and which side each one lives on

Three that matter, and one of them the project already used — plus an optional fourth that
is only held, never matched. Two values have to agree across two places, so those are the
only ones worth being careful with.

| secret | pod | web | what it proves |
| --- | :-: | :-: | --- |
| `WMRM_POD_TOKEN` | ✅ **one per pod** | stored in `pods.token` | **both directions.** web → pod as a bearer token; pod → web as the root of the key its reports are signed with |
| `CRON_SECRET` | — | ✅ **and on wmrm-cron** | that a sweep really came from the cron Worker. Not Access: the trigger reaches the app over a service binding, which never traverses the edge, so Access never sees it |
| `R2_*` (four) | ✅ | ✅ | reading the source and publishing the result |
| `WMRM_MEZON_WEBHOOK_URL` | optional | — | nothing — it *is* the credential. Held, not matched: leaving it unset only means no chat notification |

Generate it with hex rather than base64 — the value passes through a RunPod environment
variable and an HTML form, and base64's `+`, `/` and `=` each have a way of being mangled on
that trip:

```bash
openssl rand -hex 24          # WMRM_POD_TOKEN -- a different one per pod
openssl rand -hex 32          # CRON_SECRET -- the same value on wmrm-service and wmrm-cron
```

`/api/cron/tick` **refuses to run** when `CRON_SECRET` is unset — 503, not "allow". A sweep
that cancels jobs on pods is not something to leave reachable by anyone who guesses the
path, and a half-configured deploy should fail where someone can see it.

**`WMRM_POD_TOKEN` must be different on every pod.** Sharing one means a single compromised
pod controls all of them, and it would also let any pod sign reports for any other — the
same reasoning that decides how R2 credentials are handled.

**There is no separate webhook secret**, and that is deliberate. A pod signs its reports
with `SHA-256(WMRM_POD_TOKEN || "wmrm-webhook-v1")`; the web app looks up which pod owns the
job, reads that pod's token, and derives the same key. The value that already had to match is
the only one there is.

That removed two problems rather than one. A fleet-wide shared secret let any pod forge
another's reports. And "configure this identical value on the web app and on every pod" is
the quietest misconfiguration available — one character out and every report is refused
while the job runs perfectly and merely looks stalled.

**The tokens are stored in D1 in the clear**, which was a decision rather than an oversight.
Hashing them is not available: the control plane has to *send* the token to call the pod, and
a hash only checks one you were already given. Encrypting them was implemented and then
removed, because the key would have to live in the same Worker that reads the table — so
anyone who can read `pods` can read the key, and anyone who can read the Worker's secrets
can read the table. It bought a third secret to keep in step and no secrecy from anyone. What
does limit the damage is that each token is a random string, scoped to one pod, used for
nothing else, and revoked by restarting that pod with a new one.

There is no Cloudflare Access service token here. `/api/pod/*` is reached through an Access
**Bypass** policy and the HMAC is what authenticates the report — one mechanism for one hop,
rather than two that can disagree with each other.

### Registering it

The pod does not announce itself. Bring it up with a token you chose, then paste its URL
and that same token into the **Pods** page of the web app. Saving probes the machine
immediately and shows what it found — GPU, `archList`, free disk, `wmrm` version — so a
wrong URL or a mistyped token is visible then, rather than in a job that fails hours later.

A pod is saved even when the probe fails; its health records why. Refusing to save a
machine that is merely asleep would mean typing everything again once it wakes up.

### What it does not do

No queue. The pod takes one job at a time and answers **409** when it is busy — deciding
what runs next is the scheduler's job, and a queue on both sides is two queues to
disagree. No retries of its own, for the same reason. No cleanup of old work directories
beyond `DELETE /jobs/{id}`.

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

This is why `unblend` is the CPU answer and why an opaque mark is the harder case: its
transform is fitted once and constant, so it has no per-frame guess to vary and
**cannot** flicker (measured 0.98 correlation with true motion). The paragraph above
applies to `high`, `fast` and `draft`, not to `unblend` or `video`.

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
