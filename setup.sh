#!/usr/bin/env bash
#
# One command from a bare machine to a working `wmrm`.
#
#   ./setup.sh
#
# Safe to re-run: every step checks before it acts, so a half-finished setup can be
# resumed rather than started over.
#
# The only line that genuinely differs per machine is torch, and it is the one people
# get wrong -- a CPU-only wheel on a GPU box runs the default engine 400x slower, and a
# torchvision from a different index than torch fails with an error that names neither
# package. Both are decided here from what nvidia-smi reports.
#
# Overrides, for when the guess is wrong:
#
#   CUDA=cu121 ./setup.sh        # older driver (< 550)
#   CUDA=cpu   ./setup.sh        # force the CPU build
#   PYTHON=3.11 ./setup.sh       # different interpreter
#   SKIP_FFMPEG=1 ./setup.sh     # ffmpeg is handled elsewhere
#   SKIP_WEIGHTS=1 ./setup.sh    # do not pre-download the ~190 MB of model weights

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYTHON="${PYTHON:-3.12}"
VENV="${VENV:-$HERE/.venv}"

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mok\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m%s\033[0m\n' "$*"; }
die()  { printf '\n\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 1. ffmpeg -- not a Python dependency, and everything else is useless without it
# --------------------------------------------------------------------------- #
log "ffmpeg"
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  ok "$(ffmpeg -version | head -1 | cut -c1-60)"
elif [[ -n "${SKIP_FFMPEG:-}" ]]; then
  warn "missing, but SKIP_FFMPEG is set -- nothing will run until it is on PATH"
elif [[ $EUID -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
  # Root in a container is the normal case for this tool, so install rather than
  # printing a command for someone to copy.
  warn "not found -- installing"
  apt-get update -qq && apt-get install -y -qq ffmpeg
  ok "installed $(ffmpeg -version | head -1 | cut -c1-40)"
else
  die "ffmpeg and ffprobe are required. Install them, then re-run:
    sudo apt update && sudo apt install -y ffmpeg
  Or set SKIP_FFMPEG=1 if you handle it another way."
fi

# --------------------------------------------------------------------------- #
# 2. uv
# --------------------------------------------------------------------------- #
log "uv"
if command -v uv >/dev/null 2>&1; then
  ok "$(uv --version)"
else
  warn "not found -- installing"
  # The official installer rather than pip: it needs no Python to already exist, and
  # it can fetch interpreters, which is what makes step 3 work on a bare image.
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  elif command -v pip3 >/dev/null 2>&1; then
    pip3 install --quiet uv
  else
    die "need curl, wget or pip3 to install uv"
  fi
  # The installer drops it in one of these and does not touch this shell's PATH.
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [[ -x "$d/uv" ]] && export PATH="$d:$PATH"
  done
  command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH -- open a new shell and re-run"
  ok "$(uv --version)"
fi

# --------------------------------------------------------------------------- #
# 3. venv
# --------------------------------------------------------------------------- #
log "virtualenv ($VENV, python $PYTHON)"
if [[ -x "$VENV/bin/python" ]]; then
  ok "exists: $("$VENV/bin/python" --version)"
else
  uv venv "$VENV" --python "$PYTHON"
  ok "created: $("$VENV/bin/python" --version)"
fi
export VIRTUAL_ENV="$VENV"
PY="$VENV/bin/python"

# --------------------------------------------------------------------------- #
# 4. torch -- the machine-specific line
# --------------------------------------------------------------------------- #
log "torch"
if [[ -n "${CUDA:-}" ]]; then
  cuda="$CUDA"
  ok "index pinned by CUDA=$cuda"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  # Two independent constraints, and reading only one of them is how this went
  # wrong before:
  #
  #   driver  -- an index newer than the driver fails to initialise CUDA at all.
  #   card    -- an index older than the card's compute capability installs fine,
  #              reports the right GPU name and VRAM, loads every model, and then
  #              dies at the first kernel launch with "no kernel image is
  #              available for execution on the device".
  #
  # This used to check only the driver, so an RTX PRO 4000 Blackwell (sm_120) on a
  # new driver picked cu124, whose newest architecture is sm_90. Everything looked
  # correct right up to the point it wasn't.
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  major="${driver%%.*}"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  # compute_cap is "12.0" for Blackwell, "8.9" for Ada, "8.6" for Ampere. Older
  # nvidia-smi does not know the field, hence the fallback.
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"

  if (( major >= 550 )); then cuda=cu124; else cuda=cu121; fi
  cap_major="${cap%%.*}"
  if [[ -n "$cap" && "$cap_major" =~ ^[0-9]+$ ]] && (( cap_major >= 12 )); then
    # Blackwell and newer. cu124 tops out at sm_90, so it is not an option here
    # regardless of how new the driver is.
    cuda=cu128
    ok "$gpu_name (compute $cap), driver $driver -> $cuda (needs >= cu128 for sm_${cap/./})"
  elif [[ -z "$cap" ]]; then
    warn "$gpu_name, driver $driver -> $cuda (this nvidia-smi cannot report"
    warn "compute_cap, so the card's architecture was not checked -- the"
    warn "verification step below is what will catch a mismatch)"
  else
    ok "$gpu_name (compute $cap), driver $driver -> $cuda"
  fi
else
  cuda=cpu
  warn "no GPU detected -> CPU build. The default engine (ProPainter) needs a GPU and"
  warn "will refuse to run; use --quality unblend, which is built for the CPU."
fi

torch_index="https://download.pytorch.org/whl/$cuda"
have_torch="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
want_gpu=$([[ "$cuda" == cpu ]] && echo 0 || echo 1)
have_gpu=$("$PY" -c 'import torch; print(1 if torch.version.cuda else 0)' 2>/dev/null || echo 0)

if [[ -n "$have_torch" && "$want_gpu" == "$have_gpu" ]]; then
  ok "already installed: $have_torch"
else
  [[ -n "$have_torch" ]] && warn "replacing $have_torch: it is the wrong build for this machine"
  # torch and torchvision in one command, from one index. Installed separately or from
  # different indexes they resolve to a mismatched pair, and the symptom is
  # `operator torchvision::nms does not exist`, which mentions neither of them.
  #
  # PyPI has to stay in the search set, and this is not optional any more.
  # `--index-url` REPLACES PyPI rather than adding to it, and the pytorch index has since
  # pruned the old nvidia wheels: every torch it still serves for cu124 pins
  # `nvidia-cudnn-cu12==9.1.0.70` exactly, while the oldest cudnn left on that index is
  # 9.18.0.77. PyPI still has 9.1.0.70. With one index the resolution is genuinely
  # unsatisfiable, and uv reports it as a pile of unrelated hints about aarch64 wheels and
  # Python ABI tags that send you looking at your interpreter version instead.
  #
  # `unsafe-best-match` is required, not belt-and-braces: uv's default `first-index` stops
  # at the first index that has the package NAME, and cu124 does have nvidia-cudnn-cu12 --
  # just not the pinned version. So it would keep failing with PyPI merely listed.
  #
  # The cost of best-match is that a newer plain torch on PyPI can win over the +cuXXX
  # build from the chosen index. The verify step below prints what was actually installed,
  # which is where you check that.
  uv pip install \
    --index-url "$torch_index" \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    torch torchvision
  ok "installed from $torch_index (+ PyPI for the nvidia runtime wheels)"
fi

# --------------------------------------------------------------------------- #
# 5. everything else
# --------------------------------------------------------------------------- #
log "python dependencies"
# --no-deps: it pins pillow==9.5, which does not build on 3.12.
uv pip install --quiet --no-deps simple-lama-inpainting
uv pip install --quiet "pillow>=10" opencv-python-headless numpy
# ProPainter's own imports. It is a research repo with no packaging, so its
# requirements are ours to satisfy.
uv pip install --quiet av addict einops future scipy scikit-image imageio \
                       imageio-ffmpeg pyyaml requests timm matplotlib
uv pip install --quiet boto3
ok "dependencies installed"

log "wmrm"
uv pip install --quiet --no-deps -e .
ok "$("$VENV/bin/wmrm" --version 2>/dev/null || echo 'installed')"

# --------------------------------------------------------------------------- #
# 6. prove it, in the environment that will actually be used
# --------------------------------------------------------------------------- #
log "verify"
"$PY" - <<'EOF'
import sys, shutil
from pathlib import Path

bad = []

import torch
gpu = torch.cuda.is_available()
print(f"    torch {torch.__version__}  cuda build {torch.version.cuda}  "
      f"cuda available {gpu}")
if torch.version.cuda and not gpu:
    bad.append("torch has a CUDA build but cannot see a device -- check the driver")
if gpu:
    print(f"    device: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # The check that the index choice above is only a guess about. Everything
    # else here passes on a wheel with no kernels for this card: it reports the
    # right name, the right VRAM, and cuda available True. Only the architecture
    # list distinguishes a usable install from one that dies at the first kernel
    # launch, ~10s into a run, with a message naming neither the card nor the
    # wheel.
    major, minor = torch.cuda.get_device_capability(0)
    want, arches = f"sm_{major}{minor}", torch.cuda.get_arch_list()
    print(f"    arch: card needs {want}, torch has "
          f"{', '.join(arches) if arches else 'nothing'}")
    if arches and want not in arches:
        newest = max((a for a in arches if a.startswith("sm_")), default="none",
                     key=lambda a: int(a[3:]))
        bad.append(
            f"this torch has no kernels for this GPU: card is {want}, the wheel's "
            f"newest is {newest}. It will load models fine and then fail at the "
            f"first kernel launch. Re-run with a newer index, e.g. "
            f"CUDA=cu130 ./setup.sh")

# The mismatch that names neither package when it fails.
try:
    import torchvision
    torch.ops.torchvision.nms
    print(f"    torchvision {torchvision.__version__}  nms ok")
except Exception as exc:
    bad.append(f"torchvision is not usable ({exc}). It must come from the same index "
               f"as torch -- re-run with CUDA= set explicitly.")

sys.path.insert(0, "src")
from wmrm.video import find_repo, describe_device
try:
    repo = find_repo()
    print(f"    propainter: {repo}")
    if not (repo / "wmrm_worker.py").is_file():
        bad.append(f"{repo} has no wmrm_worker.py -- run scripts/vendor-propainter.sh")
except Exception as exc:
    bad.append(str(exc).splitlines()[0])
print(f"    engine device: {describe_device()}")

for tool in ("ffmpeg", "ffprobe"):
    if not shutil.which(tool):
        bad.append(f"{tool} not on PATH")

if bad:
    print()
    for b in bad:
        print(f"    PROBLEM: {b}")
    sys.exit(1)
EOF

# --------------------------------------------------------------------------- #
# 7. weights -- pulled now so the first real run is not also a 190 MB download
# --------------------------------------------------------------------------- #
if [[ -z "${SKIP_WEIGHTS:-}" ]]; then
  log "model weights (~190 MB, once)"
  # Loading the models is also the only end-to-end proof that torch, torchvision and
  # the vendored checkout work together. Done on the CPU deliberately: it exercises
  # the same code and does not need the card to be free.
  "$PY" - <<'EOF'
import sys
sys.path.insert(0, "src")
from wmrm.video import find_repo
sys.path.insert(0, str(find_repo()))
from wmrm_worker import ProPainterWorker, WorkerOpts
ProPainterWorker(device="cpu", opts=WorkerOpts(fp16=False))
print("    models load")
EOF
  ok "weights present"
else
  warn "SKIP_WEIGHTS set -- the first run will download ~190 MB"
fi

log "done"
cat <<EOF
    source $VENV/bin/activate
    wmrm run YOUR.mp4          # detects the watermark and removes it
    wmrm batch ./inbox         # a whole folder, models loaded once
EOF
