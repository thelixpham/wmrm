#!/usr/bin/env bash
#
# Bring a RunPod Pod up as a wmrm worker, then serve.
#
#   bash deploy/pod-entrypoint.sh
#
# Idempotent: safe to run on every boot. That is not a nicety -- a Pod's container
# filesystem is discarded on stop/restart and only the volume survives, so this script
# runs again every time the machine comes back and must not undo or duplicate its own
# work.
#
# THE RULE THIS SCRIPT EXISTS TO ENFORCE: everything that must outlive a restart lives
# under /workspace. RunPod's own docs are explicit that editing or restarting a Pod
# "erases all data not stored in /workspace or a network volume", and that includes
# site-packages. A venv installed into the system Python disappears; so do the LaMa
# weights (196 MB) and the ProPainter weights (~190 MB), which then re-download on the
# first job after every restart.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${WMRM_REPO:-$WORKSPACE/wmrm}"
VENV="${WMRM_VENV:-$WORKSPACE/.venv}"

# Exported so `wmrm serve` and every child `wmrm run` agree on where things are. The
# work and state directories are namespaced by pod id inside the app, which is what
# keeps two pods off one path if they ever share a network volume.
export WMRM_POD_ID="${WMRM_POD_ID:-${RUNPOD_POD_ID:-local}}"
export WMRM_WORK_DIR="${WMRM_WORK_DIR:-$WORKSPACE/wmrm-work}"
export WMRM_STATE="${WMRM_STATE:-$WORKSPACE/wmrm-state}"
export TORCH_HOME="${TORCH_HOME:-$WORKSPACE/.cache/torch}"
export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
export PYTHONUNBUFFERED=1

PORT="${WMRM_PORT:-8000}"

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------- #
# preconditions
# ---------------------------------------------------------------------------- #

[[ -d "$WORKSPACE" ]] || die "$WORKSPACE does not exist. On RunPod this is the volume \
mount; without it nothing installed here survives a restart, so refusing rather than \
appearing to work."

[[ -d "$REPO" ]] || die "no wmrm checkout at $REPO (override with WMRM_REPO)"

# ---------------------------------------------------------------------------- #
# token -- generated once, then reused for the life of the volume
# ---------------------------------------------------------------------------- #
#
# The token cannot be regenerated on each boot, and this is the part that is easy to get
# wrong: the key a pod signs its reports with is DERIVED FROM THIS TOKEN, and the control
# plane holds the copy that was pasted into /pods. Mint a fresh one on restart and both
# directions break at once -- dispatch is refused by the pod, and any report it does send
# is refused by the app -- while /health still answers 200 and the pod looks fine.
#
# So it lives on the volume, which is the only thing here that survives a restart, and is
# namespaced by pod id for the same reason the work and state directories are: two pods
# sharing a network volume must not share an identity.
TOKEN_FILE="${WMRM_TOKEN_FILE:-$WMRM_STATE/$WMRM_POD_ID/pod-token}"

if [[ -n "${WMRM_POD_TOKEN:-}" ]]; then
  # An explicit value wins, always. This is the path for a token the control plane issued
  # first, and it is not written to the file -- what the operator passes for one boot
  # should not silently become the default for every boot after it.
  printf 'token   : from WMRM_POD_TOKEN (env)\n'
elif [[ -s "$TOKEN_FILE" ]]; then
  WMRM_POD_TOKEN="$(tr -d '\n' < "$TOKEN_FILE")"
  export WMRM_POD_TOKEN
  printf 'token   : reused from %s\n' "$TOKEN_FILE"
else
  mkdir -p "$(dirname "$TOKEN_FILE")"
  # umask in a subshell so the file is 0600 from the moment it exists rather than being
  # created readable and tightened afterwards.
  ( umask 077; head -c 24 /dev/urandom | base64 | tr -d '\n' > "$TOKEN_FILE" )
  WMRM_POD_TOKEN="$(cat "$TOKEN_FILE")"
  export WMRM_POD_TOKEN

  # Printed once, on the boot that created it, because there is no other way to learn it
  # and it has to be pasted into the control plane by hand. Later boots print the path
  # instead -- a pod's console output is visible in the RunPod dashboard, so repeating a
  # secret there every restart is a cost with no reader.
  printf '\n\033[1m==> new pod token minted\033[0m\n'
  printf '    %s\n\n' "$WMRM_POD_TOKEN"
  printf '    Paste it into the control plane at /pods, together with this endpoint.\n'
  printf '    Stored at %s -- read it again with:\n' "$TOKEN_FILE"
  printf '      cat %s\n' "$TOKEN_FILE"
fi

# This pod fetches its own source and publishes its own output, so it needs R2 access.
# Without it only kind='local' and kind='url' jobs can run, and /health says so.
missing_r2=()
for v in R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET; do
  [[ -n "${!v:-}" ]] || missing_r2+=("$v")
done
if [[ -z "${R2_ACCOUNT_ID:-}${R2_ENDPOINT:-}" ]]; then
  missing_r2+=("R2_ACCOUNT_ID (or R2_ENDPOINT)")
fi
if (( ${#missing_r2[@]} )); then
  warn "R2 is not configured -- missing: ${missing_r2[*]}
       Jobs with input.kind='r2' or output.kind='r2' will be refused with 400."
fi

log "pod $WMRM_POD_ID"
mkdir -p "$WMRM_WORK_DIR" "$WMRM_STATE" "$TORCH_HOME"

# ---------------------------------------------------------------------------- #
# ffmpeg
# ---------------------------------------------------------------------------- #

if ! command -v ffmpeg >/dev/null 2>&1; then
  log "installing ffmpeg"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq ffmpeg
  else
    die "ffmpeg is missing and this image has no apt-get to install it with"
  fi
fi
printf 'ffmpeg  : %s\n' "$(command -v ffmpeg)"

# ---------------------------------------------------------------------------- #
# venv on the volume
# ---------------------------------------------------------------------------- #

if [[ ! -x "$VENV/bin/python" ]]; then
  log "creating venv at $VENV"
  # uv when available (it is much faster and is what the README uses), else stdlib venv.
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV" --python 3.12
  else
    python3 -m venv "$VENV"
  fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
printf 'python  : %s\n' "$(python -V 2>&1)"

pip_install() {
  if command -v uv >/dev/null 2>&1; then
    VIRTUAL_ENV="$VENV" uv pip install "$@"
  else
    python -m pip install -q "$@"
  fi
}

# torch is deliberately NOT installed here. The right wheel depends on the card, and
# guessing produces the worst possible failure: a cu124 wheel on a Blackwell card
# installs cleanly, reports `cuda (...)`, loads the models, then dies at the first
# kernel launch. Install it once, for this machine, per the README, and let /health's
# archList report what you actually got.
if ! python -c "import torch" >/dev/null 2>&1; then
  warn "torch is not installed in $VENV.
       CPU-only engines (unblend/fast/draft) will still work; video and high will not.
       Install the wheel that matches this card -- see wmrm/README.md 'Install'."
fi

log "installing wmrm + serve extra"
pip_install -e "$REPO[serve]" 2>&1 | tail -3 || die "install failed"

# ---------------------------------------------------------------------------- #
# report what this machine turned out to be
# ---------------------------------------------------------------------------- #

log "machine"
python - <<'PY'
import json
from wmrm.server.probe import free_gb, probe_machine
import os
m = probe_machine()
g = m["gpu"]
print(f"ffmpeg  : {m['ffmpeg']}   nvdec: {m['nvdec']}")
print(f"torch   : {g['torch']}")
print(f"cuda    : {g['cuda']}  {g['name'] or ''}")
print(f"archList: {g['archList']}")
if g["error"]:
    print(f"gpu note: {g['error']}")
if g["cuda"] and g["archList"] and not any(a >= "sm_120" for a in g["archList"]):
    # Only a note: sm_120 matters on Blackwell cards, and this cannot tell which card
    # will be attached on the next boot.
    print("note    : no sm_120 in archList -- on an RTX 50-series / RTX PRO Blackwell "
          "card this wheel will die at the first kernel launch. Use a cu128+ index.")
print(f"free    : {free_gb(os.environ['WMRM_WORK_DIR']):.1f} GiB in "
      f"{os.environ['WMRM_WORK_DIR']}")
PY

# ---------------------------------------------------------------------------- #
# serve
# ---------------------------------------------------------------------------- #

log "serving on 0.0.0.0:$PORT"
# 0.0.0.0 because the RunPod proxy cannot reach a server bound to localhost, and one
# worker because job state is per-process and on disk.
exec wmrm serve --host 0.0.0.0 --port "$PORT"
