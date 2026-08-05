#!/usr/bin/env bash
#
# One-shot pipeline. Results always land in outbox/.
#
#   ./run.sh                        every video in inbox/
#   ./run.sh clip.mp4               one file
#   ./run.sh a.mp4 b.mp4 c.mov      several files
#   ./run.sh /data/videos           another folder
#   ./run.sh /data/videos x.mp4     folders and files mixed
#
# How it picks the watermark box:
#   - PRESET exists  -> use it. Deterministic, no guessing.
#   - otherwise      -> detect once on the first video, SAVE it as PRESET, then
#                       use that one box for everything.
#
# So the first run calibrates itself and every run after it is repeatable. Check
# the preview PNG it points you at -- detection is a guess, not a guarantee.

set -euo pipefail

# ----------------------------------------------------------------------------- #
# config -- override with environment variables
# ----------------------------------------------------------------------------- #
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INBOX="${INBOX:-$HERE/inbox}"     # used only when no arguments are given
OUTBOX="${OUTBOX:-$HERE/outbox}"
PRESET="${PRESET:-$HERE/preset.json}"
CORNER="${CORNER:-tr}"            # tr tl br bl -- where the watermark sits
QUALITY="${QUALITY:-}"            # empty = best (LaMa). "fast" for CPU-bound boxes
EXTRA="${EXTRA:-}"                # anything else, e.g. EXTRA="--device cuda"
FORCE="${FORCE:-}"                # set to 1 to redo files already in outbox

# ----------------------------------------------------------------------------- #

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not on PATH (apt install ffmpeg)"

# Prefer the project venv so the script works from a bare shell too.
if [[ -x "$HERE/.venv/bin/wmrm" ]]; then
  WMRM="$HERE/.venv/bin/wmrm"
elif command -v wmrm >/dev/null 2>&1; then
  WMRM="$(command -v wmrm)"
else
  die "wmrm not found. Install it: see README.md"
fi

is_video() {
  case "${1,,}" in
    *.mp4|*.mov|*.mkv|*.webm|*.avi|*.m4v) return 0 ;;
    *) return 1 ;;
  esac
}

collect_dir() {   # append every video in a directory, skipping our own outputs
  local d="$1" f
  shopt -s nullglob nocaseglob
  for f in "$d"/*.mp4 "$d"/*.mov "$d"/*.mkv "$d"/*.webm "$d"/*.avi "$d"/*.m4v; do
    [[ "$f" == *-clean.* ]] || videos+=("$f")
  done
  shopt -u nullglob nocaseglob
}

videos=()
if (( $# == 0 )); then
  mkdir -p "$INBOX"
  collect_dir "$INBOX"
  (( ${#videos[@]} == 0 )) && { log "nothing to do"; echo "Put videos in: $INBOX"; exit 0; }
else
  for arg in "$@"; do
    if [[ -d "$arg" ]]; then
      collect_dir "$arg"
    elif [[ -f "$arg" ]]; then
      # An explicit file is processed even if it is named *-clean.*: you asked
      # for it by name, so second-guessing you would be wrong.
      is_video "$arg" || die "not a video: $arg"
      videos+=("$arg")
    else
      die "no such file or directory: $arg"
    fi
  done
  (( ${#videos[@]} == 0 )) && die "no videos found in the paths you gave"
fi

mkdir -p "$OUTBOX"
log "${#videos[@]} video(s) to consider"

args=()
[[ -n "$QUALITY" ]] && args+=(--quality "$QUALITY")
# shellcheck disable=SC2206  # deliberate word-splitting of user-supplied flags
[[ -n "$EXTRA" ]] && args+=($EXTRA)

# --- one box for the whole run ------------------------------------------------- #
# Detect once, save it, then process everything from the saved preset. Detecting
# per file is what makes unattended runs dangerous: each file gets a different
# box, so one bad guess is invisible among the good ones.
if [[ -f "$PRESET" ]]; then
  log "using saved box from $(basename "$PRESET")"
else
  log "no preset yet -- detecting on $(basename "${videos[0]}")"
  "$WMRM" detect "${videos[0]}" --corner "$CORNER" --preset "$PRESET"
fi

# --- process ------------------------------------------------------------------ #
processed=0 skipped=0 failed=0 failed_names=()
for src in "${videos[@]}"; do
  base="$(basename "$src")"
  out="$OUTBOX/${base%.*}-clean.${base##*.}"

  if [[ -f "$out" && -z "$FORCE" ]]; then
    echo "skip (exists): $(basename "$out")"
    skipped=$((skipped + 1))
    continue
  fi

  log "$base"
  if "$WMRM" run "$src" -o "$out" --preset "$PRESET" "${args[@]}"; then
    processed=$((processed + 1))
  else
    warn "FAILED: $base"
    failed=$((failed + 1)); failed_names+=("$base")
  fi
done

log "done: $processed processed, $skipped skipped, $failed failed"
echo "Results: $OUTBOX"
(( failed )) && warn "failed: ${failed_names[*]}"

preview="${videos[0]%.*}-preview-zoom.png"
if [[ -f "$preview" ]]; then
  echo
  echo "Confirm the red box covered the whole watermark:"
  echo "  $preview"
  echo
  echo "If it was wrong: rm $PRESET, then measure and save the real box:"
  echo "  $WMRM grid ${videos[0]} --corner $CORNER"
  echo "  $WMRM detect ${videos[0]} --box X,Y,W,H --preset $PRESET"
  echo "and run this again."
fi

(( failed )) && exit 1
exit 0
