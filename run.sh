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
#   - PRESET exists  -> use it for everything. Deterministic, no guessing.
#   - DETECT=each    -> detect per video (the default). Right when the batch is
#                       mixed: a box measured on one source is meaningless for a
#                       different crop, a different logo or a different placement,
#                       and normalized coordinates only rescue a resolution change.
#   - DETECT=once    -> detect on the first video, save it as PRESET, reuse it.
#                       Right when every file is the same watermark from the same
#                       pipeline: one box, one preview, one decision.
#
# Detecting per video buys a better fit and costs the single point where a human
# confirmed the box. COVERAGE=1 (also the default) buys that back: every detected
# box is checked against the frame it came from before a pixel is touched, and a box
# that is provably too small stops that file instead of quietly shipping a video
# with watermark fringe left in it. Verdicts are summarised at the end, so one bad
# guess among fifty good ones is loud rather than invisible.
#
# Detection is still a guess with measured failure modes -- see the README table.
# Look at the preview PNGs this prints at the end.

set -euo pipefail

# ----------------------------------------------------------------------------- #
# config -- override with environment variables
# ----------------------------------------------------------------------------- #
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INBOX="${INBOX:-$HERE/inbox}"     # used only when no arguments are given
OUTBOX="${OUTBOX:-$HERE/outbox}"
# Was PRESET named on the command line, or is this just the default path? The
# difference decides whether an existing preset.json is an instruction or a leftover.
if [[ -n "${PRESET:-}" ]]; then PRESET_GIVEN=1; else PRESET_GIVEN=0; fi
PRESET="${PRESET:-$HERE/preset.json}"
CORNER="${CORNER:-tr}"            # tr tl br bl -- where the watermark sits
DETECT="${DETECT:-each}"          # each = detect per video, once = one box for all
COVERAGE="${COVERAGE:-1}"         # 1 = gate every detected box on `wmrm coverage`
CLEAN="${CLEAN:-1}"               # 1 = delete our own leftovers for files that passed
QUALITY="${QUALITY:-}"            # empty = the CLI default (un-blend). Override with
                                  # "high" for an opaque mark, "fast" if ffmpeg-only
EXTRA="${EXTRA:-}"                # anything else, e.g. EXTRA="--device cuda"
FORCE="${FORCE:-}"                # set to 1 to redo files already in outbox

# ----------------------------------------------------------------------------- #

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not on PATH (apt install ffmpeg)"

# Prefer the project venv so the script works from a bare shell too.
#
# Third branch matters more than it looks: the venv is often created somewhere
# other than $HERE/.venv (activated from the parent directory, for instance), and
# some install paths leave the package importable without putting the console
# script on PATH. The module entry point is equivalent, so use it instead of
# failing on a technicality.
if [[ -x "$HERE/.venv/bin/wmrm" ]]; then
  WMRM=("$HERE/.venv/bin/wmrm")
elif command -v wmrm >/dev/null 2>&1; then
  WMRM=("$(command -v wmrm)")
elif python -c "import wmrm" >/dev/null 2>&1; then
  WMRM=(python -m wmrm.cli)
else
  # Say what was tried, not just that it failed -- "not found" alone sends people
  # to reinstall something that is already installed in the wrong place.
  die "wmrm not found. Looked for:
    1. $HERE/.venv/bin/wmrm            $([[ -e "$HERE/.venv" ]] && echo "(.venv exists, but no wmrm in it)" || echo "(no .venv here)")
    2. 'wmrm' on PATH                  (VIRTUAL_ENV=${VIRTUAL_ENV:-unset})
    3. 'import wmrm' in $(command -v python || echo python)

  The package itself is probably not installed -- an active venv only gives you
  the dependencies. From $HERE run:

    pip install -e .        # or: uv pip install -e .

  Then check it with:  wmrm --help"
fi
WMRM_SHOW="${WMRM[*]}"

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

# --- how the box gets decided -------------------------------------------------- #
# A preset you *named* always wins: it is the only box a human deliberately measured
# and froze, and guessing over the top of it would throw away the one thing here that
# is not a guess.
#
# A preset that merely happens to exist at the default path does not win, and that
# asymmetry is the point. It used to: a leftover preset.json from an earlier
# DETECT=once run would silently stop all detection, so the tool quietly did
# something other than what the default asks for. Now it says so and carries on.
[[ "$DETECT" == "each" || "$DETECT" == "once" ]] \
  || die "DETECT must be 'each' or 'once', got '$DETECT'"

shared_preset=""
if [[ "$PRESET_GIVEN" == "1" ]]; then
  # Naming a preset path is a request for one shared, saved box -- whether or not the
  # file exists yet. Existing: use it. Missing: calibrate once and write it there,
  # which is the only reading of `PRESET=x.json` that makes sense on a first run.
  if [[ -f "$PRESET" ]]; then
    log "using saved box from $PRESET"
  else
    log "no preset yet -- detecting on $(basename "${videos[0]}"), saving to $PRESET"
    "${WMRM[@]}" detect "${videos[0]}" --corner "$CORNER" --preset "$PRESET"
  fi
  shared_preset="$PRESET"
elif [[ "$DETECT" == "once" ]]; then
  if [[ -f "$PRESET" ]]; then
    log "using saved box from $(basename "$PRESET") for all ${#videos[@]} file(s)"
  else
    log "no preset yet -- DETECT=once, detecting on $(basename "${videos[0]}")"
    "${WMRM[@]}" detect "${videos[0]}" --corner "$CORNER" --preset "$PRESET"
  fi
  shared_preset="$PRESET"
else
  if [[ -f "$PRESET" ]]; then
    warn "note: $(basename "$PRESET") exists but is being ignored -- the default is"
    warn "      to detect each video on its own. DETECT=once uses it instead."
  fi
  log "detecting each video on its own"
fi

# --- cleanup ------------------------------------------------------------------- #
# Delete what this script generated, and only for files nothing needs re-checking on.
# Never the input, never the output, never a file we did not create: preview PNGs sit
# next to someone's source footage, so the pattern has to be exact rather than a glob
# over that directory.
#
# The condition is the point. A file whose box was confirmed `covered` has nothing
# left to look at, so its preview is 465 KB of noise beside the original. A file that
# came back INCONCLUSIVE has exactly one thing that can still settle it -- the preview
# -- so deleting that would be deleting the evidence.
cleaned_bytes=0
clean_up() {   # <src> <preset> <base>
  [[ "$CLEAN" == "1" ]] || return 0
  local src="$1" preset="$2" base="$3" f
  # Keep everything for the files flagged for a human.
  for flagged in ${inconclusive[@]+"${inconclusive[@]}"}; do
    [[ "$flagged" == "$base" ]] && return 0
  done

  local victims=("${src%.*}-preview.png" "${src%.*}-preview-zoom.png")
  # Per-file presets only. A shared preset is either yours or this run's calibration
  # for every remaining file -- deleting it mid-run would break the files after this
  # one, and deleting it at the end would throw away the repeatability it exists for.
  [[ -z "$shared_preset" && "$preset" == "$presetdir/"* ]] && victims+=("$preset")

  for f in "${victims[@]}"; do
    if [[ -f "$f" ]]; then
      cleaned_bytes=$(( cleaned_bytes + $(stat -c %s "$f" 2>/dev/null || echo 0) ))
      rm -f -- "$f"
    fi
  done
}

# Per-file presets are kept for the files that need a second look. They are the record
# of what box each output was actually made with, which is the first thing anyone asks
# when one output looks wrong.
#
# Created on first use, not up front: when one shared box is in play this directory
# would stay empty, and an empty dot-directory appearing in outbox/ is a surprise for
# anything that lists that folder expecting only results.
presetdir="$OUTBOX/.presets"

# --- process ------------------------------------------------------------------ #
processed=0 skipped=0 failed=0 failed_names=()
undercovered=() inconclusive=() previews=()

for src in "${videos[@]}"; do
  base="$(basename "$src")"
  out="$OUTBOX/${base%.*}-clean.${base##*.}"

  if [[ -f "$out" && -z "$FORCE" ]]; then
    echo "skip (exists): $(basename "$out")"
    skipped=$((skipped + 1))
    continue
  fi

  log "$base"

  # 1. the box for this file
  if [[ -n "$shared_preset" ]]; then
    preset="$shared_preset"
  else
    mkdir -p "$presetdir"
    preset="$presetdir/${base%.*}.json"
    if ! "${WMRM[@]}" detect "$src" --corner "$CORNER" --preset "$preset"; then
      warn "FAILED (detect found nothing): $base"
      failed=$((failed + 1)); failed_names+=("$base [detect]")
      continue
    fi
    p="${src%.*}-preview-zoom.png"
    [[ -f "$p" ]] && previews+=("$p")
  fi

  # 2. gate it. Exit codes from `wmrm coverage`: 0 covered, 1 under-covered,
  #    2 inconclusive. Only 1 is a provable defect, so only 1 stops the file --
  #    inconclusive means the background is static and no statistic can answer,
  #    which is a reason to look at the preview, not a reason to refuse the work.
  if [[ "$COVERAGE" == "1" ]]; then
    set +e
    "${WMRM[@]}" coverage "$src" --preset "$preset"
    cov=$?
    set -e
    case "$cov" in
      0) ;;
      1) warn "SKIPPED: $base -- box is too small, watermark extends outside it."
         warn "         Re-run with a bigger box, or COVERAGE=0 to process anyway."
         undercovered+=("$base")
         failed=$((failed + 1)); failed_names+=("$base [under-covered]")
         continue ;;
      2) warn "$base -- coverage INCONCLUSIVE (static background). Processing; "
         warn "         confirm this one by eye."
         inconclusive+=("$base") ;;
      *) warn "$base -- coverage check itself failed (exit $cov). Processing; "
         warn "         confirm this one by eye."
         inconclusive+=("$base") ;;
    esac
  fi

  # 3. process
  if "${WMRM[@]}" run "$src" -o "$out" --preset "$preset" "${args[@]}"; then
    processed=$((processed + 1))
    clean_up "$src" "$preset" "$base"
  else
    warn "FAILED: $base"
    failed=$((failed + 1)); failed_names+=("$base")
  fi
done

log "done: $processed processed, $skipped skipped, $failed failed"
echo "Results: $OUTBOX"
(( failed )) && warn "failed: ${failed_names[*]}"

# The whole point of gating per file is that a bad box is loud. Repeat the verdicts
# here: nobody scrolls back through fifty files of log.
if (( ${#undercovered[@]} )); then
  echo
  warn "NOT PROCESSED -- detected box too small (${#undercovered[@]} file(s)):"
  for f in "${undercovered[@]}"; do warn "  $f"; done
  warn "For each, measure the box by hand and freeze it:"
  warn "  $WMRM_SHOW grid <file> --corner $CORNER"
  warn "  $WMRM_SHOW coverage <file> --box X,Y,W,H"
  warn "  $WMRM_SHOW detect <file> --box X,Y,W,H --preset $PRESET"
fi
if (( ${#inconclusive[@]} )); then
  echo
  warn "PROCESSED BUT UNVERIFIED -- coverage could not answer (${#inconclusive[@]} file(s)):"
  for f in "${inconclusive[@]}"; do warn "  $f"; done
  warn "The background is static there, so no statistic separates mark from wall."
  warn "Check these by eye before shipping them."
fi

if [[ -n "$shared_preset" ]]; then
  preview="${videos[0]%.*}-preview-zoom.png"
  if [[ -f "$preview" ]]; then
    echo
    echo "One box was used for everything. Confirm it covered the whole watermark:"
    echo "  $preview"
    echo
    echo "If it was wrong: rm $PRESET, then measure and save the real box:"
    echo "  $WMRM_SHOW grid ${videos[0]} --corner $CORNER"
    echo "  $WMRM_SHOW detect ${videos[0]} --box X,Y,W,H --preset $PRESET"
    echo "and run this again."
  fi
else
  # Only the previews still on disk. CLEAN deletes the ones for files that passed the
  # coverage gate, so what is left here is exactly the set worth opening -- pointing
  # at a path we just removed would be worse than saying nothing.
  kept=()
  for p in ${previews[@]+"${previews[@]}"}; do
    [[ -f "$p" ]] && kept+=("$p")
  done
  if (( ${#kept[@]} )); then
    echo
    echo "Previews kept for the files that still need an eye (${#kept[@]}):"
    for p in "${kept[@]:0:10}"; do echo "  $p"; done
    (( ${#kept[@]} > 10 )) && echo "  ... and $(( ${#kept[@]} - 10 )) more"
    echo "Boxes used: $presetdir"
  fi
fi

if (( cleaned_bytes )); then
  echo
  printf 'cleaned up %s of previews and per-file presets for the files that passed.\n' \
    "$(numfmt --to=iec --suffix=B "$cleaned_bytes" 2>/dev/null || echo "$cleaned_bytes bytes")"
  echo "CLEAN=0 keeps them."
fi

(( failed )) && exit 1
exit 0
