#!/usr/bin/env bash
# Re-copy the vendored ProPainter inference code from upstream.
#
# The vendored tree carries no .git of its own -- it is ordinary source in this
# repo. That is deliberate, and it is exactly why this script exists: without it
# there would be no record of where the code came from and no way to move to a
# newer upstream except by hand.
#
#   scripts/vendor-propainter.sh                  # re-vendor at the pinned commit
#   scripts/vendor-propainter.sh <ref>            # ... at another ref/tag/branch
#   scripts/vendor-propainter.sh <ref> --from DIR # ... from a checkout you have
#
# Afterwards: read the diff, run tests/test_propainter_parity.py, and update the
# commit hash in vendor/README.md. The parity test is not optional -- our worker
# duplicates upstream's inference body, so an upstream change to it is invisible
# until that test fails.
set -euo pipefail

PIN=e870e79321c31b733e2031af5aa2fb1fe3ac7eec
UPSTREAM=https://github.com/sczhou/ProPainter.git

# Only the inference import chain, reachable from inference_propainter.py.
# Everything else upstream ships is demo assets, sample clips or training code:
# 303 MB of checkout for 932 KB of code we run. vendor/README.md has the table.
KEEP=(model core RAFT utils configs inference_propainter.py LICENSE requirements.txt)

# Ours, not upstream's. A re-vendor must not touch these.
OURS=(wmrm_worker.py)

here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dst=$here/vendor/ProPainter

ref=${1:-$PIN}
from=""
if [[ ${2:-} == --from ]]; then
    from=${3:?--from needs a directory}
fi

tmp=$(mktemp -d -t vendor-pp-XXXXXX)
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

if [[ -n $from ]]; then
    src=$(cd "$from" && pwd)
    echo "source   : $src (existing checkout)"
    if [[ -d $src/.git ]]; then
        echo "         : at $(git -C "$src" rev-parse HEAD)"
        if [[ $(git -C "$src" rev-parse HEAD) != "$ref"* ]]; then
            echo "WARNING  : checkout is not at $ref -- vendoring what is on disk"
        fi
    fi
else
    echo "source   : $UPSTREAM at $ref"
    # Full clone, not --depth 1: a depth-1 clone cannot check out an arbitrary
    # commit, and pinning to a commit rather than a branch tip is the point.
    git clone --quiet "$UPSTREAM" "$tmp/ProPainter"
    git -C "$tmp/ProPainter" checkout --quiet "$ref"
    src=$tmp/ProPainter
fi

for f in "${KEEP[@]}"; do
    [[ -e $src/$f ]] || { echo "error: $src/$f is missing -- wrong ref?" >&2; exit 1; }
done

# Stage the new tree beside the old one and swap, so an interrupted run cannot
# leave a half-copied vendor directory that imports but misbehaves.
stage=$tmp/stage
mkdir -p "$stage"
for f in "${KEEP[@]}"; do
    cp -r "$src/$f" "$stage/"
done
find "$stage" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$stage" -name '*.py[co]' -delete
# No .git, by design: this is repo source now, not a nested checkout.
rm -rf "$stage/.git"

# Carry our own files and the downloaded weights across an upgrade.
for f in "${OURS[@]}"; do
    [[ -e $dst/$f ]] && cp -r "$dst/$f" "$stage/"
done
if [[ -d $dst/weights ]]; then
    mv "$dst/weights" "$stage/weights"
    echo "weights  : kept $(du -sh "$stage/weights" | cut -f1) already downloaded"
elif [[ -d $src/weights ]]; then
    cp -r "$src/weights" "$stage/weights"
    echo "weights  : copied $(du -sh "$stage/weights" | cut -f1) from the source checkout"
else
    echo "weights  : none present; they download on first run (~190 MB)"
fi

mkdir -p "$here/vendor"
rm -rf "$dst"
mv "$stage" "$dst"

code=$(du -sh --exclude=weights "$dst" | cut -f1)
echo "vendored : $dst ($code of code, $(find "$dst" -name '*.py' -not -path '*/weights/*' | wc -l) .py files)"
echo
echo "next:"
echo "  git -C $here status --short vendor/"
echo "  python tests/test_propainter_parity.py"
echo "  # then update the commit hash in vendor/README.md if the ref changed"
