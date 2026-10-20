#!/bin/bash
# Sync this experiment to Polaris. Safe to re-run -- rsync sends only what changed.
#
#   ./polaris/push.sh                 # sync
#   ./polaris/push.sh --dry-run       # show what would change, send nothing
#
# Override the destination with OPAL_REMOTE / OPAL_REMOTE_PATH.
#
# Explicit include-list, not an exclude-list. An experiment directory
# accumulates things that must never be pushed -- a venv, run outputs, pulled
# SafeTensors -- and with an exclude-list every new one is pushed by default
# until someone remembers to add it. The list below is the whole payload.
#
# NOT pushed, deliberately:
#   .venv/              built for this laptop; wrong arch, no CUDA. setup.sh
#                       builds the real one with uv on the node.
#   ../flowcept/        setup.sh clones both from GitHub on Polaris. Pushing
#   ../vllm-attn-connector/   them means ~112 MB of .git for no benefit.
#   results/            runs live on /eagle, and pulled runs come BACK from there.
#
# data/opal_chains.csv IS pushed (~1 MB): unlike spotterai's gated dataset it is
# the benchmark itself, it is small, and regenerating it on the node is not
# possible. It is also what makes the run reproducible -- the chains are the
# experiment.
set -euo pipefail

REMOTE="${OPAL_REMOTE:-agueroudji@polaris.alcf.anl.gov}"
REMOTE_PATH="${OPAL_REMOTE_PATH:-/lus/eagle/projects/radix-io/agueroudji/opal-attn}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PAYLOAD=(configs polaris scripts src data README.md Makefile requirements-local.txt .gitignore)

# The two upstream repos are cloned on Polaris rather than pushed, so anything
# uncommitted here would silently not make it. Warn -- do not block: most of the
# time this is exactly what you want.
for repo in flowcept vllm-attn-connector; do
    d="$(cd "$HERE/.." && pwd)/$repo"
    [[ -d "$d/.git" ]] || continue
    if [[ -n "$(git -C "$d" status --porcelain 2>/dev/null)" ]]; then
        echo "!! $repo has uncommitted changes, and is NOT pushed (setup.sh clones it)." >&2
        echo "   Commit and push it to GitHub first, or those edits will not be on Polaris." >&2
        git -C "$d" status --short | sed 's/^/     /' >&2
    fi
done

cd "$HERE"

# Stamp what is being sent. Running stale code on the remote is the single most
# confusing failure mode here -- it looks like a bug in the scripts rather than
# an old copy of them -- so setup.sh and smoke_test.sh print this stamp.
STAMP=".opal-push-stamp"
{
    echo "pushed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "pushed_from=$(hostname)"
    echo "payload_sha=$(find "${PAYLOAD[@]}" -type f \
        -not -path '*/__pycache__/*' -not -name '*.pyc' \
        -exec sha256sum {} + 2>/dev/null | sort | sha256sum | cut -c1-12)"
} > "$STAMP"

rsync -avzh --relative --mkpath "$@" \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    "${PAYLOAD[@]}" "$STAMP" "$REMOTE:$REMOTE_PATH/"
rc=$?

if [[ "$rc" == "0" && "$*" != *--dry-run* ]]; then
    echo
    echo "pushed: $(grep payload_sha "$STAMP" | cut -d= -f2)  ($(grep pushed_at "$STAMP" | cut -d= -f2))"
    echo "On Polaris, setup.sh and smoke_test.sh report this same stamp."
fi
exit "$rc"
