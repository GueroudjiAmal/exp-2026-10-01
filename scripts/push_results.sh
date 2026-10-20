#!/bin/bash
# Ship a finished run directory to a remote host. The mirror of polaris/push.sh,
# which sends source one way; this sends results the other.
#
#   ./scripts/push_results.sh results/12345
#   ./scripts/push_results.sh results/12345 --to user@host:/data/opal
#   ./scripts/push_results.sh results/12345 --no-tensors --verify
#   ./scripts/push_results.sh results/12345 --dry-run
#   ./scripts/push_results.sh results/12345 --verify-only   # check an old push
#
# Destination comes from --to, else $OPAL_RESULTS_REMOTE. It is an rsync target,
# so `user@host:/path` or a plain local path both work -- the latter is useful
# for an attached disk, and exercises the same code.
#
# Runs from anywhere: the laptop after pulling a run back, or straight off a
# Polaris login node, which is the faster route for a large run since it skips
# the round trip. There is no module or venv requirement beyond python3 for
# --verify.
#
# WHAT GETS SENT, and why it is an include-list. A run directory holds two very
# different things: a few hundred KB of JSONL that is useless to lose, and tens
# of MB of SafeTensors that can be large and are sometimes not wanted. Naming
# both explicitly means --no-tensors is a real choice rather than a guess about
# what some exclude pattern happens to catch.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_DIR=""
DEST="${OPAL_RESULTS_REMOTE:-}"
NO_TENSORS=0
VERIFY=0
VERIFY_ONLY=0
RSYNC_ARGS=()

usage() { sed -n '2,30p' "$0" | sed 's/^# \?//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --to)         DEST="$2"; shift 2 ;;
        --no-tensors) NO_TENSORS=1; shift ;;
        --verify)     VERIFY=1; shift ;;
        # Re-check an archive without resending it. Worth having separately:
        # with --verify the rsync runs first and would silently REPAIR a
        # corrupted remote file, so a combined run can never report a failure.
        --verify-only) VERIFY=1; VERIFY_ONLY=1; shift ;;
        --dry-run)    RSYNC_ARGS+=(--dry-run); shift ;;
        -h|--help)    usage 0 ;;
        -*)           RSYNC_ARGS+=("$1"); shift ;;
        *)            RUN_DIR="$1"; shift ;;
    esac
done

[[ -n "$RUN_DIR" ]] || { echo "!! no run directory given" >&2; usage 2; }
[[ -d "$RUN_DIR" ]] || { echo "!! not a directory: $RUN_DIR" >&2; exit 2; }
if [[ -z "$DEST" ]]; then
    echo "!! no destination. Pass --to user@host:/path, or set OPAL_RESULTS_REMOTE." >&2
    exit 2
fi

RUN_DIR="$(cd "$RUN_DIR" && pwd)"
NAME="$(basename "$RUN_DIR")"

# Refuse to ship something that is not a run. Pushing an empty or half-written
# directory and discovering it a week later is the failure this prevents.
if [[ ! -f "$RUN_DIR/turns.jsonl" ]]; then
    echo "!! $RUN_DIR has no turns.jsonl -- not a finished run directory." >&2
    echo "   (A run still in progress has not written it yet.)" >&2
    exit 2
fi
shopt -s nullglob
BUFFERS=("$RUN_DIR"/flowcept_buffer*.jsonl)
TENSORS=("$RUN_DIR"/*/*.safetensors)
shopt -u nullglob
if [[ ${#BUFFERS[@]} == 0 ]]; then
    echo "!! no flowcept_buffer*.jsonl in $RUN_DIR -- the provenance is missing," >&2
    echo "   so what is left cannot be verified or queried. Refusing." >&2
    exit 2
fi

echo "run        $RUN_DIR"
echo "records    $(wc -l < "$RUN_DIR/turns.jsonl") turns, ${#BUFFERS[@]} buffer file(s)"
echo "tensors    ${#TENSORS[@]} files, $(du -sh --exclude='*.jsonl' --exclude='*.json' "$RUN_DIR" 2>/dev/null | cut -f1)"
echo "dest       ${DEST%/}/${NAME}"

# --- the transfer -------------------------------------------------------------
# -a archive, -z compress (JSONL compresses well; SafeTensors barely, but the
# cost is small), -h human sizes, --partial so an interrupted large transfer
# resumes instead of restarting, --info=progress2 for one aggregate progress
# line rather than one per file.
#
# Deliberately NOT --delete: this is additive. A destination holding other runs
# must not be pruned because one run was pushed into it.
INCLUDES=(--include='*/' --include='*.jsonl' --include='*.json')
if [[ "$NO_TENSORS" == "0" ]]; then
    INCLUDES+=(--include='*.safetensors')
else
    echo "note       --no-tensors: sending provenance only."
    echo "           analyze.py --no-tensors reads that; the tensor-level checks"
    echo "           and the attention report need the files."
fi
INCLUDES+=(--exclude='*')

if [[ "$VERIFY_ONLY" == "1" ]]; then
    echo "mode       --verify-only: not sending anything"
else
    rsync -azh --partial --info=progress2 --mkpath \
        "${INCLUDES[@]}" "${RSYNC_ARGS[@]}" \
        "$RUN_DIR" "${DEST%/}/"
fi

if [[ " ${RSYNC_ARGS[*]} " == *" --dry-run "* ]]; then
    echo; echo "dry run: nothing sent."
    exit 0
fi

# --- verification -------------------------------------------------------------
# Against the sha256 the CONNECTOR recorded when it wrote each file, not a
# manifest generated here. Hashing the local copy and comparing to the remote
# copy only proves the copy matches the copy; comparing both to what provenance
# says proves the bytes are the ones whose statistics were captured.
if [[ "$VERIFY" == "1" ]]; then
    echo
    if [[ "$NO_TENSORS" == "1" && "$VERIFY_ONLY" == "0" ]]; then
        echo "verify     skipped: --no-tensors sent no tensor files."
        exit 0
    fi
    MANIFEST="$(mktemp)"
    trap 'rm -f "$MANIFEST"' EXIT
    PYBIN="${HERE}/.venv/bin/python"; [[ -x "$PYBIN" ]] || PYBIN=python3
    # One implementation of "which file should hash to what", in
    # report.checksum_manifest -- this used to be a second copy inline here,
    # and two copies of a verification rule is one too many.
    PYTHONPATH="${HERE}/src" "$PYBIN" - "$RUN_DIR" > "$MANIFEST" <<'PYMANIFEST'
import sys
from pathlib import Path
from opal_attn import report

run = Path(sys.argv[1])
buf, _notes = report.find_run_buffer(run)
if buf is None:
    sys.exit(f"no provenance in {run}")
manifest = report.checksum_manifest(report.load_buffer(buf))
for rel, sha in sorted(manifest.items()):
    print(f"{sha}  {rel}")
print(f"manifest: {len(manifest)} file(s)", file=sys.stderr)
PYMANIFEST
    COUNT="$(wc -l < "$MANIFEST")"
    if [[ "$COUNT" == "0" ]]; then
        echo "verify     no sha256 in the provenance -- the connector ran with" >&2
        echo "           checksum:false, so there is nothing to check against." >&2
        exit 0
    fi
    # `sha256sum -c` exits non-zero and names each bad file on stderr. Let that
    # propagate: a verification failure must fail the script, or the exit code
    # of an archival step means nothing in a pipeline.
    if [[ "$DEST" != *:* ]]; then
        ( cd "${DEST%/}/${NAME}" && sha256sum -c --quiet - ) < "$MANIFEST" \
            && echo "verify     OK: $COUNT file(s) match the recorded sha256"
    else
        HOST="${DEST%%:*}"; RPATH="${DEST#*:}"
        ssh "$HOST" "cd '${RPATH%/}/${NAME}' && sha256sum -c --quiet -" < "$MANIFEST" \
            && echo "verify     OK: $COUNT file(s) match the recorded sha256"
    fi
fi
