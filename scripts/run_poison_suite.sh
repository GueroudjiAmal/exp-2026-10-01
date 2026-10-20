#!/bin/bash
# The poisoning suite: one clean twin plus every arm, run STRICTLY IN SEQUENCE
# on one GPU, all with provenance and attention capture.
#
#   ./scripts/run_poison_suite.sh                 # all arms
#   ./scripts/run_poison_suite.sh qc history      # just these
#   DRY=1 ./scripts/run_poison_suite.sh           # pre-flight only, no GPU
#
# WHY SEQUENTIAL, AND WHY ONE SCRIPT. vLLM's numerics depend on batch
# composition, so two arms scheduled differently can disagree on a few answers
# for reasons that have nothing to do with the poison. Every arm therefore gets
# identical --max-num-seqs, --max-model-len, --limit and model, and they run
# back to back with nothing else on the card. Holding that by hand across seven
# invocations is how it stops being held.
#
# T=0.0 ON PURPOSE. At T=0.6 a clean/poisoned pair differs by sampling as well
# as by the poison, and with a handful of expected flips the sampling could be
# the whole effect. Greedy was checked on this model: 20/20 correct, median
# 2790 decode steps, cap never reached, no repetition loop.
#
# PRE-FLIGHT FIRST. Every arm is dry-run before any GPU work, which re-checks
# token parity against THIS model. A manifest that does not balance aborts the
# suite in seconds instead of after an hour of capture.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MODEL="${OPAL_MODEL:-Qwen/Qwen3.5-35B-A3B}"
LIMIT="${OPAL_LIMIT:-20}"
SPECS="${OPAL_POISON_DIR:-configs/poison}"
ALL=(clean outfmt rules qc schema history null)
ARMS=("${@:-}")
[[ -z "${ARMS[*]}" ]] && ARMS=("${ALL[@]}")

COMMON=(--model "$MODEL"
        --enable-thinking --max-tokens 8192 --max-model-len 32768
        --temperature 0.0
        --max-num-seqs 8 --limit "$LIMIT" --cuda-graphs)

spec_for() { [[ "$1" == clean ]] && echo "" || echo "$SPECS/$1.json"; }

echo "=== pre-flight (no GPU): parity and context budget for every arm ==="
for arm in "${ARMS[@]}"; do
    S="$(spec_for "$arm")"
    printf '  %-9s ' "$arm"
    PY="$HERE/.venv/bin/python"; [[ -x "$PY" ]] || PY="$HERE/.venv-gpu/bin/python"
    PYTHONPATH="$HERE/src" "$PY" scripts/run_experiment.py --dry-run \
        --limit "$LIMIT" --model "$MODEL" --out-dir "/tmp/opal-preflight-$arm" \
        ${S:+--poison "$S"} 2>&1 \
      | grep -E "^poison|^FAILURES" | tr '\n' ' '
    echo
done
rm -rf /tmp/opal-preflight-*
echo

if [[ -n "${DRY:-}" ]]; then echo "DRY=1, stopping before the GPU."; exit 0; fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MAP="results/poison-suite-$STAMP.json"
mkdir -p results
echo "{" > "$MAP"

echo "=== $# arm(s), sequential, capture on ==="
i=0
for arm in "${ARMS[@]}"; do
    S="$(spec_for "$arm")"
    i=$((i+1))
    echo
    echo "--- [$i/${#ARMS[@]}] $arm  ($(date -u +%H:%M:%SZ)) ---"
    # --fg: block until this arm finishes. run_gpu.sh refuses to start while
    # another run is live, so detaching here would serialise by error instead.
    ./scripts/run_gpu.sh --fg --label "poison-$arm" \
        "${COMMON[@]}" ${S:+--poison "$S"}
    RUN="$(readlink results/latest)"
    printf '  "%s": "%s"%s\n' "$arm" "$RUN" \
        "$([[ $i -lt ${#ARMS[@]} ]] && echo , )" >> "$MAP"
    echo "    -> results/$RUN"
done
echo "}" >> "$MAP"

echo
echo "suite map: $MAP"
echo "analyse:   for d in \$(python3 -c \"import json;print(' '.join(json.load(open('$MAP')).values()))\"); do make analyze RUN=results/\$d; done"
