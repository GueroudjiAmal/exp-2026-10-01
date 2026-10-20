#!/bin/bash
# One reference-style heat map per run, for the SAME turn, so the arms can be
# read side by side.
#
#   ./scripts/heatmap_suite.sh <suite.json> <chain-id> <round> [extra args]
#
# Example:
#   ./scripts/heatmap_suite.sh results/poison-suite-20261006T235421Z.json \
#       exp32-chain-0020-hard 5
#
# Writes results/heatmaps/<round>-<chain>/<arm>.txt, one file per run, plus
# index.txt listing the answer each arm gave. Same turn, same format, so a
# diff between two files is exactly the effect of that arm's poison.
#
# The format is the one from row_analogue_reasoned.txt: a column per answer
# item, heat as max-with-decay over that item's decode steps, every line of
# the prompt shown.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

SUITE="${1:?usage: heatmap_suite.sh <suite.json> <chain-id> <round> [args...]}"
CHAIN="${2:?missing chain id}"
ROUND="${3:?missing round}"
shift 3

PY="$HERE/.venv/bin/python"; [[ -x "$PY" ]] || PY="$HERE/.venv-gpu/bin/python"
export PYTHONPATH="$HERE/src:$HERE/../vllm-attn-connector/src"

OUT="results/heatmaps/r${ROUND}-${CHAIN}"
mkdir -p "$OUT"
IDX="$OUT/index.txt"
: > "$IDX"

# Arm order comes from the suite map; clean first so a diff reads as
# "what the poison changed" rather than the other way round.
ARMS=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); \
print('\n'.join(['clean']+[k for k in d if k!='clean']))" "$SUITE")

for arm in $ARMS; do
    RUN=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" \
          "$SUITE" "$arm")
    printf '  %-9s ' "$arm"
    if "$PY" scripts/heatmap.py "results/$RUN" --chain "$CHAIN" --round "$ROUND" \
            "$@" > "$OUT/$arm.txt" 2> "$OUT/$arm.err"; then
        rm -f "$OUT/$arm.err"
        # second line of the render carries answered/truth/verdict
        sed -n '2p' "$OUT/$arm.txt" | tee -a /dev/stderr >> "$IDX"
        sed -i "\$s/^/$arm: /" "$IDX"
    else
        echo "FAILED -- $(tail -1 "$OUT/$arm.err")"
        echo "$arm: FAILED" >> "$IDX"
    fi
done

echo
echo "wrote $OUT/  ($(ls "$OUT"/*.txt 2>/dev/null | wc -l) files)"
echo "compare two arms:  diff -y --width=200 $OUT/clean.txt $OUT/qc.txt | less"
