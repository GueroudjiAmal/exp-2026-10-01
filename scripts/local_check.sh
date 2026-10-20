#!/bin/bash
# The local arm, end to end. No CUDA, no vLLM, no network beyond the tokenizer.
#
#   ./scripts/local_check.sh            # all stages
#   ./scripts/local_check.sh 2          # just one
#
# Stages are ordered by how cheap it is to find out:
#   0  environment      venv, imports, the sibling checkouts on PYTHONPATH
#   1  aggregations     the connector's own CPU tests: reduction + segment planning
#   2  flowcept         provenance in, provenance out, and report.py's join
#   3  dry run (small)  3 chains tokenized, segmented and budgeted
#   4  dry run (full)   all 100 chains; writes the plan the Polaris job sizes from
#
# What NONE of this covers: the probe, the Triton kernel, the KV cache layout,
# and whether the captured numbers are real attention. Those need a GPU and are
# polaris/smoke_test.sh stage 3.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

GENESIS="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="${HERE}/src:${GENESIS}/vllm-attn-connector/src:${GENESIS}/flowcept/src"
export FLOWCEPT_SETTINGS_PATH="${FLOWCEPT_SETTINGS_PATH:-${HERE}/configs/flowcept_local.yaml}"
PY="${HERE}/.venv/bin/python"
ONLY="${1:-}"
FAILED=0

if [[ ! -x "$PY" ]]; then
    echo "!! no venv at ${HERE}/.venv -- run: make venv" >&2
    exit 1
fi

stage() {
    [[ -n "$ONLY" && "$ONLY" != "$1" ]] && return 1
    echo
    echo "=============================================================="
    echo "  stage $1: $2"
    echo "=============================================================="
    return 0
}
ok()   { echo "  PASS: $*"; }
fail() { echo "  FAIL: $*" >&2; FAILED=1; }

# ---------------------------------------------------------------- 0
if stage 0 "environment"; then
    echo "  python   $($PY -V 2>&1)"
    echo "  settings $FLOWCEPT_SETTINGS_PATH"
    $PY -c "import torch; print('  torch   ', torch.__version__, 'cuda', torch.cuda.is_available())"
    for mod in transformers flowcept opal_attn; do
        if $PY -c "import $mod" 2>/dev/null; then ok "import $mod"; else fail "import $mod"; fi
    done
    # The connector PACKAGE needs vLLM and cannot import here -- that is
    # expected. Its READER does not, and must, because analysis of a Polaris
    # run happens on this machine. opal_attn.provenance is the shim that gets
    # at the reader without dragging the package in; if it breaks, the results
    # of every GPU run become unreadable off-node.
    if $PY -c "from opal_attn.provenance import load_provenance" 2>/dev/null; then
        ok "provenance reader resolves without vLLM"
    else
        fail "provenance reader (opal_attn.provenance)"
    fi
    # vllm must NOT be importable here; if it is, someone installed a CUDA
    # wheel on a machine with no CUDA and stage 3 will fail confusingly.
    if $PY -c "import vllm" 2>/dev/null; then
        echo "  note: vllm imports locally -- unexpected on this machine, but harmless"
    else
        ok "vllm absent, as expected on a CPU-only box"
    fi
    [[ -f data/opal_chains.csv ]] && ok "data/opal_chains.csv present" \
                                  || fail "data/opal_chains.csv missing"
    # A stray CSV outside data/ is a decoy. EVERY script reads
    # data/opal_chains.csv, and polaris/push.sh ships the data/ directory but
    # not a root-level file -- so editing the wrong copy changes nothing,
    # locally or on Polaris, and nothing complains. Identical copies are merely
    # untidy; differing ones mean work has already been done on the file that
    # is not being read.
    for stray in ./opal_chains.csv ../opal_chains.csv; do
        [[ -f "$stray" ]] || continue
        if cmp -s "$stray" data/opal_chains.csv; then
            echo "  note: $stray duplicates data/opal_chains.csv byte for byte."
            echo "        Nothing reads it. Delete it to avoid editing the decoy."
        else
            fail "$stray DIFFERS from data/opal_chains.csv, and data/ is the one
        every script reads. Reconcile them before running anything:
          diff <(cut -d, -f1,2 '$stray') <(cut -d, -f1,2 data/opal_chains.csv)"
        fi
    done
fi

# ---------------------------------------------------------------- 1
if stage 1 "connector aggregations (CPU)"; then
    if $PY -m pytest "${GENESIS}/vllm-attn-connector/tests/test_aggregations.py" -q; then
        ok "streaming reduction and segment planning match their references"
    else
        fail "connector aggregation tests"
    fi
fi

# ---------------------------------------------------------------- 2
if stage 2 "flowcept provenance round-trip"; then
    if $PY scripts/flowcept_smoke.py; then
        ok "interceptor -> offline dump -> report.py join"
    else
        fail "flowcept round-trip"
    fi
fi

# ---------------------------------------------------------------- 3
if stage 3 "dry run, 3 chains"; then
    if $PY scripts/run_experiment.py --dry-run --limit 3 \
            --out-dir results/local-check-small; then
        ok "3 chains planned"
    else
        fail "small dry run"
    fi
fi

# ---------------------------------------------------------------- 4
if stage 4 "dry run, all chains"; then
    if $PY scripts/run_experiment.py --dry-run --out-dir results/plan; then
        ok "all chains planned; results/plan/plan.json is what Polaris sizes from"
    else
        fail "full dry run"
    fi
fi

echo
if [[ "$FAILED" == "0" ]]; then
    echo "local check: PASS"
else
    echo "local check: FAILED" >&2
fi
exit "$FAILED"
