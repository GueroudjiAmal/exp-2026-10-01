#!/bin/bash
# Staged bring-up test for Polaris. Run it INSIDE an interactive debug job:
#
#   qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 \
#        -q debug -A $OPAL_PROJECT
#   cd $OPAL_ROOT && ./polaris/smoke_test.sh
#
# Stages are ordered by what fails first and how cheap it is to find out. Each
# isolates ONE failure mode, so when something breaks you know what broke rather
# than staring at a vLLM traceback.
#
#   0  environment   python, vllm version + both APIs, GPUs, offline flags
#   1  imports       vllm / flowcept / vllm_attn_connector, connector by name
#   2  engine        tiny model generates, NO connector   <- driver/wheel check
#   3  capture       the connector's own e2e against a real engine
#   4  chains        5 OPAL chains end to end, provenance verified
#
# Stage 2 excludes the connector entirely, so a failure there is vLLM-on-Polaris
# and nothing to do with this work. Stage 3 is the connector's own 43-45
# assertions. Stage 4 is this experiment.
#
# Run all of them, or just one:  ./polaris/smoke_test.sh 3
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
# shellcheck disable=SC1091
source ./polaris/env.sh

SMOKE_MODEL="${SMOKE_MODEL:-facebook/opt-125m}"
MODEL="${OPAL_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
WORK="${TMPDIR:-/tmp}/opal-smoke.$$"
mkdir -p "$WORK"
ONLY="${1:-}"
FAILED=0

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

# Bail out before stage 0 if setup never completed. Without this, every stage
# emits a ModuleNotFoundError and the real cause is buried under four tracebacks.
if [[ ! -f "${OPAL_VENV:-/nonexistent}/bin/activate" ]]; then
    echo "!! No venv at ${OPAL_VENV:-unset}. Run polaris/setup.sh first, in a job." >&2
    exit 1
fi

# ---------------------------------------------------------------- 0
if stage 0 "environment"; then
    opal_report_stamp
    echo "  python   $(python -V 2>&1)"
    echo "  root     $OPAL_ROOT"
    echo "  flowcept $FLOWCEPT_SETTINGS_PATH"
    echo "  offline  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
    if python polaris/check_vllm_api.py --import; then
        ok "vllm version, KV connector API, backend registry, FLASH_ATTN"
    else
        fail "vllm API check"
    fi
    if nvidia-smi -L; then ok "GPUs visible"; else fail "no GPUs in this allocation"; fi
    [[ -f data/opal_chains.csv ]] && ok "benchmark CSV present" \
                                  || fail "data/opal_chains.csv missing"
fi

# ---------------------------------------------------------------- 1
if stage 1 "imports, and the connector resolves by name"; then
    if python -c "import vllm, flowcept, vllm_attn_connector" 2>&1; then
        ok "vllm / flowcept / vllm_attn_connector import"
    else
        fail "imports"
    fi
    # The engine resolves the connector through kv_connector_module_path, by
    # STRING, at construction. A typo there is not an ImportError anywhere --
    # it is a runtime failure inside engine startup, so check it the same way
    # the engine will.
    if python -c "
import importlib
m = importlib.import_module('vllm_attn_connector')
assert hasattr(m, 'AttnConnector'), 'AttnConnector not exported'
assert hasattr(m, 'install_probe'), 'install_probe not exported'
print('  AttnConnector + install_probe resolve from the module path')
"; then
        ok "kv_connector_module_path resolves"
    else
        fail "connector does not resolve by name"
    fi
fi

# ---------------------------------------------------------------- 2
if stage 2 "engine runs, NO connector  (pure vLLM on Polaris)"; then
    if python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='${SMOKE_MODEL}', max_model_len=512, gpu_memory_utilization=0.5,
          enforce_eager=True)
out = llm.generate(['The capital of France is'],
                   SamplingParams(temperature=0.0, max_tokens=8))[0].outputs[0].text
print('  generated:', repr(out))
assert out.strip(), 'empty generation'
"; then
        ok "pure vLLM generates -- CUDA wheels match the driver"
    else
        fail "pure vLLM. This is NOT the connector; fix the install first."
    fi
fi

# ---------------------------------------------------------------- 3
if stage 3 "capture: the connector's own end-to-end"; then
    # Its 43-45 assertions cover what this experiment cannot: the probe under
    # CUDA graphs, the Triton kernel, cache layout, mass conservation, and that
    # attention actually MOVES between decode steps. Run it with graphs ON --
    # that is how a real run goes, and an all-zero capture still "passes" mass
    # conservation, so this is the check that catches it.
    E2E="${OPAL_ROOT}/vllm-attn-connector/tests/e2e_smoke.py"
    if [[ -f "$E2E" ]]; then
        ( cd "$WORK" && python "$E2E" --ranges )
        if [[ $? == 0 ]]; then ok "connector e2e (declared ranges, graphs on)"
        else fail "connector e2e -- layout, kernel, probe or emission"; fi
    else
        fail "no e2e_smoke.py at $E2E (did setup.sh clone the connector?)"
    fi
fi

# ---------------------------------------------------------------- 4
if stage 4 "this experiment: 5 OPAL chains end to end"; then
    OUT="${WORK}/chains"
    mkdir -p "$OUT"
    ( cd "$OUT" && python "${OPAL_ROOT}/scripts/run_experiment.py" \
        --csv "${OPAL_ROOT}/data/opal_chains.csv" \
        --model "$MODEL" --limit 5 --out-dir "$OUT" \
        --workflow-id "opal-attn-smoke-$$" --cuda-graphs )
    rc=$?
    if [[ $rc == 0 ]]; then
        ok "5 chains captured and verified"
        echo "  artifacts: $OUT"
    else
        fail "chain run reported $rc failed check(s) -- see above"
    fi
fi

echo
if [[ "$FAILED" == "0" ]]; then
    echo "smoke test: PASS   (scratch: $WORK)"
else
    echo "smoke test: FAILED (scratch: $WORK)" >&2
fi
exit "$FAILED"
