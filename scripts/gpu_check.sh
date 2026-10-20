#!/bin/bash
# Staged bring-up on a local NVIDIA box. The Polaris ladder, minus PBS, modules
# and the offline pinning.
#
#   ./scripts/gpu_check.sh          # all stages
#   ./scripts/gpu_check.sh 3        # just one
#
#   0  environment   GPU, driver, vllm version + BOTH APIs, FLASH_ATTN resolves
#   1  imports       vllm / flowcept / vllm_attn_connector, connector by name
#   2  engine        tiny model generates, NO connector  <- driver/wheel check
#   3  capture       the connector's own e2e: probe, kernel, layout, mass
#   4  chains        3 OPAL chains end to end, provenance verified
#
# Stage 2 excludes the connector entirely, so a failure there is vLLM-on-this-box
# and nothing to do with this work.
#
# Stage 3 is the load-bearing one: it is the only stage that can catch an
# all-zero capture. Mass conservation passes trivially on zeros (0 + 1 == 1), so
# "the output looks structurally fine" is not evidence; the e2e asserts attn_sum
# totals one softmax unit per decode step, which zeros cannot fake.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
# shellcheck source=scripts/lib_siblings.sh
source "${HERE}/scripts/lib_siblings.sh"
if ! opal_resolve_siblings; then
    echo "!! run ./scripts/gpu_setup.sh first -- it fetches them" >&2
    exit 1
fi

GPU_VENV="${GPU_VENV:-${HERE}/.venv-gpu}"
PY="${GPU_VENV}/bin/python"
MODEL="${OPAL_MODEL:-Qwen/Qwen3.5-35B-A3B}"
SMOKE_MODEL="${SMOKE_MODEL:-facebook/opt-125m}"
# Fraction of device memory the tiny smoke models may reserve. Small on
# purpose: 0.5 is unremarkable on a discrete card but means ~64 GB on GB10's
# unified pool -- memory the HOST also needs -- for a 125M model that wants a
# few hundred MB. Raise it only if a stage reports too little KV cache.
SMOKE_UTIL="${OPAL_SMOKE_UTIL:-0.15}"
WORK="${TMPDIR:-/tmp}/opal-gpu-check.$$"
mkdir -p "$WORK"
ONLY="${1:-}"
FAILED=0

# Single-process engine: the connector then runs in THIS process, so the records
# never cross a process boundary and Flowcept needs no message queue. With
# multiprocessing on, the emitted records leave the process and a Redis is
# required -- which is why this is set here and not left to chance.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONPATH="${HERE}/src:${OPAL_CONNECTOR_SRC}:${OPAL_FLOWCEPT_SRC}"
# The venv's bin/ must be on PATH even though we invoke its python by full
# path: FlashInfer's JIT runs `ninja` as a SUBPROCESS, resolved through PATH,
# and pip installs ninja into the venv rather than system-wide.
export PATH="${GPU_VENV}/bin:${PATH}"
export FLOWCEPT_SETTINGS_PATH="${FLOWCEPT_SETTINGS_PATH:-${HERE}/configs/flowcept_local_gpu.yaml}"

if [[ ! -x "$PY" ]]; then
    echo "!! no GPU venv at $GPU_VENV -- run ./scripts/gpu_setup.sh first" >&2
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
    echo "  flowcept $FLOWCEPT_SETTINGS_PATH"
    echo "  sources:"; opal_report_siblings
    if nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; then
        ok "GPU visible"
    else
        fail "no GPU"
    fi
    $PY -c "import torch; print('  torch   ', torch.__version__, 'cuda', torch.cuda.is_available())"
    if $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        ok "torch sees CUDA"
    else
        fail "torch cannot see CUDA -- wheels vs driver"
    fi
    if $PY polaris/check_vllm_api.py --import; then
        ok "vllm version, KV connector API, backend registry, FLASH_ATTN"
    else
        fail "vllm API check"
    fi
    # Headers, before Triton: Triton builds a C shim that #includes <Python.h>
    # the first time it JITs, so a missing header turns into a compile error
    # inside stage 2 or 3 rather than anything that names the real cause.
    if $PY -c 'import os,sysconfig,sys; sys.exit(0 if os.path.exists(
        os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)'; then
        ok "Python.h present (Triton can build its CUDA shim)"
    else
        fail "no Python.h -- Triton cannot compile. Install python3.12-dev, or
        rebuild the venv on a uv-managed interpreter: rm -rf .venv-gpu && ./scripts/gpu_setup.sh"
    fi

    # ninja: FlashInfer JITs during warmup and execs it. Missing, the engine
    # fails at startup with a FileNotFoundError that names neither.
    if command -v ninja >/dev/null 2>&1; then
        ok "ninja present ($(command -v ninja))"
    else
        fail "no ninja on PATH -- FlashInfer cannot JIT and the engine will not
        start. Fix: .venv-gpu/bin/python -m pip install ninja
        (or: sudo apt-get install -y ninja-build)"
    fi

    # flashinfer imports during attention-backend selection, so a version
    # mismatch between flashinfer-python and flashinfer-cubin breaks every
    # model. Check the import itself rather than the versions -- that is what
    # actually has to work.
    if $PY -c "import flashinfer" 2>/dev/null; then
        ok "flashinfer imports ($($PY -c 'import flashinfer;print(flashinfer.__version__)' 2>/dev/null))"
    else
        fail "flashinfer does not import. Most likely a flashinfer-cubin whose
        version does not match flashinfer-python. Fix:
          .venv-gpu/bin/pip uninstall -y flashinfer-cubin"
    fi

    # nvcc: Triton ships its own compiler, but FlashInfer compiles real .cu
    # files -- notably the fused-MoE kernels an MoE model needs at warmup. A
    # box with only the driver and no CUDA toolkit runs dense models fine and
    # dies on MoE ones, with the failure surfacing as "ninja: build stopped".
    if command -v nvcc >/dev/null 2>&1; then
        ok "nvcc present ($(nvcc --version | awk '/release/{print $5,$6}'))"
    else
        echo "  note: no nvcc on PATH. Dense models are fine; an MoE model"
        echo "        (Qwen3-30B-A3B and friends) will fail at warmup when"
        echo "        FlashInfer tries to build its fused-MoE kernels."
    fi

    # Triton is what the connector JITs its q.Kt kernel with. Without it the
    # kernel silently falls back to the pure-torch reference -- correct, but
    # materially slower, and worth knowing before you time anything.
    if $PY -c "import triton; print('  triton  ', triton.__version__)"; then
        ok "triton present (the fast q.Kt path)"
    else
        echo "  note: no triton -- the connector falls back to pure torch." \
             "Correct, slower."
    fi
    # gpu_sizing.py (stage 4) reads this. Build it here rather than telling the
    # user to go and do it: it needs only the tokenizer, takes a couple of
    # minutes, and a box that skipped it gets a silently wrong --max-num-seqs.
    if [[ -f results/plan/plan.json ]]; then
        ok "results/plan/plan.json present"
    else
        echo "  building results/plan/plan.json (tokenizes all 430 turns)..."
        if $PY scripts/run_experiment.py --dry-run --out-dir results/plan >/dev/null; then
            ok "plan built"
        else
            fail "could not build the plan -- gpu_sizing has nothing to read"
        fi
    fi
fi

# ---------------------------------------------------------------- 1
if stage 1 "imports, and the connector resolves by name"; then
    if $PY -c "import vllm, flowcept, vllm_attn_connector"; then
        ok "vllm / flowcept / vllm_attn_connector import"
    else
        fail "imports"
    fi
    # The engine resolves the connector through kv_connector_module_path, by
    # STRING, at construction. A typo there is not an ImportError anywhere -- it
    # is a runtime failure inside engine startup, so check it the way the engine
    # will.
    if $PY -c "
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
if stage 2 "engine runs, NO connector  (pure vLLM on this box)"; then
    if $PY -c "
from vllm import LLM, SamplingParams
llm = LLM(model='${SMOKE_MODEL}', max_model_len=512,
          gpu_memory_utilization=${SMOKE_UTIL}, enforce_eager=True)
out = llm.generate(['The capital of France is'],
                   SamplingParams(temperature=0.0, max_tokens=8))[0].outputs[0].text
print('  generated:', repr(out))
assert out.strip(), 'empty generation'
"; then
        ok "pure vLLM generates -- CUDA wheels match the driver"
    else
        fail "pure vLLM. This is NOT the connector; fix the install first."
        cat >&2 <<MSG

  Reproduce it on its own, with full vLLM logging, to see the real cause:

    cd $HERE
    VLLM_LOGGING_LEVEL=DEBUG .venv-gpu/bin/python -c "
from vllm import LLM
LLM(model='${SMOKE_MODEL}', max_model_len=512,
    gpu_memory_utilization=${SMOKE_UTIL}, enforce_eager=True)"

  Things that have actually caused this, in rough order:
    - no Python.h, so Triton cannot build its shim  (stage 0 checks this now)
    - gpu_memory_utilization too high for a UNIFIED pool; try OPAL_SMOKE_UTIL=0.05
    - the model is not cached and the box has no outbound HF access
    - CUDA wheels vs driver: compare 'nvidia-smi' against torch.version.cuda
MSG
    fi
fi

# ---------------------------------------------------------------- 3
if stage 3 "capture: the connector's own end-to-end"; then
    E2E="${OPAL_CONNECTOR_DIR}/tests/e2e_smoke.py"
    if [[ -f "$E2E" ]]; then
        ( cd "$WORK" && $PY "$E2E" --ranges )
        if [[ $? == 0 ]]; then
            ok "connector e2e (declared ranges, CUDA graphs on)"
        else
            fail "connector e2e -- layout, kernel, probe or emission"
        fi
    else
        fail "no e2e_smoke.py at $E2E"
    fi
fi

# ---------------------------------------------------------------- 4
if stage 4 "this experiment: 3 OPAL chains end to end"; then
    # Three chains, not five: this is a desktop card, and the point is to prove
    # the path works before you commit to all 100. Flags come from gpu_sizing,
    # which reads the dry run and this card's actual VRAM.
    echo "  sizing this card:"
    $PY scripts/gpu_sizing.py --model "$MODEL" | sed 's/^/    /'
    SEQS="$($PY scripts/gpu_sizing.py --model "$MODEL" 2>/dev/null \
            | awk '/--max-num-seqs/ {print $2}')"
    SEQS="${SEQS:-4}"
    OUT="${WORK}/chains"
    mkdir -p "$OUT"
    echo "  running 3 chains with --max-num-seqs $SEQS"
    ( cd "$OUT" && $PY "${HERE}/scripts/run_experiment.py" \
        --csv "${HERE}/data/opal_chains.csv" \
        --model "$MODEL" --limit 3 --out-dir "$OUT" \
        --max-num-seqs "$SEQS" \
        --workflow-id "opal-attn-gpucheck-$$" )
    rc=$?
    if [[ $rc == 0 ]]; then
        ok "3 chains captured and verified"
        echo "  artifacts: $OUT"
    else
        fail "chain run reported $rc failed check(s) -- see above"
    fi
fi

echo
if [[ "$FAILED" == "0" ]]; then
    echo "gpu check: PASS   (scratch: $WORK)"
else
    echo "gpu check: FAILED (scratch: $WORK)" >&2
fi
exit "$FAILED"
