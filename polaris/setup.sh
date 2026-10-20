#!/bin/bash
# Build the Polaris environment. Run INSIDE A JOB (a GPU is needed to verify the
# vLLM install, and the proxy gives outbound access):
#
#   qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 \
#        -q debug -A $OPAL_PROJECT
#   cd $OPAL_ROOT && ./polaris/setup.sh
#
# No conda. The conda/2025-09-* modules pin gcc-native/14.2 and
# cray-hdf5-parallel/1.14.3.5, neither of which exists under the current Cray PE,
# so they cannot be loaded at all. This uses spack-pe-base with an explicit
# toolchain and a uv-built venv instead.
#
# Stages everything a run needs -- wheels, model weights, tokenizer -- into
# $OPAL_ROOT. Runs are then pinned offline (polaris/env.sh) so they cannot
# silently fetch anything.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${OPAL_PROJECT:?set OPAL_PROJECT to your ALCF project/allocation name}"
: "${OPAL_ROOT:=/lus/eagle/projects/${OPAL_PROJECT}/${USER}/opal-attn}"
OPAL_VENV="${OPAL_VENV:-${OPAL_ROOT}/.venv}"
# Defaulted here, not inherited: setup.sh deliberately does NOT source env.sh
# (that pins runs offline), so anything env.sh exports has to be repeated.
OPAL_DATA="${OPAL_DATA:-${OPAL_ROOT}/data}"

# vLLM >= 0.27.0 is the floor: the connector imports KVConnectorWorkerMetadata
# and SupportsHMA from vllm's connector base, and neither exists before 0.26.
# It is tested against 0.27.x; 0.28.0 is what spotterai's Polaris env resolved
# to and is the default here.
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"

read -r -a MODELS <<< "${OPAL_MODELS:-Qwen/Qwen3-4B-Instruct-2507}"
SMOKE_MODEL="${SMOKE_MODEL:-facebook/opt-125m}"
MODELS+=("$SMOKE_MODEL")

if [[ -z "${PBS_JOBID:-}" ]]; then
    cat >&2 <<MSG
!! Not inside a PBS job. Step 6 verifies vLLM by importing it on a GPU, which a
   login node cannot do. Get a node, then re-run:

     qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 \\
          -q debug -A ${OPAL_PROJECT}
     cd ${OPAL_ROOT} && ./polaris/setup.sh
MSG
    exit 1
fi

# Proxy first: everything below downloads, and compute nodes reach out only
# through it. env.sh is deliberately NOT sourced here -- it pins runs offline.
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov}"

if ! curl -sI --max-time 20 https://huggingface.co >/dev/null 2>&1; then
    echo "!! no outbound access via ${https_proxy} -- nothing can be staged." >&2
    exit 1
fi
echo "==> outbound OK via ${https_proxy}"
echo "==> root: $OPAL_ROOT"
mkdir -p "$OPAL_ROOT"

# --- 1. source layout ---------------------------------------------------------
clone_or_keep() {
    local name="$1" url="$2"
    if [[ -d "$OPAL_ROOT/$name/.git" ]]; then
        echo "==> $name already present, not touching it (git pull it yourself)"
    else
        echo "==> cloning $name"
        git clone "$url" "$OPAL_ROOT/$name"
    fi
}
clone_or_keep flowcept             https://github.com/spotter-ai-genesis/flowcept.git
clone_or_keep vllm-attn-connector  https://github.com/spotter-ai-genesis/vllm-attn-connector.git

WORKSPACE="$(cd "$HERE/.." && pwd)"
if [[ "$WORKSPACE" != "$OPAL_ROOT" ]]; then
    for d in configs scripts src polaris data; do
        [[ -d "$WORKSPACE/$d" ]] && cp -r "$WORKSPACE/$d" "$OPAL_ROOT/"
    done
fi

# --- 2. modules ---------------------------------------------------------------
# shellcheck source=polaris/lib.sh
source "$HERE/lib.sh"
opal_report_stamp
opal_load_modules || { echo "!! module setup failed" >&2; exit 1; }
echo "==> modules loaded; CC=$CC CXX=$CXX"
echo "==> CUDA: ${CUDA_HOME:-${CUDATOOLKIT_HOME:-unset}}"
nvidia-smi -L || echo "!! no GPUs visible in this allocation"

# --- 3. venv via uv -----------------------------------------------------------
opal_require_uv || exit 1
if [[ ! -f "$OPAL_VENV/bin/activate" ]]; then
    echo "==> creating venv at $OPAL_VENV"
    uv venv "$OPAL_VENV" --python 3.12 --seed
fi
# shellcheck disable=SC1091
source "$OPAL_VENV/bin/activate"
export VIRTUAL_ENV="$OPAL_VENV"
echo "==> python $(python -V 2>&1) at $(command -v python)"

# Triton JITs the connector's q.K^T kernel and builds a C shim that includes
# <Python.h>. The ALCF module stack's python has headers, but a uv-managed one
# may not be what got picked -- check now rather than at the first engine start.
if ! python -c 'import os,sysconfig,sys; sys.exit(0 if os.path.exists(
    os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)'; then
    echo "!! no Python.h for this interpreter; Triton cannot build its CUDA shim." >&2
    echo "   Rebuild the venv on a uv-managed python (it bundles headers):" >&2
    echo "     rm -rf $OPAL_VENV && ./polaris/setup.sh" >&2
    exit 1
fi

# --- 4. packages --------------------------------------------------------------
echo "==> installing vLLM ${VLLM_VERSION}"
uv pip install "vllm==${VLLM_VERSION}"

# Static check before anything expensive: is the connector API this build needs
# actually present in the installed vLLM source?
python "$OPAL_ROOT/polaris/check_vllm_api.py" || exit 1

# Same PEP 639 / setuptools >= 77 clash as on a local GPU box; the helper
# explains it. The clones live directly under $OPAL_ROOT here rather than in a
# vendor/ directory, so point it at them explicitly.
# shellcheck source=scripts/lib_siblings.sh
source "$OPAL_ROOT/scripts/lib_siblings.sh"
opal_fix_pep639_license "$OPAL_ROOT/vllm-attn-connector" --owned
opal_fix_nvml_memory_probe "$OPAL_ROOT/flowcept" --owned

echo "==> installing flowcept + vllm-attn-connector (editable)"
uv pip install -e "$OPAL_ROOT/flowcept[telemetry,nvidia]"
uv pip install -e "$OPAL_ROOT/vllm-attn-connector"
uv pip install "huggingface_hub[cli]" pandas pyarrow

uv pip freeze > "$OPAL_ROOT/requirements.polaris.lock"
RESOLVED_VLLM="$(python -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || echo unknown)"
echo "==> vllm $RESOLVED_VLLM  (pin it: export VLLM_VERSION=$RESOLVED_VLLM)"

# --- 5. weights ---------------------------------------------------------------
export HF_HOME="${HF_HOME:-${OPAL_ROOT}/hf}"
mkdir -p "$HF_HOME" "$OPAL_DATA"

# ALCF preloads common weights read-only here. If yours is present you can skip
# the download: export HF_HUB_CACHE=/eagle/datasets/model-weights/hub
SHARED_HUB="/eagle/datasets/model-weights/hub"
for m in "${MODELS[@]}"; do
    if [[ -d "${SHARED_HUB}/models--${m//\//--}" ]]; then
        echo "==> $m is already in the shared hub"
        echo "    to use it instead: export HF_HUB_CACHE=$SHARED_HUB"
    fi
    echo "==> caching model $m"
    hf download "$m" || huggingface-cli download "$m"
done

# The benchmark is a CSV that travels with the source, not a gated download --
# so unlike spotterai's BioMysteryBench there is nothing to accept here.
if [[ -f "$OPAL_ROOT/data/opal_chains.csv" ]]; then
    echo "==> benchmark present: $(wc -l < "$OPAL_ROOT/data/opal_chains.csv") CSV lines"
else
    echo "!! data/opal_chains.csv is missing -- push.sh should have sent it" >&2
    exit 1
fi

# --- 6. does vLLM import on a GPU? --------------------------------------------
# Proves the CUDA wheels work against this driver -- the most likely thing to be
# wrong, and cheapest to find out here rather than inside a batch job.
echo "==> importing vLLM on a real GPU"
if ! python "$OPAL_ROOT/polaris/check_vllm_api.py" --import; then
    cat >&2 <<'MSG'
!! vLLM has the right API but will not import on this node.
   Usually the pip CUDA wheels vs the driver. Options:
     - older pin:  VLLM_VERSION=0.27.1 ./polaris/setup.sh   (0.27.0 is the floor)
     - compare:    nvidia-smi   vs   the loaded cudatoolkit-standalone version
   Everything else staged fine; only this step failed.
MSG
    exit 1
fi

# --- 7. warm the compile cache ------------------------------------------------
# The first engine start JIT-compiles for a while; do not let that eat a batch
# job's walltime. Warm the EVAL model -- the cache is keyed on model shape, so
# warming the smoke model would not help. No enforce_eager: that skips the very
# compilation being cached. Non-fatal.
echo "==> warming the compile cache for ${MODELS[0]}"
VLLM_CACHE_ROOT="${OPAL_ROOT}/.cache/vllm" \
    python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='${MODELS[0]}', max_model_len=8192, gpu_memory_utilization=0.85)
print(llm.generate(['hello'], SamplingParams(max_tokens=4))[0].outputs[0].text)
" || echo "!! warmup failed (non-fatal; the cache is just cold)"

cat <<SUMMARY

==> done.
    root      $OPAL_ROOT
    venv      $OPAL_VENV
    vllm      $RESOLVED_VLLM
    HF_HOME   $HF_HOME
    lockfile  $OPAL_ROOT/requirements.polaris.lock

    Next, in THIS same job:
      source polaris/env.sh
      ./polaris/smoke_test.sh

    Then the real run:
      qsub -A $OPAL_PROJECT polaris/job.pbs
SUMMARY
