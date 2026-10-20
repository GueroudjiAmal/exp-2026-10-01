#!/bin/bash
# Sourced by job.pbs, smoke_test.sh, and by hand. Safe on a login node.
#   source polaris/env.sh
#
# Everything is driven off OPAL_ROOT and OPAL_PROJECT; set them in your shell
# or ~/.bashrc.

# ---- what you must set -------------------------------------------------------
: "${OPAL_PROJECT:?set OPAL_PROJECT to your ALCF project/allocation name}"
: "${OPAL_ROOT:=/lus/eagle/projects/${OPAL_PROJECT}/${USER}/opal-attn}"

# ---- derived layout ----------------------------------------------------------
export OPAL_PROJECT OPAL_ROOT
export OPAL_VENV="${OPAL_VENV:-${OPAL_ROOT}/.venv}"
export OPAL_RUNS="${OPAL_RUNS:-${OPAL_ROOT}/runs}"
export OPAL_DATA="${OPAL_DATA:-${OPAL_ROOT}/data}"

# Model weights and tokenizers. Must be on a project filesystem: $HOME has a
# small quota and is not meant for model weights.
export HF_HOME="${HF_HOME:-${OPAL_ROOT}/hf}"
export HF_HUB_DISABLE_TELEMETRY=1

# The two sibling checkouts. setup.sh clones them here; the connector is used
# via PYTHONPATH as well as pip so a `git pull` takes effect without reinstalling.
export OPAL_CONNECTOR_SRC="${OPAL_CONNECTOR_SRC:-${OPAL_ROOT}/vllm-attn-connector/src}"
export OPAL_FLOWCEPT_SRC="${OPAL_FLOWCEPT_SRC:-${OPAL_ROOT}/flowcept/src}"
export PYTHONPATH="${OPAL_ROOT}/src:${OPAL_CONNECTOR_SRC}:${OPAL_FLOWCEPT_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

# ---- are we in a job? --------------------------------------------------------
# PBS exports PBS_JOBID inside a job (batch or interactive) and nowhere else.
if [[ -n "${PBS_JOBID:-}" ]]; then OPAL_OFFLINE=1; else OPAL_OFFLINE=0; fi
export OPAL_OFFLINE

if [[ "${OPAL_OFFLINE}" == "1" ]]; then
    # Offline by CHOICE, not necessity -- compute nodes do reach out through the
    # ALCF proxy, and setup.sh depends on that to stage weights. But a run that
    # silently downloads is slower and not reproducible: you cannot tell
    # afterwards which weights it used. A run is pinned to what was staged, and
    # a cache miss fails immediately instead of quietly fetching something.
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export VLLM_NO_USAGE_STATS=1
    export DO_NOT_TRACK=1
    unset http_proxy https_proxy ftp_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY
else
    # Login node: leave the proxy in place for pip/git in the same shell.
    export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
    export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$https_proxy"
    export no_proxy="${no_proxy:-admin,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.alcf.anl.gov}"
fi

# ---- compile / JIT caches ----------------------------------------------------
# ALCF's vLLM guidance is TMPDIR=/tmp (and RAY_TMPDIR=/tmp): Triton and
# torch.compile write many small files and Lustre handles that badly. This job
# leans on Triton harder than most -- the connector JITs its own q.K^T kernel --
# so getting this wrong shows up as a very slow first request.
# VLLM_CACHE_ROOT is the exception: torch.compile artifacts are worth keeping
# between jobs, so they live on the project filesystem.
export TMPDIR="${TMPDIR:-/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp}"
_opal_scratch="${TMPDIR}/${USER}-opal.${PBS_JOBID:-login}"
mkdir -p "${_opal_scratch}" 2>/dev/null || true
export TRITON_CACHE_DIR="${_opal_scratch}/triton"
export TORCHINDUCTOR_CACHE_DIR="${_opal_scratch}/inductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${OPAL_ROOT}/.cache/vllm}"

# ---- vLLM --------------------------------------------------------------------
# Single-process engine: the KV connector then runs in THIS process, so the
# records it emits never cross a process boundary and Flowcept needs no message
# queue. That is what makes a Redis-free compute node possible, and it is the
# configuration the connector is tested under.
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if [[ -n "${PBS_JOBID:-}" ]]; then
    # TP>1 needs vLLM to know its own high-speed-network address, and localhost
    # exempted from any proxy (straight from the ALCF vLLM guide; no-op at TP=1).
    #
    # `|| true` is load-bearing: this file is sourced by job.pbs under
    # `set -euo pipefail`, and if the HSN name does not resolve the pipeline
    # fails, the assignment fails, and the job dies at `source env.sh` with no
    # output at all.
    _hsn=$(getent hosts "$(hostname).hsn.cm.polaris.alcf.anl.gov" 2>/dev/null \
           | awk '{print $1}' | sort | head -n1) || true
    if [[ -n "${_hsn:-}" ]]; then
        export VLLM_HOST_IP="$_hsn"
    else
        echo "opal: could not resolve the HSN address; VLLM_HOST_IP unset (fine at TP=1)" >&2
    fi
    export no_proxy="localhost,127.0.0.1"
    unset _hsn
fi

# ---- flowcept ----------------------------------------------------------------
export FLOWCEPT_SETTINGS_PATH="${FLOWCEPT_SETTINGS_PATH:-${OPAL_ROOT}/configs/flowcept_polaris.yaml}"

# ---- modules + venv ----------------------------------------------------------
# shellcheck source=polaris/lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

opal_load_modules || echo "opal: continuing without the module env" >&2

if [[ -f "${OPAL_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${OPAL_VENV}/bin/activate"
    # Activation already does this, but job.pbs and smoke_test.sh also invoke
    # the venv's python by full path in places, and FlashInfer resolves `ninja`
    # through PATH at JIT time. The module env provides one too.
    export PATH="${OPAL_VENV}/bin:${PATH}"
else
    echo "opal: no venv at ${OPAL_VENV} -- run polaris/setup.sh first" >&2
fi

# ALCF known issue: the environment can abort with
#   MPIDI_CRAY_init: GPU_SUPPORT_ENABLED is requested, but GTL library is not linked
# The documented fix is importing mpi4py first thing. Opt-in, and a last resort:
# ALCF's PyTorch page warns that setting CUDA_VISIBLE_DEVICES AFTER mpi4py is
# imported can fail, because `from mpi4py import MPI` implicitly runs
# MPI_COMM_WORLD init -- and setting CUDA_VISIBLE_DEVICES per worker is exactly
# what vLLM does at TP>1. So this fix and tensor parallelism tend to fight.
export OPAL_MPI4PY_FIX="${OPAL_MPI4PY_FIX:-0}"

unset _opal_scratch
