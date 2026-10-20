#!/bin/bash
# Build a CUDA venv for running the experiment on a local NVIDIA box.
#
#   ./scripts/gpu_setup.sh
#   VLLM_VERSION=0.27.1 ./scripts/gpu_setup.sh     # pin a different vLLM
#
# SEPARATE venv from the CPU one. `.venv` holds a CPU torch for the dry run and
# the aggregation tests; this builds `.venv-gpu` with the CUDA wheels. They
# cannot share: installing CUDA torch over the CPU build leaves the dry run
# dependent on a driver it does not need, and a laptop that later loses the GPU
# then cannot plan either.
#
# uv, for the same reasons as on Polaris (polaris/lib.sh): it resolves the vLLM
# dependency tree far faster than pip, and `uv pip freeze` gives a lockfile
# recording exactly what a run used.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
# shellcheck source=scripts/lib_siblings.sh
source "${HERE}/scripts/lib_siblings.sh"

GPU_VENV="${GPU_VENV:-${HERE}/.venv-gpu}"
# >= 0.27.0 is the floor: the connector imports KVConnectorWorkerMetadata and
# SupportsHMA from vllm's connector base, and neither exists before 0.26.
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    cat >&2 <<'MSG'
!! nvidia-smi not found. This script is for a machine with an NVIDIA GPU.
   On a CPU-only box use the local arm instead:  make venv && make check
MSG
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if ! command -v uv >/dev/null 2>&1; then
    echo "==> installing uv into ${HERE}/bin"
    mkdir -p "${HERE}/bin"
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="${HERE}/bin" UV_NO_MODIFY_PATH=1 sh
    export PATH="${HERE}/bin:$PATH"
fi

# Python DEVELOPMENT HEADERS are required, not optional. Triton compiles a
# small C shim (cuda_utils.c) the first time it JITs a kernel, and that
# #includes <Python.h>; so does torch.compile's inductor. Without them you get
#
#   /tmp/.../cuda_utils.c:9:10: fatal error: Python.h: No such file or directory
#
# ...deep inside the first engine start, long after the install. A distro
# python3.12 without python3.12-dev has no headers, which is the common case on
# a freshly imaged box.
#
# There is no way to avoid needing them: this connector JITs its own q.K^T
# kernel, so Triton must be able to build. (--enforce-eager dodges
# torch.compile but not that.)
have_headers() {
    "$1" -c 'import os,sysconfig,sys; sys.exit(0 if os.path.exists(
        os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)' 2>/dev/null
}

create_venv() {   # create_venv [extra uv args...]
    echo "==> creating $GPU_VENV${1:+ ($*)}"
    uv venv "$GPU_VENV" --python 3.12 --seed "$@"
}

if [[ ! -f "$GPU_VENV/bin/activate" ]]; then
    create_venv
    if ! have_headers "$GPU_VENV/bin/python"; then
        # uv's own managed CPython ships its headers, so this needs no root.
        # Safe to redo here: nothing has been installed into the venv yet.
        echo "==> no Python.h in that interpreter; rebuilding on a uv-managed one" >&2
        rm -rf "$GPU_VENV"
        create_venv --python-preference only-managed
    fi
fi
# shellcheck disable=SC1091
source "$GPU_VENV/bin/activate"
export VIRTUAL_ENV="$GPU_VENV"
echo "==> python $(python -V 2>&1) at $(command -v python)"

if ! have_headers "$GPU_VENV/bin/python"; then
    INC="$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
    cat >&2 <<MSG
!! No Python.h under $INC
   Triton cannot build its CUDA shim without it, so the connector's kernel
   cannot JIT and the first engine start will fail. Fix it one of two ways:

   with root:
     sudo apt-get install -y python3.12-dev     # or python3-devel on RHEL

   without root -- rebuild this venv on a uv-managed interpreter, which
   bundles its headers (reinstalls vLLM, ~10 min):
     rm -rf "$GPU_VENV"
     ./scripts/gpu_setup.sh
MSG
    exit 1
fi
echo "==> Python.h present ($(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])'))"

echo "==> installing vLLM ${VLLM_VERSION} (pulls CUDA torch and triton)"
uv pip install "vllm==${VLLM_VERSION}"

# Static check before the editable installs: does this vLLM have BOTH APIs the
# connector needs -- the KV connector base and the attention backend registry?
python polaris/check_vllm_api.py || exit 1

# --clone: in the genesis workspace the siblings are already next door and are
# used as-is (local edits intact); on a bare box they are fetched into vendor/.
echo "==> resolving flowcept + vllm-attn-connector"
opal_resolve_siblings --clone || exit 1
opal_report_siblings

# Build-system compatibility, before the install rather than after it fails.
opal_fix_pep639_license "$OPAL_FLOWCEPT_DIR"
opal_fix_nvml_memory_probe "$OPAL_FLOWCEPT_DIR"
opal_fix_pep639_license "$OPAL_CONNECTOR_DIR"

echo "==> installing them (editable)"
uv pip install -e "${OPAL_FLOWCEPT_DIR}[telemetry,nvidia]"
uv pip install -e "${OPAL_CONNECTOR_DIR}"
# ninja is NOT optional. FlashInfer JIT-compiles kernels during vLLM's warmup
# and shells out to `ninja`; without it the engine dies at startup with a bare
#   FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
# nested inside "Engine core initialization failed", which names neither
# flashinfer nor ninja in its summary line.
uv pip install transformers "huggingface_hub[cli]" ninja

uv pip freeze > "${HERE}/requirements.gpu.lock"
RESOLVED="$(python -c 'import importlib.metadata as m; print(m.version("vllm"))')"

# The real test: does it import against THIS driver, and does the backend the
# probe overrides resolve? A probe that fails to install is silent -- the
# capture still writes valid records and every number in them is zero.
echo "==> verifying on the GPU"
python polaris/check_vllm_api.py --import || exit 1

# ninja lands in the venv's bin/, which is only on PATH when the venv is
# ACTIVATED. Scripts here invoke .venv-gpu/bin/python directly instead, so
# confirm the binary is reachable the way a run will actually look for it.
if ! PATH="$GPU_VENV/bin:$PATH" command -v ninja >/dev/null 2>&1; then
    echo "!! ninja is still not on PATH even with $GPU_VENV/bin prepended." >&2
    echo "   FlashInfer cannot JIT without it. Install it system-wide:" >&2
    echo "     sudo apt-get install -y ninja-build" >&2
    exit 1
fi
echo "==> ninja: $(PATH="$GPU_VENV/bin:$PATH" command -v ninja)"

cat <<SUMMARY

==> done.
    venv      $GPU_VENV
    vllm      $RESOLVED
    lockfile  ${HERE}/requirements.gpu.lock

    Next:
      make plan                 # if you have not already
      ./scripts/gpu_check.sh    # staged bring-up, ends with 3 real chains
SUMMARY
