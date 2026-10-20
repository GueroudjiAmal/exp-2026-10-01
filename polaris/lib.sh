# Shared Polaris shell helpers. Sourced, never executed.

# Load the Polaris module environment. Returns 0 on success, 1 on failure.
#
# This is the sequence that actually works on Polaris. Notably it does NOT use
# the conda module: conda/2025-09-* pins gcc-native/14.2 and
# cray-hdf5-parallel/1.14.3.5, neither of which exists under the current Cray PE,
# so `module load conda` fails on login and compute nodes alike. spack-pe-base
# plus an explicit toolchain avoids that entirely.
#
# `set -u` has to come off first: lmod's `module` function reads LOADEDMODULES
# and _LMFILES_, which are unbound before anything is loaded, so under `set -u`
# `module load` dies with an unbound-variable error that reads like a missing
# module. Restored from $- afterwards -- `$(set +o)` does not round-trip errexit
# through command substitution.
opal_load_modules() {
    local flags="$-"
    local rc=0
    set +eu

    module use /soft/modulefiles                  || rc=1
    module load spack-pe-base                     || rc=1
    module load cmake ninja \
                "${OPAL_GCC_MODULE:-gcc-native/14}" \
                "${OPAL_CUDA_MODULE:-cudatoolkit-standalone/12.9.0}" || rc=1
    # darshan's I/O interposition upsets the vLLM/torch process tree, and this
    # job writes a few hundred small SafeTensors files. It is in the Polaris
    # default set, so it has to be removed rather than just not added.
    module unload darshan 2>/dev/null

    export CC=cc
    export CXX=CC

    case "$flags" in *e*) set -e ;; esac
    case "$flags" in *u*) set -u ;; esac

    [[ "$rc" == "0" ]] || echo "opal: module setup failed (errors above)" >&2
    return "$rc"
}

# uv builds the venv. Install it if absent.
#
# Auto-installing is right here specifically because of WHERE this is called
# from: setup.sh has already exported the ALCF proxy and verified outbound
# access by the time it needs uv. Telling the user to install it by hand sends
# them to a shell with no proxy set -- a second wall immediately after the first.
#
# Installs into the project (UV_INSTALL_DIR), not $HOME: it belongs with the
# rest of the staged environment, and a project filesystem is where it survives.
opal_require_uv() {
    local install_dir="${1:-${OPAL_ROOT:-$PWD}/bin}"

    # Already reachable, or from a previous run of this function / uv's installer.
    for candidate in "$install_dir" "$HOME/.local/bin"; do
        if [[ -x "$candidate/uv" ]]; then
            export PATH="$candidate:$PATH"
            return 0
        fi
    done
    command -v uv >/dev/null 2>&1 && return 0

    if [[ -z "${https_proxy:-}${HTTPS_PROXY:-}" ]] \
       && ! curl -sI --max-time 10 https://astral.sh >/dev/null 2>&1; then
        echo "opal: uv is missing and there is no outbound access to install it." >&2
        echo "         On a Polaris compute node, export the proxy first:" >&2
        echo "           export https_proxy=http://proxy.alcf.anl.gov:3128" >&2
        echo "           export http_proxy=http://proxy.alcf.anl.gov:3128" >&2
        echo "         then: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        return 1
    fi

    echo "opal: uv not found; installing it into $install_dir" >&2
    mkdir -p "$install_dir"
    if ! curl -LsSf https://astral.sh/uv/install.sh \
         | env UV_INSTALL_DIR="$install_dir" UV_NO_MODIFY_PATH=1 sh >&2; then
        echo "opal: uv install failed. Install by hand with the proxy exported:" >&2
        echo "           curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        return 1
    fi

    export PATH="$install_dir:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "opal: uv installed to $install_dir but is still not on PATH" >&2
        return 1
    fi
    echo "opal: uv $(uv --version 2>/dev/null || echo '?') ready" >&2
    return 0
}

# Report which pushed revision this checkout is, so "stale code on the remote"
# is visible instead of surfacing as a mystery argparse or module error.
opal_report_stamp() {
    local stamp="${OPAL_ROOT:-$PWD}/.opal-push-stamp"
    if [[ -f "$stamp" ]]; then
        echo "  revision  $(grep -m1 '^payload_sha=' "$stamp" | cut -d= -f2) " \
             "(pushed $(grep -m1 '^pushed_at=' "$stamp" | cut -d= -f2))"
    else
        echo "  revision  no push stamp -- this tree may predate polaris/push.sh"
    fi
}
