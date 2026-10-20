# Resolve (and if needed, fetch) the two checkouts this experiment builds on.
# Sourced by gpu_setup.sh and gpu_check.sh. Never executed.
#
# Two situations, and they must both work from the same script:
#
#   IN THE GENESIS WORKSPACE -- ../flowcept and ../vllm-attn-connector are
#   already there, usually with local edits. Use them as-is. Cloning over them
#   would discard work.
#
#   ON A BARE REMOTE BOX -- only this experiment directory was rsynced. The
#   siblings have to come from GitHub, and they go into `vendor/` INSIDE the
#   experiment directory rather than next to it: the parent may not be writable,
#   and scattering clones around someone's $HOME is rude. It also keeps the whole
#   experiment removable with one `rm -rf`.
#
# Exports OPAL_FLOWCEPT_DIR and OPAL_CONNECTOR_DIR (repo roots), and
# OPAL_FLOWCEPT_SRC / OPAL_CONNECTOR_SRC (their src/ dirs, for PYTHONPATH).

OPAL_FLOWCEPT_REPO="${OPAL_FLOWCEPT_REPO:-https://github.com/spotter-ai-genesis/flowcept.git}"
OPAL_CONNECTOR_REPO="${OPAL_CONNECTOR_REPO:-https://github.com/spotter-ai-genesis/vllm-attn-connector.git}"

# opal_resolve_siblings [--clone]
#   Without --clone, report what is present and return 1 if anything is missing.
#   With --clone, fetch whatever is missing into vendor/ first.
opal_resolve_siblings() {
    local want_clone=0
    [[ "${1:-}" == "--clone" ]] && want_clone=1

    local here parent vendor
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    parent="$(cd "$here/.." && pwd)"
    vendor="${OPAL_VENDOR_DIR:-$here/vendor}"

    local missing=0
    local name repo dir var_dir var_src
    for name in flowcept vllm-attn-connector; do
        if [[ "$name" == "flowcept" ]]; then
            repo="$OPAL_FLOWCEPT_REPO"; var_dir=OPAL_FLOWCEPT_DIR; var_src=OPAL_FLOWCEPT_SRC
        else
            repo="$OPAL_CONNECTOR_REPO"; var_dir=OPAL_CONNECTOR_DIR; var_src=OPAL_CONNECTOR_SRC
        fi

        if [[ -d "$parent/$name/src" ]]; then
            dir="$parent/$name"                       # the genesis workspace
        elif [[ -d "$vendor/$name/src" ]]; then
            dir="$vendor/$name"                       # cloned here previously
        elif [[ "$want_clone" == "1" ]]; then
            echo "==> cloning $name into $vendor" >&2
            mkdir -p "$vendor"
            if ! git clone --depth 1 "$repo" "$vendor/$name" >&2; then
                echo "!! could not clone $repo" >&2
                echo "   If the box has no outbound access, rsync the checkout" >&2
                echo "   yourself:  rsync -az ../$name BOX:$vendor/" >&2
                return 1
            fi
            dir="$vendor/$name"
        else
            echo "!! $name not found (looked in $parent/$name and $vendor/$name)" >&2
            missing=1
            continue
        fi
        export "$var_dir=$dir"
        export "$var_src=$dir/src"
    done
    return "$missing"
}

# One line per checkout: where it came from and what revision it is. Running
# against stale or unexpected code is the confusing failure here, same as the
# push stamp on Polaris -- so make it visible rather than discoverable.
opal_report_siblings() {
    local dir
    for dir in "${OPAL_FLOWCEPT_DIR:-}" "${OPAL_CONNECTOR_DIR:-}"; do
        [[ -n "$dir" ]] || continue
        local rev dirty
        rev="$(git -C "$dir" rev-parse --short HEAD 2>/dev/null || echo '?')"
        dirty=""
        [[ -n "$(git -C "$dir" status --porcelain 2>/dev/null)" ]] && dirty=" (dirty)"
        printf '  %-22s %s  %s%s\n' "$(basename "$dir")" "$rev" "$dir" "$dirty"
    done
}

# setuptools >= 77 (PEP 639) refuses a pyproject that carries BOTH a license
# expression (`license = "MIT"`) and a legacy `License ::` classifier:
#
#   InvalidConfigError: License classifiers have been superseded by license
#   expressions. Please remove: License :: OSI Approved :: MIT License
#
# vllm-attn-connector currently has both, so an editable install fails on any
# box whose setuptools is current -- which a fresh uv venv always is. The
# upstream fix is one deleted line; until that lands, drop the classifier from
# the local copy.
#
# Only ever touches a clone under vendor/, which setup created and owns. A
# checkout the user keeps elsewhere (the genesis workspace) is reported and
# left alone -- silently editing someone's working tree is not setup's business.
# Pass --owned when the CALLER created the clone (polaris/setup.sh clones into
# $OPAL_ROOT directly, with no vendor/ in the path, so the path alone cannot
# tell us). Without it, only a vendor/ clone is considered ours to edit.
opal_fix_pep639_license() {
    local dir="$1" pyproject="$1/pyproject.toml" owned=0
    [[ "${2:-}" == "--owned" ]] && owned=1
    [[ "$dir" == */vendor/* ]] && owned=1
    [[ -f "$pyproject" ]] || return 0
    grep -q 'License :: OSI Approved' "$pyproject" || return 0
    grep -qE '^license *= *"' "$pyproject" || return 0

    if [[ "$owned" == "1" ]]; then
        echo "==> $(basename "$dir"): removing the legacy License classifier" >&2
        echo "    (PEP 639: setuptools >= 77 rejects it alongside license = \"...\")" >&2
        sed -i '/License :: OSI Approved/d' "$pyproject"
    else
        cat >&2 <<MSG
!! $pyproject declares both a license expression and a legacy
   'License :: OSI Approved' classifier. setuptools >= 77 rejects that, so the
   editable install below will fail. This checkout is yours, so fix it there:

     sed -i '/License :: OSI Approved/d' "$pyproject"

   and commit it, so clones on other machines get it too.
MSG
    fi
}

# Flowcept's NVIDIA telemetry calls nvmlDeviceGetMemoryInfo() unconditionally,
# before checking whether "used" was even requested
# (telemetry_capture.py __get_gpu_info_nvidia). On a unified-memory part --
# GB10 / DGX Spark, GH200 -- that call raises NVMLError_NotSupported, because
# there is no discrete framebuffer to report. The exception is caught and
# logged, so it is not fatal, but it fires on EVERY capture (once per task, so
# 430 times in a full run) and it takes temperature, power, name and id down
# with it even though all four work fine on that hardware.
#
# Making the call lazy fixes both: ask for memory only when "used" is
# configured. Idempotent, and ownership rules match opal_fix_pep639_license.
opal_fix_nvml_memory_probe() {
    local dir="$1" owned=0
    [[ "${2:-}" == "--owned" ]] && owned=1
    [[ "$dir" == */vendor/* ]] && owned=1
    local f="$dir/src/flowcept/flowceptor/telemetry_capture.py"
    [[ -f "$f" ]] || return 0
    grep -q '^        nvidia_info = nvmlDeviceGetMemoryInfo(device)$' "$f" || return 0

    if [[ "$owned" != "1" ]]; then
        echo "!! $f calls nvmlDeviceGetMemoryInfo() unconditionally, which fails" >&2
        echo "   on unified-memory GPUs and disables all GPU telemetry there." >&2
        echo "   This checkout is yours; patch and commit it if you want the fix." >&2
        return 0
    fi

    echo "==> flowcept: making the NVML memory query lazy (unified-memory GPUs)" >&2
    python3 - "$f" <<'PYFIX'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
s = p.read_text()
before = """        device = nvmlDeviceGetHandleByIndex(gpu_ix)
        nvidia_info = nvmlDeviceGetMemoryInfo(device)
        flowcept_gpu_info = {}

        if "used" in gpu_conf:
            flowcept_gpu_info["used"] = nvidia_info.used
"""
after = """        device = nvmlDeviceGetHandleByIndex(gpu_ix)
        flowcept_gpu_info = {}

        # Queried only when asked for: on unified-memory parts (GB10, GH200)
        # nvmlDeviceGetMemoryInfo raises NVMLError_NotSupported, and calling it
        # unconditionally loses every other metric too.
        if "used" in gpu_conf:
            flowcept_gpu_info["used"] = nvmlDeviceGetMemoryInfo(device).used
"""
if before not in s:
    sys.exit("flowcept telemetry_capture.py does not match the expected text; not patching")
p.write_text(s.replace(before, after))
print("   patched", p)
PYFIX
}
