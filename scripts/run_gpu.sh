#!/bin/bash
# Launch an experiment run on the machine with the GPU. One command, no env to
# remember, a fresh timestamped directory every time.
#
#   ./scripts/run_gpu.sh                      # all 100 chains, detached
#   ./scripts/run_gpu.sh --label smoke --limit 5
#   ./scripts/run_gpu.sh --fg --limit 3       # foreground, watch it
#   ./scripts/run_gpu.sh --limit 50 --model Qwen/Qwen3-14B
#   ./scripts/run_gpu.sh --no-connector       # baseline arm, no capture
#
# Anything it does not recognise is passed through to run_experiment.py, so
# --model, --top-pct, --max-tokens, --max-model-len and friends all work.
#
# This script does NOT build anything. If the venv is missing it says so and
# stops -- building is scripts/gpu_setup.sh's job, and silently installing
# packages from a launcher is how a "run" ends up reinstalling its own
# dependencies.
#
# Each run gets results/<UTC timestamp>[-label]/ and NEVER reuses a directory:
# Flowcept keeps old dumps (delete_previous_file: false) and names them with a
# pid, so a second run in one directory leaves two buffers while turns.jsonl is
# overwritten -- and an analysis can then pair one run's answers with another's
# provenance without erroring.
#
# results/latest is a RELATIVE symlink to the newest run, so it survives being
# copied to another machine. Every message below names the run's real
# directory, because `latest` moves as soon as the next run starts.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

case "${1:-}" in -h|--help)
    sed -n '2,26p' "$0" | sed 's/^# \?//'; exit 0 ;;
esac

GPU_VENV="${GPU_VENV:-${HERE}/.venv-gpu}"
PY="${GPU_VENV}/bin/python"
[[ -x "$PY" ]] || {
    echo "!! no GPU venv at $GPU_VENV" >&2
    echo "   build it first:  ./scripts/gpu_setup.sh" >&2
    exit 1
}

# --- the environment a run needs, so nobody has to remember it ---------------
# PATH: FlashInfer JITs at warmup and execs `ninja`, resolved through PATH;
#       pip installs it into the venv, not system-wide.
# VLLM_ENABLE_V1_MULTIPROCESSING=0: keeps the engine in-process, so the
#       connector emits provenance in-process and Flowcept needs no message
#       queue. With multiprocessing on, records cross a process boundary and a
#       reachable Redis becomes mandatory.
# MAX_JOBS: caps the parallelism of CUDA compilation at warmup. ninja defaults
#       to one nvcc per core (~20 here); each is memory-hungry, and on a
#       UNIFIED-memory part they compete with model weights already resident in
#       the same pool, so the OOM killer takes one and ninja reports "exit
#       status 137". Measured: 2 works with a 57 GiB MoE model; 4 untested.
#       Bounded cost -- FlashInfer caches what it builds, so only the first run
#       with a given model and compute capability compiles at all.
export PATH="${GPU_VENV}/bin:${PATH}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export FLOWCEPT_SETTINGS_PATH="${FLOWCEPT_SETTINGS_PATH:-${HERE}/configs/flowcept_local_gpu.yaml}"
export MAX_JOBS="${MAX_JOBS:-2}"

FG=0
LABEL=""
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fg)    FG=1; shift ;;
        --label) LABEL="$2"; shift 2 ;;
        *)       PASS+=("$1"); shift ;;
    esac
done

# --- refuse to stack runs -----------------------------------------------------
# Two engines on one card means both crawl or both OOM, and on a unified-memory
# part the second can take the host with it.
#
# `kill -0` alone is not enough: run.pid outlives the run, and after a reboot or
# enough process churn that number belongs to something else -- which would
# block every future run with a message naming an innocent process.
is_our_run() {
    local pid="$1"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q run_experiment.py
}

if [[ -L results/latest && -f results/latest/run.pid ]] \
   && is_our_run "$(cat results/latest/run.pid 2>/dev/null)"; then
    ACTIVE="results/$(readlink results/latest)"
    echo "!! a run is already going: ${ACTIVE}" >&2
    echo "   pid $(cat results/latest/run.pid)" >&2
    echo "   tail -f ${ACTIVE}/run.log        to watch" >&2
    echo "   kill \$(cat ${ACTIVE}/run.pid)   to stop" >&2
    exit 1
fi

RUN_ID="$(date -u +%Y%m%d-%H%M%SZ)${LABEL:+-$LABEL}"
RUN="${HERE}/results/${RUN_ID}"
REL="results/${RUN_ID}"
mkdir -p "$RUN"

# --- size the card, unless the caller already decided -------------------------
# gpu_sizing reads results/plan/plan.json (the dry run) and this device's real
# memory, and prints the flags that fit. Doing it here means the numbers follow
# the hardware rather than being pasted from another machine.
has_flag() { [[ " ${PASS[*]:-} " == *" $1 "* ]]; }

# The plan must be built with THIS RUN's model and wrapper, not the default.
# It is a tokenization: prompt lengths come from the model's own tokenizer, and
# Qwen3.5's vocabulary is 248044 against Qwen3's 151643 -- the same
# conversation is 3614 tokens on one and 3534 on the other. A plan built with
# the wrong tokenizer feeds gpu_sizing wrong prompt lengths, so --max-num-seqs
# is sized for a workload that does not exist.
#
# It also runs the dry run's checks against the real template. One of those --
# "answer segments track conversation depth" -- catches a chat template whose
# assistant marker the range regex mishandles, which is exactly how a Qwen3.5
# generation prompt got mis-declared as an answer segment. Catching that here
# costs two minutes of CPU; catching it after a capture costs the capture.
MODEL_ARG=()
SYS_ARG=()
for i in "${!PASS[@]}"; do
    [[ "${PASS[$i]}" == "--model" ]] && MODEL_ARG=(--model "${PASS[$((i+1))]}")
    [[ "${PASS[$i]}" == "--system-prompt" ]] && SYS_ARG=(--system-prompt "${PASS[$((i+1))]}")
done
# One plan per (model, wrapper), so switching models does not silently reuse
# the previous one's token counts.
# Keyed on the CSV's CONTENT too, not just the model and wrapper. A poisoning
# arm runs the same model over a MODIFIED benchmark: same path, different
# bytes. Without the content hash the clean arm's cached plan would be reused
# for the poisoned one, and its token counts would describe the wrong prompts
# -- silently, because the file name never changed.
CSV_ARG=()
for i in "${!PASS[@]}"; do
    [[ "${PASS[$i]}" == "--csv" ]] && CSV_ARG=("${PASS[$((i+1))]}")
done
CSV_PATH="${CSV_ARG[0]:-${HERE}/data/opal_chains.csv}"
CSV_HASH="$(sha256sum "$CSV_PATH" 2>/dev/null | cut -c1-12)"
PLAN_KEY="$(printf '%s|%s|%s' "${MODEL_ARG[1]:-default}" "${SYS_ARG[1]:-}" \
            "${CSV_HASH:-nocsv}" | sha256sum | cut -c1-12)"
PLAN_DIR="results/plan/${PLAN_KEY}"
if [[ ! -f "${PLAN_DIR}/plan.json" ]]; then
    echo "==> planning for ${MODEL_ARG[1]:-the default model} (tokenizes 430 turns, ~2 min)"
    if ! "$PY" scripts/run_experiment.py --dry-run --out-dir "$PLAN_DIR" \
            "${MODEL_ARG[@]}" "${SYS_ARG[@]}"; then
        echo "!! the dry run reported failures for this model. Those are about" >&2
        echo "   the prompts and the chat template, not the GPU, and they will" >&2
        echo "   not fix themselves during a capture. Refusing to start." >&2
        # The run directory was created before planning; it is empty and would
        # otherwise sit in results/ looking like a run that produced nothing.
        rm -rf "$PLAN_DIR"
        rmdir "$RUN" 2>/dev/null || true
        exit 1
    fi
fi

SIZING="$("$PY" scripts/gpu_sizing.py "${MODEL_ARG[@]}" --plan "${PLAN_DIR}/plan.json" \
          ${OPAL_UTIL:+--util "$OPAL_UTIL"} 2>/dev/null || true)"
auto_of() { awk -v f="$1" '$1==f {print $2}' <<<"$SIZING" | head -1; }

if ! has_flag --max-num-seqs; then
    N="$(auto_of --max-num-seqs)"; [[ -n "$N" ]] && PASS+=(--max-num-seqs "$N")
fi
if ! has_flag --gpu-memory-utilization; then
    U="${OPAL_UTIL:-$(auto_of --gpu-memory-utilization)}"
    [[ -n "$U" ]] && PASS+=(--gpu-memory-utilization "$U")
fi
# CUDA graphs on unless the sizing said the pools will not fit, or the caller
# already chose. The connector's e2e covers the graph path either way, so this
# is a speed decision, not a correctness one.
if ! has_flag --cuda-graphs && ! grep -q "CUDA graphs left OFF" <<<"$SIZING"; then
    PASS+=(--cuda-graphs)
fi

CMD=("$PY" scripts/run_experiment.py
     --csv "${HERE}/data/opal_chains.csv"
     --out-dir "$RUN"
     --workflow-id "opal-${RUN_ID}"
     "${PASS[@]}")

{
    echo "run_id=${RUN_ID}"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname)"
    echo "max_jobs=${MAX_JOBS}"
    echo "flowcept=${FLOWCEPT_SETTINGS_PATH}"
    echo "cmd=${CMD[*]}"
} > "$RUN/launch.txt"

# RELATIVE: an absolute link records the path on the machine that made it, so
# after an rsync it points into the other filesystem and resolves to nothing.
ln -sfn "$RUN_ID" "${HERE}/results/latest"

echo "==> ${REL}"
grep -E '^detected:' <<<"$SIZING" | sed 's/^/    /' || true
echo "    ${CMD[*]}"
echo

if [[ "$FG" == "1" ]]; then
    exec "${CMD[@]}"
fi

# setsid detaches from this shell's process group so a dropped ssh connection
# cannot signal it; nohup covers the hangup itself.
setsid nohup "${CMD[@]}" > "$RUN/run.log" 2>&1 < /dev/null &
echo $! > "$RUN/run.pid"
sleep 3
# A run that dies on startup -- bad flag, OOM at load, a missing model -- looks
# exactly like one that started fine until you go looking.
if kill -0 "$(cat "$RUN/run.pid")" 2>/dev/null; then
    cat <<MSG
started, pid $(cat "$RUN/run.pid")

  tail -f ${REL}/run.log
  kill \$(cat ${REL}/run.pid)                  # to stop it

when it finishes:
  ${GPU_VENV}/bin/python scripts/analyze.py ${REL}

(results/latest also points here, but it moves when the next run starts --
 use the path above for anything that must still mean this run.)
MSG
else
    echo "!! it exited immediately. Last lines of ${REL}/run.log:" >&2
    tail -n 30 "$RUN/run.log" >&2
    exit 1
fi
