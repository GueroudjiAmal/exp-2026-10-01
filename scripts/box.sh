#!/bin/bash
# Drive a remote NVIDIA box over ssh: push code, bring it up, run, pull results.
#
#   export OPAL_BOX=user@gpuhost
#   export OPAL_BOX_PATH=/home/user/opal-attn        # optional, this is the default shape
#
#   ./scripts/box.sh push          # rsync source + data (not venvs, not results)
#   ./scripts/box.sh sync-connector # push local vllm-attn-connector edits
#   ./scripts/box.sh setup         # build .venv-gpu there (clones the two siblings)
#   ./scripts/box.sh check         # the staged GPU ladder, ends with 3 real chains
#   ./scripts/box.sh sizing        # what fits on that card, as flags
#   ./scripts/box.sh run [args]    # start the full run DETACHED; survives disconnect
#   ./scripts/box.sh status        # is it still going, and the tail of its log
#   ./scripts/box.sh logs          # follow the log
#   ./scripts/box.sh pull [name]   # bring runs back into ./results/
#   ./scripts/box.sh shell         # ssh in, cd'd and ready
#
# Typical first time:
#   ./scripts/box.sh push && ./scripts/box.sh setup && ./scripts/box.sh check
# Then:
#   ./scripts/box.sh run --limit 100
#   ./scripts/box.sh status
#   ./scripts/box.sh pull
#
# WHY DETACHED. The full 100-chain run is long, and an ssh session that drops
# takes its child processes with it. `run` starts the job under setsid+nohup
# with its own log and pidfile, so a dropped connection costs nothing and
# `status` works from anywhere. This is the piece a scheduler would normally
# provide and a bare GPU box does not.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

BOX="${OPAL_BOX:-}"
BOX_PATH="${OPAL_BOX_PATH:-}"
SSH_OPTS="${OPAL_SSH_OPTS:-}"

usage() { sed -n '2,30p' "$0" | sed 's/^# \?//'; exit "${1:-0}"; }
[[ $# -gt 0 ]] || usage 2
CMD="$1"; shift
[[ "$CMD" == "help" || "$CMD" == "-h" || "$CMD" == "--help" ]] && usage 0

if [[ -z "$BOX" ]]; then
    echo "!! set OPAL_BOX to the machine, e.g.  export OPAL_BOX=user@gpuhost" >&2
    exit 2
fi

# OPAL_BOX is an ssh destination, not a URL. Pasting a browser-shaped address
# is the obvious mistake, and left alone it surfaces three layers down as
# "Could not resolve hostname http" -- because rsync reads everything before
# the first ':' as the host. Strip what is unambiguous, reject the rest here
# where the message can say what to type.
BOX="${BOX#http://}"; BOX="${BOX#https://}"; BOX="${BOX#ssh://}"
BOX="${BOX%/}"
if [[ "$BOX" == */* || "$BOX" == *:* ]]; then
    echo "!! OPAL_BOX=\"${OPAL_BOX}\" is not an ssh destination." >&2
    echo "   It must be  user@host  -- no scheme, no path, no colon." >&2
    echo "   The path goes in OPAL_BOX_PATH instead:" >&2
    echo >&2
    echo "     export OPAL_BOX=${USER}@${BOX%%[:/]*}" >&2
    echo "     export OPAL_BOX_PATH=/home/${USER}/..." >&2
    exit 2
fi
if [[ "$BOX" != *@* ]]; then
    # Legal (ssh falls back to the local username) but almost never intended
    # when the remote account differs, which it usually does.
    echo "note: OPAL_BOX=$BOX has no user; ssh will use '${USER}'." >&2
fi
# Default the remote path to a directory of the same name, RELATIVE to the
# remote home. Relative rather than an absolute /home/<user>/... guess: rsync
# resolves a relative destination against the remote home, and `bash -l` starts
# there too, so this is correct whatever the account's home actually is --
# /home/x, /users/x, /lus/..., or anything else. Naming it after this
# experiment keeps two experiments from landing on top of each other.
BOX_PATH="${BOX_PATH:-$(basename "$HERE")}"
BOX_PATH="${BOX_PATH%/}"

# shellcheck disable=SC2086
rsh() { ssh $SSH_OPTS "$BOX" "$@"; }

# Run a command on the box, fed over STDIN rather than embedded in a quoted
# argument. `ssh host "bash -lc '...'"` needs the command to survive two rounds
# of shell quoting, and any single quote in it -- from a flag, a path, a python
# snippet -- silently truncates the command into something that still runs. The
# heredoc has no such failure mode.
#
# `bash -l` because the venv, CUDA and any module setup live in the user's
# profile, which a non-interactive `ssh host cmd` does not read.
# `-s` so the script arrives on stdin.
remote() {
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "$BOX" "bash -l -s" <<REMOTE_SCRIPT
set -uo pipefail
cd "$BOX_PATH" 2>/dev/null || {
    echo "!! $BOX_PATH does not exist on \$(hostname). Run: ./scripts/box.sh push" >&2
    exit 97
}
$1
REMOTE_SCRIPT
}

# Shell-quote this command's arguments for the remote shell.
#
# `remote "./scripts/run_gpu.sh$(qargs "$@")"` looks right and is not: $* joins with
# spaces and the remote shell re-splits on them, so any argument containing a
# space arrives as several. The reference system prompt is 138 characters of
# prose containing single quotes -- it would arrive word-split and with its
# quoting destroyed. printf %q emits each argument in a form the remote shell
# parses back to exactly the original string.
qargs() {
    local out=""
    local a
    for a in "$@"; do
        out+=" $(printf '%q' "$a")"
    done
    printf '%s' "$out"
}

# Is the far side actually set up? Called before anything that needs the venv,
# so "you have not run setup yet" is said once and plainly instead of arriving
# as a ModuleNotFoundError from four different scripts.
require_venv() {
    local rc=0
    remote 'test -x .venv-gpu/bin/python' || rc=$?
    case "$rc" in
        0) return 0 ;;
        97) exit 1 ;;   # remote() already explained it
        *) echo "!! no .venv-gpu on $BOX. Run: ./scripts/box.sh setup" >&2; exit 1 ;;
    esac
}

LOG="run.log"
PIDFILE="run.pid"

case "$CMD" in

push)
    # Explicit include-list, same reasoning as polaris/push.sh: this directory
    # accumulates venvs, vendored clones and run outputs, and with an
    # exclude-list each new one ships by default until someone notices.
    PAYLOAD=(configs scripts src polaris data README.md Makefile
             requirements-local.txt .gitignore)
    # polaris/ comes along only for check_vllm_api.py, which gpu_setup.sh runs.
    STAMP=".opal-push-stamp"
    {
        echo "pushed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "pushed_from=$(hostname)"
        echo "payload_sha=$(find "${PAYLOAD[@]}" -type f \
            -not -path '*/__pycache__/*' -not -name '*.pyc' \
            -exec sha256sum {} + 2>/dev/null | sort | sha256sum | cut -c1-12)"
    } > "$STAMP"
    echo "==> $BOX:$BOX_PATH"
    # An absolute destination is taken literally, so a path whose leading
    # component is wrong lands somewhere unwritable -- commonly the filesystem
    # root, when a "$HOME/" or "/home/$USER/" prefix was dropped. Say so before
    # rsync does, because its message names mkdir rather than the cause.
    if [[ "$BOX_PATH" == /* && "$BOX_PATH" != /home/* && "$BOX_PATH" != /Users/* \
          && "$BOX_PATH" != /lus/* && "$BOX_PATH" != /scratch/* \
          && "$BOX_PATH" != /data/* && "$BOX_PATH" != /work/* ]]; then
        echo "note: OPAL_BOX_PATH=$BOX_PATH is an absolute path outside the usual" >&2
        echo "      writable roots. If this is not deliberate, drop it entirely --" >&2
        echo "      the default is a directory of this name under the remote home." >&2
    fi
    # shellcheck disable=SC2086
    if ! rsync -azh --relative --mkpath ${SSH_OPTS:+-e "ssh $SSH_OPTS"} "$@" \
        --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/' \
        "${PAYLOAD[@]}" "$STAMP" "$BOX:$BOX_PATH/"; then
        echo >&2
        echo "!! push failed. The usual causes, in order:" >&2
        echo "   - OPAL_BOX_PATH is not writable by you on that host." >&2
        echo "     Unset it to use the default (~/$(basename "$HERE")), or give" >&2
        echo "     a path under your remote home." >&2
        echo "   - no ssh key, so every call prompts. Fix it once:" >&2
        echo "       ssh-copy-id ${BOX}" >&2
        echo "     or reuse one connection:" >&2
        echo "       export OPAL_SSH_OPTS='-o ControlMaster=auto -o ControlPath=/tmp/opal-%r@%h:%p -o ControlPersist=4h'" >&2
        exit 1
    fi
    echo "pushed: $(grep payload_sha "$STAMP" | cut -d= -f2)"
    ;;

sync-connector)
    # push.sh deliberately excludes the sibling repos, because gpu_setup.sh
    # clones them from GitHub -- so a local connector edit never reaches the
    # box through a normal push. That has now caused three rounds of "the fix
    # is not there", each one diagnosed by hand with long absolute paths typed
    # at a prompt. This is that copy, as one command.
    #
    # Local layout:  <parent>/vllm-attn-connector/src/...
    # Box layout:    <BOX_PATH>/vendor/vllm-attn-connector/src/...
    # The nesting differs, which is exactly why relative paths kept failing.
    SRC="$(cd "$HERE/.." && pwd)/vllm-attn-connector/src/vllm_attn_connector"
    DST="${BOX_PATH}/vendor/vllm-attn-connector/src/vllm_attn_connector"
    [[ -d "$SRC" ]] || {
        echo "!! no connector checkout at $SRC" >&2
        echo >&2
        # Almost always this command was run ON the box. It is a PUSH: it
        # copies the local checkout to the box, so it needs the laptop's
        # sibling layout (<parent>/vllm-attn-connector). On the box the
        # connector lives at <BOX_PATH>/vendor/vllm-attn-connector instead,
        # and the parent directory has no sibling at all -- which is exactly
        # the path named above.
        if [[ -d "$HERE/vendor/vllm-attn-connector" ]]; then
            echo "   This looks like the box itself: there is a vendor/ copy at" >&2
            echo "     $HERE/vendor/vllm-attn-connector" >&2
            echo "   sync-connector PUSHES from your laptop to here. Run it there:" >&2
        else
            echo "   sync-connector pushes a local checkout to the box, so run it" >&2
            echo "   from the machine that HAS one, next to this experiment:" >&2
        fi
        echo >&2
        echo "     cd ~/Work/genesis/exp-2026-10-01-opal-attn && ./scripts/box.sh sync-connector" >&2
        exit 1
    }
    echo "==> $SRC"
    echo "    -> $BOX:$DST"
    # shellcheck disable=SC2086
    rsync -avz ${SSH_OPTS:+-e "ssh $SSH_OPTS"} \
        --include='*.py' --exclude='*' \
        "$SRC/" "$BOX:$DST/"
    # Verify by CHECKSUM, not by grepping for marker strings. Markers rot:
    # this check looked for PAD_N, a later rewrite replaced it with tiling, and
    # a correct sync then reported "expect 3, got 0" -- a false alarm that
    # costs exactly as much attention as a real one. A hash compares the files
    # themselves and never needs updating.
    echo
    echo "verifying what landed:"
    ( cd "$SRC" && sha256sum ./*.py ) \
        | sed 's#\./##' > "${TMPDIR:-/tmp}/opal-conn-manifest.$$"
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "$BOX" \
        "cd '$DST' && sha256sum -c --quiet - && echo '  all files match the local checkout'" \
        < "${TMPDIR:-/tmp}/opal-conn-manifest.$$" \
        || echo "!! some files differ from the local checkout (named above)" >&2
    rm -f "${TMPDIR:-/tmp}/opal-conn-manifest.$$"
    ;;

setup)
    # Clones flowcept + vllm-attn-connector into vendor/ on the box, builds
    # .venv-gpu with uv, and verifies vLLM imports against that driver.
    remote "./scripts/gpu_setup.sh$(qargs "$@")"
    ;;

check)
    require_venv
    remote "./scripts/gpu_check.sh$(qargs "$@")"
    ;;

sizing)
    require_venv
    # Generate the plan with the GPU venv explicitly. `make plan` would do it
    # too, but only because the Makefile falls back -- being direct here means
    # this does not depend on that fallback staying in place.
    remote "test -f results/plan/plan.json || \
    .venv-gpu/bin/python scripts/run_experiment.py --dry-run --out-dir results/plan >/dev/null
.venv-gpu/bin/python scripts/gpu_sizing.py$(qargs "$@")"
    ;;

run)
    require_venv
    # Delegates to scripts/run_gpu.sh ON THE BOX rather than reimplementing the
    # launch here. That script owns the timestamped run directory, the
    # environment, the card sizing and the detach; duplicating any of it would
    # mean two launch paths that quietly drift apart.
    remote "./scripts/run_gpu.sh$(qargs "$@")"
    ;;

status)
    remote "if test -f results/latest/run.pid && kill -0 \$(cat results/latest/run.pid) 2>/dev/null; then
    echo \"RUNNING  pid \$(cat results/latest/run.pid)  \$(readlink -f results/latest)\"
else
    echo 'not running'
fi
if test -e results/latest; then
    echo
    cat results/latest/launch.txt 2>/dev/null
    echo
    tail -n 20 results/latest/run.log 2>/dev/null
fi
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || true"
    ;;

logs)
    # `latest` is right here -- the intent is "whatever is running now" -- but
    # print which directory that resolved to, so the output is attributable.
    remote "test -e results/latest || { echo 'no runs yet' >&2; exit 1; }
echo \"== results/\$(readlink results/latest)/run.log\"
tail -f results/latest/run.log"
    ;;

stop)
    remote "if test -f results/latest/run.pid && kill \$(cat results/latest/run.pid) 2>/dev/null; then
    echo stopped
else
    echo 'nothing running'
fi"
    ;;

pull)
    # Bring runs back. Named run, or everything.
    WHAT="${1:-}"
    mkdir -p results

    # PRE-FLIGHT. This is the one command that reaches the box with raw rsync
    # instead of remote(), so it never inherited remote()'s "does BOX_PATH
    # exist" check. Without it rsync reports
    #     change_dir "/home/<user>/exp-2026-10-01-opal-attn/results" failed
    # which names a path nobody chose and does not mention the actual cause:
    # OPAL_BOX_PATH was unset in THIS shell, so the default -- a directory of
    # this name under the remote home -- was used instead. The variable does
    # not survive opening a new terminal, which is exactly when this happens.
    if ! rsh "test -d ${BOX_PATH@Q}/results" 2>/dev/null; then
        echo "!! $BOX:$BOX_PATH/results does not exist." >&2
        echo >&2
        if [[ -z "${OPAL_BOX_PATH:-}" ]]; then
            echo "   OPAL_BOX_PATH is UNSET in this shell, so pull fell back to" >&2
            echo "   the default, relative to your remote home:" >&2
            echo >&2
            echo "       $BOX_PATH" >&2
            echo >&2
            echo "   If the experiment lives somewhere else on the box, set it" >&2
            echo "   and try again -- and put it in ~/.bashrc so the next" >&2
            echo "   terminal has it too:" >&2
            echo >&2
            echo "       export OPAL_BOX_PATH=/path/to/exp-2026-10-01-opal-attn" >&2
        else
            echo "   OPAL_BOX_PATH=$OPAL_BOX_PATH" >&2
            echo "   The path is wrong, or no run has finished there yet" >&2
            echo "   (results/ is created by the first run, not by setup)." >&2
        fi
        echo >&2
        echo "   Checkouts found on the box:" >&2
        rsh "find ~ -maxdepth 5 -type d -name '$(basename "$HERE")' \
             -not -path '*/.*' 2>/dev/null | sed 's|^|       |' \
             || true" >&2
        exit 1
    fi

    # A named run that is not there deserves the list, not a transfer of
    # nothing: rsync exits 0 having copied no files when the source glob
    # matches nothing, which reads as success.
    if [[ -n "$WHAT" ]] && ! rsh "test -e ${BOX_PATH@Q}/results/${WHAT@Q}" 2>/dev/null; then
        echo "!! no run named '$WHAT' in $BOX:$BOX_PATH/results" >&2
        echo "   what is there, newest first:" >&2
        rsh "ls -1t ${BOX_PATH@Q}/results | sed 's|^|       |'" >&2
        exit 1
    fi

    echo "==> pulling from $BOX:$BOX_PATH/results/${WHAT}"
    # shellcheck disable=SC2086
    rsync -azh --partial --info=progress2 ${SSH_OPTS:+-e "ssh $SSH_OPTS"} \
        "$BOX:$BOX_PATH/results/${WHAT:+$WHAT}" results/
    echo
    echo "analyse it:  make analyze RUN=results/<name>"
    ;;

shell)
    rsh -t "cd '$BOX_PATH' && bash -l"
    ;;

*)
    echo "!! unknown command: $CMD" >&2
    usage 2
    ;;
esac
