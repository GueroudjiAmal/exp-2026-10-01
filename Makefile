# OPAL chain attention-provenance experiment.
#
#   make venv         build the local CPU venv
#   make check        the whole local arm (no GPU needed)
#   make plan         dry run over all 100 chains -> results/plan/plan.json
#   make push         rsync source+config+CSV to Polaris
#   make analyze RUN=results/12345     report on a pulled run
#   make heatmap RUN=results/12345 CHAIN=exp32-chain-0020-hard ROUND=5
#   make reproduce-reference        the reference heat map (GPU; see README)
#   make push-results RUN=results/12345 TO=user@host:/data/opal
#
# Driving a REMOTE NVIDIA box over ssh (export OPAL_BOX=user@host first):
#   make box-push / box-setup / box-check / box-run / box-status / box-pull
#
# On a machine WITH an NVIDIA GPU:
#   make gpu-venv     build .venv-gpu with vLLM + the connector (CUDA)
#   make gpu-sizing   what fits on this card, as flags
#   make gpu-check    staged bring-up, ends with 3 real chains
#   make gpu-run      the full 100-chain run, locally (new timestamped dir)
#
# `make run` does not exist on purpose: on a CPU-only box there is nothing to
# run, and on a GPU box `make gpu-check` should pass first.

SHELL := /bin/bash
HERE  := $(shell pwd)
GEN   := $(shell cd .. && pwd)
# The CPU venv when there is one, else the GPU venv, else bare python3. A box
# that only ever ran gpu_setup.sh has no .venv at all, and `make plan` there
# must still work -- it is what gpu_sizing.py reads.
PY    := $(shell if [ -x "$(HERE)/.venv/bin/python" ]; then echo "$(HERE)/.venv/bin/python"; \
                 elif [ -x "$(HERE)/.venv-gpu/bin/python" ]; then echo "$(HERE)/.venv-gpu/bin/python"; \
                 else echo python3; fi)
GPUPY := $(HERE)/.venv-gpu/bin/python

export PYTHONPATH := $(HERE)/src:$(GEN)/vllm-attn-connector/src:$(GEN)/flowcept/src
export FLOWCEPT_SETTINGS_PATH ?= $(HERE)/configs/flowcept_local.yaml

.PHONY: help venv check plan smoke tests push push-results analyze heatmap reproduce-reference clean \
        gpu-venv gpu-sizing gpu-check gpu-run \
        box-push box-setup box-check box-sizing box-run box-status box-logs box-pull

help:
	@awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' Makefile

venv:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements-local.txt
	$(PY) -m pip install -q -e ../flowcept[telemetry]
	@echo "venv ready: $$($(PY) -V)"

check:
	./scripts/local_check.sh

plan:
	$(PY) scripts/run_experiment.py --dry-run --out-dir results/plan

smoke:
	$(PY) scripts/flowcept_smoke.py

tests:
	$(PY) -m pytest tests -q
	$(PY) -m pytest $(GEN)/vllm-attn-connector/tests/test_aggregations.py -q

push:
	./polaris/push.sh

analyze:
	@test -n "$(RUN)" || { echo "usage: make analyze RUN=results/<jobid>" >&2; exit 2; }
	$(PY) scripts/analyze.py "$(RUN)"

# One turn, drawn: per-line heat with a column per answer item. CHAIN and
# ROUND pick the turn; everything else passes through to the script.
heatmap:
	@test -n "$(RUN)" || { echo "usage: make heatmap RUN=results/<jobid> CHAIN=<id> ROUND=<n>" >&2; exit 2; }
	$(PY) scripts/heatmap.py "$(RUN)" $(if $(CHAIN),--chain "$(CHAIN)") $(if $(ROUND),--round "$(ROUND)") $(OPAL_ARGS)

# TO defaults to $OPAL_RESULTS_REMOTE when unset; --verify checks the transfer
# against the sha256 the connector recorded, not against a fresh local hash.
push-results:
	@test -n "$(RUN)" || { echo "usage: make push-results RUN=results/<jobid> [TO=host:/path]" >&2; exit 2; }
	./scripts/push_results.sh "$(RUN)" $(if $(TO),--to "$(TO)") --verify

# ---- local NVIDIA GPU -------------------------------------------------------
# Separate venv: CUDA torch and the CPU torch in .venv cannot coexist, and
# keeping them apart means the dry run still works if the GPU goes away.

gpu-venv:
	./scripts/gpu_setup.sh

gpu-sizing: plan
	$(GPUPY) scripts/gpu_sizing.py || $(PY) scripts/gpu_sizing.py

gpu-check:
	./scripts/gpu_check.sh

# Deliberately NOT depending on gpu-check: that would re-run a 3-chain capture
# every time. Run `make gpu-check` once after setup, then this.
# One launcher for every run, local or remote: scripts/run_gpu.sh. It makes a
# fresh timestamped directory, sizes the card, sets the environment and
# detaches. OUT is no longer a knob -- reusing a directory is what mixes two
# runs' provenance.
gpu-run:
	./scripts/run_gpu.sh $(OPAL_ARGS)

# ---- remote NVIDIA box over ssh ---------------------------------------------
# All of these need OPAL_BOX=user@host; OPAL_BOX_PATH defaults to a directory
# named after this experiment under the remote user's home.

box-push:   ; ./scripts/box.sh push
box-setup:  ; ./scripts/box.sh setup
box-check:  ; ./scripts/box.sh check
box-sizing: ; ./scripts/box.sh sizing
box-run:    ; ./scripts/box.sh run $(OPAL_ARGS)
box-status: ; ./scripts/box.sh status
box-logs:   ; ./scripts/box.sh logs
box-pull:   ; ./scripts/box.sh pull $(RUN)

# Reproduce the reference heat map (row_analogue_reasoned.txt). Needs a GPU.
# This is a DEVIATION from the default protocol, and deliberately spelled out
# rather than hidden behind a boolean: the wrapper below is 34 tokens and
# shifts every token index in every conversation, and ground-truth history
# means no mistake can propagate. Both are properties of the reference
# artifact, not of this experiment's headline configuration.
OPAL_REFERENCE_SYSTEM := You are answering a self-contained data-analysis question. Work only from the text given. Reply with exactly one line, starting 'ANSWER:'.

# The model is PINNED, not inherited from DEFAULT_MODEL. The reference is
# Qwen3 line, not Qwen3.5: the same text tokenizes to 3534 tokens under
# Qwen3-4B-Instruct-2507 (vocab 151643) and 3614 under Qwen3.5-35B-A3B (vocab
# 248044), and 3534 is what the reference reports. Inheriting the default
# would silently stop reproducing it the day the default moves.
OPAL_REFERENCE_MODEL := Qwen/Qwen3-4B-Instruct-2507

reproduce-reference:
	./scripts/run_gpu.sh --label reference \
	  --model "$(OPAL_REFERENCE_MODEL)" \
	  --history ground-truth \
	  --system-prompt "$(OPAL_REFERENCE_SYSTEM)" \
	  --chain-ids exp32-chain-0020-hard $(OPAL_ARGS)

clean:
	rm -rf results/local-check-small results/dryrun-smoke
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
