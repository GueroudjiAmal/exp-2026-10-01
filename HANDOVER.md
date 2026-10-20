# Handover — OPAL attention provenance

Start here. This is the front door; the detail lives in two other documents and
you should not read them first.

| document | what it is | read when |
|---|---|---|
| **HANDOVER.md** (this) | orientation, setup, first result, current status | now |
| [README.md](README.md) | reference: architecture, every arm, every run artifact | when something is unclear |
| [POISONING.md](POISONING.md) | the poisoning study: workflows, arms, findings | before touching Track B |

---

## 1. What this is, in one paragraph

The OPAL benchmark is 100 synthetic multi-round data-analysis investigations
(430 rounds) over a plant-phenotyping study. This harness replays them through
one vLLM engine and captures, **for every generated token, which prompt tokens
it attended to** — recomputed exactly, not estimated — and records it as
Flowcept provenance. The point is to ask whether a model that answers correctly
is answering *for the right reasons*, and whether attention over the prompt
predicts where that model can be attacked.

Capture is done by `vllm-attn-connector`, an out-of-tree vLLM KV connector. It
needs CUDA. Everything else — planning, scoring, analysis, figures, heat maps —
runs on any laptop.

---

## 2. READ THIS BEFORE YOU INSTALL ANYTHING

**The connector on GitHub does not work on Qwen3.5.** Four fixes are required
and none of them is committed upstream. `scripts/gpu_setup.sh` clones from
GitHub, so a fresh setup gets the broken version, and the failure is silent:
you get a complete, structurally valid capture in which **every number is zero
or NaN**, and twelve of the fourteen verification checks still pass.

The patch is in this repo:

```bash
vendor-patches/vllm-attn-connector-hybrid-kv.patch     # 391 lines, applies cleanly to HEAD
```

Apply it to the connector checkout after setup:

```bash
cd <wherever vllm-attn-connector lives>
patch -p1 < <this repo>/vendor-patches/vllm-attn-connector-hybrid-kv.patch
```

What it fixes, all one root cause — *the connector assumed a uniform,
single-group KV cache*, which Qwen3.5 (hybrid full-attention + gated-delta-net,
multimodal config) is not:

1. `tl.arange` over the paged block size — fails when it is not a power of two.
   Qwen3.5's attention page is 1056 tokens, sized to match its GDN state. The
   kernel now walks the block in fixed 128-token tiles.
2. `num_attention_heads` read from the top-level HF config — `None` for a
   multimodal model, so it silently fell back to `num_kv_heads` and scored 2 of
   16 heads. Now reads `hf_text_config`; the fallback warns.
3. Block tables built from `new_block_ids[0]` — hardcoded group 0. Qwen3.5 has
   **four** KV cache groups and full attention is group 3, so the kernel
   indexed the cache with another group's block numbers and produced NaN. Now
   per-group.
4. The logits scratch was `torch.empty`, so anything unwritten surfaced as NaN
   rather than an obvious zero. Now zeroed.

**Better than patching: commit and push these upstream.** They are correct,
tested, and every fresh machine will otherwise re-discover them. The patch is a
stopgap, not the plan.

---

## 3. Setup

Three environments, by what you have.

```bash
make venv          # any laptop. Analysis, figures, dry runs, heat maps. No GPU.
make gpu-venv      # on an NVIDIA box. vLLM 0.28.0 + flowcept + the connector.
                   # Then apply the patch from section 2.
```

For a remote GPU box, put these in your shell profile — they do not survive a
new terminal, and forgetting them is the single most common failure:

```bash
export OPAL_BOX=user@host
export OPAL_BOX_PATH=/absolute/path/to/exp-2026-10-01-opal-attn
```

Polaris has its own path; see [polaris/RUNBOOK.md](polaris/RUNBOOK.md).

Verify the laptop side works before anything else:

```bash
make check          # 5 stages, ~3 min. Must pass.
```

---

## 4. Your first result, end to end

```bash
# laptop
./scripts/box.sh push
./scripts/box.sh sync-connector      # verifies by sha256 against your checkout

# on the box
./scripts/gpu_check.sh               # staged bring-up; stage 3 is the one that matters
./scripts/run_gpu.sh --label smoke --limit 3 \
  --enable-thinking --max-tokens 8192 --max-model-len 32768 \
  --temperature 0.0 --max-num-seqs 8 --cuda-graphs

# anywhere
make analyze RUN=results/<stamp>-smoke
./scripts/heatmap.py results/<stamp>-smoke --chain <a chain in the run> --round 3
```

The model is **Qwen3.5-35B-A3B** and it is the default everywhere — in
`run_experiment.py`, `gpu_check.sh` and `gpu_sizing.py` — so `--model` is only
needed to override it. `--temperature 0.0` is not optional if you want runs to
be comparable: two clean runs at `T=0` with pinned `--max-num-seqs` come back
bit-identical, and nothing downstream is interpretable without that.

**Read these three lines of `analyze` output before any number:**

```
the query probe fired (N layers)          zero -> wrong backend, every value is 0
the capture is NON-ZERO                   every other check passes on zeros
attn_sum totals one softmax unit per step zeros cannot fake this
```

If any of those fails, the run is void regardless of how good the rest looks.
This is not hypothetical — five consecutive Qwen3.5 runs passed everything else
while being entirely zero.

---

## 5. The three tracks

**Provenance.** Capture a run, analyse where attention went.
`./scripts/run_gpu.sh --label x ...` then `make analyze RUN=...`. See README
§"What a run directory contains" and §"Reading the results".

**Reference heat map — historical, not part of the forward plan.**
`make reproduce-reference` reproduces `row_analogue_reasoned.txt`, an artifact
made before this project settled on Qwen3.5. The target therefore PINS
Qwen3-4B rather than inheriting the default, because the reference is 3534
tokens under that vocabulary and 3614 under Qwen3.5's — which is how it was
identified. It is the only place another model appears; leave the pin alone
and do not take it as a model recommendation.

**Poisoning.** `POISONING.md`, workflows 1–3. Already run; results in
`results/2026100[678]-*-poison-*`.

---

## 6. What is established, and what is not

From `POISONING.md`, reproduced here because people quote headline numbers
without reading the caveats:

**Established.**
- The capture is deterministic. Two clean runs at `T=0` with pinned
  `--max-num-seqs` came back bit-identical: 90/90 identical replies, `attn_sum`
  max difference `0.000e+00`. Conditional on TP=1, one GPU, greedy.
- **Efficacy does not track attention density.** The densest span flipped 1 of
  87; a mid-density span flipped 80%. What predicted efficacy was *what the
  poison attacked*, not how much attention its span draws.
- No prefill blind spot on this workload.

**Not established — do not write these up.**
- *Attention detects successful poisoning.* It does not. Split by outcome, the
  turns that actually flipped show +8.6% and −3.9%, at or below the null
  control's +6.5%. The large arm-level deltas came from turns the model
  **resisted**. The signature measures the model *noticing*, not the poison
  *succeeding*.
- *Any single flip rate.* The same poison reads 81.8%, 70.9% or 87.1% depending
  on the denominator. Fix the population before quoting.
- *Anything about the `history` arm's attention* — its substitution breaks token
  parity, so the length filter drops exactly the rounds where it acts.

---

## 7. Traps that have each cost a day

- **Round is confounded with task type.** Round N is almost always the same task
  across chains, so `by_round` is difficulty, not depth. Use `propagation`.
- **Distance is confounded with everything.** Attention density falls 10–17x
  from the nearest to the farthest quartile *within every section kind*. Any
  cross-section comparison must be at matched distance or it is a position
  ranking wearing a content label.
- **55% of attention mass is on the sink and newlines** — position 0 alone is
  38%. Renormalise over content mass, always.
- **Share is not density.** `db_lookup` has the largest share and near-lowest
  per-token density; its share was length.
- **`--max-steps` keeps the FIRST N decode steps.** On a thinking model the
  answer comes last, so capping discards exactly what you want. Use `--top-pct`
  to control size instead.
- **Character-length-preserving is not token-length-preserving.**
  `AGREES`→`DENIES` is 6 characters both ways and 3→2 tokens, which shifts every
  index after it. `build_poison_manifest.py` enforces token parity per chain, and refuses an
  unbalanced substitution.
- **Never reuse a run directory.** Flowcept keeps old dumps and names them with
  a pid, so a second run leaves two buffers while `turns.jsonl` is overwritten.
  `run_gpu.sh` timestamps every run for this reason.
- **Correct ≠ textually identical.** Scoring normalises case, whitespace and
  separator spacing; history insertion does not. At 100% accuracy 89/90 replies
  matched ground truth byte-for-byte — one did not, and that was enough to
  invalidate a planned determinism test.

---

## 8. State of the repo as handed over

- 35 run directories in `results/`, **14 GB**. Provenance runs, the reference,
  the gt/self-history pair, and the full poisoning suite including a 16k arm.
- `results/figures/`, `results/heatmaps/` — generated output.
- The connector patch is **uncommitted upstream** (section 2).
- `make check` passes; `ruff check src scripts` is clean.
- `box.sh push` carries source, config and the benchmark CSV — **not** the
  connector, not `results/`, and not `vendor-patches/`. Use
  `box.sh sync-connector` for the connector and `box.sh pull` for results.
