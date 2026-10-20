# Attention provenance under poisoning — how to use this

A prototype for one question: **when a model answers a multi-round question,
can the attention it paid tell you whether its evidence was tampered with?**

It runs the OPAL benchmark through vLLM, records which prompt tokens every
generated token attended to, then deliberately corrupts one span of the input
and measures what moved. Everything below is additive to the capture
experiment described in [README.md](README.md); read that first for the
capture itself.

---

## What the thing actually does

**The benchmark.** `data/opal_chains.csv` is 100 synthetic multi-round
investigations, 430 rounds total. Each round pastes in some tool outputs —
data tables, a quality-control table, a schema dictionary — states rules, and
asks for one line back. The questions are built so a shortcut gives a
plausible wrong answer: the highest measurement is usually disqualified by a
quality gate, and the CSV's `explanation` column names which wrong answer each
shortcut produces. Later rounds refer to *"the dose you identified in round 1"*
without repeating it, so the model must carry its own answers forward.

**The capture.** `vllm-attn-connector` records, per decode step, how much
attention each prompt token received. Two products matter:

- `attn_sum` — one number per prompt token, summed over all decode steps,
  recorded **before** the top-10% filter. Exact and unbiased. `sum(attn_sum)`
  equals the number of decode steps, which is a free integrity check.
- `topk_pos` / `val_all_max` — per-step detail, but only the strongest 10%
  of positions within each declared segment.

Use `attn_sum` for any aggregate. Use the per-step arrays only when you need
step resolution and can live with the selection.

**The poisoning.** One span of round 1 is replaced with text that tokenizes to
the *same number of tokens*, so prompt length and batch shape are unchanged.
The run is then paired against an untouched twin.

---

## Setup

```bash
make venv            # CPU venv: analysis, figures, dry runs. No GPU needed.
make gpu-venv        # on the GPU box: vLLM + the connector
```

Analysis runs anywhere. Only capture needs CUDA.

---

## Workflow 1 — capture a run

```bash
./scripts/run_gpu.sh --label mylabel \
  --model Qwen/Qwen3.5-35B-A3B \
  --enable-thinking --max-tokens 8192 --max-model-len 32768 \
  --temperature 0.0 --max-num-seqs 8 --limit 20 --cuda-graphs
```

Then, anywhere:

```bash
make analyze RUN=results/<stamp>-mylabel
```

Read these three lines before anything else:

- `the query probe fired (N layers)` — zero means the engine chose a backend
  the probe is not on and **every captured value is 0**
- `the capture is NON-ZERO` — every other structural check passes on zeros
- `attn_sum totals one softmax unit per decode step` — zeros cannot fake this

The analysis also prints attention density per segment kind, banded by
distance from the decode point. Read that table, not the raw shares — see
*Four normalisations* below.

## Workflow 2 — poison a run

```bash
# 1. build the manifests (no GPU; seconds)
python scripts/build_poison_manifest.py --limit 20 --model Qwen/Qwen3.5-35B-A3B

# 2. pre-flight every arm, still no GPU
DRY=1 ./scripts/run_poison_suite.sh

# 3. the suite: clean twin + every arm, sequential, capture on
./scripts/run_poison_suite.sh
./scripts/run_poison_suite.sh clean schema qc      # or a subset
```

Step 1 writes `configs/poison/<arm>.json`, one entry per chain, and refuses
any substitution that changes the token count. Step 2 re-checks that against
the model you are about to run — a manifest built for one tokenizer need not
balance under another.

Coverage is not uniform and that is the data: seven of the first twenty chains
have no quality gate in round 1, so the `qc`, `rules` and `schema` arms cannot
apply to them. Each manifest records its own chains; `coverage.json` has the
all-arms intersection.

## Workflow 3 — analyse the poisoning

```bash
python scripts/poison_report.py results/poison-suite-<stamp>.json \
       --floor <a-second-clean-run>
python scripts/poison_deep.py   results/poison-suite-<stamp>.json
python scripts/poison_figures.py results/poison-suite-<stamp>-report.json \
       --matrix exp109-chain-0011-hard:1
./scripts/heatmap_suite.sh results/poison-suite-<stamp>.json \
       exp32-chain-0020-hard 5
```

| script | what it answers |
|---|---|
| `poison_report.py` | did answers change, and did attention on the poisoned span move — paired per turn, with a per-round breakdown. `--floor` takes a second clean run and reports the nondeterminism baseline. |
| `poison_deep.py` | whole-prompt divergence, entropy shift, step-0 divergence, thinking cost, and a looked × flipped table |
| `poison_figures.py` | SVG. `--matrix` is the one to use: every run side by side, rows are prompt lines |
| `heatmap_suite.sh` | the per-turn ASCII chart, one file per run, same turn, so a `diff` is the poison's effect |

---

## The arms

Round 1 always asks where a measurement peaks. Answering needs three steps:
pick the right trait (schema table), find its maximum (data table), discard
any dose below the usable-plant floor (QC table). Each arm attacks one.

| arm | edit | attacks |
|---|---|---|
| `clean` | nothing | the paired baseline |
| `qc` | `floor for a panel claim is 12.` → `01.` | removes the quality gate |
| `rules` | `must clear the usable-plant floor` → `may ignore …` | the same corruption as an instruction |
| `schema` | trait `entity_level` `plant` → `area` | hides which trait is the right one |
| `outfmt` | the `ANSWER:` template line | the densest-attention span |
| `history` | round 1's assistant turn → a wrong answer | plants a false memory for later rounds |
| `null` | an experiment ID nothing depends on | false-positive control |

`qc` and `rules` flip ~80% of answers. `schema`, `outfmt` and `null` flip
essentially none.

**Two arms are weaker than they look.** The token-parity gate rejected every
aggressive `outfmt` candidate and selected `dose` → `peak`, which means the
same thing — so that arm never tested what it was designed to. And `history`
substitutes `ANSWER: 0` everywhere, the lowest dose in every series and
conspicuous. Rebuild both before citing them.

---

## Four normalisations, each of which replaced a wrong answer

Every one of these was discovered by a result that did not survive it.

1. **Mass from `attn_sum`, not the per-step arrays.** Selection inflates
   declared segments, because each gets a guaranteed one-position quota.
2. **Renormalise over content.** Position 0 (the attention sink), blank lines
   and chat markers carry no information and take **15%** of the mass on
   Qwen3.5-35B-A3B and **55%** on Qwen3-4B. A raw share compares sink
   behaviour as much as attention.
3. **Divide by span length.** Otherwise you rank spans by size. This is what
   killed "the SESSION briefing is the most-attended prose" — it was 140
   tokens long, not specially attended.
4. **Band by distance from the decode point.** Density falls **10–18×** from
   the nearest instance of a span kind to the farthest, which swamps every
   between-kind difference. Compare at matched distance or not at all.

---

## Comparing two runs: never by index

Token parity holds for the injected span, but `--history model` puts the
model's own replies into later prompts — and once a poison flips an answer,
those replies differ in length and every index after them shifts. Measured on
one chain: rounds 1–3 stayed aligned by luck, round 4 came out 2652 vs 2642
tokens.

Line index is not safe either. It held there because each divergent answer was
one line, but a reply that wraps differently changes the line count and
silently shifts every row after it.

So correspondence is **established, not assumed**:

```python
from opal_attn.report import compare_turn
r = compare_turn(attn_clean, lines_clean, attn_pois, lines_pois)
r["moved"]           # total |change| over lines whose text is IDENTICAL
r["changed_pairs"]   # corresponding position, different text
r["n_only_a"]        # no counterpart — never compared
```

`align_prompts` diffs the two line sequences by text, so it survives
insertions and deletions, not only equal-length substitutions.

---

## What the prototype has shown, and what it has not

**Established.**

- The capture is deterministic. Two clean runs at `T=0` with pinned
  `--max-num-seqs` came back bit-identical — 90/90 identical replies,
  `attn_sum` max difference `0.000e+00`. So any measured change is signal.
  (Conditional on TP=1, one GPU, greedy. It will not survive TP>1.)
- Efficacy does **not** track attention density. The densest span flipped
  1 of 87; a mid-density one flipped 80%. What predicted efficacy was *what
  the poison attacked*, not how much attention its span draws.
- No prefill blind spot on this workload: no answer ever changed without the
  poisoned span being retained at some decode step.

**Not established — do not write these up as results.**

- *Attention detects successful poisoning.* It does not. Split by outcome, the
  turns that actually flipped show **+8.6%** (`qc`) and **−3.9%** (`rules`) —
  at or below the `null` control's +6.5%. The large arm-level deltas came from
  the turns the model **resisted** (+52%) and the ones it deliberated to
  exhaustion (+31%). The signature measures the model *noticing*, not the
  poison *succeeding*, and those are inversely related.
- *Any single flip rate.* The same `qc` poison reads 81.8%, 70.9% or 87.1%
  depending on which turns are in the denominator. Truncation is
  poison-dependent, so arms are not compared on equal footing. Fix the
  population before quoting a number.
- *Anything about the `history` arm's attention.* Its substitution breaks
  token parity, so the length filter drops exactly the rounds where it acts.

---

## Traps worth knowing

- **`--max-steps` keeps the FIRST N decode steps.** On a thinking model those
  are the reasoning; the `ANSWER:` tokens come last. It is the obvious way to
  shrink tensors and it discards exactly what a heat map aggregates over. Use
  `--top-pct` instead.
- **Truncated replies are not answers.** A reply cut off mid-reasoning gets a
  `truncated` verdict from `finish_reason`, never a scraped one — a reasoning
  model quotes the output format back to itself (`` the line should read
  `ANSWER: 0; DISAGREES` ``) and the regex will find that. Before this existed
  a 90-turn run reported 0.8778 accuracy where the completed replies were
  78/78.
- **Never reuse a run directory.** Flowcept keeps old dumps and names them by
  pid, so a second run leaves two buffers while `turns.jsonl` is overwritten.
- **Pin `--max-num-seqs` across arms.** vLLM's numerics depend on batch
  composition; greedy decoding does not protect you.
- **Thinking mode changes what a heat map means.** With reasoning on, the
  answer items are a handful of steps at the end of a ~2400-step trace, so the
  chart describes the commitment moment, not the derivation.
