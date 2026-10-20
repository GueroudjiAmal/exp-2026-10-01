# OPAL chains → attention provenance

**2026-10-01.** Replay the OPAL benchmark's 100 multi-round chains through one
vLLM engine and capture, for every generated token, which prompt tokens it
attended to — as Flowcept provenance.

The question the setup is built to answer: **when a model answers round 5 of an
investigation, is it looking at round 5's evidence, or at its own earlier
answers?** Each chain's history is the model's *own* replies, so a mistake at
round 2 is in the context at round 5, and attention over the history is
measurable rather than assumed.

Two arms, deliberately unequal:

| arm | where | what it proves |
|---|---|---|
| **CPU** | this laptop, no NVIDIA device | everything up to the GPU: chains, tokenization, segments, context budget, the Flowcept round-trip, the join |
| **local GPU** | any NVIDIA box | the real thing, at whatever scale the card allows |
| **Polaris** | 1 node, 1×A100 | the real thing, all 100 chains |

The local-GPU and Polaris arms run **the same code with the same flags**. The
only differences are where the venv comes from (uv either way) and that Polaris
adds modules, PBS and offline pinning.

---

## Pieces

```
src/opal_attn/
    chains.py       load opal_chains.csv, group into chains, select subsets
    ranges.py       declare attention segments in TOKEN space  <- easy to get silently wrong
                    and the per-line token table a heat map is drawn on
    answers.py      parse a reply, score it, name the documented shortcut it took
    runner.py       the engine driver: round-synchronous batching across chains
    plan.py         the dry run -- everything the real run does, minus the GPU
    report.py       verify a capture, then report what it shows
    provenance.py   read the connector's SafeTensors without needing vLLM
    poison.py       corrupt one span, token-length preserving -- POISONING.md installed
scripts/
    run_experiment.py   the run, and `--dry-run` the plan
    analyze.py          verify + report a finished run directory, on any machine
    failures.py         where a run failed, and whether it looked at the evidence
    heatmap.py          ONE turn, drawn: per-line heat, a column per answer item
    heatmap_suite.sh    the same turn for every run of a suite, one file each
    mismatch.py         turns where runner and connector disagree on length
    debug_attention.py  why a report came back empty: inspect one run's tensors
poisoning -- see POISONING.md
    build_poison_manifest.py  per-chain arms, token parity enforced
    run_poison_suite.sh       clean twin + every arm, sequential, capture on
    poison_report.py          flips and attention on the poisoned span
    poison_deep.py            divergence, entropy, looked x flipped, cost
    poison_figures.py         SVG figures
    compare.py          several runs side by side
    flowcept_smoke.py   provenance in, provenance out, no GPU
    local_check.sh      the CPU arm, staged
    gpu_setup.sh        build .venv-gpu on a local NVIDIA box
    gpu_sizing.py       what fits on this card, printed as flags
    gpu_check.sh        the local-GPU ladder, ends with 3 real chains
    run_gpu.sh          launch a run: fresh timestamped dir, sized, detached
    push_results.sh     ship a finished run to a remote host, and verify it
    box.sh              drive a remote NVIDIA box: push, setup, run, pull
    lib_siblings.sh     find (or clone) flowcept + the connector
polaris/
    env.sh setup.sh job.pbs smoke_test.sh push.sh lib.sh check_vllm_api.py
    RUNBOOK.md          the sequence, in order
configs/
    flowcept_local.yaml       CPU box; GPU telemetry off
    flowcept_local_gpu.yaml   local NVIDIA box; same, telemetry on
    flowcept_polaris.yaml     compute node; same shape again
    (all three fully offline -- provenance to JSONL, no Redis, no Mongo)
data/opal_chains.csv    100 chains, 430 rounds
```

Depends on two sibling checkouts, used from source:
`../vllm-attn-connector` and `../flowcept`.

---

## CPU arm

Works anywhere, including on a GPU box — and worth running there first, since
it catches everything that is not CUDA.

```bash
make venv
make check
```

Five stages, ~3 minutes after the first tokenizer download:

| stage | what it covers |
|---|---|
| 0 environment | venv, imports, and that the provenance reader resolves *without* vLLM |
| 1 aggregations | the connector's own CPU tests: streaming reduction, segment planning |
| 2 flowcept | interceptor → offline JSONL dump → `report.py`'s join |
| 3 dry run, 3 chains | tokenize, segment, budget |
| 4 dry run, 100 chains | writes `results/plan/plan.json` |

Current output of stage 4, against `Qwen/Qwen3-4B-Instruct-2507`:

```
turns_planned                      430
chains_truncated                   0
prompt_tokens_total                892843
prompt_tokens_max                  5116
prompt_tokens_median               1625
context_headroom_tokens            3044
segments_median                    8
kept_per_step_median               159
safetensors_mib_upper_bound        53.1
```

So `max_model_len=8192` fits every chain with 3044 tokens to spare, and the
whole capture is ~53 MiB of tensors. Both numbers come from the local arm; the
Polaris walltime request is sized from them.

### Why this particular laptop cannot capture

It has an Intel Iris Xe iGPU and no NVIDIA device. The connector needs CUDA
specifically, not just "a GPU": it allocates a `torch.cuda.Stream()` to run the
`q·Kᵀ` recomputation off the critical path, and its probe overrides a CUDA
attention backend. CPU and OpenVINO vLLM have neither, so there is no
configuration of *this* machine that captures attention.

On a machine that does have one, see the next section — nothing below is
Polaris-specific.

---

## Local NVIDIA GPU arm

Same code as Polaris, no scheduler. One extra venv, because CUDA torch and the
CPU torch in `.venv` cannot coexist — and keeping them apart means the dry run
still works if the GPU goes away.

```bash
make gpu-venv      # builds .venv-gpu: uv, vLLM 0.28.0, flowcept, the connector
make gpu-check     # staged bring-up, ends with 3 real chains captured
./scripts/run_gpu.sh --label smoke --limit 5     # rehearse
./scripts/run_gpu.sh                             # all 100 chains
```

`run_gpu.sh` is the only launcher, used by `make gpu-run` and by
`box.sh run` over ssh, so there is one code path rather than three that drift.
It picks a fresh `results/<UTC timestamp>[-label]/` every time, points
`results/latest` at it, sets `PATH`/`VLLM_ENABLE_V1_MULTIPROCESSING`/
`FLOWCEPT_SETTINGS_PATH`, derives `--max-num-seqs` and
`--gpu-memory-utilization` from `gpu_sizing.py` unless you pass them, detaches
under `setsid nohup`, and refuses to start while another run is live. Unknown
flags pass through to `run_experiment.py`.

**Never reuse a run directory.** Flowcept keeps old dumps
(`delete_previous_file: false`) and names them with a pid, so a second run in
one directory leaves two buffers while `turns.jsonl` is overwritten — and an
analysis can then pair one run's answers with another's provenance without
erroring. The timestamped directory exists to make that impossible.

### Sizing the card first

The binding constraint is the KV cache, not the weights. Qwen3-4B-Instruct-2507
is 36 layers × 8 KV heads × 128, so **144 KiB of KV per token** — high for a 4B
model — and a chain's final round reaches ~5.1k tokens, so one long sequence
alone holds ~720 MiB. `scripts/gpu_sizing.py` reads `results/plan/plan.json` and
your actual VRAM and answers in flags:

| VRAM | bf16 | concurrent at the longest round | suggested |
|---|---|---|---|
| 8 GiB | does not fit | — | use the AWQ 4-bit build |
| 12 GiB | fits, barely | 2 | AWQ 4-bit raises it to 10 |
| 16 GiB | yes | 8 | `--max-num-seqs 8 --cuda-graphs` |
| 24 GiB | yes | 18 | `--max-num-seqs 18 --cuda-graphs` |
| 40 GiB | yes | 38 | `--max-num-seqs 38 --cuda-graphs` |

Capture itself costs almost no memory — the probe's query buffers are tens of
MB and its scratch is under a MB per request. It costs *time*, not VRAM.

Below ~4 concurrent long sequences the later rounds serialise and the run is
dominated by rounds 5–7. That is when to reach for either the 4-bit build
(`cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit`, ~4.9 GiB freed) or `--limit`.

### Unified-memory parts (GB10 / DGX Spark, GH200)

`nvidia-smi --query-gpu=memory.total` returns `[N/A]` on these — there is no
discrete framebuffer, the CPU and GPU share one pool. `gpu_sizing.py` asks
torch instead, so run it from inside `.venv-gpu`.

Two consequences:

- **`gpu_memory_utilization` is a fraction of the whole shared pool**, so a
  value that is routine on a discrete card starves the host. Start around
  `--gpu-memory-utilization 0.4` and raise it once a run is stable.
- **Pin the attention backend.** The probe overrides exactly one backend, and
  on Blackwell vLLM may select FlashInfer rather than FlashAttention. If they
  disagree the probe never fires and *every captured value is zero*, with no
  error anywhere. `--attn-backend NAME` sets `VLLM_ATTENTION_BACKEND` and the
  probe's override together, so they cannot drift apart.

A run now records `probe_layers_seen`, and `verify` fails loudly when it is 0.
That is the check to look at first on new hardware — before accuracy, before
anything.

### The ladder

`./scripts/gpu_check.sh` is `polaris/smoke_test.sh` without PBS or modules:

| stage | a failure here means |
|---|---|
| 0 environment | driver, vLLM version, missing API, no Triton |
| 1 imports | packaging, or `kv_connector_module_path` |
| 2 engine, **no connector** | pure vLLM on this box — wheels vs driver |
| 3 the connector's own e2e | layout, Triton kernel, probe, or emission |
| 4 three OPAL chains | this experiment: ranges, the join, the report |

**Stage 3 is the one that matters.** It is the only stage that catches an
all-zero capture: mass conservation passes trivially on zeros (`0 + 1 == 1`), so
a structurally valid output proves nothing. The e2e asserts `attn_sum` totals
one softmax unit per decode step, which zeros cannot fake. The silent way to get
zeros is `install_probe()` returning False, which is why `check_vllm_api.py`
verifies the attention backend registry and not just the KV connector API.

### Two environment settings that are not optional

`VLLM_ENABLE_V1_MULTIPROCESSING=0` — the engine then runs in-process, so the
connector emits provenance in-process too and Flowcept needs no message queue.
With multiprocessing on, records leave the process and you need a reachable
Redis. Both `gpu_check.sh` and `make gpu-run` set it.

`FLOWCEPT_SETTINGS_PATH=configs/flowcept_local_gpu.yaml` — identical to the CPU
config except GPU telemetry is on. Still fully offline: provenance goes to
JSONL, same as Polaris, so `analyze.py` reads a local run and a Polaris run the
same way.

---

## Remote NVIDIA box over ssh

For a GPU workstation or lab server that is not Polaris — no scheduler, no
modules. One script drives the whole loop:

```bash
export OPAL_BOX=user@gpuhost
# export OPAL_BOX_PATH=/data/opal        # default: ~/exp-2026-10-01-opal-attn

./scripts/box.sh push          # rsync source + data (no venvs, no results)
./scripts/box.sh setup         # build .venv-gpu there; clones the two siblings
./scripts/box.sh check         # the staged ladder, ends with 3 real chains
./scripts/box.sh sizing        # what fits on that card, as flags

./scripts/box.sh run --limit 100
./scripts/box.sh status        # alive? plus the log tail and nvidia-smi
./scripts/box.sh logs          # follow it
./scripts/box.sh pull          # bring runs back into ./results/
make analyze RUN=results/<name>
```

`make box-push`, `box-setup`, `box-check`, `box-run`, `box-status`, `box-pull`
wrap the same commands.

**The run is detached.** `setsid nohup`, its own log and pidfile, so a dropped
ssh connection costs nothing and `status` works from anywhere. This is the one
thing a scheduler would give you and a bare box does not. `run` refuses to
start a second run while one is live — two engines on one card means both OOM
or both crawl — and it reports immediately if the job dies on startup rather
than letting you discover it an hour later.

**The siblings are fetched on the box.** `flowcept` and `vllm-attn-connector`
are not pushed (they are ~112 MB of `.git` and belong upstream). On a bare box
`gpu_setup.sh` clones them into `vendor/` *inside* the experiment directory —
not next to it, since the parent may not be writable and scattering clones
through someone's `$HOME` is rude. In the genesis workspace the existing
siblings are used as-is, local edits intact. `gpu_check.sh` prints the revision
of each, so running against stale code is visible rather than mysterious.

If the box has no outbound access, rsync the checkouts yourself —
`rsync -az ../flowcept BOX:<path>/vendor/` — and setup will use them.

## Polaris arm

For all 100 chains on an A100, or when you have no local card. Read
[`polaris/RUNBOOK.md`](polaris/RUNBOOK.md). In short:

```bash
./polaris/push.sh                                   # laptop
qsub -I -l select=1 -l filesystems=home:eagle -q debug -A $OPAL_PROJECT
cd $OPAL_ROOT && ./polaris/setup.sh && ./polaris/smoke_test.sh
OPAL_LIMIT=5 qsub -A $OPAL_PROJECT -q debug polaris/job.pbs   # rehearse
qsub -A $OPAL_PROJECT polaris/job.pbs                          # the run
```

Set `OPAL_PROJECT=radix-io` and `OPAL_ROOT=/lus/eagle/projects/radix-io/$USER/opal-attn`
in `~/.bashrc` first.

Adapted from `../spotterai/polaris/`, which already solved the Polaris-specific
problems: no conda module (it pins a gcc and an HDF5 that do not exist under the
current Cray PE), `set -u` off around lmod, `darshan` unloaded, Triton and
inductor caches on `/tmp` rather than Lustre, and `|| true` on the HSN lookup so
a failed resolve does not kill the job at `source env.sh` with no output.

The compute node is **fully offline by choice**. It can reach out through the
ALCF proxy, and `setup.sh` depends on that — but a run that silently downloads
is not reproducible, because you cannot tell afterwards which weights it used.
Runs are pinned to what was staged and a cache miss fails immediately.

---

## Three decisions worth knowing about

### Round-synchronous batching

Rounds inside a chain are strictly sequential: round N's prompt contains the
model's own round N−1 answer. Chains are independent. So the driver issues one
batched `generate` per *round index* — at round r it submits the r-th turn of
every chain that has one. That is 7 engine calls for 430 generations instead of
430.

### The join is on request id, not prompt length

The connector's own 5-round example matches records to turns by prompt token
count. With one chain that is fine. With 430 turns, collisions are near-certain,
and a wrong join is silent — you get a plausible report about the wrong text. So
the runner records `RequestOutput.request_id` and `report.py` matches it against
`task["used"]["request_id"]`, which the Flowcept adapter stores with the
connector's `:g<group>` suffix stripped. Prompt length is then checked as an
*independent witness* that the join is right, rather than used as the key.

### The chains are replayed verbatim

Nothing is prepended, appended or reworded. An OPAL row already carries its own
SESSION briefing, TOOLS list, TERMS, RULES and OUTPUT FORMAT — including the
`ANSWER: <...>` line it wants back — so a system prompt of our own would restate
those rules in different words, shift every token index in the conversation, and
make the measured attention partly a property of the wrapper instead of the
benchmark. The conversation is exactly the chain's prompts and the model's
replies. (Checked for Qwen: with no system message the chat template inserts no
default one.)

### `tool_ranges` from the CSV is not used

The column records character spans under the generator's own regex tokenizer,
which splits far more coarsely than BPE — and the error is not a constant
factor, because it depends how much of the text is numbers, which OPAL tool
outputs mostly are. Worse, CSV spans are relative to a round's own prompt; once
that prompt is turn N of a conversation every index moves by the system prompt,
all previous rounds, all previous answers and the chat template's markers. The
shift depends on *whose* answers are in the history, so it cannot be
precomputed. `ranges.declare_ranges()` re-finds the blocks in the rendered
conversation and maps them through the tokenizer's own offset mapping. This
reads the chain's structure; it does not alter the text.

The dry run checks this the only way that works without a GPU: round *r* must
carry exactly *r−1* answer segments. If the chat template's assistant marker
ever stops matching, every history segment silently vanishes and that count
catches it. It currently passes on all 430 turns.

---

## What a run directory contains

One run, one directory, nothing of it anywhere else:

```
results/20261002-120000Z/
  launch.txt         the exact command, host and time
  run.log            stdout and stderr
  run_meta.json      model, settings, timings, probe evidence, purge count
  turns.jsonl        one line per (chain, round): reply, verdict, segments
  prompts.jsonl      the rendered conversation and its line table, per turn
  summary.json       verdicts, accuracy, propagation, where attention went
  provenance.jsonl   the Flowcept records -- exactly the turns
  attention/         the SafeTensors those records point at
  raw/               Flowcept's own dump, before the flush nudge was removed
```

`run_experiment.py` chdirs into the run directory before Flowcept starts.
That is not cosmetic: Flowcept's `dump_buffer.path` is relative to the working
directory, so without it the provenance dump lands wherever the command was
launched from — the repository root, for every run before this change.

`prompts.jsonl` is what makes a heat map drawable later. It holds, per turn,
the fully rendered conversation and a LINE TABLE -- every physical line with
the token span it occupies. It is stored rather than recomputed because
recomputing needs the tokenizer *and* the chat template exactly as they were
on the day: a template that has drifted since would re-derive different spans
for the same captured tensors and never fail. This is the same reason
`tool_ranges` is not used. It is kept out of `turns.jsonl` because it is
thousands of tokens per turn and `turns.jsonl` is read by hand.

`provenance.jsonl` is canonical. Flowcept's own dump is kept under `raw/`
because the two legitimately differ by one record: the engine needs a
one-token request after the final round to flush that round's records, and
that request is not part of the experiment, so it is removed from
`provenance.jsonl` and its tensor file deleted. `run_meta.json` records how
many were purged.

`results/latest` is a RELATIVE symlink to the newest run, so it survives
`box.sh pull`. Use it for tailing a live run; name the timestamped directory
explicitly for anything you want to still mean the same thing tomorrow, since
`latest` moves the moment another run starts.

**Run directories are never reused.** `run_gpu.sh` timestamps every one. A
second run in one directory leaves two Flowcept dumps while `turns.jsonl` is
overwritten, and an analysis can then pair one run's answers with another's
provenance without erroring.

## Moving a run between machines

Runs are produced on Polaris or a GPU box and read somewhere else, so shipping
them is a first-class step rather than an afterthought.

```bash
# pull one back
rsync -avz agueroudji@polaris.alcf.anl.gov:$OPAL_ROOT/runs/<jobid> ./results/

# push one anywhere (laptop or straight off a Polaris login node)
make push-results RUN=results/<jobid> TO=user@host:/data/opal
./scripts/push_results.sh results/<jobid> --to user@host:/data/opal --verify
./scripts/push_results.sh results/<jobid> --no-tensors      # provenance only
./scripts/push_results.sh results/<jobid> --verify-only     # re-check an old push
```

The destination can also come from `$OPAL_RESULTS_REMOTE`, and a plain local
path works too — useful for an attached disk, and it exercises the same code.

**`--verify` checks against the connector's own sha256**, recorded when each
file was written, not against a hash taken afterwards. Hashing the source and
comparing to the destination only proves the copy matches the copy; comparing
both to what the provenance claims proves the bytes are the ones whose
statistics were captured. A mismatch names the file and exits non-zero.

**`--verify-only` exists separately** because `--verify` rsyncs first, which
would silently *repair* a corrupted remote file — so a combined run can never
report a failure. Use it to re-check an archive later.

Two refusals, both deliberate: a directory with no `turns.jsonl` is not a
finished run, and one with no `flowcept_buffer*.jsonl` has lost the provenance
that makes the tensors interpretable. Neither is worth discovering a week later.

### Why a moved run still reads

The connector records an **absolute** `file://` uri, resolved on the machine
that captured it, and `store.load` reads that path verbatim. The moment a run
directory moves, that path stops existing and every tensor read fails with a
FileNotFoundError naming `/lus/eagle`.

`opal_attn.provenance` rebases it: files live at
`<connector-workflow-id>/<request>_g<n>.safetensors`, so the last two path
components are stable wherever the directory lands. The uri is rewritten only
when it does not resolve, so a run analysed in place is untouched. Pass
`run_dir=` to `load_provenance`; `analyze.py` does it for you.

## Reading the results

```bash
make analyze RUN=results/<jobid>
```

`analyze.py` runs two separate things and does not merge them:

**Verification** — is the provenance trustworthy? A record per turn, prompt
counts agreeing between runner and connector, declared ranges present verbatim
as segments, kept mass plus discarded residual summing to one, the manifest
agreeing with the file on disk. These are pass/fail.

**Report** — what does the run show? Accuracy by tier, round and task type;
where chains first break and how they do after; which wrong answers are a
*documented shortcut* the CSV predicts rather than noise; and `hottest_by_origin`,
which is the point of the whole exercise — whether peak attention at each turn
landed on the current round's evidence or on the conversation history.

**Density** -- attention per span kind, per token, banded by distance from the
decode point. Read this rather than raw shares. Three normalisations apply and
each replaced a wrong answer: mass comes from `attn_sum` (unbiased,
pre-selection) not the top_pct arrays; it is renormalised over CONTENT,
because position 0, blank lines and chat markers take 15% of the mass on
Qwen3.5-35B-A3B and 55% on Qwen3-4B; and it is divided by span length and
banded by distance, or the table just ranks spans by how long they are and how
close they usually sit. Density falls 10-18x from the nearest instance of a
kind to the farthest, which swamps every between-kind difference.

None of the report is asserted. OPAL is hard, and a run with perfect provenance
and 20% accuracy is a successful run.

## Reading ONE turn

`analyze.py` and `failures.py` aggregate over a run. To see where a single
answer came from:

```bash
make heatmap RUN=results/<jobid> CHAIN=exp32-chain-0020-hard ROUND=5
./scripts/heatmap.py results/<jobid> --chain ... --round 5 --top 40
```

Heat is `val_all_max`: per line the max over its retained positions,
aggregated across that item's decode steps with exponential decay `0.5**k`.

**One column per answer item.** An OPAL answer like
`<accession>; <MATCHES or DIVERGES>` is two questions answered by two disjoint
runs of decode steps, and they consult different evidence -- the accession
comes from the conversation history, the verdict from the current round.
Aggregating the reply as a whole averages that distinction away, which is why
the runner records `gen_char_ends`: it maps each decode step back to the span
of text it wrote, so the steps can be split on the separator the benchmark's
own OUTPUT FORMAT specifies.

Max-with-decay rather than a sum, because the objective is to RANK lines so
that noise does not outrank evidence -- not to estimate total attention mass.
A sum rewards long lines and lines merely adjacent to many retained positions;
an undecayed max lets the tail of an item, by which point the answer is
already committed, outvote the step that actually chose it.

Each column is scaled to its own maximum: the two items are not on a
comparable scale and sharing one makes the weaker column vanish.

Note what this is NOT: `hottest_segments` in the run-level report samples a
single decode step and reduces to one argmax per turn, so on a two-part answer
it only ever sees the second part. The two numbers answer different questions
and will not agree.

### Reproducing the reference heat map

`row_analogue_reasoned.txt` is round 5 of `exp32-chain-0020-hard`. It was
produced by this harness, but **not** by its default configuration, and the
two differences both matter:

| | reference | this harness's default |
|---|---|---|
| history | the CSV's answers | the model's own replies |
| system prompt | 138 chars, 34 tokens | none |

Verified rather than assumed: the body is `data/opal_chains.csv` untouched
(eleven distinctive lines across all five rounds, byte-identical), all four
history answers equal the CSV's `correct_answer`, and the system block is the
only other difference. The chain also exists in the connector's
`tests/bio_example.csv`, but that copy says "This is a 5-round investigation."
where the reference says 6 -- so the reference is this CSV, and cannot have
come from `e2e_example.py`.

```bash
make reproduce-reference          # 1 chain, GPU, writes results/<stamp>-reference
make heatmap RUN=results/<stamp>-reference CHAIN=exp32-chain-0020-hard ROUND=5
```

With the wrapper on, round 5 of that chain renders to **exactly 3534 tokens**
and 273 lines, which is what the reference reports. That is the prompt side
reproduced bit for bit; only the attention values then depend on the run.

`--system-prompt` takes the literal string rather than a boolean, so
`run_meta.json` records what was actually used and a different wrapper needs
no code change. It is recorded even when empty: a missing key cannot
distinguish "no system prompt" from "a run predating the flag", and those
shift every token index differently.

**This is a deviation, kept visible.** An OPAL row already carries its own
SESSION briefing, TOOLS, TERMS, RULES and an OUTPUT FORMAT naming the
`ANSWER: <...>` line, so the wrapper restates the benchmark's rules in
different words and moves every index in every conversation. The default
stays off. Over all 100 chains it costs 34 tokens a turn --
`context_headroom_tokens` 3044 -> 3010, still no truncation.

## Poisoning

Corrupting one span deliberately and measuring what moved is a layer on top of
this one: see **[POISONING.md](POISONING.md)**. It covers the arms, the
token-parity constraint, how to compare two runs without trusting token
indices, and -- at length -- which of its results do not hold up.

## Known limitations

Inherited from the connector, and they bound what the results can claim:

- **Prefill attention is not captured.** Only decode queries are scored. If
  content was assembled into a late prompt position during prefill, decode
  attention points there rather than at the original source.
- **Attention magnitude is not signed.** High attention means "this token was
  consulted", not "this token was used affirmatively".
- **Reduced over layers and heads.** The output says which head supplied the
  max, not which layer produced a score.
- **TP > 1 is handled but untested on real hardware.** The default here is TP=1.
- **`--max-steps` keeps the FIRST N decode steps.** On a reasoning model those
  are the thinking; the `ANSWER:` tokens come last. It is the obvious way to
  shrink a capture and it discards exactly what a per-answer heat map
  aggregates over. Trim with `--top-pct` instead.
- **A truncated reply is not an answer.** `finish_reason == "length"` gets its
  own `truncated` verdict, because a reasoning model quotes the output format
  back to itself while working and `parse_answer` will scrape that. Before
  this existed a 90-turn run reported 0.8778 accuracy where the replies that
  actually finished were 78/78.
