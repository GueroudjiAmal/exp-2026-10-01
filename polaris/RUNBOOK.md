# Polaris runbook

Everything up to the batch job runs in **one interactive job**. This is the
sequence; `../README.md` is the why.

In `~/.bashrc` on Polaris:

```bash
export OPAL_PROJECT=radix-io
export OPAL_ROOT=/lus/eagle/projects/radix-io/$USER/opal-attn
```

No `HF_TOKEN` is needed — the benchmark is a CSV that travels with the source,
and `Qwen/Qwen3-4B-Instruct-2507` is not gated.

---

## 1 · Laptop: validate, then push

```bash
cd /home/agueroudji/Work/genesis/exp-2026-10-01-opal-attn
make check          # the whole local arm; must pass before you burn node hours
./polaris/push.sh
```

Note the `pushed: <sha>` line. Source, config and the benchmark CSV only — no
venv, no `flowcept/` or `vllm-attn-connector/` (setup clones those), no results.

## 2 · Get the node

```bash
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=2:00:00 \
     -q debug -A $OPAL_PROJECT
cd $OPAL_ROOT
```

Everything below is inside this job. `setup.sh` refuses to run outside one.

## 3 · Stage the environment (once, ~20–40 min)

```bash
./polaris/setup.sh
```

proxy → modules → uv → venv → vLLM 0.28.0 → **two** API checks → flowcept +
connector → weights → **import vLLM on the GPU** → warm the compile cache.

Three lines to watch scroll past:

- **the API check** — it verifies the KV connector base *and* the attention
  backend registry. The second is the one people forget: a vLLM can have the
  connector API and no `register_backend`, and then `install_probe()` returns
  False, the capture still writes structurally valid records, and every number
  in them is zero.
- **step 6, `importing vLLM on a real GPU`** — the CUDA-wheels-vs-driver
  question. Passing it essentially proves the environment.
- **step 7, the warmup** — non-fatal, but skipping it means the batch job pays
  the JIT cost out of its own walltime.

## 4 · Every new shell in the job

```bash
source polaris/env.sh
```

Sets the venv, the offline flags, `FLOWCEPT_SETTINGS_PATH`, `PYTHONPATH`, and
the Triton/inductor cache dirs. Without it every later command fails on an unset
variable or a missing module. Check `revision <sha>` matches what push.sh
printed.

## 5 · The ladder

```bash
./polaris/smoke_test.sh          # or one stage: ./polaris/smoke_test.sh 3
```

| stage | what a failure here means |
|---|---|
| 0 environment | wrong vLLM version, missing API, or no GPU in the allocation |
| 1 imports | packaging, or `kv_connector_module_path` |
| 2 engine, **no connector** | **pure vLLM** — CUDA wheels vs driver |
| 3 the connector's own e2e | layout, Triton kernel, probe, or emission |
| 4 five OPAL chains | this experiment: ranges, the join, the report |

Stage 2 excludes the connector entirely, so a failure there is vLLM-on-Polaris
and nothing to do with this work.

**Stage 3 is the load-bearing one.** It runs the connector's 43–45 assertions
against a real engine with CUDA graphs on, which is the only thing that catches
an all-zero capture — mass conservation passes trivially on zeros (`0 + 1 == 1`),
so "the output looks structurally fine" is not evidence. Stage 3 asserts
`attn_sum` totals one softmax unit per decode step, which zeros cannot fake.

## 6 · Rehearse in the debug queue

```bash
OPAL_LIMIT=5 qsub -A $OPAL_PROJECT -q debug -l walltime=00:30:00 polaris/job.pbs
```

Same code path as the real run, five chains. Confirms the batch environment,
the run directory layout and the post-run analysis before committing two hours.

## 7 · The real run

```bash
qsub -A $OPAL_PROJECT polaris/job.pbs
```

Defaults: all 100 chains, `Qwen/Qwen3-4B-Instruct-2507`, TP=1, `top_pct=10`,
`max_tokens=32`, `max_model_len=8192`, CUDA graphs on. Override any of them
with the `OPAL_*` environment variables listed at the top of `job.pbs`.

Artifacts land in `$OPAL_RUNS/<jobid>/`:

| file | what it is |
|---|---|
| `turns.jsonl` | one record per (chain, round): answer, correctness, declared segments, request id |
| `run_meta.json` | model, settings, timings, any chain the context window truncated |
| `summary.json` | accuracy breakdown + verification results |
| `flowcept_buffer*.jsonl` | the Flowcept provenance dump |
| `<request>_g0.safetensors` | the attention statistics, one file per request |

The job re-runs `analyze.py` against the files on disk at the end, rather than
trusting the in-memory buffer the run already checked — the dump is what
survives the job, and it is all an off-node analysis will ever see.

## 8 · Overhead measurement (optional)

```bash
OPAL_EXTRA_ARGS=--no-connector OPAL_LIMIT=20 qsub -A $OPAL_PROJECT -q debug polaris/job.pbs
OPAL_LIMIT=20                                 qsub -A $OPAL_PROJECT -q debug polaris/job.pbs
```

Two pitfalls, both of which produced wrong numbers during the connector's own
development: **interleave the arms in one session** — GPU clock and thermal
state drift between sessions by more than the effect being measured — and
**discard the first run**, which pays warmup the others do not. Compare
`generate_seconds` in `run_meta.json`.

## 9 · Move the run off the node

Pull it to the laptop:

```bash
rsync -avz agueroudji@polaris.alcf.anl.gov:$OPAL_ROOT/runs/<jobid> ./results/
make analyze RUN=results/<jobid>
```

Or push it somewhere else directly — faster for a large run, since it skips the
round trip through your laptop. `push_results.sh` needs nothing but rsync and
python3, so it works from a Polaris login node:

```bash
# on Polaris
cd $OPAL_ROOT
./scripts/push_results.sh runs/<jobid> --to user@archive:/data/opal --verify
```

`--verify` checks the transfer against the sha256 the **connector** recorded
when it wrote each file, so it proves the bytes are the ones whose statistics
were captured — not merely that the copy matches the copy. `--verify-only`
re-checks an existing push without resending (and without silently repairing
it, which a plain `--verify` run would).

Analysis needs no GPU and no vLLM anywhere: `opal_attn.provenance` loads the
connector's SafeTensors reader directly off disk, and rebases the records'
absolute `file://` uris onto wherever the directory now lives. Add
`--no-tensors` to `analyze.py` if you moved the JSONL but not the tensor
files.
