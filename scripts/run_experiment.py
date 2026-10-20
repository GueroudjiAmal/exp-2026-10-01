#!/usr/bin/env python
"""Run the OPAL chain experiment, or plan it without a GPU.

    python scripts/run_experiment.py --dry-run          # local: no CUDA needed
    python scripts/run_experiment.py                    # real: needs a GPU

Exit status is the number of failed checks, so this is usable as a test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from opal_attn import plan as planning  # noqa: E402
from opal_attn.chains import load_chains, select  # noqa: E402

# The project runs on Qwen3.5. Anything else is a deliberate override --
# notably `make reproduce-reference`, which PINS Qwen3-4B because the
# artifact it reproduces was made with it (3534 tokens under that
# vocabulary, 3614 under this one).
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(HERE / "data" / "opal_chains.csv"))
    p.add_argument("--model", default=os.environ.get("OPAL_MODEL", DEFAULT_MODEL))
    p.add_argument("--tokenizer", default=None,
                   help="defaults to --model; set it to plan without the weights")
    p.add_argument("--out-dir", default=os.environ.get("OPAL_OUT", str(HERE / "results" / "latest")))
    p.add_argument("--workflow-id", default=os.environ.get("OPAL_WORKFLOW_ID", "opal-attn"))

    # The OPAL rows are self-contained -- SESSION, TOOLS, TERMS, RULES and an
    # OUTPUT FORMAT that already names the "ANSWER: <...>" line -- so the
    # default is NO system prompt and the conversation is exactly the
    # benchmark's text. Anything here restates the benchmark's own rules in
    # different words and shifts every token index in every conversation,
    # which makes the measured attention partly a property of the wrapper.
    #
    # A literal string rather than a boolean, so run_meta.json records what was
    # actually used and a different wrapper needs no code change.
    p.add_argument("--system-prompt", default=os.environ.get("OPAL_SYSTEM_PROMPT", ""),
                   help="prepend a system turn to every chain. OFF by default; "
                        "see `make reproduce-reference` for the one wrapper "
                        "with a known provenance.")

    p.add_argument("--poison", default=None, metavar="SPEC.json",
                   help="apply a poison manifest built by "
                        "scripts/build_poison_manifest.py. Token parity is "
                        "re-checked against this run's tokenizer and an "
                        "unbalanced substitution aborts the run.")

    p.add_argument("--history", choices=["model", "ground-truth"], default="model",
                   help="what the conversation carries forward. 'model' (the "
                        "default) replays each chain with the model's own "
                        "answers, so an early mistake propagates. "
                        "'ground-truth' substitutes the CSV's answer, giving a "
                        "control arm in which nothing can propagate -- the "
                        "accuracy gap between the two is the propagation "
                        "effect, measured directly.")

    sel = p.add_argument_group("chain selection")
    sel.add_argument("--limit", type=int, default=0, help="first N chains; 0 = all 100")
    sel.add_argument("--tier", choices=["low", "medium", "hard"], default=None)
    sel.add_argument("--chain-ids", nargs="+", default=None)

    cap = p.add_argument_group("capture")
    cap.add_argument("--top-pct", type=float, default=10.0)
    cap.add_argument("--chunk-size", type=int, default=32)
    cap.add_argument("--max-steps", type=int, default=0, help="0 = every decode step")
    cap.add_argument("--layer-stats", action="store_true",
                     help="per-head diagnostics; roughly +50%% capture cost")

    eng = p.add_argument_group("engine")
    eng.add_argument("--max-model-len", type=int, default=8192)
    eng.add_argument("--max-tokens", type=int, default=32)
    # Greedy BY DEFAULT, because a benchmark number you cannot reproduce is
    # not a benchmark number. But greedy is not free: Qwen explicitly advises
    # against it for their THINKING models, where it degrades quality and can
    # produce endless repetition. A reasoning model therefore needs these
    # exposed -- running one at temperature 0 is not "the careful choice", it
    # is running it outside the regime its authors measured.
    #
    # Recorded in run_meta either way, so a run always says how it sampled.
    eng.add_argument("--temperature", type=float, default=0.0,
                     help="0.0 = greedy (default, reproducible). Qwen3/3.5 "
                          "thinking models want ~0.6 with --top-p 0.95.")
    eng.add_argument("--top-p", type=float, default=1.0,
                     help="nucleus sampling; only meaningful above "
                          "temperature 0")
    eng.add_argument("--max-num-seqs", type=int, default=64)
    eng.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    eng.add_argument("--tensor-parallel-size", type=int, default=1)
    eng.add_argument("--cuda-graphs", action="store_true",
                     help="on Polaris this should be ON; off is for small cards")
    eng.add_argument("--prefix-caching", action="store_true",
                     help="let the engine reuse cached prefixes. OFF by "
                          "default during capture: the connector tracks block "
                          "tables from newly-allocated blocks only, so a "
                          "cache hit makes it score a truncated prompt and "
                          "silently mis-attribute every position after the "
                          "gap. Costs re-prefilling each round; buys correct "
                          "provenance on long prompts.")
    eng.add_argument("--enable-thinking", action="store_true",
                     help="let a reasoning model emit its <think> block before "
                          "answering. OFF by default: every OPAL row demands "
                          "'exactly one line and nothing else', so this is a "
                          "deliberate deviation from the benchmark's protocol, "
                          "not the headline configuration. parse_answer takes "
                          "the LAST ANSWER: line, so a model that reasons then "
                          "complies still scores. Needs --max-tokens >= 512.")
    eng.add_argument("--attn-backend", default=os.environ.get("OPAL_ATTN_BACKEND", ""),
                     help="pin BOTH vLLM's attention backend and the probe's "
                          "override to this name (e.g. FLASH_ATTN, FLASHINFER, "
                          "TRITON_ATTN). Leave unset to use the connector's "
                          "FLASH_ATTN default. Needed wherever vLLM would "
                          "otherwise pick a backend the probe is not on.")
    eng.add_argument("--no-connector", action="store_true",
                     help="baseline arm: identical workload, no capture")

    p.add_argument("--dry-run", action="store_true",
                   help="plan only: tokenize, segment and budget, then stop")
    return p


def do_dry_run(args) -> int:
    from transformers import AutoTokenizer

    chains = select(load_chains(args.csv), limit=args.limit, tier=args.tier,
                    chain_ids=args.chain_ids)
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    # The dry run is the pre-flight: it must apply the poison too, or an
    # unbalanced manifest passes every check here and fails on the GPU.
    if args.poison:
        from opal_attn.poison import apply_manifest
        chains, _hist, pmeta = apply_manifest(chains, args.poison, tok)
        print(f"poison {pmeta['arm']!r} ({pmeta['form']}/{pmeta['kind']}) "
              f"applied to {pmeta['chains_poisoned']}/{len(chains)} chains")
    print(f"chains: {len(chains)}   rounds: {sum(len(c) for c in chains)}   "
          f"tokenizer: {args.tokenizer or args.model}")

    pl = planning.build_plan(
        chains, tok, model=args.model, max_model_len=args.max_model_len,
        max_tokens=args.max_tokens, top_pct=args.top_pct,
        chunk_size=args.chunk_size, max_steps=args.max_steps,
        enable_thinking=args.enable_thinking,
        system_prompt=args.system_prompt)
    # The planner always uses ground-truth history, so its token counts are
    # exact for --history ground-truth and an estimate for --history model
    # (the model's replies tokenize to different lengths).
    proj = planning.project(pl)

    print("\n=== projection ===")
    for k, v in proj.items():
        print(f"  {k:<34} {v}")

    if pl.dropped:
        print(f"\n=== {len(pl.dropped)} chain(s) truncated by the context window ===")
        for cid, why in list(pl.dropped.items())[:10]:
            print(f"  {cid}: {why}")

    print("\n=== checks (no GPU) ===")
    checks = planning.check(pl)
    for ok, msg in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}")

    path = planning.write_plan(args.out_dir, pl, proj)
    print(f"\nplan written to {path}")
    failed = sum(1 for ok, _ in checks if not ok)
    print(f"FAILURES: {failed}")
    return failed


def do_run(args) -> int:
    from opal_attn import report, runner

    turns, buffer, meta = runner.run(args)
    out = Path(args.out_dir)

    # provenance.jsonl is the canonical record set: exactly the turns, with the
    # flush nudge removed. Flowcept's own dump is moved under raw/ so the two
    # cannot be confused -- they differ by that one record.
    with (out / "provenance.jsonl").open("w") as fh:
        for rec in buffer:
            fh.write(json.dumps(rec, default=str) + "\n")
    meta["raw_dumps"] = runner.tidy_run_dir(out)
    runner.write_results(out, turns, meta)

    turn_dicts = report.load_turns(out / "turns.jsonl")
    load_provenance = None
    if not args.no_connector:
        # Prefers the installed package; falls back to loading store.py off
        # disk, which is what lets the same analysis run on a laptop.
        from opal_attn.provenance import load_provenance

    print("\n=== capture verification ===")
    checks = report.verify(turn_dicts, buffer, top_pct=args.top_pct,
                           load_provenance=load_provenance, run_dir=out, meta=meta)
    for ok, msg in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}")

    summary = report.summarise(turn_dicts)
    print("\n=== accuracy (reported, not asserted) ===")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")

    if load_provenance is not None:
        hot = report.hottest_segments(turn_dicts, buffer, load_provenance,
                                      run_dir=out)
        print("\n=== where attention went ===")
        for k, v in hot.items():
            print(f"  {k:<30} {v}")
        summary["attention"] = hot

    summary["run_meta"] = meta
    summary["verification"] = [{"passed": ok, "check": m} for ok, m in checks]
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    n_tensors = len(list((out / "attention").rglob("*.safetensors")))
    print(f"\nartifacts in {out}")
    for name in ("run_meta.json", "turns.jsonl", "summary.json",
                 "provenance.jsonl"):
        if (out / name).is_file():
            print(f"  {name:20} {(out / name).stat().st_size:>10,} B")
    print(f"  {'attention/':20} {n_tensors:>10,} files")
    if (out / "raw").is_dir():
        print(f"  {'raw/':20} {len(list((out / 'raw').iterdir())):>10,} files "
              f"(Flowcept's own dump, pre-purge)")

    failed = sum(1 for ok, _ in checks if not ok)
    print(f"FAILURES: {failed}")
    return failed


def main() -> int:
    args = build_parser().parse_args()
    # Resolve every path before anything chdirs into the run directory, which
    # runner.run does so Flowcept's relative dump lands with its run.
    args.out_dir = str(Path(args.out_dir).resolve())
    args.csv = str(Path(args.csv).resolve())
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    return do_dry_run(args) if args.dry_run else do_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
