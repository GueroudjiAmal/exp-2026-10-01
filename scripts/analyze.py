#!/usr/bin/env python
"""Verify and report on a finished run directory -- on any machine.

Capture happens on Polaris; this is how you look at it afterwards, including
from a laptop with no CUDA and no vLLM. It reads `turns.jsonl`, the Flowcept
JSONL dump and the SafeTensors files, and re-runs exactly the checks and the
report that `run_experiment.py` prints at the end of a run.

    python scripts/analyze.py results/12345                     # on the node
    python scripts/analyze.py ./pulled-run --no-tensors         # descriptors only

`--no-tensors` skips anything that opens a SafeTensors file, for when you have
pulled back the JSONL but not the (larger) tensor directory.

Exit status is the number of failed checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from opal_attn import report  # noqa: E402
from opal_attn.answers import classify, parse_answer  # noqa: E402
from opal_attn.chains import load_chains  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--no-tensors", action="store_true",
                    help="descriptor-level checks only; do not open any file")
    ap.add_argument("--top-pct", type=float, default=None,
                    help="expected top_pct; defaults to what run_meta.json says")
    ap.add_argument("--rescore", nargs="?", const=str(HERE / "data" / "opal_chains.csv"),
                    default=None, metavar="CSV",
                    help="re-grade the stored replies against the CSV with the "
                         "current scoring rules, instead of trusting the `ok` "
                         "the run recorded. Needs no GPU: it reads the `got` "
                         "text already in turns.jsonl. Reports what changed.")
    args = ap.parse_args()

    run = args.run_dir
    turns_path = run / "turns.jsonl"
    if not turns_path.is_file():
        raise SystemExit(f"no turns.jsonl in {run}")
    meta = json.loads((run / "run_meta.json").read_text()) if (run / "run_meta.json").is_file() else {}
    buf_path, notes = report.find_run_buffer(run, meta.get("workflow_id"))
    for n in notes:
        print(f"!! {n}")
    if buf_path is None:
        raise SystemExit(f"no provenance.jsonl or flowcept_buffer*.jsonl in {run}")

    turns = report.load_turns(turns_path)
    buffer = report.load_buffer(buf_path)
    top_pct = args.top_pct if args.top_pct is not None else meta.get("top_pct", 10.0)

    load_provenance = None
    if not args.no_tensors:
        try:
            from opal_attn.provenance import load_provenance
        except ImportError as e:
            print(f"note: {e}\n      continuing with descriptor-level checks only")

    if args.rescore:
        # Scoring rules change (set-valued and composite answers were being
        # marked wrong); the model's replies do not. Re-grading from the stored
        # text is cheap and auditable, and beats re-running 430 generations.
        rows = {(r.chain_id, r.round_no): r
                for c in load_chains(args.rescore) for r in c.rounds}
        changed = []
        for t in turns:
            row = rows.get((t["chain_id"], t["round_no"]))
            if row is None:
                continue
            # Re-parse from the RAW reply, not the stored `got`: the stored
            # value was produced by the old parser, which fell back to the last
            # line when there was no ANSWER:. Re-grading that would carry the
            # old leniency into the new verdict.
            got = parse_answer(t.get("raw"))
            verdict, shortcut = classify(got, row,
                                         t.get('finish_reason'))
            was = t.get("verdict") or ("correct" if t.get("ok") else "wrong")
            if verdict != was:
                changed.append((t["chain_id"], t["round_no"], t["task_type"],
                                got, t["want"], was, verdict))
            t["got"], t["verdict"] = got, verdict
            t["ok"], t["shortcut"] = verdict == "correct", shortcut
        print(f"rescored   {len(turns)} turns against {args.rescore}")
        import collections as _c
        moves = _c.Counter(f"{w} -> {n}" for *_x, w, n in changed)
        print(f"           {len(changed)} verdict(s) changed")
        for move, k in moves.most_common():
            print(f"             {k:4d}  {move}")
        for cid, rd, task, got, want, was, now in changed[:6]:
            print(f"             e.g. {cid} r{rd} {task}: {got!r} vs {want!r}"
                  f"  {was} -> {now}")
        if len(changed) > 10:
            print(f"             ... and {len(changed) - 10} more")
        print()

    print(f"run        {run}")
    print(f"buffer     {buf_path.name}  ({len(buffer)} records)")
    print(f"turns      {len(turns)}")
    if meta:
        print(f"model      {meta.get('model')}  connector={meta.get('connector')} "
              f"tp={meta.get('tensor_parallel_size')}")

    print("\n=== capture verification ===")
    # run_dir matters: the provenance records an absolute uri from the machine
    # that captured the run, so a directory pulled off Polaris needs rebasing.
    checks = report.verify(turns, buffer, top_pct=top_pct,
                           load_provenance=load_provenance, run_dir=run, meta=meta)
    for ok, msg in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}")

    summary = report.summarise(turns, meta)
    print("\n=== accuracy (reported, not asserted) ===")
    for k, v in summary.items():
        print(f"  {k:<30} {v}")

    prompts_path = run / "prompts.jsonl"
    if load_provenance is not None and prompts_path.is_file():
        dens = report.density_by_distance(
            turns, buffer, load_provenance, report.load_prompts(prompts_path),
            run_dir=run)
        print("\n=== attention density (content-renormalised, per token, "
              "by distance from the decode point) ===")
        print("  structural share: " + "  ".join(
            f"{k}={v}" for k, v in sorted(dens["structural_share"].items(),
                                          key=lambda kv: -kv[1])))
        hdr = "".join(f"{str(tuple(b)):>14s}" for b in dens["bands"])
        print(f"  {'kind':18s}{hdr}{'n':>7s}")
        for k, row in sorted(dens["density"].items(),
                             key=lambda kv: -(kv[1]["by_band"][2] or -1)):
            cells = "".join(f"{(str(x) if x is not None else '-'):>14s}"
                            for x in row["by_band"])
            print(f"  {k:18s}{cells}{row['instances']:7d}")
        summary["density_by_distance"] = dens

    if load_provenance is not None:
        hot = report.hottest_segments(turns, buffer, load_provenance, run_dir=run)
        print("\n=== where attention went ===")
        for k, v in hot.items():
            print(f"  {k:<30} {v}")
        summary["attention"] = hot

    summary["verification"] = [{"passed": ok, "check": m} for ok, m in checks]
    (run / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {run / 'analysis.json'}")

    failed = sum(1 for ok, _ in checks if not ok)
    print(f"FAILURES: {failed}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
