#!/usr/bin/env python
"""Where did this run fail -- which chains, which depth, and did it look?

    python scripts/failures.py results/<run> [--csv data/opal_chains.csv]
    python scripts/failures.py results/<run> --no-tensors   # skip attention

Answers three questions, in order of how much they constrain an explanation:

  1. WHICH turns fail -- by difficulty tier and by round depth, separated,
     because round is confounded with task type in this benchmark and the two
     look identical if you only read `by_round`.
  2. WHERE in a chain -- is a failure the chain's FIRST, or downstream of an
     earlier one? Only the first failures are about the question; the rest may
     be about the context.
  3. DID IT LOOK -- for failures the CSV attributes to ignoring a specific
     tool, how much attention landed on that tool's span, compared with turns
     of the same task that were answered correctly. This is the question the
     capture exists to answer: did the model consult the evidence and misread
     it, or never consult it.

Verdicts are re-graded from the stored raw replies under the current rules, so
runs made before a scoring change stay comparable.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from opal_attn import report  # noqa: E402
from opal_attn.answers import classify, parse_answer  # noqa: E402
from opal_attn.chains import load_chains  # noqa: E402

# Which segment kind each documented shortcut says was ignored. Only shortcuts
# that name a tool are listed; "stepping to the next column" is a misreading of
# a table that WAS consulted, so it has no target and is left out.
SHORTCUT_TARGET = {
    "ignoring the quality gate": "qc_lookup",
    "skipping the exclusions": "qc_lookup",
    "ignoring empty_mask": "qc_lookup",
    "ignoring the schedule": "design_lookup",
    "judging a source about another species": "literature_lookup",
    "taking the largest round-to-round value without the within-round table":
        "db_lookup",
}


def pct(n, d):
    return f"{n/d:.0%}" if d else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--csv", type=Path, default=HERE / "data" / "opal_chains.csv")
    ap.add_argument("--no-tensors", action="store_true")
    args = ap.parse_args()

    turns = report.load_turns(args.run / "turns.jsonl")
    rows = {(r.chain_id, r.round_no): r
            for c in load_chains(args.csv) for r in c.rounds}
    for t in turns:
        row = rows.get((t["chain_id"], t["round_no"]))
        if row is not None:
            got = parse_answer(t.get("raw"))
            t["verdict"], t["shortcut"] = classify(got, row)
            t["got"] = got

    scored = [t for t in turns if t["verdict"] in ("correct", "wrong")]
    bad = [t for t in scored if t["verdict"] == "wrong"]
    print(f"{args.run.name}: {len(turns)} turns, {len(scored)} scored, "
          f"{len(bad)} wrong\n")

    # ---- 1. tier vs depth, separately ------------------------------------
    print("=== 1. who fails ===")
    print("  by tier (difficulty is declared in the chain id):")
    for tier in ("low", "medium", "hard"):
        g = [t for t in scored if t["tier"] == tier]
        w = [t for t in g if t["verdict"] == "wrong"]
        print(f"    {tier:8} {len(w):4}/{len(g):4} wrong  {pct(len(w), len(g))}")

    print("  by round depth (CONFOUNDED with task type -- each round is almost")
    print("  always the same task across chains, so this is difficulty, not depth):")
    by_round = collections.defaultdict(list)
    for t in scored:
        by_round[t["round_no"]].append(t)
    for rd in sorted(by_round):
        g = by_round[rd]
        w = [t for t in g if t["verdict"] == "wrong"]
        tasks = {t["task_type"] for t in g}
        print(f"    round {rd}  {len(w):4}/{len(g):4} wrong  {pct(len(w), len(g)):>5}"
              f"   task: {'/'.join(sorted(tasks))}")

    # ---- 2. first failure vs downstream ----------------------------------
    print("\n=== 2. where in the chain ===")
    by_chain = collections.defaultdict(list)
    for t in turns:
        by_chain[t["chain_id"]].append(t)
    first_at, downstream = collections.Counter(), [0, 0]
    clean_chains = 0
    for ts in by_chain.values():
        ts.sort(key=lambda x: x["round_no"])
        wrongs = [t for t in ts if t["verdict"] == "wrong"]
        if not wrongs:
            clean_chains += 1
            continue
        first_at[wrongs[0]["round_no"]] += 1
        after = [t for t in ts if t["round_no"] > wrongs[0]["round_no"]
                 and t["verdict"] in ("correct", "wrong")]
        downstream[0] += sum(t["verdict"] == "wrong" for t in after)
        downstream[1] += len(after)
    print(f"  chains with no wrong answer : {clean_chains}/{len(by_chain)}")
    print(f"  chain's FIRST failure at round: "
          f"{dict(sorted(first_at.items()))}")
    print(f"  turns after a chain's first failure: {downstream[0]}/{downstream[1]} "
          f"wrong ({pct(downstream[0], downstream[1])})")

    print("\n  how the failures are explained:")
    sc = collections.Counter(t["shortcut"] for t in bad if t["shortcut"])
    for name, n in sc.most_common():
        print(f"    {n:4}  {name}")
    print(f"    {sum(1 for t in bad if not t['shortcut']):4}  (no documented "
          f"shortcut matches)")

    # ---- 3. did it look? --------------------------------------------------
    if args.no_tensors:
        return 0
    print("\n=== 3. did it look at the evidence? ===")
    buf_path, notes = report.find_run_buffer(args.run)
    for n in notes:
        print(f"  !! {n}")
    if buf_path is None:
        print("  no provenance; skipping")
        return 0
    buffer = report.load_buffer(buf_path)
    tasks_idx = report.index_tasks(buffer)
    from opal_attn.provenance import load_provenance

    # mass per segment KIND for one turn: the share of retained attention that
    # landed in segments of that kind, averaged over recorded decode steps.
    def mass_by_kind(t):
        rec = tasks_idx.get(t.get("request_id"))
        if rec is None:
            return None
        a = load_provenance(rec, run_dir=args.run)
        pos, val = a["topk_pos"].tolist(), a["val_all_max"].tolist()
        segs = [(lo, hi) for lo, hi, _k in a["segments"].tolist()]
        kind_of = {}
        for d in t.get("segments", []):
            kind_of[(d["lo"], d["hi"])] = d["kind"]
        if not pos or not any(p > 0 for row in pos for p in row):
            return None
        out, total = collections.Counter(), 0.0
        for prow, vrow in zip(pos, val, strict=True):
            for p, v in zip(prow, vrow, strict=True):
                if p <= 0:
                    continue
                for lo, hi in segs:
                    if lo <= p < hi:
                        out[kind_of.get((lo, hi), "prose gap")] += v
                        total += v
                        break
        return {k: v / total for k, v in out.items()} if total else None

    # Compare turns that took a tool-ignoring shortcut against turns of the
    # SAME task answered correctly. Same task matters: segment kinds and their
    # sizes differ per task, so a cross-task average would compare apples to
    # oranges.
    groups = collections.defaultdict(lambda: {"shortcut": [], "correct": []})
    examined = 0
    for t in scored:
        target = SHORTCUT_TARGET.get(t.get("shortcut") or "")
        if t["verdict"] == "wrong" and not target:
            continue
        m = mass_by_kind(t)
        if m is None:
            continue
        examined += 1
        if t["verdict"] == "wrong":
            groups[(t["task_type"], target)]["shortcut"].append(m.get(target, 0.0))
        else:
            for tgt in set(SHORTCUT_TARGET.values()):
                if tgt in m:
                    groups[(t["task_type"], tgt)]["correct"].append(m[tgt])

    print(f"  turns with usable attention: {examined}")
    print(f"  {'task / ignored span':<46} {'shortcut':>10} {'correct':>10} {'delta':>8}")
    any_row = False
    for (task, tgt), d in sorted(groups.items()):
        if len(d["shortcut"]) < 3 or len(d["correct"]) < 3:
            continue
        a, b = statistics.mean(d["shortcut"]), statistics.mean(d["correct"])
        print(f"  {task + ' / ' + tgt:<46} {a:>10.3f} {b:>10.3f} {a-b:>+8.3f}"
              f"   n={len(d['shortcut'])}/{len(d['correct'])}")
        any_row = True
    if not any_row:
        print("  (no task had >=3 turns in both groups -- needs more chains)")
    else:
        print("\n  negative delta = the shortcut turns put LESS attention on the")
        print("  span the CSV says they ignored, i.e. they did not look.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
