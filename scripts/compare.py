#!/usr/bin/env python
"""Tabulate several runs side by side.

    python scripts/compare.py results/2026*-4b results/2026*-14b ...
    python scripts/compare.py results/2026*            # everything

Reads each run's summary.json and run_meta.json. Runs are only comparable when
they covered the SAME chains, so the chain count and turn count are shown
first -- a difference there means the rest of the table is apples to oranges.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))


def load(run: Path) -> tuple[dict | None, str]:
    """`(run, reason_it_was_skipped)`. A skipped run must say WHY.

    "no run directories with a summary.json" is useless when the cause could
    be an unmatched shell glob (bash passes the pattern through literally), a
    run still in progress (summary.json is written last), or a run that died.
    Each of those needs a different response.
    """
    if not run.exists():
        return None, ("no such path -- if this looks like a glob, it matched "
                      "nothing and the shell passed it through literally")
    if not run.is_dir():
        return None, "not a directory"
    s, m = run / "summary.json", run / "run_meta.json"
    if not s.is_file():
        if (run / "run.pid").is_file():
            return None, ("no summary.json yet -- this run looks still in "
                          "progress (run.pid present; summary.json is written "
                          "last)")
        if (run / "run.log").is_file():
            tail = (run / "run.log").read_text(errors="replace").strip().splitlines()
            last = tail[-1][:90] if tail else "(empty)"
            return None, f"no summary.json; the run did not finish. Log ends: {last}"
        return None, "no summary.json and no run.log -- not a run directory"
    out = json.loads(s.read_text())
    out["_meta"] = json.loads(m.read_text()) if m.is_file() else {}
    out["_name"] = run.name
    return out, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()

    runs, skipped = [], []
    for path in sorted(args.runs):
        r, why = load(path)
        (runs.append(r) if r else skipped.append((path, why)))
    for path, why in skipped:
        print(f"skipped {path}: {why}", file=sys.stderr)
    if not runs:
        print("\nNothing to compare. Finished runs in results/:",
              file=sys.stderr)
        base = HERE / "results"
        found = sorted(d for d in base.glob("*") if (d / "summary.json").is_file()) \
            if base.is_dir() else []
        for d in found:
            print(f"  {d.relative_to(HERE)}", file=sys.stderr)
        if not found:
            print("  (none -- no run has completed in this directory)",
                  file=sys.stderr)
        return 2

    def meta(r, k, default="-"):
        return r["_meta"].get(k, default)

    rows = [
        ("model",            lambda r: str(meta(r, "model")).split("/")[-1]),
        ("max_tokens",       lambda r: meta(r, "max_tokens")),
        ("chains",           lambda r: meta(r, "chains")),
        ("turns",            lambda r: r.get("turns")),
        ("scored",           lambda r: r.get("scored", r.get("turns"))),
        ("correct",          lambda r: r.get("correct")),
        ("accuracy",         lambda r: r.get("accuracy")),
        ("format_error",     lambda r: r.get("verdicts", {}).get("format_error", 0)),
        ("unscorable",       lambda r: r.get("verdicts", {}).get("unscorable", 0)),
        ("chains_all_right", lambda r: r.get("chains_fully_correct")),
        ("generate_s",       lambda r: meta(r, "generate_seconds")),
        ("probe_layers",     lambda r: meta(r, "probe_layers_seen")),
    ]

    w = max(len(r["_name"]) for r in runs) + 2
    print(f"{'':<18}" + "".join(f"{r['_name']:<{w}}" for r in runs))
    for label, fn in rows:
        print(f"{label:<18}" + "".join(f"{str(fn(r)):<{w}}" for r in runs))

    # Comparability, stated rather than assumed.
    sigs = {(meta(r, "chains"), r.get("turns")) for r in runs}
    if len(sigs) > 1:
        print("\n!! these runs did NOT cover the same turns -- the numbers above "
              "are not directly comparable")

    # With exactly two runs over the same turns, the net difference hides what
    # happened: +8 could be 8 gains and 0 losses, or 20 gains and 12 losses.
    # Those mean different things, and the turns are PAIRED -- same chains,
    # same model, greedy decoding -- so the flip matrix is the right readout,
    # not two independent proportions.
    if len(runs) == 2:
        import json as _json
        from pathlib import Path as _P

        def verdicts(name):
            for cand in args.runs:
                if _P(cand).name == name:
                    rows = [_json.loads(ln) for ln in
                            (_P(cand) / "turns.jsonl").read_text().splitlines() if ln.strip()]
                    return {(r["chain_id"], r["round_no"]):
                            (r.get("verdict") or ("correct" if r["ok"] else "wrong"))
                            for r in rows}
            return {}

        a, b = (verdicts(r["_name"]) for r in runs)
        shared = sorted(set(a) & set(b))
        if shared:
            both = sum(a[k] == "correct" == b[k] for k in shared)
            only_a = sum(a[k] == "correct" and b[k] != "correct" for k in shared)
            only_b = sum(b[k] == "correct" and a[k] != "correct" for k in shared)
            neither = len(shared) - both - only_a - only_b
            print(f"\npaired over {len(shared)} shared turns:")
            print(f"  both correct            {both}")
            print(f"  only {runs[0]['_name'][-18:]:<18}  {only_a}")
            print(f"  only {runs[1]['_name'][-18:]:<18}  {only_b}")
            print(f"  neither                 {neither}")
            print(f"  turns that FLIPPED      {only_a + only_b}  "
                  f"(net {only_a - only_b:+d} to the first)")
            if only_a + only_b:
                # McNemar, exact. Only the DISCORDANT pairs carry information:
                # a turn both runs got right, or both got wrong, says nothing
                # about the variable under test. Under the null each flip is a
                # coin toss, so the question is whether 12-vs-4 (say) is a
                # plausible draw from 16 tosses. Treating the two accuracies as
                # independent proportions instead would overstate the evidence,
                # because the turns are the same turns.
                from math import comb
                n_d, hi = only_a + only_b, max(only_a, only_b)
                pval = min(1.0, 2 * sum(comb(n_d, i)
                                        for i in range(hi, n_d + 1)) / 2 ** n_d)
                verdict = ("significant at 0.05" if pval < 0.05
                           else "NOT significant at 0.05")
                print(f"  McNemar exact two-sided p = {pval:.4f}  ({verdict})")
                if pval >= 0.05:
                    print("  the direction may still be real -- this says the "
                          "sample cannot\n  establish it. More chains is the "
                          "remedy, not more interpretation.")

    print("\nper-task accuracy (scored turns only):")
    tasks = sorted({t for r in runs for t in r.get("by_task", {})})
    print(f"{'':<26}" + "".join(f"{r['_name'][:w-2]:<{w}}" for r in runs))
    for t in tasks:
        cells = []
        for r in runs:
            d = r.get("by_task", {}).get(t)
            cells.append("-" if not d else
                         ("n/a" if d.get("acc") is None else
                          f"{d['acc']:.2f} ({d['n']})"))
        print(f"  {t:<24}" + "".join(f"{c:<{w}}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
