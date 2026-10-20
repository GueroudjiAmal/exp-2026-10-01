#!/usr/bin/env python
"""Build one poison manifest per arm, with token parity enforced per chain.

    python scripts/build_poison_manifest.py --limit 20 --model Qwen/Qwen3.5-35B-A3B

Writes configs/poison/<arm>.json and prints a coverage table.

WHY A BUILD STEP. The text to poison differs per chain, so an arm is not one
edit but twenty, each of which must (a) match exactly once and (b) preserve the
token count under the run's own tokenizer. Both are checked here, with no GPU,
so an unbalanced substitution is a build error rather than a silently re-based
distance axis discovered after a capture.

COVERAGE IS NOT UNIFORM AND THAT IS THE DATA, NOT A BUG. Seven of the first
twenty chains have no quality gate in round 1 at all -- one rule, one tool --
so the qc, rules and schema arms cannot apply to them. Each arm therefore
records the chains it covers, and cross-arm comparisons are valid only on the
intersection. Per-arm clean-vs-poisoned pairing is unaffected, since that
pairs a chain with itself.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from opal_attn.chains import load_chains, select  # noqa: E402
from opal_attn.poison import PoisonError, find_balanced, locate  # noqa: E402

# Each arm: (name, form, kind, locator -> (find, [candidate replacements])).
# Candidates are SEMANTICALLY equivalent phrasings of one intended corruption;
# the tokenizer picks whichever preserves the token count. Meaning is fixed by
# the author, wording varies only to satisfy the measurement's constraint.

_FLOOR = re.compile(r"floor for a panel claim is (\d+)\.")
_GATE = re.compile(r"must clear the usable-plant floor")
_OUTFMT = re.compile(r"ANSWER: <dose in mg kg-1, digits only>")
_SCHEMA = re.compile(r"^(\s+\S+\s+\S+\s+\S+\s+\S+\s+)plant\s*$", re.M)


def _qc(prompt, row):  # noqa: ARG001
    m = _FLOOR.search(prompt)
    if not m:
        return None
    d = len(m.group(1))
    lows = ["0" * (d - 1) + "1", "0" * (d - 1) + "2", "0" * d, "1".rjust(d, "0")]
    return (m.group(0), [f"floor for a panel claim is {v}." for v in dict.fromkeys(lows)])


def _rules(prompt, row):  # noqa: ARG001
    if not _GATE.search(prompt):
        return None
    return ("must clear the usable-plant floor",
            ["may ignore the usable-plant floor",
             "need not clear the usable-plant floor",
             "can disregard the usable-plant floor",
             "may overlook the usable-plant floor"])


def _outfmt(prompt, row):  # noqa: ARG001
    if not _OUTFMT.search(prompt):
        return None
    return ("ANSWER: <dose in mg kg-1, digits only>",
            ["ANSWER: <dose in mg kg-1, highest only>",
             "ANSWER: <dose in mg kg-1, largest only>",
             "ANSWER: <dose in mg kg-1, maximum only>",
             "ANSWER: <peak in mg kg-1, digits only>"])


def _schema(prompt, row):  # noqa: ARG001
    hits = _SCHEMA.findall(prompt)
    if len(hits) != 1:
        return None
    m = _SCHEMA.search(prompt)
    return (m.group(0).rstrip("\n"),
            [m.group(1) + "region", m.group(1) + "sector", m.group(1) + "area"])


def _null(prompt, row):  # noqa: ARG001
    """Change a number that cannot affect the answer: the rounds_scheduled-style
    trailing count in a non-decisive design row, else the last digit of the
    experiment id in the SESSION line. Deliberately inert."""
    m = re.search(r"(experiment )(\d+)( at the)", prompt)
    if not m:
        return None
    alt = str(int(m.group(2)) + 1).rjust(len(m.group(2)), "0")
    return (m.group(0), [f"{m.group(1)}{alt}{m.group(3)}"])


def _history(prompt, row):
    """Substitute round 1's answer with a different, valid-looking dose."""
    doses = re.findall(r"\bt(\d+)\b", prompt)
    alt = next((d for d in dict.fromkeys(doses) if d != row.correct_answer), None)
    if alt is None:
        return None
    return (None, [f"ANSWER: {alt}"])


ARMS = [
    ("outfmt",  "prose",      "output_format", _outfmt,  "prompt"),
    ("rules",   "prose",      "rules",         _rules,   "prompt"),
    ("qc",      "data-table", "qc_lookup",     _qc,      "prompt"),
    ("schema",  "data-table", "schema_lookup", _schema,  "prompt"),
    ("history", "history",    "answer",        _history, "history"),
    ("null",    "control",    "session",       _null,    "prompt"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(HERE / "data" / "opal_chains.csv"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=str(HERE / "configs" / "poison"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    chains = select(load_chains(args.csv), limit=args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    coverage: dict[str, set[str]] = {}
    print(f"{'arm':10s}{'form':12s}{'covered':>9s}{'no anchor':>11s}{'unbalanced':>12s}")
    for name, form, kind, locator, where in ARMS:
        spec = {"arm": name, "form": form, "kind": kind,
                "model": args.model, "round_no": 1, "where": where, "chains": {}}
        noanchor = unbalanced = 0
        for c in chains:
            row = c.rounds[0]
            found = locator(row.prompt, row)
            if found is None:
                noanchor += 1
                continue
            find, cands = found
            if where == "history":
                spec["chains"][c.chain_id] = {"replace": cands[0]}
                continue
            try:
                ranked = find_balanced(tok, row.prompt, find, cands)
            except PoisonError:
                noanchor += 1
                continue
            best = next((c_ for c_, d in ranked if d == 0), None)
            if best is None:
                unbalanced += 1
                continue
            lo, hi = locate(row.prompt, find)
            spec["chains"][c.chain_id] = {"find": find, "replace": best,
                                          "char_lo": lo, "char_hi": hi}
        (out / f"{name}.json").write_text(json.dumps(spec, indent=2) + "\n")
        coverage[name] = set(spec["chains"])
        print(f"{name:10s}{form:12s}{len(spec['chains']):9d}{noanchor:11d}{unbalanced:12d}")

    common = set.intersection(*coverage.values()) if coverage else set()
    print(f"\ncommon chain set (all arms apply): {len(common)} of {len(chains)}")
    print("cross-arm comparisons are valid only on that intersection; "
          "per-arm clean-vs-poisoned pairing uses each arm's own coverage.")
    (out / "coverage.json").write_text(json.dumps(
        {"per_arm": {k: sorted(v) for k, v in coverage.items()},
         "common": sorted(common)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
