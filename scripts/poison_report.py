#!/usr/bin/env python
"""How effective was each poison, and did the provenance see it?

    python scripts/poison_report.py results/poison-suite-<stamp>.json

Reads the suite map written by run_poison_suite.sh and reports, per arm,
against the clean twin:

  EFFECT   -- did the answer change, on turns where BOTH arms completed
  COST     -- did the poison push the model out of its token budget
  EVIDENCE -- did attention on the poisoned span change, paired per turn

THREE THINGS THIS IS CAREFUL ABOUT.

Truncated turns are not scored. A reply cut off mid-reasoning never committed
to an answer, and `parse_answer` will happily scrape a fragment out of the
unfinished thought. Flips are therefore counted only where clean AND poisoned
both finished; truncation is reported separately as its own effect, because a
poison that makes the model think past its budget has done something real even
though it changed no answer.

Attention mass comes from `attn_sum`, which the connector accumulates over all
positions BEFORE top_pct selection. It can be summed over any span, including
one the run never declared, so clean and poisoned use identical segmentation
and the delta is attributable to the text rather than to the partition.

Shares are renormalised over CONTENT tokens -- position 0, blank lines and
chat markers carry no information and take 15% of the mass on this model and
55% on Qwen3-4B, so a raw share is partly a measure of sink behaviour.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from opal_attn import report  # noqa: E402


def _content_mass(s, lines):
    return sum(sum(s[max(lo, 1):hi]) for lo, hi, _r, tx in lines
               if hi > lo and tx.strip() and not tx.startswith("<|im_"))


_OFFS: dict[int, list] = {}


def _span_of(text, needle, tok):
    """Token span of `needle`'s FIRST occurrence; None if absent.

    First rather than unique: the poison sits in round 1, which is the start
    of every rendered conversation, and a value may legitimately recur in a
    later round's tables.
    """
    i = text.find(needle)
    if i < 0:
        return None
    key = hash(text)
    if key not in _OFFS:
        _OFFS[key] = tok(text, add_special_tokens=False,
                         return_offsets_mapping=True)["offset_mapping"]
    offs = _OFFS[key]
    lo = next((j for j, (a, b) in enumerate(offs) if b > i), None)
    hi = next((j for j, (a, b) in enumerate(offs) if a >= i + len(needle)), len(offs))
    return (lo, hi) if lo is not None and hi > lo else None


def load_arm(run_dir: Path):
    meta = json.loads((run_dir / "run_meta.json").read_text())
    turns = report.load_turns(run_dir / "turns.jsonl")
    prompts = report.load_prompts(run_dir / "prompts.jsonl")
    buf, _ = report.find_run_buffer(run_dir, meta.get("workflow_id"))
    tasks = report.index_tasks(report.load_buffer(buf))
    by = {(t["chain_id"], t["round_no"]): t for t in turns}
    return meta, by, prompts, tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", type=Path)
    ap.add_argument("--specs", default=str(HERE / "configs" / "poison"))
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--floor", default=None, metavar="RUN",
                    help="a SECOND clean run with identical config. Its "
                         "difference from the first clean run on the very "
                         "same spans is the noise floor: at T=0 with "
                         "token-parity-preserving poisons the prompts and "
                         "batch shapes are identical, so whatever separates "
                         "two clean runs is engine nondeterminism and nothing "
                         "else. Without it a delta has no denominator.")
    args = ap.parse_args()

    suite = json.loads(args.suite.read_text())
    clean_dir = HERE / "results" / suite["clean"]
    meta0 = json.loads((clean_dir / "run_meta.json").read_text())

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(meta0["tokenizer"])
    load_prov = None
    if not args.no_attention:
        from opal_attn.provenance import load_provenance as load_prov

    _m0, clean, cprompts, ctasks = load_arm(clean_dir)
    floor_dir = (HERE / "results" / args.floor) if args.floor else None
    floor = fprompts = ftasks = None
    if floor_dir is not None:
        _mf, floor, fprompts, ftasks = load_arm(floor_dir)
        same = [k for k in clean if k in floor
                and clean[k].get("finish_reason") == "stop"
                and floor[k].get("finish_reason") == "stop"]
        diff = [k for k in same if clean[k]["got"] != floor[k]["got"]]
        print(f"noise floor: {floor_dir.name}")
        print(f"  answers differing between two identical clean runs: "
              f"{len(diff)}/{len(same)}")
        for k in diff[:4]:
            print(f"    {k[0]} r{k[1]}: {clean[k]['got']!r} vs {floor[k]['got']!r}")
        print()

    print(f"clean twin: {suite['clean']}   model {meta0['model']}  "
          f"T={meta0['temperature']}  max_tokens={meta0['max_tokens']}\n")
    print(f"{'arm':9s}{'n pois':>7s}{'both done':>10s}{'FLIPS':>7s}{'flip%':>7s}"
          f"{'trunc p/c':>11s}{'att clean':>10s}{'att pois':>9s}{'delta%':>8s}"
          f"{'FLOOR' if args.floor else '':>9s}")

    rows = []
    for arm, d in suite.items():
        if arm == "clean":
            continue
        run = HERE / "results" / d
        meta, poi, pprompts, ptasks = load_arm(run)
        spec = json.loads((Path(args.specs) / f"{arm}.json").read_text())
        pchains = set(spec["chains"])

        keys = [k for k in poi if k[0] in pchains and k in clean]
        both = [k for k in keys
                if poi[k].get("finish_reason") == "stop"
                and clean[k].get("finish_reason") == "stop"]
        flips = [k for k in both if poi[k]["got"] != clean[k]["got"]]
        tr_p = sum(1 for k in keys if poi[k].get("finish_reason") == "length")
        tr_c = sum(1 for k in keys if clean[k].get("finish_reason") == "length")

        ac = ap_ = None
        paired = []
        if load_prov is not None:
            cm, pm = [], []  # per-turn density on the poisoned span
            # PAIRED per-turn records. The medians above cannot show the one
            # thing the 16k pair revealed -- that efficacy and detectability
            # both fall as the model deliberates longer -- because that is an
            # interaction, and an interaction needs the turns, not a summary.
            paired = []
            for k in keys:
                ent = spec["chains"][k[0]]
                needle_c = ent.get("find") or None
                needle_p = ent["replace"]
                if spec.get("where") == "history":
                    needle_c, needle_p = None, needle_p
                # The run directory must travel with the arm. The connector
                # records an absolute uri from the capture host, and
                # provenance rebases it onto the directory it is given -- so
                # handing the clean record the POISONED directory makes every
                # clean read miss, silently, and both medians come back empty.
                for store, turns_, prompts_, tasks_, needle, rdir in (
                        (cm, clean, cprompts, ctasks, needle_c, clean_dir),
                        (pm, poi, pprompts, ptasks, needle_p, run)):
                    t = turns_[k]
                    pr = prompts_.get(t.get("request_id"))
                    rec = tasks_.get(t.get("request_id"))
                    if pr is None or rec is None:
                        continue
                    nd = needle
                    if nd is None:              # history clean arm: round-1 answer
                        a1 = [g for g in t["segments"]
                              if g["kind"] == "answer" and g.get("round_hint") == 1]
                        if not a1:
                            continue
                        sp = (a1[0]["lo"], a1[0]["hi"])
                    else:
                        sp = _span_of(pr["text"], nd, tok)
                        if sp is None:
                            continue
                    try:
                        arr = report._read(load_prov, rec, rdir)
                    except Exception:           # noqa: BLE001
                        continue
                    if "attn_sum" not in arr:
                        continue
                    s = arr["attn_sum"].tolist()
                    cmass = _content_mass(s, pr["lines"])
                    if cmass <= 0:
                        continue
                    lo, hi = max(sp[0], 1), sp[1]
                    if hi > lo:
                        store.append(sum(s[lo:hi]) / cmass / (hi - lo) * 1000)
                if len(cm) == len(pm) and cm:
                    paired.append({
                        "key": f"{k[0]}:r{k[1]}",
                        "clean": cm[-1], "pois": pm[-1],
                        "steps": len(poi[k].get("gen_char_ends") or []),
                        "flipped": bool(
                            poi[k].get("finish_reason") == "stop"
                            and clean[k].get("finish_reason") == "stop"
                            and poi[k]["got"] != clean[k]["got"]),
                        "truncated": poi[k].get("finish_reason") == "length"})
            if cm and pm:
                ac, ap_ = statistics.median(cm), statistics.median(pm)

        # The same span measured between the two CLEAN runs. Computed per
        # arm rather than once, because the floor is span-dependent: a short
        # dense span and a long sparse one do not carry the same jitter.
        fl = None
        if floor is not None and load_prov is not None:
            f1, f2 = [], []
            for k in keys:
                ent = spec["chains"][k[0]]
                nd = ent.get("find") or None
                for store, turns_, prompts_, tasks_, rdir in (
                        (f1, clean, cprompts, ctasks, clean_dir),
                        (f2, floor, fprompts, ftasks, floor_dir)):
                    t = turns_.get(k)
                    if t is None:
                        continue
                    pr = prompts_.get(t.get("request_id"))
                    rec = tasks_.get(t.get("request_id"))
                    if pr is None or rec is None:
                        continue
                    if nd is None:
                        a1 = [g for g in t["segments"] if g["kind"] == "answer"
                              and g.get("round_hint") == 1]
                        if not a1:
                            continue
                        sp = (a1[0]["lo"], a1[0]["hi"])
                    else:
                        sp = _span_of(pr["text"], nd, tok)
                        if sp is None:
                            continue
                    try:
                        arr = report._read(load_prov, rec, rdir)
                    except Exception:                       # noqa: BLE001
                        continue
                    if "attn_sum" not in arr:
                        continue
                    sarr = arr["attn_sum"].tolist()
                    cmass = _content_mass(sarr, pr["lines"])
                    lo, hi = max(sp[0], 1), sp[1]
                    if cmass > 0 and hi > lo:
                        store.append(sum(sarr[lo:hi]) / cmass / (hi - lo) * 1000)
            if f1 and f2:
                m1, m2 = statistics.median(f1), statistics.median(f2)
                fl = 100.0 * (m2 - m1) / m1 if m1 > 0 else None

        pct = 100.0 * len(flips) / len(both) if both else float("nan")
        dl = (100.0 * (ap_ - ac) / ac) if (ac and ap_ is not None and ac > 0) else None
        print(f"{arm:9s}{len(pchains):7d}{len(both):10d}{len(flips):7d}{pct:6.1f}%"
              f"{f'{tr_p}/{tr_c}':>11s}"
              f"{(f'{ac:.3f}' if ac else '-'):>10s}{(f'{ap_:.3f}' if ap_ else '-'):>9s}"
              f"{(f'{dl:+.0f}%' if dl is not None else '-'):>8s}"
              f"{(f'{fl:+.1f}%' if fl is not None else ''):>9s}")
        rows.append({"arm": arm, "form": spec["form"], "kind": spec["kind"],
                     "chains": len(pchains), "comparable": len(both),
                     "flips": len(flips), "flip_pct": round(pct, 2),
                     "truncated_poisoned": tr_p, "truncated_clean": tr_c,
                     "att_density_clean": ac, "att_density_poisoned": ap_,
                     "att_delta_pct": dl, "att_floor_pct": fl,
                     # Kept so a figure can show the SPREAD, not just the
                     # median: a median moved by two outliers and a median
                     # moved by every turn look identical in a table.
                     "att_turns_clean": cm if load_prov is not None else [],
                     "att_turns_poisoned": pm if load_prov is not None else [],
                     "paired": paired if load_prov is not None else [],
                     "flipped_turns": [f"{c}:r{r}" for c, r in flips]})

    # PER ROUND. The poison is injected at round 1 and recedes as the
    # conversation grows, so this is the distance axis -- observed, not
    # manipulated. It is the only view that can test whether an effect
    # persists or washes out, which is the whole laundering question.
    print(f"\n{'arm':9s}" + "".join(f"{'r'+str(r):>9s}" for r in range(1, 8)))
    for r_ in rows:
        arm = r_["arm"]
        run = HERE / "results" / suite[arm]
        _m, poi, _pp, _pt = load_arm(run)
        spec = json.loads((Path(args.specs) / f"{arm}.json").read_text())
        pch = set(spec["chains"])
        cells = ""
        for rn in range(1, 8):
            ks = [k for k in poi if k[0] in pch and k[1] == rn and k in clean
                  and poi[k].get("finish_reason") == "stop"
                  and clean[k].get("finish_reason") == "stop"]
            if not ks:
                cells += f"{'-':>9s}"
                continue
            f = sum(1 for k in ks if poi[k]["got"] != clean[k]["got"])
            cells += f"{f}/{len(ks)}".rjust(9)
        print(f"{arm:9s}{cells}")
    print("flips/comparable per round. '-' = no turn where both arms completed.")

    out = args.suite.with_name(args.suite.stem + "-report.json")
    out.write_text(json.dumps({"clean": suite["clean"], "arms": rows}, indent=2) + "\n")
    print(f"\nwrote {out}")
    print("\nflip% is over turns where BOTH arms completed. 'trunc p/c' is "
          "truncated turns poisoned/clean -- an arm that truncates more has "
          "changed behaviour even where it changed no answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
