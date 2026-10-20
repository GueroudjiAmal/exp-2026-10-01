#!/usr/bin/env python
"""Six ways to see a poison in the provenance, beyond "did the answer change".

    python scripts/poison_deep.py results/poison-suite-<stamp>.json

Every measure here is PAIRED per turn against the clean twin and needs no
labels, so each is usable as a detector, not only as a description.

  1 MASS DISPLACED      total variation between the two attention
                        distributions over all prompt positions. Bounded 0-1
                        and reads directly: "this fraction of the model's
                        attention moved". The span-only delta in
                        poison_report.py cannot see mass that leaves the
                        poisoned span -- and it does: poisoning the quality
                        gate doubles attention on schema_lookup, a span the
                        edit never touched.

  2 WHERE IT WENT       the same displacement, signed, per segment kind. Says
                        which evidence the model abandoned and which it took
                        up instead.

  3 CONCENTRATION       Shannon entropy of the attention distribution, as an
                        effective number of positions attended. Separates a
                        poison that REDIRECTS attention (entropy flat, mass
                        moves) from one that CONFUSES (entropy rises, the
                        model spreads out looking for something).

  4 STEP-0 DIVERGENCE   attention at the FIRST decode step, before a single
                        character of output can differ. If it has already
                        moved, provenance saw the poison land before the text
                        did -- which is the difference between an explanation
                        and an early warning.

  5 LOOKED x FLIPPED    a 2x2. "Flipped without ever retaining the poisoned
                        span" is the prefill blind spot made countable: the
                        connector scores decode queries only, so a poison
                        consumed during prefill changes the answer and leaves
                        no decode trace.

  6 THINKING COST       decode steps, clean vs poisoned. A label-free detector
                        that uses no attention at all, and therefore the
                        baseline the attention measures have to beat. If
                        reasoning length detects as well, provenance is
                        adding nothing here and should say so.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from opal_attn import report  # noqa: E402


def _norm(v):
    t = sum(v)
    return [x / t for x in v] if t > 0 else v


def tv(p, q):
    """Total variation. Equal-length only; a length mismatch means the
    token-parity guarantee failed and the pair is not comparable."""
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q, strict=True))


def eff_support(p):
    h = -sum(x * math.log(x) for x in p if x > 0)
    return math.exp(h)


def load(run: Path):
    meta = json.loads((run / "run_meta.json").read_text())
    turns = {(t["chain_id"], t["round_no"]): t
             for t in report.load_turns(run / "turns.jsonl")}
    prompts = report.load_prompts(run / "prompts.jsonl")
    buf, _ = report.find_run_buffer(run, meta.get("workflow_id"))
    return meta, turns, prompts, report.index_tasks(report.load_buffer(buf))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", type=Path)
    ap.add_argument("--specs", default=str(HERE / "configs" / "poison"))
    args = ap.parse_args()

    from opal_attn.provenance import load_provenance
    suite = json.loads(args.suite.read_text())
    cdir = HERE / "results" / suite["clean"]
    _mc, cturns, cprompts, ctasks = load(cdir)

    print(f"clean twin {suite['clean']}\n")
    print(f"{'arm':9s}{'n':>4s}{'mass moved':>12s}{'step-0':>9s}"
          f"{'eff.support':>13s}{'steps c→p':>13s}{'looked&flip':>12s}"
          f"{'BLIND':>7s}{'resisted':>9s}")

    out_rows = []
    for arm, d in suite.items():
        if arm == "clean":
            continue
        run = HERE / "results" / d
        _mp, pturns, pprompts, ptasks = load(run)
        spec = json.loads((Path(args.specs) / f"{arm}.json").read_text())
        pch = set(spec["chains"])

        tvs, s0s, des, dsteps = [], [], [], []
        flow: dict[str, float] = {}
        lf = lnf = nlf = nlnf = 0
        for k in sorted(set(cturns) & set(pturns)):
            if k[0] not in pch:
                continue
            ct, pt = cturns[k], pturns[k]
            cpr, ppr = cprompts.get(ct["request_id"]), pprompts.get(pt["request_id"])
            crec, prec = ctasks.get(ct["request_id"]), ptasks.get(pt["request_id"])
            if not (cpr and ppr and crec and prec):
                continue
            try:
                ca = report._read(load_provenance, crec, cdir)
                pa = report._read(load_provenance, prec, run)
            except Exception:                                   # noqa: BLE001
                continue
            cs, ps = ca["attn_sum"].tolist(), pa["attn_sum"].tolist()
            if len(cs) != len(ps):
                continue
            cp, pp = _norm(cs), _norm(ps)
            tvs.append(tv(cp, pp))
            des.append(eff_support(pp) / eff_support(cp))

            # 2 — signed flow per declared kind, on the poisoned run's spans
            for g in pt.get("segments", []):
                lo, hi = max(g["lo"], 1), g["hi"]
                if hi > lo:
                    flow[g["kind"]] = flow.get(g["kind"], 0.0) + \
                        (sum(pp[lo:hi]) - sum(cp[lo:hi]))

            # 4 — first decode step only
            try:
                c0 = _norm(ca["val_all_max"].tolist()[0])
                p0 = _norm(pa["val_all_max"].tolist()[0])
                if len(c0) == len(p0):
                    s0s.append(tv(c0, p0))
            except Exception:                                   # noqa: BLE001
                pass

            # 5 — was the poisoned span ever retained, and did the answer move?
            ent = spec["chains"][k[0]]
            needle = ent.get("replace")
            sp = None
            if spec.get("where") != "history" and needle:
                i = ppr["text"].find(needle)
                if i >= 0:
                    # Token span via the line table: find the line whose
                    # character range contains the poison, and take its span.
                    acc, sp = 0, None
                    for lo, hi, _r, tx in ppr["lines"]:
                        if acc <= i < acc + len(tx) + 1 and hi > lo:
                            sp = (lo, hi)
                            break
                        acc += len(tx) + 1
            if sp:
                pos = {p for row in pa["topk_pos"].tolist() for p in row}
                looked = any(sp[0] <= p < sp[1] for p in pos)
                moved = (ct.get("finish_reason") == "stop"
                         and pt.get("finish_reason") == "stop"
                         and ct["got"] != pt["got"])
                lf += looked and moved
                lnf += looked and not moved
                nlf += (not looked) and moved
                nlnf += (not looked) and not moved

            # 6 — thinking cost
            gc, gp = len(ct.get("gen_char_ends") or []), len(pt.get("gen_char_ends") or [])
            if gc and gp:
                dsteps.append((gc, gp))

        if not tvs:
            continue
        mtv = statistics.median(tvs)
        ms0 = statistics.median(s0s) if s0s else None
        mde = statistics.median(des)
        mc = statistics.median([a for a, _ in dsteps]) if dsteps else 0
        mp = statistics.median([b for _, b in dsteps]) if dsteps else 0
        print(f"{arm:9s}{len(tvs):4d}{mtv*100:11.1f}%"
              f"{(f'{ms0*100:.1f}%' if ms0 is not None else '-'):>9s}"
              f"{mde:12.2f}x{f'{mc:.0f}→{mp:.0f}':>13s}"
              f"{lf:12d}{nlf:7d}{lnf:9d}")
        top = sorted(flow.items(), key=lambda kv: -abs(kv[1]))[:3]
        out_rows.append({"arm": arm, "n": len(tvs), "mass_moved": mtv,
                         "step0_tv": ms0, "eff_support_ratio": mde,
                         "steps_clean": mc, "steps_poisoned": mp,
                         "looked_flipped": lf, "blind_flipped": nlf,
                         "looked_resisted": lnf, "neither": nlnf,
                         "flow_top": {k: round(v, 4) for k, v in top}})

    print("\nmass moved = total variation of the whole attention distribution "
          "(0-1).\nstep-0 = the same at the FIRST decode step, before any "
          "output can differ.\neff.support = entropy ratio; >1 means attention "
          "spread out.\nBLIND = answer changed although the poisoned span was "
          "never retained.\n")
    print("where the mass went (top kinds, signed, summed over turns):")
    for r in out_rows:
        s = "  ".join(f"{k} {v:+.3f}" for k, v in r["flow_top"].items())
        print(f"  {r['arm']:9s} {s}")

    o = args.suite.with_name(args.suite.stem + "-deep.json")
    o.write_text(json.dumps(out_rows, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
