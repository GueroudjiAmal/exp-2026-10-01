#!/usr/bin/env python
"""Draw one turn's attention over the prompt, one column per answer item.

    python scripts/heatmap.py results/<run> --chain exp32-chain-0020-hard --round 5
    python scripts/heatmap.py results/<run> --request-id 1234
    python scripts/heatmap.py results/<run> --chain ... --round 5 --top 40

This is the per-turn counterpart to `scripts/failures.py`, which aggregates
over a whole run. The question here is the one the chains were built for, asked
of a single turn: when the model answered round 5, which LINES of the
conversation was it looking at -- and did the two halves of a two-part answer
look at different things?

HEAT is `val_all_max`: per line the max over its retained positions,
aggregated across that item's decode steps with exponential decay `0.5**k`.
Max-with-decay ranks lines so that noise does not outrank evidence, which is a
different objective from estimating total attention mass -- see
`report.heat_lines` for why each choice is the way it is.

Each column is scaled to its OWN maximum. The two items of one answer are not
on a comparable scale, and sharing one would make the weaker column vanish.

Only decode attention is captured, so this shows what the model consulted while
WRITING the answer, not what it assembled during prefill. A blank line is
either level 0 or a line on which no position survived top_pct selection; the
two are not distinguishable after the fact and the legend says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from opal_attn import report  # noqa: E402
from opal_attn.chains import load_chains  # noqa: E402

RAMP = " .:-=+*#%@"      # level 0..9


def level(v: float, vmax: float) -> str:
    if v <= 0 or vmax <= 0:
        return RAMP[0]
    return RAMP[min(9, max(1, round(9 * v / vmax)))]


def gutter_labels(prompt: dict, segments: list[dict]) -> list[str]:
    """`r4·qc` for each line, from the declared segment that contains it.

    Lines in undeclared prose get a blank gutter rather than a guess. That the
    gutter is blank for the round's own question text, the RULES and the
    OUTPUT FORMAT footer is not a rendering gap -- those genuinely are not
    declared segments, and a line there is exactly what the run-level report
    buckets as an unattributed "prose gap".
    """
    spans = sorted(((s["lo"], s["hi"], s["kind"], s.get("round_hint", 0))
                    for s in segments), key=lambda x: x[0])
    out = []
    for lo, hi, _r, _text in ((ln[0], ln[1], ln[2], ln[3])
                              for ln in prompt.get("lines") or []):
        tag = ""
        if hi > lo:
            for slo, shi, kind, rh in spans:
                if slo < hi and lo < shi:          # overlaps this segment
                    short = {"answer": "ANS"}.get(kind, kind.split("_")[0][:3])
                    tag = f"r{rh}·{short}"
                    break
        out.append(tag)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--chain")
    ap.add_argument("--round", type=int)
    ap.add_argument("--request-id")
    ap.add_argument("--csv", default=str(HERE / "data" / "opal_chains.csv"),
                    help="for the benchmark's own explanation of the turn")
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=0,
                    help="show only the N hottest lines; default is all of them")
    ap.add_argument("--width", type=int, default=86, help="columns of line text")
    args = ap.parse_args()

    run = args.run_dir
    turns = report.load_turns(run / "turns.jsonl")
    prompts_path = run / "prompts.jsonl"
    if not prompts_path.is_file():
        raise SystemExit(
            f"no prompts.jsonl in {run}. Runs captured before the line table "
            f"existed cannot be drawn: the spans cannot be recomputed safely "
            f"without the exact tokenizer and chat template of that run.")
    prompts = report.load_prompts(prompts_path)

    if args.request_id:
        sel = [t for t in turns if t.get("request_id") == args.request_id]
    else:
        sel = [t for t in turns
               if (not args.chain or t.get("chain_id") == args.chain)
               and (args.round is None or t.get("round_no") == args.round)]
    if not sel:
        raise SystemExit("no turn matched; pass --chain/--round or --request-id")
    if len(sel) > 1:
        raise SystemExit(f"{len(sel)} turns matched; narrow it with --round")
    turn = sel[0]

    meta_path = run / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    buf_path, notes = report.find_run_buffer(run, meta.get("workflow_id"))
    for n in notes:
        print(f"!! {n}")
    if buf_path is None:
        raise SystemExit(f"no provenance in {run}")
    tasks = report.index_tasks(report.load_buffer(buf_path))
    rec = tasks.get(turn.get("request_id"))
    if rec is None:
        raise SystemExit(f"no attention record joined to request "
                         f"{turn.get('request_id')!r}")
    prompt = prompts.get(turn.get("request_id"))
    if prompt is None:
        raise SystemExit(f"no prompt line table for request "
                         f"{turn.get('request_id')!r}")

    from opal_attn.provenance import load_provenance
    field = report.heat_lines(turn, rec, load_provenance, prompt,
                              run_dir=run, decay=args.decay)
    items = field["items"]

    expl = ""
    try:
        rows = {(r.chain_id, r.round_no): r
                for c in load_chains(args.csv) for r in c.rounds}
        row = rows.get((turn["chain_id"], turn["round_no"]))
        expl = row.explanation if row else ""
    except Exception:
        pass

    verdict = turn.get("verdict") or ("correct" if turn.get("ok") else "wrong")
    print(f"{turn['chain_id']} round {turn['round_no']} | {turn['task_type']}")
    print(f"answered {turn.get('got')!r} | truth {turn.get('want')!r} "
          f"| {verdict.upper()}")
    if turn.get("finish_reason") and turn["finish_reason"] != "stop":
        print(f"!! finish_reason={turn['finish_reason']} -- the reply was cut "
              f"off, so the later decode steps below are missing")
    if expl:
        print(f"\nBENCHMARK EXPLANATION (CSV `explanation` column):\n   {expl}")

    print("\nHEAT = val_all_max; per line the max over its retained positions;")
    print(f"aggregated across each item's content steps with exponential "
          f"decay {args.decay}**k.")
    for n, it in enumerate(items, 1):
        steps = it["steps"]
        rng = f"steps {steps[0]}-{steps[-1]}" if steps else "no steps"
        print(f"  col{n} = item{n} {it['label']!r} ({rng})")
    print(f"  ramp {RAMP!r} = level 0..9, each column scaled to its own max.")
    print(f"  blank heat = level 0, or no position on that line survived "
          f"top-{meta.get('top_pct', '?')}% selection.")

    gutters = gutter_labels(prompt, turn.get("segments") or [])
    maxima = [max(it["heat"].values(), default=0.0) for it in items]
    lines = prompt["lines"]
    # `text.split("\n")` yields a final empty element for the trailing newline
    # after the generation prompt. It holds no tokens and can never be hot;
    # dropping it is what makes the line count here agree with the reference
    # rendering rather than being one more.
    if lines and not lines[-1][3] and lines[-1][1] <= lines[-1][0]:
        lines = lines[:-1]

    order = range(len(lines))
    if args.top:
        best = sorted(order, key=lambda i: -max(
            (it["heat"].get(i, 0.0) / m if m else 0.0)
            for it, m in zip(items, maxima, strict=True)))[:args.top]
        order = sorted(best)
        print(f"  showing the {len(order)} hottest lines of {len(lines)}.")

    print(f"  {len(lines)} lines, {field['n_steps']} decode steps captured.")
    print("=" * (13 + len(items) + 1 + args.width))
    prev = None
    for i in order:
        if prev is not None and i != prev + 1:
            print(" " * 13 + "~" * (len(items) + 1 + 20) + "  (lines skipped)")
        prev = i
        cells = "".join(level(it["heat"].get(i, 0.0), m)
                        for it, m in zip(items, maxima, strict=True))
        text = lines[i][3][:args.width]
        print(f"{gutters[i]:<11} {cells} {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
