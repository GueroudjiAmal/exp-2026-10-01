#!/usr/bin/env python
"""Why did `hottest_by_kind` come back empty? Inspect one run's tensors.

    python scripts/debug_attention.py results/<run> [n]

`hottest_segments` skips a turn when the retained-position row it looks at
holds nothing usable. That is silent by design -- one odd record should not
abort a report -- but when EVERY turn skips, the report is empty and the
reason is invisible. This prints the state it decides on.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from opal_attn import report  # noqa: E402
from opal_attn.provenance import load_provenance  # noqa: E402


def main() -> int:
    run = Path(sys.argv[1])
    n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    turns = report.load_turns(run / "turns.jsonl")
    buf_path, _ = report.find_run_buffer(run)
    buffer = report.load_buffer(buf_path)
    tasks = report.index_tasks(buffer)
    matched = [t for t in turns if t.get("request_id") in tasks]
    print(f"turns {len(turns)}  joined {len(matched)}  buffer {buf_path.name}\n")

    why: Counter = Counter()
    for i, t in enumerate(matched):
        a = load_provenance(tasks[t["request_id"]], run_dir=run)
        pos = a["topk_pos"].tolist()
        val = a["val_all_max"].tolist()
        segs = [(lo, hi) for lo, hi, _k in a["segments"].tolist()]

        if not pos:
            why["topk_pos has no rows"] += 1
            continue
        step = max(0, len(pos) - 2)          # what hottest_segments uses
        row = pos[step]
        positive = [p for p in row if p > 0]
        in_seg = [p for p in positive if any(lo <= p < hi for lo, hi in segs)]

        if i < n_show:
            print(f"--- {t['chain_id']} r{t['round_no']} ---")
            print(f"  shapes        topk_pos={len(pos)}x{len(pos[0]) if pos else 0}  "
                  f"segments={len(segs)}  declared_in_turn={len(t['segments'])}")
            print(f"  prompt tokens {t['n_prompt_tokens']}  "
                  f"segments span  {segs[0] if segs else '-'} .. {segs[-1] if segs else '-'}")
            print(f"  step used     {step} of {len(pos)}")
            print(f"  row[{step}]      first 8 = {row[:8]}")
            print(f"                 >0: {len(positive)}/{len(row)}   "
                  f"inside a segment: {len(in_seg)}")
            print(f"  val row       first 4 = {[round(v, 4) for v in val[step][:4]]}")
            nonzero_rows = sum(1 for r in pos if any(p > 0 for p in r))
            print(f"  rows with any position > 0: {nonzero_rows}/{len(pos)}")
            print()

        if not positive:
            why["every retained position <= 0 at that step"] += 1
        elif not in_seg:
            why["positions fall outside every segment"] += 1
        else:
            why["usable"] += 1

    print("verdict across all joined turns:")
    for k, v in why.most_common():
        print(f"  {v:5}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
