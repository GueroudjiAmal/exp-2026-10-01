#!/usr/bin/env python
"""Which turns disagree on prompt length, and what it costs them.

    python scripts/mismatch.py results/<run>

The runner counts the token ids it submitted; the connector counts the keys it
found in the cache. They should be identical -- that agreement is what makes
the request-id join trustworthy rather than coincidental. When they differ,
the segment plan was built on a different length than the prompt actually had,
so declared ranges stop landing verbatim and every position in that record is
attributed to the wrong span.

Prints the direction and size of each disagreement, which is what says whether
the cause is prefix caching (connector sees fewer keys), chunked prefill
(settles late), or something else.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from opal_attn import report  # noqa: E402
from opal_attn.provenance import load_provenance  # noqa: E402


def main() -> int:
    run = Path(sys.argv[1])
    turns = report.load_turns(run / "turns.jsonl")
    buf, _ = report.find_run_buffer(run)
    if buf is None:
        raise SystemExit(f"no provenance in {run}")
    tasks = report.index_tasks(report.load_buffer(buf))

    bad = []
    for t in turns:
        rec = tasks.get(t.get("request_id"))
        if rec is None:
            continue
        conn = (rec.get("used") or {}).get("num_prompt_tokens")
        if conn == t["n_prompt_tokens"]:
            continue
        a = load_provenance(rec, run_dir=run)
        segs = {(lo, hi) for lo, hi, _k in a["segments"].tolist()}
        decl = {(s["lo"], s["hi"]) for s in t["segments"]}
        bad.append((t, conn, sorted(decl - segs), len(decl), len(segs),
                    a["segments"].tolist()[-1] if len(segs) else None))

    print(f"{len(bad)} of {len(turns)} turns disagree\n")
    for t, conn, missing, n_decl, n_seg, last_seg in bad:
        d = conn - t["n_prompt_tokens"]
        print(f"  {t['chain_id']} r{t['round_no']} ({t['task_type']})")
        print(f"    runner submitted {t['n_prompt_tokens']} tokens, "
              f"connector scored {conn}  ({d:+d})")
        print(f"    declared {n_decl} ranges, emitted {n_seg} segments, "
              f"last segment {last_seg}")
        if missing:
            print(f"    declared ranges NOT emitted: {missing[:3]}")
    if bad:
        diffs = {conn - t["n_prompt_tokens"] for t, conn, *_ in bad}
        print(f"\n  distinct differences: {sorted(diffs)}")
        print("  a constant negative offset points at prefix caching; a varying")
        print("  one at chunked prefill settling at different points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
