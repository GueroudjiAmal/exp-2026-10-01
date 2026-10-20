#!/usr/bin/env python
"""Local Flowcept smoke: provenance in, provenance out, no GPU.

Exercises the exact path the real run uses -- the vLLM interceptor that the
connector calls, Flowcept's offline JSONL dump, and `report.py`'s join -- with
synthetic records in place of the connector's. That is worth doing locally
because the join is where a real run's results are most easily lost: a record
whose `request_id` does not match what the runner stored is indistinguishable
from a record that was never emitted.

What it does NOT prove: that the attention numbers are right. Nothing here
touches a GPU, a kernel, or a KV cache.

    FLOWCEPT_SETTINGS_PATH=configs/flowcept_local.yaml \
        python scripts/flowcept_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
os.environ.setdefault("FLOWCEPT_SETTINGS_PATH",
                      str(HERE / "configs" / "flowcept_local.yaml"))

from opal_attn import report  # noqa: E402

WORKFLOW_ID = "opal-attn-smoke"
N = 5


def fake_stats(i: int, n_prompt: int, n_steps: int, k: int) -> dict:
    """The descriptor shape the connector emits -- a reference, not the data."""
    return {
        "uri": f"file:///dev/null/{i}_g0.safetensors",
        "format": "safetensors", "bytes": 1234, "sha256": "0" * 64,
        "kv_cache_group_id": 0, "written_by_tp_rank": 0,
        "segment_mode": "variable",
        "decode_steps_dropped": 0, "decode_steps_nonfinite": 0, "restarts": 0,
        "tensors": {
            "topk_pos": {"shape": [n_steps, k], "dtype": "int32"},
            "val_all_max": {"shape": [n_steps, k], "dtype": "float32"},
            "attn_sum": {"shape": [n_prompt], "dtype": "float32"},
        },
    }


def main() -> int:
    from flowcept import Flowcept
    from flowcept.flowceptor.adapters.vllm.vllm_interceptor import VLLMInterceptor

    work = Path(tempfile.mkdtemp(prefix="opal-smoke-"))
    os.chdir(work)      # dump_buffer path is relative to cwd

    turns = []
    with Flowcept("vllm", workflow_id=WORKFLOW_ID,
                  workflow_name="opal_attn_smoke") as fc:
        itc = VLLMInterceptor.get_instance()
        # The connector registers its own workflow, parented to ours, and hangs
        # the run-constant config there. report.attention_config() must find it
        # by the FIELD rather than by a known id -- this is what checks that.
        child = f"vllm-attn-{WORKFLOW_ID}"
        itc.send_model_workflow(
            child,
            {"model": "synthetic/none", "tokenizer": "synthetic/none"},
            parent_workflow_id=WORKFLOW_ID,
            attention_config={"connector": "vllm-attn-connector",
                              "metric": "decode_attention",
                              "top_pct": 10.0, "chunk_size": 32},
        )
        for i in range(N):
            n_prompt, n_steps, k = 900 + 300 * i, 32, 90 + 30 * i
            rid = f"synthetic-{i}"
            itc.capture_request(
                workflow_id=child, request_id=f"{rid}:g0",
                attention_stats=fake_stats(i, n_prompt, n_steps, k),
                num_prompt_tokens=n_prompt, num_decode_tokens=n_steps,
                activity="decode_attention",
            )
            turns.append({
                "chain_id": "synthetic-chain", "tier": "low", "round_no": i + 1,
                "task_type": "synthetic", "request_id": rid,
                "n_prompt_tokens": n_prompt, "n_segments": 3,
                "segments": [{"kind": "db_lookup", "lo": 10, "hi": 40,
                              "round_hint": i + 1}],
                "got": "40", "want": "40" if i % 2 == 0 else "41",
                "verdict": "correct" if i % 2 == 0 else "wrong",
                "ok": i % 2 == 0, "shortcut": None, "raw": "ANSWER: 40",
                "gen_seconds": 0.0,
            })
        buffer = list(fc.get_buffer())

    checks = report.verify(turns, buffer, top_pct=10.0, load_provenance=None)
    summary = report.summarise(turns)

    print(f"buffer records : {len(buffer)}")
    print(f"dump written   : {sorted(p.name for p in work.glob('*.jsonl'))}")
    print("\n=== report.verify on synthetic records ===")
    for ok, msg in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
    print("\n=== report.summarise ===")
    print(f"  accuracy {summary['accuracy']}  "
          f"({summary['correct']}/{summary['turns']})")

    # The dump on disk must contain what get_buffer() returned, or the Polaris
    # arm -- which has ONLY the dump -- is reading something different from what
    # the local arm validated.
    dumps = sorted(work.glob("*.jsonl"))
    dump_ok = False
    if dumps:
        recs = report.load_buffer(dumps[0])
        dump_ok = len(report.attention_tasks(recs)) == N
        print(f"\n  {'PASS' if dump_ok else 'FAIL'}  the JSONL dump holds all "
              f"{N} tasks ({len(report.attention_tasks(recs))} found)")
    else:
        print("\n  FAIL  no JSONL dump was written")

    failed = sum(1 for ok, _ in checks if not ok) + (0 if dump_ok else 1)
    print(f"\nscratch: {work}")
    print(f"FAILURES: {failed}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
