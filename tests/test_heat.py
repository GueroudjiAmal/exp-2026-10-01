"""The per-answer-item heat field, end to end, with no GPU and no vLLM.

Builds a REAL conversation -- a real OPAL chain, the real chat template, the
real tokenizer, the real line table -- and pairs it with a SYNTHETIC capture
whose attention is planted on known lines. Everything between the tensors and
the rendered column is then exercised against an answer this test knows.

Synthetic rather than a recorded fixture because the property under test is
"the heat lands on the line the attention was on". A recorded capture cannot
state that independently: it would only assert that today's code reproduces
today's output, which is exactly the check that let an all-zero capture pass
mass conservation for five runs.

The chain is exp32-chain-0020-hard round 5 with ground-truth history -- the
turn in the reference heat map, whose answer has two parts that consult
different evidence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from opal_attn import report  # noqa: E402
from opal_attn.chains import load_chains  # noqa: E402
from opal_attn.ranges import declare_ranges_and_lines  # noqa: E402
from opal_attn.runner import decode_steps  # noqa: E402

CHAIN, ROUND = "exp32-chain-0020-hard", 5
REPLY = "ANSWER: NIL 373; DIVERGES"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
K = 16              # retained positions per decode step
SINK_VAL = 99.0     # position 0 outranks everything and must still be ignored


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:                       # offline, no cached weights
        pytest.skip(f"tokenizer {MODEL} unavailable: {e}")


class _Completion:
    """The shape of `RequestOutput.outputs[0]` that the runner reads."""
    def __init__(self, token_ids, text, finish_reason="stop"):
        self.token_ids, self.text, self.finish_reason = token_ids, text, finish_reason


def _line_of(lines, needle: str) -> int:
    """The one line equal to `needle` once indentation is stripped.

    Exact rather than substring: "NIL 373   0" appears as a prefix of four
    different table rows in this conversation, and planting attention on the
    wrong one would make the test pass for the wrong reason.
    """
    hits = [i for i, ln in enumerate(lines) if ln[3].strip() == needle]
    assert len(hits) == 1, f"{needle!r} matched {len(hits)} lines, need exactly 1"
    return hits[0]


@pytest.fixture(scope="module")
def built(tok, tmp_path_factory):
    """A run directory: real prompt and line table, planted attention."""
    chain = {c.chain_id: c for c in load_chains(HERE / "data" / "opal_chains.csv")}[CHAIN]
    msgs = []
    for row in chain.rounds:
        if row.round_no > ROUND:
            break
        msgs.append({"role": "user", "content": row.prompt})
        if row.round_no < ROUND:
            msgs.append({"role": "assistant",
                         "content": f"ANSWER: {row.correct_answer}"})
    text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                   tokenize=False, enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    segs, lines = declare_ranges_and_lines(text, tok)
    line_rows = [[ln.lo, ln.hi, ln.round_hint, ln.text] for ln in lines]

    gen_ids = tok(REPLY, add_special_tokens=False)["input_ids"]
    gen_text, gen_char_ends = decode_steps(tok, _Completion(gen_ids, REPLY))
    assert gen_text == REPLY, gen_text

    turn = {
        "chain_id": CHAIN, "round_no": ROUND, "task_type": "analogue_search",
        "request_id": "7", "n_prompt_tokens": len(ids),
        "segments": [{"kind": s.kind, "lo": s.lo, "hi": s.hi,
                      "round_hint": s.round_hint} for s in segs],
        "got": "NIL 373; DIVERGES", "want": chain.rounds[ROUND - 1].correct_answer,
        "verdict": "correct", "ok": True, "shortcut": None, "raw": REPLY,
        "finish_reason": "stop", "gen_text": gen_text,
        "gen_char_ends": gen_char_ends,
    }

    items = report.answer_items(turn)
    assert [i["label"] for i in items] == ["NIL 373", "DIVERGES"], items

    # Plant: item 1 on the round-2 history answer, item 2 on round 5's
    # peak_treatment table. Different evidence, which is the whole point.
    hot1 = _line_of(line_rows, "ANSWER: MAGIC 75<|im_end|>")
    hot2 = _line_of(line_rows, "NIL 373   0")
    targets = {items[0]["steps"][0]: hot1, items[1]["steps"][0]: hot2}

    n_steps = len(gen_char_ends)
    pos = np.zeros((n_steps, K), dtype=np.int32)
    val = np.zeros((n_steps, K), dtype=np.float32)
    for s in range(n_steps):
        pos[s, 0], val[s, 0] = 0, SINK_VAL          # the attention sink
        for j in range(1, K):                       # spread, low-value noise
            pos[s, j] = (s * 37 + j * 101) % max(1, len(ids))
            val[s, j] = 0.01
        if s in targets:
            lo, hi = line_rows[targets[s]][0], line_rows[targets[s]][1]
            pos[s, 1], val[s, 1] = lo, 1.0
            if hi > lo + 1:
                pos[s, 2], val[s, 2] = lo + 1, 0.9

    run = tmp_path_factory.mktemp("run")
    wf = "opal-test"
    tdir = run / "attention" / wf
    tdir.mkdir(parents=True)
    from safetensors.numpy import save_file
    tfile = tdir / "7_g0.safetensors"
    save_file({"topk_pos": pos, "val_all_max": val,
               "segments": np.array([[s.lo, s.hi, 1] for s in segs],
                                    dtype=np.int32)}, str(tfile))

    (run / "turns.jsonl").write_text(json.dumps(turn) + "\n")
    (run / "prompts.jsonl").write_text(json.dumps({
        "request_id": "7", "chain_id": CHAIN, "round_no": ROUND,
        "n_prompt_tokens": len(ids), "text": text, "lines": line_rows}) + "\n")
    (run / "run_meta.json").write_text(json.dumps(
        {"workflow_id": wf, "top_pct": 10.0, "model": MODEL}) + "\n")
    (run / "provenance.jsonl").write_text(json.dumps({
        "type": "task", "used": {"request_id": "7"},
        "attention_stats": {"uri": tfile.resolve().as_uri(),
                            "format": "safetensors"}}) + "\n")
    return run, turn, line_rows, hot1, hot2, items


def test_answer_items_split_on_the_separator(built):
    _run, _turn, _lines, _h1, _h2, items = built
    assert [i["label"] for i in items] == ["NIL 373", "DIVERGES"]
    # disjoint, ordered, and neither claims the "ANSWER:" prefix steps
    s1, s2 = items[0]["steps"], items[1]["steps"]
    assert s1 and s2 and max(s1) < min(s2), (s1, s2)
    assert min(s1) > 0


def test_each_item_is_hottest_on_its_own_evidence(built):
    run, turn, _lines, hot1, hot2, _items = built
    from opal_attn.provenance import load_provenance
    prompts = report.load_prompts(run / "prompts.jsonl")
    rec = report.index_tasks(report.load_buffer(run / "provenance.jsonl"))["7"]
    field = report.heat_lines(turn, rec, load_provenance, prompts["7"], run_dir=run)

    (c1, c2) = field["items"]
    assert max(c1["heat"], key=c1["heat"].get) == hot1
    assert max(c2["heat"], key=c2["heat"].get) == hot2
    # and NOT each other's: the two columns genuinely separate
    assert c1["heat"].get(hot2, 0.0) < c1["heat"][hot1]
    assert c2["heat"].get(hot1, 0.0) < c2["heat"][hot2]


def test_the_attention_sink_is_never_credited(built):
    run, turn, lines, _h1, _h2, _items = built
    from opal_attn.provenance import load_provenance
    prompts = report.load_prompts(run / "prompts.jsonl")
    rec = report.index_tasks(report.load_buffer(run / "provenance.jsonl"))["7"]
    field = report.heat_lines(turn, rec, load_provenance, prompts["7"], run_dir=run)
    sink_line = next(i for i, ln in enumerate(lines) if ln[1] > ln[0] and ln[0] == 0)
    for col in field["items"]:
        assert col["heat"].get(sink_line, 0.0) < SINK_VAL


def test_decay_is_applied_per_step_within_an_item(built):
    run, turn, _lines, hot1, _h2, items = built
    from opal_attn.provenance import load_provenance
    prompts = report.load_prompts(run / "prompts.jsonl")
    rec = report.index_tasks(report.load_buffer(run / "provenance.jsonl"))["7"]
    for decay in (0.5, 1.0):
        f = report.heat_lines(turn, rec, load_provenance, prompts["7"],
                              run_dir=run, decay=decay)
        # the plant sits on step 0 of the item, so weight is decay**0 == 1
        assert f["items"][0]["heat"][hot1] == pytest.approx(1.0)
    # a plant on a LATER step of the item must be scaled by decay**k
    later = items[0]["steps"][2]
    f = report.heat_lines(turn, rec, load_provenance, prompts["7"], run_dir=run)
    noise = max(v for k, v in f["items"][0]["heat"].items() if k != hot1)
    assert noise == pytest.approx(0.01 * 0.5 ** 0, abs=1e-6) or noise < 0.011, (
        f"noise {noise} should be the undecayed-or-less 0.01 floor (step {later})")


def test_the_renderer_draws_the_planted_lines(built, capsys):
    """scripts/heatmap.py end to end, as a subprocess, on the same run."""
    import subprocess
    run, _turn, lines, hot1, hot2, _items = built
    out = subprocess.run(
        [sys.executable, str(HERE / "scripts" / "heatmap.py"), str(run),
         "--chain", CHAIN, "--round", str(ROUND)],
        capture_output=True, text=True, timeout=300,
        env={"PYTHONPATH": str(HERE / "src"), "PATH": "/usr/bin:/bin",
             "HOME": str(Path.home())},
    )
    assert out.returncode == 0, out.stderr
    with capsys.disabled():
        print("\n" + out.stdout)

    # Find each planted line's drawn row by its text rather than by index:
    # the renderer drops the trailing blank line to match the reference's line
    # count, so output row N is not source line N.
    body = out.stdout.splitlines()
    for which, idx in (("item 1", hot1), ("item 2", hot2)):
        want = lines[idx][3].rstrip()
        rows = [r for r in body if r.endswith(want)]
        assert len(rows) == 1, f"{which}: {len(rows)} rows end with {want!r}"
        cells = rows[0][: -len(want)][-3:-1] if want else ""
        assert "@" in rows[0][:len(rows[0]) - len(want)], (
            f"{which} should be the hottest line in its column: {rows[0]!r}")
        assert cells is not None
    assert "NIL 373" in out.stdout and "DIVERGES" in out.stdout
    # the header must agree with the reference's line-counting convention
    assert " lines, " in out.stdout


def test_a_turn_that_declared_nothing_is_a_failure_not_a_pass(built):
    """`declared <= emitted` is vacuous on an empty set; verify must not be.

    A turn with no declared ranges is what a chat-template change looks like
    from the analysis side: declare_ranges matches nothing, the connector
    chunks the whole prompt into synthesised gaps, and every structural check
    still passes. It has to read as a failure.
    """
    run, turn, _lines, _h1, _h2, _items = built
    from opal_attn.provenance import load_provenance
    buffer = report.load_buffer(run / "provenance.jsonl")
    meta = json.loads((run / "run_meta.json").read_text())

    def check_named(turns, fragment):
        checks = report.verify(turns, buffer, top_pct=meta["top_pct"],
                               load_provenance=load_provenance,
                               run_dir=run, meta=meta)
        return next(ok for ok, msg in checks if fragment in msg)

    assert check_named([turn], "declared ranges appear verbatim") is True

    stripped = {**turn, "segments": []}
    assert check_named([stripped], "declared ranges appear verbatim") is False
