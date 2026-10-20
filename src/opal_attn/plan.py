"""The dry run: everything the real run does, up to the point a GPU is needed.

This is the local arm. It loads the chains, renders every turn through the real
chat template, tokenizes with the real tokenizer, declares the real segments,
and applies the real context budget -- then stops instead of calling an engine.

It exists because almost every way this experiment can be wrong is visible
here: a chain that silently truncates at round 5, a chat template that moves
every declared index, a tokenizer mismatch between the planner and the run, a
segment that lands next to a tool block rather than on it, an output budget that
does not fit the filesystem. None of those need CUDA to find, and all of them
cost node hours to find on Polaris.

What it cannot check is the capture itself: the probe, the kernel, the cache
layout. Those are GPU-only and are what `polaris/smoke_test.sh` stage 3 is for.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .chains import Chain
from .ranges import declare_ranges, kept_entries
from .runner import chat_kwargs


@dataclass
class PlannedTurn:
    chain_id: str
    tier: str
    round_no: int
    task_type: str
    n_prompt_tokens: int
    n_segments: int
    segment_kinds: list[str]
    tool_blocks: int
    answer_blocks: int
    kept_per_step: int
    fits: bool


@dataclass
class Plan:
    model: str
    turns: list[PlannedTurn] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    config: dict = field(default_factory=dict)


def build_plan(chains: list[Chain], tok, *, model: str, max_model_len: int,
               max_tokens: int, top_pct: float, chunk_size: int,
               max_steps: int, enable_thinking: bool = False,
               system_prompt: str = "") -> Plan:
    """Render and measure every turn, with GROUND-TRUTH answers as history.

    The real run uses the model's own replies, which tokenize to a different
    length -- so these prompt counts are an estimate, not a promise. They are
    close (an OPAL answer is a short line) and they are the only estimate
    available without a GPU, but the context-budget margin below exists because
    of exactly this gap.
    """
    plan = Plan(model=model, config={
        "max_model_len": max_model_len, "max_tokens": max_tokens,
        "top_pct": top_pct, "chunk_size": chunk_size, "max_steps": max_steps,
        "history": "ground-truth answers (the run uses the model's own)",
        "system_prompt": system_prompt,
    })
    for chain in chains:
        # No system prompt unless one was asked for; see runner.py. It is
        # seeded HERE as well as in the runner, or the context budget this
        # computes describes a shorter prompt than the engine will see.
        messages: list[dict] = ([{"role": "system", "content": system_prompt}]
                                if system_prompt else [])
        for row in chain.rounds:
            messages.append({"role": "user", "content": row.prompt})
            # Same kwargs as the real run, or these token counts describe a
            # different prompt than the one the engine will see.
            text = tok.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=False,
                                           **chat_kwargs(enable_thinking))
            ids = tok(text, add_special_tokens=False)["input_ids"]
            n = len(ids)
            fits = n + max_tokens + 16 <= max_model_len
            segs = declare_ranges(text, tok)
            plan.turns.append(PlannedTurn(
                chain_id=chain.chain_id, tier=chain.tier, round_no=row.round_no,
                task_type=row.task_type, n_prompt_tokens=n,
                n_segments=len(segs),
                segment_kinds=sorted({s.kind for s in segs}),
                tool_blocks=sum(1 for s in segs if s.kind != "answer"),
                answer_blocks=sum(1 for s in segs if s.kind == "answer"),
                kept_per_step=kept_entries(segs, n, top_pct, chunk_size),
                fits=fits,
            ))
            if not fits:
                plan.dropped[chain.chain_id] = (
                    f"round {row.round_no} needs {n} tokens, over {max_model_len}")
                break
            # Stand-in history. The "ANSWER: <...>" shape is the row's OWN
            # declared OUTPUT FORMAT, not a format imposed here -- but the
            # model's real reply will tokenize to a different length, which is
            # why these counts are an estimate and the window keeps headroom.
            messages.append({"role": "assistant",
                             "content": f"ANSWER: {row.correct_answer}"})
    return plan


def project(plan: Plan) -> dict:
    """Output size and context headroom, from the planned turns.

    The file-size model is the connector's own, from its README:

        floats = k*G*4  (positions, two values, head id)
               + G      (residual)
               + 2T     (attn_sum, attn_peak)
               + T      (prompt_token_ids)
               + 3*n_seg

    at 4 bytes each. `G` is decode steps recorded -- the full generation unless
    `max_steps` caps it. Taking G = max_tokens is therefore an UPPER bound: a
    model that stops early at an EOS writes fewer rows.
    """
    cfg = plan.config
    live = [t for t in plan.turns if t.fits]
    g = cfg["max_steps"] or cfg["max_tokens"]
    total = 0
    for t in live:
        floats = (t.kept_per_step * g * 4) + g + 3 * t.n_prompt_tokens + 3 * t.n_segments
        total += floats * 4
    lens = [t.n_prompt_tokens for t in live] or [0]
    return {
        "turns_planned": len(live),
        "turns_dropped": len(plan.turns) - len(live),
        "chains_truncated": len(plan.dropped),
        "prompt_tokens_total": sum(lens),
        "prompt_tokens_max": max(lens),
        "prompt_tokens_median": int(statistics.median(lens)),
        "context_headroom_tokens": cfg["max_model_len"] - max(lens) - cfg["max_tokens"],
        "decode_steps_assumed": g,
        "segments_median": int(statistics.median(
            [t.n_segments for t in live] or [0])),
        "kept_per_step_median": int(statistics.median(
            [t.kept_per_step for t in live] or [0])),
        "safetensors_bytes_upper_bound": total,
        "safetensors_mib_upper_bound": round(total / 2**20, 1),
    }


def check(plan: Plan) -> list[tuple[bool, str]]:
    """Assertions that do not need a GPU. `(passed, message)` per check."""
    live = [t for t in plan.turns if t.fits]
    out: list[tuple[bool, str]] = []

    out.append((bool(live), f"turns planned ({len(live)})"))
    if not live:
        return out

    out.append((not plan.dropped,
                f"every chain fits the context window "
                f"({len(plan.dropped)} truncated)"))
    out.append((all(t.n_segments > 0 for t in live),
                "every turn declares at least one segment"))
    out.append((all(t.tool_blocks > 0 for t in live),
                "every turn finds at least one tool block"))

    # Round 1 has no history, so it must have no answer segment; every later
    # round must have exactly round_no-1 of them. This is the check that catches
    # a chat template whose assistant marker the range regex does not match --
    # the failure that would otherwise silently drop all history segments.
    bad = [t for t in live
           if t.answer_blocks != t.round_no - 1]
    out.append((not bad,
                f"answer segments track conversation depth "
                f"({len(bad)} turns disagree"
                + (f", e.g. {bad[0].chain_id} r{bad[0].round_no}: "
                   f"{bad[0].answer_blocks} vs {bad[0].round_no - 1}" if bad else "")
                + ")"))

    # Prompts must grow monotonically inside a chain; if they do not, the
    # history is not accumulating and every later round is being answered blind.
    shrunk = []
    per_chain: dict[str, list[PlannedTurn]] = {}
    for t in live:
        per_chain.setdefault(t.chain_id, []).append(t)
    for cid, ts in per_chain.items():
        lens = [t.n_prompt_tokens for t in sorted(ts, key=lambda x: x.round_no)]
        if lens != sorted(lens) or len(set(lens)) != len(lens):
            shrunk.append(cid)
    out.append((not shrunk,
                f"prompts grow every round within a chain ({len(shrunk)} do not)"))

    return out


def write_plan(out_dir: str | Path, plan: Plan, projection: dict) -> Path:
    """Persist the plan. The Polaris job reads it to size its walltime request."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plan.json"
    path.write_text(json.dumps({
        "model": plan.model,
        "config": plan.config,
        "projection": projection,
        "dropped": plan.dropped,
        "turns": [t.__dict__ for t in plan.turns],
    }, indent=2) + "\n")
    return path
