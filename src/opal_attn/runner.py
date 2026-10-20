"""Replaying OPAL chains through one vLLM engine, with attention capture.

Scheduling: ROUND-SYNCHRONOUS BATCHING. Rounds inside a chain are strictly
sequential -- round N's prompt contains the model's own round N-1 answer, so it
cannot be built until that answer exists. Chains, however, are independent. So
the driver runs one batched `generate` per round index: at round r it submits
the r-th turn of every chain that has one and is still alive.

That is 7 engine calls for 430 generations rather than 430, and it is what makes
capture affordable -- the connector's cost is per request per decode step, which
overlaps across a batch on the side stream, whereas 430 serial calls pay the
engine's own ramp-up 430 times.

Everything one run produces lands in ONE directory, and nothing of one run
lands in another's:

    <run>/run_meta.json      configuration, timings, probe evidence
    <run>/turns.jsonl        one line per (chain, round)
    <run>/summary.json       verdicts, accuracy, propagation, attention
    <run>/provenance.jsonl   the Flowcept records, flush nudge removed
    <run>/attention/         the SafeTensors the records point at
    <run>/raw/               Flowcept's own dump, before the nudge was removed

That requires a chdir: Flowcept's `dump_buffer.path` is relative to the
working directory, so without one its dump lands wherever the command happened
to be launched from -- which, for every run so far, was the repository root
rather than the run directory.

Joining records back to chains: on vLLM's request id, taken from the
`RequestOutput` and matched against `task["used"]["request_id"]`, which the
Flowcept adapter stores with the connector's `:g<group>` suffix stripped.
Matching on prompt length (as the connector's own 5-round example does) is fine
for one chain and wrong for a hundred, where collisions are near-certain.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .answers import CORRECT, classify, parse_answer
from .chains import Chain
from .ranges import Line, Segment, as_pairs, declare_ranges_and_lines

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# NO system prompt is added. An OPAL row is already self-contained -- it carries
# its own SESSION briefing, TOOLS list, TERMS, RULES and OUTPUT FORMAT, and the
# OUTPUT FORMAT already specifies the "ANSWER: <...>" line. Prepending an
# instruction of our own would restate those rules in different words, shift
# every token index in the conversation, and make the measured attention a
# property of our wrapper rather than of the benchmark. The conversation is
# exactly: the chain's prompts, and the model's replies.
#
# Verified for Qwen chat templates: with no system message the template inserts
# no default one, so the rendered conversation starts at the first user turn.


@dataclass
class TurnResult:
    chain_id: str
    tier: str
    round_no: int
    task_type: str
    request_id: str | None
    n_prompt_tokens: int
    n_segments: int
    segments: list[dict]
    got: str | None          # None when the reply had no ANSWER: line
    want: str
    verdict: str             # correct | wrong | unscorable | format_error
    ok: bool                 # verdict == "correct"; kept for older consumers
    shortcut: str | None
    raw: str
    gen_seconds: float
    # Decode-step bookkeeping. `raw` is the reply as a human reads it; these
    # say WHICH DECODE STEP produced which part of it, which is what lets an
    # analysis aggregate attention per answer item rather than over the reply
    # as a whole. OPAL's multi-part formats ("<accession>; <MATCHES|DIVERGES>")
    # are two separate questions answered by two disjoint runs of steps, and
    # they attend to different evidence -- the first to the history, the second
    # to the current round. Collapsing them hides exactly the effect these
    # chains were built to measure.
    #
    # `gen_char_ends[i]` is the length of the decoded reply after step i, so
    # step i covers `gen_text[gen_char_ends[i-1]:gen_char_ends[i]]`. A step
    # that does not advance it emitted no visible text (EOS, or a special
    # token). Offsets are computed against `gen_text`, NOT `raw`, because `raw`
    # is stripped and stripping moves every index.
    finish_reason: str | None = None
    gen_text: str = ""
    gen_char_ends: list[int] = field(default_factory=list)


@dataclass
class ChainState:
    """One chain's live conversation. The history is the MODEL's own answers.

    `messages` starts EMPTY unless `system` is set: by default the first entry
    is the chain's own round-1 prompt. See the note above on why nothing is
    prepended by default, and `--system-prompt` for the one way to override it.
    """
    chain: Chain
    system: str = ""
    messages: list[dict] = field(default_factory=list)
    alive: bool = True
    stopped_because: str | None = None

    def __post_init__(self) -> None:
        if self.system:
            self.messages.append({"role": "system", "content": self.system})


# Hybrid reasoning models (Qwen3-14B, Qwen3-32B, ...) emit a <think> block
# unless the chat template is told otherwise, and OPAL demands "exactly one
# line and nothing else" in a 32-token budget -- so thinking consumes the whole
# allowance and no ANSWER: line is ever produced. Measured: 214/214 format
# errors on Qwen3-14B.
#
# Passed unconditionally. A template that does not define `enable_thinking`
# ignores the kwarg (verified for Qwen3-*-Instruct-2507, whose template never
# mentions it), so one code path covers both families.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_thinking(text: str, truncated: bool = False) -> str:
    """The reply without its reasoning block.

    Only what the model would show a user belongs in the conversation history.
    An unterminated block -- the generation ran out of budget mid-thought --
    leaves nothing to keep, so the turn contributes an empty assistant message
    rather than several thousand tokens of abandoned reasoning.

    THE OPENING TAG IS OFTEN NOT IN THE OUTPUT. A Qwen thinking template ends
    the generation prompt with "<think>\n", so the completion STARTS inside
    the block and the model only ever emits the closing "</think>". Matching
    on a `<think>...</think>` pair therefore misses the common case entirely,
    and looking for a stray "<think>" misses it too.

    Measured on a 90-turn Qwen3.5 run before this was fixed: a turn following
    a truncated reply carried a prompt 4743 tokens larger than its
    predecessor, against 602 after a completed one -- the whole abandoned
    ramble, inherited by every later round, which is what pushed chains past
    max_model_len and killed them at round 5.

    So: anything up to and including the last "</think>" is reasoning. With no
    closing tag at all, a truncated reply is reasoning ALL THE WAY DOWN and
    contributes nothing; an untruncated one never entered a block and is kept
    whole.
    """
    out = text or ""
    if "</think>" in out:           # closed: the answer is what follows
        out = out.rsplit("</think>", 1)[1]
    elif "<think>" in out:          # opened in the OUTPUT, never closed
        out = out.split("<think>", 1)[0]
    elif truncated:                 # opened in the PROMPT, never closed
        out = ""
    return out.strip()


def chat_kwargs(enable_thinking: bool = False) -> dict:
    """Template kwargs. Off by default, because OPAL asks for one line.

    Passed unconditionally: a template that does not define `enable_thinking`
    ignores it (verified for Qwen3-*-Instruct-2507, whose template never
    mentions it), so one code path covers thinking and non-thinking families.
    """
    return {"enable_thinking": bool(enable_thinking)}


def decode_steps(tok, completion) -> tuple[str, list[int]]:
    """`(decoded_text, char_end_per_decode_step)` for one completion.

    The connector records attention per DECODE STEP; the reply is text. This
    is the bridge: `ends[i]` is how long the decoded reply is after step i, so
    step i wrote `text[ends[i-1]:ends[i]]`. An analysis can then ask "which
    steps produced the second half of the answer" and aggregate only those.

    Decoded prefix by prefix rather than token by token because a byte-level
    BPE token is not independently decodable -- a multi-byte character is split
    across tokens, and decoding them singly yields replacement characters and
    the wrong lengths. `skip_special_tokens=True` means an EOS step leaves the
    length unchanged, which is exactly the signal "this step emitted no text".

    One decode call per step, each over at most `max_tokens` tokens: a few
    seconds across a 430-turn run, against a generate phase measured in
    minutes.
    """
    ids = list(getattr(completion, "token_ids", None) or [])
    text, ends = "", []
    for i in range(len(ids)):
        text = tok.decode(ids[:i + 1], skip_special_tokens=True)
        ends.append(len(text))
    return text, ends


def write_prompt_line(fh, request_id, chain_id: str, round_no: int,
                      n_prompt_tokens: int, text: str, lines) -> None:
    """Append one turn's rendered prompt and line table to `prompts.jsonl`.

    Kept OUT of turns.jsonl deliberately. turns.jsonl is read by every script
    and by hand; the rendered conversation is thousands of tokens per turn and
    would bury it. This file is opened only by something drawing a heat map.

    The line table is stored rather than recomputed because recomputing it
    needs the tokenizer AND the exact chat template that produced the run --
    and a template that has changed since would silently re-derive different
    spans for the same captured tensors, which is the failure this experiment
    already refuses to risk for `tool_ranges`.
    """
    fh.write(json.dumps({
        "request_id": request_id,
        "chain_id": chain_id,
        "round_no": round_no,
        "n_prompt_tokens": n_prompt_tokens,
        "text": text,
        "lines": [[ln.lo, ln.hi, ln.round_hint, ln.text] for ln in lines],
    }) + "\n")


def build_turn(state: ChainState, round_idx: int, tok, max_model_len: int,
               max_tokens: int, enable_thinking: bool = False
               ) -> tuple[list[int], str, list[Segment], list[Line]] | None:
    """Render chain `state`'s round `round_idx` turn, or None if it cannot run.

    Returns `(token_ids, rendered_text, segments, lines)`. The token ids are
    what the engine is handed -- not a string for it to re-tokenize -- so the
    segment indices declared here are provably the indices the connector
    scores. Letting the engine tokenize separately risks a leading BOS shifting
    every declared range by one, which no assertion downstream would catch.

    The line table comes from the same tokenization as the segments, for the
    same reason and at no extra cost.
    """
    row = state.chain.rounds[round_idx]
    state.messages.append({"role": "user", "content": row.prompt})
    text = tok.apply_chat_template(state.messages, add_generation_prompt=True,
                                   tokenize=False,
                                   **chat_kwargs(enable_thinking))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) + max_tokens + 16 > max_model_len:
        state.messages.pop()            # leave the conversation as it was
        state.alive = False
        state.stopped_because = (f"round {row.round_no} needs {len(ids)} prompt "
                                 f"tokens, over the {max_model_len} window")
        return None
    return (ids, text, *declare_ranges_and_lines(text, tok))


def run(args) -> tuple[list[TurnResult], list[dict], dict]:
    """Execute the experiment. Returns `(turns, flowcept_buffer, run_meta)`."""
    from .chains import load_chains, select

    chains = select(load_chains(args.csv), limit=args.limit, tier=args.tier,
                    chain_ids=args.chain_ids)
    if not chains:
        raise SystemExit("no chains selected")

    registry = None
    if not args.no_connector:
        import vllm_attn_connector
        from vllm_attn_connector import REGISTRY as registry

        # The probe overrides ONE attention backend. If the engine then selects
        # a different one -- which Blackwell and other newer parts make likely,
        # since vLLM may prefer FlashInfer there -- the probe never fires and
        # every captured number is zero, with no error anywhere.
        #
        # So pin both ends to the same name instead of hoping they agree.
        # VLLM_ATTENTION_BACKEND must be set before the engine is built.
        backend = args.attn_backend
        if backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = backend
        # MUST precede engine construction: `_cached_get_attn_backend` is
        # @cache'd, so the backend class is resolved once and never re-read.
        assert vllm_attn_connector.install_probe(backend or None), (
            f"probe failed to install over {backend or 'FLASH_ATTN'}; "
            f"try --attn-backend TRITON_ATTN")

    from flowcept import Flowcept
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.config import KVTransferConfig

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    # POISON, applied here because it needs the tokenizer: every substitution
    # is re-checked for token-length parity against THIS model, not the one it
    # was authored on. A manifest built for another tokenizer fails loudly
    # rather than shifting every index and re-basing the distance axis.
    #
    # History arms are not text edits -- they replace the assistant message the
    # loop appends -- so they are carried separately and consulted below.
    poison_meta, hist_override = None, {}
    if getattr(args, "poison", None):
        from .poison import apply_manifest
        chains, hist_override, poison_meta = apply_manifest(chains, args.poison, tok)
        print(f"poison {poison_meta['arm']!r} "
              f"({poison_meta['form']}/{poison_meta['kind']}) applied to "
              f"{poison_meta['chains_poisoned']}/{len(chains)} chains")

    # Measured: Qwen3-14B with thinking on and a 32-token budget produced 214
    # format errors out of 214 -- the <think> block consumed the entire
    # allowance and no ANSWER: line was ever reached. Say so before burning the
    # run rather than after.
    if args.enable_thinking and args.max_tokens < 256:
        print(f"!! --enable-thinking with --max-tokens {args.max_tokens}: the "
              f"reasoning block will very likely consume the whole budget and "
              f"every turn will be a format_error. Use >= 512.")
    if args.enable_thinking and not args.max_steps:
        print(f"   note: capture records EVERY decode step, so a {args.max_tokens}"
              f"-token budget makes each tensor file roughly "
              f"{args.max_tokens // 32}x the size of a 32-token run. "
              f"--max-steps caps it.")

    states = [ChainState(c, args.system_prompt) for c in chains]
    turns: list[TurnResult] = []

    # `--no-connector` is the baseline arm of an overhead measurement: identical
    # workload, identical engine settings, no capture at all.
    # Tensors in their own subdirectory, so the run directory's top level is
    # the handful of files a human reads. The connector adds its own
    # <workflow-id>/ level beneath this.
    out_root = Path(args.out_dir).resolve()
    attn_dir = out_root / "attention"
    attn_dir.mkdir(parents=True, exist_ok=True)

    kv_cfg = None if args.no_connector else KVTransferConfig(
        kv_connector="AttnConnector",
        kv_connector_module_path="vllm_attn_connector",
        kv_role="kv_producer",      # never loads KV; "kv_both" defers block frees
        kv_connector_extra_config={
            "workflow_id": args.workflow_id,
            "out_dir": str(attn_dir),
            "top_pct": args.top_pct,
            "chunk_size": args.chunk_size,
            "max_steps": args.max_steps,
            "layer_stats": args.layer_stats,
        },
    )

    # Opened before the chdir and addressed absolutely, like every other
    # artifact path here -- relative paths are what used to scatter a run.
    prompts_fh = (out_root / "prompts.jsonl").open("w")

    gen_seconds = 0.0
    t_wall = time.perf_counter()
    # Flowcept's offline dump path is RELATIVE to the working directory, so the
    # only way to keep it with the run it belongs to is to be standing in the
    # run directory when Flowcept starts. Every path from here on is absolute.
    os.chdir(out_root)
    with Flowcept("vllm", workflow_id=args.workflow_id,
                  workflow_name="opal_attn_chains") as fc:
        llm = LLM(
            model=args.model,
            kv_transfer_config=kv_cfg,
            # PREFIX CACHING OFF during capture, unless asked for.
            #
            # The connector builds each request's block table by accumulating
            # `new_block_ids`, which reports only NEWLY ALLOCATED blocks.
            # Blocks served from the prefix cache were never new, so they never
            # enter the table, and the connector scores only the prefix it
            # happens to know about. Measured on Qwen3.5 (1056-token pages): a
            # 4421-token prompt scored as 4224 -- four blocks instead of five --
            # losing 197 tokens and two declared ranges, with nothing logged.
            #
            # It only bites on prompts long enough to span blocks, which is why
            # 18 of 20 turns looked fine. That is the worst kind of defect to
            # leave on: it corrupts the longest, most interesting turns and
            # passes every check on the short ones.
            enable_prefix_caching=args.prefix_caching,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            enforce_eager=not args.cuda_graphs,
        )

        max_rounds = max(len(c) for c in chains)
        for round_idx in range(max_rounds):
            batch, owners = [], []
            for st in states:
                if not st.alive or round_idx >= len(st.chain):
                    continue
                built = build_turn(st, round_idx, tok, args.max_model_len,
                                   args.max_tokens, args.enable_thinking)
                if built is None:
                    continue
                ids, text, segs, lines = built
                batch.append(TokensPrompt(prompt_token_ids=ids))
                owners.append((st, ids, text, segs, lines))
            if not batch:
                continue

            # Declared ranges only travel when there is a connector to
            # receive them. With --no-connector the engine logs
            #   "Got kv_transfer_params, but no KVConnector found"
            # once PER REQUEST and ignores them. Harmless -- the baseline arm
            # captures nothing by definition -- but one line per turn buries
            # any warning that does matter, and the baseline exists to be read
            # against the capture arm's log.
            params = [
                SamplingParams(
                    temperature=args.temperature, top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    **({} if args.no_connector else {"extra_args": {
                        "kv_transfer_params": {"ranges": as_pairs(segs)}}}),
                )
                for _st, _ids, _text, segs, _lines in owners
            ]
            print(f"round {round_idx + 1}: {len(batch)} chains, "
                  f"{sum(len(i) for _s, i, _t, _g, _l in owners)} prompt tokens")
            _t0 = time.perf_counter()
            outs = llm.generate(batch, params, use_tqdm=True)
            gen_seconds += time.perf_counter() - _t0

            # strict=True: vLLM returns outputs in submission order, but if it
            # ever returned fewer, zip would silently pair each turn with
            # ANOTHER chain's reply and every downstream number would be wrong
            # and plausible. Fail instead.
            for (st, ids, text, segs, lines), out in zip(owners, outs, strict=True):
                row = st.chain.rounds[round_idx]
                rid = getattr(out, "request_id", None)
                comp = out.outputs[0]
                raw = comp.text
                gen_text, gen_char_ends = decode_steps(tok, comp)
                got = parse_answer(raw)
                verdict, shortcut = classify(
                    got, row, getattr(comp, "finish_reason", None))
                # WHAT GOES INTO THE HISTORY -- the experiment's main lever.
                #
                #   "model"        the model's own reply, right or wrong. An
                #                  early mistake stays in the context and can
                #                  propagate. This is the condition the chains
                #                  were written for.
                #   "ground-truth" the CSV's answer instead. Every round is
                #                  answered from a correct history, so nothing
                #                  can propagate. The control arm: the gap
                #                  between the two IS the propagation effect,
                #                  measured directly rather than inferred from
                #                  comparing chains that happened to err early
                #                  against chains that did not.
                #
                # The ground-truth text uses the row's OWN declared OUTPUT
                # FORMAT ("ANSWER: <...>"), so nothing is injected that the
                # benchmark did not already specify.
                #
                # The <think> block never enters the history either way. Qwen's
                # guidance is that reasoning is not carried between turns, and
                # carrying it cost a run: a 1024-token block per turn inflated
                # the context until chains hit max_model_len and stopped early
                # -- 179 turns completed out of 214.
                ov = hist_override.get((st.chain.chain_id, row.round_no))
                if ov is not None:
                    history_answer = ov
                elif args.history == "ground-truth":
                    history_answer = f"ANSWER: {row.correct_answer}"
                else:
                    history_answer = strip_thinking(
                        raw, getattr(comp, "finish_reason", None) == "length")
                st.messages.append({"role": "assistant",
                                    "content": history_answer})
                turns.append(TurnResult(
                    chain_id=st.chain.chain_id, tier=st.chain.tier,
                    round_no=row.round_no, task_type=row.task_type,
                    request_id=rid,
                    n_prompt_tokens=len(ids), n_segments=len(segs),
                    segments=[asdict(s) for s in segs],
                    got=got, want=row.correct_answer,
                    verdict=verdict, ok=(verdict == CORRECT), shortcut=shortcut,
                    raw=raw.strip(), gen_seconds=0.0,
                    finish_reason=getattr(comp, "finish_reason", None),
                    gen_text=gen_text, gen_char_ends=gen_char_ends,
                ))
                write_prompt_line(prompts_fh, rid, st.chain.chain_id,
                                  row.round_no, len(ids), text, lines)

        prompts_fh.close()

        # ONE one-token nudge, after the last round rather than after each.
        # The connector flushes a finished request at the NEXT scheduler step,
        # and every round except the final one already has a next round to
        # provide those steps -- so only the last batch needs prompting. Its
        # record is purged below, leaving exactly one artifact per real turn.
        nudge_ids = tok(".", add_special_tokens=False)["input_ids"]
        llm.generate([TokensPrompt(prompt_token_ids=nudge_ids)],
                     SamplingParams(temperature=0.0, max_tokens=1),
                     use_tqdm=False)

        del llm
        # vLLM initialises an NCCL process group even at world_size=1 and does
        # not tear it down, so torch prints a resource-leak warning as the very
        # last line of every run. Harmless -- the process is exiting -- but it
        # is the final thing on screen after a long job, which makes a clean
        # run look like it ended badly. Destroy it explicitly instead.
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:       # never let cleanup cost us the results
            pass
        buffer = list(fc.get_buffer())

    # Drop the flush nudge so the artifact is exactly the experiment.
    buffer, purged = purge_unmatched(buffer, turns, out_root)

    # Did the probe actually run? `layers_seen` is populated the first time the
    # copy executes in a layer's forward. Zero layers means the override never
    # sat on the backend the engine chose, and the capture is all zeros -- a
    # failure that is otherwise completely silent, because the records are
    # still structurally valid and mass conservation still "passes" on zeros.
    probe_layers = len(getattr(registry, "layers_seen", ()) or ()) if registry else 0

    meta = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "workflow_id": args.workflow_id,
        "chains": len(chains),
        "turns": len(turns),
        "generate_seconds": round(gen_seconds, 3),
        "wall_seconds": round(time.perf_counter() - t_wall, 3),
        "connector": "off" if args.no_connector else "on",
        "attn_backend": args.attn_backend or "FLASH_ATTN (default)",
        # Recorded ALWAYS, empty string included. A run directory must be able
        # to answer "was there a wrapper" without inference: a missing key is
        # ambiguous between "no system prompt" and "a run predating the flag",
        # and those shift every token index differently.
        "system_prompt": args.system_prompt,
        "poison": poison_meta,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "enable_thinking": bool(args.enable_thinking),
        "history": args.history,
        "prefix_caching": bool(args.prefix_caching),
        "probe_layers_seen": probe_layers,
        "flush_records_purged": purged,
        "cuda_graphs": bool(args.cuda_graphs),
        "top_pct": args.top_pct,
        "chunk_size": args.chunk_size,
        "max_steps": args.max_steps,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "stopped": {s.chain.chain_id: s.stopped_because
                    for s in states if s.stopped_because},
    }
    return turns, buffer, meta


def tidy_run_dir(out_dir: str | Path) -> list[str]:
    """Move Flowcept's own dump under raw/, leaving one canonical copy on top.

    Flowcept writes `flowcept_buffer_<workflow>_<pid>_<tid>.jsonl` and keeps
    older ones (`delete_previous_file: false`). Those hold the buffer BEFORE
    the flush nudge was removed, so they disagree with `provenance.jsonl` by
    one record. Both are worth keeping -- one is the evidence, one is the
    experiment -- but they must not sit side by side where a reader, or
    `analyze.py`, might take either.
    """
    out = Path(out_dir)
    raw = out / "raw"
    moved = []
    for f in sorted(out.glob("flowcept_buffer*.jsonl")):
        raw.mkdir(exist_ok=True)
        f.rename(raw / f.name)
        moved.append(f.name)
    return moved


def purge_unmatched(buffer: list[dict], turns: list[TurnResult],
                    out_dir: str | Path) -> tuple[list[dict], int]:
    """Remove records that belong to no turn, and delete their tensor files.

    The only such record is the flush nudge -- a one-token "." request the
    engine needs in order to emit the final round's records. It is real work
    the connector captured, but it is not part of the experiment, and leaving
    it in means the artifact contains N+1 files for N turns and every count
    downstream is off by one.

    Matching mirrors `report.index_tasks`: vLLM's internal id is
    "<counter>-<hex>" while RequestOutput gives "<counter>", so compare on the
    leading component too.

    Files are only unlinked when they resolve INSIDE out_dir. A record whose
    uri points elsewhere is left alone and merely dropped from the buffer --
    deleting by a path that came out of a data file is not something this
    should do on trust.
    """
    import re as _re
    keep_ids = {t.request_id for t in turns if t.request_id}
    out_root = Path(out_dir).resolve()
    kept, purged = [], 0

    for rec in buffer:
        desc = rec.get("attention_stats") or {}
        rid = (rec.get("used") or {}).get("request_id")
        if rec.get("type") != "task" or not desc.get("uri") or not rid:
            kept.append(rec)
            continue
        base = _re.sub(r"-[0-9a-f]{6,}$", "", rid)
        if rid in keep_ids or base in keep_ids:
            kept.append(rec)
            continue
        uri = desc["uri"]
        path = Path(uri[len("file://"):] if uri.startswith("file://") else uri)
        try:
            if path.resolve().is_relative_to(out_root) and path.is_file():
                path.unlink()
        except (OSError, ValueError):
            pass
        purged += 1
    return kept, purged


def write_results(out_dir: str | Path, turns: list[TurnResult], meta: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "turns.jsonl").open("w") as fh:
        for t in turns:
            fh.write(json.dumps(asdict(t)) + "\n")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
