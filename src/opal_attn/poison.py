"""Poisoning OPAL chains, with the edit held to the same token count.

The design this serves, in one line: inject ONCE at round 1, observe the
effect at every later round. Form is manipulated (history / prose /
data-table); distance is observed, not manipulated, because the two cannot be
separated -- placing a poison further back also places it upstream of more
inference, and that confound is the thing the chains exist to study.

WHY TOKEN PARITY IS A HARD GATE. A substitution that changes the token count
shifts every index after it. That breaks two things at once: the clean twin no
longer shares spans with the poisoned run, and every distance-from-decode in
the analysis is re-based -- and distance is the axis the whole design rests on.

Character parity does NOT give token parity. Measured on Qwen3-4B-Instruct-2507:

    primary -> derived    7 chars both    1 -> 1 tokens   balanced
    AGREES  -> DENIES     6 chars both    3 -> 2 tokens   SHIFTS
    fc1     -> s3d        3 chars both    2 -> 3 tokens   SHIFTS

BPE merges depend on content, so equal strings are not equal token counts. And
it is per-tokenizer: a pair that balances on Qwen3-4B need not balance on
Qwen3.5, whose vocabulary is 248k against 151k. The check therefore runs
against the model actually being used, every time, not once at authoring time.

The count is compared on the WHOLE rendered prompt rather than on the
substring, because a merge can span the boundary: "0.528" alone may tokenize
to five tokens while "t0 0.528 " tokenizes differently either side of the edit.
Only the end-to-end count is the thing that matters.

NOTHING HERE IS DECLARED AS A SEGMENT. `attn_sum` is accumulated over all
positions before top_pct selection, so a poisoned span's attention mass can be
summed post hoc on a run that never declared it. Declaring would change the
partition, hand the span a guaranteed retention quota, and make every earlier
run non-comparable -- for no gain. Poisons carry spans; they do not create
segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .chains import Chain, Round


class PoisonError(ValueError):
    """A poison that cannot be applied as specified. Never silently skipped."""


@dataclass(frozen=True)
class Poison:
    """One arm. `find` must occur EXACTLY once in the target round's prompt.

    `where`:
        "prompt"  -- edit the round's text (prose and data-table arms)
        "history" -- replace the assistant message the runner appends after
                     this round (the history arm). The text is never shown to
                     the model as a tool output; it is substituted for what
                     the model said, which is what makes it laundering rather
                     than injection.

    `legitimises` names the CSV ablation this poison makes CORRECT, where one
    applies. That is the sharpest cell in the design: a model that takes the
    documented shortcut now scores right and a careful one scores wrong, so
    accuracy inverts and only provenance separates them. `answers.ablations()`
    already parses these names out of the `explanation` column.

    `trigger` is reserved for conditional (backdoor) variants -- a predicate
    naming the condition under which the poison is active. Unused today; it is
    here so adding one later is a field, not a rewrite.
    """

    name: str
    form: str                       # history | prose | data-table
    kind: str                       # target segment kind, for reporting
    # `find` is meaningless for where="history": there is no text to match,
    # the runner substitutes the whole assistant message. Defaulted so a
    # history arm need not carry a dummy, and validated below so a prompt arm
    # cannot omit it by accident.
    find: str = ""
    replace: str = ""
    where: str = "prompt"
    round_no: int = 1
    legitimises: str | None = None
    trigger: str | None = None

    def __post_init__(self) -> None:
        if self.where not in ("prompt", "history"):
            raise PoisonError(f"{self.name}: where must be 'prompt' or "
                              f"'history', not {self.where!r}")
        if self.where == "prompt" and not self.find:
            raise PoisonError(f"{self.name}: a prompt poison needs `find`")
        if not self.replace:
            raise PoisonError(f"{self.name}: `replace` is empty. A deletion "
                              f"cannot be token-length preserving; pad it to "
                              f"the same token count instead.")


@dataclass
class PoisonRecord:
    """What was actually done to one chain, for run_meta and the analysis."""

    name: str
    form: str
    kind: str
    chain_id: str
    round_no: int
    where: str
    char_lo: int
    char_hi: int
    before: str
    after: str
    tokens_before: int
    tokens_after: int
    legitimises: str | None = None

    @property
    def balanced(self) -> bool:
        return self.tokens_before == self.tokens_after


def locate(text: str, find: str) -> tuple[int, int]:
    """The one char span of `find`, or an error naming why it is not one.

    Uniqueness is required rather than "first match": OPAL tables repeat
    values across rows, and poisoning a different cell than intended would be
    invisible afterwards -- the run would look fine and measure something else.
    """
    n = text.count(find)
    if n != 1:
        raise PoisonError(
            f"{find!r} occurs {n} times, need exactly 1. Widen the pattern "
            f"with surrounding context until it is unique.")
    lo = text.index(find)
    return lo, lo + len(find)


def token_parity(tok, text: str, p: Poison) -> tuple[int, int]:
    """`(tokens_before, tokens_after)` for the WHOLE text, under `tok`.

    Whole-text rather than substring, because a BPE merge can straddle the
    edit boundary and change the count outside the span itself.
    """
    lo, hi = locate(text, p.find)
    after = text[:lo] + p.replace + text[hi:]
    n_before = len(tok(text, add_special_tokens=False)["input_ids"])
    n_after = len(tok(after, add_special_tokens=False)["input_ids"])
    return n_before, n_after


def apply_to_round(p: Poison, row: Round, tok, *, strict: bool = True
                   ) -> tuple[Round, PoisonRecord]:
    """Poison one round's prompt. Raises unless the token count is preserved.

    `strict=False` records the imbalance and applies anyway -- only for
    deliberately exploring a substitution that cannot be balanced, in which
    case that arm's distances must be re-derived and NOT compared with the
    others.
    """
    lo, hi = locate(row.prompt, p.find)
    n_before, n_after = token_parity(tok, row.prompt, p)
    if strict and n_before != n_after:
        raise PoisonError(
            f"poison {p.name!r} is not token-length preserving under this "
            f"tokenizer: {n_before} -> {n_after} ({n_after - n_before:+d}). "
            f"Every index after the edit would shift, which re-bases the "
            f"distance axis and breaks the clean twin. Pad the replacement to "
            f"the same token count, pick another substitution, or pass "
            f"strict=False and treat this arm as non-comparable.")
    poisoned = replace(row, prompt=row.prompt[:lo] + p.replace + row.prompt[hi:])
    rec = PoisonRecord(
        name=p.name, form=p.form, kind=p.kind, chain_id=row.chain_id,
        round_no=row.round_no, where=p.where, char_lo=lo, char_hi=lo + len(p.replace),
        before=p.find, after=p.replace, tokens_before=n_before,
        tokens_after=n_after, legitimises=p.legitimises)
    return poisoned, rec


@dataclass
class PoisonedChain:
    chain: Chain
    records: list[PoisonRecord] = field(default_factory=list)
    # round_no -> the assistant text to substitute, for `where="history"`.
    # The runner reads this instead of the model's own reply for that round.
    history_overrides: dict[int, str] = field(default_factory=dict)


def apply_to_chain(chain: Chain, poisons: list[Poison], tok, *,
                   strict: bool = True) -> PoisonedChain:
    """Apply every poison that targets this chain. Order is not significant.

    A poison whose `find` is absent from the target round raises, rather than
    being skipped: an arm that silently applies to 3 of 20 chains produces a
    number that looks like an effect size and is not one.
    """
    rounds = {r.round_no: r for r in chain.rounds}
    out = PoisonedChain(chain=Chain(chain.chain_id, list(chain.rounds)))
    by_no = {r.round_no: i for i, r in enumerate(out.chain.rounds)}

    for p in poisons:
        row = rounds.get(p.round_no)
        if row is None:
            raise PoisonError(
                f"poison {p.name!r} targets round {p.round_no}, which "
                f"{chain.chain_id} does not have ({len(chain.rounds)} rounds)")
        if p.where == "history":
            out.history_overrides[p.round_no] = p.replace
            out.records.append(PoisonRecord(
                name=p.name, form=p.form, kind=p.kind, chain_id=chain.chain_id,
                round_no=p.round_no, where=p.where, char_lo=-1, char_hi=-1,
                before="<the model's own reply>", after=p.replace,
                tokens_before=0, tokens_after=0, legitimises=p.legitimises))
            continue
        i = by_no[p.round_no]
        poisoned, rec = apply_to_round(p, out.chain.rounds[i], tok, strict=strict)
        out.chain.rounds[i] = poisoned
        out.records.append(rec)
    return out


def check(poisons: list[Poison], chains: list[Chain], tok) -> list[dict]:
    """Dry-run every poison against every chain. No edits, no GPU.

    Run this before a capture. A poison that is ambiguous, absent or
    unbalanced is a spec bug, and finding it here costs seconds rather than a
    run and an analysis that quietly measured the wrong span.
    """
    out = []
    for p in poisons:
        ok = bad = 0
        reasons: list[str] = []
        for c in chains:
            row = next((r for r in c.rounds if r.round_no == p.round_no), None)
            if row is None or p.where == "history":
                ok += p.where == "history"
                continue
            try:
                n0, n1 = token_parity(tok, row.prompt, p)
                if n0 == n1:
                    ok += 1
                else:
                    bad += 1
                    reasons.append(f"{c.chain_id}: {n0}->{n1} tokens")
            except PoisonError as e:
                bad += 1
                reasons.append(f"{c.chain_id}: {e}")
        out.append({"name": p.name, "form": p.form, "kind": p.kind,
                    "applies": ok, "fails": bad, "first_failures": reasons[:3]})
    return out


def find_balanced(tok, text: str, find: str, candidates: list[str]
                  ) -> list[tuple[str, int]]:
    """`[(candidate, delta)]` sorted with the token-balanced ones first.

    The parity gate is easy to state and hard to satisfy by hand: adding a
    digit to a table cell costs a token, and an English rewrite of the same
    character length almost never lands on the same count. Rejecting those is
    correct, but rejection alone just moves the work to guesswork.

    So: supply several SEMANTICALLY EQUIVALENT phrasings of the poison and let
    the tokenizer choose. The ones with delta 0 are usable as-is; the rest are
    reported with their cost so a near-miss can be padded deliberately rather
    than discovered during a run.

    Searching over phrasing rather than over meaning is what keeps this
    honest. The arm's semantics are fixed by the author; only the wording
    varies, and only to satisfy a constraint of the measurement.
    """
    lo, hi = locate(text, find)
    n0 = len(tok(text, add_special_tokens=False)["input_ids"])
    out = []
    for cand in candidates:
        n1 = len(tok(text[:lo] + cand + text[hi:],
                     add_special_tokens=False)["input_ids"])
        out.append((cand, n1 - n0))
    return sorted(out, key=lambda kv: (abs(kv[1]), len(kv[0])))


def apply_manifest(chains: list[Chain], spec_path, tok, *, strict: bool = True
                   ) -> tuple[list[Chain], dict, dict]:
    """`(chains, history_overrides, meta)` for a manifest built per arm.

    Shared by the real run and the dry run on purpose. The dry run is the
    pre-flight, so it has to exercise the SAME substitution path: a manifest
    whose parity fails under this model must abort without a GPU, not after
    one. An earlier version applied poisons only in the runner, and the dry
    run then reported identical token counts for every arm -- which looks like
    a passing check and is actually a check that never ran.
    """
    import json as _json
    from pathlib import Path as _Path

    spec = _json.loads(_Path(spec_path).read_text())
    out = list(chains)
    hist: dict[tuple[str, int], str] = {}
    applied: list[str] = []

    for i, c in enumerate(out):
        ent = spec["chains"].get(c.chain_id)
        if ent is None:
            continue
        p = Poison(name=spec["arm"], form=spec["form"], kind=spec["kind"],
                   find=ent.get("find", ""), replace=ent["replace"],
                   where=spec.get("where", "prompt"),
                   round_no=spec.get("round_no", 1))
        pc = apply_to_chain(c, [p], tok, strict=strict)
        out[i] = pc.chain
        for rn, txt in pc.history_overrides.items():
            hist[(c.chain_id, rn)] = txt
        applied.append(c.chain_id)

    meta = {"arm": spec["arm"], "form": spec["form"], "kind": spec["kind"],
            "where": spec.get("where", "prompt"),
            "round_no": spec.get("round_no", 1),
            "spec": str(_Path(spec_path).resolve()),
            "built_for_model": spec.get("model"),
            "chains_poisoned": len(applied), "chain_ids": sorted(applied)}
    return out, hist, meta
