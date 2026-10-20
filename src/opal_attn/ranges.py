"""Declaring per-request attention segments in TOKEN space.

The connector partitions a prompt into segments and keeps the strongest
`top_pct` of positions within each. Declaring the segments ourselves -- one per
tool block, one per prior assistant answer -- is what makes the captured
attention interpretable: "round 4 attended to round 2's qc_lookup" rather than
"round 4 attended to tokens 1800-1830".

The CSV's own `tool_ranges` column is NOT usable, for two reasons. Both produce
attention attributed to the wrong text, and neither fails loudly.

1. WRONG TOKENIZER. The CSV records character spans under the generator's regex
   tokenizer, which splits far more coarsely than BPE. The error is not even a
   constant factor, since it depends how much of the text is numbers and
   punctuation -- and OPAL tool outputs are mostly numbers.

2. HISTORY SHIFTS EVERYTHING. CSV spans are relative to a round's own prompt.
   Once that prompt is turn N of a conversation, every index moves by the system
   prompt, all previous rounds, all previous answers, and the chat template's own
   markers. The shift also depends on WHOSE answers are in the history: these
   chains replay with the model's own replies, which tokenize to different
   lengths than ground truth. So it cannot be precomputed per round -- it has to
   be derived from the exact string handed to the tokenizer on that turn.

Hence: re-find the blocks in the rendered conversation, and map character spans
to token spans through the tokenizer's own offset mapping.

The same offset table also yields a LINE table (`declare_ranges_and_lines`).
Segments are the unit the connector selects within; lines are the unit a human
reads a heat map in. Both come from one tokenization because tokenizing a
multi-thousand-token conversation twice per turn is the slow part of a dry run.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

# `>>> db_lookup(...)` opens a tool block; the block runs to the next one or to
# the OUTPUT FORMAT footer, whichever comes first.
_TOOL_RE = re.compile(r"^>>> (\w+)", re.M)
# Qwen-style chat markers. An assistant turn in the HISTORY is one segment.
#
# `<|im_end|>` is REQUIRED, not optional. A history answer always carries one;
# the trailing generation prompt never does, because the model has not spoken
# yet. Accepting end-of-string as a terminator instead made the generation
# prompt look like an answer, and the only thing standing between that and a
# bogus segment was a test for an empty body.
#
# That test held for Qwen3-4B-Instruct-2507, whose template ends the render at
# "<|im_start|>assistant\n". It does NOT hold for Qwen3.5, whose template emits
# an empty "<think>\n\n</think>" reasoning block into the generation prompt
# even with enable_thinking=False -- a non-empty body, so the final turn was
# declared a history answer, mislabelled as the current round's own reply, and
# counted against the "round r carries r-1 answer segments" check.
_ANSWER_RE = re.compile(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", re.S)


@dataclass
class Segment:
    kind: str   # tool name, or "answer"
    lo: int     # token index, inclusive
    hi: int     # token index, exclusive
    round_hint: int = 0   # which round of the conversation the span came from


@dataclass
class Line:
    """One physical line of the rendered conversation, in token space.

    `lo >= hi` means the line occupies no tokens of its own (a blank line
    swallowed by a neighbouring token). Such lines are kept so that a rendered
    heat map has the same line numbering as the text it describes.
    """
    text: str
    lo: int
    hi: int
    round_hint: int = 0


def char_to_token_spans(text: str, tok):
    """`(offsets, tspan, n_tokens)` for one rendered conversation.

    `tspan(char_lo, char_hi) -> (tok_lo, tok_hi)` in the SAME index space as
    the `input_ids` the engine is handed -- which is the whole point, since a
    declared range is only meaningful if it indexes the tokens the connector
    scores.

    Zero-width offsets are excluded from the SEARCH (a token covering no
    characters can never be the boundary of a character span) but not from the
    INDEXING: the returned indices are positions in the full token sequence.
    Collapsing the two is a silent corruption -- many tokenizers map special
    tokens such as `<|im_start|>` to (0, 0), and indexing a filtered list would
    then shift every declared range left by the number of specials before it,
    with no assertion downstream able to notice. Qwen happens to emit real
    offsets for its specials, so this costs nothing there and is the difference
    between right and wrong everywhere else.

    Lookups are binary searches rather than scans: the line table asks for one
    span per line, and a linear probe per span makes a 270-line conversation
    quadratic.
    """
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    pairs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    n_tokens = len(pairs)

    keep = [i for i, (a, b) in enumerate(pairs) if b > a]
    offs = [pairs[i] for i in keep]
    starts = [a for a, _b in offs]
    ends = [b for _a, b in offs]

    def tspan(cs: int, ce: int) -> tuple[int | None, int]:
        i = bisect_right(ends, cs)          # first kept token ending past cs
        if i >= len(offs):
            return None, n_tokens
        j = bisect_left(starts, ce)         # first kept token starting at/after ce
        return keep[i], (keep[j] if j < len(keep) else n_tokens)

    return offs, tspan, n_tokens


def _round_marker_positions(text: str) -> list[int]:
    return [m.start() for m in re.finditer(r"<\|im_start\|>user", text)]


def declare_ranges(text: str, tok) -> list[Segment]:
    """One segment per tool block and per assistant answer in the conversation.

    `text` is the fully rendered conversation -- what `apply_chat_template`
    produced and what the tokenizer will see -- not a single round's prompt.

    Returned segments are sorted, non-overlapping and clamped to each other.
    Gaps between them (prose, rules, the chat template's own markers) are left
    alone: the connector turns every gap into a segment of its own, so the
    partition stays complete and `topk_residual` stays exact.
    """
    return declare_ranges_and_lines(text, tok)[0]


def declare_ranges_and_lines(text: str, tok) -> tuple[list[Segment], list[Line]]:
    """`declare_ranges`, plus every physical line of `text` in token space.

    One tokenization serves both. The line table is what a heat map is drawn
    on: the connector selects positions WITHIN a segment, but "which line did
    the model look at" is the question a reader actually asks, and a tool block
    is twenty lines of table.

    Lines carry `round_hint` too, so a position in undeclared prose -- the
    round's own question text, the RULES, the OUTPUT FORMAT footer -- can still
    be attributed to the round it came from. `hottest_segments` cannot do that,
    which is why it reports such positions as an unattributed "prose gap".
    """
    _offs, tspan, n_tokens = char_to_token_spans(text, tok)
    turn_starts = _round_marker_positions(text)

    def round_of(pos: int) -> int:
        return sum(1 for s in turn_starts if s <= pos)

    out: list[Segment] = []
    for m in _TOOL_RE.finditer(text):
        start = m.start()
        nxt = text.find("\n>>> ", start + 1)
        fmt = text.find("\nOUTPUT FORMAT", start)
        end = min([x for x in (nxt, fmt) if x > 0], default=len(text))
        lo, hi = tspan(start, end)
        if lo is not None and hi > lo:
            out.append(Segment(m.group(1), lo, hi, round_of(start)))

    for m in _ANSWER_RE.finditer(text):
        body = m.group(1).strip()
        if not body:            # the trailing generation prompt, not an answer
            continue
        lo, hi = tspan(m.start(1), m.start(1) + len(m.group(1)))
        if lo is not None and hi > lo:
            out.append(Segment("answer", lo, hi, round_of(m.start())))

    out.sort(key=lambda s: s.lo)
    clean: list[Segment] = []
    for s in out:               # the connector truncates overlaps anyway; be explicit
        if clean and s.lo < clean[-1].hi:
            s.lo = clean[-1].hi
        if s.hi > s.lo:
            clean.append(s)

    lines: list[Line] = []
    cursor = 0
    for raw_line in text.split("\n"):
        c0, c1 = cursor, cursor + len(raw_line)
        cursor = c1 + 1                      # the newline itself
        lo, hi = tspan(c0, c1) if c1 > c0 else (None, 0)
        if lo is None:
            lines.append(Line(raw_line, -1, -1, round_of(c0)))
        else:
            lines.append(Line(raw_line, lo, max(lo, hi), round_of(c0)))

    # Close the gaps so the table TILES the token space. A line's character
    # span stops before its newline, and a byte-level tokenizer often folds
    # that newline (plus the next line's indent) into one token -- so spans
    # taken literally leave holes, and a retained position landing in one would
    # be attributed to no line at all. Each line is extended to where the next
    # line's first token begins, which gives every token exactly one owner and
    # puts a line-terminating newline on the line it terminates.
    real = [ln for ln in lines if ln.hi > ln.lo]
    for prev, nxt in zip(real, real[1:], strict=False):   # deliberately offset
        prev.hi = max(prev.hi, nxt.lo)
    if real:
        real[-1].hi = max(real[-1].hi, n_tokens)
    return clean, lines


def as_pairs(segments: list[Segment]) -> list[list[int]]:
    """The `[[lo, hi], ...]` form the connector takes in `kv_transfer_params`."""
    return [[s.lo, s.hi] for s in segments]


def kept_entries(segments: list[Segment], n_prompt: int, top_pct: float,
                 chunk_size: int = 32) -> int:
    """How many positions the connector will retain per decode step.

    Mirrors the connector's own arithmetic -- `max(1, round(top_pct/100 * len))`
    per segment, over the declared segments AND the gaps it synthesises between
    them. Used by the planner to project file sizes before burning node hours.
    """
    spans = [(s.lo, s.hi) for s in segments]
    filled: list[tuple[int, int]] = []
    cursor = 0
    for lo, hi in spans:
        if lo > cursor:
            # the connector chunks an undeclared gap into `chunk_size` blocks
            for a in range(cursor, lo, chunk_size or (lo - cursor)):
                filled.append((a, min(a + (chunk_size or (lo - cursor)), lo)))
        filled.append((lo, hi))
        cursor = max(cursor, hi)
    if cursor < n_prompt:
        step = chunk_size or (n_prompt - cursor)
        for a in range(cursor, n_prompt, step):
            filled.append((a, min(a + step, n_prompt)))
    return sum(max(1, round(top_pct / 100.0 * (hi - lo))) for lo, hi in filled)

# Header -> section name, matched against a single rendered line. Used by
# `sections_from_lines`, which needs no tokenizer.
_SECTION_HEADERS = [
    ("session",       re.compile(r"^SESSION$")),
    ("tools",         re.compile(r"^TOOLS \(")),
    ("terms",         re.compile(r"^TERMS$")),
    ("question",      re.compile(r"^ROUND \d+ - TASK")),
    ("rules",         re.compile(r"^RULES$")),
    ("tool_outputs",  re.compile(r"^TOOL OUTPUTS$")),
    ("output_format", re.compile(r"^OUTPUT FORMAT$")),
]


def sections_from_lines(lines) -> list[Segment]:
    """Prose sections as SEPARATE INSTANCES, from a `prompts.jsonl` line table.

    `lines` is the stored `[[lo, hi, round_hint, text], ...]`, so this needs no
    tokenizer and works on a run captured with a model whose tokenizer is not
    available locally.

    ONE SEGMENT PER OCCURRENCE, which is the whole point. A round-7 turn
    contains seven "OUTPUT FORMAT" blocks at very different distances from the
    decode point, and attention density falls 10-18x across that range. Folding
    them into one span per turn -- which an earlier version of this analysis
    did -- silently keeps only the nearest and makes every prose section look
    like it only ever appears close to the answer. The tool segments were
    already per-instance, so prose and tools were being compared on different
    units.
    """
    out: list[Segment] = []
    cur: Segment | None = None
    for lo, hi, round_hint, text in lines:
        name = next((k for k, rx in _SECTION_HEADERS if rx.match(text)), None)
        if name is not None:
            if cur is not None and cur.hi > cur.lo:
                out.append(cur)
            cur = Segment(name, lo if hi > lo else -1, hi, round_hint)
            continue
        if cur is not None and hi > lo:
            if cur.lo < 0:
                cur.lo = lo
            cur.hi = hi
    if cur is not None and cur.hi > cur.lo:
        out.append(cur)
    return [s for s in out if s.lo >= 0 and s.hi > s.lo]
