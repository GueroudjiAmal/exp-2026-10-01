"""Scoring a model reply against an OPAL row.

Accuracy is REPORTED, never asserted. OPAL is hard and a low score is a fact
about the model, not a bug in the capture -- a run whose provenance is perfect
and whose accuracy is 20% is a successful run.

The `explanation` column is the useful part. It names the wrong answers you get
by ignoring one particular tool result, so a wrong reply can often be classified
as a specific documented shortcut rather than noise. "The model skipped the QC
table" is a finding; "the model was wrong" is not.
"""

from __future__ import annotations

import re

ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)

# `... ; db_lookup gives 1000; qc_lookup gives 0 on 5 plants ...`
_ABLATION_RE = re.compile(r"(?:^|; )(?:ablations: )?([a-z][^;]*?) gives ([^;)]+)")


def parse_answer(text: str) -> str | None:
    """The model's final `ANSWER:` line, or None if it never produced one.

    The last match wins, not the first: a model that reasons aloud before
    complying often emits a provisional `ANSWER:` mid-stream.

    Returns None rather than guessing. Every OPAL row ends with an explicit
    OUTPUT FORMAT demanding one line beginning "ANSWER:", so a reply without
    one did not follow the instruction -- and scoring its last line anyway
    silently converts a format failure into a content result, which is a
    different measurement reported under the same name.
    """
    hits = ANSWER_RE.findall(text or "")
    return hits[-1].strip().rstrip(".") if hits else None


def _norm(s: str) -> str:
    """Normalise for comparison: case, whitespace, trailing period, and the
    spacing AROUND separators.

    "fc1,rgb2,msc1" and "fc1, rgb2, msc1" are the same answer -- the benchmark
    asks for "comma separated" and never says whether to put a space after the
    comma. Likewise "WELL_REPLICATED;9" and "WELL_REPLICATED; 9".

    This is the only leniency. Membership and ORDER are untouched, so a
    different set, a missing member, or a different sequence all still score
    wrong -- which matters, because these prompts DO fix the order ("in table
    column order") and the field sequence.
    """
    t = re.sub(r"\s+", " ", s or "").strip().strip(".").lower()
    return re.sub(r"\s*([,;])\s*", r"\1 ", t)


def same(got: str, want: str) -> bool:
    """Exact comparison, up to case, whitespace and a trailing period.

    Nothing else is forgiven. Only meaningful where `scorable(want)` holds.
    """
    return _norm(got) == _norm(want)


def ablations(explanation: str) -> dict[str, str]:
    """Documented wrong answers from the `explanation` column, by shortcut name."""
    return {m.group(1).strip(): m.group(2).strip()
            for m in _ABLATION_RE.finditer(explanation or "")}


def _head(value: str) -> str:
    """The comparable part of an answer or a parsed ablation value.

    The column is prose, so two things survive parsing and have to be trimmed
    here: a composite value cut at its first ";" ("1000; DISAGREES" arrives as
    "1000"), and values trailing explanatory words ("0 on 5 plants").
    """
    value = re.split(r"[;@]", value or "", maxsplit=1)[0]
    value = re.split(r"\s+(?:on|which|is)\s", value, maxsplit=1)[0]
    return value.strip()


def matches_ablation(got: str, value: str) -> bool:
    return same(got, value) or same(_head(got), _head(value))


# A turn's outcome. Kept distinct on purpose: lumping them together is what
# made two task types look like model failures when they were grading failures.
CORRECT = "correct"
WRONG = "wrong"
FORMAT_ERROR = "format_error"    # the reply had no ANSWER: line at all
# The reply hit the token cap mid-sentence, so the model never committed to an
# answer. NOT scorable, and not the same as a wrong answer.
#
# This needs its own verdict because `parse_answer` cannot tell the difference
# on text alone. A reasoning model quotes the output format back to itself
# while it works ("the line should read `ANSWER: 0; DISAGREES`"), and states
# provisional conclusions the same way. The regex finds those, the LAST one
# wins, and a turn the model never finished is scored as content.
#
# Measured on a 90-turn Qwen3.5 run: 12 replies hit the cap. Eleven scored
# "wrong" and one scored "correct" -- a false positive, from a fragment that
# happened to match. Reported accuracy was 0.8778; over the 78 replies that
# actually finished it was 1.0000. The 12 were noise presented as signal.
TRUNCATED = "truncated"
# There was an UNSCORABLE verdict here, for answers containing "," or ";". It
# was wrong: the prompts specify the order and the field sequence --
# "<modalities, comma separated, in table column order>" and
# "<WELL_REPLICATED or UNDERPOWERED>; <usable plants>" -- so exact comparison
# IS the correct grading for them, and excluding them discarded 199 of 430
# turns, including task types that graded fine. Older turns.jsonl files may
# still carry the string; report.py keeps tolerating it.


def classify(got: str | None, row, finish_reason: str | None = None
             ) -> tuple[str, str | None]:
    """`(verdict, shortcut_name)` for one reply.

    Every well-formed reply is graded by exact comparison against the CSV's
    answer, normalised only for case, whitespace and separator spacing. A reply
    with no ANSWER: line is a FORMAT_ERROR instead -- there is nothing to
    compare it against.

    `finish_reason` comes from the engine and OVERRIDES the text. "length"
    means the reply was cut off at the token cap, so whatever the regex found
    was mid-thought, not an answer. Trusting the text there invents a verdict
    out of an unfinished sentence -- in both directions.
    """
    if finish_reason == "length":
        return TRUNCATED, None
    if got is None:
        return FORMAT_ERROR, None
    if same(got, row.correct_answer):
        return CORRECT, None
    for name, wrong in ablations(row.explanation).items():
        if matches_ablation(got, wrong):
            return WRONG, name
    return WRONG, None
