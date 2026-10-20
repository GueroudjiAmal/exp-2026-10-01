"""Loading `opal_chains.csv` and grouping its rows into chains.

The CSV is one row per (chain, round): chain_id, round_no, task_type, prompt,
tool_ranges, correct_answer, explanation. Rows of a chain are ordered by
round_no and must be replayed in that order -- a later round refers to what an
earlier one established and does not repeat it.

`tool_ranges` is deliberately NOT used; see `ranges.declare_ranges` for the two
reasons it cannot be.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

# OPAL prompts carry whole tool-output tables in one field, well past the
# default 128 KB limit.
csv.field_size_limit(10 ** 7)

# chain ids look like `exp114-chain-0001-low`; the tail is the difficulty tier.
_CHAIN_ID_RE = re.compile(r"^(?P<exp>exp\d+)-chain-(?P<num>\d+)-(?P<tier>\w+)$")


@dataclass
class Round:
    chain_id: str
    round_no: int
    task_type: str
    prompt: str
    correct_answer: str
    explanation: str


@dataclass
class Chain:
    chain_id: str
    rounds: list[Round] = field(default_factory=list)

    @property
    def tier(self) -> str:
        m = _CHAIN_ID_RE.match(self.chain_id)
        return m.group("tier") if m else "unknown"

    @property
    def experiment(self) -> str:
        m = _CHAIN_ID_RE.match(self.chain_id)
        return m.group("exp") if m else "unknown"

    def __len__(self) -> int:
        return len(self.rounds)


def load_chains(path: str | Path) -> list[Chain]:
    """Every chain in the CSV, rounds sorted, chains in first-appearance order."""
    by_id: dict[str, Chain] = {}
    with Path(path).open(newline="") as fh:
        for raw in csv.DictReader(fh):
            cid = raw["chain_id"]
            by_id.setdefault(cid, Chain(cid)).rounds.append(
                Round(
                    chain_id=cid,
                    round_no=int(raw["round_no"]),
                    task_type=raw["task_type"],
                    prompt=raw["prompt"],
                    correct_answer=raw["correct_answer"],
                    explanation=raw["explanation"],
                )
            )
    chains = list(by_id.values())
    for c in chains:
        c.rounds.sort(key=lambda r: r.round_no)
    return chains


def select(
    chains: list[Chain],
    *,
    limit: int = 0,
    tier: str | None = None,
    chain_ids: list[str] | None = None,
) -> list[Chain]:
    """Narrow the chain set. `limit=0` means all of them.

    Selection is applied in the order tier -> explicit ids -> limit, and never
    shuffles: a run with `--limit 10` is the first ten chains of the file every
    time, so two runs with the same flags compare directly.
    """
    out = chains
    if tier:
        out = [c for c in out if c.tier == tier]
    if chain_ids:
        wanted = set(chain_ids)
        out = [c for c in out if c.chain_id in wanted]
    if limit > 0:
        out = out[:limit]
    return out
