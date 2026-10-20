"""Verifying a capture and reporting what it shows.

Two separate jobs, deliberately not merged:

`verify` asks whether the provenance is TRUSTWORTHY -- a record per turn, the
declared segments present verbatim, the kept mass plus the discarded residual
summing to one, the manifest agreeing with the file on disk. These are
pass/fail, and a failure means the capture is not usable.

`summarise` asks what the run SHOWS -- accuracy, where chains break, which
segment the model attended to hardest. None of it is asserted. OPAL is hard;
a low score is a fact about the model.
"""

from __future__ import annotations

import bisect
import collections
import json
import re
from pathlib import Path

# vLLM's engine-internal request id is "<counter>-<hex>", while the
# RequestOutput handed back by LLM.generate carries only "<counter>". The
# connector sees the former, the runner records the latter, so a literal
# comparison joins nothing.
#
# Matched narrowly -- a trailing dash plus >=6 hex characters -- rather than by
# splitting on the first "-", so an id that legitimately contains dashes is not
# silently truncated into a wrong (but plausible) key.
_REQ_SUFFIX_RE = re.compile(r"^(?P<base>.+)-[0-9a-f]{6,}$")


def _id_aliases(request_id: str) -> set[str]:
    """Every spelling of one request id that a join might legitimately use."""
    out = {request_id}
    m = _REQ_SUFFIX_RE.match(request_id)
    if m:
        out.add(m.group("base"))
    return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_buffer(path: str | Path) -> list[dict]:
    """Flowcept's offline JSONL dump, or a list already in memory."""
    recs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        recs.extend(obj if isinstance(obj, list) else [obj])
    return recs


def load_turns(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]


def attention_tasks(buffer: list[dict]) -> list[dict]:
    """Every task record that points at a tensor file."""
    return [r for r in buffer
            if r.get("type") == "task"
            and (r.get("attention_stats") or {}).get("uri")]


def index_tasks(buffer: list[dict]) -> dict[str, dict]:
    """Attention tasks keyed by every alias of their request id.

    The Flowcept adapter stores the id with the connector's `:g<group>` suffix
    already stripped, but what remains is vLLM's INTERNAL id
    ("<counter>-<hex>"), not the "<counter>" that `RequestOutput.request_id`
    returns to the caller. Both spellings are indexed so a lookup by either
    succeeds.

    Note this dict is longer than the number of records -- use
    `attention_tasks` to count them.
    """
    out: dict[str, dict] = {}
    for r in attention_tasks(buffer):
        rid = (r.get("used") or {}).get("request_id")
        if not rid:
            continue
        for alias in _id_aliases(rid):
            out.setdefault(alias, r)
    return out


def attention_config(buffer: list[dict]) -> dict:
    """Per-run connector settings, recorded once on the connector's workflow.

    The connector creates its OWN workflow (`vllm-attn-<uuid>`) whose parent is
    the workflow id passed in `kv_connector_extra_config`, and hangs the config
    there -- so this looks for the record that HAS the field rather than for a
    particular id.
    """
    for w in buffer:
        if w.get("type") == "workflow" and w.get("attention_config"):
            return w["attention_config"]
    return {}


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def checksum_manifest(buffer: list[dict]) -> dict[str, str]:
    """`{relative path: sha256}` for every tensor file the provenance names.

    The connector already hashes each file as it writes it (`checksum: true`,
    the default), so a transfer can be verified against the provenance itself
    rather than against a manifest generated after the fact -- which would only
    prove the copy matches the copy. Paths are the last two components,
    `<connector-workflow-id>/<request>_g<n>.safetensors`, which is how they sit
    inside a run directory on any host.

    Empty when the connector ran with `checksum: false`; callers should say so
    rather than silently verifying nothing.
    """
    out: dict[str, str] = {}
    for rec in buffer:
        desc = rec.get("attention_stats") or {}
        uri, sha = desc.get("uri"), desc.get("sha256")
        if not uri or not sha:
            continue
        # "attention/<connector-workflow-id>/<request>_g<n>.safetensors" --
        # the path relative to the run directory, which is how the file sits
        # on any host it is copied to.
        parts = uri.rstrip("/").split("/")
        out["attention/" + "/".join(parts[-2:])] = sha
    return out


def find_run_buffer(run_dir, workflow_id: str | None = None
                    ) -> tuple[Path | None, list[str]]:
    """The canonical record set for a run directory, plus anything worth saying.

    Precedence, and why each step exists:

    `provenance.jsonl` is what the run declares as its records -- exactly the
    turns, flush nudge removed. Anything under `raw/` is Flowcept's own dump
    and disagrees with it by that one record.

    Older runs have neither, just `flowcept_buffer_<workflow>_<pid>_<tid>.jsonl`
    in the run directory. There can be SEVERAL: `delete_previous_file: false`
    keeps old ones and the pid makes each name unique, so a directory reused
    for two runs holds both while `turns.jsonl` holds only the second. Picking
    by size (an earlier version of this) can then pair one run's answers with
    another's provenance -- which does not error, it just reports something
    plausible. `append_workflow_id_to_path` puts the workflow in the name, so
    match on that when the caller knows it, and fall back to newest.

    Returns `(path, notes)`; notes are for the caller to print. Keeping this
    side-effect free matters because `push_results.sh` calls it to build a
    checksum manifest on stdout.
    """
    run_dir = Path(run_dir)
    notes: list[str] = []

    canonical = run_dir / "provenance.jsonl"
    if canonical.is_file():
        return canonical, notes

    cands = sorted(run_dir.glob("flowcept_buffer*.jsonl")) or \
        sorted((run_dir / "raw").glob("flowcept_buffer*.jsonl"))
    if not cands:
        return None, notes

    if workflow_id:
        exact = [p for p in cands if workflow_id in p.name]
        if exact:
            if len(cands) > len(exact):
                notes.append(f"{len(cands)} buffer files here; using the "
                             f"{len(exact)} matching workflow {workflow_id!r}")
            return max(exact, key=lambda p: p.stat().st_mtime), notes
        notes.append(f"none of the {len(cands)} buffer files name workflow "
                     f"{workflow_id!r}; falling back to the newest")

    if len(cands) > 1:
        notes.append(f"{len(cands)} buffer files in {run_dir} -- using the "
                     f"newest. A directory should hold ONE run; mixing them "
                     f"pairs one run's turns with another's provenance.")
        notes += [f"   {c.name}  ({c.stat().st_size} B)" for c in
                  sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)]
    return max(cands, key=lambda p: p.stat().st_mtime), notes


def _read(load_provenance, rec: dict, run_dir):
    """Call the loader with `run_dir` when it accepts one.

    `opal_attn.provenance.load_provenance` rebases a stale absolute uri onto a
    moved run directory; `vllm_attn_connector.load_provenance` does not take
    the argument at all. Supporting both keeps this module usable with either.
    """
    if run_dir is None:
        return load_provenance(rec)
    try:
        return load_provenance(rec, run_dir=run_dir)
    except TypeError:
        return load_provenance(rec)


def _shape(nested) -> list[int]:
    shape = []
    cur = nested
    while isinstance(cur, list):
        shape.append(len(cur))
        cur = cur[0] if cur else None
    return shape


def verify(turns: list[dict], buffer: list[dict], *, top_pct: float,
           load_provenance=None, run_dir=None, meta=None) -> list[tuple[bool, str]]:
    """Trustworthiness checks. `(passed, message)` per check.

    Separate from `summarise` on purpose: these are pass/fail statements about
    whether the provenance can be believed. What the run SHOWS is a different
    question, and mixing the two invites reading a low score as a broken
    capture, or a clean capture as a good result.
    """
    checks: list[tuple[bool, str]] = []

    def chk(cond, msg):
        checks.append((bool(cond), msg))

    # First, because every other check passes on an all-zero capture. The probe
    # overrides one attention backend; if the engine picked a different one it
    # never fired, and the records are valid-looking zeros.
    cfg_early = attention_config(buffer)
    if meta and meta.get("connector") == "on":
        seen = meta.get("probe_layers_seen", 0)
        scored = (cfg_early or {}).get("num_layers_scored") or 0
        chk(seen > 0,
            f"the query probe fired ({seen} layers; backend "
            f"{meta.get('attn_backend')}). Zero means the engine used a "
            f"different backend and every value is 0 -- re-run with "
            f"--attn-backend set to what the engine logged.")
        # "> 0" is not enough. A hybrid model -- only some layers full
        # attention, the rest linear -- lets the probe attach to a handful and
        # look healthy while the connector scores nothing: Qwen3.5-35B-A3B
        # reported 10 of 40 and produced an entirely zero capture. Compare the
        # probe's reach against the layers the connector says it scores.
        if scored:
            chk(seen >= scored,
                f"the probe covers every scored layer ({seen} probed vs "
                f"{scored} scored). Fewer probed than scored means the two "
                f"disagree about which layers exist -- common in hybrid or "
                f"multimodal models, and it yields a zero capture.")

    tasks = index_tasks(buffer)
    records = attention_tasks(buffer)
    chk(records, f"attention records emitted ({len(records)})")
    if not tasks:
        return checks

    cfg = attention_config(buffer)
    chk(cfg, "attention_config recorded on the workflow")
    if cfg:
        chk(cfg.get("top_pct") == top_pct,
            f"top_pct honoured ({cfg.get('top_pct')} vs {top_pct})")
        chk(cfg.get("metric") == "decode_attention",
            f"metric is decode_attention ({cfg.get('metric')!r})")

    matched = [t for t in turns if t.get("request_id") in tasks]
    # When NOTHING joins, the ids themselves are the only useful information,
    # so print them here rather than making someone write a one-off script.
    detail = ""
    if not matched:
        tids = [repr(t.get("request_id")) for t in turns[:4]]
        rids = [repr((r.get("used") or {}).get("request_id")) for r in records[:4]]
        detail = (f"\n          turns say   : {', '.join(tids)}"
                  f"\n          records say : {', '.join(rids)}")
    # `>= len(turns) - 2` tolerates records lost to the teardown flush, but on
    # a short run that tolerance swallows the whole thing (0 >= 1-2 is true),
    # so require at least one join as well.
    chk(matched and len(matched) >= len(turns) - 2,
        f"a record per turn ({len(matched)} of {len(turns)} joined by "
        f"request id){detail}")
    # After the flush nudge is purged the artifact should be exactly the
    # experiment: one record per turn, no strays.
    chk(len(records) <= len(turns),
        f"no records beyond the turns ({len(records)} records for "
        f"{len(turns)} turns)")

    # An independent witness that the join is the right one: the connector
    # counts keys in the cache, the runner counts ids it submitted. If those
    # agree, the match is not coincidental.
    # .get, not [], throughout: a record with fields missing is a record to
    # REPORT, not a reason to abandon the remaining checks. Comparing against
    # a None sentinel makes a malformed record register as a mismatch.
    mism = [t for t in matched
            if (tasks[t["request_id"]].get("used") or {}).get("num_prompt_tokens")
            != t.get("n_prompt_tokens")]
    chk(matched and not mism,
        f"prompt token counts agree between runner and connector "
        f"({len(mism)} disagree"
        + ("; NOTHING JOINED, so this proves nothing" if not matched else "") + ")")

    modes = {(tasks[t["request_id"]].get("attention_stats") or {}).get("segment_mode")
             for t in matched}
    chk(modes == {"variable"},
        f"declared ranges were used on every request (modes={modes or '-'})")

    unhealthy = [t for t in matched
                 if (tasks[t["request_id"]].get("attention_stats") or {}).get(
                     "decode_steps_nonfinite", 0)
                 or (tasks[t["request_id"]].get("attention_stats") or {}).get(
                     "restarts", 0)]
    chk(not unhealthy, f"no non-finite steps or restarts ({len(unhealthy)} flagged)")

    errored = [t for t in matched
               if (tasks[t["request_id"]].get("attention_stats") or {}).get("error")]
    chk(not errored, f"no write errors reported ({len(errored)} flagged)")

    if load_provenance is None:
        checks.append((True, "tensor-level checks skipped (no loader passed)"))
        return checks

    # A sample, not everything: reading every file is I/O bound and the
    # invariants are per-record, so a sample spanning short and long prompts
    # catches a systematic break. Deepest turns first -- they have the most
    # segments and the most history, so they fail first if anything is wrong.
    sample = sorted(matched, key=lambda t: -t["n_prompt_tokens"])[:12]
    seg_ok = mass_ok = manifest_ok = live_ok = sum_ok = 0
    unreadable: list[str] = []
    no_ranges: list[str] = []
    for t in sample:
        rec = tasks[t["request_id"]]
        # A file that is missing, truncated or corrupt must come back as a
        # FAILED CHECK, never an exception -- raising here would abandon every
        # remaining check, and a partial file is exactly what an interrupted
        # transfer leaves behind.
        try:
            arrays = {k: v.tolist()
                      for k, v in _read(load_provenance, rec, run_dir).items()}
        except Exception as exc:                            # noqa: BLE001
            unreadable.append(f"{t['request_id']}: {type(exc).__name__}")
            continue

        declared = {(s["lo"], s["hi"]) for s in t["segments"]}
        emitted = {(lo, hi) for lo, hi, _k in arrays.get("segments", [])}
        # A subset test is VACUOUS on an empty declared set -- set() <= anything
        # is True -- so a turn that declared nothing would be counted as a turn
        # whose ranges survived intact. Those are opposites. Declaring nothing
        # is what a chat template change looks like from here: the tool and
        # answer patterns stop matching, declare_ranges returns [], the
        # connector happily chunks the whole prompt into synthesised gaps, and
        # every structural check downstream still passes.
        if not declared:
            no_ranges.append(str(t["request_id"]))
        else:
            seg_ok += declared <= emitted

        try:
            resid = arrays["topk_residual"][0]
            step0 = resid[0] if isinstance(resid, list) else resid
            mass_ok += abs(sum(arrays["val_all_avg"][0]) + step0 - 1.0) < 1e-3
        except (KeyError, IndexError, TypeError):
            unreadable.append(f"{t['request_id']}: missing mass tensors")

        # IS THERE ANYTHING IN IT? Every other tensor check passes on an
        # all-zero capture: mass conservation reads 0 + 1 == 1, the manifest
        # describes zeros as happily as numbers, and the declared ranges are
        # still present as segments. A probe that never fired therefore
        # produces a record set that looks perfect and means nothing --
        # observed on Qwen3.5, where all 214 records held topk_pos == -1 and
        # val == 0 while fourteen checks passed.
        live_ok += any(p > 0 for row in arrays.get("topk_pos", []) for p in row)

        # attn_sum is the column sum over ALL positions, accumulated across
        # decode steps, so it must total one softmax unit per recorded step.
        # Zeros cannot fake this, which is what makes it the load-bearing
        # check -- it is what the connector's own e2e asserts.
        try:
            steps = len(arrays["topk_pos"])
            total = sum(arrays["attn_sum"])
            sum_ok += steps > 0 and abs(total - steps) < max(0.05 * steps, 0.5)
        except (KeyError, TypeError):
            pass

        manifest = (rec.get("attention_stats") or {}).get("tensors", {})
        manifest_ok += bool(manifest) and all(
            name in arrays and list(spec["shape"]) == _shape(arrays[name])
            for name, spec in manifest.items())

    n = len(sample)
    # `x == n` is trivially true at n == 0. A check that passes because it
    # examined nothing is worse than no check: it reads as evidence.
    if n == 0:
        chk(False, "no records to sample -- the tensor-level checks did not run")
        return checks
    chk(not unreadable,
        f"every sampled tensor file reads back ({len(unreadable)} did not"
        + (f"; first: {unreadable[0]}" if unreadable else "") + ")")
    chk(live_ok == n,
        f"the capture is NON-ZERO ({live_ok}/{n} files hold a retained "
        f"position). Zero means the probe never fired on the backend the "
        f"engine used -- every other check below passes on zeros.")
    chk(sum_ok == n,
        f"attn_sum totals one softmax unit per decode step ({sum_ok}/{n})")
    chk(seg_ok == n and not no_ranges,
        f"declared ranges appear verbatim as segments ({seg_ok}/{n})"
        + (f" -- but {len(no_ranges)} sampled turn(s) declared NO ranges at "
           f"all (first: {no_ranges[0]}). The subset test passes vacuously on "
           f"those, so this is a failure. It means declare_ranges matched "
           f"nothing, which a chat template change causes silently."
           if no_ranges else ""))
    chk(mass_ok == n, f"kept mass + residual == 1 ({mass_ok}/{n})")
    chk(manifest_ok == n, f"manifest matches the file on disk ({manifest_ok}/{n})")
    return checks


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _verdict(t: dict) -> str:
    """This turn's outcome, tolerating runs made before verdicts existed."""
    v = t.get("verdict")
    if v:
        return v
    return "correct" if t.get("ok") else "wrong"


def summarise(turns: list[dict], meta: dict | None = None) -> dict:
    """Accuracy and error structure. Reported, never asserted.

    ACCURACY IS OVER SCORED TURNS ONLY: correct + wrong. A turn the model did
    not answer in the required format is counted and reported as a
    format_error but kept out of the denominator -- putting it in would mix
    "the model was wrong" with "the model did not answer".

    Runs made before the `unscorable` verdict was removed may still carry it;
    those turns stay out of the denominator too, so their old numbers are
    reproduced rather than silently changed. Re-grade them with
    `analyze.py --rescore` to score them under the current rules.
    """
    n = len(turns)
    counts = collections.Counter(_verdict(t) for t in turns)
    # A chain the runner KILLED contributes no turns, so nothing below can
    # notice it: accuracy is computed over whatever survived, and reads as a
    # complete result. A chain dies when its next prompt no longer fits
    # max_model_len, which --history model makes likely the moment replies get
    # long -- and the planner cannot predict it, because it budgets with
    # ground-truth answers. So surface it here, next to the number it
    # invalidates, rather than leaving it in run_meta for someone to find.
    stopped = dict((meta or {}).get("stopped") or {})
    scored = [t for t in turns if _verdict(t) in ("correct", "wrong")]
    ok = sum(1 for t in scored if _verdict(t) == "correct")
    by_chain: dict[str, list[dict]] = {}
    for t in turns:
        by_chain.setdefault(t["chain_id"], []).append(t)

    complete, first_bad_round, after_first = 0, [], [0, 0]
    for ts in by_chain.values():
        ts = sorted(ts, key=lambda x: x["round_no"])
        # "wrong" only. An unscorable or malformed turn is not evidence the
        # chain went wrong, so it neither breaks a clean chain nor starts the
        # after-error window.
        bad = [i for i, t in enumerate(ts) if _verdict(t) == "wrong"]
        # "no wrong answers" is not the same as "got it right". A chain whose
        # every turn was a FORMAT ERROR has no wrong answers either, and the
        # first version of this counted all 50 such chains as fully correct --
        # a 14B run that produced no answers at all scored 50/50 here. Require
        # at least one correct turn and no failures of any kind.
        verdicts = [_verdict(t) for t in ts]
        if not bad and "format_error" not in verdicts and "correct" in verdicts:
            complete += 1
            continue
        if not bad:
            continue        # nothing answered wrongly, but not a clean chain
        fb = bad[0]
        first_bad_round.append(ts[fb]["round_no"])
        rest = [t for t in ts[fb + 1:] if _verdict(t) in ("correct", "wrong")]
        after_first[0] += sum(1 for t in rest if _verdict(t) == "correct")
        after_first[1] += len(rest)

    out_stopped = {}
    if stopped:
        out_stopped = {
            "chains_stopped_early": len(stopped),
            "chains_stopped_early_why": stopped,
            "chains_stopped_early_note":
                "these chains ran out of context and contributed no turns "
                "after the round named. Accuracy below is over the turns that "
                "RAN, so it describes a smaller experiment than the one "
                "requested -- raise --max-model-len or lower --max-tokens.",
        }

    return {
        "turns": n,
        "verdicts": dict(counts),
        **out_stopped,
        "scored": len(scored),
        "correct": ok,
        "accuracy": round(ok / len(scored), 4) if scored else 0.0,
        "accuracy_denominator": "scored turns (excludes unscorable + format_error)",
        "format_error_rate": round(counts.get("format_error", 0) / n, 4) if n else 0.0,
        # Printed next to accuracy ON PURPOSE. Truncated turns leave the
        # denominator, so a run that cuts replies off reports a HIGHER
        # accuracy over fewer turns -- the same shape as the `unscorable`
        # verdict that was removed from this file. The rate is the thing that
        # says whether the accuracy above describes the experiment or a
        # surviving fraction of it.
        "truncation_rate": round(counts.get("truncated", 0) / n, 4) if n else 0.0,
        "chains": len(by_chain),
        "chains_fully_correct": complete,
        "first_error_round": dict(sorted(collections.Counter(first_bad_round).items())),
        "accuracy_after_first_error": (
            round(after_first[0] / after_first[1], 4) if after_first[1] else None),
        "by_tier": _group_acc(turns, "tier"),
        # CONFOUNDED with task type, and not readable as a depth effect: in
        # opal_chains.csv round N is almost always the same task for every
        # chain (1 dose_response_shape, 2 transition_sensitivity, 3
        # cross_modality_peak, 4 modality_coverage, 6 variance_regime, 7
        # literature_audit; only round 5 is mixed). A drop at round 4 therefore
        # says modality_coverage is hard, NOT that four rounds of history hurt.
        # Use `propagation` below for the depth question.
        "by_round": _group_acc(turns, "round_no"),
        "by_round_caveat": ("round is confounded with task type in this "
                            "benchmark; see `propagation` for the depth effect"),
        "by_task": _group_acc(turns, "task_type"),
        "propagation": propagation(turns),
        "documented_shortcuts": dict(collections.Counter(
            t["shortcut"] for t in turns if t.get("shortcut")).most_common()),
        "wrong_unexplained": sum(1 for t in turns
                                 if _verdict(t) == "wrong" and not t.get("shortcut")),
    }


def propagation(turns: list[dict]) -> dict:
    """Does an earlier mistake make later rounds worse, holding task fixed?

    This is the question the chains exist to ask, and neither `by_round` nor
    `accuracy_after_first_error` answers it: both compare different TASKS,
    because round and task are locked together in this benchmark. Later rounds
    are also harder rounds, so a decline proves nothing on its own.

    Within one task type that confound is gone -- every turn being compared is
    the same question -- so the contrast between "the chain was clean until
    now" and "the chain already went wrong" isolates the effect of carrying a
    bad answer in the context.

    Reported per task and as a total weighted across them. `delta` is
    after-error minus clean, so a negative number is evidence of propagation.
    """
    by_chain: dict[str, list[dict]] = {}
    for t in turns:
        by_chain.setdefault(t["chain_id"], []).append(t)

    acc: dict[str, dict[str, list[int]]] = {}
    for ts in by_chain.values():
        erred = False
        for t in sorted(ts, key=lambda x: x["round_no"]):
            v = _verdict(t)
            if v in ("correct", "wrong"):
                b = acc.setdefault(t.get("task_type", "unknown"),
                                   {"clean": [0, 0], "after_error": [0, 0]})
                slot = b["after_error" if erred else "clean"]
                slot[0] += v == "correct"
                slot[1] += 1
            if v == "wrong":
                erred = True

    out, tot = {}, {"clean": [0, 0], "after_error": [0, 0]}
    for task, b in sorted(acc.items()):
        row = {}
        for k in ("clean", "after_error"):
            c, n = b[k]
            tot[k][0] += c
            tot[k][1] += n
            row[k] = {"correct": c, "n": n,
                      "acc": round(c / n, 4) if n else None}
        if row["clean"]["acc"] is not None and row["after_error"]["acc"] is not None:
            row["delta"] = round(row["after_error"]["acc"] - row["clean"]["acc"], 4)
        out[task] = row

    overall = {}
    for k in ("clean", "after_error"):
        c, n = tot[k]
        overall[k] = {"correct": c, "n": n, "acc": round(c / n, 4) if n else None}
    if overall["clean"]["acc"] is not None and overall["after_error"]["acc"] is not None:
        overall["delta"] = round(overall["after_error"]["acc"]
                                 - overall["clean"]["acc"], 4)
    out["ALL"] = overall
    return out


def _group_acc(turns: list[dict], key: str) -> dict:
    """Accuracy grouped by one field, over SCORED turns only.

    `n` is how many turns in the group could be graded; `skipped` is how many
    could not. A group that is entirely unscorable reports acc=None rather
    than 0.0, which would read as "the model got them all wrong".

    Tolerates records lacking the field (older runs) by bucketing them under
    "unknown" instead of aborting the report.
    """
    g: dict = {}
    for t in turns:
        a = g.setdefault(t.get(key, "unknown"), [0, 0, 0])
        v = _verdict(t)
        if v in ("correct", "wrong"):
            a[0] += v == "correct"
            a[1] += 1
        else:
            a[2] += 1
    out = {}
    for k, (c, n, skipped) in sorted(g.items(), key=lambda kv: str(kv[0])):
        row = {"correct": c, "n": n,
               "acc": round(c / n, 4) if n else None}
        if skipped:
            row["skipped"] = skipped
        out[str(k)] = row
    return out


def hottest_segments(turns: list[dict], buffer: list[dict], load_provenance,
                     limit: int = 0, run_dir=None) -> dict:
    """Which KIND of segment the model attended to hardest, per turn.

    Peak `val_all_max` within each segment at the final content decode step --
    the probe that separated evidence from noise best on the connector's own
    5-round example. Aggregated by segment kind, and by whether the span came
    from the CURRENT round or from the conversation history, which is the
    question these chains exist to ask.
    """
    tasks = index_tasks(buffer)
    hot_kind: collections.Counter = collections.Counter()
    hot_origin: collections.Counter = collections.Counter()
    matched = [t for t in turns if t.get("request_id") in tasks]
    if limit:
        matched = matched[:limit]

    for t in matched:
        arrays = _read(load_provenance, tasks[t["request_id"]], run_dir)
        pos = arrays["topk_pos"].tolist()
        val = arrays["val_all_max"].tolist()
        segs = [(lo, hi) for lo, hi, _k in arrays["segments"].tolist()]
        if not pos:
            continue
        step = max(0, len(pos) - 2)     # last content step, before any EOS row
        peak: dict[int, float] = {}
        for p, v in zip(pos[step], val[step], strict=True):
            if p <= 0:                  # position 0 is the sink; never selected
                continue
            for si, (lo, hi) in enumerate(segs):
                if lo <= p < hi:
                    peak[si] = max(peak.get(si, 0.0), v)
                    break
        if not peak:
            continue
        lo, hi = segs[max(peak, key=peak.get)]
        declared = next((s for s in t["segments"]
                         if (s["lo"], s["hi"]) == (lo, hi)), None)
        hot_kind[declared["kind"] if declared else "prose gap"] += 1
        if declared:
            hot_origin["current round" if declared["round_hint"] >= t["round_no"]
                       else "history"] += 1
        else:
            hot_origin["prose gap"] += 1

    return {"turns_examined": len(matched),
            "hottest_by_kind": dict(hot_kind.most_common()),
            "hottest_by_origin": dict(hot_origin.most_common())}


# --------------------------------------------------------------------------
# per-answer-item heat, over lines
# --------------------------------------------------------------------------
#
# `hottest_segments` above answers "which segment kind won, once per turn".
# That is a census, and it is blunt in three ways that matter for reading a
# single turn:
#
#   it samples ONE decode step (`len(pos) - 2`), so on a two-part answer it
#   only ever sees the last part;
#   it reduces to a single argmax, so there is nothing to draw;
#   it works in SEGMENTS, and a tool block is twenty lines of table.
#
# What follows is the other thing: for ONE turn, heat per line per answer
# item. An OPAL answer like "<accession>; <MATCHES|DIVERGES>" is two questions
# answered by two disjoint runs of decode steps, and they consult different
# evidence -- the accession comes from the history, the verdict from the
# current round. Aggregating them together averages the distinction away.

def load_prompts(path: str | Path) -> dict[str, dict]:
    """`prompts.jsonl` keyed by every alias of its request id.

    Written by the runner because a line table cannot be safely recomputed
    later: it needs the tokenizer AND the chat template exactly as they were,
    and a template that has drifted since would re-derive different spans for
    the same captured tensors without failing.
    """
    out: dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for alias in _id_aliases(rec.get("request_id") or ""):
            if alias:
                out.setdefault(alias, rec)
    return out


def answer_items(turn: dict) -> list[dict]:
    """The parts of a reply's ANSWER: line, each with the steps that wrote it.

    `[{"label": "NIL 373", "steps": [3, 4, 5, 6, 7]}, ...]`.

    Split on ";" because that is the separator every multi-part OUTPUT FORMAT
    in this benchmark specifies ("<accession label>; <MATCHES or DIVERGES>").
    A single-part answer yields one item, so callers need no special case.

    Returns [] when the turn predates `gen_char_ends` or produced no ANSWER:
    line -- there is then no honest step-to-text mapping, and inventing one by
    assuming a fixed prefix length would silently attribute the wrong steps.
    """
    text = turn.get("gen_text") or ""
    ends = turn.get("gen_char_ends") or []
    if not text or not ends:
        return []

    hits = list(re.finditer(r"ANSWER\s*:\s*(.+?)\s*$", text,
                            re.IGNORECASE | re.MULTILINE))
    if not hits:
        return []
    payload = hits[-1]                       # the last one, as parse_answer does
    base, body = payload.start(1), payload.group(1)

    spans, cursor = [], 0
    for part in body.split(";"):
        c0 = base + cursor
        cursor += len(part) + 1              # the ";" itself
        stripped = part.strip()
        if not stripped:
            continue
        lead = len(part) - len(part.lstrip())
        spans.append((stripped, c0 + lead, c0 + lead + len(stripped)))

    items = []
    for label, c0, c1 in spans:
        steps = [i for i, end in enumerate(ends)
                 if end > c0 and (ends[i - 1] if i else 0) < c1]
        if steps:
            items.append({"label": label, "steps": steps})
    return items


def _line_index(line_los: list[int], pos: int) -> int | None:
    """Which line a token position falls on. Lines tile [0, n_tokens)."""
    i = bisect.bisect_right(line_los, pos) - 1
    return i if i >= 0 else None


def heat_lines(turn: dict, rec: dict, load_provenance, prompt: dict, *,
               run_dir=None, decay: float = 0.5) -> dict:
    """Per-line attention heat for one turn, one column per answer item.

    For each item, over its decode steps in order (k = 0, 1, 2, ...):

        heat[line] = max over steps of  decay**k * max(val_all_max on that line)

    Max rather than sum, and decayed rather than flat, because the objective
    is NOT to estimate how much total attention a line received -- it is to
    rank lines so that noise does not outrank evidence. A sum rewards long
    lines and rewards lines that are merely adjacent to many retained
    positions; an undecayed max lets the tail of an item, which is mostly
    committed by then, outvote the step that actually chose the answer.

    Position 0 is skipped: it is the attention sink, selected on every step of
    every request, and it is not evidence of anything.

    Returns raw values. Scaling to a ramp belongs to whatever draws it, and
    each column scales to its OWN max -- the two items of one answer are not
    on a comparable scale and pretending otherwise makes the weaker one vanish.
    """
    arrays = _read(load_provenance, rec, run_dir)
    pos = arrays["topk_pos"].tolist()
    val = arrays["val_all_max"].tolist()
    lines = prompt.get("lines") or []
    covered = [(i, ln) for i, ln in enumerate(lines) if ln[1] > ln[0]]
    line_los = [ln[0] for _i, ln in covered]

    items = answer_items(turn)
    if not items:
        # No step mapping available. One column over every step that retained
        # anything is still drawable, and is honest about what it is.
        items = [{"label": "(whole reply)", "steps": list(range(len(pos)))}]

    out_items = []
    for item in items:
        heat: dict[int, float] = {}
        used = []
        for k, step in enumerate(s for s in item["steps"] if s < len(pos)):
            w = decay ** k
            used.append(step)
            for p, v in zip(pos[step], val[step], strict=True):
                if p <= 0:
                    continue
                j = _line_index(line_los, p)
                if j is None:
                    continue
                li = covered[j][0]
                heat[li] = max(heat.get(li, 0.0), w * float(v))
        out_items.append({"label": item["label"], "steps": used, "heat": heat})

    return {"request_id": turn.get("request_id"),
            "chain_id": turn.get("chain_id"),
            "round_no": turn.get("round_no"),
            "task_type": turn.get("task_type"),
            "n_steps": len(pos),
            "n_lines": len(lines),
            "decay": decay,
            "items": out_items}


def density_by_distance(turns: list[dict], buffer: list[dict], load_provenance,
                        prompts: dict, *, run_dir=None,
                        bands=((0, 400), (400, 900), (900, 1600), (1600, 10**9))
                        ) -> dict:
    """Attention DENSITY per span kind, by distance from the decode point.

    Three normalisations, each of which replaced a wrong answer:

    MASS FROM `attn_sum`, not from the per-step top_pct arrays. attn_sum is
    accumulated before selection, so it is unbiased and can be summed over any
    span -- including spans the run never declared. Selection inflates
    declared segments, because every segment holds a guaranteed quota.

    RENORMALISED OVER CONTENT. Position 0, blank lines and chat markers carry
    no information and take 55% of the mass on Qwen3-4B against 15% on
    Qwen3.5-35B-A3B. A share-of-total comparison between two models is
    therefore largely a comparison of their sink behaviour.

    PER TOKEN, AND AT MATCHED DISTANCE. Without dividing by span length, this
    ranks spans by size. Without banding by distance, it ranks them by how
    close they usually sit: density falls 10-18x from the nearest instance of
    a kind to the farthest, which swamps every between-kind difference.

    Spans come from the declared segments AND from `sections_from_lines`, both
    per instance -- a round-7 turn has seven OUTPUT FORMAT blocks at different
    distances and they are seven observations, not one.
    """
    import statistics

    from .ranges import sections_from_lines

    tasks = index_tasks(buffer)
    pts: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    struct: collections.Counter = collections.Counter()
    n = 0

    for t in turns:
        rec = tasks.get(t.get("request_id"))
        pr = prompts.get(t.get("request_id"))
        if rec is None or pr is None:
            continue
        try:
            arrays = _read(load_provenance, rec, run_dir)
        except Exception:                                   # noqa: BLE001
            continue
        if "attn_sum" not in arrays:
            continue
        s = arrays["attn_sum"].tolist()
        total = sum(s)
        if total <= 0:
            continue
        lines = pr.get("lines") or []
        n_tok = len(s)

        content = 0.0
        for lo, hi, _r, text in lines:
            if hi <= lo:
                continue
            m = sum(s[max(lo, 1):hi])
            if not text.strip():
                struct["blank lines"] += m / total
            elif text.startswith("<|im_"):
                struct["chat markers"] += m / total
            else:
                struct["content lines"] += m / total
                content += m
        struct["position 0 (sink)"] += s[0] / total
        if content <= 0:
            continue
        n += 1

        spans = [(g["lo"], g["hi"], g["kind"]) for g in t.get("segments", [])]
        spans += [(x.lo, x.hi, x.kind) for x in sections_from_lines(lines)]
        for lo, hi, kind in spans:
            lo = max(lo, 1)
            if hi > lo:
                pts[kind].append((n_tok - hi,
                                  sum(s[lo:hi]) / content / (hi - lo) * 1000))

    out: dict = {"turns": n,
                 "structural_share": {k: round(v / n, 4) for k, v in struct.items()} if n else {},
                 "bands": [list(b) for b in bands], "density": {}}
    for kind, v in pts.items():
        row = []
        for lo, hi in bands:
            g = [d for x, d in v if lo <= x < hi]
            row.append(round(statistics.median(g), 4) if len(g) >= 5 else None)
        out["density"][kind] = {"by_band": row, "instances": len(v)}
    return out


# --------------------------------------------------------------------------
# comparing two runs over EQUIVALENT areas
# --------------------------------------------------------------------------
#
# Token index is not a valid key across runs. Token-length parity holds for
# the injected span, but `--history model` puts the model's OWN replies into
# later prompts -- and once a poison flips an answer those replies differ in
# length, so every index after them shifts. Measured on one chain: rounds 1-3
# stayed aligned by luck, round 4 came out 2652 vs 2642 tokens.
#
# Line index is not valid either. It happened to hold here (196 lines both
# sides) because the divergent answers were one line each, but an answer that
# wraps differently, or a `<think>` block, changes the line count and silently
# shifts every row after it.
#
# So correspondence is ESTABLISHED, not assumed: align the two line sequences
# by their text, and compare only the pairs the alignment matched. Lines with
# no counterpart are reported, never compared.

def align_prompts(lines_a: list, lines_b: list) -> dict:
    """Correspondence between two runs' line tables, from their text.

    Returns `{"equal": [(i, j)...], "changed": [(i, j)...], "only_a": [...],
    "only_b": [...]}`.

    `equal`   -- same text. Comparing attention here is apples to apples, and
                 this is where a redistribution effect must show up to count.
    `changed` -- corresponding position, different text: the poisoned line
                 itself, and any assistant turn whose answer diverged. Worth
                 measuring, but it is a different line, so it is kept apart
                 from `equal` rather than averaged in with it.
    `only_*`  -- no counterpart at all. Never compared.

    Uses difflib on the line TEXT, so it survives insertions and deletions,
    not just equal-length substitutions -- which is the case index arithmetic
    cannot handle and the one that actually arises.
    """
    import difflib

    ta = [ln[3] for ln in lines_a]
    tb = [ln[3] for ln in lines_b]
    sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    out: dict[str, list] = {"equal": [], "changed": [], "only_a": [], "only_b": []}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out["equal"] += [(i, j) for i, j in zip(range(i1, i2),
                                                    range(j1, j2), strict=True)]
        elif tag == "replace":
            # Pair them off positionally WITHIN the replace block; any excess
            # on either side has no counterpart.
            n = min(i2 - i1, j2 - j1)
            out["changed"] += [(i1 + k, j1 + k) for k in range(n)]
            out["only_a"] += list(range(i1 + n, i2))
            out["only_b"] += list(range(j1 + n, j2))
        elif tag == "delete":
            out["only_a"] += list(range(i1, i2))
        elif tag == "insert":
            out["only_b"] += list(range(j1, j2))
    return out


def line_density(attn_sum: list, lines: list) -> dict:
    """`{line index: attention per token}`, content-renormalised.

    Content excludes position 0 (the sink), blank lines and chat markers --
    together 15% of the mass on Qwen3.5-35B-A3B and 55% on Qwen3-4B, so a raw
    share compares sink behaviour as much as attention.
    """
    content = sum(sum(attn_sum[max(lo, 1):hi]) for lo, hi, _r, tx in lines
                  if hi > lo and tx.strip() and not tx.startswith("<|im_"))
    if content <= 0:
        return {}
    out = {}
    for i, (lo, hi, _r, tx) in enumerate(lines):
        lo = max(lo, 1)
        if hi > lo and tx.strip() and not tx.startswith("<|im_"):
            out[i] = sum(attn_sum[lo:hi]) / content / (hi - lo) * 1000
    return out


def compare_turn(attn_a: list, lines_a: list, attn_b: list, lines_b: list
                 ) -> dict:
    """Paired per-line attention between two runs, over equivalent areas only.

    `moved` is the total absolute change across lines that are TEXTUALLY
    IDENTICAL in both runs. That is the redistribution the edit caused
    elsewhere in the prompt -- it cannot be contaminated by the edit itself,
    because the edited line is not in that set.
    """
    da, db = line_density(attn_a, lines_a), line_density(attn_b, lines_b)
    al = align_prompts(lines_a, lines_b)
    eq = [(i, j) for i, j in al["equal"] if i in da and j in db]
    ch = [(i, j) for i, j in al["changed"] if i in da and j in db]
    return {
        "n_equal": len(eq), "n_changed": len(ch),
        "n_only_a": len(al["only_a"]), "n_only_b": len(al["only_b"]),
        "moved": sum(abs(db[j] - da[i]) for i, j in eq),
        "equal_pairs": [(i, j, da[i], db[j]) for i, j in eq],
        "changed_pairs": [(i, j, da[i], db[j]) for i, j in ch],
    }
