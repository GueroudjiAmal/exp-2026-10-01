#!/usr/bin/env python
"""Figures for the poisoning suite, as standalone SVG. No plotting library.

    python scripts/poison_figures.py results/poison-suite-<stamp>-report.json
    python scripts/poison_figures.py <report.json> --heat qc:exp109-chain-0011-hard:1

Writes into results/figures/. SVG because it is vector (drops into a paper or
scales on a slide), needs no matplotlib -- which this machine does not have and
cannot install offline -- and is a text format, so a figure diffs in git.

THREE FIGURES, each answering one question:

  heat-matrix-*.svg        One turn, every run side by side. Rows are prompt
                           lines, columns runs, one colour scale throughout.

  outcome-vs-attention.svg Per turn: decode steps against attention change,
                           split by outcome. This is the one that matters --
                           arm-level medians hide that the turns which
                           actually flipped show almost no attention change,
                           while the resisted and truncated ones show a lot.

Two earlier figures were removed rather than fixed. `effect-vs-detection`
plotted flip rate against an arm-level median delta, and the outcome split
showed that median is driven by the turns the poison did NOT work on.
`span-shift` plotted the same median with its spread, so it inherited the
same defect.

Colour follows the data's job, per the project's viz rules: two conditions are
categorical identity (validated slots 1 and 2, blue/orange); magnitude is one
sequential hue; a signed difference is diverging blue-gray-red. Text never
wears a series colour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
CLEAN, POIS = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
NEUTRAL = "#f0efec"
DIV_HI = ["#f0efec", "#f6c9c9", "#ec9a9a", "#e34948", "#d03b3b"]   # red arm
DIV_LO = ["#f0efec", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6"]   # blue arm
FONT = ("font-family=\"Inter,-apple-system,Segoe UI,Helvetica,Arial,sans-serif\" "
        "font-variant-numeric=\"tabular-nums\"")


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">'
            f'<rect width="{w}" height="{h}" fill="{SURFACE}"/>{body}</svg>')


def txt(x, y, s, size=12, fill=INK, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" {FONT} font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
            f'{esc(s)}</text>')


def ramp(v, vmax, steps):
    """Sequential step for a magnitude; lightest means near zero."""
    if vmax <= 0 or v <= 0:
        return steps[0]
    i = min(len(steps) - 1, max(0, round((len(steps) - 1) * (v / vmax) ** 0.6)))
    return steps[i]


def diverging(v, vmax):
    if vmax <= 0 or abs(v) < vmax * 0.02:
        return NEUTRAL
    arm = DIV_HI if v > 0 else DIV_LO
    i = min(len(arm) - 1, max(1, round((len(arm) - 1) * (abs(v) / vmax) ** 0.6)))
    return arm[i]



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, help="…-report.json from poison_report.py")
    ap.add_argument("--matrix", default=None, metavar="CHAIN:ROUND",
                    help="draw ONE turn across every run, side by side")
    ap.add_argument("--out", default=str(HERE / "results" / "figures"))
    args = ap.parse_args()

    rep = json.loads(args.report.read_text())
    arms = rep["arms"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []

    # Always drawn: it is the figure the arm-level summary gets wrong.
    f = fig_outcome_scatter(arms, out / "outcome-vs-attention.svg")
    if f:
        made.append(f)

    if args.matrix:
        from opal_attn import report as rp
        from opal_attn.provenance import load_provenance
        chain, rnd = args.matrix.split(":")
        rnd = int(rnd)
        suite = json.loads(Path(str(args.report).replace("-report", "")).read_text())
        order = ["clean"] + [a["arm"] for a in arms]
        heats, marks, lines_ref = {}, {}, None
        for arm in order:
            run = HERE / "results" / suite[arm]
            meta = json.loads((run / "run_meta.json").read_text())
            turns = {(t["chain_id"], t["round_no"]): t
                     for t in rp.load_turns(run / "turns.jsonl")}
            prompts = rp.load_prompts(run / "prompts.jsonl")
            buf, _ = rp.find_run_buffer(run, meta.get("workflow_id"))
            tasks = rp.index_tasks(rp.load_buffer(buf))
            t = turns.get((chain, rnd))
            if t is None:
                continue
            pr = prompts[t["request_id"]]
            arr = rp._read(load_provenance, tasks[t["request_id"]], run)
            s_ = arr["attn_sum"].tolist()
            lines = pr["lines"]
            cm = _content_mass(s_, lines)
            heats[arm] = {i: sum(s_[max(lo, 1):hi]) / cm / (hi - max(lo, 1)) * 1000
                          for i, (lo, hi, _r, tx) in enumerate(lines)
                          if hi > lo and tx.strip() and not tx.startswith("<|im_")}
            lines_ref = lines
            if arm != "clean":
                spec = json.loads((HERE / "configs" / "poison" /
                                   f"{arm}.json").read_text())
                ent = spec["chains"].get(chain)
                if ent:
                    key = ent["replace"].split("\n")[0][:26]
                    marks[arm] = {i for i, ln in enumerate(lines) if key in ln[3]}
        order = [a for a in order if a in heats]
        f = fig_heat_matrix(lines_ref, heats, order, marks,
                            f"{chain} · round {rnd} · "
                            f"{len(order)} runs, identical prompts except the poison",
                            out / f"heat-matrix-{chain}-r{rnd}.svg")
        if f:
            made.append(f)

    for m in made:
        print(f"wrote {m}  ({m.stat().st_size} B)")
    return 0


def _content_mass(s, lines):
    return sum(sum(s[max(lo, 1):hi]) for lo, hi, _r, tx in lines
               if hi > lo and tx.strip() and not tx.startswith("<|im_"))




def fig_heat_matrix(lines, heats, order, marks, title, out: Path,
                    rows_shown=48):
    """One turn, every run side by side. Rows are prompt lines, columns runs.

    This is the figure the pairwise diffs were trying to be. A reader wants to
    scan ONE row across all seven arms and see where it lights up; two-column
    before/after panels make that a memory exercise.

    ONE COLOUR SCALE FOR THE WHOLE GRID. Per-column scaling would make every
    arm look equally hot -- each column normalised to its own max -- which is
    exactly the comparison the figure exists to prevent. vmax is global and
    stated in the caption.

    Rows are the union of the hottest lines across arms and every arm's own
    poisoned line, so a poison that attracted nothing is still visible as a
    cold cell rather than being filtered out for being cold.
    """
    hottest = sorted(range(len(lines)),
                     key=lambda i: -max(h.get(i, 0.0) for h in heats.values()))
    keep = sorted(set(hottest[:rows_shown]) | {i for v in marks.values() for i in v})
    cw, rh, T = 46, 14, 104
    labw = int(max((len(lines[i][3][:44]) for i in keep), default=20) * 5.7)
    L = 26
    W = L + cw * len(order) + 14 + labw + 20
    H = T + rh * len(keep) + 64
    vmax = max([h.get(i, 0.0) for h in heats.values() for i in keep] + [1e-9])

    body = [txt(L, 30, "Attention per prompt line, every run side by side",
                15, INK, weight="600"),
            txt(L, 48, title, 11, INK2),
            txt(L, 66, f"one colour scale across the whole grid (0 – {vmax:.2f}) · "
                       f"● marks that run's poisoned line", 11, INK2)]
    for j, arm in enumerate(order):
        x = L + cw * j + cw / 2
        body.append(txt(x, T - 10, arm, 10, INK, "middle",
                        weight="600" if arm != "clean" else "400"))
    for r, i in enumerate(keep):
        y = T + r * rh
        for j, arm in enumerate(order):
            v = heats[arm].get(i, 0.0)
            body.append(f'<rect x="{L+cw*j+1}" y="{y+1}" width="{cw-2}" '
                        f'height="{rh-2}" fill="{ramp(v, vmax, SEQ)}"/>')
            if i in marks.get(arm, ()):
                # ring in the surface colour so it reads on any ramp step
                body.append(f'<circle cx="{L+cw*j+cw/2:.1f}" cy="{y+rh/2:.1f}" '
                            f'r="3.2" fill="none" stroke="{SURFACE}" '
                            f'stroke-width="2"/>')
        marked = any(i in v for v in marks.values())
        body.append(txt(L + cw * len(order) + 12, y + rh - 4,
                        lines[i][3][:44], 10, INK if marked else INK2,
                        weight="600" if marked else "400"))
    # scale strip
    sy = H - 36
    for k, col in enumerate(SEQ):
        body.append(f'<rect x="{L+k*22}" y="{sy}" width="22" height="10" fill="{col}"/>')
    body.append(txt(L, sy + 26, "low", 10, INK2))
    body.append(txt(L + len(SEQ) * 22, sy + 26, "high", 10, INK2, "end"))
    body.append(txt(L + len(SEQ) * 22 + 16, sy + 9,
                    "attention density per token, content-normalised", 10, INK2))
    out.write_text(svg(W, H, "".join(body), "Attention per line across runs"))
    return out


OUT_COL = {"flipped": "#2a78d6", "resisted": "#eb6834", "truncated": "#1baf7a"}


def fig_outcome_scatter(arms, out: Path):
    """Decode steps against attention change, one point per turn, by outcome.

    THIS IS THE FIGURE THAT CORRECTS THE SUMMARY TABLE. Arm-level medians say
    qc moved attention +29% and flipped 82% of answers, which reads as "the
    signal tracks the damage". Split by outcome it is the reverse: on qc the
    turns that actually FLIPPED show +8.6%, barely above the null control's
    +6.5%, while the turns the model RESISTED show +52% and the ones it
    deliberated to exhaustion show +31%.

    So the attention signature is not a measure of a poison succeeding. It is
    a measure of the model NOTICING -- re-reading the contradiction, spending
    tokens on it, and often recovering. A poison that is simply swallowed is
    read once and leaves almost nothing behind.

    Three outcomes are three identities, so categorical colour, slots 1-3
    (documented as validating all-pairs, which a scatter needs). A legend
    carries identity; the points carry no labels because there are hundreds.
    """
    import statistics as _st
    rows = [a for a in arms if a.get("paired")]
    if not rows:
        return None
    nc = 3
    nr = (len(rows) + nc - 1) // nc
    pw, ph, gx, gy = 196, 150, 34, 56
    L, T = 56, 96
    W = L + nc * (pw + gx) + 16
    H = T + nr * (ph + gy) + 30
    pts = [(x, a) for a in rows for x in a["paired"] if x.get("clean", 0) > 0]
    dys = [100 * (x["pois"] - x["clean"]) / x["clean"] for x, _a in pts]
    lo, hi = min(dys + [0]), max(dys + [0])
    lo, hi = min(lo, -20), max(hi, 20)
    xmax = max((x["steps"] for x, _a in pts), default=8192) or 8192

    body = [txt(L, 32, "A poison that works leaves almost no trace; "
                       "one the model fights leaves a large one", 15, INK,
                weight="600"),
            txt(L, 50, "every turn · x = decode steps the model spent · "
                       "y = change in attention on the poisoned span", 11, INK2)]
    for i, (lab, col) in enumerate(OUT_COL.items()):
        x = L + i * 118
        body.append(f'<circle cx="{x}" cy="66" r="5" fill="{col}"/>')
        body.append(txt(x + 10, 70, lab, 11, INK))

    for idx, a in enumerate(rows):
        r, c = divmod(idx, nc)
        ox, oy = L + c * (pw + gx), T + r * (ph + gy)
        y0 = oy + ph - ph * (0 - lo) / (hi - lo)
        body.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" '
                    f'fill="none" stroke="{GRID}" stroke-width="1"/>')
        body.append(f'<line x1="{ox}" y1="{y0:.1f}" x2="{ox+pw}" y2="{y0:.1f}" '
                    f'stroke="{INK2}" stroke-width="1" stroke-dasharray="3 3"/>')
        groups = {"flipped": [], "resisted": [], "truncated": []}
        for x in a["paired"]:
            if x.get("clean", 0) <= 0:
                continue
            k = ("truncated" if x["truncated"]
                 else "flipped" if x["flipped"] else "resisted")
            groups[k].append(x)
        for k, col in OUT_COL.items():
            for x in groups[k]:
                dy = 100 * (x["pois"] - x["clean"]) / x["clean"]
                px = ox + pw * min(x["steps"], xmax) / xmax
                py = oy + ph - ph * (dy - lo) / (hi - lo)
                body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" '
                            f'fill="{col}" opacity="0.55"/>')
        # median marker per outcome: the point of the figure is the contrast
        for k, col in OUT_COL.items():
            g = [100 * (x["pois"] - x["clean"]) / x["clean"] for x in groups[k]]
            if len(g) < 3:
                continue
            m = _st.median(g)
            py = oy + ph - ph * (m - lo) / (hi - lo)
            body.append(f'<line x1="{ox+pw-20}" y1="{py:.1f}" x2="{ox+pw}" '
                        f'y2="{py:.1f}" stroke="{col}" stroke-width="3"/>')
        n_f = len(groups["flipped"])
        body.append(txt(ox, oy - 8, f"{a['arm']}   {n_f} flipped of "
                                    f"{len(a['paired'])}", 12, INK, weight="600"))
        if c == 0:
            for g in (lo, 0, hi):
                yy = oy + ph - ph * (g - lo) / (hi - lo)
                body.append(txt(ox - 8, yy + 4, f"{g:+.0f}%", 10, INK2, "end"))
        body.append(txt(ox + pw / 2, oy + ph + 16,
                        f"0 – {int(xmax)} decode steps", 10, INK2, "middle"))
    out.write_text(svg(W, H, "".join(body), "Outcome against attention change"))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
