"""E2 (tweet urn) figures — deliverables (a)-(e) plus the prominence panel.

House style copied from e1_figures.py so both tabs of the shared artifact read as one
document: greyscale ramp, white mark edges, dpi 120, small legends.

Deliverables, as agreed at the start of the session (docs/tweet_urn_plan.md §4):
  a  funnel per bin, every denominator kept
  b  evidence-type (flag) mix per bin, per claim AND per post
  c  score distribution per bin, E1 weights CITED not re-fitted
  d  evidence quality of the RETRIEVED documents per bin
  e  threshold sweep: nudge rate per cutoff, per bin, per claim AND per post
  f  claim prominence — the largest single driver, and why the deck must show it

Post-level figures use COMPLETE posts only and worst-case aggregation (any nudging
claim nudges the post), per Daniel 2026-08-05: per-post is the headline unit.
Intervals are 95% Wilson on POST counts, never on document counts — using documents
as the denominator is what produced a phantom gradient in the 7% tranche.

  uv run python -m eval.scripts.build_eval.e2_figures
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

RES = Path("eval/data/urn_runs/e2_tweets/results-00.jsonl")
PROM = Path("eval/data/survey_claims/e2_claim_prominence.parquet")
DRAW = Path("eval/data/survey_claims/e2_draw_screen.parquet")
OUT = Path("eval/data/urn_runs/e2_tweets/figures")

BINS = ["0-30", "30-50", "50-70", "70-90", "90-100"]
FLAGS = ("5", "4", "3", "X", "I", "2", "1")
FLAG_GREY = ["0.10", "0.30", "0.50", "0.62", "0.74", "0.86", "0.95"]
GREY = ["0.15", "0.4", "0.6", "0.8"]
DPI = 150
# Rendered inside an artifact at ~860px, so type must survive downscaling. Titles are
# NOT drawn in matplotlib: the page supplies headings, and reserving axes space for a
# title is what collided with legends and annotations in the first cut.
plt.rcParams.update({
    "font.size": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.edgecolor": "0.35", "axes.linewidth": 0.8,
    "xtick.color": "0.3", "ytick.color": "0.3", "figure.constrained_layout.use": True,
})
# READ from the E1 metrics file, never pasted: this block was the fourth copy of
# these constants in the repo and they had already drifted apart once.
# 2026-09-14: headline_metrics.json is now the SIX-FLAG fit (read-v6.1: no "3" class,
# so a read-v5 "3" folds into "X"), keyed by flag; the three-voice collapse that used
# to happen here is gone.
# Its other convention comes with it -- the fit pads every claim to PAD_TO slots
# with silent documents, so scoring here pads too; the weights are only valid
# under the convention they were estimated with.
_E1M = W = E1T = None   # loaded by _load_e1() from main(); never read at import time
PAD_TO = 10


def _load_e1() -> None:
    global _E1M, W, E1T
    _E1M = json.loads(Path("eval/data/urn_runs/e1_ctx/headline_metrics.json").read_text())["overall"]
    W = dict(_E1M["weights"])
    E1T = _E1M["threshold"]


def _score(rec) -> float:
    """Sum of per-flag weights, padded to PAD_TO slots with silent ("I") reads.

    The shipped urn is SIX-flag (read-v6.1, 2026-09-14): a read-v5 "3"
    (contested/mixed) document folds into "X" (on-claim context)."""
    n = 0
    s = 0.0
    for e in rec["results"]:
        d = e["read"]["direction"]
        w = W.get("X" if d == "3" else d)
        if w is not None:
            s += w
            n += 1
    return s + max(0, PAD_TO - n) * W["I"]


# muted slate ramp, dark = least reliable bin; clay is reserved for refute/false
RAMP = ["#24404f", "#48697e", "#7095a8", "#a3bfcc", "#cfdfe6"]
LOCAL = {"gpb.org", "nydailynews.com", "nypost.com",
         "floridadaily.com", "heartlandsignal.com"}


def wilson(k, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load():
    recs = [json.loads(l) for l in open(RES)]
    prom = {r["claim_id"]: r for r in pl.read_parquet(PROM).iter_rows(named=True)}
    drawn = collections.Counter()
    for r in pl.read_parquet(DRAW).iter_rows(named=True):
        drawn[r["post_id"]] += 1
    seen = collections.Counter(r["post_id"] for r in recs)
    complete = {p for p, n in seen.items() if n >= drawn.get(p, 10 ** 9)}
    for r in recs:
        p = prom.get(r["review_url"], {})
        r["coverage"] = p.get("coverage", "unknown")
        r["complete"] = r["post_id"] in complete
        r["_score"] = None if r.get("excluded") else _score(r)
    run = [r for r in recs if not r.get("excluded")]
    return recs, run


def posts(run, pred=None):
    """Complete posts -> list of their claim records."""
    by = collections.defaultdict(list)
    for r in run:
        if r["complete"] and (pred is None or pred(r)):
            by[r["post_id"]].append(r)
    return by


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------- a. funnel
def fig_funnel(recs, run):
    stages = ["drawn", "run", "≥1 doc", "≥1 directional"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(BINS))
    wdt = 0.2
    for i, st in enumerate(stages):
        vals = []
        for b in BINS:
            d = [r for r in recs if r["bin"] == b]
            rn = [r for r in run if r["bin"] == b]
            if st == "drawn":
                vals.append(len(d))
            elif st == "run":
                vals.append(len(rn))
            elif st == "≥1 doc":
                vals.append(sum(1 for r in rn if r["results"]))
            else:
                vals.append(sum(1 for r in rn if any(
                    e["read"]["direction"] in "54321" for e in r["results"])))
        ax.bar(x + (i - 1.5) * wdt, vals, wdt, color=GREY[i],
               edgecolor="white", linewidth=0.5, label=st)
    ax.set_xticks(x)
    ax.set_xticklabels(BINS)
    ax.set_xlabel("NewsGuard bin")
    ax.set_ylabel("claims")
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), handlelength=1.4, columnspacing=1.6)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, "e2_funnel.png")


# ------------------------------------------------------------- b. flag mix
def fig_flagmix(run):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bottom = np.zeros(len(BINS))
    for f, col in zip(FLAGS, FLAG_GREY):
        vals = []
        for b in BINS:
            c = collections.Counter(e["read"]["direction"]
                                    for r in run if r["bin"] == b for e in r["results"])
            vals.append(100 * c[f] / max(sum(c.values()), 1))
        vals = np.array(vals)
        ax.bar(BINS, vals, 0.62, bottom=bottom, color=col,
               edgecolor="white", linewidth=0.5, label=f)
        for xi, (v, bo) in enumerate(zip(vals, bottom)):
            if v >= 5:
                ax.text(xi, bo + v / 2, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, color="0.98" if col < "0.5" else "0.15")
        bottom += vals
    ax.set_ylabel("% of documents read")
    ax.set_xlabel("NewsGuard bin")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              handlelength=1.2, columnspacing=1.4,
              title="flag  ·  5 4 support · 3 X no direction · 2 1 refute · I no bearing",
              title_fontsize=9)
    save(fig, "e2_flag_mix.png")


# ------------------------------------------------- b2. post-level signal rate
def fig_post_signal(run):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs, ys, los, his, ns = [], [], [], [], []
    for b in BINS:
        by = posts(run, lambda r, b=b: r["bin"] == b)
        n = len(by)
        k = sum(1 for v in by.values()
                if any(e["read"]["direction"] in "54321" for r in v for e in r["results"]))
        lo, hi = wilson(k, n)
        xs.append(b); ys.append(100 * k / max(n, 1))
        los.append(100 * lo); his.append(100 * hi); ns.append(n)
    err = np.array([np.array(ys) - np.array(los), np.array(his) - np.array(ys)])
    ax.bar(xs, ys, 0.6, color="0.35", edgecolor="white", linewidth=0.5)
    ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor="0.10", capsize=4, linewidth=1.1)
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax.text(i, 5, f"n={n}", ha="center", fontsize=9.5, color="0.98")
        ax.text(i, y + (his[i] - y) + 3.5, f"{y:.1f}%", ha="center", fontsize=10,
                color="0.15")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of posts with ≥1 directional read")
    ax.set_xlabel("NewsGuard bin")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)

    save(fig, "e2_post_signal.png")


# ------------------------------------------------------ c. score distribution
def fig_score(run, name="e2_score_distribution.png"):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    data = [[r["_score"] for r in run if r["bin"] == b] for b in BINS]
    bp = ax.boxplot(data, labels=BINS, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="0.05", linewidth=1.4))
    for patch, med, c, mc in zip(bp["boxes"], bp["medians"], RAMP,
                                 ["white", "white", "0.05", "0.05", "0.05"]):
        patch.set_facecolor(c)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)
        med.set_color(mc)
    ax.axhline(E1T, color="0.10", linestyle="--", linewidth=1.1)
    ax.text(1.02, E1T, "nudge\nthreshold", transform=ax.get_yaxis_transform(),
            fontsize=9, color="0.25", va="center", ha="left")
    ax.set_ylabel("claim log-odds  (E1 weights)")
    ax.set_xlabel("NewsGuard bin")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)

    save(fig, name)


# -------------------------------------------------------- d. evidence quality
def fig_quality(run):
    """Small multiples, not a twin axis. Two y-scales on one frame forced the legend
    into the data and made a flat series look like it shared the other's scale."""
    panels = [("mean NewsGuard score\nof retrieved evidence", "ng", (85, 95)),
              ("% of documents\nNG-rated", "rated", (0, 80)),
              ("% primary\nsources", "primary", (0, 30)),
              ("independent voices\nper claim", "voices", (0, 12))]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.3))
    for ax, (lab, key, ylim) in zip(axes, panels):
        vals = []
        for b in BINS:
            docs = [e for r in run if r["bin"] == b for e in r["results"]]
            if key == "ng":
                ngs = [e["ng"] for e in docs if e.get("ng") is not None]
                vals.append(np.mean(ngs) if ngs else 0)
            elif key == "rated":
                vals.append(100 * sum(1 for e in docs if e.get("ng") is not None)
                            / max(len(docs), 1))
            elif key == "primary":
                vals.append(100 * sum(1 for e in docs if e.get("rel") == "PRIMARY")
                            / max(len(docs), 1))
            else:
                vpc = [len({e.get("voice") for e in r["results"]})
                       for r in run if r["bin"] == b]
                vals.append(np.mean(vpc) if vpc else 0)
        ax.plot(range(len(BINS)), vals, "o-", color="0.20", linewidth=1.6,
                markersize=5, markeredgecolor="white", markeredgewidth=0.9)
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels(BINS, rotation=45, ha="right", fontsize=8.5)
        ax.set_ylim(*ylim)
        ax.set_title(lab, fontsize=9.5, color="0.25", pad=8)
        ax.grid(axis="y", color="0.92", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("")
    fig.supxlabel("NewsGuard bin of the POSTING outlet", fontsize=10, color="0.3")
    save(fig, "e2_evidence_quality.png")


# ------------------------------------------- e0. nudge rate at the E1 threshold
def fig_nudge_rate(run, name="e2_nudge_rate.png"):
    """The E2 headline as one bar chart: % of complete posts whose worst claim
    crosses E1's fitted operating point, per bin. Same unit and intervals as
    fig_post_signal; the full sweep across cutoffs stays in fig_sweep."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs, ys, los, his, ns = [], [], [], [], []
    for b in BINS:
        by = posts(run, lambda r, b=b: r["bin"] == b)
        n = len(by)
        k = sum(1 for v in by.values() if min(x["_score"] for x in v) <= E1T)
        lo, hi = wilson(k, n)
        xs.append(b); ys.append(100 * k / max(n, 1))
        los.append(100 * lo); his.append(100 * hi); ns.append(n)
    err = np.array([np.array(ys) - np.array(los), np.array(his) - np.array(ys)])
    ax.bar(xs, ys, 0.6, color=RAMP, edgecolor="white", linewidth=0.5)
    ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor="0.10", capsize=4, linewidth=1.1)
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax.text(i, y + (his[i] - y) + 0.6, f"{y:.1f}%", ha="center", fontsize=10.5,
                color="0.15")
        ax.text(i, 0.5, f"n={n}", ha="center", fontsize=9,
                color="0.98" if i < 2 else "0.15")
    ax.set_ylim(0, 17)
    ax.set_ylabel("% of posts crossing the nudge threshold")
    ax.set_xlabel("NewsGuard bin of the posting outlet", labelpad=12)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, name)


# ---------------------------------------------------------- e. threshold sweep
def fig_sweep(run):
    cuts = np.linspace(-10, 0, 41)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for b, c in zip(BINS, RAMP):
        by = posts(run, lambda r, b=b: r["bin"] == b)
        n = len(by)
        ys = [100 * sum(1 for v in by.values()
                        if min(x["_score"] for x in v) <= t) / max(n, 1) for t in cuts]
        ax.plot(cuts, ys, "-", color=c, linewidth=1.7, label=f"NG {b}  (n={n})")
    ax.axvline(E1T, color="0.10", linestyle="--", linewidth=1.1)
    ax.text(E1T - 0.25, 47, "nudge threshold ", fontsize=9, color="0.25",
            rotation=90, va="top", ha="right")
    ax.set_xlabel("nudge cutoff  (claim log-odds)")
    ax.set_ylabel("% of posts nudged")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)
    ax.grid(color="0.93", linewidth=0.6)
    ax.set_axisbelow(True)

    save(fig, "e2_threshold_sweep.png")


# ------------------------------------------------------------- f. prominence
def fig_prominence(run):
    """Coverage effect alone. The outlet ranking moved to its own tall figure —
    50 labels cannot fit beside a bar chart without colliding."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    covs = ["many", "few", "none"]
    labs = ["many outlets\nwould report it", "a few would", "essentially\nnobody would"]
    ys, los, his, ns = [], [], [], []
    for c in covs:
        sel = [r for r in run if r["coverage"] == c]
        k = sum(1 for r in sel if any(e["read"]["direction"] in "54321"
                                      for e in r["results"]))
        lo, hi = wilson(k, len(sel))
        ys.append(100 * k / max(len(sel), 1)); los.append(100 * lo); his.append(100 * hi)
        ns.append(len(sel))
    err = np.array([np.array(ys) - np.array(los), np.array(his) - np.array(ys)])
    ax.bar(labs, ys, 0.58, color=[RAMP[0], RAMP[2], RAMP[4]],
           edgecolor="white", linewidth=0.5)
    ax.errorbar(labs, ys, yerr=err, fmt="none", ecolor="0.10", capsize=5, linewidth=1.1)
    for i, (y, n) in enumerate(zip(ys, ns)):
        ax.text(i, y + (his[i] - y) + 3, f"{y:.1f}%", ha="center", fontsize=11,
                color="0.12")
        ax.text(i, 5, f"n={n:,}", ha="center", fontsize=9.5,
                color="0.98" if i == 0 else "0.15")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of claims finding directional evidence")
    ax.set_xlabel("expected independent coverage, scored before retrieval",
                  labelpad=12)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.grid(axis="y", color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, "e2_prominence.png")


def _outlet_pts(run):
    """Per outlet: mean directional docs per claim, 95% CI of the mean, NG score."""
    import statistics as st
    roster = {r["domain"]: r["ng_score"]
              for r in pl.read_csv("eval/data/survey_claims/source_roster.csv").iter_rows(named=True)}
    FOCUS = {**{d: "local / metro" for d in LOCAL},
             **{d: "international" for d in ("reuters.com", "the-sun.com",
                                             "thegrayzone.com", "mintpressnews.com",
                                             "consortiumnews.com")}}
    pts = []
    for d in sorted({r["publisher_site"] for r in run}):
        sel = [r for r in run if r["publisher_site"] == d]
        cs = [sum(1 for e in r["results"] if e["read"]["direction"] in ("5", "4", "1", "2"))
              for r in sel]
        m = st.mean(cs)
        half = 1.96 * st.stdev(cs) / len(cs) ** 0.5
        pts.append(dict(v=m, lo=m - half, hi=m + half, d=d, n=len(cs),
                        ng=roster[d], bin=sel[0]["bin"],
                        focus=FOCUS.get(d, "national / specialist (US)")))
    return pts


OUTLET_STYLE = {"local / metro": dict(marker="o", s=46, c="#24404f"),
                "international": dict(marker="s", s=36, c="#a05c46"),
                "national / specialist (US)": dict(marker="o", s=26, c="#aab3b8")}


def _draw_outlet_rows(ax, rows):
    """rows: list of (y, pt-or-None-header-label). Draws points, CIs, labels."""
    ys, labels = [], []
    for y, item in rows:
        ys.append(y)
        if isinstance(item, str):
            labels.append(item)
            continue
        labels.append(f'{item["d"]}  \u00b7  {item["ng"]:.0f}')
        ax.plot([item["lo"], item["hi"]], [y, y], "-", color="0.82",
                linewidth=1.6, zorder=1)
        ax.text(item["hi"] + 0.12, y, f'{item["v"]:.1f}', va="center",
                fontsize=8.2, color="0.45")
    for fc, stl in OUTLET_STYLE.items():
        sel = [(y, it) for y, it in rows if not isinstance(it, str) and it["focus"] == fc]
        if sel:
            ax.scatter([it["v"] for _, it in sel], [y for y, _ in sel],
                       edgecolors="white", linewidths=0.8, zorder=3, label=fc, **stl)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.8)
    for tick, (_, it) in zip(ax.get_yticklabels(), rows):
        if isinstance(it, str):
            tick.set_color("0.15"); tick.set_fontweight("bold")
        else:
            tick.set_color("0.2")
    ax.grid(axis="x", color="0.93", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("supporting or refuting documents per claim, outlet average",
                  labelpad=10)


def fig_outlets(run):
    """Every outlet ranked by mean directional docs per claim; NG score in the
    label; markers = hand-assigned EDITORIAL coverage focus (Daniel 2026-08-18:
    the local-is-not-the-issue argument must not rest on an LLM call).
    Metric switched from %-of-claims-with-any to mean count, Daniel 2026-08-19."""
    pts = sorted(_outlet_pts(run), key=lambda p: p["v"])
    rows = list(enumerate(pts))
    fig, ax = plt.subplots(figsize=(7.8, 10.8))
    _draw_outlet_rows(ax, rows)
    ax.set_ylim(-1, len(pts))
    ax.set_xlim(0, 9.2)
    ax.legend(frameon=False, loc="lower right", fontsize=9,
              title="editorial coverage focus", title_fontsize=9, borderaxespad=1.2)
    save(fig, "e2_outlets.png")


def fig_outlets_by_bin(run):
    """Appendix twin of fig_outlets: same points grouped by NewsGuard bin,
    least reliable bin at the top."""
    pts = _outlet_pts(run)
    rows, y = [], 0
    for b in reversed(BINS):          # build bottom-up so 0-30 lands on top
        grp = sorted((p for p in pts if p["bin"] == b), key=lambda p: p["v"])
        for p in grp:
            rows.append((y, p)); y += 1
        rows.append((y, f"NewsGuard {b}")); y += 2
    fig, ax = plt.subplots(figsize=(7.8, 11.6))
    _draw_outlet_rows(ax, rows)
    ax.set_ylim(-1, y - 1)
    ax.set_xlim(0, 9.2)
    ax.legend(frameon=False, loc="lower right", fontsize=9,
              title="editorial coverage focus", title_fontsize=9, borderaxespad=1.2)
    save(fig, "e2_outlets_by_bin.png")


def main():
    _load_e1()
    recs, run = load()
    print(f"E2 figures — {len(recs)} records, {len(run)} run")
    fig_funnel(recs, run)
    fig_flagmix(run)
    fig_post_signal(run)
    fig_nudge_rate(run)
    fig_score(run)
    run_x = [r for r in run if r["publisher_site"] != "occupydemocrats.com"]
    fig_nudge_rate(run_x, name="e2_nudge_rate_ex_od.png")
    fig_score(run_x, name="e2_score_distribution_ex_od.png")
    fig_quality(run)
    fig_sweep(run)
    fig_prominence(run)
    fig_outlets(run)
    fig_outlets_by_bin(run)


if __name__ == "__main__":
    main()
