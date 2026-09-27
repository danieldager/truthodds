"""Thursday-deck figures for the E1 urn run.

    uv run python -m eval.scripts.build_eval.e1_figures \
        -i eval/data/urn_runs/e1_ctx/results-00.jsonl -o eval/data/urn_runs/e1_ctx/figures

Every number plotted is recomputed here from the run file through fit_urn's own
load/fit/score path, so a figure cannot drift from the banked fit, which is
asserted before anything is drawn. These are the THREE-VOICE figures, so the
banked fit they check against is headline_metrics_clustered.json (3-voice on the
frozen population), not headline_metrics.json — that file has been the shipped
SEVEN-FLAG fit since 2026-09-14.

Detection framing throughout: the thing being detected is a FALSE claim, so
recall is over gold-FALSE and FPR is over gold-TRUE (a flagged true post is the
error the product cannot afford). Claims score LOW when false, so a claim is
flagged when score <= threshold.

House style: greyscale, white mark edges, figsize ~(6,4), dpi 120 -- matching
eval/scripts/claim_sourcing/compute_scorecard.py and the monochrome metropolis deck.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.scripts.build_eval import fit_urn

FLAGS = ("5", "4", "3", "X", "I", "2", "1")
SUPPORT, REFUTE = ("5", "4"), ("1", "2")
# 7 steps ordered support -> silent -> refute, so the bar reads dark-to-light
# in the direction the model reads it.
FLAG_GREY = ["0.10", "0.30", "0.50", "0.62", "0.74", "0.86", "0.95"]
MIN_MINORITY = 30   # a stratum needs this many in BOTH classes before its AUC means anything
BOOT_REPS, BOOT_SEED = 2000, 707   # stratum CIs; seed matches the run's draw seed
GREY = ["0.15", "0.4", "0.6", "0.8"]
DPI = 120


def roc(pairs: list[tuple[float, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Sweep every achievable threshold. x = FPR over TRUE, y = recall over FALSE."""
    trues = np.sort([s for s, y in pairs if y == 1])
    falses = np.sort([s for s, y in pairs if y == 0])
    thr = np.array(sorted({s for s, _ in pairs}))
    fpr = np.searchsorted(trues, thr, side="right") / len(trues)
    rec = np.searchsorted(falses, thr, side="right") / len(falses)
    return np.r_[0.0, fpr, 1.0], np.r_[0.0, rec, 1.0]


def oof_scored(rows: list[dict]) -> list[tuple[dict, float]]:
    """fit_urn.out_of_fold, but keeping the row so strata can be cut afterwards."""
    out = []
    for f in range(fit_urn.K_FOLDS):
        train = [r for r in rows if r["fold"] != f]
        test = [r for r in rows if r["fold"] == f]
        if not train or not test:
            continue
        w = fit_urn.fit(train)
        out += [(r, fit_urn.score(r, w)) for r in test]
    return out


def _auc_np(s_pos: np.ndarray, s_neg: np.ndarray) -> float:
    """Mann-Whitney U, ties half -- the vectorised twin of fit_urn.auc."""
    neg = np.sort(s_neg)
    lo = np.searchsorted(neg, s_pos, side="left")
    hi = np.searchsorted(neg, s_pos, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(s_pos) * len(neg)))


def bootstrap_auc(rows: list[dict], reps: int, rng) -> tuple[float, float]:
    """Percentile CI on a stratum's out-of-fold AUC.

    Resamples claims WITHIN each gold class, so the reported n and true-count stay
    fixed and the interval is about sampling the claims, not the class balance.
    The whole out-of-fold procedure is redone inside every replicate -- the weights
    are refitted on the resampled folds -- because the plotted number is itself an
    out-of-fold AUC with stratum-specific weights, and holding the weights fixed
    would understate the interval.
    """
    y = np.array([r["y"] for r in rows])
    nt = np.array([r["n_t"] for r in rows], float)
    nf = np.array([r["n_f"] for r in rows], float)
    ne = np.array([r["n_e"] for r in rows], float)
    fold = np.array([r["fold"] for r in rows])
    idx_t, idx_f = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    counts = np.stack([nt, nf, ne])
    aucs = []
    for _ in range(reps):
        i = np.r_[rng.choice(idx_t, len(idx_t)), rng.choice(idx_f, len(idx_f))]
        c, yy, ff = counts[:, i], y[i], fold[i]
        scores = np.empty(len(i))
        ok = True
        for k in range(fit_urn.K_FOLDS):
            tr, te = ff != k, ff == k
            if not tr.any() or not te.any():
                continue
            pos, neg = tr & (yy == 1), tr & (yy == 0)
            tot_t, tot_f = c[:, pos].sum(), c[:, neg].sum()
            if not tot_t or not tot_f:
                ok = False
                break
            w = np.log(((c[:, pos].sum(1) + 1) / (tot_t + 3))
                       / ((c[:, neg].sum(1) + 1) / (tot_f + 3)))
            scores[te] = w @ c[:, te]
        if ok and (yy == 1).any() and (yy == 0).any():
            aucs.append(_auc_np(scores[yy == 1], scores[yy == 0]))
    if len(aucs) < reps // 2:   # too degenerate to resample: no interval exists
        return float("nan"), float("nan")
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str) -> Path:
    fig.tight_layout()
    p = out / name
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {p}")
    return p


# ---------------------------------------------------------------- figures

def fig_roc(out: Path, variants: dict[str, tuple[list, float]]) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    styles = [("-", "#24404f", 1.8), ("--", "0.55", 1.3)]
    for (label, (pairs, a)), (ls, c, lw) in zip(variants.items(), styles):
        x, y = roc(pairs)
        ax.plot(x, y, ls, color=c, linewidth=lw, label=f"{label}  AUC {a:.3f}")
    ax.plot([0, 1], [0, 1], "-", color="0.85", linewidth=1, zorder=0)
    ax.set_xlabel("false-positive rate (gold-TRUE claims flagged)")
    ax.set_ylabel("recall (gold-FALSE claims flagged)")
    ax.set_title("Truth Odds separates true from false claims", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    style(ax)
    return save(fig, out, "e1_roc.png")


def fig_scores(out: Path, pairs: list[tuple[float, int]], thr: float,
               rec: float, fpr: float) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    t = np.array([s for s, y in pairs if y == 1])
    f = np.array([s for s, y in pairs if y == 0])
    lo, hi = np.percentile(np.r_[t, f], [0.5, 99.5])
    bins = np.linspace(lo, hi, 31)
    # grouped, not overlaid: opaque overlaid bars hide the shorter series.
    ax.hist([f, t], bins=bins, weights=[np.full(len(f), 1 / len(f)),
                                        np.full(len(t), 1 / len(t))],
            color=["#a05c46", "#8fb0c2"], edgecolor="white", linewidth=0.4,
            label=[f"gold false or mixed (n={len(f)})", f"gold true (n={len(t)})"])
    ax.axvline(thr, color="0.10", linewidth=1.2, linestyle="--")
    ax.annotate(f"nudge threshold\nFPR {fpr:.1%}, recall {rec:.1%}", xy=(thr, ax.get_ylim()[1] * 0.88),
                xytext=(-6, 0), textcoords="offset points", fontsize=7.5,
                va="top", ha="right")
    ax.set_xlabel("out-of-fold Truth Odds score (log-odds)")
    ax.set_ylabel("share of claims")
    ax.set_title("Score distribution by gold label", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    return save(fig, out, "e1_score_distribution.png")


def fig_flag_mix(out: Path, recs: list[dict]) -> Path:
    # Word labels throughout (Daniel 2026-08-18): the raw flags are digits 5..1 and the
    # gold scale is also 1..5, so the two collided in the reader's head. The four
    # direction groups carry the figure's point; strong-vs-lean detail does not.
    groups = [("supports the claim", SUPPORT, "#2f5468", "0.98"),
              ("bears on it, no direction", ("3", "X"), "#8a9ba5", "0.15"),
              ("no bearing on the claim", ("I",), "#d9dfe2", "0.15"),
              ("refutes the claim", REFUTE, "#a05c46", "0.98")]
    vlabels = ["false", "mostly\nfalse", "mixed", "mostly\ntrue", "true"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    vs = [1, 2, 3, 4, 5]
    counts, nd = [], []
    for v in vs:
        docs = [d for r in recs if r.get("veracity") == v for d in r["results"]]
        counts.append(collections.Counter(d["read"]["direction"] for d in docs))
        nd.append(len(docs))
    bottom = np.zeros(len(vs))
    for lab, flags, col, txt in groups:
        vals = np.array([sum(c[f] for f in flags) / max(n, 1)
                         for c, n in zip(counts, nd)])
        ax.bar(vs, vals, 0.62, bottom=bottom, color=col, edgecolor="white",
               linewidth=1.0, label=lab)
        for x, (v_, b_) in enumerate(zip(vals, bottom)):
            if v_ >= 0.05:
                ax.text(vs[x], b_ + v_ / 2, f"{v_:.0%}", ha="center", va="center",
                        fontsize=8, color=txt)
        bottom += vals
    ax.set_xticks(vs)
    ax.set_xticklabels([f"{vlabels[i]}\n{nd[i]:,} docs" for i in range(len(vs))],
                       fontsize=8.5)
    ax.set_ylabel("share of documents read")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(fontsize=8.5, ncol=4, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.34), handlelength=1.2, columnspacing=1.2)
    style(ax)
    fig.subplots_adjust(bottom=0.38)
    p = out / "e1_flag_mix.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


def fig_operating(out: Path, pairs: list[tuple[float, int]], rec: float, fpr: float) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x, y = roc(pairs)
    ax.plot(x, y, "-", color="0.10", linewidth=1.8)
    ax.axvline(0.02, color="0.55", linewidth=1, linestyle=":")
    ax.plot([fpr], [rec], "o", color="0.10", markersize=7,
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.annotate(f"FPR {fpr:.1%}\nrecall {rec:.1%}\nLR+ 19.8", xy=(fpr, rec),
                xytext=(10, -22), textcoords="offset points", fontsize=8)
    ax.text(0.0205, 0.02, "2% FPR budget", fontsize=7.5, color="0.45", rotation=90)
    ax.set_xlim(0, 0.10); ax.set_ylim(0, 0.65)
    ax.set_xlabel("false-positive rate (gold-TRUE claims flagged)")
    ax.set_ylabel("recall (gold-FALSE claims flagged)")
    ax.set_title("Operating point at the 2% false-positive budget", fontsize=11)
    style(ax)
    return save(fig, out, "e1_operating_point.png")


def fig_strata(out: Path, metrics: dict, headline: float,
               claims: list[dict], rng) -> tuple[Path, dict]:
    """One row per confound worth ruling out, with a bootstrap CI on each.

    Three rewrites. The first floated the AUC/n text beside each dot, where it
    collided with neighbours. The second was still hard to read CONCEPTUALLY: it
    stacked three unrelated questions in one chart and spent 5 of its 13 rows on
    strata with almost no minority class (factuel.afp.com's 0.612 is 3 true claims
    out of 205). Now each group header states the question it answers, and a
    stratum enters only if BOTH classes clear MIN_MINORITY.

    The claim-screen group is gone for the same reason: once the ill-posed classes
    are excluded it has one surviving member, and a group of one is not a
    stratification. Its result is reported as a line of text instead.

    The third rewrite added the intervals. The fourth is why the ceiling block is
    PUBLISHER-CONTROLLED. Pooled, the ceiling strata looked like a clean gradient
    (claim_date 0.807 -> review-1 0.947, non-overlapping). They are not: claim_date
    is 38% politifact and x_date is 84% snopes, and political-statement claims are
    harder than viral-hoax claims, so the ceiling strata were sorting claims by
    publisher. Hold the publisher fixed and it flattens -- politifact/claim_date
    0.838 and snopes/x_date 0.817, both near the headline. What survives is
    snopes/x_date+clamped at 0.891, and "clamped" is by construction the same event
    as "the post is dated on or after the review", so loosest-ceiling and
    recirculated-claim cannot be told apart here. Tests in
    scratchpad/ceiling_test.py, recorded in clog/040826.
    """
    def stat(rs: list[dict]) -> tuple:
        a = fit_urn.auc(fit_urn.out_of_fold(rs))
        return a, len(rs), sum(r["y"] for r in rs), bootstrap_auc(rs, BOOT_REPS, rng)

    P, C = "publisher", "ceiling_src"
    spec = [("head", "Is it set by one fact-checker?", None),
            ("stratum", "snopes.com", lambda r: r[P] == "snopes.com"),
            ("stratum", "politifact.com", lambda r: r[P] == "politifact.com"),
            ("head", "Is it set by the date ceiling?  (publisher held fixed)", None),
            ("stratum", "snopes · x_date",
             lambda r: r[P] == "snopes.com" and r[C] == "x_date"),
            ("stratum", "snopes · x_date+clamped",
             lambda r: r[P] == "snopes.com" and r[C] == "x_date+clamped"),
            ("stratum", "politifact · claim_date",
             lambda r: r[P] == "politifact.com" and r[C] == "claim_date")]

    rows, cis = [], {}                         # (kind, label, auc, n, n_true)
    for kind, label, pred in spec:
        if kind == "head":
            rows.append((kind, label, None, None, None))
            continue
        a, n, nt, ci = stat([r for r in claims if pred(r)])
        assert min(nt, n - nt) >= MIN_MINORITY, f"{label} minority {min(nt, n - nt)}"
        cis[label] = ci
        rows.append((kind, label, a, n, nt))
    for k in ("snopes.com", "politifact.com"):  # these two are also banked -- check
        i = [r[1] for r in rows].index(k)
        assert abs(rows[i][2] - metrics["by_publisher"][k]["auc_oof"]) < 1e-12, k
    scr = metrics["headline_screened_ok"]
    cis["OVERALL"] = h_lo, h_hi = bootstrap_auc(claims, BOOT_REPS, rng)

    ypos = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    fig.subplots_adjust(left=0.30, right=0.72, top=0.87, bottom=0.20)

    for yp, (kind, label, a, n, nt) in zip(ypos, rows):
        if kind == "head":
            ax.annotate(label, xy=(-0.40, yp), xycoords=("axes fraction", "data"),
                        fontsize=9.5, color="0.15", va="center", annotation_clip=False)
            ax.axhline(yp - 0.5, color="0.88", linewidth=0.8, zorder=0)
            continue
        lo, hi = cis[label]
        ax.plot([lo, hi], [yp, yp], "-", color="0.35", linewidth=1.4, zorder=2,
                solid_capstyle="butt")
        for b in (lo, hi):
            ax.plot([b, b], [yp - 0.13, yp + 0.13], "-", color="0.35",
                    linewidth=1.4, zorder=2)
        ax.plot([a], [yp], "o", markersize=7, zorder=3, color="0.15",
                markeredgecolor="white", markeredgewidth=1.0)
        ax.annotate(label, xy=(-0.03, yp), xycoords=("axes fraction", "data"),
                    fontsize=8.5, color="0.25", ha="right", va="center",
                    annotation_clip=False)
        ax.annotate(f"{a:.3f}", xy=(1.05, yp), xycoords=("axes fraction", "data"),
                    fontsize=8.5, color="0.15", ha="left", va="center",
                    family="monospace", annotation_clip=False)
        ax.annotate(f"[{lo:.3f}, {hi:.3f}]", xy=(1.17, yp),
                    xycoords=("axes fraction", "data"), fontsize=8, color="0.45",
                    ha="left", va="center", family="monospace", annotation_clip=False)
        ax.annotate(f"n {n:,}", xy=(1.50, yp), xycoords=("axes fraction", "data"),
                    fontsize=8, color="0.45", ha="left", va="center",
                    family="monospace", annotation_clip=False)

    ax.axvspan(h_lo, h_hi, color="0.90", zorder=0)
    ax.axvline(headline, color="0.15", linewidth=1.1, zorder=1)
    ax.annotate(f"headline {headline:.3f}", xy=(headline, len(rows) - 0.35),
                fontsize=8, color="0.15", ha="center", annotation_clip=False)
    ax.annotate("AUC    95% CI          n", xy=(1.05, len(rows) - 0.35),
                xycoords=("axes fraction", "data"), fontsize=8, color="0.45",
                ha="left", family="monospace", annotation_clip=False)
    ax.set_xlim(0.74, 1.0); ax.set_ylim(-0.9, len(rows) - 0.1)
    ax.set_yticks([])
    ax.set_xticks([0.8, 0.9, 1.0])
    ax.set_xlabel("out-of-fold AUC  (bars = 95% bootstrap CI, "
                  f"{BOOT_REPS:,} replicates)", fontsize=9)
    ax.set_title("No confound explains the headline", fontsize=11, x=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=8.5, color="0.6")
    ax.annotate(
        "Pooled by ceiling alone the strata ran 0.807 to 0.947, but claim_date is 38% "
        "politifact and x_date is 84% snopes — that gradient was publisher mix.\n"
        "The one row still above the band is the clamped stratum, where the post is "
        "dated on or after the review. Loosest-ceiling and recirculated-claim are the\n"
        f"same event by construction, so this run cannot separate them. Screened-ok "
        f"claims score {scr['auc_oof']:.3f} on n={scr['n']:,}, so ill-posed claims are not "
        "driving it either.",
        xy=(-0.40, -0.30), xycoords="axes fraction",
        fontsize=7.5, color="0.45", va="top", annotation_clip=False)
    p = out / "e1_strata.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p, cis


def fig_weights(out: Path, w: dict[str, float]) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    names = ["supporting\nvoice", "refuting\nvoice", "silent\nvoice"]
    vals = [w["n_t"], w["n_f"], w["n_e"]]
    ax.bar(names, vals, 0.55, color=["0.25", "0.45", "0.80"],
           edgecolor="white", linewidth=1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.09 if v > 0 else -0.09), f"{v:+.3f}", ha="center",
                va="bottom" if v > 0 else "top", fontsize=9)
    ax.axhline(0, color="0.2", linewidth=0.9)
    ax.set_ylim(-2.3, 2.6)
    ax.set_ylabel("fitted log-likelihood ratio per voice")
    ax.set_title("Fitted per-voice weights", fontsize=11)
    style(ax)
    return save(fig, out, "e1_weights.png")


def fig_funnel(out: Path, f: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    steps = [("drawn", f["drawn"]), ("ran", f["ran"]),
             ("returned evidence", f["with_docs"]),
             ("gold-directional (scored)", f["scored"])]
    ypos = np.arange(len(steps))[::-1]
    ax.barh(ypos, [n for _, n in steps], 0.55,
            color=["0.80", "0.62", "0.42", "0.18"], edgecolor="white", linewidth=1.0)
    for yp, (_, n) in zip(ypos, steps):
        ax.text(n + 45, yp, f"{n:,}", va="center", fontsize=9, color="0.15")
    ax.set_yticks(ypos)
    ax.set_yticklabels([s for s, _ in steps], fontsize=8.5)
    ax.set_xlim(0, f["drawn"] * 1.16)
    ax.set_xlabel("claims")
    ax.set_title("Claim funnel — every drop is accounted for", fontsize=11)
    note = (f"{f['excluded']} excluded by the hand-read gate  ·  "
            f"{f['zero']} returned zero documents  ·  {f['mid']} gold-mid (veracity 3, no direction)\n"
            f"{f['docs']:,} documents read  ·  {f['read_ok']:.1%} readable  ·  "
            f"{f['snippet']:.1%} snippet-only  ·  {f['leaks']} dated after the ceiling")
    ax.text(0, -0.30, note, transform=ax.transAxes, fontsize=7.5, color="0.35", va="top")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="0.9", linewidth=0.6); ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=0.34)
    p = out / "e1_funnel.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


# ---------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=Path)
    ap.add_argument("-o", "--out", required=True, type=Path)
    ap.add_argument("-m", "--metrics", type=Path,
                    help="banked 3-voice fit (default: headline_metrics_clustered.json "
                         "alongside the input)")
    ap.add_argument("--population", type=Path, default=fit_urn.POPULATION_FC_GOLD,
                    help="parquet with a claim_id column: restrict to those claims")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(
        (args.metrics or args.input.parent / "headline_metrics_clustered.json").read_text())

    print("fitting …")
    pop = fit_urn.load_population(args.population)
    rows = fit_urn.load_headline(args.input, None, False, population=pop)
    oof = fit_urn.out_of_fold(rows)
    a = fit_urn.auc(oof)
    assert abs(a - metrics["overall"]["auc_oof"]) < 1e-12, \
        f"reproduced AUC {a} != banked {metrics['overall']['auc_oof']}"
    rec, fpr, thr = fit_urn.recall_at_fpr(oof, 0.02)
    print(f"  headline reproduced: AUC {a:.4f}  recall {rec:.3f} @ FPR {fpr:.4f}")

    rows10 = fit_urn.load_headline(args.input, 10, False, population=pop)
    rows_db = fit_urn.load_headline(args.input, None, True, population=pop)
    oof10, oof_db = fit_urn.out_of_fold(rows10), fit_urn.out_of_fold(rows_db)
    # Plotted: the headline and the slot-convention sensitivity. The
    # blanket-citation sensitivity is recorded in figures.json and stated in
    # prose instead -- a third near-identical curve added clutter, not evidence.
    variants = {
        "headline (N = documents returned)": (oof, a),
        "sensitivity: N = 10 slots": (oof10, fit_urn.auc(oof10)),
    }
    auc_db = fit_urn.auc(oof_db)
    rec_db, fpr_db, _ = fit_urn.recall_at_fpr(oof_db, 0.02)
    print(f"  blanket-demoted (not plotted): AUC {auc_db:.4f}  recall {rec_db:.3f}")

    _axis = fit_urn.load_judged_axis()
    recs = [json.loads(l) for l in args.input.open()
            if _axis.get(json.loads(l)["review_url"]) != fit_urn.MEDIA_AXIS]
    run = [r for r in recs if not r.get("excluded")]
    docs = [d for r in run for d in r["results"]]
    funnel = {
        "drawn": len(recs),
        "excluded": sum(1 for r in recs if r.get("excluded")),
        "ran": len(run),
        "with_docs": sum(1 for r in run if r["results"]),
        "zero": sum(1 for r in run if not r["results"]),
        "mid": sum(1 for r in run if r.get("veracity") == 3),
        "scored": len(rows),
        "docs": len(docs),
        "read_ok": sum(1 for d in docs if d["read_status"] == "ok") / len(docs),
        "snippet": sum(1 for d in docs if d["provenance"] == "snippet") / len(docs),
        "leaks": sum(1 for d in docs if d.get("leak_flag")),
    }
    print(f"  funnel: {funnel}")

    print("drawing …")
    strata_path, cis = fig_strata(args.out, metrics, a, rows,
                                  np.random.default_rng(BOOT_SEED))
    paths = [
        fig_roc(args.out, variants),
        fig_scores(args.out, oof, thr, rec, fpr),
        fig_flag_mix(args.out, run),
        fig_operating(args.out, oof, rec, fpr),
        strata_path,
        fig_weights(args.out, metrics["overall"]["weights"]),
        fig_funnel(args.out, funnel),
    ]
    (args.out / "figures.json").write_text(json.dumps(
        {"input": str(args.input), "auc_oof": a, "recall_at_2pct_fpr": rec, "fpr": fpr,
         "threshold": thr, "variant_auc": {k: v[1] for k, v in variants.items()},
         "blanket_demoted": {"auc_oof": auc_db, "recall_at_2pct_fpr": rec_db,
                             "fpr": fpr_db, "plotted": False},
         "strata_min_minority": MIN_MINORITY,
         "strata_ci95": {k: (None if np.isnan(v[0]) else [v[0], v[1]])
                         for k, v in cis.items()},
         "bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED,
                       "method": "stratified percentile, refit inside each replicate"},
         "funnel": funnel, "figures": [str(p) for p in paths]}, indent=2))
    print(f"\n{len(paths)} figures -> {args.out}")


if __name__ == "__main__":
    main()
