"""Score distributions per gold veracity label + the fit-on-extremes experiment.

Figure (appendix of the results brief): out-of-fold Truth Odds score per 5-point
gold label, scored with the committed fit (train = directional labels 1/2/4/5).

Experiment (Daniel 2026-08-19): does the fit need the "mostly" labels at all, and
what happens when mixed (3) is counted as false AT EVAL time only?
  fit variants:  A committed (train 1/2/4/5)   B extremes (train 1/5 only)
  eval sets:     directional (1/2 vs 4/5)      expanded (1/2/3 vs 4/5)

    uv run python -m eval.scripts.build_eval.e1_veracity_bands \
        -i eval/data/urn_runs/e1_ctx/results-00.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.scripts.build_eval import fit_urn
from eval.scripts.build_eval.fit_urn import (
    FLAG_TO_VOICE, K_FOLDS, auc, fit, fold_of, recall_at_fpr, score)

# read, never pasted (this was the fifth copy of the operating point in the repo).
# The scores drawn here are THREE-VOICE, so the line must be the three-voice
# operating point: headline_metrics.json has been the shipped 7-flag fit since
# 2026-09-14 and its threshold is on a different scale. The clustered sibling
# stores the nested choice per fold; the line is their mean, as fit_urn means it.
E1T = None   # loaded by _load_e1t() from main(); never read at import time


def _load_e1t() -> None:
    global E1T
    m = json.loads(fit_urn.E1_METRICS_CLUSTERED.read_text())["overall"]
    E1T = sum(m["thresholds_by_fold"]) / len(m["thresholds_by_fold"])


LABELS = [(1, "false"), (2, "mostly\nfalse"), (3, "mixed"),
          (4, "mostly\ntrue"), (5, "true")]
COLORS = ["#a05c46", "#bd8873", "#8a9ba5", "#8fb0c2", "#2f5468"]
MEDIAN = ["white", "0.05", "0.05", "0.05", "white"]
DPI = 150
plt.rcParams.update({
    "font.size": 10.5, "axes.labelsize": 10.5, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.edgecolor": "0.35", "axes.linewidth": 0.8,
})


def load_all(path: Path) -> list[dict]:
    """Like fit_urn.load_headline but KEEPS every veracity label (incl. mixed).

    Media-provenance claims are excluded here too (Daniel 2026-08-20) so the band
    figure describes the same corpus as the headline."""
    axis = fit_urn.load_judged_axis()
    out = []
    for line in path.open():
        r = json.loads(line)
        v = r.get("veracity")
        if v not in (1, 2, 3, 4, 5):
            continue
        if axis.get(r["review_url"]) == fit_urn.MEDIA_AXIS:
            continue
        counts = collections.Counter()
        for d in r.get("results") or []:
            voice = FLAG_TO_VOICE.get((d.get("read") or {}).get("direction"))
            if voice:
                counts[voice] += 1
        returned = sum(counts.values())
        if not returned:
            continue
        out.append({
            "v": int(v),
            "n_t": counts["supports"], "n_f": counts["refutes"],
            "n_e": max(0, returned - counts["supports"] - counts["refutes"]),
            "fold": fold_of(r["review_url"], K_FOLDS),
        })
    return out


def oof_scores(rows: list[dict], train_labels: set[int]) -> None:
    """Attach r['score']: out-of-fold, weights fitted on train_labels only."""
    for k in range(K_FOLDS):
        train = [dict(r, y=1 if r["v"] >= 4 else 0)
                 for r in rows if r["fold"] != k and r["v"] in train_labels]
        w = fit(train)
        for r in rows:
            if r["fold"] == k:
                r["score"] = score(r, w)


def report(rows: list[dict], eval_labels: dict[int, int], name: str) -> None:
    pairs = [(r["score"], eval_labels[r["v"]]) for r in rows if r["v"] in eval_labels]
    n_t = sum(y for _, y in pairs)
    rec, fpr, thr = recall_at_fpr(pairs, 0.02)
    print(f"  {name:44s} n={len(pairs):4d} (T {n_t} / F {len(pairs)-n_t})  "
          f"AUC {auc(pairs):.3f}  recall@FPR<=2% {rec:.3f} (FPR {fpr:.3f}, thr {thr:.2f})")


def main() -> None:
    _load_e1t()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=Path)
    args = ap.parse_args()
    rows = load_all(args.input)
    directional = {1: 0, 2: 0, 4: 1, 5: 1}
    expanded = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}

    print("fit A — committed (train on 1/2/4/5):")
    oof_scores(rows, {1, 2, 4, 5})
    report(rows, directional, "eval directional (1/2 vs 4/5)")
    report(rows, expanded, "eval expanded (1/2/3 vs 4/5)")

    import statistics as st
    print("\nper-label score stats (fit A, out of fold):")
    for v, name in LABELS:
        xs = [r["score"] for r in rows if r["v"] == v]
        below = sum(1 for x in xs if x <= E1T)
        print(f"  {name.replace(chr(10), ' '):13s} n={len(xs):4d}  "
              f"median {st.median(xs):+6.2f}  mean {st.mean(xs):+6.2f}  "
              f"below E1T {below/len(xs):6.1%}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    data = [[r["score"] for r in rows if r["v"] == v] for v, _ in LABELS]
    ticks = [f"{name}\nn={len(d)}" for (_, name), d in zip(LABELS, data)]
    bp = ax.boxplot(data, labels=ticks, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(linewidth=1.4))
    for patch, med, c, mc in zip(bp["boxes"], bp["medians"], COLORS, MEDIAN):
        patch.set_facecolor(c)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)
        med.set_color(mc)
    ax.axhline(E1T, color="0.10", linestyle="--", linewidth=1.1)
    ax.text(1.02, E1T, "nudge\nthreshold", transform=ax.get_yaxis_transform(),
            fontsize=9, color="0.25", va="center", ha="left")
    ax.set_ylabel("claim score, out of fold  (log-odds)")
    ax.set_xlabel("gold fact-check label", labelpad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    outdir = args.input.parent / "figures"
    fig.savefig(outdir / "e1_score_by_veracity.png", dpi=DPI, bbox_inches="tight")
    print(f"\n  wrote {outdir / 'e1_score_by_veracity.png'}")

    print("\nfit B — extremes only (train on 1/5):")
    oof_scores(rows, {1, 5})
    report(rows, directional, "eval directional (1/2 vs 4/5)")
    report(rows, expanded, "eval expanded (1/2/3 vs 4/5)")


if __name__ == "__main__":
    main()
