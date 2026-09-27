"""
Plot evaluation results from docs/eval_results.csv and docs/eval_claims.csv.

Usage:
    uv run python3 docs/plot_results.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_CSV = Path(__file__).parent / "eval_results.csv"
CLAIMS_CSV  = Path(__file__).parent / "eval_claims.csv"

LABELS = ["Refuted", "Supported", "Not Enough Evidence", "Conflicting Evidence"]
LABEL_COLS = ["acc_refuted", "acc_supported", "acc_nei", "acc_conflicting"]


def ece(confidences: Sequence[float], corrects: Sequence[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted average gap between mean confidence
    and accuracy across `n_bins` equal-width confidence buckets in [0, 1]."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(corrects, dtype=float)
    if conf.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # bin index 0..n_bins-1; clip so confidence==1.0 lands in last bin
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    total = conf.size
    score = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        score += (mask.sum() / total) * abs(conf[mask].mean() - corr[mask].mean())
    return float(score)


def roc_curve(confidences: Sequence[float], corrects: Sequence[bool]):
    """Return (fpr, tpr) arrays for an ROC curve. `corrects=True` is the positive class.
    Ties on confidence are collapsed to a single point (sklearn convention)."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(corrects, dtype=bool)
    if conf.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-conf, kind="mergesort")
    conf = conf[order]
    corr = corr[order]
    P = int(corr.sum())
    N = int((~corr).sum())
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tps = np.cumsum(corr.astype(int))
    fps = np.cumsum((~corr).astype(int))
    # collapse runs of equal confidence to the last index in each run
    distinct = np.r_[np.diff(conf) != 0, True]
    tps = tps[distinct]
    fps = fps[distinct]
    tpr = np.r_[0.0, tps / P]
    fpr = np.r_[0.0, fps / N]
    return fpr, tpr


def auroc(confidences: Sequence[float], corrects: Sequence[bool]) -> float:
    """Area under the ROC curve via trapezoid rule. Returns 0.5 on degenerate input
    (empty, all-correct, or all-incorrect — no curve is defined)."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(corrects, dtype=bool)
    if conf.size == 0 or corr.sum() == 0 or (~corr).sum() == 0:
        return 0.5
    fpr, tpr = roc_curve(conf, corr)
    return float(np.trapezoid(tpr, fpr))


def plot_roc_curve(
    claims_df: pd.DataFrame,
    runs_df: pd.DataFrame | None = None,
    out_path: Path | None = None,
) -> None:
    """One ROC curve per approach. Joins claims_df with runs_df on run_id to
    recover the approach label; falls back to grouping all claims into one curve."""
    if runs_df is not None and "approach" in runs_df.columns:
        df = claims_df.merge(runs_df[["run_id", "approach"]], on="run_id", how="left")
        df["approach"] = df["approach"].fillna("unknown")
    else:
        df = claims_df.assign(approach="all")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")
    for approach, grp in df.groupby("approach"):
        confs = grp["post_confidence"].astype(float).to_numpy()
        corrs = grp["correct"].astype(bool).to_numpy()
        fpr, tpr = roc_curve(confs, corrs)
        auc = auroc(confs, corrs)
        ax.plot(fpr, tpr, marker=".", linewidth=1.5,
                label=f"{approach}  (AUC={auc:.3f}, n={len(grp)})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve — confidence vs. correctness")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if out_path is None:
        out_path = Path(__file__).parent / "roc_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(Path.cwd()) if out_path.is_absolute() else out_path}")


def plot_reliability_diagram(
    confidences: Sequence[float],
    corrects: Sequence[bool],
    n_bins: int = 10,
    out_path: Path | None = None,
) -> None:
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(corrects, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = 1.0 / n_bins
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)

    bin_acc = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        bin_count[b] = mask.sum()
        if mask.any():
            bin_acc[b] = corr[mask].mean()

    ece_val = ece(conf, corr, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    has_data = ~np.isnan(bin_acc)
    ax.bar(centers[has_data], bin_acc[has_data], width=width * 0.95,
           color="#4c72b0", edgecolor="black", label="Accuracy per bin")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    for b in range(n_bins):
        if bin_count[b] > 0:
            ax.text(centers[b], bin_acc[b] + 0.02, str(bin_count[b]),
                    ha="center", fontsize=7, color="#333")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram  (ECE = {ece_val:.4f}, n = {conf.size})")
    ax.legend(loc="upper left")
    plt.tight_layout()
    if out_path is None:
        out_path = Path(__file__).parent / "reliability_diagram.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(Path.cwd()) if out_path.is_absolute() else out_path}  (ECE={ece_val:.4f})")


def plot_accuracy_by_approach(df: pd.DataFrame) -> None:
    grouped = df.groupby("approach")["accuracy"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(grouped["approach"], grouped["accuracy"], color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall accuracy by approach (mean across runs)")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "accuracy_by_approach.png", dpi=150)
    print("Saved: docs/accuracy_by_approach.png")


def plot_per_label_accuracy(df: pd.DataFrame) -> None:
    latest = df.sort_values("timestamp").groupby("approach").last().reset_index()
    x = range(len(LABELS))
    width = 0.8 / max(len(latest), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, row in latest.iterrows():
        vals = [row[c] for c in LABEL_COLS]
        offsets = [xi + (i - len(latest) / 2) * width for xi in x]
        bars = ax.bar(offsets, vals, width=width * 0.9, label=row["approach"])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-label accuracy by approach (latest run each)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "per_label_accuracy.png", dpi=150)
    print("Saved: docs/per_label_accuracy.png")


def plot_accuracy_over_runs(df: pd.DataFrame) -> None:
    df = df.sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(10, 4))
    for approach, grp in df.groupby("approach"):
        ax.plot(range(len(grp)), grp["accuracy"], marker="o", label=approach)
        for j, (_, row) in enumerate(grp.iterrows()):
            ax.annotate(f"n={row['n_claims']}", (j, row["accuracy"]),
                        textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy over time by approach")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "accuracy_over_runs.png", dpi=150)
    print("Saved: docs/accuracy_over_runs.png")


def _self_test_ece() -> None:
    """Quick sanity checks on dummy data."""
    # Perfectly calibrated: 50% confidence, 50% correct → ECE ≈ 0
    conf = [0.5] * 100
    corr = [True] * 50 + [False] * 50
    assert ece(conf, corr, n_bins=10) < 1e-9, "perfectly calibrated case failed"
    # Overconfident: 100% confidence, 0% correct → ECE = 1.0
    assert abs(ece([1.0] * 10, [False] * 10, n_bins=10) - 1.0) < 1e-9
    # Empty
    assert ece([], [], n_bins=10) == 0.0
    print("ece() self-test passed")


def _self_test_auroc() -> None:
    """Quick sanity checks on dummy data."""
    # Perfect ranker: every correct has higher confidence than every incorrect → AUC = 1.0
    conf = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    corr = [True, True, True, False, False, False]
    assert abs(auroc(conf, corr) - 1.0) < 1e-9, "perfect ranker failed"
    # Worst possible ranker: AUC = 0.0
    assert abs(auroc(conf, [False, False, False, True, True, True])) < 1e-9
    # Uninformative (all same confidence) → AUC = 0.5
    assert abs(auroc([0.5] * 6, [True, True, True, False, False, False]) - 0.5) < 1e-9
    # Degenerate cases
    assert auroc([], []) == 0.5
    assert auroc([0.5, 0.6], [True, True]) == 0.5
    print("auroc() self-test passed")


def main() -> None:
    _self_test_ece()
    _self_test_auroc()

    runs_df = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else None

    if CLAIMS_CSV.exists():
        claims = pd.read_csv(CLAIMS_CSV)
        claims = claims.dropna(subset=["post_confidence", "correct"])
        if len(claims):
            confidences = claims["post_confidence"].astype(float).tolist()
            corrects = claims["correct"].astype(bool).tolist()
            plot_reliability_diagram(confidences, corrects, n_bins=10)
            plot_roc_curve(claims, runs_df)
        else:
            print(f"No claim-level data in {CLAIMS_CSV}, skipping reliability/ROC")
    else:
        print(f"No claims file at {CLAIMS_CSV}, skipping reliability/ROC")

    if runs_df is None:
        print(f"No results file found at {RESULTS_CSV}")
        return

    df = runs_df
    print(f"Loaded {len(df)} run(s) from {RESULTS_CSV}")
    cols = ["run_id", "approach", "n_claims", "seed", "accuracy"]
    if "ece_score" in df.columns:
        cols.append("ece_score")
    if "auroc_score" in df.columns:
        cols.append("auroc_score")
    print(df[cols].to_string(index=False))

    plot_accuracy_by_approach(df)
    plot_per_label_accuracy(df)
    plot_accuracy_over_runs(df)


if __name__ == "__main__":
    main()
