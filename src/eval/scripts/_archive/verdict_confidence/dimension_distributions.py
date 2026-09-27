"""Distribution + calibration read on the Likert verdict-confidence dimensions.

Answers, from a run's verdict data, the core question **does lower confidence predict
lower accuracy?** plus two diagnostics: (1) are the 1-5 dims *well-used* (full scale, per
the prompt's "use 2 and 4 deliberately"), and (2) are the three confidence dims
*non-redundant* (Spearman; |rho|>~0.9 => the decomposition is cosmetic).

Two input shapes, auto-detected:
  - PANEL  (agreement.parquet): per-model arrays `{dim}_by_model` for all four dims +
    `deployment_binary`/`gold`/`deployment_veracity`. Correctness = deployment_binary==gold.
    The only file with all four dims => the multi-dimensional read.
  - SINGLE (flag_eval.parquet): scalar `veracity` + ready-made `bare_correct`/`gold`. One
    signal, high power => the powered veracity-vs-accuracy read.

Metric formulas (AURC risk-coverage, equal-mass binning) mirror `score.py`; we re-implement
the few small helpers here so the script stays standalone (no embedding/model imports).

Confidence orientation: every signal is oriented so HIGHER = MORE CONFIDENT, then we ask
whether higher confidence => higher P(verdict correct). `veracity` itself is a DIRECTION
(1 false .. 5 true), so its confidence proxy is the magnitude |veracity-3|; cross-model
disagreement enters as -std (low spread = confident).

Run (from src/):
  uv run python -m eval.scripts.verdict_confidence.dimension_distributions
  uv run python -m eval.scripts.verdict_confidence.dimension_distributions --files eval/scripts/verdict_confidence/data/agreement.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

CONF_DIMS = ("evidence_sufficiency", "evidence_agreement", "source_reliability")
ALL_DIMS = ("veracity",) + CONF_DIMS
LEVELS = (1, 2, 3, 4, 5)
DEFAULT_FILES = (
    "eval/scripts/verdict_confidence/data/agreement.parquet",
    "eval/scripts/verdict_confidence/data/flag_eval.parquet",
)


# --- metric helpers (formulas mirror score.py) -------------------------------------------

def _aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower = better). Abstain low-confidence first.
    Random-order AURC ~= the marginal error rate, so AURC < base-err == useful signal."""
    n = len(conf)
    if n == 0:
        return float("nan")
    order = np.argsort(-conf)  # most confident first
    err = (1 - correct[order]).cumsum() / np.arange(1, n + 1)
    return float(err.mean())


def _auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of confidence ranking correctness (>0.5 => confident items more often correct)."""
    if len(np.unique(correct)) < 2:
        return float("nan")
    return float(roc_auc_score(correct, conf))


def _aucpr(conf: np.ndarray, correct: np.ndarray) -> float:
    """AUC-PR for the 'correct' class; baseline = base rate of correct."""
    if len(np.unique(correct)) < 2:
        return float("nan")
    return float(average_precision_score(correct, conf))


def _pb(conf: np.ndarray, correct: np.ndarray) -> float:
    if len(np.unique(correct)) < 2 or len(np.unique(conf)) < 2:
        return float("nan")
    return float(pointbiserialr(correct, conf).correlation)


def _acc_by_equal_mass(conf: np.ndarray, correct: np.ndarray, n_bins: int = 3) -> list[dict]:
    """Split into equal-mass bins of ascending confidence; report mean conf + accuracy.
    Equal-mass (quantile) binning handles the heavy Likert ties gracefully (cf. score.py ECE)."""
    n = len(conf)
    n_bins = max(1, min(n_bins, n))
    order = np.argsort(conf)  # ascending confidence
    out = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        out.append({"n": len(chunk), "conf": float(conf[chunk].mean()),
                    "acc": float(correct[chunk].mean())})
    return out


def _fmt_signal_row(name: str, conf: np.ndarray, correct: np.ndarray) -> str:
    return (f"  {name:<26} pb={_pb(conf, correct):+.3f}  AUROC={_auroc(conf, correct):.3f}  "
            f"AUC-PR={_aucpr(conf, correct):.3f}  AURC={_aurc(conf, correct):.3f}")


# --- distribution / scale-usage -----------------------------------------------------------

def _level_fracs(vals: np.ndarray) -> dict[int, float]:
    n = len(vals)
    return {lv: (float((vals == lv).mean()) if n else float("nan")) for lv in LEVELS}


def _print_dist_row(name: str, vals: np.ndarray) -> None:
    f = _level_fracs(vals)
    cells = " ".join(f"{lv}:{100 * f[lv]:5.1f}%" for lv in LEVELS)
    mid = 100 * np.isin(vals, [2, 4]).mean()
    flag = "  <-- COLLAPSED to {1,3,5}" if mid < 10 else ""
    print(f"  {name:<24} (n={len(vals):>4})  {cells}   mean={vals.mean():.2f}  mid(2&4)={mid:4.1f}%{flag}")


# --- redundancy ---------------------------------------------------------------------------

def _print_redundancy(means: dict[str, np.ndarray], source: str) -> None:
    print(f"\n### Redundancy — Spearman among confidence dims ({source}); |rho|>~0.9 => cosmetic")
    for i in range(len(CONF_DIMS)):
        for j in range(i + 1, len(CONF_DIMS)):
            rho = spearmanr(means[CONF_DIMS[i]], means[CONF_DIMS[j]]).correlation
            flag = "  <-- REDUNDANT" if (not np.isnan(rho) and abs(rho) > 0.9) else ""
            print(f"  {CONF_DIMS[i]:>22} ~ {CONF_DIMS[j]:<22} rho={rho:+.3f}{flag}")
    print("  -- each confidence dim vs veracity (direction, not confidence) --")
    for d in CONF_DIMS:
        rho = spearmanr(means["veracity"], means[d]).correlation
        print(f"  {'veracity':>22} ~ {d:<22} rho={rho:+.3f}")


# --- confident-error asymmetry ------------------------------------------------------------

def _print_confident_error(veracity: np.ndarray, correct: np.ndarray, gold: list[str]) -> None:
    print("\n### Confident-error asymmetry (where do confident verdicts fail?)")
    err_by_level = {lv: int(((veracity == lv) & (correct == 0)).sum()) for lv in LEVELS}
    tot = int((correct == 0).sum())
    print(f"  errors by veracity level: {err_by_level}   (total errors={tot})")
    if tot:
        print(f"  share at v=5 (confident-true): {100 * err_by_level[5] / tot:.0f}%   "
              f"at v=1 (confident-false): {100 * err_by_level[1] / tot:.0f}%")
    for lv, lab in [(5, "confident-TRUE  (v=5)"), (1, "confident-FALSE (v=1)")]:
        m = veracity == lv
        if m.sum():
            errs = [gold[k] for k in range(len(gold)) if veracity[k] == lv and correct[k] == 0]
            breakdown = {g: errs.count(g) for g in sorted(set(errs))}
            print(f"  {lab}: n={int(m.sum()):<4} err_rate={1 - correct[m].mean():.3f}  "
                  f"errors_by_gold={breakdown or '{}'}")


# --- PANEL mode (agreement.parquet) -------------------------------------------------------

def run_panel(df: pl.DataFrame) -> None:
    df = df.filter(pl.col("error").is_null()) if "error" in df.columns else df
    n = df.height
    models = df["model_order"][0].to_list() if "model_order" in df.columns else []
    print(f"\n{'=' * 78}\nPANEL mode  n={n}  models={models}")

    by_model = {d: df[f"{d}_by_model"].to_list() for d in ALL_DIMS}
    means = {d: np.array([np.mean(x) for x in by_model[d]]) for d in ALL_DIMS}
    ver_std = np.array([np.std(x) for x in by_model["veracity"]])
    dep_ver = df["deployment_veracity"].to_numpy()
    correct = np.array([int(b == g) for b, g in zip(df["deployment_binary"], df["gold"])])
    gold = df["gold"].to_list()
    print(f"deployment accuracy (deployment_binary==gold): {correct.mean():.3f}  "
          f"(gold pass={gold.count('pass')}/{n})")

    # 1. distributions / scale usage (pooled over the 4 models => 4*n obs per dim)
    print("\n### 1. Per-dimension scale usage (pooled across models)")
    for d in ALL_DIMS:
        _print_dist_row(d, np.array([x for row in by_model[d] for x in row]))
    _print_dist_row("deployment_veracity", dep_ver)

    print("\n  cross-model spread per dim (how much the panel diverges):")
    for d in ALL_DIMS:
        sp = np.array([np.std(x) for x in by_model[d]])
        rg = np.array([np.ptp(x) for x in by_model[d]])
        print(f"    {d:<22} mean_std={sp.mean():.2f}  %unanimous={100 * (sp == 0).mean():3.0f}%  "
              f"%range>=2={100 * (rg >= 2).mean():3.0f}%")

    # 2. redundancy
    _print_redundancy(means, "per-model means")

    # 3. calibration / discrimination — the core question
    base_err = 1 - correct.mean()
    print(f"\n### 3. Does confidence predict correctness?  base-rate correct={correct.mean():.3f} "
          f"(random-order AURC~={base_err:.3f})")
    signals = {
        "|deployment_ver-3|": np.abs(dep_ver - 3).astype(float),
        "|panel_mean_ver-3|": np.abs(means["veracity"] - 3),
        "cross_model_agreement(-std)": -ver_std,
        "evidence_sufficiency_mean": means["evidence_sufficiency"],
        "evidence_agreement_mean": means["evidence_agreement"],
        "source_reliability_mean": means["source_reliability"],
        "weakest_link_min(3 dims)": np.minimum.reduce(
            [means[d] for d in CONF_DIMS]),
    }
    print("  signal (higher = more confident)        corr / discrimination / selective")
    for name in sorted(signals, key=lambda k: _aurc(signals[k], correct)):
        print(_fmt_signal_row(name, signals[name], correct))

    print("\n  accuracy by confidence tertile (ascending) for the two strongest + weakest signals:")
    for name in ("cross_model_agreement(-std)", "evidence_sufficiency_mean",
                 "source_reliability_mean", "|deployment_ver-3|"):
        bins = _acc_by_equal_mass(signals[name], correct, n_bins=3)
        cells = "  ".join(f"[conf~{b['conf']:+.2f} n={b['n']}: acc={b['acc']:.2f}]" for b in bins)
        print(f"    {name:<28} {cells}")

    # 4. confident-error asymmetry (deployment veracity)
    _print_confident_error(dep_ver, correct, gold)


# --- SINGLE mode (flag_eval.parquet) ------------------------------------------------------

def run_single(df: pl.DataFrame) -> None:
    df = df.filter(pl.col("error").is_null()) if "error" in df.columns else df
    n = df.height
    ver = df["veracity"].to_numpy()
    correct = df["bare_correct"].to_numpy() if "bare_correct" in df.columns else \
        np.array([int(p == g) for p, g in zip(df["bare_pred"], df["gold"])])
    gold = df["gold"].to_list()
    print(f"\n{'=' * 78}\nSINGLE mode  n={n}  bare accuracy={correct.mean():.3f}  "
          f"(gold pass={gold.count('pass')}/{n})")

    # 1. distribution / scale usage
    print("\n### 1. veracity scale usage")
    _print_dist_row("veracity", ver)
    print(f"  level-3 (neutral) usage: {100 * (ver == 3).mean():.1f}%  "
          f"poles {{1,5}}: {100 * np.isin(ver, [1, 5]).mean():.1f}%")

    # 3. calibration — powered single-signal read
    base_err = 1 - correct.mean()
    print(f"\n### 3. Does confidence predict correctness?  base-rate correct={correct.mean():.3f} "
          f"(random-order AURC~={base_err:.3f})")
    print("  accuracy by veracity LEVEL (raw direction):")
    for lv in LEVELS:
        m = ver == lv
        if m.sum():
            pred = "pass" if lv >= 4 else "flag"
            print(f"    v={lv}  n={int(m.sum()):<5} acc={correct[m].mean():.3f}  (bare pred={pred})")
    print("  accuracy by |veracity-3| (confidence magnitude):")
    mag = np.abs(ver - 3).astype(float)
    for mv in (0, 1, 2):
        m = mag == mv
        if m.sum():
            print(f"    |v-3|={mv}  n={int(m.sum()):<5} acc={correct[m].mean():.3f}")
    print("\n  magnitude signal (higher = more confident):")
    print(_fmt_signal_row("|veracity-3|", mag, correct))
    print(f"  (raw veracity vs correct: pb={_pb(ver.astype(float), correct):+.3f} "
          f"— negative => higher 'true' calls are LESS often correct)")
    print("  risk by coverage, abstain low |v-3| first:")
    order = np.argsort(-mag)
    for cov in (0.25, 0.50, 0.75, 1.0):
        k = max(1, int(cov * n))
        print(f"    coverage {cov:>4.0%}: risk(err)={1 - correct[order[:k]].mean():.3f}  (n={k})")

    # 4. confident-error asymmetry
    _print_confident_error(ver, correct, gold)


# --- dispatch -----------------------------------------------------------------------------

def detect_mode(df: pl.DataFrame) -> str:
    if "veracity_by_model" in df.columns:
        return "panel"
    if "veracity" in df.columns:
        return "single"
    raise ValueError("unrecognised schema: need 'veracity_by_model' (panel) or 'veracity' (single)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", default=list(DEFAULT_FILES),
                    help="parquet file(s); panel (agreement) or single (flag_eval) shape, auto-detected")
    args = ap.parse_args()

    for f in args.files:
        path = Path(f)
        df = pl.read_parquet(path)
        print(f"\n\n{'#' * 78}\n# {path.name}  ({df.height} rows)\n{'#' * 78}")
        mode = detect_mode(df)
        (run_panel if mode == "panel" else run_single)(df)


if __name__ == "__main__":
    main()
