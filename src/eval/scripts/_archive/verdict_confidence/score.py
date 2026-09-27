"""Step 3 — scorer: confidence signals (Part A) + calibration metrics (Part B).

Reads resamples.parquet (from resample_run.py) and, per claim, derives:
  - the binary prediction (majority vote: pass iff >half the K samples have veracity>=4),
  - Part-A confidence signals (predictive-entropy confidences, vote agreement, Likert
    mean/min/stability, semantic consistency of the justifications),
then evaluates them (Part B): AUC-PR for PASS discrimination, class-wise ECE (equal-mass
bins), balanced Brier, and AURC (risk-coverage) per signal — the AURC ranking answers "which
signal best predicts correctness". Plus the inter-dim Spearman redundancy check.

Everything is reported for the four slices: {without, with} synthetic × {incl, excl} artifact,
headline = text-only. See docs/confidence_metrics.md for the what/why/how of each number.

Run (from src/):
  uv run python -m eval.scripts.verdict_confidence.score
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from pipeline.embedding import embed

IN = Path("eval/scripts/verdict_confidence/data/resamples.parquet")
OUT = Path("eval/scripts/verdict_confidence/data/score_report.json")
CONF_DIMS = ("evidence_sufficiency", "evidence_agreement", "source_reliability")
SEMANTIC_TAU = 0.9  # cosine >= tau => same meaning cluster (paraphrase-MiniLM)
SIGNALS = ("veracity_conf", "agreement", "likert_mean", "likert_min",
           "likert_stability", "joint_conf", "semantic_consistency")


# --- Part-A per-claim signal helpers ------------------------------------------------------

def _entropy_conf(samples: list[int], levels: int = 5) -> float:
    """1 - normalized predictive entropy over the 1..levels histogram (1 = unanimous)."""
    counts = np.bincount(np.array(samples) - 1, minlength=levels).astype(float)
    p = counts[counts > 0] / counts.sum()
    h = -(p * np.log(p)).sum()
    return float(1 - h / np.log(levels))


def _semantic_consistency(justifs: list[str]) -> float:
    """1 - normalized entropy over meaning-clusters of the K justifications (discrete semantic
    entropy; no logprobs). Greedy cosine clustering against each cluster's first member."""
    k = len(justifs)
    if k <= 1:
        return 1.0
    vecs = [embed(j) for j in justifs]
    reps: list[int] = []  # one representative index per cluster
    sizes: list[int] = []
    for vi, v in enumerate(vecs):
        for ci, ri in enumerate(reps):
            if float(np.dot(v, vecs[ri])) >= SEMANTIC_TAU:
                sizes[ci] += 1
                break
        else:
            reps.append(vi)
            sizes.append(1)
    p = np.array(sizes, dtype=float) / k
    h = -(p * np.log(p)).sum()
    return float(1 - h / np.log(k))


def per_claim_signals(df: pl.DataFrame) -> pl.DataFrame:
    """Add prediction + correctness + the 7 candidate confidence signals to each claim row."""
    rows = []
    for r in df.to_dicts():
        ver = r["veracity_samples"]
        k = len(ver)
        pass_frac = float(np.mean([v >= 4 for v in ver]))
        pred = "pass" if pass_frac > 0.5 else "flag"
        dim_means = {d: float(np.mean(r[f"{d}_samples"])) for d in CONF_DIMS}
        dim_stds = {d: float(np.std(r[f"{d}_samples"])) for d in CONF_DIMS}
        rows.append({
            "claim_id": r["claim_id"], "gold": r["gold_binary_label"],
            "modality": r["modality"], "synthetic": r["synthetic"],
            "pred": pred, "correct": int(pred == r["gold_binary_label"]),
            "p_pass": pass_frac,
            **{f"mean_{d}": dim_means[d] for d in CONF_DIMS},
            # --- candidate confidence signals (all in [0,1], read as P(verdict correct)) ---
            "veracity_conf": _entropy_conf(ver),
            "agreement": max(pass_frac, 1 - pass_frac),
            "likert_mean": (np.mean(list(dim_means.values())) - 1) / 4,
            "likert_min": (min(dim_means.values()) - 1) / 4,
            "likert_stability": 1 - np.mean(list(dim_stds.values())) / 2,  # max std on 1-5 is 2
            "joint_conf": float(np.mean([_entropy_conf(r[f"{d}_samples"]) for d in CONF_DIMS])),
            "semantic_consistency": _semantic_consistency(r["justification_samples"]),
        })
    return pl.DataFrame(rows)


# --- Part-B evaluation-metric helpers -----------------------------------------------------

def _ece_equal_mass(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Equal-mass (quantile-binned) ECE: size-weighted |mean conf - accuracy| per bin."""
    n = len(conf)
    if n == 0:
        return float("nan")
    n_bins = max(1, min(n_bins, n))
    order = np.argsort(conf)
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        ece += len(chunk) / n * abs(conf[chunk].mean() - correct[chunk].mean())
    return float(ece)


def _classwise_ece(s: pl.DataFrame, signal: str) -> dict:
    out = {}
    for cls in ("pass", "flag"):
        sub = s.filter(pl.col("pred") == cls)
        out[cls] = (_ece_equal_mass(sub[signal].to_numpy(), sub["correct"].to_numpy())
                    if sub.height else float("nan"))
    vals = [v for v in out.values() if not np.isnan(v)]
    out["macro"] = float(np.mean(vals)) if vals else float("nan")
    return out


def _balanced_brier(s: pl.DataFrame, signal: str) -> float:
    briers = []
    for cls in ("pass", "flag"):
        sub = s.filter(pl.col("pred") == cls)
        if sub.height:
            briers.append(float(((sub[signal].to_numpy() - sub["correct"].to_numpy()) ** 2).mean()))
    return float(np.mean(briers)) if briers else float("nan")


def _aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower = better). Abstain low-confidence first."""
    n = len(conf)
    if n == 0:
        return float("nan")
    order = np.argsort(-conf)  # most confident first
    err = (1 - correct[order]).cumsum() / np.arange(1, n + 1)  # risk at each coverage
    return float(err.mean())


def evaluate_slice(s: pl.DataFrame) -> dict:
    """Verdict-quality + per-signal calibration for one slice."""
    y = (s["gold"] == "pass").to_numpy().astype(int)
    pred = (s["pred"] == "pass").to_numpy().astype(int)
    res = {"n": s.height, "n_pass_gold": int(y.sum()), "n_flag_gold": int((1 - y).sum()),
           "accuracy": float((s["correct"].to_numpy()).mean())}
    # PASS-class discrimination (rare class)
    if y.sum() and y.sum() < len(y):
        res["auc_pr_pass"] = float(average_precision_score(y, s["p_pass"].to_numpy()))
        p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=[1], zero_division=0)
        res["pass_precision"], res["pass_recall"], res["pass_f1"] = float(p[0]), float(r[0]), float(f1[0])
    else:
        res["auc_pr_pass"] = float("nan")  # single-class slice — undefined
    res["signals"] = {
        sig: {"ece": _classwise_ece(s, sig), "balanced_brier": _balanced_brier(s, sig),
              "aurc": _aurc(s[sig].to_numpy(), s["correct"].to_numpy())}
        for sig in SIGNALS
    }
    return res


def _print_slice(name: str, res: dict) -> None:
    print(f"\n### {name}")
    print(f"  N={res['n']}  gold pass/flag={res['n_pass_gold']}/{res['n_flag_gold']}  "
          f"acc={res['accuracy']:.3f}  AUC-PR(pass)={res['auc_pr_pass']:.3f}")
    if "pass_f1" in res:
        print(f"  PASS  P={res['pass_precision']:.3f} R={res['pass_recall']:.3f} F1={res['pass_f1']:.3f}")
    print(f"  {'signal':<22} {'ECE_pass':>9} {'ECE_flag':>9} {'ECE_macro':>10} {'bBrier':>8} {'AURC':>7}")
    for sig in sorted(SIGNALS, key=lambda x: (np.isnan(res['signals'][x]['aurc']), res['signals'][x]['aurc'])):
        m = res["signals"][sig]
        print(f"  {sig:<22} {m['ece']['pass']:>9.3f} {m['ece']['flag']:>9.3f} "
              f"{m['ece']['macro']:>10.3f} {m['balanced_brier']:>8.3f} {m['aurc']:>7.3f}")


def main() -> None:
    df = pl.read_parquet(IN).filter(pl.col("error").is_null() & pl.col("gold_binary_label").is_not_null())
    print(f"loaded {df.height} scored claims from {IN}")
    sig = per_claim_signals(df)

    slices = {
        "WITHOUT synthetic · INCL artifact": sig.filter(~pl.col("synthetic")),
        "WITHOUT synthetic · EXCL artifact (headline)": sig.filter(~pl.col("synthetic") & (pl.col("modality") == "text")),
        "WITH synthetic · INCL artifact": sig,
        "WITH synthetic · EXCL artifact": sig.filter(pl.col("modality") == "text"),
    }
    report = {}
    for name, s in slices.items():
        if s.height == 0:
            continue
        res = evaluate_slice(s)
        report[name] = res
        _print_slice(name, res)

    # --- Inter-dimension Spearman redundancy check (A5), on the full text set ---
    print("\n### Inter-dimension Spearman (redundancy check; |rho|>0.9 => collapse a dim)")
    txt = sig.filter(pl.col("modality") == "text")
    mat = np.column_stack([txt[f"mean_{d}"].to_numpy() for d in CONF_DIMS])
    spear = {}
    for i in range(len(CONF_DIMS)):
        for j in range(i + 1, len(CONF_DIMS)):
            rho = float(spearmanr(mat[:, i], mat[:, j]).correlation) if txt.height > 2 else float("nan")
            spear[f"{CONF_DIMS[i]} ~ {CONF_DIMS[j]}"] = rho
            flag = "  <-- REDUNDANT" if (not np.isnan(rho) and abs(rho) > 0.9) else ""
            print(f"  {CONF_DIMS[i]:>22} ~ {CONF_DIMS[j]:<22} rho={rho:+.3f}{flag}")

    OUT.write_text(json.dumps({"slices": report, "interdim_spearman": spear}, indent=1, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
