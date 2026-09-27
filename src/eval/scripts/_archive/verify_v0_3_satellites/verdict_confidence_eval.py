"""WS4 — confidence-dimension evaluation for the Stage-3 verifier.

Joins a claims_verdict file (gold) with a verdicts parquet (verify_run output) on claim_id and runs
two reads on the three confidence dims (evidence_sufficiency, evidence_agreement, source_reliability),
porting `_archive/verdict_confidence/dimension_distributions.py` and adding the construct-validity test:

  PREDICTIVENESS — does a higher dim predict a more-correct verdict? Correctness = |pred-gold|<=1
    (off-by-one on the 1-5 scale). Per-dim AUROC / AURC / AUC-PR / point-biserial, redundancy
    (Spearman), scale-usage (collapse-to-{1,3,5} flag), and the confident-error asymmetry. (Brier
    needs calibrated probabilities/logprobs → deferred to the calibrated-confidence step.)

  CONSTRUCT VALIDITY (the stronger, non-circular test) — do the dims encode what they claim? Group by
    `rating_subtype` and test:
      H1  evidence_agreement LOWER on `mixed` than on clear_true/clear_false  (contested => conflict)
      H2  evidence_sufficiency LOWER on `unprovable` than on resolved          (no-evidence => low suff)
      H3  source_reliability shows NO separation across subtypes               (re-confirm it's a dud)
    Flat dims across subtypes ⇒ the prompt isn't giving the dims range ⇒ a WS5 target.

  uv run python -m eval.scripts.verdict_confidence_eval \
      -c .../claims_verdict_dev_bal.parquet -v .../verdicts_verdict_dev_bal.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import mannwhitneyu, pointbiserialr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

CONF_DIMS = ("evidence_sufficiency", "evidence_agreement", "source_reliability")
ALL_DIMS = ("veracity",) + CONF_DIMS
LEVELS = (1, 2, 3, 4, 5)
RESOLVED = ("clear_true", "clear_false", "mostly_true", "mostly_false")


# --- metric helpers (mirror score.py / the archived analyzer) ----------------------------

def _aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    n = len(conf)
    if n == 0:
        return float("nan")
    order = np.argsort(-conf)
    err = (1 - correct[order]).cumsum() / np.arange(1, n + 1)
    return float(err.mean())


def _auroc(conf, correct):
    return float(roc_auc_score(correct, conf)) if len(np.unique(correct)) > 1 else float("nan")


def _aucpr(conf, correct):
    return float(average_precision_score(correct, conf)) if len(np.unique(correct)) > 1 else float("nan")


def _pb(conf, correct):
    if len(np.unique(correct)) < 2 or len(np.unique(conf)) < 2:
        return float("nan")
    return float(pointbiserialr(correct, conf).correlation)


def _signal_row(name, conf, correct):
    return (f"  {name:<26} pb={_pb(conf, correct):+.3f}  AUROC={_auroc(conf, correct):.3f}  "
            f"AUC-PR={_aucpr(conf, correct):.3f}  AURC={_aurc(conf, correct):.3f}")


def _dist_row(name, vals):
    n = len(vals)
    f = {lv: (float((vals == lv).mean()) if n else float("nan")) for lv in LEVELS}
    cells = " ".join(f"{lv}:{100 * f[lv]:5.1f}%" for lv in LEVELS)
    mid = 100 * np.isin(vals, [2, 4]).mean() if n else float("nan")
    flag = "  <-- COLLAPSED to {1,3,5}" if (n and mid < 10) else ""
    print(f"  {name:<22} (n={n:>4})  {cells}  mean={vals.mean():.2f}  mid(2&4)={mid:4.1f}%{flag}")


def _acc_by_tertile(conf, correct, n_bins=3):
    n = len(conf)
    n_bins = max(1, min(n_bins, n))
    order = np.argsort(conf)
    out = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            out.append({"n": len(chunk), "conf": float(conf[chunk].mean()), "acc": float(correct[chunk].mean())})
    return out


def _mwu(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney U one-sided (a < b); return (p, rank-biserial effect size)."""
    if len(a) < 3 or len(b) < 3:
        return float("nan"), float("nan")
    u, p = mannwhitneyu(a, b, alternative="less")
    rbc = 1 - 2 * u / (len(a) * len(b))  # rank-biserial; >0 => a tends below b
    return float(p), float(rbc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--claims", type=Path, required=True)
    ap.add_argument("-v", "--verdicts", type=Path, required=True)
    ap.add_argument("--content-only", action="store_true", help="restrict to judged_axis==content")
    args = ap.parse_args()

    claims = pl.read_parquet(args.claims)
    vcols = pl.read_parquet(args.verdicts)
    keep = [c for c in ("claim_id", *ALL_DIMS, "error", "n_urls_seen") if c in vcols.columns]
    df = claims.join(vcols.select(keep), on="claim_id", how="inner")
    if "error" in df.columns:
        df = df.filter(pl.col("error").is_null())
    df = df.filter(pl.col("veracity") > 0)
    if args.content_only:
        df = df.filter(pl.col("judged_axis") == "content")
    n = df.height
    if n == 0:
        print("no scorable rows.")
        return

    ver = df["veracity"].to_numpy()
    gold = df["gold_veracity"].to_numpy()
    dims = {d: df[d].to_numpy().astype(float) for d in CONF_DIMS}
    correct = (np.abs(ver - gold) <= 1).astype(int)  # off-by-one == "verdict right enough"
    subtype = df["rating_subtype"].to_list()

    print(f"\n{'='*78}\nCONFIDENCE EVAL  n={n}  (correct = |pred_veracity - gold| <= 1)")
    print(f"verdict off-by-one accuracy: {correct.mean():.3f}   exact: {(ver==gold).mean():.3f}")

    # 1. scale usage
    print("\n### 1. Per-dimension scale usage (is the 1-5 range used, or collapsed?)")
    for d in ALL_DIMS:
        _dist_row(d, df[d].to_numpy())

    # 2. redundancy
    print("\n### 2. Redundancy — Spearman among confidence dims (|rho|>~0.9 => cosmetic)")
    for i in range(len(CONF_DIMS)):
        for j in range(i + 1, len(CONF_DIMS)):
            rho = spearmanr(dims[CONF_DIMS[i]], dims[CONF_DIMS[j]]).correlation
            flag = "  <-- REDUNDANT" if (not np.isnan(rho) and abs(rho) > 0.9) else ""
            print(f"  {CONF_DIMS[i]:>22} ~ {CONF_DIMS[j]:<22} rho={rho:+.3f}{flag}")
    for d in CONF_DIMS:
        print(f"  {'veracity':>22} ~ {d:<22} rho={spearmanr(ver, dims[d]).correlation:+.3f}")

    # 3. predictiveness
    base_err = 1 - correct.mean()
    print(f"\n### 3. Does confidence predict verdict-correctness?  base correct={correct.mean():.3f} "
          f"(random-order AURC~={base_err:.3f}; AURC below that = useful)")
    signals = {**{f"{d}": dims[d] for d in CONF_DIMS},
               "weakest_link_min(3)": np.minimum.reduce([dims[d] for d in CONF_DIMS]),
               "|veracity-3|(magnitude)": np.abs(ver - 3).astype(float)}
    for name in sorted(signals, key=lambda k: _aurc(signals[k], correct)):
        print(_signal_row(name, signals[name], correct))
    print("\n  accuracy by confidence tertile (ascending):")
    for name in ("evidence_sufficiency", "evidence_agreement", "source_reliability"):
        bins = _acc_by_tertile(signals[name], correct)
        cells = "  ".join(f"[~{b['conf']:.2f} n={b['n']}: acc={b['acc']:.2f}]" for b in bins)
        print(f"    {name:<22} {cells}")

    # confident-error asymmetry
    print("\n  confident-error asymmetry (error rate by PREDICTED veracity level):")
    for lv in LEVELS:
        m = ver == lv
        if m.sum():
            print(f"    pred v={lv}  n={int(m.sum()):<4} off-by-one acc={correct[m].mean():.3f}  "
                  f"exact={(ver[m]==gold[m]).mean():.3f}")

    # 4. construct validity
    print(f"\n{'='*78}\n### 4. CONSTRUCT VALIDITY — dims grouped by rating_subtype")
    sub_arr = np.array(subtype)
    print(f"  {'subtype':<14}{'n':>5}  " + "".join(f"{d.split('_')[1][:4]:>8}" for d in CONF_DIMS) + "   (mean dim)")
    for st in ("clear_true", "mostly_true", "mixed", "unprovable", "mostly_false", "clear_false", "altered_media"):
        m = sub_arr == st
        if m.sum():
            means = "".join(f"{dims[d][m].mean():>8.2f}" for d in CONF_DIMS)
            print(f"  {st:<14}{int(m.sum()):>5}  {means}")

    def grp(dim, sts):
        m = np.isin(sub_arr, sts)
        return dims[dim][m]

    print("\n  Hypothesis tests (Mann-Whitney one-sided; rbc>0 = first group lower):")
    # H1: agreement lower on mixed than on clear_true/clear_false
    p, e = _mwu(grp("evidence_agreement", ["mixed"]), grp("evidence_agreement", ["clear_true", "clear_false"]))
    print(f"  H1  agreement(mixed) < agreement(clear_true|clear_false):   p={p:.4f}  rbc={e:+.3f}  "
          f"{'✓' if (p==p and p<0.05) else '✗ (no separation)'}")
    # H2: sufficiency lower on unprovable than on resolved
    p, e = _mwu(grp("evidence_sufficiency", ["unprovable"]), grp("evidence_sufficiency", list(RESOLVED)))
    print(f"  H2  sufficiency(unprovable) < sufficiency(resolved):        p={p:.4f}  rbc={e:+.3f}  "
          f"{'✓' if (p==p and p<0.05) else '✗ (no separation)'}")
    # H3: source_reliability shows no separation (mixed vs resolved); expect NS
    p, e = _mwu(grp("source_reliability", ["mixed", "unprovable"]), grp("source_reliability", list(RESOLVED)))
    print(f"  H3  source_reliability(unresolved) vs (resolved):          p={p:.4f}  rbc={e:+.3f}  "
          f"{'NS => dud, as expected' if not (p==p and p<0.05) else 'separates (unexpected)'}")


if __name__ == "__main__":
    main()
