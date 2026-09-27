"""Analyze the multi-model agreement run (agreement.parquet).

Answers the Stage-1 viability questions:
  1. Cross-model SPREAD per label (veracity + all 3 confidence dims) — where do models diverge?
  2. Binary-verdict agreement distribution, by stratum.
  3. Does agreement predict correctness? (error rate among unanimous vs split panels; mean
     veracity spread on correct vs incorrect deployment verdicts.)
  4. Bonus: does a 4-model majority ENSEMBLE beat gpt-oss alone vs gold?

Confidence framing: gpt-oss is the deployment verdict; panel agreement is a confidence wrapper
on it. Run (from src/): uv run python -m eval.scripts.verdict_confidence.agreement_analyze
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

IN = Path("eval/scripts/verdict_confidence/data/agreement.parquet")
DIMS = ("veracity", "evidence_sufficiency", "evidence_agreement", "source_reliability")


def main() -> None:
    df = pl.read_parquet(IN).filter(pl.col("error").is_null())
    n = df.height
    print(f"loaded {n} claims (panel of {len(df['model_order'][0])} models)\n")

    rows = []
    for r in df.to_dicts():
        ver = np.array(r["veracity_by_model"])
        pass_votes = int((ver >= 4).sum())
        n_models = len(ver)
        maj_pass = pass_votes > n_models / 2
        tie = pass_votes == n_models / 2
        ensemble = "tie" if tie else ("pass" if maj_pass else "flag")
        rows.append({
            "claim_id": r["claim_id"], "stratum": r["stratum"], "gold": r["gold"],
            "deployment_binary": r["deployment_binary"],
            "deployment_correct": int(r["deployment_binary"] == r["gold"]),
            "ensemble": ensemble,
            "ensemble_correct": int(ensemble == r["gold"]),
            "pass_votes": pass_votes,
            "unanimous": int(pass_votes in (0, n_models)),
            "agreement": max(pass_votes, n_models - pass_votes) / n_models,
            **{f"{d}_spread": float(np.std(r[f"{d}_by_model"])) for d in DIMS},
            **{f"{d}_range": int(np.ptp(r[f"{d}_by_model"])) for d in DIMS},
        })
    a = pl.DataFrame(rows)

    # 1. Per-dim cross-model spread
    print("### 1. Cross-model spread per label (how much the 4 models diverge)")
    print(f"  {'dim':<22} {'mean_std':>8} {'mean_range':>11} {'%unanimous':>11} {'%spread>=2':>11}")
    for d in DIMS:
        sp, rg = a[f"{d}_spread"].to_numpy(), a[f"{d}_range"].to_numpy()
        print(f"  {d:<22} {sp.mean():>8.2f} {rg.mean():>11.2f} "
              f"{100*(sp == 0).mean():>10.0f}% {100*(rg >= 2).mean():>10.0f}%")

    # 2. Binary agreement distribution
    print("\n### 2. Binary verdict — pass-vote distribution (0..4 models voting PASS)")
    print(a.group_by("pass_votes").agg(pl.len().alias("n")).sort("pass_votes"))
    print(f"  unanimous (0 or 4): {int(a['unanimous'].sum())}/{n} = {100*a['unanimous'].mean():.0f}%")
    print("  by stratum (mean agreement, mean veracity spread):")
    print(a.group_by("stratum").agg(
        pl.len().alias("n"), pl.col("agreement").mean().round(3).alias("agree"),
        pl.col("veracity_spread").mean().round(2).alias("ver_spread"),
        pl.col("deployment_correct").mean().round(3).alias("deploy_acc")).sort("stratum"))

    # 3. Does agreement predict correctness?
    print("\n### 3. Does agreement predict correctness? (deployment = gpt-oss)")
    for label, sub in [("panel UNANIMOUS", a.filter(pl.col("unanimous") == 1)),
                       ("panel SPLIT", a.filter(pl.col("unanimous") == 0))]:
        if sub.height:
            print(f"  {label:<16} n={sub.height:<3} deployment error rate="
                  f"{1 - sub['deployment_correct'].mean():.3f}")
    cor = a.filter(pl.col("deployment_correct") == 1)["veracity_spread"]
    inc = a.filter(pl.col("deployment_correct") == 0)["veracity_spread"]
    print(f"  mean veracity spread:  correct={cor.mean():.2f} (n={cor.len()})  "
          f"incorrect={inc.mean():.2f} (n={inc.len()})  "
          f"[higher spread on incorrect => spread predicts error]")

    # 4. Ensemble vs gpt-oss alone
    print("\n### 4. Ensemble (4-model majority) vs gpt-oss alone")
    print(f"  gpt-oss deployment accuracy: {a['deployment_correct'].mean():.3f}")
    nontie = a.filter(pl.col("ensemble") != "tie")
    print(f"  ensemble accuracy (excl ties): {nontie['ensemble_correct'].mean():.3f} "
          f"(n={nontie.height}, ties={a.height - nontie.height})")

    # 5. Per-claim eyeball
    print("\n### 5. Per-claim (veracity by model, spread, votes, correct)")
    show = a.join(df.select("claim_id", "veracity_by_model"), on="claim_id").sort(["stratum", "claim_id"])
    print(f"  {'id':<7} {'stratum':<14} {'gold':<5} {'veracity_4':<16} {'spr':>4} {'votes':>5} {'dep✓':>5}")
    for r in show.to_dicts():
        print(f"  {r['claim_id']:<7} {r['stratum']:<14} {r['gold']:<5} "
              f"{str(r['veracity_by_model']):<16} {r['veracity_spread']:>4.1f} "
              f"{r['pass_votes']:>5} {'Y' if r['deployment_correct'] else 'N':>5}")


if __name__ == "__main__":
    main()
