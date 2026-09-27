"""Per-STAGE agreement between two Claimify runs (gpt-oss-120b vs qwen3-32b).

Instead of only comparing the final binary (has_claim), this compares the two
models at EACH of the three Claimify stages, so we can see WHERE they diverge:

  Stage 1 SELECTION      : verifiable (bool)      -- on all posts
  Stage 2 DISAMBIGUATION : can_disambiguate (bool) + confidence (cat)
                           -- conditioned on BOTH models reaching stage 2
                              (i.e. both said verifiable=true)
  Stage 3 DECOMPOSITION  : n_claims                -- conditioned on BOTH
                              reaching stage 3 (both can_disambiguate=true)

Stages are conditional: a model only runs stage 2 if its stage 1 said yes, so we
compare each stage on the subset where both models actually executed it.
Read-only.

  uv run python -m eval.scripts.feed_study.analyze_claimify_stages
  uv run python -m eval.scripts.feed_study.analyze_claimify_stages \\
      --a claimify_gpt-oss-120b.parquet --b claimify_qwen3-32b.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

D = Path(__file__).parent / "data"


def _resolve(p: Path) -> Path:
    return p if p.exists() else (D / p if (D / p).exists() else p)


def _kappa(a: list, b: list) -> float:
    """Cohen's kappa over arbitrary categorical labels."""
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b), key=str)
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    conf = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        conf[idx[x]][idx[y]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    row = [sum(conf[i]) for i in range(k)]
    col = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(k))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def _label(k: float) -> str:
    if k != k:
        return "n/a"
    return ("worse-than-chance" if k < 0 else "slight" if k < 0.2 else "fair" if k < 0.4
            else "moderate" if k < 0.6 else "substantial" if k < 0.8 else "almost-perfect")


def _binary_block(a: list[bool], b: list[bool], na: str, nb: str, yes="yes", no="no") -> None:
    n = len(a)
    if n == 0:
        print("  (no shared rows)")
        return
    agree = sum(1 for x, y in zip(a, b) if x == y)
    both = sum(1 for x, y in zip(a, b) if x and y)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)
    ao = sum(1 for x, y in zip(a, b) if x and not y)
    bo = sum(1 for x, y in zip(a, b) if not x and y)
    k = _kappa(a, b)
    print(f"  n={n} | {na} {yes}={both + ao} ({100*(both+ao)/n:.1f}%) | "
          f"{nb} {yes}={both + bo} ({100*(both+bo)/n:.1f}%)")
    print(f"  %agree={100*agree/n:.1f}  kappa={k:.3f} ({_label(k)})")
    print(f"  both {yes}={both}  {na}-only={ao}  {nb}-only={bo}  both {no}={neither}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, default=D / "claimify_gpt-oss-120b.parquet")
    ap.add_argument("--b", type=Path, default=D / "claimify_qwen3-32b.parquet")
    args = ap.parse_args()

    cols = ["post_id", "outcome", "verifiable", "can_disambiguate", "confidence", "n_claims"]
    g = pl.read_parquet(_resolve(args.a)).filter(pl.col("error").is_null()).select(cols)
    q = pl.read_parquet(_resolve(args.b)).filter(pl.col("error").is_null()).select(cols)
    na, nb = args.a.stem.replace("claimify_", ""), args.b.stem.replace("claimify_", "")
    j = g.join(q, on="post_id", suffix="_q")
    n = j.height
    print(f"paired posts (both error-free): {n}\n")

    # ---- per-model outcome funnel ----
    print("=" * 64)
    print("OUTCOME FUNNEL per model")
    for name, df in ((na, g), (nb, q)):
        f = dict(df.group_by("outcome").len().iter_rows())
        print(f"  {name:14s}: no_verifiable={f.get('no_verifiable_content',0)}  "
              f"abstained={f.get('abstained_ambiguous',0)}  claims={f.get('claims',0)}")

    # ---- Stage 1: SELECTION (verifiable) on all posts ----
    print("\n" + "=" * 64)
    print("STAGE 1 — SELECTION: verifiable?  (all posts)")
    _binary_block(j["verifiable"].to_list(), j["verifiable_q"].to_list(), na, nb,
                  yes="verifiable", no="not")

    # ---- Stage 2: DISAMBIGUATION, conditioned on both verifiable ----
    both_ver = j.filter(pl.col("verifiable") & pl.col("verifiable_q"))
    print("\n" + "=" * 64)
    print(f"STAGE 2 — DISAMBIGUATION: can_disambiguate?  "
          f"(subset: both models said verifiable, n={both_ver.height})")
    _binary_block(both_ver["can_disambiguate"].fill_null(False).to_list(),
                  both_ver["can_disambiguate_q"].fill_null(False).to_list(), na, nb,
                  yes="resolvable", no="abstain")
    # confidence agreement, where both produced a confidence (both can_disambiguate)
    both_dis = both_ver.filter(pl.col("can_disambiguate") & pl.col("can_disambiguate_q"))
    if both_dis.height:
        ca = both_dis["confidence"].to_list()
        cb = both_dis["confidence_q"].to_list()
        ex = 100 * sum(1 for x, y in zip(ca, cb) if x == y) / len(ca)
        print(f"\n  confidence agreement (both resolvable, n={both_dis.height}): "
              f"exact={ex:.1f}%  kappa={_kappa(ca, cb):.3f}")
        for name, col in ((na, ca), (nb, cb)):
            dist = {c: col.count(c) for c in ("high", "medium", "low")}
            print(f"    {name:14s} confidence dist: {dist}")

    # ---- Stage 3: DECOMPOSITION (n_claims), conditioned on both reaching it ----
    both_dec = j.filter(
        pl.col("verifiable") & pl.col("verifiable_q")
        & pl.col("can_disambiguate").fill_null(False) & pl.col("can_disambiguate_q").fill_null(False)
    )
    print("\n" + "=" * 64)
    print(f"STAGE 3 — DECOMPOSITION: n_claims  (subset: both reached decomposition, n={both_dec.height})")
    if both_dec.height:
        ga = both_dec["n_claims"].to_list()
        qa = both_dec["n_claims_q"].to_list()
        exact = 100 * sum(1 for x, y in zip(ga, qa) if x == y) / len(ga)
        within1 = 100 * sum(1 for x, y in zip(ga, qa) if abs(x - y) <= 1) / len(ga)
        print(f"  {na} mean n_claims={sum(ga)/len(ga):.2f} (total {sum(ga)}) | "
              f"{nb} mean n_claims={sum(qa)/len(qa):.2f} (total {sum(qa)})")
        print(f"  exact n_claims match={exact:.1f}%  |diff|<=1={within1:.1f}%")

    # ---- where does the FINAL disagreement enter? ----
    print("\n" + "=" * 64)
    print("WHERE FINAL has_claim DISAGREEMENT ENTERS")
    j2 = j.with_columns(
        (pl.col("outcome") == "claims").alias("hc"),
        (pl.col("outcome_q") == "claims").alias("hc_q"),
    )
    disagree = j2.filter(pl.col("hc") != pl.col("hc_q"))
    # classify the disagreement by the stage at which the two diverged
    sel_div = disagree.filter(pl.col("verifiable") != pl.col("verifiable_q")).height
    dis_div = disagree.filter(
        (pl.col("verifiable") == pl.col("verifiable_q"))
        & (pl.col("can_disambiguate").fill_null(False) != pl.col("can_disambiguate_q").fill_null(False))
    ).height
    print(f"  total final disagreements: {disagree.height}")
    print(f"    diverged at SELECTION (verifiable differs):       {sel_div}")
    print(f"    agreed on verifiable, diverged at DISAMBIGUATION: {dis_div}")


if __name__ == "__main__":
    main()
