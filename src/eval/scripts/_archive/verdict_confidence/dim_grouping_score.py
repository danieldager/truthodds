"""Score the dimension-grouping + reasoning A/B (reads data/dim_grouping.parquet).

Answers the two experiments:
  Exp 2 (grouping: all5 / split_2_3 / per_dim, no reasoning) — does splitting the call change
        (a) cross-dim INDEPENDENCE, (b) SCALE USAGE (2 & 4 vs 1/3/5), (c) how well the scores
        PREDICT correctness, and at what cost?
  Exp 1 (reasoning: all5/none vs all5/medium) — does thinking change accuracy/calibration, and
        what does it cost?

Metrics implemented numpy-only: Spearman (rank→Pearson) and AUROC (Mann-Whitney, tie-aware).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from eval.scripts.verdict_confidence.dim_grouping_prompts import DIMS

OUT = Path("eval/scripts/verdict_confidence/data/dim_grouping.parquet")
CONF_DIMS = ("evidence_sufficiency", "evidence_agreement", "source_reliability")


def _ranks(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(len(x))
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, r)
    return (sums / counts)[inv]


def spearman(a: list, b: list) -> float | None:
    m = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(m) < 3:
        return None
    x = _ranks(np.array([p[0] for p in m], float)); y = _ranks(np.array([p[1] for p in m], float))
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def auroc(scores: list, pos: list) -> float | None:
    """AUROC of `scores` predicting the positive label (pos=1). Tie-aware via rank-sum."""
    m = [(s, p) for s, p in zip(scores, pos) if s is not None]
    n_pos = sum(p for _, p in m); n_neg = len(m) - n_pos
    if not n_pos or not n_neg:
        return None
    r = _ranks(np.array([s for s, _ in m], float))
    rank_pos = sum(rk for rk, (_, p) in zip(r, m) if p) + n_pos  # +n_pos: 0-based→1-based
    return float((rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def at24(vals: list) -> float:
    v = [x for x in vals if x is not None]
    return 100 * sum(1 for x in v if x in (2, 4)) / len(v) if v else 0.0


def dist(vals: list) -> dict:
    v = [x for x in vals if x is not None]
    return {k: v.count(k) for k in (1, 2, 3, 4, 5)}


def cell(df: pl.DataFrame, config: str, reasoning: str) -> pl.DataFrame:
    return df.filter((pl.col("config") == config) & (pl.col("reasoning") == reasoning) & pl.col("error").is_null())


def independence(g: pl.DataFrame) -> tuple[float | None, float | None]:
    """(mean |off-diagonal Spearman| over all dim pairs, halo = mean |corr(veracity, conf dims)|)."""
    cols = {d: g[d].to_list() for d in DIMS}
    off = [spearman(cols[a], cols[b]) for i, a in enumerate(DIMS) for b in DIMS[i + 1:]]
    off = [abs(c) for c in off if c is not None]
    halo = [spearman(cols["veracity"], cols[c]) for c in CONF_DIMS]
    halo = [abs(c) for c in halo if c is not None]
    return (float(np.mean(off)) if off else None, float(np.mean(halo)) if halo else None)


def correctness(g: pl.DataFrame) -> dict:
    """gold is binary pass/flag. veracity high → pass; contextual_integrity high → pass."""
    gold_pass = [1 if x == "pass" else 0 for x in g["gold"].to_list()]
    return {
        "auroc_veracity": auroc(g["veracity"].to_list(), gold_pass),
        "auroc_ctx_integ": auroc(g["contextual_integrity"].to_list(), gold_pass),
        "auroc_suff": auroc(g["evidence_sufficiency"].to_list(), gold_pass),
    }


def cost(g: pl.DataFrame) -> dict:
    return {"out_tok": g["completion_tokens"].mean(), "in_tok": g["prompt_tokens"].mean(),
            "cached": g["cached_tokens"].mean(), "n_calls": g["n_calls"].mean(),
            "lat": g["latency_s"].mean()}


def fmt(x, p=2):
    return "  n/a" if x is None else f"{x:.{p}f}"


def main() -> None:
    df = pl.read_parquet(OUT)
    ok = df.filter(pl.col("error").is_null())
    print(f"rows: {df.height} | errors: {df.height - ok.height} | "
          f"gold: {dict(zip(*[ok.filter(pl.col('config')=='all5')['gold'].value_counts().sort('count')[c].to_list() for c in ('gold','count')]))}")
    print(f"modality: {ok.filter(pl.col('config')=='all5')['modality'].value_counts().to_dicts()}\n")

    # ---- Exp 2: grouping (no reasoning) ----
    print("=" * 92)
    print("EXP 2 — DIMENSION GROUPING (reasoning OFF). Same fixed analysis across configs.")
    print("=" * 92)
    print(f"{'config':11s} {'n':>3s} {'ver%@2|4':>8s} {'ctx%@2|4':>8s} {'mean|corr|':>10s} "
          f"{'halo':>6s} {'AUROC_ver':>9s} {'AUROC_ctx':>9s} {'out_tok':>7s} {'lat_s':>6s}")
    for cfg in ("all5", "split_2_3", "per_dim"):
        g = cell(df, cfg, "none")
        if not g.height:
            continue
        mo, halo = independence(g); c = correctness(g); k = cost(g)
        print(f"{cfg:11s} {g.height:3d} {at24(g['veracity'].to_list()):7.0f}% "
              f"{at24(g['contextual_integrity'].to_list()):7.0f}% {fmt(mo):>10s} {fmt(halo):>6s} "
              f"{fmt(c['auroc_veracity']):>9s} {fmt(c['auroc_ctx_integ']):>9s} "
              f"{k['out_tok']:7.0f} {k['lat']:6.1f}")
    print("\nveracity score distribution (1→5) per config:")
    for cfg in ("all5", "split_2_3", "per_dim"):
        g = cell(df, cfg, "none")
        if g.height:
            print(f"  {cfg:11s} {dist(g['veracity'].to_list())}")

    # ---- Exp 1: reasoning on vs off (all5) ----
    print("\n" + "=" * 92)
    print("EXP 1 — REASONING (all5): none vs medium. Same fixed analysis.")
    print("=" * 92)
    print(f"{'reasoning':9s} {'n':>3s} {'ver%@2|4':>8s} {'AUROC_ver':>9s} {'AUROC_suff':>10s} "
          f"{'out_tok':>7s} {'in_tok':>7s} {'lat_s':>6s}")
    for rea in ("none", "medium"):
        g = cell(df, "all5", rea)
        if not g.height:
            continue
        c = correctness(g); k = cost(g)
        print(f"{rea:9s} {g.height:3d} {at24(g['veracity'].to_list()):7.0f}% {fmt(c['auroc_veracity']):>9s} "
              f"{fmt(c['auroc_suff']):>10s} {k['out_tok']:7.0f} {k['in_tok']:7.0f} {k['lat']:6.1f}")

    # paired veracity flip (same claims)
    a = cell(df, "all5", "none").select("claim_id", v_none="veracity")
    b = cell(df, "all5", "medium").select("claim_id", v_med="veracity")
    j = a.join(b, on="claim_id").drop_nulls()
    if j.height:
        flips = j.filter(pl.col("v_none") != pl.col("v_med")).height
        mad = (j["v_none"] - j["v_med"]).abs().mean()
        print(f"\npaired (n={j.height}): veracity changed on {flips} claims; mean|Δveracity|={mad:.2f}")


if __name__ == "__main__":
    main()
