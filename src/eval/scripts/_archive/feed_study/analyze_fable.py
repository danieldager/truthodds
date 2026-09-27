"""Exploratory analysis of two FABLE runs (gpt-oss-120b vs qwen3-32b).

Works for both post mode and claim mode — pass the two parquets with --a/--b.
The join key is auto-detected: per-claim if `claim_index` varies (claim mode),
else per-post. Read-only. Prints:
  1. fable_total distribution per run (histogram + percentiles).
  2. per-dimension means per run.
  3. threshold sweep on fable_total: for T in 6..25, binarize both at
     (total >= T) and report inter-run % agreement + Cohen's kappa.
  4. example units binned low / medium / high, both runs' dim vectors side by
     side, for hand inspection.

  uv run python -m eval.scripts.feed_study.analyze_fable          # post mode (default)
  uv run python -m eval.scripts.feed_study.analyze_fable \\
      --a data/fable_claim_gpt-oss-120b.parquet --b data/fable_claim_qwen3-32b.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

D = Path(__file__).parent / "data"
DIMS = ["fragmentation", "actionability", "believability", "spread_likelihood", "exploitativeness"]


def _kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    cats = [False, True]
    idx = {c: i for i, c in enumerate(cats)}
    conf = [[0, 0], [0, 0]]
    for x, y in zip(a, b):
        conf[idx[x]][idx[y]] += 1
    po = (conf[0][0] + conf[1][1]) / n
    row = [conf[0][0] + conf[0][1], conf[1][0] + conf[1][1]]
    col = [conf[0][0] + conf[1][0], conf[0][1] + conf[1][1]]
    pe = (row[0] / n) * (col[0] / n) + (row[1] / n) * (col[1] / n)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def _hist(df: pl.DataFrame, name: str) -> None:
    n = df.height
    print(f"\n{name}  (n={n})")
    counts = dict(df.group_by("fable_total").len().sort("fable_total").iter_rows())
    mx = max(counts.values()) if counts else 1
    for t in range(5, 26):
        c = counts.get(t, 0)
        bar = "#" * round(40 * c / mx) if c else ""
        if c:
            print(f"  total {t:2d} | {c:4d} {bar}")
    q = df.select(
        pl.col("fable_total").quantile(0.25).alias("p25"),
        pl.col("fable_total").median().alias("p50"),
        pl.col("fable_total").quantile(0.75).alias("p75"),
        pl.col("fable_total").mean().alias("mean"),
        pl.col("fable_total").max().alias("max"),
    ).row(0)
    print(f"  p25={q[0]} median={q[1]} p75={q[2]} mean={q[3]:.2f} max={q[4]}")
    print("  per-dim mean: " + ", ".join(f"{d}={df[d].mean():.2f}" for d in DIMS))


def _resolve(p: Path) -> Path:
    return p if p.exists() else (D / p if (D / p).exists() else p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, default=D / "fable_post_gpt-oss-120b.parquet")
    ap.add_argument("--b", type=Path, default=D / "fable_post_qwen3-32b.parquet")
    args = ap.parse_args()

    g = pl.read_parquet(_resolve(args.a)).filter(pl.col("error").is_null())
    q = pl.read_parquet(_resolve(args.b)).filter(pl.col("error").is_null())
    name_a, name_b = args.a.stem, args.b.stem

    # Per-claim key if claim_index actually varies (claim mode); else per-post.
    keys = ["post_id"]
    if "claim_index" in g.columns and g["claim_index"].max() is not None and g["claim_index"].max() > -1:
        keys = ["post_id", "claim_index"]
    print(f"join key: {keys}")

    print("=" * 70)
    print("1-2. DISTRIBUTIONS")
    _hist(g, name_a)
    _hist(q, name_b)

    # join for paired analysis
    j = (
        g.select(keys + ["unit_text", "fable_total"] + DIMS)
        .join(q.select(keys + ["fable_total"] + DIMS), on=keys, suffix="_q")
    )
    print("\n" + "=" * 70)
    print(f"3. THRESHOLD SWEEP (paired n={j.height}) — total>=T, inter-model agreement")
    print("   T |  gpt yes  qwen yes |  %agree   kappa")
    best = (None, -2.0)
    gt = j["fable_total"].to_list()
    qt = j["fable_total_q"].to_list()
    for T in range(6, 26):
        a = [v >= T for v in gt]
        b = [v >= T for v in qt]
        agree = 100.0 * sum(1 for x, y in zip(a, b) if x == y) / len(a)
        k = _kappa(a, b)
        flag = ""
        if all(a) or not any(a) or all(b) or not any(b):
            flag = "  (degenerate)"
        print(f"  {T:2d} | {sum(a):7d}  {sum(b):7d} | {agree:6.1f}%  {k:6.3f}{flag}")
        if not flag and k > best[1]:
            best = (T, k)
    if best[0] is not None:
        print(f"   -> max non-degenerate kappa at total>={best[0]} (kappa={best[1]:.3f})")

    print("\n" + "=" * 70)
    print("4. EXAMPLE POSTS (binned by mean of the two totals), scores side by side")
    j2 = j.with_columns(((pl.col("fable_total") + pl.col("fable_total_q")) / 2).alias("avg"))
    bins = [("LOW (avg<=6)", pl.col("avg") <= 6),
            ("MEDIUM (7<=avg<=10)", (pl.col("avg") >= 7) & (pl.col("avg") <= 10)),
            ("HIGH (avg>=12)", pl.col("avg") >= 12)]
    for title, expr in bins:
        sub = j2.filter(expr).sort("avg", descending=("HIGH" in title))
        # spread the sample across the bin
        take = min(5, sub.height)
        idxs = [round(i * (sub.height - 1) / max(take - 1, 1)) for i in range(take)] if sub.height else []
        print(f"\n--- {title}  ({sub.height} posts) ---")
        for i in idxs:
            r = sub.row(i, named=True)
            gv = [r[d] for d in DIMS]
            qv = [r[f"{d}_q"] for d in DIMS]
            print(f"  gpt{gv}={r['fable_total']:2d}  qwen{qv}={r['fable_total_q']:2d}")
            print(f"    | {(r['unit_text'] or '')[:120].replace(chr(10),' ')}")


if __name__ == "__main__":
    main()
