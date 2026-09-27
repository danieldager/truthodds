"""Build the canonical 12-month multi-source eval dataset + a frozen DEV/TEST split.

  uv run python -m eval.scripts.build_dataset_splits          # dev 1000 / test = rest
  uv run python -m eval.scripts.build_dataset_splits --dev 1000

Combines the 8 per-source `*_harvest.parquet` files (already a dense 12 months, all enriched with
`judged_axis`, `has_image`, `has_video`, `sources`, image/quote cols) into one deduped set, then adds a
deterministic `split` column: **DEV = 1000 rows** (the lowest md5(review_url) hashes), **TEST = the
rest**. The split is frozen/reproducible (md5, no RNG) so it's identical on every re-run / future session.
NOT image-only and NOT oversampled (per Daniel 2026-06-26) — the natural ~25%-image mix is preserved so a
downstream authenticity router's precision is measurable on the text-only majority too.

Writes `eval/data/dataset_12mo.parquet` (all rows + `split`). Does NOT touch `factcheck_textonly.parquet`
(the combiner's protected dev-200 baseline) or `combine_dataset.py` (parallel session owns that combiner).
"""
from __future__ import annotations

import argparse
import glob
import hashlib

import polars as pl

OUT = "eval/data/dataset_12mo.parquet"


def combined() -> pl.DataFrame:
    frames = [pl.read_parquet(f) for f in sorted(glob.glob("eval/data/*_harvest.parquet"))]
    shared = sorted(set.intersection(*[set(f.columns) for f in frames]))
    return pl.concat([f.select(shared) for f in frames], how="diagonal_relaxed").unique("review_url")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", type=int, default=1000)
    args = ap.parse_args()

    df = combined()
    df = df.with_columns(
        pl.col("review_url").map_elements(
            lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % (10 ** 12), return_dtype=pl.Int64
        ).alias("_h")
    )
    cut = df["_h"].sort()[args.dev - 1]  # dev = the args.dev lowest hashes (md5 space ~1e12 → no ties)
    df = df.with_columns(
        pl.when(pl.col("_h") <= cut).then(pl.lit("dev")).otherwise(pl.lit("test")).alias("split")
    ).drop("_h")
    ndev = df.filter(pl.col("split") == "dev").height
    df.write_parquet(OUT)

    print(f"combined {df.height} fact-checks (12mo, 8 sources) → {OUT}")
    print(f"split: dev {ndev} | test {df.height - ndev}")
    for col in ("judged_axis", "has_image", "has_video", "language_code"):
        if col not in df.columns:
            continue
        print(f"\n  {col} by split:")
        g = df.group_by("split", col).len().sort("split", col)
        for s in ("dev", "test"):
            sub = g.filter(pl.col("split") == s)
            tot = sub["len"].sum()
            cells = "  ".join(f"{r[col]}={r['len']}" for r in sub.to_dicts())
            print(f"    {s:4s} (n={tot}): {cells}")


if __name__ == "__main__":
    main()
