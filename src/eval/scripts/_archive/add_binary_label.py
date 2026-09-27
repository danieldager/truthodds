"""Add the binary deployment label to an existing eval_v1.parquet without
re-running the LLM pass. Idempotent.
"""
from pathlib import Path

import polars as pl

PARQUET = Path("eval/data/eval_v1.parquet")


def main() -> None:
    df = pl.read_parquet(PARQUET)
    df = df.with_columns(
        binary_label=pl.when(pl.col("harmonised_label") == "Supported")
        .then(pl.lit("pass"))
        .when(pl.col("harmonised_label").is_null())
        .then(pl.lit(None))
        .otherwise(pl.lit("flag"))
    )
    df.write_parquet(PARQUET)
    print(f"Updated {PARQUET}")
    print(df.group_by("binary_label").agg(pl.len().alias("n")).sort("n", descending=True))


if __name__ == "__main__":
    main()
