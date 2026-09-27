"""Assign a dev/test split to eval_v1 (+ a soft_flag tag for the misleading bucket).

dev ≈ 30% (for signal selection / threshold tuning), test ≈ 70% (held-out headline). Stratified
by binary_label × modality so both sets carry the rare PASS class + the artifact bucket
proportionally. The 60 stage-1 agreement claims (already eyeballed) are FORCED into dev so the
test set stays pristine. Seeded for reproducibility.

soft_flag = publisher rating where the atomic claim can be TRUE but the post is flagged for
framing (misleading/mixture/half-true/missing-context/…) — the measurement bucket from clog
14:30. Carried so calibration can be reported truth-only (separating soft_flag), like modality.

Run (from src/): uv run python -m eval.scripts.verdict_confidence.splits
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

MODALITY = Path("eval/scripts/verdict_confidence/data/eval_v1_modality.parquet")
STAGE1 = Path("eval/scripts/verdict_confidence/data/stage1_sample.parquet")
OUT = Path("eval/scripts/verdict_confidence/data/eval_v1_splits.parquet")
SOFT_RX = (r"mislead|mixture|half[ -]?true|missing context|needs context|out of context|"
           r"lacks context|cherry|partly|exaggerat|outdated|misrepresent|overstat|spin")
DEV_FRAC_OF_REMAINDER = 0.28   # + the forced stage-1 60 ⇒ ≈30% dev overall


def main() -> None:
    df = (pl.read_parquet(MODALITY).with_row_index("ridx")
          .with_columns(claim_id=pl.format("ev{}", "ridx"),
                        soft_flag=pl.col("original_rating").fill_null("").str.to_lowercase()
                        .str.contains(SOFT_RX)))
    forced_dev = set(pl.read_parquet(STAGE1)["claim_id"].to_list())  # the 60 already-seen claims

    remaining = df.filter(~pl.col("claim_id").is_in(forced_dev))
    parts = remaining.partition_by("binary_label", "modality", as_dict=False)
    dev_extra = pl.concat([p.sample(fraction=DEV_FRAC_OF_REMAINDER, seed=42) for p in parts])
    dev_ids = forced_dev | set(dev_extra["claim_id"].to_list())

    df = df.with_columns(
        split=pl.when(pl.col("claim_id").is_in(dev_ids)).then(pl.lit("dev")).otherwise(pl.lit("test")))
    df.write_parquet(OUT)

    print(f"wrote {OUT}  ({df.height} rows; dev forced from stage-1: {len(forced_dev)})")
    print(df.group_by("split").agg(
        pl.len().alias("n"),
        pl.col("binary_label").eq("pass").sum().alias("pass"),
        pl.col("binary_label").eq("flag").sum().alias("flag"),
        (pl.col("modality") == "artifact").sum().alias("artifact"),
        pl.col("soft_flag").sum().alias("soft_flag")).sort("split"))
    # confirm test is clean of stage-1
    leak = df.filter((pl.col("split") == "test") & pl.col("claim_id").is_in(forced_dev)).height
    print(f"stage-1 claims leaked into test: {leak} (must be 0)")


if __name__ == "__main__":
    main()
