"""Modality tagger — step 1 of the verdict-confidence study.

Classifies each gold claim's publisher `original_rating` into:
  - "artifact" — image/video / AI-generated / altered / miscaptioned / satire verdicts a
    TEXT-only verifier structurally cannot reproduce (≈ all Refuted; ~17.6% of eval_v1).
  - "text"     — everything else (text-verifiable claims).

Every metric in the study is reported twice — INCLUDING and EXCLUDING the artifact bucket —
with the text-only number as the headline (docs/verdict_confidence_design.md §1).

The regex is pinned in the design brief; it reproduces the 2026-06-13 audit's 17.6% share
on eval_v1 (clog/130626.md). The bare `doctor` token was tightened to `doctored` (it matched
only a verifiable "resident doctors" pay-stat — a false positive — and zero real artifacts).
A few prose false negatives remain (AI artifacts phrased "generated using AI" / "made using
artificial intelligence" that `ai-gener` misses) — net ~a handful of rows, left as-is.

Run (from src/):
  uv run python -m eval.scripts.verdict_confidence.modality
Writes a working copy with the `modality` column to data/eval_v1_modality.parquet and prints
the artifact share + its cross-tab with binary_label. Does NOT mutate the canonical gold.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

# Artifact/satire rating regex, matched case-insensitively against original_rating.
# `doctored` (not bare `doctor`): the audit showed `doctor` matched ONLY a false positive
# ("resident doctors" pay-stat) and zero real artifacts, so this strictly removes the FP.
ARTIFACT_RX = (
    r"alter|doctored|ai-gener|ai gener|miscaption|manipulat|deepfake|"
    r"digitally|photoshop|fabricat|staged|morph|satir|parod"
)


def tag_modality(df: pl.DataFrame, rating_col: str = "original_rating") -> pl.DataFrame:
    """Add a `modality` column ∈ {"text", "artifact"} derived from the publisher rating text."""
    return df.with_columns(
        pl.when(pl.col(rating_col).fill_null("").str.to_lowercase().str.contains(ARTIFACT_RX))
        .then(pl.lit("artifact"))
        .otherwise(pl.lit("text"))
        .alias("modality")
    )


IN = Path("eval/data/eval_v1.parquet")
OUT = Path("eval/scripts/verdict_confidence/data/eval_v1_modality.parquet")


def main() -> None:
    df = tag_modality(pl.read_parquet(IN))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)

    n = df.height
    n_art = int((df["modality"] == "artifact").sum())
    print(f"wrote {OUT}  ({n} rows)")
    print(f"artifact: {n_art}/{n} = {n_art / n:.3%}   (audit target ~17.6%)")
    print(
        df.group_by("modality", "binary_label")
        .agg(pl.len().alias("n"))
        .sort(["modality", "n"], descending=[False, True])
    )


if __name__ == "__main__":
    main()
