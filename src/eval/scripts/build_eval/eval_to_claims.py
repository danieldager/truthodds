"""Adapt a harvested/eval parquet -> the claims-shaped parquet verify_run consumes.

Reads an eval/harvest parquet (harmonised_label + review_url + claim_text + ...)
and emits the claims schema the Tier-3 grading harness expects
(verification_grading/verify_run.py): a stable claim_id (sha1[:16] of
"<claim_text>||<review_url>", matching load_claims.stable_id), claim_text,
gold_label (= harmonised_label, CE normalised to the verifier's ".../Cherrypicking"
string), plus binary_label, claim_date, review_url, fact_checking_article (= review_url,
so verify_run excludes the fact-check page from retrieval) and publisher_site. No API calls.

  uv run python -m eval.scripts.build_eval.eval_to_claims \
      -i eval/data/eval_v1.parquet -o eval/data/claims_from_eval.parquet
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import polars as pl

DEFAULT_INPUT = Path("eval/data/eval_v1.parquet")


def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    out = args.output or args.input.with_name(args.input.stem + "_claims.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(args.input)
    # Only rows with usable claim text AND a harmonised label can be graded.
    df = df.filter(
        pl.col("claim_text").is_not_null()
        & (pl.col("claim_text").str.strip_chars() != "")
        & pl.col("harmonised_label").is_not_null()
    )

    claim_ids = [
        stable_id(f"{row['claim_text']}||{row['review_url']}")
        for row in df.iter_rows(named=True)
    ]
    out_df = df.select(
        pl.Series("claim_id", claim_ids, dtype=pl.Utf8),
        pl.col("claim_text"),
        # Normalise the CE label to the verifier's 4-class string so score.py's exact-match
        # scoring lands it (harmonize emits "Conflicting Evidence"; the verifier's
        # verdict_4class emits the ".../Cherrypicking" variant).
        pl.when(pl.col("harmonised_label") == "Conflicting Evidence")
        .then(pl.lit("Conflicting Evidence/Cherrypicking"))
        .otherwise(pl.col("harmonised_label"))
        .alias("gold_label"),
        pl.col("binary_label"),
        pl.col("claim_date"),
        pl.col("review_url"),
        # verify_run builds exclude_urls from `fact_checking_article` to stop the verifier
        # retrieving the fact-check page itself (leakage). For FCT rows that page is review_url.
        pl.col("review_url").alias("fact_checking_article"),
        pl.col("publisher_site"),
    )

    before = out_df.height
    out_df = out_df.unique(subset="claim_id", keep="first", maintain_order=True)
    if out_df.height < before:
        print(f"dropped {before - out_df.height} duplicate claim_id(s)")

    out_df.write_parquet(out)
    print(f"wrote {out_df.height} rows -> {out}")
    print(f"unique claim_id: {out_df['claim_id'].n_unique()} / {out_df.height}")
    print("\ngold_label distribution:")
    print(out_df.group_by("gold_label").agg(pl.len().alias("n")).sort("n", descending=True))


if __name__ == "__main__":
    main()
