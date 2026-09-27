"""Combine the per-source harvest parquets into the final TEXT-ONLY fact-check dataset.

  uv run python -m eval.scripts.combine_dataset            # write eval/data/factcheck_textonly.parquet
  uv run python -m eval.scripts.combine_dataset --dry-run  # print the breakdown only, write nothing

Concatenates every eval/data/*_harvest.parquet (schema-relaxed) and DROPS artifact-axis rows —
image/video-authenticity claims that are unusable in a text-only dataset because we don't store the
media (w/ Daniel, clog 250626). The kept set is `judged_axis in {content, attribution}`. Untagged
rows (a source not yet enriched) are reported and excluded, never silently dropped.

The per-source `*_harvest.parquet` files stay the full harvest (all axes) — this step is the
filtered *deliverable*, so dropping artifact here is non-destructive / re-runnable.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import polars as pl

OUT = Path("eval/data/factcheck_textonly.parquet")


def _section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the breakdown, write nothing")
    args = ap.parse_args()

    files = sorted(glob.glob("eval/data/*_harvest.parquet"))
    combined = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")

    # Gold corrections (review_url -> label) for the llm_map slips the dev-200 audit caught
    # (clog 250626) — applied here so they survive re-harvest/re-combine. Recompute binary_label.
    CORRECTIONS = {
        "https://fullfact.org/immigration/asylum-seekers-freedom-pass-london/": "Refuted",   # was Supported (backwards)
        "https://fullfact.org/conflict/firework-video-false-claims-tehran-japan/": "Refuted",  # was NEE; found positive counter-evidence
    }
    combined = combined.with_columns(
        harmonised_label=pl.col("review_url").replace_strict(
            list(CORRECTIONS), list(CORRECTIONS.values()), default=pl.col("harmonised_label"))
    ).with_columns(
        binary_label=pl.when(pl.col("harmonised_label") == "Supported").then(pl.lit("pass"))
        .when(pl.col("harmonised_label").is_null()).then(pl.lit(None)).otherwise(pl.lit("flag"))
    )

    # Drop junk/placeholder ratings (Full Fact "Test" records) — not real fact-checks.
    _rn = pl.col("original_rating").fill_null("").str.strip_chars().str.to_lowercase()
    n_junk = combined.filter(_rn == "test").height
    combined = combined.filter(_rn != "test")

    n_all = combined.height
    untagged = combined.filter(pl.col("judged_axis").is_null())
    if untagged.height:
        bad = untagged.group_by("publisher_site").agg(pl.len().alias("n")).sort("n", descending=True)
        print(f"WARNING: {untagged.height}/{n_all} rows have no judged_axis (source not enriched) "
              f"— EXCLUDED:\n{bad.to_dicts()}")

    artifact = combined.filter(pl.col("judged_axis") == "artifact").height
    text = combined.filter(pl.col("judged_axis").is_in(["content", "attribution"]))

    _section("DROP-ARTIFACT FILTER")
    print(f"all rows ............ {n_all}  (junk 'Test' dropped: {n_junk})")
    print(f"artifact dropped .... {artifact}")
    print(f"untagged excluded ... {untagged.height}")
    print(f"text-only kept ...... {text.height}")

    _section("BY SOURCE (text-only)")
    bs = (text.group_by("publisher_site")
          .agg(pl.len().alias("n"),
               (pl.col("judged_axis") == "content").sum().alias("content"),
               (pl.col("judged_axis") == "attribution").sum().alias("attrib"),
               pl.col("raw_tier").is_not_null().sum().alias("pairs"))
          .sort("n", descending=True))
    for r in bs.to_dicts():
        print(f"  {r['publisher_site']:<22}{r['n']:>5}  (content {r['content']:>4}, "
              f"attrib {r['attrib']:>3}, extraction-pairs {r['pairs']:>4})")

    _section("LANGUAGE")
    for r in text.group_by("language_code").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts():
        print(f"  {r['language_code'] or '?':<6}{r['n']:>5}")

    _section("VERIFICATION DATASET  (text-only, has a harmonised verdict)")
    ver = text.filter(pl.col("harmonised_label").is_not_null())
    print(f"  usable verification rows: {ver.height}/{text.height}")
    for r in ver.group_by("harmonised_label").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts():
        print(f"    {r['harmonised_label']:<22}{r['n']:>5}")
    for r in text.group_by("binary_label").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts():
        print(f"    binary={str(r['binary_label']):<10}{r['n']:>5}")

    _section("EXTRACTION DATASET  (raw -> normalized pairs)")
    pairs = text.filter(pl.col("raw_tier").is_not_null())
    print(f"  usable extraction pairs: {pairs.height}/{text.height}")
    if "raw_tier" in pairs.columns:
        for r in pairs.group_by("raw_tier").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts():
            print(f"    tier={r['raw_tier']:<8}{r['n']:>5}")

    _section("DATES")
    print(f"  review_date span: {text['review_date'].min()} -> {text['review_date'].max()}")
    if "claim_date" in text.columns:
        cd = text.filter(pl.col("claim_date").is_not_null()).height
        print(f"  claim_date present: {cd}/{text.height}")

    if args.dry_run:
        print("\n[dry-run] nothing written.")
        return
    text.write_parquet(OUT)
    print(f"\nWrote {OUT}: {text.height} text-only rows.")


if __name__ == "__main__":
    main()
