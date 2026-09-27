"""Newschecker harvester -> rolling, deduped parquet (Google FCT API only).

  uv run python -m eval.scripts.harvest_newschecker --months 3
  uv run python -m eval.scripts.harvest_newschecker --months 3 --max 15   # validate

Pulls the Newschecker EN vertical (newschecker.in) straight from the GFC API — claim, verdict and
claimant are all in the API claim dicts, so there's no article fetch. Harmonises the clean enum
verdict via the existing RULES["newschecker.in"] table, with an LLM fallback for any unexpected
string. Same shared schema as the other harvesters; the verbatim raw claim / context is
reconstructed later by eval.scripts.fetch_sources (note: claimant is generic "Social Media Post").
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import polars as pl

from config import EXTRACTION_MODEL
from eval.harmonize import llm_map, rule_map
from eval.newschecker import VERTICALS, harvest_records

OUT = Path("eval/data/newschecker_harvest.parquet")
SCHEMA = {
    "claim_text": pl.Utf8, "original_rating": pl.Utf8, "rating_value": pl.Utf8,
    "claimant": pl.Utf8, "publisher_site": pl.Utf8, "publisher_name": pl.Utf8,
    "review_url": pl.Utf8, "review_date": pl.Utf8, "language_code": pl.Utf8,
    "claim_source_url": pl.Utf8, "raw_claim": pl.Utf8, "context": pl.Utf8, "sources": pl.List(pl.Utf8),
    "harmonised_label": pl.Utf8,
}


def _row(rec: dict) -> dict:
    return {
        "claim_text": rec["normalized_claim"], "original_rating": rec["verdict_raw"],
        "rating_value": rec["rating_value"], "claimant": rec["claimant"],
        "publisher_site": rec["publisher_site"], "publisher_name": rec["publisher_name"],
        "review_url": rec["review_url"], "review_date": rec["review_date"],
        "language_code": rec["language_code"], "claim_source_url": rec["claim_source_url"],
        "raw_claim": None, "context": rec["context"], "sources": None,
    }


def _harmonise(site: str, rating: str | None) -> str | None:
    """Existing rule table first; LLM fallback for any string not in the enum."""
    lab = rule_map(site, rating)
    if lab is None and rating:
        try:
            lab = llm_map(EXTRACTION_MODEL, rating)
        except Exception as e:  # network / non-canonical → leave null, reported below
            print(f"  llm_map failed for {rating!r}: {e}")
    return lab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=8)
    args = ap.parse_args()

    since_iso = (datetime.date.today() - datetime.timedelta(days=round(args.months * 30.44))).isoformat()
    max_age = round(args.months * 30.44) + 7  # pad for ingestion lag, mirroring the other harvesters

    seen = set(pl.read_parquet(OUT)["review_url"].to_list()) if OUT.exists() else set()
    recs: list[dict] = []
    for site, lang in VERTICALS.items():
        got = harvest_records(site, lang, max_age_days=max_age, max_pages=args.max_pages)
        recs += got
        print(f"{site} ({lang}): {len(got)} records from API")

    recs = [r for r in recs
            if r.get("review_date") and r["review_date"] >= since_iso and r["review_url"] not in seen]
    if args.max:
        recs = recs[: args.max]
    print(f"in-window + new (vs {OUT.name}): {len(recs)}")
    if not recs:
        print("0 rows — nothing written.")
        return

    rows = []
    for r in recs:
        row = _row(r)
        row["harmonised_label"] = _harmonise(r["publisher_site"], r["verdict_raw"])
        rows.append(row)

    new = pl.DataFrame(rows, schema_overrides=SCHEMA).with_columns(
        binary_label=pl.when(pl.col("harmonised_label") == "Supported").then(pl.lit("pass"))
        .when(pl.col("harmonised_label").is_null()).then(pl.lit(None)).otherwise(pl.lit("flag"))
    )
    combined = new if not OUT.exists() else pl.concat([pl.read_parquet(OUT), new], how="diagonal_relaxed")
    combined.write_parquet(OUT)

    miss = new.filter(pl.col("harmonised_label").is_null()).height
    print(f"\nWrote {OUT}: +{new.height} new, {combined.height} total")
    print(f"rule/LLM-miss (null label): {miss}")
    print("by language:", new.group_by("language_code").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())
    print("harmonised:", new.group_by("harmonised_label").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())


if __name__ == "__main__":
    main()
