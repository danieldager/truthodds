"""FactCheck.org harvester -> rolling, deduped parquet (GFC API discovery + page-scrape sources).

  uv run python -m eval.scripts.harvest_factcheckorg --months 12
  uv run python -m eval.scripts.harvest_factcheckorg --months 12 --max 10   # validate

claim + verdict + claimant + claimDate from the GFC API; a page scrape adds the cited evidence
(eval.factcheckorg). Verdict harmonised by the rule table (RULES["factcheck.org"]) with an llm_map
fallback for the free-text minority. Same shared schema; raw side (rare) filled by fetch_sources.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import polars as pl

from config import EXTRACTION_MODEL
from eval.factcheckorg import PUBLISHER_SITE, harvest_records
from eval.harmonize import llm_map, rule_map

OUT = Path("eval/data/factcheckorg_harvest.parquet")
SCHEMA = {
    "claim_text": pl.Utf8, "original_rating": pl.Utf8, "rating_value": pl.Utf8,
    "claimant": pl.Utf8, "publisher_site": pl.Utf8, "publisher_name": pl.Utf8,
    "review_url": pl.Utf8, "review_date": pl.Utf8, "claim_date": pl.Utf8, "language_code": pl.Utf8,
    "claim_source_url": pl.Utf8, "raw_claim": pl.Utf8, "context": pl.Utf8, "sources": pl.List(pl.Utf8),
    "harmonised_label": pl.Utf8,
}


def _row(rec: dict) -> dict:
    return {
        "claim_text": rec["normalized_claim"], "original_rating": rec["verdict_raw"],
        "rating_value": rec.get("rating_value"), "claimant": rec.get("claimant"),
        "publisher_site": rec["publisher_site"], "publisher_name": rec.get("publisher_name") or "FactCheck.org",
        "review_url": rec["review_url"], "review_date": rec["review_date"], "claim_date": rec.get("claim_date"),
        "language_code": rec["language_code"], "claim_source_url": rec.get("claim_source_url"),
        "raw_claim": None, "context": rec.get("context"), "sources": rec.get("sources"),
    }


def _harmonise(rating: str | None) -> str | None:
    """Rule table first (RULES['factcheck.org']); LLM fallback for any free-text string."""
    lab = rule_map(PUBLISHER_SITE, rating)
    if lab is None and rating:
        try:
            lab = llm_map(EXTRACTION_MODEL, rating)
        except Exception as e:  # noqa: BLE001
            print(f"  llm_map failed for {rating!r}: {e}")
    return lab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=8)
    args = ap.parse_args()

    since_iso = (datetime.date.today() - datetime.timedelta(days=round(args.months * 30.44))).isoformat()
    seen = set(pl.read_parquet(OUT)["review_url"].to_list()) if OUT.exists() else set()
    recs = harvest_records(max_age_days=round(args.months * 30.44) + 7, max_pages=args.max_pages)
    recs = [r for r in recs
            if r.get("review_date") and r["review_date"] >= since_iso and r["review_url"] not in seen]
    if args.max:
        recs = recs[: args.max]
    print(f"{PUBLISHER_SITE}: {len(recs)} in-window + new records")
    if not recs:
        print("0 rows — nothing written.")
        return

    rows = [{**_row(r), "harmonised_label": _harmonise(r["verdict_raw"])} for r in recs]
    new = pl.DataFrame(rows, schema_overrides=SCHEMA).with_columns(
        binary_label=pl.when(pl.col("harmonised_label") == "Supported").then(pl.lit("pass"))
        .when(pl.col("harmonised_label").is_null()).then(pl.lit(None)).otherwise(pl.lit("flag"))
    )
    combined = new if not OUT.exists() else pl.concat([pl.read_parquet(OUT), new], how="diagonal_relaxed")
    combined.write_parquet(OUT)

    cit = new.filter(pl.col("sources").list.len() > 0).height
    miss = new.filter(pl.col("harmonised_label").is_null()).height
    print(f"\nWrote {OUT}: +{new.height} new, {combined.height} total")
    print(f"cited sources: {cit}/{new.height} ({cit/new.height*100:.0f}%)  |  "
          f"claim_date: {new.filter(pl.col('claim_date').is_not_null()).height}/{new.height}  |  rule/LLM-miss: {miss}")
    print("harmonised:", new.group_by("harmonised_label").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())


if __name__ == "__main__":
    main()
