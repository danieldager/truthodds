"""Direct PolitiFact harvester -> rolling, deduped parquet.

  uv run python -m eval.scripts.harvesters.harvest_politifact            # pull the RSS feed
  uv run python -m eval.scripts.harvesters.harvest_politifact --max 5    # cap (debug)

Supplements the Google FCT API, which under-ingests PolitiFact (~83% miss in a
settled window — clog/230626). Pulls the fact-check RSS, parses each article
(eval.politifact), harmonises the Truth-O-Meter rating with the SAME rule table
build_eval uses (PolitiFact is fully rule-covered — no LLM), and appends rows new
by review_url to eval/data/politifact_harvest.parquet.

Carries the extraction-gold extras the API never exposes: raw claim statement,
claimant, claim date, justification, and cited-source URLs.
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from tqdm import tqdm

from eval.harmonize import rule_map
from eval.politifact import FLIP_SLUGS, archive_urls, parse_article, rss_items

OUT = Path("eval/data/politifact_harvest.parquet")
SCHEMA = {
    "claim_text": pl.Utf8, "claim_source_url": pl.Utf8, "body_quote": pl.Utf8,
    "claimant": pl.Utf8, "claim_date": pl.Utf8,
    "claim_date_raw": pl.Utf8, "claim_venue": pl.Utf8, "publisher_site": pl.Utf8,
    "publisher_name": pl.Utf8, "review_url": pl.Utf8, "review_title": pl.Utf8,
    "review_date": pl.Utf8, "language_code": pl.Utf8, "original_rating": pl.Utf8,
    "rating_slug": pl.Utf8, "justification": pl.Utf8, "sources": pl.List(pl.Utf8),
}


def _harmonise(df: pl.DataFrame) -> pl.DataFrame:
    labels = [rule_map(r["publisher_site"], r["original_rating"]) for r in df.iter_rows(named=True)]
    return df.with_columns(harmonised_label=pl.Series(labels, dtype=pl.Utf8)).with_columns(
        binary_label=pl.when(pl.col("harmonised_label") == "Supported").then(pl.lit("pass"))
        .when(pl.col("harmonised_label").is_null()).then(pl.lit(None))
        .otherwise(pl.lit("flag"))
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=None,
                    help="backfill the paginated archive this many months back (default: RSS only)")
    ap.add_argument("--max", type=int, default=None, help="cap articles (debug)")
    ap.add_argument("--sleep", type=float, default=0.3, help="politeness delay between articles")
    args = ap.parse_args()

    if args.months:
        since = date.today() - timedelta(days=round(args.months * 30.44))
        print(f"Backfill: archive since {since} (~{args.months} months)")
        items = list(archive_urls(since))
    else:
        items = rss_items()
    if args.max:
        items = items[: args.max]
    print(f"source: {len(items)} fact-checks")

    # Skip URLs already harvested (RSS only carries ~20 recent items, so this is cheap).
    seen = set()
    if OUT.exists():
        seen = set(pl.read_parquet(OUT)["review_url"].to_list())
    todo = [it for it in items if it["url"] not in seen]
    print(f"new (not in {OUT.name}): {len(todo)} of {len(items)}")

    rows = []
    for it in tqdm(todo, desc="parse"):
        try:
            row = parse_article(it["url"])
        except Exception as e:  # noqa: BLE001
            print(f"  [fail] {it['url']}: {e}")
            continue
        if row.get("rating_slug") in FLIP_SLUGS:  # Flip-O-Meter, not a veracity rating
            continue
        if it.get("title"):
            row["review_title"] = it["title"]
        if it.get("review_date"):
            row["review_date"] = it["review_date"]
        rows.append(row)
        time.sleep(args.sleep)

    if not rows:
        print("0 new rows — nothing written.")
        return

    new = _harmonise(pl.DataFrame(rows, schema_overrides=SCHEMA))
    combined = new if not OUT.exists() else pl.concat([pl.read_parquet(OUT), new], how="diagonal_relaxed")
    combined.write_parquet(OUT)

    print(f"\nWrote {OUT}: +{new.height} new, {combined.height} total")
    print("=== new rows by harmonised_label ===")
    print(new.group_by("harmonised_label").agg(pl.len().alias("n")).sort("n", descending=True))
    print("=== new rows by original_rating ===")
    print(new.group_by("original_rating").agg(pl.len().alias("n")).sort("n", descending=True))
    print(f"avg sources/row: {new['sources'].list.len().mean():.1f}  |  "
          f"rows with claim_date: {new.filter(pl.col('claim_date').is_not_null()).height}/{new.height}")


if __name__ == "__main__":
    main()
