"""20 Minutes "Fake Off" (FR) harvester -> rolling, deduped parquet.

  uv run python -m eval.scripts.harvesters.harvest_twentyminutes --months 3

GFC API discovery (FR) + on-page ClaimReview (eval.twentyminutes), numeric verdict harmonised via
eval.harmonize.numeric_map. Same shared schema; raw_claim filled later by eval.scripts.harvesters.fetch_sources.
"""
from __future__ import annotations

import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from eval.harmonize import numeric_map
from eval.twentyminutes import discover, parse_article

OUT = Path("eval/data/twentyminutes_harvest.parquet")
SCHEMA = {
    "claim_text": pl.Utf8, "original_rating": pl.Utf8, "rating_value": pl.Utf8,
    "claimant": pl.Utf8, "publisher_site": pl.Utf8, "publisher_name": pl.Utf8,
    "review_url": pl.Utf8, "review_date": pl.Utf8, "language_code": pl.Utf8,
    "claim_source_url": pl.Utf8, "raw_claim": pl.Utf8, "context": pl.Utf8, "sources": pl.List(pl.Utf8),
}


def _row(rec: dict) -> dict:
    return {
        "claim_text": rec["normalized_claim"], "original_rating": rec["verdict_raw"],
        "rating_value": rec["rating_value"], "claimant": rec["claimant"],
        "publisher_site": rec["publisher_site"], "publisher_name": rec["publisher_name"],
        "review_url": rec["review_url"], "review_date": rec["review_date"],
        "language_code": rec["language_code"], "claim_source_url": rec["claim_source_url"],
        "raw_claim": None, "context": rec["context"], "sources": rec.get("sources"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=20, help="GFC API discovery pages (100/page)")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    since = datetime.date.today() - datetime.timedelta(days=round(args.months * 30.44))
    since_iso = since.isoformat()
    urls = discover(max_age_days=round(args.months * 30.44) + 7, max_pages=args.max_pages)
    if args.max:
        urls = urls[: args.max]
    seen = set(pl.read_parquet(OUT)["review_url"].to_list()) if OUT.exists() else set()
    todo = [u for u in urls if u not in seen]
    print(f"API discovery (FR): {len(urls)} URLs, {len(todo)} new (vs {OUT.name})")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(tqdm(ex.map(parse_article, todo), total=len(todo), desc="20min"))

    rows = [_row(r) for r in recs if r and r.get("review_date") and r["review_date"] >= since_iso]
    print(f"parsed with ClaimReview + in-window: {len(rows)} of {len(todo)}")
    if not rows:
        print("0 rows — nothing written.")
        return

    new = pl.DataFrame(rows, schema_overrides=SCHEMA).with_columns(
        harmonised_label=pl.col("rating_value").map_elements(numeric_map, return_dtype=pl.Utf8)
    ).with_columns(
        binary_label=pl.when(pl.col("harmonised_label") == "Supported").then(pl.lit("pass"))
        .when(pl.col("harmonised_label").is_null()).then(pl.lit(None)).otherwise(pl.lit("flag"))
    )
    combined = new if not OUT.exists() else pl.concat([pl.read_parquet(OUT), new], how="diagonal_relaxed")
    combined.write_parquet(OUT)

    src = new.filter(pl.col("claim_source_url").is_not_null()).height
    print(f"\nWrote {OUT}: +{new.height} new, {combined.height} total")
    print(f"claim_source_url present: {src}/{new.height} ({src/new.height*100:.0f}%)")
    print("harmonised:", new.group_by("harmonised_label").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())


if __name__ == "__main__":
    main()
