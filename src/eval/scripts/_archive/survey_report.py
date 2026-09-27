"""Phase 1 report: monthly volume per publisher + cost estimate."""
import json
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

import polars as pl

SURVEY_PATH = Path("eval/data/survey/12mo.jsonl")
PAGE_SIZE_HARVEST = 100  # what we'd use in production


def main() -> None:
    rows = []
    for line in SURVEY_PATH.read_text().splitlines():
        c = json.loads(line)
        for r in c.get("claimReview", []):
            rows.append(
                {
                    "publisher_site": r.get("publisher", {}).get("site"),
                    "publisher_name": r.get("publisher", {}).get("name"),
                    "review_date": r.get("reviewDate"),
                    "textual_rating": r.get("textualRating"),
                    "language_code": r.get("languageCode"),
                    "url": r.get("url"),
                    "harvest_site": c.get("_harvest_publisher_site"),
                }
            )

    df = pl.DataFrame(rows)
    print(f"Total review rows: {len(df)}")
    print(f"Unique URLs: {df['url'].n_unique()}")

    # Monthly volume per harvest_site (= what we'd see if we ran the harvester for that publisher)
    df_m = (
        df.with_columns(month=pl.col("review_date").str.slice(0, 7))
        .group_by(["harvest_site", "month"])
        .agg(pl.len().alias("n"))
        .sort(["harvest_site", "month"])
    )

    # Wide table
    pivot = df_m.pivot(values="n", index="month", on="harvest_site").sort("month")
    print()
    print("=== Monthly volume per publisher (review_date) ===")
    print(pivot)

    # Per-publisher totals + cost estimate (requests/month)
    print()
    print("=== Per-publisher: average claims/month → requests/month estimate ===")
    print(f"(assuming pageSize={PAGE_SIZE_HARVEST}, monthly harvest with maxAgeDays~30)")
    print()
    print(f"  {'publisher':<28} {'12-mo total':>12} {'avg/mo':>10} {'req/mo':>10}")
    total_req_per_month = 0
    for site in sorted(df['harvest_site'].unique().to_list()):
        n = df.filter(pl.col("harvest_site") == site).height
        avg = n / 12
        req = max(1, ceil(avg / PAGE_SIZE_HARVEST))
        total_req_per_month += req
        print(f"  {site:<28} {n:>12} {avg:>10.1f} {req:>10}")
    print()
    print(f"  Estimated total requests per monthly harvest: ~{total_req_per_month}")
    print(f"  (Realistic ceiling with some pagination overhead: ~{total_req_per_month + len(df['harvest_site'].unique())})")

    # Top rating strings
    print()
    print("=== Top 30 textualRating strings (harmonisation signal) ===")
    ratings = (
        df.group_by("textual_rating")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(30)
    )
    print(ratings)


if __name__ == "__main__":
    main()
