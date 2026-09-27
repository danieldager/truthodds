"""Lead Stories harvester -> rolling, deduped parquet (the first generic-ClaimReview source).

  uv run python -m eval.scripts.harvesters.harvest_leadstories --months 3
  uv run python -m eval.scripts.harvesters.harvest_leadstories --months 3 --max 30   # validate

Discovers the window from sitemap.txt (eval.leadstories), parses each article's ClaimReview
(eval.claimreview), harmonises the numeric `ratingValue` (eval.harmonize.numeric_map — Lead
Stories' `alternateName` is a topical tag, not a verdict), and writes the SHARED record schema
(eval-aligned column names + the bonus `claim_source_url`/`raw_claim`/`context`). The verbatim
raw claim (`raw_claim`) is left null here; the separate source-fetch pass fills it from
`claim_source_url` best-effort.
"""
from __future__ import annotations

import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from eval.harmonize import refine_numeric
from eval.leadstories import parse_article, sitemap_urls

OUT = Path("eval/data/leadstories_harvest.parquet")
SCHEMA = {
    "claim_text": pl.Utf8, "original_rating": pl.Utf8, "rating_value": pl.Utf8,
    "claimant": pl.Utf8, "publisher_site": pl.Utf8, "publisher_name": pl.Utf8,
    "review_url": pl.Utf8, "review_date": pl.Utf8, "language_code": pl.Utf8,
    "claim_source_url": pl.Utf8, "raw_claim": pl.Utf8, "context": pl.Utf8, "sources": pl.List(pl.Utf8),
}


def _row(rec: dict) -> dict:
    """Generic ClaimReview record -> eval-aligned shared row."""
    return {
        "claim_text": rec["normalized_claim"],
        "original_rating": rec["verdict_raw"],       # Lead Stories' topical tag (kept for info)
        "rating_value": rec["rating_value"],
        "claimant": rec["claimant"],
        "publisher_site": rec["publisher_site"],
        "publisher_name": rec["publisher_name"],
        "review_url": rec["review_url"],
        "review_date": rec["review_date"],
        "language_code": rec["language_code"],
        "claim_source_url": rec["claim_source_url"],
        "raw_claim": None,                           # filled by the source-fetch pass
        "context": rec["context"],
        "sources": rec.get("sources"),               # evidence cited (checked post excluded)
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    since = datetime.date.today() - datetime.timedelta(days=round(args.months * 30.44))
    since_iso = since.isoformat()
    urls = sitemap_urls(since)
    if args.max:
        urls = urls[: args.max]
    seen = set(pl.read_parquet(OUT)["review_url"].to_list()) if OUT.exists() else set()
    todo = [u for u in urls if u not in seen]
    print(f"window since {since_iso}: {len(urls)} candidates, {len(todo)} new (vs {OUT.name})")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(tqdm(ex.map(parse_article, todo), total=len(todo), desc="leadstories"))

    rows = [_row(r) for r in recs if r and r.get("review_date") and r["review_date"] >= since_iso]
    print(f"parsed with ClaimReview + in-window: {len(rows)} of {len(todo)}")
    if not rows:
        print("0 rows — nothing written.")
        return

    new = pl.DataFrame(rows, schema_overrides=SCHEMA).with_columns(
        harmonised_label=pl.struct(["original_rating", "rating_value"]).map_elements(
            lambda s: refine_numeric(s["original_rating"], s["rating_value"]), return_dtype=pl.Utf8)
    ).with_columns(
        binary_label=pl.when(pl.col("harmonised_label") == "Supported").then(pl.lit("pass"))
        .when(pl.col("harmonised_label").is_null()).then(pl.lit(None)).otherwise(pl.lit("flag"))
    )
    combined = new if not OUT.exists() else pl.concat([pl.read_parquet(OUT), new], how="diagonal_relaxed")
    combined.write_parquet(OUT)

    src = new.filter(pl.col("claim_source_url").is_not_null()).height
    print(f"\nWrote {OUT}: +{new.height} new, {combined.height} total")
    print(f"claim_source_url present: {src}/{new.height} ({src/new.height*100:.0f}%) — extraction-pair seeds")
    print("harmonised:", new.group_by("harmonised_label").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())
    print("date span:", new["review_date"].min(), "->", new["review_date"].max())


if __name__ == "__main__":
    main()
