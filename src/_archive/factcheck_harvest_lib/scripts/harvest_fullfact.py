"""Full Fact harvester -> rolling, deduped parquet (GFC API discovery + on-page ClaimReview scrape).

  uv run python -m eval.scripts.harvest_fullfact --months 12
  uv run python -m eval.scripts.harvest_fullfact --months 12 --max 15   # validate

Discovers review URLs via the GFC API (eval.fullfact.discover), scrapes each page for the inline
ClaimReview (claim + free-text verdict), the embedded checked-post URL, and the evidence Full Fact
cites inline in the body (eval.fullfact.parse_article). Verdict is a free-text sentence → harmonised
by harmonize.llm_map (PAID — one call per new row). Same shared schema as the other harvesters; the
verbatim raw claim + claim_date are filled later by eval.scripts.fetch_sources (X-post resolution).
"""
from __future__ import annotations

import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from config import EXTRACTION_MODEL
from eval.fullfact import PUBLISHER_SITE, discover, parse_article
from eval.harmonize import llm_map

OUT = Path("eval/data/fullfact_harvest.parquet")
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
        "rating_value": rec.get("rating_value"), "claimant": rec.get("claimant"),
        "publisher_site": rec["publisher_site"], "publisher_name": rec.get("publisher_name") or "Full Fact",
        "review_url": rec["review_url"], "review_date": rec["review_date"],
        "language_code": rec["language_code"], "claim_source_url": rec.get("claim_source_url"),
        "raw_claim": None, "context": rec.get("context"), "sources": rec.get("sources"),
    }


def _harmonise(rating: str | None) -> str | None:
    """LLM-only — Full Fact verdicts are free-text sentences (no rule table)."""
    if not rating:
        return None
    try:
        return llm_map(EXTRACTION_MODEL, rating)
    except Exception as e:  # network / non-canonical → leave null, reported below
        print(f"  llm_map failed for {rating!r}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=8)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    since_iso = (datetime.date.today() - datetime.timedelta(days=round(args.months * 30.44))).isoformat()
    urls = discover(max_age_days=round(args.months * 30.44) + 7, max_pages=args.max_pages)
    if args.max:
        urls = urls[: args.max]
    seen = set(pl.read_parquet(OUT)["review_url"].to_list()) if OUT.exists() else set()
    todo = [u for u in urls if u not in seen]
    print(f"API discovery: {len(urls)} URLs, {len(todo)} new (vs {OUT.name})")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(tqdm(ex.map(parse_article, todo), total=len(todo), desc="fullfact"))
    recs = [r for r in recs if r and r.get("review_date") and r["review_date"] >= since_iso]
    print(f"parsed with ClaimReview + in-window: {len(recs)}")
    if not recs:
        print("0 rows — nothing written.")
        return

    with ThreadPoolExecutor(max_workers=args.workers) as ex:  # PAID: one llm_map per row
        labels = list(tqdm(ex.map(lambda r: _harmonise(r["verdict_raw"]), recs), total=len(recs), desc="harmonise"))
    rows = [{**_row(r), "harmonised_label": lab} for r, lab in zip(recs, labels)]

    new = pl.DataFrame(rows, schema_overrides=SCHEMA).with_columns(
        binary_label=pl.when(pl.col("harmonised_label") == "Supported").then(pl.lit("pass"))
        .when(pl.col("harmonised_label").is_null()).then(pl.lit(None)).otherwise(pl.lit("flag"))
    )
    combined = new if not OUT.exists() else pl.concat([pl.read_parquet(OUT), new], how="diagonal_relaxed")
    combined.write_parquet(OUT)

    src = new.filter(pl.col("claim_source_url").is_not_null()).height
    cit = new.filter(pl.col("sources").list.len() > 0).height
    miss = new.filter(pl.col("harmonised_label").is_null()).height
    print(f"\nWrote {OUT}: +{new.height} new, {combined.height} total")
    print(f"checked-post URL: {src}/{new.height} ({src/new.height*100:.0f}%)  |  "
          f"cited sources: {cit}/{new.height} ({cit/new.height*100:.0f}%)  |  LLM-miss: {miss}")
    print("harmonised:", new.group_by("harmonised_label").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())


if __name__ == "__main__":
    main()
