"""Periodic Google Fact Check Tools harvester -> rolling, deduped eval dataset.

For each curated publisher, pulls recent ClaimReviews within a rolling window
(deliberately overlapping month-to-month to absorb fact-checker ingestion lag),
projects them with the same 10-field schema as build_eval.load_raw, harmonises
ratings identically (per-publisher rule table -> LLM fallback for fullfact.org and
rule-misses), and appends the rows that are new (by review_url) to a rolling master.

  uv run python -m eval.scripts.build_eval.harvest_incremental \
      --window-days 38 --max-pages 5

Outputs:
  - eval/data/eval_master.parquet              rolling, deduped master (rewritten)
  - eval/data/snapshots/eval_<YYYYMMDD>.parquet  the new rows from this run
  - eval/data/harvest_state.json               {last_run, n_total, n_new}

The Google Fact Check Tools API is free; defaults stay small. The FCT key comes
from GOOGLE_FCTAPI_KEY (loaded by config.py through the eval.harvest import).
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import polars as pl
from tqdm import tqdm

from config import FCTAPI_KEY
from eval.harmonize import GROQ_MODEL, llm_map, rule_map
from eval.harvest import paginate_claims

DATA_DIR = Path("eval/data")
MASTER = DATA_DIR / "eval_master.parquet"
SNAP_DIR = DATA_DIR / "snapshots"
STATE = DATA_DIR / "harvest_state.json"

# Copied verbatim from build_eval.py — keep in sync.
CURATED_PUBS = {
    "snopes.com", "factcheck.afp.com", "newschecker.in", "verafiles.org",
    "rumorscanner.com", "politifact.com", "factcheck.org", "fullfact.org",
}
LLM_PUBLISHERS = {"fullfact.org"}  # publishers whose ratings need the LLM

WINDOW_DAYS = 38  # monthly cadence, deliberately overlapping for ingestion lag
MAX_PAGES = 5     # hard cap on API pages per publisher

# The 10-field projection of build_eval.load_raw, pinned so an empty pull still
# yields a typed frame (every field is free text in eval_v1).
PROJECTION_SCHEMA = {
    "claim_text": pl.Utf8, "claimant": pl.Utf8, "claim_date": pl.Utf8,
    "publisher_site": pl.Utf8, "publisher_name": pl.Utf8, "review_url": pl.Utf8,
    "review_title": pl.Utf8, "review_date": pl.Utf8, "language_code": pl.Utf8,
    "original_rating": pl.Utf8,
}


def harvest_raw(window_days: int, max_pages: int, language: str) -> pl.DataFrame:
    """Pull + project ClaimReviews for the curated publishers (same projection as
    build_eval.load_raw). Dedups by review_url within the pull."""
    seen_urls: set[str] = set()
    rows = []
    for site in sorted(CURATED_PUBS):
        n0 = len(rows)
        for c in paginate_claims(
            review_publisher_site_filter=site,
            max_age_days=window_days,
            language_code=language,
            page_size=100,
            max_pages=max_pages,
        ):
            for r in c.get("claimReview", []):
                url = r.get("url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                rows.append(
                    {
                        "claim_text": c.get("text"),
                        "claimant": c.get("claimant"),
                        "claim_date": c.get("claimDate"),
                        "publisher_site": (r.get("publisher") or {}).get("site"),
                        "publisher_name": (r.get("publisher") or {}).get("name"),
                        "review_url": url,
                        "review_title": r.get("title"),
                        "review_date": r.get("reviewDate"),
                        "language_code": r.get("languageCode"),
                        "original_rating": r.get("textualRating"),
                    }
                )
        print(f"  {site:<20} +{len(rows) - n0}")
    df = pl.DataFrame(rows, schema=PROJECTION_SCHEMA)
    # A multi-review claim returned under one site filter can carry a non-curated
    # publisher's review; keep only curated ones (matches build_eval's filter).
    return df.filter(pl.col("publisher_site").is_in(CURATED_PUBS))


def harmonise(df: pl.DataFrame) -> pl.DataFrame:
    """Harmonise ratings to the 4-class scheme — verbatim logic of build_eval.main():
    rule table first, LLM fallback for fullfact.org + rule-misses, then binary_label."""
    rule_labels = [
        rule_map(row["publisher_site"], row["original_rating"])
        for row in df.iter_rows(named=True)
    ]
    df = df.with_columns(rule_label=pl.Series(rule_labels, dtype=pl.Utf8))

    needs_llm_mask = (
        pl.col("publisher_site").is_in(LLM_PUBLISHERS) | pl.col("rule_label").is_null()
    )
    needs_llm = df.filter(needs_llm_mask)
    rule_only = df.filter(~needs_llm_mask)
    print(
        f"Rule-based covers {len(rule_only)}; LLM needed for {len(needs_llm)} "
        f"({len(needs_llm)/max(len(df),1)*100:.1f}%)"
    )

    llm_labels = []
    failures = 0
    for row in tqdm(needs_llm.iter_rows(named=True), total=len(needs_llm), desc="LLM"):
        try:
            llm_labels.append(llm_map(GROQ_MODEL, row["original_rating"] or ""))
        except Exception as e:  # noqa: BLE001
            failures += 1
            llm_labels.append(None)
            print(f"  [fail] {row['publisher_site']} {row['original_rating']!r}: {e}")

    needs_llm = needs_llm.with_columns(llm_label=pl.Series(llm_labels, dtype=pl.Utf8))
    rule_only = rule_only.with_columns(llm_label=pl.lit(None).cast(pl.Utf8))

    merged = pl.concat([rule_only, needs_llm]).with_columns(
        harmonised_label=pl.coalesce(pl.col("rule_label"), pl.col("llm_label")),
        harmonisation_source=pl.when(pl.col("rule_label").is_not_null())
        .then(pl.lit("rule"))
        .otherwise(pl.lit("llm")),
    )
    # Binary "deployment" label: PASS only if the claim is verified true.
    # Refuted / Conflicting Evidence / Not Enough Evidence all -> FLAG.
    merged = merged.with_columns(
        binary_label=pl.when(pl.col("harmonised_label") == "Supported")
        .then(pl.lit("pass"))
        .when(pl.col("harmonised_label").is_null())
        .then(pl.lit(None))
        .otherwise(pl.lit("flag"))
    )
    if failures:
        print(f"LLM harmonisation failures (left unlabeled): {failures}")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES)
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    if not FCTAPI_KEY:
        raise SystemExit(
            "GOOGLE_FCTAPI_KEY not set (config.FCTAPI_KEY is empty). "
            "Add it to src/.env and retry."
        )

    SNAP_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Harvesting {len(CURATED_PUBS)} publishers | window={args.window_days}d "
        f"max_pages={args.max_pages} lang={args.language}"
    )
    raw = harvest_raw(args.window_days, args.max_pages, args.language)
    print(f"Pulled {len(raw)} unique ClaimReviews (curated publishers)")

    # DEDUP first (anti-join on review_url) so we never re-harmonise — and never
    # re-spend LLM calls on — rows already in the master.
    if MASTER.exists():
        master = pl.read_parquet(MASTER)
        new_raw = raw.join(master.select("review_url"), on="review_url", how="anti")
    else:
        master = None
        new_raw = raw
    print(f"New after dedup vs master: {len(new_raw)} of {len(raw)} pulled")

    new = harmonise(new_raw)
    combined = new if master is None else pl.concat([master, new])

    combined.write_parquet(MASTER)
    today = datetime.date.today()
    snap_path = SNAP_DIR / f"eval_{today.strftime('%Y%m%d')}.parquet"
    if not new.is_empty():
        new.write_parquet(snap_path)

    state = {"last_run": today.isoformat(), "n_total": combined.height, "n_new": new.height}
    STATE.write_text(json.dumps(state, indent=2))

    print("\n=== Harvest summary ===")
    print(f"pulled (this run):  {raw.height}")
    print(f"new after dedup:    {new.height}")
    print(f"master total:       {combined.height} -> {MASTER}")
    print(f"snapshot:           {snap_path if not new.is_empty() else '(none — 0 new rows)'}")
    print(f"state:              {STATE}")

    print("\n=== New rows by harmonised_label ===")
    print(new.group_by("harmonised_label").agg(pl.len().alias("n")).sort("n", descending=True))
    print("=== New rows by binary_label ===")
    print(new.group_by("binary_label").agg(pl.len().alias("n")).sort("n", descending=True))


if __name__ == "__main__":
    main()
