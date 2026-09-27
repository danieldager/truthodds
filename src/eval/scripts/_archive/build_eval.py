"""Build the harmonised eval dataset (v1).

- Load both raw survey JSONLs.
- Filter to 8 curated publishers × last 6 months (Nov 2025 – Apr 2026).
- Apply rule-based harmonisation for 7 publishers.
- LLM-harmonise Full Fact via Groq + Qwen3-32B.
- Save as Parquet (eval_v1.parquet) and a TSV summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from tqdm import tqdm

from eval.harmonize import GROQ_MODEL, llm_map, rule_map

SOURCE_JSONLS = [
    Path("eval/data/survey/12mo.jsonl"),
    Path("eval/data/survey/api_publishers.jsonl"),
]
OUT_PARQUET = Path("eval/data/eval_v1.parquet")
OUT_SUMMARY = Path("eval/data/eval_v1_summary.tsv")

CURATED_PUBS = {
    "snopes.com", "factcheck.afp.com", "newschecker.in", "verafiles.org",
    "rumorscanner.com", "politifact.com", "factcheck.org", "fullfact.org",
}
WINDOW_MONTHS = {"2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"}
LLM_PUBLISHERS = {"fullfact.org"}  # publishers whose ratings need the LLM


def load_raw() -> pl.DataFrame:
    seen_urls: set[str] = set()
    rows = []
    for p in SOURCE_JSONLS:
        for line in p.read_text().splitlines():
            c = json.loads(line)
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
    return pl.DataFrame(rows)


def main() -> None:
    df = load_raw()
    print(f"Raw unique claims (across both surveys): {len(df)}")

    # Filter to curated set + window
    df = df.with_columns(month=pl.col("review_date").str.slice(0, 7)).filter(
        pl.col("publisher_site").is_in(CURATED_PUBS)
        & pl.col("month").is_in(WINDOW_MONTHS)
    )
    print(f"After curated-set + 6-month window filter: {len(df)}")

    # Apply rule-based harmonisation
    rule_labels = [
        rule_map(row["publisher_site"], row["original_rating"])
        for row in df.iter_rows(named=True)
    ]
    df = df.with_columns(rule_label=pl.Series(rule_labels))

    # Rows needing LLM: Full Fact + any rule-fallthrough on other publishers
    needs_llm_mask = (
        pl.col("publisher_site").is_in(LLM_PUBLISHERS) | pl.col("rule_label").is_null()
    )
    needs_llm = df.filter(needs_llm_mask)
    rule_only = df.filter(~needs_llm_mask)
    print(
        f"Rule-based covers {len(rule_only)}; LLM needed for {len(needs_llm)} "
        f"({len(needs_llm)/len(df)*100:.1f}%)"
    )

    # LLM pass
    llm_labels = []
    failures = 0
    for row in tqdm(needs_llm.iter_rows(named=True), total=len(needs_llm), desc="LLM"):
        try:
            llm_labels.append(llm_map(GROQ_MODEL, row["original_rating"] or ""))
        except Exception as e:
            failures += 1
            llm_labels.append(None)
            print(f"  [fail] {row['publisher_site']} {row['original_rating']!r}: {e}")

    needs_llm = needs_llm.with_columns(llm_label=pl.Series(llm_labels))
    rule_only = rule_only.with_columns(llm_label=pl.lit(None).cast(pl.Utf8))

    merged = pl.concat([rule_only, needs_llm]).with_columns(
        harmonised_label=pl.coalesce(pl.col("rule_label"), pl.col("llm_label")),
        harmonisation_source=pl.when(pl.col("rule_label").is_not_null())
        .then(pl.lit("rule"))
        .otherwise(pl.lit("llm")),
    )
    # Binary "deployment" label: PASS only if the claim is verified true.
    # Refuted / Conflicting Evidence / Not Enough Evidence all → FLAG
    # (the extension nudges whenever there's any concern, including unverifiability).
    merged = merged.with_columns(
        binary_label=pl.when(pl.col("harmonised_label") == "Supported")
        .then(pl.lit("pass"))
        .when(pl.col("harmonised_label").is_null())
        .then(pl.lit(None))
        .otherwise(pl.lit("flag"))
    )

    n_unlabeled = merged.filter(pl.col("harmonised_label").is_null()).height
    print(f"Final unlabeled (both rule and LLM failed): {n_unlabeled}")

    merged.write_parquet(OUT_PARQUET)
    print(f"Wrote {OUT_PARQUET} ({len(merged)} rows)")

    # Summary
    summary = (
        merged.group_by("harmonised_label")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    print("\n=== Label distribution ===")
    print(summary)

    by_pub = (
        merged.group_by(["publisher_site", "harmonised_label"])
        .agg(pl.len().alias("n"))
        .sort(["publisher_site", "n"], descending=[False, True])
    )
    print("\n=== Per-publisher label distribution ===")
    with pl.Config(tbl_rows=60):
        print(by_pub)
    by_pub.write_csv(OUT_SUMMARY, separator="\t")


if __name__ == "__main__":
    main()
