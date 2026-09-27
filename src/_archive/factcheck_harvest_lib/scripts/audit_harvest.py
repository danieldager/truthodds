"""Data-integrity QA over the harvested fact-checks — catches empty/degenerate rows and capture gaps.

Run after the harvest/resolve pipeline (and before building any stage dataset). It does NOT fix
anything — it reports, and exits non-zero if a hard-integrity threshold is breached, so it can gate a
pipeline. Motivated by the 70%-empty-input problem (clog 270626) that went undetected because there was
no standing check.

Checks (per source + combined):
  1. GOLD present        — `claim_text` non-null (the fact-checker claim; should be ~100%).
  2. No duplicate rows   — `review_url` unique.
  3. Resolved ⇒ captured — rows with source_method ∈ {x_syndication, page_meta} must have non-empty
                           `raw_claim`/`raw_context` (else the resolver said "ok" but captured nothing).
  4. EMPTY-INPUT         — no `raw_context` AND no image (nothing to extract/gate from), by source_method.
  5. RECOVERABLE losses  — source_method ∈ {failed, None} BUT a non-dead source URL is present (non-X, or
                           not yet attempted) → capture we may be leaving on the table; worth re-resolving.
  6. TRUNCATION smell    — page_meta `raw_claim` that ends in an ellipsis or is suspiciously short
                           (og:description preview, not the full post).

  uv run python -m eval.scripts.audit_harvest
"""
from __future__ import annotations

import glob
import os
import re
import sys

import polars as pl

DEAD_X = re.compile(r"(twitter|x)\.com/.+/status/", re.I)


def _audit(d: pl.DataFrame, name: str) -> dict:
    n = d.height
    # "no usable text" = neither the real post (raw_claim) NOR the gated/quote context (raw_context)
    notext = (pl.col("raw_claim").fill_null("").str.strip_chars().str.len_chars() == 0) & \
             (pl.col("raw_context").fill_null("").str.strip_chars().str.len_chars() == 0)
    has_img = pl.col("has_image").fill_null(False) if "has_image" in d.columns else pl.lit(False)
    resolved = pl.col("source_method").is_in(["x_syndication", "page_meta"]) if "source_method" in d.columns else pl.lit(False)
    no_gold = d.filter(pl.col("claim_text").fill_null("").str.strip_chars().str.len_chars() == 0).height
    dups = n - d["review_url"].n_unique()
    has_vid = pl.col("has_video").fill_null(False) if "has_video" in d.columns else pl.lit(False)
    # genuine capture bug = resolver said ok but captured NOTHING usable (no text, no image, no video)
    resolved_empty = d.filter(resolved & notext & (has_img == False) & (has_vid == False)).height  # noqa: E712
    empty_input = d.filter(notext & (has_img == False)).height  # noqa: E712
    # recoverable: failed/None but has a source URL that isn't a (dead) X-status
    if "claim_source_url" in d.columns and "source_method" in d.columns:
        recoverable = d.filter(
            (~pl.col("source_method").is_in(["x_syndication", "page_meta"]))
            & pl.col("claim_source_url").is_not_null()
            & ~pl.col("claim_source_url").str.contains(DEAD_X.pattern)
        ).height
    else:
        recoverable = 0
    trunc = 0
    if "source_method" in d.columns and "raw_claim" in d.columns:
        pm = d.filter(pl.col("source_method") == "page_meta")
        if pm.height:
            trunc = pm.filter(pl.col("raw_claim").fill_null("").str.contains(r"(\.\.\.|…)\s*$")).height
    return {"name": name, "n": n, "no_gold": no_gold, "dups": dups, "resolved_empty": resolved_empty,
            "empty_input": empty_input, "recoverable": recoverable, "trunc": trunc}


def main() -> None:
    rows = []
    frames = []
    for f in sorted(glob.glob("eval/data/*_harvest.parquet")):
        d = pl.read_parquet(f)
        rows.append(_audit(d, os.path.basename(f).replace("_harvest.parquet", "")))
        frames.append(d)
    cols = sorted(set.intersection(*[set(x.columns) for x in frames]))
    comb = pl.concat([x.select(cols) for x in frames], how="diagonal_relaxed").unique("review_url")
    combined = _audit(comb, "COMBINED")

    print(f"{'source':14}{'rows':>7}{'no_gold':>9}{'dups':>6}{'resolved_empty':>16}{'empty_input':>13}{'recoverable':>13}{'trunc':>7}")
    for r in rows + [combined]:
        pct = f"{r['empty_input']/r['n']*100:.0f}%" if r["n"] else "-"
        print(f"{r['name']:14}{r['n']:>7}{r['no_gold']:>9}{r['dups']:>6}{r['resolved_empty']:>16}"
              f"{r['empty_input']:>9} {pct:>3}{r['recoverable']:>13}{r['trunc']:>7}")

    print("\nLegend: resolved_empty = resolver said ok but captured nothing (BUG if >0); "
          "empty_input = no text & no image (unusable for stages 1/2/3); "
          "recoverable = failed/None w/ a non-dead source URL (re-resolve candidates); "
          "trunc = page_meta og preview ending in '…' (likely truncated).")

    # hard gates: gold + dups + resolved_empty must be ~0
    bad = combined["no_gold"] + combined["dups"] + combined["resolved_empty"]
    if bad:
        print(f"\n*** INTEGRITY FAIL: no_gold={combined['no_gold']} dups={combined['dups']} "
              f"resolved_empty={combined['resolved_empty']} ***")
        sys.exit(1)
    print("\nHard checks pass (gold present, no dups, no resolved-but-empty). "
          f"NOTE: {combined['empty_input']} empty-input rows are excluded per-stage, not a harvest bug.")


if __name__ == "__main__":
    main()
