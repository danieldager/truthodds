"""fc-gold v2 admitted selection — Daniel's publisher ruling (2026-07-23).

Admitted = the 8 clean-admit publishers from the reliability audits + politifact +
snopes (audits: eval/data/publisher_audits.md; page: fc_gold_v2_audit.html).
Excluded by ruling: leadstories, factly, vishvasnews, altnews, rumorscanner,
defacto-observatoire (aggregator). Publishers under 100 in-window rows were not
audited and stay out until audited.

Window: claim_date >= 2020 OR review_date >= 2020. Rows keep all flags
(rating_subtype/nee/contested/judged_axis) — media-authenticity and satire exclusions
are applied per-analysis, not here.

  uv run python -m eval.scripts.build_eval.build_fc_gold_selection
Output: eval/data/fc_gold_v2_admitted.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DATA = Path("eval/data")
OUT = DATA / "fc_gold_v2_admitted.parquet"

ADMITTED = {
    # clean admits
    "factcheck.afp.com", "factuel.afp.com", "fullfact.org", "aap.com.au",
    "africacheck.org", "factcheck.org", "boomlive.in", "verafiles.org",
    "ghanafact.com",
    # admit-with-caveats explicitly included by Daniel
    "politifact.com", "snopes.com",
}


def main():
    df = pl.read_parquet(DATA / "fc_gold_v2.parquet")
    sel = df.filter(
        ((pl.col("review_date") >= "2020") | (pl.col("claim_date") >= "2020"))
        & pl.col("publisher_site").is_in(sorted(ADMITTED))
    )
    sel.write_parquet(OUT)
    print(f"wrote {OUT}: {len(sel)} rows from {sel['publisher_site'].n_unique()} publishers")
    print("by veracity:", dict(sel.group_by("veracity").len().sort("veracity").iter_rows()))
    print("true side (4+5):", len(sel.filter(pl.col("veracity") >= 4)),
          "| contested:", sel["contested"].sum(), "| nee:", sel["nee"].sum())
    print("media-authenticity (altered_media subtype):",
          len(sel.filter(pl.col("rating_subtype") == "altered_media")),
          "| satire:", len(sel.filter(pl.col("rating_subtype") == "satire")))
    print(sel.group_by("publisher_site").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
