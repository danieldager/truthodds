"""Stage-3 scope FLOOR — the DETERMINISTIC media-dependency carve, as a join-able mask.

A row belongs in the Stage-3 *content-veracity* headline ONLY IF a text-only verifier searching the
web for the claim's CONTENT could in principle reach the gold verdict. This script produces the cheap,
no-API floor of that decision so the scorer can slice a defensible headline TODAY:

  uv run python -m eval.scripts.build_stage3_scope_floor   ->  eval/data/verdict_scope_floor.parquet

`stage3_in_scope` = (judged_axis == "content") AND NOT media-dependent.
`out_of_scope_reason` ∈ {attribution, artifact, video, provenance_recency, none}
  - attribution        -> "did X say Y" (separate slice, not the content headline)
  - artifact           -> media authenticity/provenance (judged_axis rule already caught it)
  - video              -> has_video; a text-only verifier can't watch the clip
  - provenance_recency -> "years old / recycled / out of context / from <year>" — the recency family
                          _AX_ARTIFACT misses; here it's a regex CANDIDATE flag.

LIMITS (why this is a floor, not the final word) — the LLM sighted scope pass
(`verdict_scope_audit.parquet`, not built yet) refines all three:
  - has_video OVER-carves: the real test is video-*locus* (does the verdict need the clip). Per the
    gate_video_locus finding ~75% of video posts carry a checkable text/poster-frame claim. The floor
    blanket-carves video; the LLM pass keeps the text-locus ones.
  - the recency regex MIS-fires (e.g. "deaths were unrelated to renewable energy" = content causation,
    not provenance). The LLM pass confirms each recency candidate.
  - the floor MISSES media_borne_unserialized (claim lives in the image, not carried by text/serial)
    and claim_proposition_mismatch (decontextualised claim_text doesn't carry the judged proposition).
    Those need the sighted model.

Never rewrites verdict_dataset.parquet (shared-read). Additive, no paid calls.
"""
from __future__ import annotations

import re
from pathlib import Path

import polars as pl

D = Path("eval/data")
GOLD = D / "verdict_dataset.parquet"
OUT = D / "verdict_scope_floor.parquet"

# recency / out-of-context family — the gap _AX_ARTIFACT misses. Candidate flag (LLM confirms).
_RECENCY = re.compile(
    r"years?\s+old|recycled|\bold (?:footage|video|photo|image|clip|picture)\b|predat"
    r"|out of context|from \d{4}|different (?:event|place|time|incident|country|location|protest)"
    r"|filmed in \d{4}|dates? back|several years|resurfac",
    re.I,
)


def main() -> None:
    g = pl.read_parquet(GOLD).filter(pl.col("in_gold"))
    g = g.with_columns([
        pl.col("has_video").fill_null(False).alias("_vid"),
        pl.col("original_rating").map_elements(
            lambda s: bool(_RECENCY.search(s or "")), return_dtype=pl.Boolean).alias("_rec"),
    ])
    # precedence: axis (attribution/artifact) first, then media-dependency within content (video, recency)
    reason = (
        pl.when(pl.col("judged_axis") == "attribution").then(pl.lit("attribution"))
        .when(pl.col("judged_axis") == "artifact").then(pl.lit("artifact"))
        .when(pl.col("_vid")).then(pl.lit("video"))
        .when(pl.col("_rec")).then(pl.lit("provenance_recency"))
        .otherwise(pl.lit("none"))
    )
    out = g.with_columns(reason.alias("out_of_scope_reason"))
    out = out.with_columns((pl.col("out_of_scope_reason") == "none").alias("stage3_in_scope"))
    out = out.select(["review_url", "split", "balanced", "gold_veracity", "rating_subtype",
                      "judged_axis", "stage3_in_scope", "out_of_scope_reason",
                      pl.col("_vid").alias("has_video"), pl.col("_rec").alias("recency_candidate")])
    out.write_parquet(OUT)

    print(f"wrote {out.height} rows -> {OUT}")
    for name, df in [("ALL in_gold", out),
                     ("balanced DEV", out.filter((pl.col("split") == "dev") & pl.col("balanced"))),
                     ("balanced TEST", out.filter((pl.col("split") == "test") & pl.col("balanced")))]:
        insc = df.filter(pl.col("stage3_in_scope"))
        print(f"\n[{name}] total {df.height} | IN-SCOPE {insc.height} "
              f"({100 * insc.height / df.height:.0f}%) | carved {df.height - insc.height}")
        print("  out_of_scope_reason:",
              dict(df.group_by("out_of_scope_reason").len().sort("len", descending=True).iter_rows()))
        print("  in-scope veracity:",
              dict(insc["gold_veracity"].value_counts().sort("gold_veracity").iter_rows()))


if __name__ == "__main__":
    main()
