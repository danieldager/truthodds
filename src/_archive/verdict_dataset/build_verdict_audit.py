"""WS2 — assemble the verdict-gold AUDIT VIEW: join the in_gold gold (verdict_dataset.parquet) to the
two-model Pass B sighted audit (audit_results.parquet, 235B 'lab' + Gemma-3-27B 'x'), and DERIVE the
borderline / needs-human flags (computed here, never asked of the model — cf. dataset_methodology §4).

  uv run python -m eval.scripts.build_verdict_audit   ->  eval/data/verdict_audit.parquet

This NEVER rewrites verdict_dataset.parquet (shared-read). The flags drive build_verdict_review.py (human
queue) and verdict_dataset_provenance.py (ledger + cleaned eval set). Decisions locked w/ Daniel 2026-06-28:
  - judged_axis: BOTH audit models agree on an axis ≠ the rule  -> reclassify (model-consensus);
    models disagree (and ≥1 ≠ rule)                              -> human review.
  - rating->veracity: EITHER model says the gold is wrong       -> human review (veracity is the headline label).
  - satire: KEPT, not dropped (a False-rated satirical claim is genuinely false content) — flagged only.
  - unprovable: surface rows a model doesn't read as v3 / subtype-not-ok (protects the gold-3 split, WS4 H2).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

D = Path("eval/data")
GOLD = D / "verdict_dataset.parquet"
AUDIT = D / "audit_results.parquet"
OUT = D / "verdict_audit.parquet"

GOLD_COLS = ["review_url", "claim_text", "original_rating", "rating_value", "publisher_site", "claimant",
             "gold_veracity", "rating_subtype", "judged_axis", "has_image", "image_paths", "raw_context",
             "language_code", "source", "split", "balanced"]
B = ["judged_axis", "is_satire", "veracity_agrees", "suggested_veracity", "rating_subtype_ok",
     "claim_matches_post", "image_supports_claim", "note"]


def main() -> None:
    gold = pl.read_parquet(GOLD).select(GOLD_COLS).rename({"judged_axis": "axis_rule"})
    aud = pl.read_parquet(AUDIT).filter(pl.col("B_lab_veracity_agrees").is_not_null())
    acols = ["key"] + [f"B_lab_{f}" for f in B] + [f"B_x_{f}" for f in B]
    aud = aud.select([c for c in acols if c in aud.columns])

    j = gold.join(aud, left_on="review_url", right_on="key", how="left")
    j = j.with_columns(pl.col("B_lab_veracity_agrees").is_not_null().alias("audited"))

    a = j.filter(pl.col("audited"))  # only audited rows carry flags
    T = True

    # ---- derived flags (pure functions of the two models' labels + the rule) ----
    # rating -> veracity disagreement: compare each model's INDEPENDENT suggested_veracity to the CURRENT
    # gold. This is robust to gold rebuilds (the harmonize free-text fix moved labels under us); the stored
    # `veracity_agrees` was recorded against the gold shown AT AUDIT TIME and goes stale, suggested_veracity
    # does not. lab_off/x_off = |suggested - gold|.
    j = j.with_columns([
        (pl.col("B_lab_suggested_veracity").is_not_null() & (pl.col("B_lab_suggested_veracity") != pl.col("gold_veracity"))).alias("lab_off"),
        (pl.col("B_x_suggested_veracity").is_not_null() & (pl.col("B_x_suggested_veracity") != pl.col("gold_veracity"))).alias("x_off"),
    ])
    j = j.with_columns([
        (pl.col("lab_off") | pl.col("x_off")).alias("veracity_disagree"),
        (pl.col("lab_off") & pl.col("x_off")).alias("veracity_both_disagree"),
        ((pl.col("B_lab_rating_subtype_ok") == False) | (pl.col("B_x_rating_subtype_ok") == False)).alias("subtype_disagree"),  # noqa: E712
        # judged_axis: both models agree with each other
        (pl.col("B_lab_judged_axis") == pl.col("B_x_judged_axis")).alias("axis_both_agree"),
        # satire: informational only (Daniel: keep, don't drop)
        ((pl.col("B_lab_is_satire") == True) | (pl.col("B_x_is_satire") == True)).alias("satire_flag"),  # noqa: E712
        # image_supports disagreement (image rows only)
        ((pl.col("B_lab_image_supports_claim") == False) | (pl.col("B_x_image_supports_claim") == False)).alias("image_unsupported"),  # noqa: E712
    ])
    # axis_reclassify_to: both models agree on an axis that differs from the rule -> model-consensus override
    j = j.with_columns(
        pl.when(pl.col("audited") & pl.col("axis_both_agree") & (pl.col("B_lab_judged_axis") != pl.col("axis_rule")))
        .then(pl.col("B_lab_judged_axis")).otherwise(None).alias("axis_reclassify_to"))
    # axis_needs_human: models disagree with each other AND at least one differs from the rule
    j = j.with_columns(
        (pl.col("audited") & (~pl.col("axis_both_agree"))
         & ((pl.col("B_lab_judged_axis") != pl.col("axis_rule")) | (pl.col("B_x_judged_axis") != pl.col("axis_rule")))
         ).alias("axis_needs_human"))
    # unprovable suspects: a model doesn't read it as the no-evidence v3
    j = j.with_columns(
        (pl.col("audited") & (pl.col("rating_subtype") == "unprovable")
         & ((pl.col("B_lab_rating_subtype_ok") == False) | (pl.col("B_x_rating_subtype_ok") == False)  # noqa: E712
            | (pl.col("B_lab_suggested_veracity") != 3) | (pl.col("B_x_suggested_veracity") != 3))
         ).alias("unprovable_suspect"))
    # the human queue
    j = j.with_columns(
        (pl.col("veracity_disagree").fill_null(False) | pl.col("axis_needs_human").fill_null(False)
         | pl.col("unprovable_suspect").fill_null(False)).alias("needs_review"))

    def reason(r):
        rs = []
        if r["veracity_disagree"]: rs.append("veracity")
        if r["axis_needs_human"]: rs.append("axis")
        if r["unprovable_suspect"]: rs.append("unprovable")
        return ",".join(rs)
    j = j.with_columns(pl.struct(["veracity_disagree", "axis_needs_human", "unprovable_suspect"])
                       .map_elements(reason, return_dtype=pl.Utf8).alias("review_reason"))

    j.write_parquet(OUT)

    # ---- report ----
    a = j.filter(pl.col("audited"))
    print(f"in_gold {j.height} | audited {a.height} ({100*a.height/j.height:.0f}%)")
    print(f"  audited by split: ", dict(a.group_by('split').len().sort('len', descending=True).iter_rows()))
    print(f"\nderived flags (audited rows):")
    print(f"  veracity_disagree (either)   {a['veracity_disagree'].sum()}")
    print(f"  veracity_both_disagree       {a['veracity_both_disagree'].sum()}")
    print(f"  subtype_disagree             {a['subtype_disagree'].sum()}")
    print(f"  axis_reclassify (both agree) {a['axis_reclassify_to'].is_not_null().sum()}")
    rec = a.filter(pl.col('axis_reclassify_to').is_not_null())
    for r0, r1, n in rec.group_by(['axis_rule', 'axis_reclassify_to']).len().sort('len', descending=True).iter_rows():
        print(f"       {r0} -> {r1}: {n}")
    print(f"  axis_needs_human (disagree)  {a['axis_needs_human'].sum()}")
    print(f"  satire_flag (kept, not drop) {a['satire_flag'].sum()}")
    print(f"  unprovable_suspect           {a['unprovable_suspect'].sum()}")
    print(f"  ---> needs_review (queue)    {a['needs_review'].sum()}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
