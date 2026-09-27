"""Build the STAGE-2 (claim-extraction) eval dataset from the core 12-month fact-check pool.

Architecture: `dataset_12mo.parquet` is the CORE — every fact-check we could harvest (grab as much as
possible). Each pipeline stage derives its OWN eval subset from the core, with documented inclusion
rules. This builds the Stage-2 subset: **you can only evaluate claim extraction on posts that have a
post to extract from.**

Derivation (core → Stage-2 subset), each filter with its reason:
  1. has gold `claim_text`            — need a target to judge extraction against.
  2. NOT video (`has_video != True`)  — the extractor can't parse video yet (skip-video rule).
  3. HAS INPUT (post text OR image)   — the CORE is ~70% "empty-input" rows (deleted/login-walled/
     API-only-no-source posts where only the fact-checker gold survives); there is nothing to extract
     from them, so they are guaranteed misses and must NOT contaminate the Stage-2 eval (clog 270626).

Deterministic dev/test split: `md5(review_url) < DEV_FRAC` → dev (pool-independent, so a row's fold is
frozen even when the core grows, e.g. extending to 24 months). Writes `stage2_dataset.parquet` (+ a
`split` col) and prints the funnel. Run after `build_dataset_splits.py`.

  uv run python -m eval.scripts.build_stage2_dataset
"""
from __future__ import annotations

import hashlib

import polars as pl

CORE = "eval/data/dataset_12mo.parquet"
OUT = "eval/data/stage2_dataset.parquet"
DEV_FRAC = 0.4


def main() -> None:
    core = pl.read_parquet(CORE)
    n0 = core.height
    # "usable" = has the REAL post text (raw_claim) OR the gated/quote context OR an image. We feed the
    # extractor raw_claim (validated distinct from the gold), so a row with raw_claim is extractable even
    # when the claim-gate left raw_context empty. (clog 270626)
    notext = (pl.col("raw_claim").fill_null("").str.strip_chars().str.len_chars() == 0) & \
             (pl.col("raw_context").fill_null("").str.strip_chars().str.len_chars() == 0)
    no_input = notext & (pl.col("has_image").fill_null(False) == False)  # noqa: E712

    d = core.filter(pl.col("claim_text").is_not_null())
    n1 = d.height
    d = d.filter(~no_input)  # keep only rows with something to extract from
    n2 = d.height
    d = d.filter(pl.col("has_video").fill_null(False) != True)  # drop ACTUAL video (keep None/False)  # noqa: E712
    n3 = d.height

    d = d.with_columns(
        pl.col("review_url").map_elements(
            lambda u: (int(hashlib.md5(u.encode()).hexdigest(), 16) % 10_000) / 10_000, return_dtype=pl.Float64
        ).alias("_h"))
    d = d.with_columns(pl.when(pl.col("_h") < DEV_FRAC).then(pl.lit("split_dev")).otherwise(pl.lit("split_test")).alias("split")).drop("_h")
    d = d.with_columns(pl.col("split").str.replace("split_", ""))
    d.write_parquet(OUT)

    print(f"=== Stage-2 dataset derivation (core → subset) → {OUT} ===")
    print(f"  core fact-checks (dataset_12mo)        : {n0}")
    print(f"  − missing gold claim_text              : {n0-n1:>5} drop  → {n1}")
    print(f"  − EMPTY-INPUT (no post text AND no img): {n1-n2:>5} drop  → {n2}")
    print(f"  − video (extractor can't parse)        : {n2-n3:>5} drop  → {n3}  STAGE-2 SET")
    nd = d.filter(pl.col("split") == "dev").height
    print(f"  split: dev {nd} / test {d.height-nd}")
    print("  by judged_axis:", d.group_by("judged_axis").len().sort("len", descending=True).to_dicts())
    print("  with image:", d.filter(pl.col("has_image") == True).height, "| text-only:", d.filter(pl.col("has_image") != True).height)  # noqa: E712
    print("  by language:", d.group_by("language_code").len().sort("len", descending=True).to_dicts())


if __name__ == "__main__":
    main()
