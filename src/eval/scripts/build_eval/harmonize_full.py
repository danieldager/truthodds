"""Harmonize the exhaustive GFC pull to the 1–5 veracity scale + NEE tag (fc-gold v2).

Daniel's spec (2026-07-23): fit everything to the 1–5 scale with the partiallys at
3 = contested, and TAG unsupported (not-enough-evidence) separately — NEE is an
evidence-state, not a veracity point. Reuses harmonize.harmonise_veracity (WS0 rules:
subtype `mixed` = contested vs `unprovable` = NEE is exactly this split), with the
claim-aware LLM fallback for the free-text tail — LLM PASS IS GATED (--llm) so the
tail size + cost are visible before any spend.

  uv run python -m eval.scripts.build_eval.harmonize_full            # rules only + tail report
  uv run python -m eval.scripts.build_eval.harmonize_full --llm      # after cost sign-off

Output: eval/data/fc_gold_v2.parquet — pull columns + veracity (1–5, null if unrated/
unresolved), rating_subtype, nee, contested, judged_axis, harmonisation_source.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.harmonize import (VERACITY_LLM_MODEL, harmonise_veracity,
                            judged_axis_from_rating, veracity_llm)

DATA_DIR = Path("eval/data")
SRC = DATA_DIR / "fc_harvest_full.parquet"
OUT = DATA_DIR / "fc_gold_v2.parquet"

# DeepInfra Llama-4-Maverick pricing (per 1M tokens) for the tail estimate; ~350 in /
# 20 out tokens per call measured on the incremental harvests.
_IN_PRICE, _OUT_PRICE, _TOK_IN, _TOK_OUT = 0.15, 0.60, 350, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="run the gated LLM fallback on the rule-miss tail")
    ap.add_argument("--limit-llm", type=int, default=0, help="cap LLM calls (0 = no cap)")
    args = ap.parse_args()

    df = pl.read_parquet(SRC)
    print(f"{len(df)} rows in {SRC.name}")

    vers, subs, srcs = [], [], []
    for rating in df["original_rating"].to_list():
        v, sub = harmonise_veracity(rating)
        vers.append(v)
        subs.append(sub)
        srcs.append("rule" if v is not None or sub == "unrated" else "")
    df = df.with_columns(
        pl.Series("veracity", vers, dtype=pl.Int64),
        pl.Series("rating_subtype", subs),
        pl.Series("harmonisation_source", srcs),
        pl.Series("judged_axis", [judged_axis_from_rating(r) for r in df["original_rating"].to_list()]),
    )

    tail = df.filter((pl.col("veracity").is_null()) & (pl.col("rating_subtype") != "unrated"))
    est = len(tail) * (_TOK_IN * _IN_PRICE + _TOK_OUT * _OUT_PRICE) / 1e6
    print(f"rule pass: {len(df) - len(tail)} resolved | LLM tail: {len(tail)} rows "
          f"(~${est:.2f} at {VERACITY_LLM_MODEL.split('/')[-1]})")

    # ---- QC artifact 1: the COMPLETE rule-behavior table. The rule pass is a pure
    # function of the rating string, so one row per distinct (publisher, rating) with
    # its mapping is EXHAUSTIVE coverage of what the rules did — eyeball this, not
    # samples. Sorted by count so the head carries the mass.
    audit = (df.group_by("publisher_site", "original_rating", "veracity",
                         "rating_subtype", "harmonisation_source")
               .len().sort("len", descending=True))
    audit.write_csv(DATA_DIR / "fc_gold_v2_mapping_audit.csv")
    print(f"mapping audit table: {len(audit)} distinct (publisher, rating) pairs "
          f"-> eval/data/fc_gold_v2_mapping_audit.csv")

    # ---- QC artifact 2: stratified semantic spot-check — 12 random (claim, rating,
    # mapping) rows per subtype, for checking that the RATING itself is faithfully
    # summarized by the mapping IN CONTEXT of the claim (catches vocabulary drift on
    # new publishers that a string table can't).
    sample = pl.concat([g.sample(min(12, len(g)), seed=7)
                        for _, g in df.group_by("rating_subtype")])
    sample.select("publisher_site", "original_rating", "veracity", "rating_subtype",
                  "claim_text", "review_url").write_csv(DATA_DIR / "fc_gold_v2_spotcheck.csv")
    print(f"spot-check sample -> eval/data/fc_gold_v2_spotcheck.csv ({len(sample)} rows)")

    if args.llm and len(tail):
        n = min(len(tail), args.limit_llm) if args.limit_llm else len(tail)
        print(f"running LLM fallback on {n} rows...", flush=True)
        idx = {r: i for i, r in enumerate(df["review_url"].to_list())}
        t0, done = time.time(), 0
        for row in tail.head(n).iter_rows(named=True):
            try:
                v, sub = veracity_llm(row["original_rating"], claim=row["claim_text"])
            except Exception as e:
                print(f"  [fail] {row['review_url']}: {e}", flush=True)
                continue
            i = idx[row["review_url"]]
            vers[i], subs[i], srcs[i] = v, sub, "llm"
            done += 1
            if done % 200 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{n} ({rate:.1f}/s, ETA {(n-done)/rate/60:.0f}m)", flush=True)
        df = df.with_columns(
            pl.Series("veracity", vers, dtype=pl.Int64),
            pl.Series("rating_subtype", subs),
            pl.Series("harmonisation_source", srcs),
        )

    df = df.with_columns(
        (pl.col("rating_subtype") == "unprovable").alias("nee"),
        ((pl.col("veracity") == 3) & (pl.col("rating_subtype") == "mixed")).alias("contested"),
    )
    df.write_parquet(OUT)
    print(f"\nwrote {OUT}: {len(df)} rows")
    print(df.group_by("veracity", "rating_subtype").len().sort("len", descending=True).head(15))
    print("\nby veracity:", dict(df.group_by("veracity").len().sort("veracity").iter_rows()))
    print("nee:", df["nee"].sum(), "| contested:", df["contested"].sum())


if __name__ == "__main__":
    main()
