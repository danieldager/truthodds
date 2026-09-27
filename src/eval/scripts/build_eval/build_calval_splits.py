"""Cut the Truth Odds CAL/VAL splits (program §1; sizes + rules Daniel 2026-07-24).

Inputs: fc_gold_v2_clustered.parquet (one row per claim cluster after the code
exclusion stack) joined to the Qwen3-235B screen output {claim_type, topic, mundane}.

Applied here, in order:
  REMAPS (harmonization audit): R1 Outdated->contested; R2 tightened
  definitive-negative reroute out of NEE (rows printed for the record); R3 AFP/BOOM
  'Misleading' title-keyword demotion -> mostly-false.
  EXCLUSIONS (flags, rows retained in the base parquet): mundane==true;
  claim_type==media_authenticity (claim-level media boundary — supersedes the
  rating-string one for retrieval experiments); judged_axis==media_authenticity
  (Daniel 2026-08-20 — the claim_type test alone is not enough: it keys on the
  claim TEXT, and extraction strips the media framing, so "this video shows a 7.1
  quake in Miyazaki" arrives as "7.1-Magnitude earthquake hits Miyazaki Prefecture"
  and screens as event_occurrence. 336 of 4,035 E1 claims were media-provenance by
  the fact-checker's own axis and every one carried a non-media claim_type).
  SPLITS: VAL = 1,000 true-side + 1,500 false-side + 400 contested/NEE, year-
  stratified with recent oversample (2024+ weighted 2x), locked untouched;
  CAL = all remaining screened trues + 5,000 year-stratified falses + remaining
  screened contested/NEE. Cluster-disjoint by construction (pool is 1 row/cluster).

  uv run python -m eval.scripts.build_eval.build_calval_splits
Outputs: eval/data/truthodds_cal.parquet, eval/data/truthodds_val.parquet,
         eval/data/truthodds_calval_provenance.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

DATA = Path("eval/data")
SEED = 404

R2_NEG = re.compile(r"(?i)^(this is (a )?fake\b|no,\s)|\bdid not say\b|\bnever said\b")
R3_TITLE = re.compile(r"(?i)doctored|altered|old (photo|video|image|footage)|"
                      r"unrelated (photo|video|image)|falsely|fabricated|^non,|photo d.archive")


def main():
    # v3 = the same 47,111 rows as v2_clustered with the FCI enrichment columns
    # (claim_date +1,143, claimant, fc_rationale, claim_date_source…). Identical
    # keys, superset schema — verified 2026-08-03 before the swap.
    base = pl.read_parquet(DATA / "fc_gold_v3.parquet")
    scr = pl.read_ndjson(DATA / "claim_screen" / "Qwen3-235B-A22B-Instruct-2507.jsonl") \
            .unique(subset=["review_url"], keep="first") \
            .select("review_url", "claim_type", "topic", "mundane")
    df = base.join(scr, on="review_url", how="inner")
    df = df.filter(pl.col("claim_date").is_null() | (pl.col("claim_date") < "2027"))  # corrupt future dates
    # Defense in depth: enforce ONE row per cluster HERE (earliest fact-check =
    # representative, Daniel 2026-07-25) — never rely on the screen file's row set
    # (it accumulates rows across rep-policy changes; the 07-25 leak assert caught
    # exactly this).
    df = (df.with_columns(pl.coalesce(pl.col("review_date"), pl.col("claim_date"))
                          .fill_null("9999").alias("_rep_date"))
            .sort(["_rep_date", "review_url"])
            .unique(subset=["cluster_id"], keep="first", maintain_order=True)
            .drop("_rep_date"))
    print(f"screened pool: {len(df)}")

    # ---- remaps (before class assignment) ----
    out_dated = pl.col("original_rating").str.contains(r"(?i)^outdated")
    r2_mask = ((pl.col("rating_subtype") == "unprovable")
               & pl.col("original_rating").str.contains(R2_NEG.pattern))
    r2_rows = df.filter(r2_mask)
    print(f"\nR2 reroute NEE->false-tier ({len(r2_rows)} rows, for the record):")
    for r in r2_rows.iter_rows(named=True):
        print(f"  [{r['publisher_site']}] {repr(r['original_rating'])[:70]}")
    r3_mask = ((pl.col("rating_subtype") == "mixed")
               & pl.col("publisher_site").is_in(["factcheck.afp.com", "factuel.afp.com", "boomlive.in"])
               & pl.col("original_rating").str.contains(r"(?i)^(misleading|trompeur)$")
               & pl.col("review_title").str.contains(R3_TITLE.pattern))
    n_r1 = int(df.select(((pl.col("rating_subtype") == "unprovable") & out_dated).sum()).item())
    df = df.with_columns(
        pl.when((pl.col("rating_subtype") == "unprovable") & out_dated)
          .then(pl.lit("mixed"))
          .when(r2_mask).then(pl.lit("clear_false"))
          .when(r3_mask).then(pl.lit("mostly_false"))
          .otherwise(pl.col("rating_subtype")).alias("rating_subtype"),
        pl.when(r2_mask).then(pl.lit(1))
          .when(r3_mask).then(pl.lit(2))
          .otherwise(pl.col("veracity")).alias("veracity"),
    ).with_columns(
        (pl.col("rating_subtype") == "unprovable").alias("nee"),
        ((pl.col("veracity") == 3) & (pl.col("rating_subtype") == "mixed")).alias("contested"),
    )
    print(f"remaps: R1 outdated->mixed {n_r1} | R2 {len(r2_rows)} | R3 {int(df.select(r3_mask.sum()).item())}")

    # ---- exclusions ----
    pool = df.filter(~pl.col("mundane") & (pl.col("claim_type") != "media_authenticity"))
    print(f"after mundane + claim_type media exclusion: {len(pool)}")

    # Second media boundary, on the axis the FACT-CHECKER graded (derive_judged_axis.py).
    axis_path = Path("eval/data/judged_axis_llm.parquet")
    if axis_path.exists():
        axis = pl.read_parquet(axis_path).select(
            pl.col("review_url"), pl.col("judged_axis_llm").alias("judged_axis"))
        pool = pool.join(axis, on="review_url", how="left")
        before = len(pool)
        pool = pool.filter(pl.col("judged_axis") != "media_authenticity")
        print(f"after judged_axis media exclusion: {len(pool)}  (-{before - len(pool)})")
    else:
        print(f"WARNING: {axis_path} missing — judged_axis media claims NOT excluded")

    pool = pool.with_columns(
        pl.coalesce(pl.col("claim_date"), pl.col("review_date")).str.slice(0, 4).alias("yr"))

    def strat_sample(g: pl.DataFrame, n: int, recent_boost: bool) -> pl.DataFrame:
        """Year-stratified draw of EXACTLY n rows (largest-remainder allocation —
        the old round()+head() trimmed the latest year, against the recency intent);
        2024+ years weighted 2x when recent_boost."""
        years = {y: c for y, c in g.group_by("yr").len().iter_rows()}
        w = {y: c * (2.0 if recent_boost and (y or "") >= "2024" else 1.0)
             for y, c in years.items()}
        tot_w = sum(w.values()) or 1.0
        exact = {y: n * wv / tot_w for y, wv in w.items()}
        alloc = {y: min(years[y], int(exact[y])) for y in w}
        # distribute the remainder by largest fractional part (capacity-aware)
        rem = n - sum(alloc.values())
        for y in sorted(w, key=lambda y: -(exact[y] - int(exact[y]))):
            if rem <= 0:
                break
            if alloc[y] < years[y]:
                alloc[y] += 1
                rem -= 1
        # ORDER-INVARIANT draw: rank within each year by blake2 hash of
        # (review_url, seed) — reproducible from row identity alone, immune to
        # upstream sort changes (the 07-25 rep-switch walked the old seeded draws).
        import hashlib
        def hkey(u):
            return hashlib.blake2b(f"{u}|{SEED}".encode(), digest_size=8).hexdigest()
        picks = []
        for (y,), grp in g.group_by("yr"):
            k = alloc.get(y, 0)
            if k > 0:
                grp = grp.with_columns(pl.col("review_url")
                                       .map_elements(hkey, return_dtype=pl.Utf8).alias("_h"))
                picks.append(grp.sort("_h").head(k).drop("_h"))
        return pl.concat(picks)

    T = pool.filter(pl.col("veracity") >= 4)
    F = pool.filter(pl.col("veracity") <= 2)
    M = pool.filter(pl.col("veracity") == 3)

    val_t = strat_sample(T, 1000, recent_boost=True)
    val_f = strat_sample(F, 1500, recent_boost=True)
    val_m = strat_sample(M, 400, recent_boost=True)
    val_urls = set(val_t["review_url"]) | set(val_f["review_url"]) | set(val_m["review_url"])

    rem = pool.filter(~pl.col("review_url").is_in(list(val_urls)))
    cal_t = rem.filter(pl.col("veracity") >= 4)
    cal_f = strat_sample(rem.filter(pl.col("veracity") <= 2), 5000, recent_boost=False)
    cal_m = rem.filter(pl.col("veracity") == 3)
    cal = pl.concat([cal_t, cal_f, cal_m])
    val = pl.concat([val_t, val_f, val_m])

    assert not (set(cal["review_url"]) & set(val["review_url"])), "split leak!"
    assert not (set(cal["cluster_id"]) & set(val["cluster_id"])), "cluster leak!"

    # Sort before writing: downstream seeded sampling (evidence_urn_run) draws by row
    # POSITION, so a nondeterministic row order here silently changes which claims any
    # seeded run picks. Caught 2026-07-27 — "seed 707, first 420" returned only 21 of
    # the original 420 pilot claims after a rebuild. Keep the sort.
    cal.sort("review_url").write_parquet(DATA / "truthodds_cal.parquet")
    val.sort("review_url").write_parquet(DATA / "truthodds_val.parquet")

    def comp(d, name):
        print(f"\n{name}: {len(d)} rows")
        print("  classes: T", len(d.filter(pl.col('veracity') >= 4)),
              "| F", len(d.filter(pl.col('veracity') <= 2)),
              "| contested", int(d['contested'].sum()), "| NEE", int(d['nee'].sum()))
        print("  years:", dict(d.group_by('yr').len().sort('yr').iter_rows()))
    comp(cal, "CAL")
    comp(val, "VAL")

    prov = {"seed": SEED, "built": "2026-07-24", "screen_prompt": "screen-v1",
            "screen_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "remaps": {"r1_outdated": n_r1, "r2_negatives": len(r2_rows),
                       "r3_title_demotion": int(df.select(r3_mask.sum()).item())},
            "exclusions": ["mundane", "claim_type==media_authenticity"],
            "cal": len(cal), "val": len(val)}
    (DATA / "truthodds_calval_provenance.json").write_text(json.dumps(prov, indent=1))
    print("\nprovenance written")


if __name__ == "__main__":
    main()
