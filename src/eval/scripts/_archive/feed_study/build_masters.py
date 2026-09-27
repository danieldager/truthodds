"""Join the sweep's per-run parquets into a post-level and claim-level master.

Read-only. Reads from --dir (default data/sample1000) using the sweep's naming:
  claimify_<model>.parquet                              (extraction: verifiable, claims)
  relevance_raw_<model>.parquet / fable_post_<model>.parquet   (RAW filters)
  {relevance,fable}_{claim,both}_score-<scorer>_ext-<extractor>.parquet
  judge_compare.parquet / judge_quality_<extractor>.parquet

Outputs (in --dir):
  master_post.parquet   — one row per post_id
  master_claim.parquet  — one row per (extractor_model, post_id, claim_index)

Models are the two extractors/scorers: gpt-oss-120b (g) and qwen3-32b (q).

  uv run python -m eval.scripts.feed_study.build_masters --dir eval/scripts/feed_study/data/sample1000 \
      --posts eval/scripts/feed_study/data/posts_sample1000.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

GPT, QWEN = "gpt-oss-120b", "qwen3-32b"
TAG = {GPT: "g", QWEN: "q"}


def _opt(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


def _explode_claims(claimify_path: Path, extractor: str) -> pl.DataFrame:
    """claimify parquet -> (extractor_model, post_id, claim_index, claim_text)."""
    cl = pl.read_parquet(claimify_path).select(["post_id", "claims"])
    ex = (cl.with_columns(pl.int_ranges(pl.col("claims").list.len()).alias("claim_index"))
            .explode(["claims", "claim_index"])
            .rename({"claims": "claim_text"})
            .filter(pl.col("claim_text").is_not_null() & (pl.col("claim_text").str.strip_chars() != "")))
    return ex.with_columns(pl.lit(extractor).alias("extractor_model"))


def build_post_master(d: Path, posts: pl.DataFrame) -> pl.DataFrame:
    m = posts.select(["post_id", "text", "lang", "source_corpus", "has_quoted",
                      "has_card", "operation"])
    for model in (GPT, QWEN):
        t = TAG[model]
        cl = _opt(d / f"claimify_{model}.parquet")
        if cl is not None:
            m = m.join(
                cl.select(
                    "post_id",
                    pl.col("verifiable").alias(f"verif_raw_{t}"),
                    pl.col("claims").list.len().alias(f"n_claims_{t}"),
                    pl.col("outcome").alias(f"outcome_{t}"),
                ), on="post_id", how="left")
        rv = _opt(d / f"relevance_raw_{model}.parquet")
        if rv is not None:
            m = m.join(rv.select("post_id", pl.col("in_scope").alias(f"relev_raw_{t}")),
                       on="post_id", how="left")
        fb = _opt(d / f"fable_post_{model}.parquet")
        if fb is not None:
            m = m.join(fb.select(
                "post_id",
                pl.col("fable_checkworthy").alias(f"harm_raw_{t}"),
                pl.col("fable_total").alias(f"harm_total_{t}"),
            ), on="post_id", how="left")
    cmp = _opt(d / "judge_compare.parquet")
    if cmp is not None:
        m = m.join(cmp.select("post_id",
                              pl.col("agreement").alias("cmp_agreement"),
                              pl.col("more_complete").alias("cmp_more_complete")),
                   on="post_id", how="left")
    return m


def _quality_long(d: Path, extractor: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (per-claim quality df, per-post coverage/flag df) for one extractor."""
    q = _opt(d / f"judge_quality_{extractor}.parquet")
    if q is None:
        return pl.DataFrame(), pl.DataFrame()
    rows, post_rows = [], []
    for r in q.iter_rows(named=True):
        post_rows.append({"post_id": r["post_id"], "coverage": r["coverage"], "flag": r["flag"]})
        try:
            scores = json.loads(r["claim_scores_json"] or "[]")
        except json.JSONDecodeError:
            scores = []
        for s in scores:
            if isinstance(s, dict) and s.get("index") is not None:
                rows.append({"post_id": r["post_id"], "claim_index": int(s["index"]),
                             "faithful": s.get("faithful"),
                             "decontextualized": s.get("decontextualized"),
                             "atomicity": s.get("atomicity")})
    per_claim = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"post_id": pl.String, "claim_index": pl.Int64, "faithful": pl.Int64,
                "decontextualized": pl.Int64, "atomicity": pl.Int64})
    per_post = pl.DataFrame(post_rows) if post_rows else pl.DataFrame(
        schema={"post_id": pl.String, "coverage": pl.Int64, "flag": pl.String})
    return per_claim, per_post


def build_claim_master(d: Path) -> pl.DataFrame:
    posts_text = None
    frames = []
    for ext in (GPT, QWEN):
        cf = d / f"claimify_{ext}.parquet"
        if not cf.exists():
            continue
        base = _explode_claims(cf, ext)  # extractor_model, post_id, claim_index, claim_text
        # scorer filter verdicts for this extractor
        for sc in (GPT, QWEN):
            for filt, vcol, alias in (("relevance", "in_scope", "relev"),
                                      ("fable", "fable_checkworthy", "harm")):
                for mode in ("claim", "both"):
                    f = _opt(d / f"{filt}_{mode}_score-{sc}_ext-{ext}.parquet")
                    if f is None:
                        continue
                    col = f"{alias}_{mode}_{TAG[sc]}"
                    base = base.join(
                        f.select("post_id", "claim_index", pl.col(vcol).alias(col)),
                        on=["post_id", "claim_index"], how="left")
        # llama quality for this extractor
        per_claim, per_post = _quality_long(d, ext)
        if per_claim.height:
            base = base.join(per_claim, on=["post_id", "claim_index"], how="left")
        if per_post.height:
            base = base.join(per_post, on="post_id", how="left")
        frames.append(base)
    if not frames:
        return pl.DataFrame()
    out = pl.concat(frames, how="diagonal_relaxed")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True, help="sweep output dir (e.g. data/sample1000)")
    ap.add_argument("--posts", type=Path, required=True, help="the posts parquet used in the sweep")
    args = ap.parse_args()

    posts = pl.read_parquet(args.posts)
    mpost = build_post_master(args.dir, posts)
    mclaim = build_claim_master(args.dir)
    mpost.write_parquet(args.dir / "master_post.parquet")
    mclaim.write_parquet(args.dir / "master_claim.parquet")
    print(f"master_post:  {mpost.height} rows, {len(mpost.columns)} cols -> {args.dir/'master_post.parquet'}")
    print(f"  cols: {mpost.columns}")
    print(f"master_claim: {mclaim.height} rows, {len(mclaim.columns)} cols -> {args.dir/'master_claim.parquet'}")
    print(f"  cols: {mclaim.columns}")
    if mclaim.height:
        print("  by extractor:", dict(mclaim.group_by("extractor_model").len().iter_rows()))


if __name__ == "__main__":
    main()
