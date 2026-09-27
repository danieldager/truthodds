"""Stage 6 — join every cascade stage into two analysis-ready tables.

Reuses ``extraction_grading/grade.py``'s builders and sanity check, extended
with the cascade-specific stages (topic embedding, LLM scope, FABLE, FCT).

Inputs (under ``--data-dir``):
    posts_x862.parquet        Stage 0  one row / post_id
    topics_x862.parquet       Stage 1  one row / post_id        (optional)
    scope_x862.parquet        Stage 2  one row / post_id        (optional)
    stage3_extractions.parquet Stage 3 one row / in-scope post_id
    stage4_judgments.parquet  Stage 4  one row / (post_id, claim_index)
    stage5_fable.parquet      Stage 5  one row / (post_id, claim_index) (optional)
    fct_x862.parquet          Stage 7  one row / check-worthy claim   (optional)

Outputs:
    graded_posts_x862.parquet   one row / post_id (all 862 preserved):
        post fields + topic/scope + extractor decision
        + per-post judge aggregates (mean/min Likert, any/n misinfo_candidate)
        + per-post FABLE aggregates (any/n checkworthy, mean total)
        + per-post FCT aggregates (any/n match).
    graded_claims_x862.parquet  one row / (post_id, claim_index):
        judge Likerts + misinfo_candidate, FABLE dims + fable_checkworthy,
        FCT match fields, with post context broadcast onto each claim.

Join key: ``post_id`` everywhere; per-claim tables also on ``claim_index``.
All joins are LEFT joins anchored on the posts frame, so out-of-scope posts
(no extraction / claim rows) are preserved with nulls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from eval.scripts.extraction_grading.grade import (  # noqa: E402
    _sanity_check,
    build_graded_claims,
    build_graded_posts,
)

DEFAULT_DATA_DIR = Path(__file__).parent / "data"

# Post-level columns broadcast onto each claim for inspection (scalars only —
# never the 384-dim post_embedding).
_POST_CONTEXT = ["post_id", "lang", "text", "own_text", "quoted_text",
                 "author_handle", "favorite_count", "retweet_count"]
_TOPIC_CONTEXT = ["post_id", "top_topic", "top_cosine", "in_scope_embed", "margin"]
_SCOPE_CONTEXT = ["post_id", "in_scope_llm", "topic_llm"]


def _read(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


def _select_present(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Select the requested columns that actually exist (schema-tolerant)."""
    return df.select([c for c in cols if c in df.columns])


def _post_features(posts: pl.DataFrame, topics: pl.DataFrame | None,
                   scope: pl.DataFrame | None) -> pl.DataFrame:
    """post_id + topic + scope columns — the per-post extras to join on.

    Excludes columns already in `posts` so build_graded_posts does not collide.
    """
    feats = posts.select("post_id")
    if topics is not None:
        feats = feats.join(_select_present(topics, _TOPIC_CONTEXT), on="post_id", how="left")
    if scope is not None:
        feats = feats.join(_select_present(scope, _SCOPE_CONTEXT), on="post_id", how="left")
    return feats


def _claim_context(posts: pl.DataFrame, topics: pl.DataFrame | None,
                   scope: pl.DataFrame | None) -> pl.DataFrame:
    """Curated post context broadcast onto each claim (no embedding column)."""
    ctx = _select_present(posts, _POST_CONTEXT)
    if topics is not None:
        ctx = ctx.join(_select_present(topics, _TOPIC_CONTEXT), on="post_id", how="left")
    if scope is not None:
        ctx = ctx.join(_select_present(scope, _SCOPE_CONTEXT), on="post_id", how="left")
    return ctx


def _classical(fable: pl.DataFrame | None, fct: pl.DataFrame | None) -> pl.DataFrame | None:
    """Per-claim FABLE + FCT fields merged on (post_id, claim_index).

    Drops each stage's claim_text/raw_response/latency_s/error so they do not
    collide with the Stage-4 judge columns in build_graded_claims. In
    particular fct's `lang` is intentionally excluded — the post-context
    `lang` is already broadcast onto each claim via _claim_context, and keeping
    fct.lang would collide (polars would suffix it `lang_right`).
    """
    frames = []
    if fable is not None:
        frames.append(_select_present(fable, [
            "post_id", "claim_index", "fragmentation", "actionability",
            "believability", "spread_likelihood", "exploitativeness",
            "fable_total", "fable_checkworthy",
        ]))
    if fct is not None:
        frames.append(_select_present(fct, [
            "post_id", "claim_index", "fct_match", "fct_publisher",
            "fct_rating", "fct_url",
        ]))
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on=["post_id", "claim_index"], how="full", coalesce=True)
    return out


def _post_aggregates(graded_posts: pl.DataFrame, fable: pl.DataFrame | None,
                     fct: pl.DataFrame | None,
                     claim_keys: pl.DataFrame) -> pl.DataFrame:
    """Graft per-post FABLE / FCT aggregates onto the graded-posts frame.

    FABLE/FCT are first restricted to the Stage-4 claim key set (`claim_keys`)
    so the per-post aggregates count exactly the claims that also survive into
    graded_claims — an orphan FCT/FABLE row (a key absent from Stage 4) can
    never inflate any_fct_match / any_fable_checkworthy.
    """
    keys = ["post_id", "claim_index"]
    if fable is not None and {"fable_checkworthy", "fable_total"}.issubset(fable.columns):
        f = fable.join(claim_keys, on=keys, how="semi")
        fable_agg = f.group_by("post_id").agg(
            pl.col("fable_checkworthy").any().alias("any_fable_checkworthy"),
            pl.col("fable_checkworthy").sum().alias("n_fable_checkworthy"),
            pl.col("fable_total").mean().alias("fable_total_mean"),
        )
        graded_posts = graded_posts.join(fable_agg, on="post_id", how="left")
    if fct is not None and "fct_match" in fct.columns:
        fc = fct.join(claim_keys, on=keys, how="semi")
        fct_agg = fc.group_by("post_id").agg(
            pl.col("fct_match").any().alias("any_fct_match"),
            pl.col("fct_match").sum().alias("n_fct_match"),
        )
        graded_posts = graded_posts.join(fct_agg, on="post_id", how="left")
    return graded_posts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()
    d = args.data_dir

    posts = _read(d / "posts_x862.parquet")
    extractions = _read(d / "stage3_extractions.parquet")
    per_claim = _read(d / "stage4_judgments.parquet")
    topics = _read(d / "topics_x862.parquet")
    scope = _read(d / "scope_x862.parquet")
    fable = _read(d / "fable_x862.parquet")
    fct = _read(d / "fct_x862.parquet")

    for name, df in (("posts_x862", posts), ("stage3_extractions", extractions),
                     ("stage4_judgments", per_claim)):
        if df is None:
            raise SystemExit(f"required parquet missing: {d / (name + '.parquet')}")

    # build_graded_posts aggregates these per-claim columns; fail with a clear
    # message (not a cryptic ColumnNotFoundError) if Stage 4 didn't write them.
    needed = {"fidelity", "decontextualized", "verifiability", "misinfo_candidate"}
    missing = needed - set(per_claim.columns)
    if missing:
        raise SystemExit(f"stage4_judgments missing required columns: {sorted(missing)}")

    print(
        "  ".join(
            f"{n}={'-' if x is None else x.height}"
            for n, x in (
                ("posts", posts), ("topics", topics), ("scope", scope),
                ("extract", extractions), ("judge", per_claim),
                ("fable", fable), ("fct", fct),
            )
        )
    )

    post_features = _post_features(posts, topics, scope)
    claim_context = _claim_context(posts, topics, scope)
    classical = _classical(fable, fct)

    claim_keys = per_claim.select(["post_id", "claim_index"])
    graded_posts = build_graded_posts(posts, extractions, per_claim, post_features)
    graded_posts = _post_aggregates(graded_posts, fable, fct, claim_keys)
    graded_claims = build_graded_claims(per_claim, claim_context, classical)

    _sanity_check(graded_posts, graded_claims)
    if graded_posts.height != posts.height:
        raise ValueError(
            f"graded_posts lost rows: {graded_posts.height} != {posts.height}"
        )

    out_posts = d / "graded_posts_x862.parquet"
    out_claims = d / "graded_claims_x862.parquet"
    graded_posts.write_parquet(out_posts)
    graded_claims.write_parquet(out_claims)
    print(
        f"wrote {graded_posts.height} rows -> {out_posts}\n"
        f"wrote {graded_claims.height} rows -> {out_claims}"
    )


if __name__ == "__main__":
    main()
