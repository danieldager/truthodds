"""Join all extraction-grading parquets into two analysis-ready tables — v4.

Inputs (under ``data/``):
    posts_n{N}.parquet                  one row per post_id
    extractions_n{N}.parquet            one row per post_id
    judgments_per_claim_n{N}.parquet    one row per (post_id, claim_index)
    features_n{N}.parquet               one row per post_id            (optional)
    classical_metrics_n{N}.parquet      one row per (post_id, claim_index)  (optional)

The v3 detection parquet has been removed; detection correctness is now
inferred from the per-claim `misinfo_candidate` boolean (the judge applies
the same misinformation criterion as the extractor).

Outputs:
    graded_posts_n{N}.parquet
        one row per post_id: post + features + extractor decision
        + per-post aggregates of Likert dims (mean, min across claims)
        + per-post `any_misinfo_candidate` (true if ANY claim is one).
    graded_claims_n{N}.parquet
        one row per (post_id, claim_index): claim Likerts + misinfo_candidate
        bool + classical metrics (if available) + post features broadcast
        onto each claim.

Join key: ``post_id`` everywhere; per-claim tables additionally on
``claim_index``. The mechanical join is a left-join on the posts parquet so
posts with `has_claim=false` (no per-claim row) are preserved.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).parent / "data"
LIKERT_DIMS = ["fidelity", "decontextualized", "verifiability"]


def _find_n(input_path: Path) -> str:
    m = re.search(r"_n(\d+)", input_path.stem)
    return m.group(1) if m else "all"


def _read_optional(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


def build_graded_claims(
    per_claim: pl.DataFrame,
    features: pl.DataFrame | None,
    classical: pl.DataFrame | None,
) -> pl.DataFrame:
    df = per_claim
    if classical is not None:
        df = df.join(classical, on=["post_id", "claim_index"], how="left")
    if features is not None:
        # Broadcast post features onto each claim. Avoid duplicating post_id.
        df = df.join(features, on="post_id", how="left")
    return df


def build_graded_posts(
    posts: pl.DataFrame,
    extractions: pl.DataFrame,
    per_claim: pl.DataFrame,
    features: pl.DataFrame | None,
) -> pl.DataFrame:
    # Per-post Likert aggregates + misinfo_candidate aggregate.
    agg_exprs = []
    for dim in LIKERT_DIMS:
        agg_exprs.append(pl.col(dim).mean().alias(f"{dim}_mean"))
        agg_exprs.append(pl.col(dim).min().alias(f"{dim}_min"))
    # Per-post: did the judge call ANY claim a misinformation candidate?
    agg_exprs.append(pl.col("misinfo_candidate").any().alias("any_misinfo_candidate"))
    agg_exprs.append(pl.col("misinfo_candidate").sum().alias("n_misinfo_candidate"))
    per_post_agg = per_claim.group_by("post_id").agg(agg_exprs)

    df = posts
    df = df.join(
        extractions.select(
            ["post_id", "has_claim", "n_claims", "claims", "latency_s", "error"]
        ).rename(
            {
                "has_claim": "extractor_has_claim",
                "latency_s": "extract_latency_s",
                "error": "extract_error",
            }
        ),
        on="post_id",
        how="left",
    )
    df = df.join(per_post_agg, on="post_id", how="left")
    if features is not None:
        df = df.join(features, on="post_id", how="left")
    return df


def _sanity_check(graded_posts: pl.DataFrame, graded_claims: pl.DataFrame) -> None:
    n_posts = graded_posts.height
    n_unique = graded_posts["post_id"].n_unique()
    if n_posts != n_unique:
        raise ValueError(
            f"graded_posts has duplicate post_ids: {n_posts} rows, "
            f"{n_unique} unique"
        )
    # Per-claim uniqueness on (post_id, claim_index)
    if graded_claims.height:
        dup = graded_claims.group_by(["post_id", "claim_index"]).len().filter(
            pl.col("len") > 1
        )
        if dup.height:
            raise ValueError(
                f"graded_claims has duplicate (post_id, claim_index) rows: "
                f"{dup.height} duplicates"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DATA_DIR / "posts_n2000.parquet",
        help="anchor input parquet; sibling files inferred from name suffix",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="override directory containing the input parquets",
    )
    args = ap.parse_args()

    data_dir = args.data_dir or args.input.parent
    n = _find_n(args.input)

    posts_path = data_dir / f"posts_n{n}.parquet"
    extractions_path = data_dir / f"extractions_n{n}.parquet"
    per_claim_path = data_dir / f"judgments_per_claim_n{n}.parquet"
    features_path = data_dir / f"features_n{n}.parquet"
    classical_path = data_dir / f"classical_metrics_n{n}.parquet"

    for required in (posts_path, extractions_path, per_claim_path):
        if not required.exists():
            raise SystemExit(f"required parquet missing: {required}")

    posts = pl.read_parquet(posts_path)
    extractions = pl.read_parquet(extractions_path)
    per_claim = pl.read_parquet(per_claim_path)
    features = _read_optional(features_path)
    classical = _read_optional(classical_path)

    print(
        f"posts={posts.height}  extractions={extractions.height}  "
        f"per_claim={per_claim.height}  "
        f"features={'-' if features is None else features.height}  "
        f"classical={'-' if classical is None else classical.height}"
    )

    graded_claims = build_graded_claims(per_claim, features, classical)
    graded_posts = build_graded_posts(posts, extractions, per_claim, features)
    _sanity_check(graded_posts, graded_claims)

    out_posts = data_dir / f"graded_posts_n{n}.parquet"
    out_claims = data_dir / f"graded_claims_n{n}.parquet"
    graded_posts.write_parquet(out_posts)
    graded_claims.write_parquet(out_claims)

    print(
        f"wrote {graded_posts.height} rows -> {out_posts}\n"
        f"wrote {graded_claims.height} rows -> {out_claims}"
    )


if __name__ == "__main__":
    main()
