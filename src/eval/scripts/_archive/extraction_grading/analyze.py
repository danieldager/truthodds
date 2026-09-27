"""Distribution analysis on the joined extraction-grading parquets — v4.

Consumes ``graded_posts_n{N}.parquet`` and ``graded_claims_n{N}.parquet``
produced by ``grade.py``. v4 changes from v3:
- LIKERT_DIMS dropped `conciseness` (always 5.0) and replaced `check_worthiness`
  with `verifiability`.
- Detection block uses the per-claim `misinfo_candidate` (bool) aggregated
  per post as `any_misinfo_candidate`, instead of a separate detection judge.
- FN is unmeasurable (we don't judge claims when `has_claim=false`), so we
  report precision only on the extractor's positives.
- Disagreement set is FP-only: posts where extractor said yes but no extracted
  claim is a misinformation candidate.

Produces:
    A. Detection — confusion (limited), n_claims bar.
    B. Per-claim Likert distributions (stacked bars + summary table).
    C. Per-post aggregates (mean + min across claims).
    D. Spearman correlations: per-post Likert ↔ numeric post features.
    E. Failure-mode buckets: claims with any Likert ≤ 2; top features per bucket.
    F. Disagreement set: per-claim FPs sampled with transcripts.
    G. (if classical metrics present) Spearman: classical ↔ LLM-judge.

All outputs land under ``results/`` next to this file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from scipy import stats

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
LIKERT_DIMS = ["fidelity", "decontextualized", "verifiability"]
FAILURE_THRESHOLD = 2

FEATURE_DENYLIST = {
    "post_id",
    "post_embedding",
    "dominant_entity_types",
}


def _find_n(path: Path) -> str:
    m = re.search(r"_n(\d+)", path.stem)
    return m.group(1) if m else "all"


def _numeric_feature_cols(df: pl.DataFrame) -> list[str]:
    out = []
    for c, t in zip(df.columns, df.dtypes):
        if c in FEATURE_DENYLIST:
            continue
        if not t.is_numeric():
            continue
        # Skip judge-derived aggregate columns and bookkeeping; keep post-intrinsic.
        if c.endswith("_mean") or c.endswith("_min"):
            continue
        if c in {
            "n_claims",
            "n_misinfo_candidate",
            "extract_latency_s",
            "char_len",
        }:
            continue
        out.append(c)
    return out


def _ensure_dirs() -> tuple[Path, Path, Path]:
    fig_dir = RESULTS_DIR / "figures"
    tab_dir = RESULTS_DIR / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR, fig_dir, tab_dir


def _save_table(df: pl.DataFrame, path: Path) -> None:
    df.write_csv(path)


# ---------------------------------------------------------------------------
# A. Detection (precision-only, FN unmeasurable)
# ---------------------------------------------------------------------------


def detection_block(graded_posts: pl.DataFrame, fig_dir: Path, tab_dir: Path) -> dict:
    """Detection via extractor's has_claim vs per-post aggregated misinfo_candidate.

    - extractor_has_claim=true & any_misinfo_candidate=true → TP
    - extractor_has_claim=true & any_misinfo_candidate=false → FP
    - extractor_has_claim=false → TN (we have no judge signal here; assume
      extractor was right). FN is unmeasurable in this design.
    """
    if graded_posts.is_empty():
        return {"note": "no graded posts"}

    ext_pos = graded_posts.filter(pl.col("extractor_has_claim"))
    n_ext_pos = ext_pos.height
    if n_ext_pos == 0:
        return {"note": "no positive extractions"}

    # any_misinfo_candidate is null when there are no per-claim rows. For
    # extractor_has_claim=true we expect at least one per-claim row → not null.
    tp = int(ext_pos["any_misinfo_candidate"].fill_null(False).sum())
    fp = n_ext_pos - tp
    n_ext_neg = graded_posts.height - n_ext_pos  # treated as TN
    precision = tp / (tp + fp) if (tp + fp) else float("nan")

    # Confusion plot: 2x2 with FN/recall noted as unmeasurable.
    cm = np.array([[n_ext_neg, np.nan], [fp, tp]])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    annot = np.array([[str(n_ext_neg), "?"],
                      [str(fp), str(tp)]])
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["judge: not misinfo", "judge: misinfo"],
        yticklabels=["ext: no claim", "ext: has claim"],
        cbar=False,
        ax=ax,
    )
    ax.set_title(
        f"Precision = {precision:.2f}  (TP={tp}  FP={fp})\n"
        f"FN unmeasurable (extractor=no → no per-claim judging)"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "detection_confusion.png", dpi=150)
    plt.close(fig)

    summary = pl.DataFrame(
        {
            "metric": ["n_posts", "n_extracted_pos", "n_extracted_neg",
                       "tp", "fp", "precision"],
            "value": [float(graded_posts.height), float(n_ext_pos),
                      float(n_ext_neg), float(tp), float(fp), precision],
        }
    )
    _save_table(summary, tab_dir / "detection_summary.csv")

    # n_claims bar
    df = graded_posts.with_columns(
        pl.col("n_claims").fill_null(0)
    )
    nc = (
        df.with_columns(
            pl.when(pl.col("n_claims") >= 4).then(4).otherwise(pl.col("n_claims"))
            .alias("n_claims_bucket")
        )
        .group_by("n_claims_bucket")
        .len()
        .sort("n_claims_bucket")
    )
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(
        [str(int(x)) if x < 4 else "4+" for x in nc["n_claims_bucket"].to_list()],
        nc["len"].to_list(),
        color="steelblue",
    )
    ax.set_xlabel("n_claims extracted")
    ax.set_ylabel("posts")
    ax.set_title(f"n_claims distribution (n={graded_posts.height})")
    fig.tight_layout()
    fig.savefig(fig_dir / "n_claims_histogram.png", dpi=150)
    plt.close(fig)
    _save_table(nc, tab_dir / "n_claims_distribution.csv")

    return {
        "n_posts": graded_posts.height,
        "n_extracted_pos": n_ext_pos,
        "n_extracted_neg": n_ext_neg,
        "tp": tp,
        "fp": fp,
        "precision": precision,
    }


# ---------------------------------------------------------------------------
# B/C. Likert distributions
# ---------------------------------------------------------------------------


def _likert_summary(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0, "median": float("nan"), "mode": float("nan"),
                "mean": float("nan"), "pct_ge4": float("nan"),
                "pct_le2": float("nan")}
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "mode": float(stats.mode(values, keepdims=False).mode),
        "mean": float(np.mean(values)),
        "pct_ge4": float((values >= 4).mean()),
        "pct_le2": float((values <= 2).mean()),
    }


def likert_block(
    graded_claims: pl.DataFrame,
    graded_posts: pl.DataFrame,
    fig_dir: Path,
    tab_dir: Path,
) -> dict:
    out: dict[str, dict] = {}
    if graded_claims.is_empty():
        return {"note": "no per-claim rows"}

    n_dims = len(LIKERT_DIMS)

    # Per-claim distributions
    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 3.5), sharey=True)
    if n_dims == 1:
        axes = [axes]
    rows = []
    for ax, dim in zip(axes, LIKERT_DIMS):
        vals = graded_claims[dim].drop_nulls().to_numpy()
        counts = [int((vals == i).sum()) for i in range(1, 6)]
        ax.bar(range(1, 6), counts, color="steelblue")
        ax.set_title(f"{dim} (per-claim, n={len(vals)})")
        ax.set_xlabel("Likert 1-5")
        s = _likert_summary(vals)
        s["dim"] = dim
        s["view"] = "per_claim"
        rows.append(s)
        out[f"per_claim_{dim}"] = s
    axes[0].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(fig_dir / "likert_per_claim.png", dpi=150)
    plt.close(fig)

    # misinfo_candidate (bool) per-claim distribution
    mc_vals = graded_claims["misinfo_candidate"].drop_nulls().to_numpy().astype(bool)
    n_true = int(mc_vals.sum())
    n_false = int(len(mc_vals) - n_true)
    out["per_claim_misinfo_candidate"] = {
        "n": int(len(mc_vals)),
        "n_true": n_true,
        "n_false": n_false,
        "pct_true": float(n_true / len(mc_vals)) if len(mc_vals) else float("nan"),
    }
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["misinfo: yes", "misinfo: no"], [n_true, n_false], color=["coral", "steelblue"])
    ax.set_ylabel("count")
    ax.set_title(f"misinfo_candidate per-claim (n={len(mc_vals)})")
    fig.tight_layout()
    fig.savefig(fig_dir / "misinfo_candidate_per_claim.png", dpi=150)
    plt.close(fig)

    # Per-post mean + min
    for view in ("mean", "min"):
        fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 3.5), sharey=True)
        if n_dims == 1:
            axes = [axes]
        for ax, dim in zip(axes, LIKERT_DIMS):
            col = f"{dim}_{view}"
            if col not in graded_posts.columns:
                continue
            vals = graded_posts[col].drop_nulls().to_numpy()
            ax.hist(vals, bins=np.arange(0.75, 5.5, 0.5), color="steelblue",
                    edgecolor="white")
            ax.set_title(f"{dim} (per-post {view}, n={len(vals)})")
            ax.set_xlabel(f"{view} of post's claims")
            s = _likert_summary(vals)
            s["dim"] = dim
            s["view"] = f"per_post_{view}"
            rows.append(s)
            out[f"per_post_{view}_{dim}"] = s
        axes[0].set_ylabel("posts")
        fig.tight_layout()
        fig.savefig(fig_dir / f"likert_per_post_{view}.png", dpi=150)
        plt.close(fig)

    summary = pl.DataFrame(rows).select(
        ["view", "dim", "n", "median", "mode", "mean", "pct_ge4", "pct_le2"]
    )
    _save_table(summary, tab_dir / "likert_summary.csv")
    return out


# ---------------------------------------------------------------------------
# D. Feature correlations
# ---------------------------------------------------------------------------


def correlation_block(
    graded_posts: pl.DataFrame, fig_dir: Path, tab_dir: Path
) -> dict:
    feat_cols = _numeric_feature_cols(graded_posts)
    if not feat_cols:
        return {"note": "no numeric post features present"}

    out: dict[str, dict] = {}
    for view in ("mean", "min"):
        rows = []
        rho_matrix = np.full((len(LIKERT_DIMS), len(feat_cols)), np.nan)
        for i, dim in enumerate(LIKERT_DIMS):
            col_name = f"{dim}_{view}"
            if col_name not in graded_posts.columns:
                continue
            for j, feat in enumerate(feat_cols):
                pair = (
                    graded_posts.select([col_name, feat])
                    .drop_nulls()
                )
                if pair.height < 5:
                    continue
                a = pair[col_name].to_numpy()
                b = pair[feat].to_numpy()
                if np.std(a) == 0 or np.std(b) == 0:
                    continue
                rho, p = stats.spearmanr(a, b)
                rho_matrix[i, j] = rho
                rows.append(
                    {"dim": dim, "feature": feat, "rho": float(rho),
                     "p": float(p), "n": int(pair.height)}
                )

        if not rows:
            out[view] = {"note": "insufficient data for correlations"}
            continue

        corr_df = pl.DataFrame(rows).sort(["dim", "rho"], descending=[False, True])
        _save_table(corr_df, tab_dir / f"spearman_per_post_{view}.csv")

        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(feat_cols)), 3.5))
        sns.heatmap(
            rho_matrix,
            xticklabels=feat_cols,
            yticklabels=LIKERT_DIMS,
            cmap="RdBu_r",
            center=0,
            vmin=-0.5,
            vmax=0.5,
            ax=ax,
            cbar_kws={"label": "Spearman ρ"},
        )
        ax.set_title(f"Spearman ρ — post-{view} Likert vs features")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(fig_dir / f"corr_heatmap_{view}.png", dpi=150)
        plt.close(fig)

        # OLS top-5 per dim
        ols_rows = []
        for dim in LIKERT_DIMS:
            col_name = f"{dim}_{view}"
            if col_name not in graded_posts.columns:
                continue
            sub = corr_df.filter(pl.col("dim") == dim).with_columns(
                pl.col("rho").abs().alias("abs_rho")
            ).sort("abs_rho", descending=True)
            top = sub.head(5)["feature"].to_list()
            if not top:
                continue
            data = graded_posts.select([col_name] + top).drop_nulls()
            if data.height < len(top) + 3:
                continue
            y = data[col_name].to_numpy()
            X = np.column_stack(
                [np.ones(data.height)] + [data[f].to_numpy() for f in top]
            )
            try:
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            y_hat = X @ beta
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            for name, b in zip(["(intercept)"] + top, beta):
                ols_rows.append(
                    {"dim": dim, "feature": name, "coef": float(b),
                     "model_r2": r2, "n": data.height}
                )
        if ols_rows:
            _save_table(pl.DataFrame(ols_rows),
                        tab_dir / f"ols_top5_per_post_{view}.csv")
        out[view] = {"n_features": len(feat_cols)}

    return out


# ---------------------------------------------------------------------------
# E. Failure-mode buckets
# ---------------------------------------------------------------------------


def failure_block(
    graded_claims: pl.DataFrame,
    graded_posts: pl.DataFrame,
    results_dir: Path,
    tab_dir: Path,
) -> dict:
    if graded_claims.is_empty():
        return {"note": "no per-claim rows"}

    feat_cols = _numeric_feature_cols(graded_posts)
    bucket_summary: list[dict] = []
    lines = ["# Failure-mode buckets", ""]
    lines.append(
        "A claim is bucketed under a dimension when its Likert on that dimension "
        f"is ≤ {FAILURE_THRESHOLD}. A claim may appear in multiple buckets. "
        "An additional misinfo_candidate=false bucket is included.\n"
    )

    total_claims = graded_claims.height

    # Likert buckets
    for dim in LIKERT_DIMS:
        bucket = graded_claims.filter(pl.col(dim) <= FAILURE_THRESHOLD)
        n_bucket = bucket.height
        pct = n_bucket / total_claims if total_claims else 0.0
        bucket_summary.append(
            {"bucket": dim, "n_claims": n_bucket, "pct_of_claims": pct}
        )
        lines.append(f"## {dim} ≤{FAILURE_THRESHOLD} (n={n_bucket}, {pct:.1%} of claims)")
        if n_bucket == 0:
            lines.append("_no claims in this bucket_\n")
            continue
        _append_bucket_section(lines, bucket, graded_posts, feat_cols, dim)

    # misinfo_candidate=false bucket
    mc_bucket = graded_claims.filter(~pl.col("misinfo_candidate"))
    n_bucket = mc_bucket.height
    pct = n_bucket / total_claims if total_claims else 0.0
    bucket_summary.append(
        {"bucket": "misinfo_candidate_false", "n_claims": n_bucket,
         "pct_of_claims": pct}
    )
    lines.append(f"## misinfo_candidate=false (n={n_bucket}, {pct:.1%} of claims)")
    if n_bucket > 0:
        _append_bucket_section(lines, mc_bucket, graded_posts, feat_cols,
                               "verifiability")

    (results_dir / "failure_buckets.md").write_text("\n".join(lines))
    _save_table(pl.DataFrame(bucket_summary), tab_dir / "failure_buckets.csv")
    return {"buckets": bucket_summary}


def _append_bucket_section(lines, bucket, graded_posts, feat_cols, display_dim):
    if feat_cols:
        bucket_posts = graded_posts.filter(
            pl.col("post_id").is_in(bucket["post_id"].unique().to_list())
        )
        corpus_posts = graded_posts.filter(pl.col("extractor_has_claim"))

        feat_rows = []
        for feat in feat_cols:
            a = bucket_posts[feat].drop_nulls().to_numpy()
            b = corpus_posts[feat].drop_nulls().to_numpy()
            if len(a) < 3 or len(b) < 3:
                continue
            mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
            s_a, s_b = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
            pooled = np.sqrt(((len(a) - 1) * s_a**2 + (len(b) - 1) * s_b**2)
                              / (len(a) + len(b) - 2))
            d = (mean_a - mean_b) / pooled if pooled > 0 else 0.0
            feat_rows.append(
                {"feature": feat, "bucket_mean": mean_a,
                 "corpus_mean": mean_b, "cohens_d": float(d)}
            )
        if feat_rows:
            top = sorted(feat_rows, key=lambda r: abs(r["cohens_d"]),
                         reverse=True)[:3]
            lines.append("Top distinguishing post features (|Cohen's d|):")
            lines.append("")
            lines.append("| feature | bucket mean | corpus mean | d |")
            lines.append("|---|---|---|---|")
            for r in top:
                lines.append(
                    f"| {r['feature']} | {r['bucket_mean']:.3f} | "
                    f"{r['corpus_mean']:.3f} | {r['cohens_d']:+.2f} |"
                )

    sample = bucket.head(3).select(["post_id", "claim_index", "claim_text",
                                    display_dim, "misinfo_candidate"])
    lines.append("\nExample claims:")
    for row in sample.iter_rows(named=True):
        lines.append(
            f"- `{row['post_id']}` [#{row['claim_index']}] "
            f"({display_dim}={row[display_dim]}, "
            f"misinfo={row['misinfo_candidate']}): "
            f"{row['claim_text']!r}"
        )
    lines.append("")


# ---------------------------------------------------------------------------
# F. Disagreement set — FP-only (extractor=true, judge says misinfo_candidate=false)
# ---------------------------------------------------------------------------


def disagreement_block(
    graded_posts: pl.DataFrame,
    graded_claims: pl.DataFrame,
    results_dir: Path,
) -> dict:
    """Posts where extractor said yes but NO extracted claim is misinfo_candidate."""
    if graded_posts.is_empty() or graded_claims.is_empty():
        return {"note": "no data"}

    ext_pos_posts = graded_posts.filter(pl.col("extractor_has_claim"))
    fp_posts = ext_pos_posts.filter(~pl.col("any_misinfo_candidate"))
    n_fp = fp_posts.height
    # Partial-agreement: post has some misinfo_candidate=true claims and some false
    partial = ext_pos_posts.filter(
        pl.col("any_misinfo_candidate")
        & (pl.col("n_misinfo_candidate") < pl.col("n_claims"))
    )
    n_partial = partial.height

    lines = ["# Extractor ↔ judge disagreement (v4)", ""]
    lines.append(
        f"{n_fp} posts out of {ext_pos_posts.height} positive extractions have "
        f"NO claim flagged as misinfo_candidate by the judge — these are FP posts.\n"
        f"{n_partial} additional posts had mixed agreement (some claims flagged, "
        f"some not).\n"
    )

    lines.append(f"## FP — all claims judged misinfo_candidate=false (n={n_fp})")
    for row in fp_posts.head(15).iter_rows(named=True):
        claims = row.get("claims") or []
        lines.append(f"- `{row['post_id']}`")
        lines.append(f"  - text: {row['text']!r}")
        if claims:
            lines.append(f"  - claims: {list(claims)}")
        # per-claim scores from graded_claims
        per_claim = graded_claims.filter(pl.col("post_id") == row["post_id"]).select(
            ["claim_index", "verifiability", "misinfo_candidate"]
        )
        for c in per_claim.iter_rows(named=True):
            lines.append(
                f"    [c{c['claim_index']}] verifiability={c['verifiability']}, "
                f"misinfo_candidate={c['misinfo_candidate']}"
            )
        lines.append("")

    lines.append(f"\n## Partial agreement (n={n_partial})")
    for row in partial.head(10).iter_rows(named=True):
        claims = row.get("claims") or []
        lines.append(f"- `{row['post_id']}`")
        lines.append(f"  - text: {row['text']!r}")
        if claims:
            lines.append(f"  - claims: {list(claims)}")
        per_claim = graded_claims.filter(pl.col("post_id") == row["post_id"]).select(
            ["claim_index", "verifiability", "misinfo_candidate"]
        )
        for c in per_claim.iter_rows(named=True):
            lines.append(
                f"    [c{c['claim_index']}] verifiability={c['verifiability']}, "
                f"misinfo_candidate={c['misinfo_candidate']}"
            )
        lines.append("")

    (results_dir / "disagreement_examples.md").write_text("\n".join(lines))
    return {"n_fp": n_fp, "n_partial": n_partial}


# ---------------------------------------------------------------------------
# G. Classical-metric correlation (optional)
# ---------------------------------------------------------------------------


def classical_block(graded_claims: pl.DataFrame, tab_dir: Path) -> dict:
    classical_cols = [
        c for c in graded_claims.columns
        if c.startswith(("nli_", "cos_", "align", "summac", "ner_overlap"))
    ]
    if not classical_cols:
        return {"note": "no classical metrics present"}

    rows = []
    for dim in LIKERT_DIMS:
        for col in classical_cols:
            pair = graded_claims.select([dim, col]).drop_nulls()
            if pair.height < 5:
                continue
            a = pair[dim].to_numpy()
            b = pair[col].to_numpy()
            if np.std(a) == 0 or np.std(b) == 0:
                continue
            rho, p = stats.spearmanr(a, b)
            rows.append(
                {"dim": dim, "classical_metric": col, "rho": float(rho),
                 "p": float(p), "n": int(pair.height)}
            )
    if rows:
        _save_table(pl.DataFrame(rows), tab_dir / "classical_vs_judge.csv")
    return {"n_pairs": len(rows)}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def write_summary(
    results: dict, results_dir: Path, n: str, graded_posts: pl.DataFrame,
    graded_claims: pl.DataFrame,
) -> None:
    det = results.get("detection", {})
    lines = [
        f"# Extraction-grading summary v4 (n={n})",
        "",
        f"- posts analysed: {graded_posts.height}",
        f"- claims analysed: {graded_claims.height}",
        "",
        "## Detection (FP-precision; FN unmeasurable)",
    ]
    if "precision" in det:
        lines.extend([
            f"- positive extractions: {det['n_extracted_pos']}  "
            f"(of {det['n_posts']} posts)",
            f"- TP (extractor=yes & any claim is misinfo_candidate): {det['tp']}",
            f"- FP (extractor=yes & NO claim is misinfo_candidate): {det['fp']}",
            f"- precision = {det['precision']:.3f}",
            f"- TN (extractor=no): {det['n_extracted_neg']} (assumed correct, "
            "FN unmeasurable)",
        ])
    else:
        lines.append(f"- {det.get('note', 'no data')}")

    lines.extend(["", "## Quality (per-claim Likert)"])
    likert = results.get("likert", {})
    for dim in LIKERT_DIMS:
        s = likert.get(f"per_claim_{dim}")
        if s:
            lines.append(
                f"- **{dim}**: median={s['median']:.1f}, mean={s['mean']:.2f}, "
                f"%≥4={s['pct_ge4']:.0%}, %≤2={s['pct_le2']:.0%}  (n={s['n']})"
            )

    mc = likert.get("per_claim_misinfo_candidate")
    if mc:
        lines.extend([
            "",
            "## Misinformation-candidate (per-claim bool)",
            f"- true: {mc['n_true']} ({mc['pct_true']:.0%})",
            f"- false: {mc['n_false']}",
            f"- n={mc['n']}",
        ])

    fails = results.get("failures", {}).get("buckets", [])
    if fails:
        lines.extend(["", "## Failure rates (per-claim)"])
        for b in fails:
            lines.append(
                f"- {b['bucket']}: {b['n_claims']} claims "
                f"({b['pct_of_claims']:.1%})"
            )

    disagree = results.get("disagreement", {})
    if "n_fp" in disagree:
        lines.extend([
            "",
            "## Disagreement (post-level)",
            f"- {disagree['n_fp']} FP posts (extractor=yes, all claims judged "
            "misinfo_candidate=false)",
            f"- {disagree['n_partial']} posts with partial agreement (some "
            "claims flagged, some not)",
        ])

    lines.extend([
        "",
        "## Artefacts",
        "- `figures/` — confusion, n_claims histogram, Likert distributions, "
        "misinfo_candidate bar, Spearman heatmaps",
        "- `tables/` — detection summary, Likert summary, Spearman correlations, "
        "OLS, failure buckets",
        "- `failure_buckets.md`, `disagreement_examples.md` — qualitative",
    ])

    (results_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i", "--input", type=Path,
        default=DATA_DIR / "graded_posts_n2000.parquet",
        help="anchor graded_posts parquet; graded_claims inferred by suffix",
    )
    ap.add_argument(
        "--data-dir", type=Path, default=None,
        help="override directory containing graded parquets",
    )
    args = ap.parse_args()

    data_dir = args.data_dir or args.input.parent
    n = _find_n(args.input)
    posts_path = data_dir / f"graded_posts_n{n}.parquet"
    claims_path = data_dir / f"graded_claims_n{n}.parquet"
    for p in (posts_path, claims_path):
        if not p.exists():
            raise SystemExit(f"missing: {p} (run grade.py first)")

    graded_posts = pl.read_parquet(posts_path)
    graded_claims = pl.read_parquet(claims_path)
    print(f"posts={graded_posts.height} claims={graded_claims.height}")

    results_dir, fig_dir, tab_dir = _ensure_dirs()
    sns.set_theme(style="whitegrid")

    results = {}
    results["detection"] = detection_block(graded_posts, fig_dir, tab_dir)
    results["likert"] = likert_block(graded_claims, graded_posts, fig_dir, tab_dir)
    results["correlations"] = correlation_block(graded_posts, fig_dir, tab_dir)
    results["failures"] = failure_block(graded_claims, graded_posts, results_dir,
                                        tab_dir)
    results["disagreement"] = disagreement_block(graded_posts, graded_claims,
                                                  results_dir)
    results["classical"] = classical_block(graded_claims, tab_dir)

    write_summary(results, results_dir, n, graded_posts, graded_claims)
    print(f"wrote results to {results_dir}/")


if __name__ == "__main__":
    main()
