"""WS4 — score predicted `veracity` (1-5) against the publisher gold (`gold_veracity`).

Joins a claims_verdict file (gold + slicing tags) with a verdicts parquet (verify_run output) on
claim_id and reports, for the ordinal veracity verdict (verdict_eval_plan.md WS4):
  - MAE on 1-5, off-by-one accuracy (|err|<=1), exact accuracy, Spearman(pred, gold).
  - NUDGE COLLAPSE: gold/pred mapped to {flag:<=2 | soft:3 | pass:>=4}; 3x3 confusion + accuracy,
    and the load-bearing **misinfo-passed** rate (gold flag-worthy <=2 but predicted pass >=4).
Headline is computed on judged_axis=='content' (factual-content claims); artifact/attribution and
the ceiling_source=='review_date' (lenient-fallback) rows are reported separately, never folded in.

  uv run python -m eval.scripts.score_veracity \
      -c eval/scripts/verification_grading/data/claims_verdict_dev_bal.parquet \
      -v eval/scripts/verification_grading/data/verdicts_verdict_dev_bal.parquet
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

SCOPE_FLOOR = "eval/data/verdict_scope_floor.parquet"


def _claim_id(url: str) -> str:  # mirror load_claims_verdict: claim_id = md5(review_url)[:16]
    return hashlib.md5((url or "").encode()).hexdigest()[:16]


def _nudge(v: int) -> str:
    return "flag" if v <= 2 else ("soft" if v == 3 else "pass")


def _metrics(gold: np.ndarray, pred: np.ndarray) -> dict:
    err = np.abs(pred - gold)
    rho = spearmanr(pred, gold).correlation if len(np.unique(pred)) > 1 and len(np.unique(gold)) > 1 else float("nan")
    return {
        "n": len(gold),
        "mae": float(err.mean()),
        "exact": float((err == 0).mean()),
        "off_by_one": float((err <= 1).mean()),
        "spearman": float(rho),
    }


def _print_metrics(title: str, gold: np.ndarray, pred: np.ndarray) -> None:
    if len(gold) == 0:
        print(f"  {title:<34} (n=0)")
        return
    m = _metrics(gold, pred)
    print(f"  {title:<34} n={m['n']:<5} MAE={m['mae']:.3f}  exact={m['exact']:.2f}  "
          f"off-by-1={m['off_by_one']:.2f}  Spearman={m['spearman']:+.3f}")


def _nudge_collapse(gold: np.ndarray, pred: np.ndarray) -> None:
    cats = ["flag", "soft", "pass"]
    gl = [_nudge(int(v)) for v in gold]
    pr = [_nudge(int(v)) for v in pred]
    acc = np.mean([g == p for g, p in zip(gl, pr)])
    print(f"\n### Nudge collapse (<=2 flag | 3 soft | >=4 pass)   accuracy={acc:.3f}  (n={len(gl)})")
    idx = {c: i for i, c in enumerate(cats)}
    cm = [[0] * 3 for _ in cats]
    for g, p in zip(gl, pr):
        cm[idx[g]][idx[p]] += 1
    print("    gold\\pred " + "".join(f"{c:>7}" for c in cats))
    for c in cats:
        print(f"    {c:<9}" + "".join(f"{cm[idx[c]][idx[d]]:>7}" for d in cats))
    # The dangerous error for a nudge tool: flag-worthy misinfo predicted as pass.
    flagworthy = np.array([g == "flag" for g in gl])
    if flagworthy.sum():
        passed = np.array([p == "pass" for p in pr])
        misinfo_passed = int((flagworthy & passed).sum())
        print(f"    MISINFO PASSED (gold<=2 -> pred>=4): {misinfo_passed}/{int(flagworthy.sum())} "
              f"= {misinfo_passed / flagworthy.sum():.3f}  (the confident-TRUE failure)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--claims", type=Path, required=True)
    ap.add_argument("-v", "--verdicts", type=Path, required=True)
    args = ap.parse_args()

    claims = pl.read_parquet(args.claims)
    vcols = pl.read_parquet(args.verdicts)
    keep = [c for c in ("claim_id", "veracity", "evidence_sufficiency", "evidence_agreement",
                        "source_reliability", "error", "n_urls_seen", "n_search_errors",
                        "rounds_used", "llm_calls", "elapsed_seconds", "stopped_reason") if c in vcols.columns]
    df = claims.join(vcols.select(keep), on="claim_id", how="inner")
    n_joined = df.height

    # Retrieval-health guard (a zero-evidence verdict scores the prior, not verdict quality).
    if "n_urls_seen" in df.columns:
        zero = df.filter(pl.col("n_urls_seen") == 0).height
        errq = df.filter(pl.col("n_search_errors") > 0).height if "n_search_errors" in df.columns else 0
        if zero or errq:
            print(f"RETRIEVAL HEALTH: {zero}/{n_joined} claims saw ZERO urls"
                  f"{f', {errq} hit search errors' if errq else ''} — high counts contaminate the score.")

    ok = df.filter(pl.col("error").is_null() if "error" in df.columns else pl.lit(True))
    ok = ok.filter(pl.col("veracity").is_not_null() & (pl.col("veracity") > 0))
    n_err = n_joined - ok.height
    if ok.height == 0:
        print("no scorable rows.")
        return

    # Stage-3 scope floor: carve the media-dependency rows WITHIN content (video / provenance_recency)
    # — they hinge on media a text-only verifier can't reach, so they belong to Phase-1b, not the
    # headline. Floor is keyed on review_url upstream; join by the derived claim_id.
    if Path(SCOPE_FLOOR).exists():
        sf = pl.read_parquet(SCOPE_FLOOR).select(
            pl.col("review_url").map_elements(_claim_id, return_dtype=pl.Utf8).alias("claim_id"),
            "stage3_in_scope", "out_of_scope_reason")
        ok = ok.join(sf, on="claim_id", how="left").with_columns(
            pl.col("stage3_in_scope").fill_null(True),
            pl.col("out_of_scope_reason").fill_null("none"))
    else:
        ok = ok.with_columns(pl.lit(True).alias("stage3_in_scope"),
                             pl.lit("none").alias("out_of_scope_reason"))

    gold = ok["gold_veracity"].to_numpy()
    pred = ok["veracity"].to_numpy()

    print(f"\nclaims: {args.claims.name}   verdicts: {args.verdicts.name}")
    print(f"joined={n_joined}  scored={ok.height}  errors/invalid={n_err}")

    # Headline population = content axis AND in stage-3 scope (text-verifiable). The media-carve
    # rows are kept and reported separately, never folded into the headline.
    content = ok.filter(pl.col("judged_axis") == "content")
    in_scope = content.filter(pl.col("stage3_in_scope"))
    carved = content.filter(~pl.col("stage3_in_scope"))
    print(f"\n{'='*70}\nVERACITY vs gold_veracity (1-5 ordinal)\n{'='*70}")
    _print_metrics("HEADLINE (content, stage3 in-scope)", in_scope["gold_veracity"].to_numpy(),
                   in_scope["veracity"].to_numpy())
    _print_metrics("content (all, incl. media-carve)", content["gold_veracity"].to_numpy(),
                   content["veracity"].to_numpy())
    _print_metrics("all axes", gold, pred)

    # Distributions (is the scale used, or collapsed?).
    def dist(a):
        return {int(k): int(v) for k, v in sorted(zip(*np.unique(a, return_counts=True)))}
    print(f"\n  gold dist: {dist(gold)}")
    print(f"  pred dist: {dist(pred)}")

    # Breakdowns (headline = content in-scope), each reported separately — never folded into the headline.
    print(f"\n### Breakdowns (content, in-scope)")
    for col, label in [("ceiling_source", "ceiling"), ("language_code", "lang"), ("source", "src")]:
        if col not in in_scope.columns:
            continue
        for val in in_scope[col].unique().sort().to_list():
            sub = in_scope.filter(pl.col(col) == val)
            _print_metrics(f"{label}={val}", sub["gold_veracity"].to_numpy(), sub["veracity"].to_numpy())
    # has_image split
    if "has_image" in in_scope.columns:
        for hv in [True, False]:
            sub = in_scope.filter(pl.col("has_image") == hv)
            _print_metrics(f"has_image={hv}", sub["gold_veracity"].to_numpy(), sub["veracity"].to_numpy())

    # Media-dependency carve (content but OUT of stage-3 scope) — the Phase-1b authentication seed,
    # reported separately: a text-only verifier structurally can't reach the gold here.
    print(f"\n### Media-dependency carve (content, OUT of stage-3 scope — Phase-1b seed, NOT verifier failures)")
    for reason in ("video", "provenance_recency"):
        sub = carved.filter(pl.col("out_of_scope_reason") == reason)
        _print_metrics(f"carved: {reason}", sub["gold_veracity"].to_numpy(), sub["veracity"].to_numpy())

    # Non-content axes (carried, not headline).
    print(f"\n### Other axes (excluded from headline)")
    for ax in ("artifact", "attribution"):
        sub = ok.filter(pl.col("judged_axis") == ax)
        _print_metrics(f"judged_axis={ax}", sub["gold_veracity"].to_numpy(), sub["veracity"].to_numpy())

    # Nudge collapse on the headline (content in-scope) population.
    _nudge_collapse(in_scope["gold_veracity"].to_numpy(), in_scope["veracity"].to_numpy())

    # Loop cost.
    if "elapsed_seconds" in ok.columns:
        print(f"\nloop: avg rounds={ok['rounds_used'].mean():.2f}  avg llm_calls={ok['llm_calls'].mean():.1f}  "
              f"avg elapsed={ok['elapsed_seconds'].mean():.1f}s")

    # A few large-error headline cases to eyeball.
    big = in_scope.with_columns((pl.col("veracity") - pl.col("gold_veracity")).abs().alias("_e")) \
        .filter(pl.col("_e") >= 2).sort("_e", descending=True)
    if big.height:
        print(f"\nlarge errors (|pred-gold|>=2), first {min(10, big.height)}/{big.height}:")
        for r in big.head(10).iter_rows(named=True):
            print(f"  gold {r['gold_veracity']} -> pred {r['veracity']}  [{r['rating_subtype']}]  "
                  f"{r['claim_text'][:80]}")


if __name__ == "__main__":
    main()
