"""Multi-model agreement harness (the self-consistency pivot).

Finding (clog/140626): resampling one model off a fixed analysis gives zero verdict spread —
the synthesis already states the verdict in prose and the Likert just transcribes it. So the
confidence signal must come from *independent judgments*, not resampling. This harness measures
INTER-MODEL agreement, production-faithful:

  1. gpt-oss-120b runs the full verify() loop — ITS retrieval, ITS evidence pool (this is what
     production uses, so retrieval stays gpt-oss).
  2. Each panel model then runs its OWN synthesis + verdict on that SAME fixed evidence pool
     (NOT on gpt-oss's analysis — that would force agreement). No extra searches.
  3. Record all panel verdicts → the spread across models (per dim) is the confidence signal.

Stage 1 = ~60 claims stratified toward the decision boundary (a signal-existence probe, NOT
representative). Resumable via claim_id. See agreement_analyze.py for the spread/agreement read.

Run (from src/):
  uv run python -m eval.scripts.verdict_confidence.agreement_run --stratified -w 4
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from pipeline.config import SEARCH_CASCADE
from pipeline.embedding import embed
from pipeline.models import AtomicClaim, EvidenceDoc
from pipeline.verify import _evaluate_likert, _synthesise, verify

# Panel: 3 vendors / 4 architectures. gpt-oss is PRIMARY (does retrieval — matches production).
PANEL = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]
PRIMARY = "openai/gpt-oss-120b"
DIMS = ("veracity", "evidence_sufficiency", "evidence_agreement", "source_reliability")
MODALITY_PARQUET = Path("eval/scripts/verdict_confidence/data/eval_v1_modality.parquet")
OUT = Path("eval/scripts/verdict_confidence/data/agreement.parquet")
SAMPLE_OUT = Path("eval/scripts/verdict_confidence/data/stage1_sample.parquet")

# Boundary ratings: where binarization is hardest and models SHOULD split.
_BOUNDARY_RX = (r"half[ -]?true|mostly true|mostly false|mixture|misleading|"
                r"partly (false|true)|missing context|needs context")


def build_stage1_sample(seed: int = 42) -> pl.DataFrame:
    """~60 claims stratified toward the decision boundary (text-modality only). Seeded for
    reproducibility. NOT representative — deliberately oversamples the hard middle."""
    df = (pl.read_parquet(MODALITY_PARQUET).with_row_index("ridx")
          .with_columns(claim_id=pl.format("ev{}", "ridx"))
          .filter((pl.col("modality") == "text") & pl.col("binary_label").is_not_null()))
    rating = pl.col("original_rating").fill_null("").str.to_lowercase()
    is_boundary = (pl.col("harmonised_label") == "Conflicting Evidence") | rating.str.contains(_BOUNDARY_RX)
    is_mostly_true = rating.str.contains("mostly true")
    strata = {
        "boundary_pass": df.filter(is_boundary & is_mostly_true),                    # target 8
        "boundary_flag": df.filter(is_boundary & ~is_mostly_true),                   # target 30
        "clear_pass": df.filter(~is_boundary & (rating == "true")),                  # target 10
        "clear_flag": df.filter(~is_boundary & rating.is_in(["false", "fake"])),     # target 12
    }
    targets = {"boundary_pass": 8, "boundary_flag": 30, "clear_pass": 10, "clear_flag": 12}
    picks = []
    for name, sub in strata.items():
        n = min(targets[name], sub.height)
        picks.append(sub.sample(n, seed=seed).with_columns(stratum=pl.lit(name)))
    return pl.concat(picks).select("claim_id", "claim_text", "binary_label", "modality",
                                   "original_rating", "stratum")


def reconstruct_pool(trace: dict) -> list[EvidenceDoc]:
    """Rebuild gpt-oss's accumulated evidence pool from the verify trace (dedup already done
    round-by-round in verify, so concatenating round docs is the full pool)."""
    return [EvidenceDoc(**d) for rnd in trace.get("rounds", []) for d in rnd["docs"]]


def run_one(row: dict) -> dict:
    t0 = time.time()
    base = {"claim_id": row["claim_id"], "claim_text": row["claim_text"],
            "gold": row["binary_label"], "modality": row["modality"],
            "original_rating": row["original_rating"], "stratum": row["stratum"],
            "model_order": list(PANEL)}
    try:
        claim = AtomicClaim(text=row["claim_text"], embedding=embed(row["claim_text"]).tolist())
        exclude = [row["exclude_url"]] if row.get("exclude_url") else []
        trace: dict = {}
        v = verify(claim, date_ceiling=None, exclude_urls=exclude,
                   providers=list(SEARCH_CASCADE), trace=trace, model=PRIMARY)
        pool = reconstruct_pool(trace)
        past_q = list(v.past_queries)

        # Each panel model: independent synthesis + verdict on the SAME fixed pool.
        panel = {}
        panel_calls = 0
        for m in PANEL:
            dec = _synthesise(claim.text, pool, past_q, m)
            lik = _evaluate_likert(claim.text, dec["analysis"], m)
            panel[m] = {**{d: lik[d] for d in DIMS}, "justification": lik["justification"],
                        "analysis": dec["analysis"]}
            panel_calls += 2

        out = {**base,
               "deployment_veracity": v.scores.veracity,
               "deployment_binary": "pass" if v.scores.veracity >= 4 else "flag",
               "deployment_analysis": v.analysis,
               "n_evidence_urls": len(v.evidence_urls), "verify_llm_calls": v.llm_calls,
               "panel_llm_calls": panel_calls, "elapsed_s": round(time.time() - t0, 1),
               "error": None}
        for d in DIMS:
            out[f"{d}_by_model"] = [panel[m][d] for m in PANEL]
        out["justification_by_model"] = [panel[m]["justification"] for m in PANEL]
        out["analysis_by_model"] = [panel[m]["analysis"] for m in PANEL]
        return out
    except Exception as exc:  # noqa: BLE001
        base.update({"deployment_veracity": 0, "deployment_binary": "", "deployment_analysis": "",
                     "n_evidence_urls": 0, "verify_llm_calls": 0, "panel_llm_calls": 0,
                     "elapsed_s": round(time.time() - t0, 1), "error": f"{type(exc).__name__}: {exc}"})
        for d in DIMS:
            base[f"{d}_by_model"] = []
        base["justification_by_model"] = []
        base["analysis_by_model"] = []
        return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratified", action="store_true", help="build+use the stage-1 stratified ~60")
    ap.add_argument("--ids", default=None, help="comma-sep claim_ids (overrides --stratified)")
    ap.add_argument("-o", "--output", type=Path, default=OUT)
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("-n", "--limit", type=int, default=None)
    args = ap.parse_args()

    sample = build_stage1_sample()
    SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.write_parquet(SAMPLE_OUT)
    print("stage-1 composition:")
    print(sample.group_by("stratum").agg(pl.len().alias("n"),
          pl.col("binary_label").eq("pass").sum().alias("pass")).sort("stratum"))

    # leakage-exclude URL: the publisher review_url (join back from the full gold table)
    review = (pl.read_parquet(MODALITY_PARQUET).with_row_index("ridx")
              .with_columns(claim_id=pl.format("ev{}", "ridx")).select("claim_id", "review_url"))
    sample = sample.join(review, on="claim_id", how="left").rename({"review_url": "exclude_url"})

    if args.ids:
        want = {s.strip() for s in args.ids.split(",")}
        sample = sample.filter(pl.col("claim_id").is_in(want))
    if args.limit:
        sample = sample.head(args.limit)

    existing_ids, existing_rows = set(), []
    if args.output.exists():
        ex = pl.read_parquet(args.output)
        existing_ids = set(ex["claim_id"].to_list())
        existing_rows = ex.to_dicts()
        print(f"existing: {len(existing_ids)} done")
    to_run = [r for r in sample.to_dicts() if r["claim_id"] not in existing_ids]
    print(f"panel={', '.join(PANEL)}\nto do: {len(to_run)}  workers: {args.workers}")
    if not to_run:
        print("nothing to do.")
        return
    embed("warmup")

    t0 = time.time()
    new_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, r): r["claim_id"] for r in to_run}
        for j, fut in enumerate(as_completed(futs), 1):
            new_rows.append(fut.result())
            if j % 5 == 0 or j == len(to_run):
                pl.DataFrame(existing_rows + new_rows).write_parquet(args.output)
                print(f"  [{j}/{len(to_run)}] done")

    df = pl.DataFrame(existing_rows + new_rows)
    df.write_parquet(args.output)
    errs = sum(1 for r in new_rows if r.get("error"))
    print(f"\ndone in {time.time() - t0:.1f}s. {df.height} rows -> {args.output}. errors: {errs}")


if __name__ == "__main__":
    main()
