"""Step 2 — K-sample self-consistency resampling harness.

Per claim: run verify() ONCE (cascade serper→exa, no date ceiling, fact-check URL excluded to
block leakage) to get the FIXED synthesis `analysis`, then resample the Likert verdict step K
times at temperature>0 off that same analysis. Writes one row per claim with the K samples as
list columns; the scorer (score.py) derives every Part-A signal + Part-B metric from this.

ONE run over eval_v1 (+ optional synthetic corrections) feeds all four metric tables — the
scorer slices by `synthetic` (with/without) and `modality` (incl/excl artifact). Resumable via
claim_id dedup; per-row errors captured, never aborts the run.

Run (from src/):
  # pilot (paid — keep small):
  uv run python -m eval.scripts.verdict_confidence.resample_run -n 8 -k 10 --temp 1.0 -w 4
  # full run (HOLD for explicit go): add --with-synthetic, drop -n
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from config import VERIFICATION_MODEL
from eval.scripts.verdict_confidence.modality import tag_modality
from pipeline.config import SEARCH_CASCADE
from pipeline.embedding import embed
from pipeline.models import AtomicClaim
from pipeline.verify import _evaluate_likert, verify

DIMS = ("veracity", "evidence_sufficiency", "evidence_agreement", "source_reliability")
MODALITY_PARQUET = Path("eval/scripts/verdict_confidence/data/eval_v1_modality.parquet")
MINED_PARQUET = Path("eval/data/mined_corrections_v1.parquet")


def load_claims(with_synthetic: bool) -> pl.DataFrame:
    """Union of eval_v1 (modality-tagged) + optional synthetic corrections, with a stable
    claim_id, a leakage-exclude URL (review_url for gold, source_url for synthetic), and a
    `synthetic` flag. Columns are normalised so the two sources concat cleanly."""
    ev = (
        pl.read_parquet(MODALITY_PARQUET)
        .with_row_index("ridx")
        .select(
            claim_id=pl.format("ev{}", "ridx"),
            claim_text="claim_text",
            binary_label="binary_label",
            modality="modality",
            exclude_url=pl.col("review_url").fill_null(""),
            synthetic=pl.lit(False),
        )
    )
    if not with_synthetic:
        return ev
    mc = (
        tag_modality(pl.read_parquet(MINED_PARQUET))
        .with_row_index("ridx")
        .select(
            claim_id=pl.format("mc{}", "ridx"),
            claim_text="claim_text",
            binary_label="binary_label",
            modality="modality",
            exclude_url=pl.col("source_url").fill_null(""),
            synthetic=pl.lit(True),
        )
    )
    return pl.concat([ev, mc])


def _error_row(row: dict, t0: float, exc: Exception) -> dict:
    base = {"claim_id": row["claim_id"], "claim_text": row["claim_text"],
            "gold_binary_label": row["binary_label"], "modality": row["modality"],
            "synthetic": row["synthetic"], "analysis": "", "n_evidence_urls": 0,
            "rounds_used": 0, "stopped_reason": "", "providers_used": [],
            "verify_llm_calls": 0, "verify_elapsed_s": round(time.time() - t0, 2),
            "resample_elapsed_s": 0.0, "error": f"{type(exc).__name__}: {exc}"}
    for d in DIMS:
        base[f"prod_{d}"] = 0
        base[f"{d}_samples"] = []
    base["justification_samples"] = []
    return base


def run_one(row: dict, k: int, temp: float, model: str) -> dict:
    t0 = time.time()
    try:
        claim = AtomicClaim(text=row["claim_text"], embedding=embed(row["claim_text"]).tolist())
        exclude = [row["exclude_url"]] if row["exclude_url"] else []
        tv = time.time()
        v = verify(claim, date_ceiling=None, exclude_urls=exclude, providers=list(SEARCH_CASCADE))
        verify_s = round(time.time() - tv, 2)

        tr = time.time()
        samples = [_evaluate_likert(row["claim_text"], v.analysis, model, temperature=temp)
                   for _ in range(k)]
        resample_s = round(time.time() - tr, 2)

        out = {"claim_id": row["claim_id"], "claim_text": row["claim_text"],
               "gold_binary_label": row["binary_label"], "modality": row["modality"],
               "synthetic": row["synthetic"], "analysis": v.analysis,
               "n_evidence_urls": len(v.evidence_urls), "rounds_used": v.rounds_used,
               "stopped_reason": v.stopped_reason, "providers_used": list(v.providers_used),
               "verify_llm_calls": v.llm_calls, "verify_elapsed_s": verify_s,
               "resample_elapsed_s": resample_s, "error": None}
        for d in DIMS:
            out[f"prod_{d}"] = v.scores.model_dump()[d]
            out[f"{d}_samples"] = [s[d] for s in samples]
        out["justification_samples"] = [s["justification"] for s in samples]
        return out
    except Exception as exc:  # noqa: BLE001
        return _error_row(row, t0, exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("eval/scripts/verdict_confidence/data/resamples.parquet"))
    ap.add_argument("-k", "--samples", type=int, default=10)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("-n", "--limit", type=int, default=None, help="cap claims (smoke/pilot)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ids", default=None,
                    help="comma-sep claim_ids to run (curated pilot; overrides -n/--offset)")
    ap.add_argument("--with-synthetic", action="store_true",
                    help="append mined_corrections_v1 (the 'with synthetic' run set)")
    args = ap.parse_args()
    model = VERIFICATION_MODEL

    claims = load_claims(args.with_synthetic)
    if args.ids:
        want = {s.strip() for s in args.ids.split(",")}
        claims = claims.filter(pl.col("claim_id").is_in(want))
    elif args.offset or args.limit is not None:
        claims = claims.slice(args.offset, args.limit)

    existing_ids: set[str] = set()
    existing_rows: list[dict] = []
    if args.output.exists():
        ex = pl.read_parquet(args.output)
        existing_ids = set(ex["claim_id"].to_list())
        existing_rows = ex.to_dicts()
        print(f"existing: {len(existing_ids)} done")

    to_run = [r for r in claims.to_dicts() if r["claim_id"] not in existing_ids]
    print(f"model={model}  cascade={'→'.join(SEARCH_CASCADE)}  K={args.samples}  temp={args.temp}")
    print(f"output: {args.output}  | to do: {len(to_run)}  workers: {args.workers}")
    if not to_run:
        print("nothing to do.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    embed("warmup")

    t0 = time.time()
    new_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, r, args.samples, args.temp, model): r["claim_id"] for r in to_run}
        for j, fut in enumerate(as_completed(futs), 1):
            new_rows.append(fut.result())
            if j % 5 == 0 or j == len(to_run):
                # checkpoint: persist partial progress so a long run is resumable on interrupt
                pl.DataFrame(existing_rows + new_rows).write_parquet(args.output)
                print(f"  [{j}/{len(to_run)}] done")

    df = pl.DataFrame(existing_rows + new_rows)
    df.write_parquet(args.output)
    errs = sum(1 for r in new_rows if r.get("error"))
    avg_v = sum(r["verify_elapsed_s"] for r in new_rows) / len(new_rows)
    avg_r = sum(r["resample_elapsed_s"] for r in new_rows) / len(new_rows)
    print(f"\ndone in {time.time() - t0:.1f}s. {df.height} total rows -> {args.output}")
    print(f"new: {len(new_rows)}, errors: {errs}, avg verify {avg_v:.1f}s, avg resample {avg_r:.1f}s")


if __name__ == "__main__":
    main()
