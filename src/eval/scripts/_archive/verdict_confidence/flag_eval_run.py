"""Flag-step eval: for each text claim, reconstruct a verdict-neutral contextualized POST, run
verify(claim, post_context=POST) — atomic veracity (context-free) + the §3b post-FLAG step — and
record both the bare (veracity>=4) and the context-aware (post_flag) binary decisions vs gold.

Tests whether including the raw post at the flag step recovers the soft_flag misses WITHOUT
over-flagging clear-PASS claims. Resumable. Run (from src/):
  uv run python -m eval.scripts.verdict_confidence.flag_eval_run --ids ev1630,ev1689,ev238,ev664   # smoke
  uv run python -m eval.scripts.verdict_confidence.flag_eval_run -w 6                               # all text
"""
from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from eval.scripts.verdict_confidence.reconstruct_context import reconstruct
from pipeline.config import SEARCH_CASCADE
from pipeline.embedding import embed
from pipeline.models import AtomicClaim
from pipeline.verify import verify

SPLITS = Path("eval/scripts/verdict_confidence/data/eval_v1_splits.parquet")
OUT = Path("eval/scripts/verdict_confidence/data/flag_eval.parquet")
# Verdict/correction words that should NOT appear in a verdict-neutral reconstruction (leak check).
LEAK_RX = re.compile(r"mislead|\bfalse\b|\bfake\b|overstat|exaggerat|debunk|no evidence|not true|"
                     r"incorrect|hoax|in reality|but in fact|out of context|misrepresent|unfounded",
                     re.I)


def _pred(is_pass: bool) -> str:
    return "pass" if is_pass else "flag"


def run_one(row: dict) -> dict:
    t0 = time.time()
    base = {"claim_id": row["claim_id"], "claim_text": row["claim_text"],
            "gold": row["binary_label"], "soft_flag": row["soft_flag"], "split": row["split"],
            "original_rating": row["original_rating"]}
    try:
        rc = reconstruct(row["claim_text"], row["claimant"], row["claim_date"], row["review_title"] or "")
        post = rc["contextualized_claim"] or row["claim_text"]
        leak = bool(LEAK_RX.search(post))

        claim = AtomicClaim(text=row["claim_text"], embedding=embed(row["claim_text"]).tolist())
        v = verify(claim, date_ceiling=None, exclude_urls=[row["review_url"]] if row["review_url"] else [],
                   providers=list(SEARCH_CASCADE), post_context=post)

        bare_pred = _pred(v.scores.veracity >= 4)
        post_pred = bare_pred if v.post_flag is None else _pred(not v.post_flag)
        return {**base, "contextualized_post": post, "recon_added": rc["added_context"],
                "leak_suspect": leak, "veracity": v.scores.veracity, "bare_pred": bare_pred,
                "post_flag": v.post_flag, "post_pred": post_pred, "post_flag_reason": v.post_flag_reason,
                "bare_correct": int(bare_pred == row["binary_label"]),
                "post_correct": int(post_pred == row["binary_label"]),
                "llm_calls": v.llm_calls, "elapsed_s": round(time.time() - t0, 1), "error": None}
    except Exception as exc:  # noqa: BLE001
        return {**base, "contextualized_post": "", "recon_added": "", "leak_suspect": False,
                "veracity": 0, "bare_pred": "", "post_flag": None, "post_pred": "",
                "post_flag_reason": "", "bare_correct": 0, "post_correct": 0, "llm_calls": 0,
                "elapsed_s": round(time.time() - t0, 1), "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=None, help="comma-sep claim_ids (smoke); else all text claims")
    ap.add_argument("-w", "--workers", type=int, default=6)
    ap.add_argument("-n", "--limit", type=int, default=None)
    args = ap.parse_args()

    df = pl.read_parquet(SPLITS).filter((pl.col("modality") == "text") & pl.col("binary_label").is_not_null())
    if args.ids:
        df = df.filter(pl.col("claim_id").is_in({s.strip() for s in args.ids.split(",")}))
    if args.limit:
        df = df.head(args.limit)

    existing_ids, existing_rows = set(), []
    if OUT.exists():
        ex = pl.read_parquet(OUT)
        existing_ids = set(ex["claim_id"].to_list())
        existing_rows = ex.to_dicts()
    to_run = [r for r in df.to_dicts() if r["claim_id"] not in existing_ids]
    print(f"text claims: {df.height}  existing: {len(existing_ids)}  to do: {len(to_run)}  workers: {args.workers}")
    if not to_run:
        print("nothing to do.")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    embed("warmup")

    t0 = time.time()
    new_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, r): r["claim_id"] for r in to_run}
        for j, fut in enumerate(as_completed(futs), 1):
            new_rows.append(fut.result())
            if j % 10 == 0 or j == len(to_run):
                pl.DataFrame(existing_rows + new_rows).write_parquet(OUT)
                print(f"  [{j}/{len(to_run)}] done ({time.time() - t0:.0f}s)")

    df_all = pl.DataFrame(existing_rows + new_rows)
    df_all.write_parquet(OUT)
    errs = sum(1 for r in new_rows if r.get("error"))
    leaks = sum(1 for r in new_rows if r.get("leak_suspect"))
    print(f"\ndone in {time.time() - t0:.0f}s. {df_all.height} rows -> {OUT}. errors: {errs}, leak-suspect: {leaks}")


if __name__ == "__main__":
    main()
