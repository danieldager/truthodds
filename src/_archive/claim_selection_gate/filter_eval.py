"""Filter-eval harness — score the production FILTER (small model + `eval.filter.FILTER_SYSTEM`, a prompt
DECOUPLED from the audit/gold) vs the filter_dev gold.

  uv run python -m eval.scripts.filter_eval --smoke
  uv run python -m eval.scripts.filter_eval --model Qwen/Qwen3-VL-30B-A3B-Instruct     # production candidate
  uv run python -m eval.scripts.filter_eval --model google/gemma-4-26B-A4B-it

Gold = `filter_dev.parquet` (235B audit + human review). Usable rows = recommend ∈ {positive, negative}
(drops excluded — unmeasurable). The filter sees only the post (blind) and FLAGS it (positive) iff
political_implication ∧ has_claim ∧ ¬obvious_joke ∧ readable. Predictions cached →
`filter_eval_<model>.parquet` (re-score is free). Reports accuracy, flag precision/recall/F1, the confusion
matrix, and rejection-recall by gold neg_reason — read against the inter-annotator CEILING (the 235B gold
labeller matches the human only ~48% on the decisive borderlines; clog 280628), so ~100% is not reachable.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import polars as pl

from eval.filter import FILTER_MODEL_DEFAULT, pass_filter
from eval.scripts._pool import pooled_checkpointed

GOLD = Path("eval/data/filter_dev.parquet")


def _pred(out: dict) -> str:
    return "positive" if out.get("flag") else "negative"   # the filter's own keep/reject decision


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=FILTER_MODEL_DEFAULT)
    ap.add_argument("--smoke", action="store_true", help="30 rows, print only")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    out_path = Path(f"eval/data/filter_eval_{args.model.split('/')[-1]}.parquet")

    rows = pl.read_parquet(GOLD).filter(pl.col("recommend").is_in(["positive", "negative"])).to_dicts()
    if args.smoke:
        rows = rows[:30]
    print(f"filter-eval: {args.model} on {len(rows)} usable rows", flush=True)

    res: dict = {}
    if out_path.exists() and not args.smoke:
        res = {r["key"]: r for r in pl.read_parquet(out_path).to_dicts()}
    todo = [i for i, r in enumerate(rows) if r["key"] not in res]

    def work(i):
        r = rows[i]
        o = pass_filter({"raw_context": r.get("raw_context"), "image_paths": r.get("image_paths")}, args.model)
        return r["key"], {"key": r["key"], "gold": r["recommend"], "neg_reason": r.get("neg_reason"),
                          "pred": _pred(o), "err": "_err" in o}

    def flush():
        if res and not args.smoke:
            pl.DataFrame(list(res.values())).write_parquet(out_path)

    if args.smoke:
        for i in range(len(rows)):
            k, p = work(i); res[k] = p
    else:
        pooled_checkpointed(todo, work, lambda k, p: res.__setitem__(k, p), flush, args.workers,
                            "filter-eval", checkpoint_every=100)
        flush()

    P = [r for r in res.values() if not r.get("err")]
    tp = sum(r["gold"] == "positive" and r["pred"] == "positive" for r in P)
    fp = sum(r["gold"] == "negative" and r["pred"] == "positive" for r in P)
    fn = sum(r["gold"] == "positive" and r["pred"] == "negative" for r in P)
    tn = sum(r["gold"] == "negative" and r["pred"] == "negative" for r in P)
    n = len(P)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"\n===== {args.model} vs filter_dev gold (n={n}, errors {len(res)-n}) =====")
    print(f"accuracy {acc:.0%} | FLAG precision {prec:.0%} recall {rec:.0%} F1 {f1:.2f}")
    print(f"confusion: TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    print("rejection recall by gold neg_reason (filter correctly says negative):")
    for reason in ("not_political", "obvious_joke", "no_claim", "human_review", "other"):
        sub = [r for r in P if r["gold"] == "negative" and r["neg_reason"] == reason]
        if sub:
            rr = sum(r["pred"] == "negative" for r in sub) / len(sub)
            print(f"   {reason:14} {sum(r['pred']=='negative' for r in sub)}/{len(sub)}  {rr:.0%}")
    print("\nCEILING context: the 235B gold labeller agrees with the human only ~48% on the decisive "
          "borderlines (~20% of rows are irreducibly subjective) → ~100% is not reachable; judge against that.")


if __name__ == "__main__":
    main()
