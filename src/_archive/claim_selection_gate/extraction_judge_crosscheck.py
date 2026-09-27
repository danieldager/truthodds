"""Cross-validate the extraction-eval judge: re-judge an existing results parquet with an INDEPENDENT
judge model and measure agreement vs the original DeepSeek-V4-Flash verdicts.

  uv run python -m eval.scripts.extraction_judge_crosscheck --results eval/data/extraction_eval_Qwen3-VL-30B-A3B-Instruct_single.parquet --judge google/gemma-3-27b-it

The extracted claims are FIXED (read from the parquet); only the judge swaps. Confirms the
match/partial/miss verdicts aren't an artifact of one judge. Reports 3-way + collapsed (match vs
not-match) agreement and Cohen's kappa over the content+attribution headline rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from eval.scripts.image_extraction_eval import JUDGE_SYS, _chat, HEADLINE_AXES
from eval.scripts._pool import pooled_checkpointed


def _judge_with(model: str, gold: str, claims: list[str]) -> str:
    if not claims:
        return "miss"
    listing = "\n".join(f"[{i}] {c}" for i, c in enumerate(claims))
    jd = _chat(model, [{"role": "system", "content": JUDGE_SYS},
               {"role": "user", "content": f"GOLD CLAIM:\n{gold}\n\nEXTRACTED CLAIMS:\n{listing}"}],
               max_tokens=400)
    return jd.get("verdict") or "error"


def _kappa(a: list[str], b: list[str], labels: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--judge", default="google/gemma-3-27b-it", help="independent judge model")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    src = pl.read_parquet(args.results).filter(
        (pl.col("verdict") != "error") & pl.col("judged_axis").is_in(HEADLINE_AXES))
    rows = src.to_dicts()
    if args.max:
        rows = rows[: args.max]
    out = Path(args.results.replace(".parquet", f"_xjudge_{args.judge.split('/')[-1]}.parquet"))
    done = set(pl.read_parquet(out)["review_url"].to_list()) if out.exists() else set()
    todo = [i for i, r in enumerate(rows) if r["review_url"] not in done]
    print(f"rows {len(rows)} | done {len(done)} | todo {len(todo)} | judge={args.judge}", flush=True)
    res: dict[str, dict] = {}

    def flush():
        if not res:
            return
        new = pl.DataFrame(list(res.values()))
        comb = new if not out.exists() else pl.concat([pl.read_parquet(out), new], how="diagonal_relaxed").unique("review_url", keep="last")
        comb.write_parquet(out)

    def work(i):
        r = rows[i]
        try:
            return i, {"v_alt": _judge_with(args.judge, r["gold"], r["claims"])}
        except Exception as e:
            return i, {"v_alt": "error", "err": str(e)[:80]}

    def apply(i, p):
        r = rows[i]
        res[r["review_url"]] = {"review_url": r["review_url"], "v_ds": r["verdict"], **p}

    if todo:
        ab = pooled_checkpointed(todo, work, apply, flush, args.workers, f"xjudge:{args.judge.split('/')[-1]}", checkpoint_every=40)
        flush()

    df = pl.read_parquet(out).filter(pl.col("v_alt") != "error")
    a = df["v_ds"].to_list()
    b = df["v_alt"].to_list()
    n = len(a)
    labels = ["match", "partial", "miss"]
    exact = sum(x == y for x, y in zip(a, b)) / n * 100
    coll = lambda v: "match" if v == "match" else "not"
    ca, cb = [coll(x) for x in a], [coll(x) for x in b]
    coll_agree = sum(x == y for x, y in zip(ca, cb)) / n * 100
    print(f"\n=== {Path(args.results).stem} judged by DeepSeek vs {args.judge} (n={n}) ===")
    print(f"  3-way exact agreement: {exact:.0f}%  (kappa {_kappa(a, b, labels):.2f})")
    print(f"  collapsed match-vs-not: {coll_agree:.0f}%  (kappa {_kappa(ca, cb, ['match', 'not']):.2f})")
    print(f"  DeepSeek match-rate {a.count('match')/n*100:.0f}% | {args.judge.split('/')[-1]} match-rate {b.count('match')/n*100:.0f}%")
    print("  confusion (rows=DeepSeek, cols=alt):")
    print(f"    {'':9s}" + "".join(f"{l:>9s}" for l in labels))
    for la in labels:
        print(f"    {la:9s}" + "".join(f"{sum(1 for x,y in zip(a,b) if x==la and y==lb):>9d}" for lb in labels))
    import os, sys; sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
