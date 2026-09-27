"""Evaluate the selection cascade against CheckThat! check-worthiness gold.

GOLD: CLEF CheckThat! 2022, Task 1, subtask-1A English (COVID-19 tweets), binary
`class_label` (1 = check-worthy). This is an EXTERNAL gold standard — the first
accuracy (not agreement) number for our selection gates.
  re-download: gitlab.com/checkthat_lab/clef2022-checkthat-lab → task1/data/subtasks-english

We run our gates (zero-shot, gpt-oss-120b) on each tweet and score the prediction
at three cascade cut points against the gold:
  V       = Selection (verifiable?)                      — prompts_claimify
  V∧R     = Selection ∧ Relevance (public consequence)   — + prompts_relevance
  V∧R∧H   = Selection ∧ Relevance ∧ Harm (FABLE flag)    — + fable
"check-worthy" ≈ verifiable + public-interest (+ harm), so the cut points test
which combination best matches human check-worthiness (and gives R/H bundled
validation). Scoring is deterministic; only the 3 gate calls/tweet cost anything.

  uv run python -m eval.scripts.feed_study.eval_checkworthy --splits test_gold dev_test
  uv run python -m eval.scripts.feed_study.eval_checkworthy --splits test_gold -n 8   # smoke
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from openai import (  # noqa: E402
    APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError,
)
from dotenv import load_dotenv  # noqa: E402

from eval.scripts.feed_study.prompts_claimify import build_selection_messages, parse_selection  # noqa: E402
from eval.scripts.feed_study.prompts_relevance import build_relevance_messages, parse_relevance  # noqa: E402
from eval.scripts.feed_study.fable import (  # noqa: E402
    build_fable_post_messages, parse_fable, fable_flag, fable_total,
)

load_dotenv(ROOT / ".env")

DATA = Path(__file__).parent / "data" / "checkthat22"
DEFAULT_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.1
MAX_TOKENS = 4000
MAX_RETRIES = 5

SCHEMA = {
    "split": pl.String, "tweet_id": pl.String, "text": pl.String, "gold": pl.Int64,
    "verifiable": pl.Boolean, "in_scope": pl.Boolean, "harm_flag": pl.Boolean,
    "harm_total": pl.Int64, "error": pl.String,
}

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"])
    return _client


def _retryable(e: Exception) -> bool:
    if isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    s = getattr(e, "status_code", None)
    return isinstance(e, APIError) and isinstance(s, int) and s >= 500


def call_llm(messages, model: str) -> str:
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = get_client().chat.completions.create(
                model=model, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            if attempt == MAX_RETRIES or not _retryable(e):
                raise
            time.sleep(min(2 ** attempt + random.random(), 30.0))
    raise RuntimeError("unreachable")


def gate_one(split: str, tweet_id: str, text: str, gold: int, model: str) -> dict:
    """Run all three gates on one tweet."""
    out = {"split": split, "tweet_id": tweet_id, "text": text, "gold": gold,
           "verifiable": None, "in_scope": None, "harm_flag": None, "harm_total": None,
           "error": None}
    try:
        out["verifiable"] = parse_selection(call_llm(build_selection_messages(text), model))["verifiable"]
        out["in_scope"] = parse_relevance(call_llm(build_relevance_messages(text), model))["in_scope"]
        dims = parse_fable(call_llm(build_fable_post_messages(text), model))
        out["harm_flag"] = fable_flag(dims)
        out["harm_total"] = fable_total(dims)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def load_splits(data: Path, splits: list[str]) -> list[tuple]:
    jobs = []
    for sp in splits:
        f = data / f"CT22_english_1A_checkworthy_{sp}.tsv"
        if not f.exists():
            raise SystemExit(f"missing split file: {f}")
        for r in csv.DictReader(f.open(encoding="utf-8"), delimiter="\t"):
            txt = (r.get("tweet_text") or "").strip()
            if txt:
                jobs.append((sp, str(r["tweet_id"]), txt, int(r["class_label"])))
    return jobs


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _prf(pred: list[bool], gold: list[int]) -> dict:
    tp = sum(1 for p, g in zip(pred, gold) if p and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p and g == 0)
    fn = sum(1 for p, g in zip(pred, gold) if not p and g == 1)
    tn = sum(1 for p, g in zip(pred, gold) if not p and g == 0)
    n = len(gold)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # negative-class F1 for macro
    pn = tn / (tn + fn) if tn + fn else 0.0
    rn = tn / (tn + fp) if tn + fp else 0.0
    f1n = 2 * pn * rn / (pn + rn) if pn + rn else 0.0
    return {"P": prec, "R": rec, "F1": f1, "macroF1": (f1 + f1n) / 2,
            "acc": (tp + tn) / n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "pred_pos": tp + fp}


def score(df: pl.DataFrame) -> str:
    ok = df.filter(pl.col("error").is_null())
    gold = ok["gold"].to_list()
    n = len(gold)
    npos = sum(gold)
    L = [f"\n{'='*72}", "CheckThat! 2022 1A-EN — check-worthiness vs our selection cascade",
         f"scored tweets: {n}  (check-worthy gold: {npos}, {100*npos/n:.0f}%)  "
         f"splits: {dict(ok.group_by('split').len().iter_rows())}", "=" * 72]
    V = ok["verifiable"].fill_null(False).to_list()
    R = ok["in_scope"].fill_null(False).to_list()
    H = ok["harm_flag"].fill_null(False).to_list()
    cuts = {
        "V  (Selection)": V,
        "V∧R (Sel∧Relevance)": [a and b for a, b in zip(V, R)],
        "V∧R∧H (full cascade)": [a and b and c for a, b, c in zip(V, R, H)],
    }
    L.append(f"\n{'cut point':24s} {'P':>6} {'R':>6} {'F1':>6} {'macroF1':>8} {'acc':>6}   "
             f"{'pred+':>6}  (TP/FP/FN/TN)")
    for name, pred in cuts.items():
        m = _prf(pred, gold)
        L.append(f"{name:24s} {m['P']:6.3f} {m['R']:6.3f} {m['F1']:6.3f} {m['macroF1']:8.3f} "
                 f"{m['acc']:6.3f}   {m['pred_pos']:6d}  ({m['tp']}/{m['fp']}/{m['fn']}/{m['tn']})")
    # baselines
    maj_acc = (n - npos) / n
    L.append(f"\nbaselines: majority(all-No) acc={maj_acc:.3f} | all-Yes P={npos/n:.3f} R=1.000 "
             f"F1={2*(npos/n)/((npos/n)+1):.3f}")
    # gate firing rates (does Relevance saturate on a single-topic COVID set?)
    L.append(f"\ngate firing rates: V={100*sum(V)/n:.0f}%  R={100*sum(R)/n:.0f}%  "
             f"H={100*sum(H)/n:.0f}%   (all tweets are COVID/health)")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--splits", nargs="+", default=["test_gold", "dev_test"])
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-w", "--workers", type=int, default=12)
    ap.add_argument("-n", "--limit", type=int, default=None)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--score-only", action="store_true", help="re-score the cached parquet, no LLM")
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    out = args.out or args.data / f"gates_{'_'.join(args.splits)}_{tag}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    all_jobs = load_splits(args.data, args.splits)
    if args.limit:
        all_jobs = all_jobs[:args.limit]

    existing = pl.read_parquet(out) if out.exists() else pl.DataFrame(schema=SCHEMA)
    if args.score_only:
        print(score(existing))
        return

    done = set(existing.filter(pl.col("error").is_null())["tweet_id"].to_list())
    jobs = [j for j in all_jobs if j[1] not in done]
    print(f"tweets: {len(all_jobs)} | to run: {len(jobs)} (cached ok: {len(done)}) | "
          f"model: {args.model} | ~{3*len(jobs)} LLM calls -> {out}")

    combined = existing
    if jobs:
        t0 = time.time(); results = []; n_err = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(gate_one, sp, tid, txt, gold, args.model) for sp, tid, txt, gold in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result(); results.append(r)
                if r["error"]:
                    n_err += 1
                    print(f"  [{i}/{len(jobs)}] {r['tweet_id']} ERROR: {r['error'][:80]}")
                elif i % 50 == 0 or i == len(jobs):
                    print(f"  [{i}/{len(jobs)}] done")
        new = pl.DataFrame(results, schema=SCHEMA)
        keep = existing.join(pl.DataFrame({"tweet_id": [j[1] for j in jobs]},
                                          schema={"tweet_id": pl.String}), on="tweet_id", how="anti")
        combined = pl.concat([keep, new]) if len(keep) else new
        combined.write_parquet(out)
        print(f"\nran {len(new)} tweets in {time.time()-t0:.1f}s. errors: {n_err}/{len(jobs)} -> {out}")

    print(score(combined))


if __name__ == "__main__":
    main()
