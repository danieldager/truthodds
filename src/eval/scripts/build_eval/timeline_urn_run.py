"""TRUE URN — evidence reads over the live timeline corpus (two-urn pivot).

The TRUE side of the tweet fit corpus. Reads the SAME instrument as the C2 FALSE
urn (`cn_false_urn.py --reads`), row-for-row: query-v3 / read-v5 via
`evidence_urn_run.run_claim`, origin site x.com excluded, and CONTEXT OFF
(`x_context=None, context_ok=False`). The context switch is the one setting that
would silently make the two urns incomparable — the FALSE side ran without post
context, so this side must too.

Parity screen (the timeline analogue of the FALSE urn's
`checkworthy & ~attribution_form & ~media_locus & ~claim_dup & ~e1_leak`):

    checkworthy AND type == "assertion" AND lang == "en"

`type == "assertion"` is `~attribution_form`; `lang == "en"` matches the FALSE
pool, which is EN-only. Topic gating (`verify_eligible`) is deliberately NOT
applied: the FALSE urn has no topic gate, and its sports/entertainment share is
24% — dropping those topics here would build the asymmetry into the fit.

Frame is stamped on every record. The urn is bimodal in age by construction
(feed ~0-2 days, search_june ~82-194 days), so the recency confound is testable
WITHIN this urn — and the June frame is age-matched to the FALSE urn's mass.

  uv run python -m eval.scripts.build_eval.timeline_urn_run --smoke     # 25 claims
  uv run python -m eval.scripts.build_eval.timeline_urn_run --reads     # full parity set

Output: eval/data/urn_runs/true_timeline/{smoke,scores}.jsonl (resume key = claim_id)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from pipeline.search import newsguard_score_map  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
    run_claim, QUERY_PROMPT_V, READ_PROMPT_V)

URN = SRC / "eval/data/tweet_corpus/timeline_urn.parquet"
OUT = SRC / "eval/data/urn_runs/true_timeline"
READS_CAP = 3.00   # USD hard cap (est $1.85 at the C2 unit cost of $0.0012/claim)
SEED = 20260826


def parity(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("checkworthy") & (pl.col("type") == "assertion")
                     & (pl.col("lang") == "en"))


def adapt(c: dict) -> dict:
    """Timeline claim -> the row shape cn_false_urn.reads() feeds run_claim."""
    return {
        "review_url": c["claim_id"], "claim_text": c["claim"],
        "publisher_site": "x.com", "claim_date": _post_date(c["created_at"]),
        "claim_type": c["type"], "topic": c.get("topic"),
        "x_context": None, "context_ok": False,
        "resolution_status": "native", "claim_resolved": None,
        "veracity": None, "rating_subtype": None,
        "review_date": None, "x_date": None, "yr": None,
        "screen_verdict": "unscreened", "screen_leak": False,
        "post_id": c["post_id"], "frame": c["frame"], "handle": c["handle"],
        "account": c.get("account"), "post_voice": c.get("voice"),
        "verify_eligible": c["verify_eligible"]}


def _post_date(created_at: str) -> str:
    """'Tue Aug 25 04:33:44 +0000 2026' -> '2026-08-25' (post date IS utterance date)."""
    import datetime as dt
    return dt.datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y").strftime("%Y-%m-%d")


def reads(smoke: bool, workers: int, cap: float) -> None:
    df = pl.read_parquet(URN)
    n0 = df.height
    df = parity(df)
    print(f"claims {n0} -> parity {df.height} "
          f"({df['post_id'].n_unique()} posts, {df['handle'].n_unique()} handles)",
          flush=True)
    print("  frames: " + json.dumps(
        {r["frame"]: r["len"] for r in df.group_by("frame").len().iter_rows(named=True)}),
        flush=True)
    rows = [adapt(c) for c in df.sort("claim_id").iter_rows(named=True)]

    if smoke:
        # stratified by frame so the shakeout exercises both age halves
        by_frame = {}
        for r in rows:
            by_frame.setdefault(r["frame"], []).append(r)
        rng = random.Random(SEED)
        rows = [r for f in sorted(by_frame)
                for r in rng.sample(by_frame[f], min(12, len(by_frame[f])))]
        out = OUT / "smoke.jsonl"
    else:
        # shuffle POSTS, not claims: every prefix is then a post-complete
        # stratified sample, so a partial run is still readable (E2, 2026-08-05)
        by_post = {}
        for r in rows:
            by_post.setdefault(r["post_id"], []).append(r)
        order = sorted(by_post)
        random.Random(SEED).shuffle(order)
        rows = [r for pid in order for r in by_post[pid]]
        out = OUT / "scores.jsonl"

    OUT.mkdir(parents=True, exist_ok=True)
    seen = {json.loads(l)["review_url"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["review_url"] not in seen]
    print(f"reads: {len(todo)}/{len(rows)} claims -> {out.name} | cap ${cap} | "
          f"{workers} workers | prompts {QUERY_PROMPT_V}/{READ_PROMPT_V}", flush=True)

    ng = newsguard_score_map()
    lock = threading.Lock()
    budget = {"spent": 0.0, "done": 0, "cached_tok": 0, "prompt_tok": 0, "stop": False}
    t0 = time.time()
    fh = open(out, "a")

    def work(row):
        if budget["stop"]:
            return
        try:
            rec = run_claim(row, ng, budget, {})
        except Exception as e:
            with lock:
                budget["done"] += 1
                print(f"  [claim-failed] {row['review_url']}: {type(e).__name__}: {e}",
                      flush=True)
            return
        for k in ("post_id", "frame", "handle", "account", "post_voice", "verify_eligible"):
            rec[k] = row[k]
        with lock:
            budget["spent"] += rec["cost"]
            budget["done"] += 1
            fh.write(json.dumps(rec) + "\n")
            n, el = budget["done"], (time.time() - t0) / 60
            if n % 20 == 0 or n == len(todo):
                fh.flush()
                proj = budget["spent"] / n * len(todo)
                print(f"  {n}/{len(todo)} | ${budget['spent']:.3f} | proj ${proj:.2f} | "
                      f"{el:.1f}m | {n/max(el,.01):.1f}/min | "
                      f"ETA {el/n*(len(todo)-n):.0f}m", flush=True)
                if proj > cap and n >= 20:
                    print(f"  BUDGET ABORT: projection ${proj:.2f} > cap ${cap}", flush=True)
                    budget["stop"] = True

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"reads done {budget['done']} | ${budget['spent']:.4f} | "
          f"{(time.time()-t0)/60:.1f}m", flush=True)
    report(out)


def report(path: Path) -> None:
    if not path.exists():
        return
    recs = [json.loads(l) for l in open(path)]
    print(f"\n== {path.name}: {len(recs)} claims, "
          f"{sum(len(r.get('results') or []) for r in recs)} docs ==", flush=True)
    for frame in sorted({r.get("frame") for r in recs}):
        sub = [r for r in recs if r.get("frame") == frame]
        f = Counter(d["read"]["direction"] for r in sub for d in (r.get("results") or [])
                    if d.get("read"))
        tot = sum(f.values()) or 1
        sup = (f["5"] + f["4"]) / tot
        ref = (f["1"] + f["2"]) / tot
        print(f"  {frame:<12} {len(sub):>5} claims | {tot:>6} docs | "
              f"support {sup:.1%} refute {ref:.1%} silent {1-sup-ref:.1%}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="24 claims, 12 per frame")
    ap.add_argument("--reads", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--cap", type=float, default=READS_CAP)
    a = ap.parse_args()
    if a.smoke or a.reads:
        reads(a.smoke, a.workers, 0.20 if a.smoke else a.cap)
    if a.report:
        report(OUT / "scores.jsonl")


if __name__ == "__main__":
    main()
