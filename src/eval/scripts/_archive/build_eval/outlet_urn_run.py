"""TRUE URN (outlet arm) — evidence reads over reputable-outlet tweets.

An alternative TRUE side for the two-urn fit: instead of a mostly-true timeline,
posts from high-reliability news accounts (NewsGuard >= 80), treated as
environmentally-valid true claims. Same instrument as `timeline_urn_run.py` and
the C2 FALSE urn, row-for-row: query-v3 / read-v5 via `evidence_urn_run.run_claim`,
CONTEXT OFF (`x_context=None, context_ok=False`).

Two differences from the timeline arm, both forced by the source:

  ORIGIN. A timeline claim originates on x.com and nowhere else. An outlet claim
  originates on x.com AND the outlet's own site, so both are excluded
  (`exclude_extra`). This does NOT solve syndication — blocking apnews.com does
  not block the same AP wire story on fifty subscriber sites — which is exactly
  why the supports weights from this arm must be compared against the timeline
  arm's before anything is adopted.

  STRATUM. `frame` (feed / search_june) has no analogue here; `ng_tier`
  (ng90 = NG >= 90, ng80 = NG 80-89) is stamped in its place and reported the
  same way.

Parity screen matches timeline_urn_run.parity(): checkworthy AND assertion AND
en. Attribution-content claims (`paired_content`) are already dropped upstream in
true_urn_assertions.parquet; the column is carried so the unfiltered variant can
be refit for free.

  uv run python -m eval.scripts.build_eval.outlet_urn_run --smoke     # 24 claims
  uv run python -m eval.scripts.build_eval.outlet_urn_run --reads     # full parity set

Output: eval/data/urn_runs/true_outlet/{smoke,scores}.jsonl (resume key = claim_id)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from eval.scripts.build_eval import outlet_aliases

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from pipeline.search import newsguard_score_map  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
    run_claim, QUERY_PROMPT_V, READ_PROMPT_V)

URN = SRC / "eval/data/tweet_corpus/true_urn_assertions.parquet"
OUT = SRC / "eval/data/urn_runs/true_outlet"
READS_CAP = 4.00   # USD hard cap (est $2.80 at the C2 unit cost of ~$0.0023/claim)
SEED = 20260827


def parity(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("checkworthy") & (pl.col("type") == "assertion")
                     & (pl.col("lang") == "en"))


def adapt(c: dict) -> dict:
    """Outlet claim -> the row shape cn_false_urn.reads() feeds run_claim."""
    return {
        "review_url": c["claim_id"], "claim_text": c["claim"],
        "publisher_site": c["domain"],
        "exclude_extra": outlet_aliases.server_side(c["domain"]),
        "claim_date": _post_date(c["created_at"]),
        "claim_type": c["type"], "topic": c.get("topic"),
        "x_context": None, "context_ok": False,
        "resolution_status": "native", "claim_resolved": None,
        "veracity": None, "rating_subtype": None,
        "review_date": None, "x_date": None, "yr": None,
        "screen_verdict": "unscreened", "screen_leak": False,
        "post_id": c["post_id"], "ng_tier": c["ng_tier"], "handle": c["handle"],
        "domain": c["domain"], "ng_score": c.get("ng_score"),
        "paired_content": c.get("paired_content"),
        "verify_eligible": c["verify_eligible"]}


def _post_date(created_at: str) -> str:
    """'2026-07-06T20:01:16.000000Z' -> '2026-07-06' (post date IS utterance date)."""
    return dt.datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime("%Y-%m-%d")


def reads(smoke: bool, workers: int, cap: float) -> None:
    df = pl.read_parquet(URN)
    n0 = df.height
    df = parity(df)
    print(f"claims {n0} -> parity {df.height} "
          f"({df['post_id'].n_unique()} posts, {df['handle'].n_unique()} handles)",
          flush=True)
    print("  tiers: " + json.dumps(
        {r["ng_tier"]: r["len"] for r in df.group_by("ng_tier").len().iter_rows(named=True)}),
        flush=True)
    rows = [adapt(c) for c in df.sort("claim_id").iter_rows(named=True)]

    if smoke:
        # stratified by tier so the shakeout exercises both reliability halves
        by_tier = {}
        for r in rows:
            by_tier.setdefault(r["ng_tier"], []).append(r)
        rng = random.Random(SEED)
        rows = [r for t in sorted(by_tier)
                for r in rng.sample(by_tier[t], min(12, len(by_tier[t])))]
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
        for k in ("post_id", "ng_tier", "handle", "domain", "ng_score",
                  "paired_content", "verify_eligible"):
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
    docs = [d for r in recs for d in (r.get("results") or [])]
    print(f"\n== {path.name}: {len(recs)} claims, {len(docs)} docs ==", flush=True)
    for tier in sorted({r.get("ng_tier") for r in recs}):
        sub = [r for r in recs if r.get("ng_tier") == tier]
        f = Counter(d["read"]["direction"] for r in sub for d in (r.get("results") or [])
                    if d.get("read"))
        tot = sum(f.values()) or 1
        sup = (f["5"] + f["4"]) / tot
        ref = (f["1"] + f["2"]) / tot
        print(f"  {tier:<12} {len(sub):>5} claims | {tot:>6} docs | "
              f"support {sup:.1%} refute {ref:.1%} silent {1-sup-ref:.1%}", flush=True)
    # Syndication check — the reason this arm exists as a comparison, not a swap.
    self_dom = sum(1 for r in recs for d in (r.get("results") or [])
                   if d.get("domain") == r.get("domain"))
    print(f"  origin-domain docs that survived exclusion: {self_dom}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="24 claims, 12 per tier")
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
