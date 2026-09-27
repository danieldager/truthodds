"""Dev-50 traced verify run — 50 checkworthy posts sampled from dev500_claims.parquet,
run through the full post-verify loop (pipeline/verify_tweet_claims.py) with FULL step tracing:
every LLM call (system + user prompt, raw JSON response, latency), every search (query,
ranked hits), every scrape. The trace feeds the annotatable review HTML
(scripts/build_dev50_review.py -> reports/).

Tracing is done with thin recording proxies around the pooled clients, so the pipeline
code runs byte-identical to production.

Usage (from src/):
    uv run python scripts/run_dev50_verify_review.py --smoke 2   # trace-format check
    uv run python scripts/run_dev50_verify_review.py             # all 50 (resumable)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import pandas as pd

from pipeline.harness import run_posts
from pipeline.pools import make_pools
from pipeline.verify_tweet_claims import CFG, LOOP_VERSION, verify_post

PARQUET = Path(__file__).resolve().parent.parent / "eval" / "data" / "survey_claims" / "dev500_claims.parquet"
V1_SAMPLE = PARQUET.parent / "dev50_verify_trace" / "sample_posts.json"  # the loop-v1 main set
SEED = 42
N_SAMPLE = 50
LOWNG_MAX = 40  # "very low" NewsGuard cutoff for the adversarial set


# =============================================================================
# Tracing proxies — same call surface as the pools, appending to a per-post sink
# =============================================================================

class _TraceLLM:
    def __init__(self, inner, sink):
        self._inner, self._sink = inner, sink

    async def chat_json(self, messages, *, max_tokens, temperature=0.0, label="llm"):
        t0 = time.monotonic()
        obj = await self._inner.chat_json(messages, max_tokens=max_tokens,
                                          temperature=temperature, label=label)
        self._sink.append({"kind": "llm", "label": label,
                           "secs": round(time.monotonic() - t0, 1),
                           "system": messages[0]["content"],
                           "user": messages[-1]["content"],
                           "response": obj})
        return obj


class _TraceSearch:
    def __init__(self, inner, sink, kind):
        self._inner, self._sink, self._kind = inner, sink, kind

    async def search_(self, query, top_k, **kw):
        t0 = time.monotonic()
        hits = await self._inner.search_(query, top_k, **kw)
        self._sink.append({"kind": self._kind, "query": query, "top_k": top_k,
                           "exclude_domains": kw.get("exclude_domains"),
                           "secs": round(time.monotonic() - t0, 1),
                           "hits": [{"url": h.get("url"), "snippet": h.get("snippet"),
                                     "date": h.get("date"),
                                     "content_chars": len(h.get("content") or "")}
                                    for h in hits]})
        return hits


class _TraceScrape:
    def __init__(self, inner, sink):
        self._inner, self._sink = inner, sink

    async def scrape(self, url):
        t0 = time.monotonic()
        txt = await self._inner.scrape(url)
        self._sink.append({"kind": "scrape", "url": url,
                           "secs": round(time.monotonic() - t0, 1),
                           "chars": len(txt) if txt else 0, "ok": bool(txt)})
        return txt


class _TracedPools:
    def __init__(self, pools, sink):
        self.cfg = pools.cfg
        self.llm = _TraceLLM(pools.llm, sink)
        self.serper = _TraceSearch(pools.serper, sink, "serper")
        self.exa = _TraceSearch(pools.exa, sink, "exa")
        self.scrape = _TraceScrape(pools.scrape, sink)


async def traced_verify(post: dict, pools) -> dict:
    trace: list[dict] = []
    result = await verify_post(post, _TracedPools(pools, trace), CFG)
    result["trace"] = trace
    for k in ("ng_score", "lean", "url"):  # review-header context
        result[k] = post.get(k)
    return result


# =============================================================================
# Sampling — posts with >=1 checkworthy claim, seeded
# =============================================================================

def build_posts() -> list[dict]:
    df = pd.read_parquet(PARQUET)
    posts = []
    for pid, g in df.groupby("post_id", sort=True):
        if not g["checkworthy"].any():
            continue
        g = g.assign(_ord=g["claim_id"].str.split(":").str[-1].astype(int)).sort_values("_ord")
        r = g.iloc[0]
        posts.append({
            "post_id": str(pid), "url": r["url"], "handle": r["handle"],
            "domain": r["domain"], "date": str(r["created_at"])[:10],
            "ng_score": None if pd.isna(r["ng_score"]) else float(r["ng_score"]),
            "lean": r["lean"], "text": r["post_text"],
            "claims": [{"c": row.claim, "t": row.type, "cw": bool(row.checkworthy)}
                       for row in g.itertuples()],
        })
    return posts


def sample_posts(which: str, out_dir: Path) -> list[dict]:
    if which == "main":
        # the loop-v1 set, byte-identical, so cross-version diffs compare the same posts
        sample = json.loads(V1_SAMPLE.read_text())
    else:  # lowng: NG < LOWNG_MAX, excluding posts already in the main set
        posts = build_posts()
        used = {p["post_id"] for p in json.loads(V1_SAMPLE.read_text())}
        pool = [p for p in posts
                if p["ng_score"] is not None and p["ng_score"] < LOWNG_MAX
                and p["post_id"] not in used]
        print(f"low-NG pool: {len(pool)} posts (NG < {LOWNG_MAX}, main-set excluded)")
        sample = random.Random(SEED).sample(pool, min(N_SAMPLE, len(pool)))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_posts.json").write_text(json.dumps(sample, ensure_ascii=False, indent=1))
    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["main", "lowng"], default="main")
    ap.add_argument("--smoke", type=int, default=0, help="run only the first N sampled posts")
    args = ap.parse_args()

    out_dir = PARQUET.parent / "runs" / f"dev50_{args.set}_{LOOP_VERSION}"
    sample = sample_posts(args.set, out_dir)
    handles = sorted({p["handle"] for p in sample})
    print(f"{args.set} set, loop {LOOP_VERSION}: {len(sample)} posts across "
          f"{len(handles)} handles: {', '.join(handles)}")
    if args.smoke:
        sample = sample[:args.smoke]

    pools = make_pools()

    async def go():
        try:
            return await run_posts(sample, traced_verify, pools, out_dir,
                                   name=f"dev50-{args.set}-{LOOP_VERSION}")
        finally:
            await pools.close()

    stats = asyncio.run(go())
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
