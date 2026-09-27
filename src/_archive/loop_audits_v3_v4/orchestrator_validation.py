"""Validation harness for the pool/orchestration layer (pipeline/pools.py + harness.py).

Runs a STUB of the verify_tweet_claims pipeline (design: docs/verify_tweet_claims_loop_design.md) end-to-end
against the REAL external services — real DeepInfra LLM calls, real Serper searches, real
scrapes — but with throwaway mini-prompts, so it measures the concurrency layer, not verdict
quality. EXA IS NEVER CALLED (1k free req/mo; its pool is docs-sized only).

Stub shape per post (mirrors the designed loop so saturation is realistic):
    OPEN (LLM: query + claims) -> serper -> scrape 3 pages (concurrent) ->
    READ per scraped doc (LLM, concurrent) -> STEP (LLM: conclude or one more round)

Stages (house rule: smoke first, then a small ramp, then STOP for sign-off):
    uv run python scripts/orchestrator_validation.py smoke              # 5 posts
    uv run python scripts/orchestrator_validation.py ramp --n 75 --k 40 # measured ramp
Both are resumable: completed posts are skipped on re-run (delete --out to start fresh).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import pandas as pd

from pipeline.harness import run_posts
from pipeline.pools import OrchestrationConfig, Pools, make_pools

TWEETS_PARQUET = Path(__file__).resolve().parent.parent / "eval" / "data" / "survey_claims" / "outlet_tweets.parquet"
DEFAULT_OUT = Path.home() / ".cache" / "factcheck_orch"

# Serper pricing is tier-dependent: $0.30-1.00 per 1k queries (2026-07-10 research).
SERPER_USD_PER_1K = (0.30, 1.00)

_MAX_TOK = {"open": 400, "read": 400, "step": 400}
_DOC_SLICE = 4000       # chars of scraped page fed to READ (stub; the real loop uses ~2,500 tok)
_MAX_ROUNDS = 2         # stub: at most 2 search rounds (design: 2-3 serper + exa escalation)
_SCRAPE_TARGET = 3      # good scrapes wanted per round


# =============================================================================
# Stub pipeline — a plain sequential async function. THE point of the architecture:
# this function knows nothing about concurrency; it just awaits pooled clients.
# =============================================================================

async def stub_pipeline(post: dict, pools: Pools) -> dict:
    text = post["text"]
    exclude = [post["domain"]] if post.get("domain") else []  # never cite the origin outlet

    opened = await pools.llm.chat_json(
        [{"role": "system", "content":
          'Given a social media post, output strict JSON: {"claims": [up to 3 short checkworthy '
          'factual claims], "query": "one keyword-style web search query for the most checkable claim"}'},
         {"role": "user", "content": text[:1500]}],
        max_tokens=_MAX_TOK["open"], label="OPEN")
    claims = [str(c) for c in (opened.get("claims") or [])][:3] or [text[:120]]
    query = str(opened.get("query") or text[:120])

    evidence: list[dict] = []
    seen: set[str] = set()
    rounds = 0
    verdict: dict = {}
    while rounds < _MAX_ROUNDS:
        rounds += 1
        results = await pools.serper.search_(query, 10, exclude_domains=exclude)
        fresh = [r for r in results if r["url"] not in seen]
        seen.update(r["url"] for r in fresh)

        # scrape down the ranked list, a batch at a time, until _SCRAPE_TARGET good pages
        docs: list[dict] = []
        queue = list(fresh)
        while queue and len(docs) < _SCRAPE_TARGET:
            batch = queue[:_SCRAPE_TARGET - len(docs)]
            queue = queue[len(batch):]
            texts = await asyncio.gather(*(pools.scrape.scrape(r["url"]) for r in batch))
            docs += [{"url": r["url"], "text": t} for r, t in zip(batch, texts) if t]

        async def read_one(doc: dict) -> dict:
            out = await pools.llm.chat_json(
                [{"role": "system", "content":
                  'Given CLAIMS and a PAGE, output strict JSON: {"stances": [{"claim": <index>, '
                  '"stance": "supports"|"refutes"|"neutral"}] } for claims the page speaks to.'},
                 {"role": "user", "content":
                  "CLAIMS:\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(claims)) +
                  f"\n\nPAGE ({doc['url']}):\n" + doc["text"][:_DOC_SLICE]}],
                max_tokens=_MAX_TOK["read"], label="READ")
            return {"url": doc["url"], "stances": out.get("stances") or []}

        evidence += list(await asyncio.gather(*(read_one(d) for d in docs)))

        verdict = await pools.llm.chat_json(
            [{"role": "system", "content":
              'Given a POST, its CLAIMS, and a stance-tagged EVIDENCE table, output strict JSON: '
              '{"resolved": true|false, "next_query": "..."|null, "veracity": 1-5}'},
             {"role": "user", "content":
              f"POST:\n{text[:1000]}\n\nCLAIMS:\n" +
              "\n".join(f"{i}. {c}" for i, c in enumerate(claims)) +
              "\n\nEVIDENCE:\n" + json.dumps(evidence)[:6000]}],
            max_tokens=_MAX_TOK["step"], label="STEP")
        if verdict.get("resolved") or not verdict.get("next_query"):
            break
        query = str(verdict["next_query"])

    return {"claims": claims, "n_rounds": rounds, "n_evidence": len(evidence),
            "veracity": verdict.get("veracity"), "resolved": bool(verdict.get("resolved"))}


# =============================================================================
# Report
# =============================================================================

def _deepinfra_pricing(model: str) -> dict | None:
    """Free metadata GET (no credits) for $/M token prices; None if unavailable."""
    import requests
    try:
        r = requests.get(f"https://api.deepinfra.com/models/{model}", timeout=10)
        p = r.json().get("pricing", {})
        if p.get("cents_per_input_token") is not None:
            # cents/token -> USD per M tokens: * 1e6 tokens / 100 cents
            return {"in_per_m": p["cents_per_input_token"] * 10_000,
                    "out_per_m": p["cents_per_output_token"] * 10_000}
    except Exception:
        pass
    return None


def report(pools: Pools, stats: dict, n_posts: int, wall_s: float) -> dict:
    m_llm, m_srp, m_scr = pools.llm.metrics, pools.serper.metrics, pools.scrape.metrics
    per_post_srp = m_srp.ok / max(1, stats["ran"])
    tok_in, tok_out = m_llm.prompt_tokens, m_llm.completion_tokens
    price = _deepinfra_pricing(pools.cfg.llm_model)

    rep = {
        "posts": {"ran": stats["ran"], "errors": stats["errors"], "wall_s": wall_s,
                  "posts_per_min": round(stats["ran"] / wall_s * 60, 2) if wall_s else None},
        "llm": {"calls_ok": m_llm.ok, "p50_s": round(m_llm.pct(0.5), 1),
                "p95_s": round(m_llm.pct(0.95), 1), "peak_in_flight": m_llm.peak_in_flight,
                "retries": {k.value: v for k, v in m_llm.retries.items()},
                "hangs": m_llm.hangs, "fatal": m_llm.fatal,
                "tokens_in": tok_in, "tokens_out": tok_out},
        "serper": {"queries_paid": m_srp.ok, "cache_hits": m_srp.cache_hits,
                   "p50_s": round(m_srp.pct(0.5), 2), "peak_in_flight": m_srp.peak_in_flight,
                   "retries": {k.value: v for k, v in m_srp.retries.items()},
                   "ratelimit_headers": m_srp.last_ratelimit or "none observed"},
        "scrape": {"attempts": m_scr.ok, "misses": m_scr.misses,
                   "p50_s": round(m_scr.pct(0.5), 1), "p95_s": round(m_scr.pct(0.95), 1),
                   "peak_in_flight": m_scr.peak_in_flight, "wall_timeouts": m_scr.hangs},
        "exa": {"calls": pools.exa.metrics.ok, "note": "never called (docs-sized only)"},
    }

    # --- 10k projection ---
    if stats["ran"]:
        scale = 10_000 / stats["ran"]
        proj: dict = {
            "wall_clock_h_at_this_K": round(wall_s * scale / 3600, 1),
            "serper_queries": int(per_post_srp * 10_000),
            "serper_usd": [round(per_post_srp * 10 * lo_hi, 2) for lo_hi in SERPER_USD_PER_1K],
            "llm_tokens_in_M": round(tok_in * scale / 1e6, 1),
            "llm_tokens_out_M": round(tok_out * scale / 1e6, 1),
        }
        if price:
            proj["llm_usd"] = round((tok_in * scale / 1e6) * price["in_per_m"]
                                    + (tok_out * scale / 1e6) * price["out_per_m"], 2)
            proj["llm_pricing_used"] = price
        else:
            proj["llm_usd"] = "pricing endpoint unavailable — tokens reported above"
        rep["projection_10k"] = proj
    return rep


# =============================================================================
# CLI
# =============================================================================

def load_posts(n: int, seed: int = 7) -> list[dict]:
    df = pd.read_parquet(TWEETS_PARQUET, columns=["post_id", "text", "domain", "lang"])
    df = df[df["text"].str.len() > 40]
    sample = df.sample(n=n, random_state=seed)
    return sample.to_dict("records")


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["smoke", "ramp"])
    ap.add_argument("--n", type=int, default=None, help="posts (smoke default 5, ramp 75)")
    ap.add_argument("--k", type=int, default=None, help="posts in flight (smoke 5, ramp 40)")
    ap.add_argument("--llm-pool", type=int, default=None, help="override LLM pool size")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    n = args.n or (5 if args.stage == "smoke" else 75)
    k = args.k or (5 if args.stage == "smoke" else 40)
    cfg = OrchestrationConfig(k_posts=k, progress_every=5.0 if args.stage == "smoke" else 10.0)
    if args.llm_pool:
        cfg.llm_pool_size = args.llm_pool
    out = args.out or (DEFAULT_OUT / args.stage)

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    posts = load_posts(n)
    pools = make_pools(cfg)
    t0 = time.monotonic()
    try:
        stats = await run_posts(posts, stub_pipeline, pools, out, name=args.stage)
    finally:
        await pools.close()
    rep = report(pools, stats, n, time.monotonic() - t0)

    print("\n" + "=" * 78)
    print(json.dumps(rep, indent=2, default=str))
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nreport -> {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
