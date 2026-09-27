"""wire-true: the E3 instrument over the KEY claims of the reputable-outlet TRUE posts.

Same population as the 2026-08-27 old-chain run (`urn_runs/true_outlet`, the 768 posts
behind true_urn_assertions.parquet), so the two chains compare post for post; same
instrument as `keyclaim_urn_run.py` (query-v3 on the production model, Serper top-10,
ceiling = post date, reader v7qa on gpt-oss-120b @ DeepInfra, every model call through reader_lab.GATE). Only the claims change:
extract-key-v2 instead of the extract -> normalize -> checkworthy chain.

ORIGIN EXCLUSION. `run_claim` builds `xd = [publisher_site] + exclude_extra`, and
search.serper_payload sends that set as server-side `-site:` operators ONLY while it
holds at most `_SERVER_XD_MAX` = 3 domains, otherwise the whole set falls back to a
client-side drop with no page-2 backfill — i.e. lost result slots for exactly the
outlets that have aliases. So the policy here is publisher_site + `outlet_aliases.
server_side()` (<= 2), which fills the budget exactly, and NOT x.com/twitter.com:
those are in SCRAPE_BLOCKLIST, which search.py already sends server-side (priority
slots 3 and 4 of 16) AND drops client-side in serper_finalize, so naming them again
would only push the outlet's own domain out of the server-side budget.

Syndication is NOT handled here — it is post hoc in `syndication.py` over the stored
documents, so it stays free and re-runnable.

    uv run python -m eval.scripts.build_eval.wire_urn_run --smoke --budget 0.5
    uv run python -m eval.scripts.build_eval.wire_urn_run --budget 6
Output: eval/data/urn_runs/wire_true/{smoke,results-00}.jsonl (resumable by review_url)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import polars as pl  # noqa: E402

from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval import evidence_urn_run as eur  # noqa: E402
from eval.scripts.build_eval import outlet_aliases, reader_lab  # noqa: E402
from eval.scripts.build_eval.keyclaim_urn_run import QUERY_MODELS, READ_MODEL, READ_PROMPT, make_query_llm, read_doc_v7qa  # noqa: E402
from eval.scripts.build_eval.reader_lab_prompts import PROMPTS  # noqa: E402
from pipeline.search import newsguard_score_map  # noqa: E402

OUTDIR = Path("eval/data/urn_runs/wire_true")
CLAIMS = OUTDIR / "keyclaims_v2.json"
TIERS = Path("eval/data/tweet_corpus/true_urn_assertions.parquet")


def adapt(c, tiers):
    return {**c, "review_url": c["claim_id"], "claim_text": c["claim"], "publisher_site": c["domain"],
            "claim_date": (c.get("created_at") or "")[:10], "claim_type": c["type"], "x_context": c.get("post_text") or "",
            "context_ok": False, "resolution_status": "native", "claim_resolved": None,
            "veracity": None, "rating_subtype": None, "review_date": None, "x_date": None, "yr": None, "topic": None,
            "exclude_extra": outlet_aliases.server_side(c["domain"]),
            "ng_tier": tiers.get(str(c["post_id"])), "checkworthy": True, "verify_eligible": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default=str(CLAIMS))
    ap.add_argument("--tiers", default=str(TIERS))
    ap.add_argument("--smoke", action="store_true", help="25 random posts")
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=96, help="claim workers; reads are sequential per claim, so this bounds concurrency")
    ap.add_argument("--provider", default="deepinfra", choices=sorted(reader_lab.PROVIDERS))
    ap.add_argument("--query-model", default="flash", choices=sorted(QUERY_MODELS))
    ap.add_argument("--gate-start", type=int, default=48)
    ap.add_argument("--gate-cap", type=int, default=96)
    ap.add_argument("--seed", type=int, default=707)
    args = ap.parse_args()
    reader_lab.PROVIDER[0] = args.provider
    reader_lab.GATE.limit = reader_lab.GATE.peak = args.gate_start
    reader_lab.GATE.cap = args.gate_cap
    eur.read_doc = read_doc_v7qa   # run_claim looks read_doc up in its module at call time
    eur.llm = make_query_llm(QUERY_MODELS[args.query_model])   # query gen through the gate too
    tiers = dict(pl.read_parquet(args.tiers).select(pl.col("post_id").cast(str), "ng_tier").unique().iter_rows())
    claims = json.load(open(args.claims))["claims"]
    rows = [adapt(c, tiers) for c in claims]
    by_post = {}
    for r in rows:
        by_post.setdefault(r["post_id"], []).append(r)
    order = sorted(by_post)
    random.Random(args.seed).shuffle(order)
    if args.smoke:
        order = order[:25]
    rows = [r for pid in order for r in by_post[pid]]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / ("smoke.jsonl" if args.smoke else "results-00.jsonl")
    seen = set()
    if out.exists():
        for l in open(out):
            try:
                seen.add(json.loads(l)["review_url"])
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in rows if r["review_url"] not in seen]
    (OUTDIR / "manifest.json").write_text(json.dumps({
        "claims_file": args.claims, "n_claims": len(rows), "n_posts": len(order), "seed": args.seed,
        "exclusion": {"publisher_site": "claim outlet domain", "exclude_extra": "outlet_aliases.server_side(domain)",
                      "note": "<= 3 domains so serper keeps them server-side; x.com/twitter.com already in SCRAPE_BLOCKLIST"},
        "prompts": {"query": eur.QUERY_PROMPT_V, "query_hash": prompt_hash(eur.QUERY_SYS),
                    "read": READ_PROMPT, "read_hash": prompt_hash(PROMPTS[READ_PROMPT]), "read_model": READ_MODEL,
                    "read_provider": args.provider, "query_model": args.query_model},
        "gate": {"start": args.gate_start, "cap": args.gate_cap}, "workers": args.workers, "context_ok": False}, indent=1))
    print(f"{len(todo)} claims ({len(order)} posts) to run -> {out.name} | budget ${args.budget} | {args.workers} workers | "
          f"query {eur.QUERY_PROMPT_V} on {eur.VERIFICATION_MODEL} | read {READ_PROMPT} on {READ_MODEL} @ {args.provider}", flush=True)
    ng_scores = newsguard_score_map()
    lock = threading.Lock()
    budget = {"spent": 0.0, "done": 0, "cached_tok": 0, "prompt_tok": 0, "stop": False}
    t0 = time.time()
    fh = open(out, "a")

    def work(row):
        if budget["stop"]:
            return
        try:
            rec = eur.run_claim(row, ng_scores, budget, {})
        except Exception as e:  # noqa: BLE001
            with lock:
                budget["done"] += 1
                print(f"  [claim-failed] {row['review_url']}: {type(e).__name__}: {e}", flush=True)
            return
        for k in ("post_id", "ng_score", "ng_tier", "handle", "domain", "why"):
            rec[k] = row.get(k)
        with lock:
            budget["spent"] += rec["cost"]
            budget["done"] += 1
            fh.write(json.dumps(rec) + "\n")
            n = budget["done"]
            if n % 20 == 0 or n == len(todo):
                fh.flush()
                el = (time.time() - t0) / 60
                rate = n / max(el, .01)
                print(f"  {n}/{len(todo)} | spent ${budget['spent']:.3f} | projected ${budget['spent'] / n * len(todo):.2f} | "
                      f"{el:.1f}m | {rate:.1f} claims/min | ETA {(len(todo) - n) / max(rate, .01):.0f}m", flush=True)
            if budget["spent"] >= args.budget:
                budget["stop"] = True
                print("  [budget cap reached, stopping]", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"done: {budget['done']} claims, ${budget['spent']:.3f}, {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
