"""Query lab (branch reader-iteration, 2026-09-10): generate the production query (v3,
QUERY_SYS from evidence_urn_run) and a candidate (v4) for a claims file, side by side,
and optionally smoke both through Serper and judge how many of the top-10 snippets
are about the claim's own event.

    uv run python -m eval.scripts.build_eval.query_lab --claims eval/data/reader_lab/extract/keyclaims_v2.json \
        --out eval/data/reader_lab/extract/queries_v2.json [--serper 30] [--provider groq]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval import evidence_urn_run as eur  # noqa: E402
from eval.scripts.build_eval.reader_lab import PROVIDER, PROVIDERS, _endpoint, _parse_json  # noqa: E402

QUERY_V4 = """\
You write ONE web search query to find independent reporting or primary evidence on a
factual claim. You get the claim, the post it came from, and the post date.

Build the query from three parts, in this order, at most 12 words, no quotes, no
operators, no question words: (1) the named actor or subject, spelled as news reports
spell it; (2) the event or action in the words a news headline would use; (3) the
decisive detail that makes the claim true or false (the number, the outcome, the
thing said, the place), because a document that lacks that detail cannot settle the
claim. Use the post only to pick names and dates, never to add its framing. If the
claim is about a statement, query the statement's content and who said it, not the
word "said". Add the month and year only when the claim's event would otherwise be
ambiguous across time.

Respond JSON only: {"query": "..."}
"""
PROMPTS = {"v3": eur.QUERY_SYS, "v4": QUERY_V4}

JUDGE_SYS = """\
You judge a web search result list for a factual claim. For each result you get a
title and snippet. Count the results that are about the claim's own event, statement
or figure (the same actor and the same specific occurrence), whether or not they
agree with the claim. Results about the same topic but a different occurrence, a
different time, or only the general subject do not count.

Respond JSON only: {"on_claim": [<result numbers>]}
"""


def llm(system, user, model, cache, key, max_tokens=200):
    ck = cache / (hashlib.sha256(f"{prompt_hash(system)}|{model}|{key}".encode()).hexdigest()[:24] + ".json")
    if ck.exists():
        return json.load(open(ck)), 0.0
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens if "gpt-oss" not in model else 4000,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    base, k = _endpoint()
    for attempt in range(5):
        r = eur._SESSION.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {k}"}, json=body, timeout=120)
        if r.status_code != 429:
            break
        time.sleep(5 * 2 ** attempt)
    r.raise_for_status()
    j = r.json()
    obj = _parse_json(j["choices"][0]["message"].get("content"))
    u = j.get("usage") or {}
    cost = u.get("estimated_cost")
    if cost is None:
        cost = (u.get("prompt_tokens") or 0) * 0.15e-6 + (u.get("completion_tokens") or 0) * 0.75e-6
    json.dump(obj, open(ck, "w"))
    return obj, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--provider", default="groq", choices=sorted(PROVIDERS))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--serper", type=int, default=0, help="smoke N claims through Serper under both queries and judge on-claim hits")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompts", default="v3,v4", help="comma list of prompt versions to generate")
    ap.add_argument("--judge-model", default=None, help="on-claim judge (default: --model); fix it across model comparisons")
    a = ap.parse_args()
    judge_model = a.judge_model or a.model
    prompts = {v: p for v, p in PROMPTS.items() if v in a.prompts.split(",")}
    PROVIDER[0] = a.provider
    claims = json.load(open(a.claims))["claims"]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(str(out) + ".cache"); cache.mkdir(exist_ok=True)
    lock = threading.Lock(); spent = [0.0]; res = {}

    def work(c):
        r = {}
        for v, sysp in prompts.items():
            if v == "v3":
                user = "\n".join([f"CLAIM: {c['claim']}", f"CONTEXT (circulation of the claim, from a neutral description): {(c.get('post_text') or '').strip()}",
                                  f"CLAIM DATE: {c.get('created_at') or ''}"])
            else:
                user = f"CLAIM: {c['claim']}\nPOST: {(c.get('post_text') or '').strip()}\nPOST DATE: {c.get('created_at') or 'unknown'}"
            cost = 0.0
            for attempt in range(3):
                try:
                    obj, cost = llm(sysp, user, a.model, cache, f"{v}|{c['claim_id']}"); break
                except Exception:  # noqa: BLE001
                    obj = {}; time.sleep(3)
            r[v] = (obj.get("query") or "").strip()[:300]
            with lock:
                spent[0] += cost
        with lock:
            res[c["claim_id"]] = r

    print(f"claims {len(claims)}  prompts v3 ({prompt_hash(eur.QUERY_SYS)}) v4 ({prompt_hash(QUERY_V4)})  model {a.model} @ {a.provider}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, c) for c in claims]):
            f.result()
    print(f"queries done  ${spent[0]:.3f}", flush=True)

    smoke = {}
    if a.serper:
        from pipeline.search import search
        random.seed(a.seed)
        sample = random.sample(claims, min(a.serper, len(claims)))
        tally = {v: [0, 0] for v in prompts}
        for c in sample:
            smoke[c["claim_id"]] = {}
            for v in prompts:
                q = res[c["claim_id"]][v]
                if not q:
                    continue
                ck = cache / ("serp_" + hashlib.sha256(q.encode()).hexdigest()[:24] + ".json")
                if ck.exists():
                    hits = json.load(open(ck))
                else:
                    eur._serper_gate()
                    try:
                        hits = search(q, 10, date_ceiling=None, exclude_domains=[d for d in [c.get("domain")] if d], min_results=0, stats={}, provider="serper")
                    except Exception as e:  # noqa: BLE001
                        print("  search failed", repr(e)[:80]); hits = []
                    hits = [{k: h.get(k) for k in ("url", "domain", "title", "snippet")} if isinstance(h, dict) else {"url": str(h)} for h in (hits or [])]
                    json.dump(hits, open(ck, "w"))
                listing = "\n".join(f"[{i}] {h.get('title') or ''} | {h.get('snippet') or ''}" for i, h in enumerate(hits, 1))
                jobj, cost = llm(JUDGE_SYS, f"CLAIM: {c['claim']}\n\nRESULTS:\n{listing}", judge_model, cache, f"judge|{v}|{c['claim_id']}|{q}")
                spent[0] += cost
                on = [i for i in (jobj.get("on_claim") or []) if isinstance(i, int) and 1 <= i <= len(hits)]
                smoke[c["claim_id"]][v] = {"query": q, "n_hits": len(hits), "on_claim": on, "domains": [h.get("domain") for h in hits]}
                tally[v][0] += len(on); tally[v][1] += len(hits)
        for v, (on, n) in tally.items():
            print(f"  serper smoke {v}: on-claim hits {on}/{n} = {on / max(1, n):.2f}  (claims {len(sample)})")

    json.dump({"prompts": {v: prompt_hash(p) for v, p in prompts.items()}, "model": a.model, "judge_model": judge_model, "queries": res, "smoke": smoke},
              open(out, "w"), indent=0, ensure_ascii=False)
    print(f"spent ${spent[0]:.3f}  wrote {out}")
    random.seed(a.seed + 1)
    for c in random.sample(claims, 10):
        print(f"\nCLAIM: {c['claim'][:140]}" + "".join(f"\n  {v}: {res[c['claim_id']][v]}" for v in prompts))


if __name__ == "__main__":
    main()
