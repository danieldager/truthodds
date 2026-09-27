"""Arm FC — a second, fact-check-targeted query.

The silent-read audit (clog/270826, 20:40) put the bottleneck in RETRIEVAL, not
the reader: missed and caught CN-false claims get the same number of documents
(9.8 vs 10.0) and completely different contents, and a fact-check domain appears
in 4.2% of missed dossiers against 14.4% of caught ones. The production query
prompt deliberately steers AWAY from that evidence ("not commentary about
whether the claim is true"), so this arm inverts exactly that one clause and
measures what the second query buys.

Union semantics: in production this query would be ADDED to the existing one, so
the arm is scored on baseline docs + FC docs, deduped by URL.

The true side is the gate. The parked query-gen thread died because new evidence
fired bad reader classes on true claims and recall@2%FPR fell even as AUC rose,
so the timeline control has equal weight here: any gain on CN-false has to
survive the count of timeline claims newly dragged into the flag zone.

    uv run python -m eval.scripts.build_eval.fc_query_test --sample
    uv run python -m eval.scripts.build_eval.fc_query_test --run --smoke 15
    uv run python -m eval.scripts.build_eval.fc_query_test --run --workers 12
    uv run python -m eval.scripts.build_eval.fc_query_test --report
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from pipeline.config import FACT_CHECK_DOMAINS  # noqa: E402
from pipeline.credibility import TRUSTED_FACTCHECKERS  # noqa: E402
from pipeline.search import SearchError, scrape, search  # noqa: E402
from pipeline.verify_tweet_claims import _domain_of, _voice_key  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
    _is_leak, _serper_gate, aggregate_reads, llm, read_doc, select_regions)
from eval.scripts.build_eval.refute_verify import w7  # noqa: E402

OUT_DIR = SRC / "eval/data/urn_runs/synth_expt"
SAMPLE = OUT_DIR / "fc_query_sample.jsonl"
RESULTS = OUT_DIR / "fc_query_results.jsonl"
C2 = SRC / "eval/data/urn_runs/c2_false"
TL = SRC / "eval/data/urn_runs/true_timeline/scores.jsonl"
SEED = 20260827
N = {"cn_false_missed": 150, "tl_unflagged": 100}
FC_DOMS = set(FACT_CHECK_DOMAINS) | {d for d in TRUSTED_FACTCHECKERS if "/" not in d}

# The production prompt's last constraint, inverted. Everything else about the
# task (one query, load-bearing proposition, <=10 words) is held fixed so the
# only moving part is what kind of source the query aims at.
FC_QUERY_SYS = (
    "You write ONE web search query whose purpose is to find out whether a factual claim "
    "has already been checked, debunked, corrected, or disputed by others. Identify the "
    "single LOAD-BEARING proposition — the thing that makes the claim true or false — and "
    "aim the query at published assessments of THAT proposition: fact-checks, corrections, "
    "retractions, denials by the people involved, or reporting that settles it. Name the "
    "entities and the specific assertion so the query cannot drift to the general topic. "
    "At most 10 words. No quotes unless a distinctive phrase is essential. "
    "Respond JSON only: {\"query\": \"...\"}")


def band_of(s: float) -> str:
    return "flag" if s <= -4.05 else ("cliff" if s <= -2.0 else "pass")


def build_sample() -> None:
    w = w7()
    rng = random.Random(SEED)
    pools = collections.defaultdict(list)

    def add(rec, pop):
        docs = [d for d in rec.get("results") or []
                if d.get("read") and d["read"].get("direction")]
        if not docs:
            return
        s = sum(w.get(d["read"]["direction"], 0.0) for d in docs)
        if s <= -4.05:                      # already flagged; nothing to gain
            return
        pools[pop].append({
            "population": pop, "review_url": rec["review_url"], "s7": round(s, 4),
            "claim": rec.get("claim_resolved") or rec.get("claim_text") or "",
            "ceiling": rec.get("ceiling"), "claim_date": rec.get("claim_date_shown"),
            "publisher_site": rec.get("publisher_site") or "x.com",
            "base_flags": [d["read"]["direction"] for d in docs],
            "base_urls": [d.get("url") for d in docs],
            "base_fc": sum(1 for d in docs if d.get("fc_domain"))})

    excl = {e["claim"][:80] for e in json.loads((C2 / "fit_exclusions.json").read_text())}
    for p in ("scores.jsonl", "scores_ext.jsonl"):
        for line in (C2 / p).open():
            r = json.loads(line)
            if (r.get("claim_text") or "")[:80] in excl:
                continue
            add(r, "cn_false_missed")
    for line in TL.open():
        add(json.loads(line), "tl_unflagged")

    rows = []
    for pop, pool in pools.items():
        rng.shuffle(pool)
        rows += pool[:N[pop]]
    with SAMPLE.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for pop in N:
        sub = [r for r in rows if r["population"] == pop]
        print(f"  {pop:18s} {len(sub):4d} claims | mean s7 {sum(r['s7'] for r in sub)/len(sub):+.2f} "
              f"| baseline fc-domain docs {sum(r['base_fc'] for r in sub)/len(sub):.2f}/claim")


def fc_query(row: dict) -> str:
    qblock = [f"CLAIM: {row['claim']}"]
    if row.get("claim_date"):
        qblock.append(f"CLAIM DATE: {row['claim_date']}")
    for _ in range(3):
        try:
            q, _, _, _ = llm([{"role": "system", "content": FC_QUERY_SYS},
                              {"role": "user", "content": "\n".join(qblock)}], max_tokens=80)
            out = (q.get("query") or "").strip()[:300]
            if out:
                return out
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise RuntimeError("fc query-gen failed")


def run_one(row: dict) -> dict:
    q = fc_query(row)
    ceil = row.get("ceiling") or None
    stats: dict = {}
    hits = None
    for attempt in range(2):
        _serper_gate()
        try:
            hits = search(q, 10, date_ceiling=ceil, exclude_domains=[row["publisher_site"]],
                          min_results=0, stats=stats, provider="serper")
            break
        except SearchError:
            if attempt == 0:
                time.sleep(60)
            else:
                raise
    seen = set(row["base_urls"])
    claim_block = (f"CLAIM: {row['claim']}\n"
                   f"(claimed on {row.get('claim_date') or 'unknown date'}; judge the "
                   f"document's bearing on this exact proposition)")
    out_docs, cost = [], 0.0
    for h in (hits or [])[:10]:
        url = h.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        dom = _domain_of(url)
        fc = any(dom == f or dom.endswith("." + f) for f in FC_DOMS)
        entry = {"url": url, "domain": dom, "voice": _voice_key(dom), "fc_domain": fc,
                 "date": h.get("date"), "leak_flag": _is_leak(h.get("date"), ceil)}
        if fc and not (h.get("date") or "").strip():
            entry["direction"] = None          # undated FC page: never read (leak risk)
            entry["read_status"] = "fc-undated"
            out_docs.append(entry)
            continue
        text = h.get("content") or scrape(url)
        if not text or len(text) < 200:
            text = h.get("snippet") or ""
        regions, _ = select_regions(text, row["claim"], q)
        if not regions:
            entry["direction"] = None
            entry["read_status"] = "empty-doc"
            out_docs.append(entry)
            continue
        rr = []
        for ids, sel in regions:
            got = None
            for _ in range(2):
                try:
                    got, c, _, _ = read_doc(claim_block, ids, sel, row["review_url"])
                    cost += c
                except Exception:  # noqa: BLE001
                    time.sleep(2)
                    continue
                if got:
                    break
            rr.append(got or {"direction": "I", "evidence": [], "reason": "",
                              "qc_flag": "read-failed"})
        agg = aggregate_reads(rr)
        entry["direction"] = agg.get("direction")
        entry["read_status"] = "ok"
        out_docs.append(entry)
    return {"review_url": row["review_url"], "population": row["population"],
            "s7_base": row["s7"], "base_flags": row["base_flags"],
            "fc_query": q, "n_new": len(out_docs), "docs": out_docs, "cost": cost}


def run(workers: int, smoke: int) -> None:
    rows = [json.loads(l) for l in SAMPLE.open()]
    done = set()
    if RESULTS.exists():
        done = {json.loads(l)["review_url"] for l in RESULTS.open()}
    todo = [r for r in rows if r["review_url"] not in done]
    if smoke:
        rng = random.Random(SEED)
        rng.shuffle(todo)
        todo = todo[:smoke]
    print(f"{len(todo)} claims, workers={workers}", flush=True)
    lock = threading.Lock(); n = [0]; cost = [0.0]; t0 = time.time()
    with RESULTS.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, r) for r in todo]
        for fut in as_completed(futs):
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {e}", flush=True); continue
            with lock:
                f.write(json.dumps(out) + "\n"); f.flush()
                n[0] += 1; cost[0] += out["cost"]
                if n[0] % 20 == 0 or n[0] == len(todo):
                    el = time.time() - t0
                    print(f"  {n[0]}/{len(todo)}  ${cost[0]:.2f}  {n[0]/el:.2f}/s  "
                          f"ETA {(len(todo)-n[0])/max(n[0]/el,.001):.0f}s", flush=True)
    print(f"done: {n[0]} claims, LLM ${cost[0]:.2f}, serper {n[0]} req", flush=True)


def report() -> None:
    w = w7()
    res = [json.loads(l) for l in RESULTS.open()]
    per = collections.defaultdict(list)
    for r in res:
        per[r["population"]].append(r)
    for pop, rows in sorted(per.items()):
        n = len(rows)
        new_docs = sum(r["n_new"] for r in rows)
        fc_docs = sum(1 for r in rows for d in r["docs"] if d["fc_domain"])
        undated = sum(1 for r in rows for d in r["docs"] if d.get("read_status") == "fc-undated")
        dirs = collections.Counter(d["direction"] for r in rows for d in r["docs"]
                                   if d.get("direction"))
        gained = sum(1 for r in rows for d in r["docs"] if d.get("direction") in ("1", "2"))
        s_new = [(r["s7_base"] + sum(w.get(d["direction"], 0.0) for d in r["docs"]
                                     if d.get("direction"))) for r in rows]
        cross = sum(1 for r, s in zip(rows, s_new) if r["s7_base"] > -4.05 and s <= -4.05)
        print(f"\n=== {pop} (n={n}) ===")
        print(f"  new docs {new_docs} ({new_docs/n:.1f}/claim) | fact-check domains {fc_docs} "
              f"({fc_docs/n:.2f}/claim) | undated FC skipped {undated}")
        print(f"  new reads: " + " ".join(f"{k}:{dirs[k]}" for k in ("5","4","3","2","1","X","I") if dirs[k]))
        print(f"  refute reads gained {gained} ({gained/n:.2f}/claim)")
        print(f"  mean s7 {sum(r['s7_base'] for r in rows)/n:+.2f} -> {sum(s_new)/n:+.2f}")
        print(f"  claims newly crossing into the flag zone: {cross}/{n} ({cross/n:.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.sample:
        build_sample()
    if a.run:
        run(a.workers, a.smoke)
    if a.report:
        report()
