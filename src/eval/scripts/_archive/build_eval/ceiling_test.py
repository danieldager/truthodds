"""Does the date ceiling cause the misses? Re-run the SAME query, later ceiling.

Search sets Google's cd_max to the claim date, so nothing published after the
claim is visible. A false claim is normally debunked AFTER it is made, so the
ceiling may be excluding the very evidence we want. Catch rate is flat across
claim years (28-33%, clog/270826), which is what you would expect if the binding
constraint travels with each claim rather than being about calendar recency.

This isolates the ceiling: same claims, same production query (reused verbatim
from the original run, so query-gen is not a moving part), only cd_max changes.
Timeline claims run as the control, since giving them hindsight must not start
manufacturing refutations for claims that are mostly true.

    uv run python -m eval.scripts.build_eval.ceiling_test --run --shift 90 --smoke 15
    uv run python -m eval.scripts.build_eval.ceiling_test --run --shift 90 --workers 24
    uv run python -m eval.scripts.build_eval.ceiling_test --report --shift 90
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
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
from pipeline.verify_tweet_claims import _domain_of  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
    _serper_gate, aggregate_reads, read_doc, select_regions)
from eval.scripts.build_eval.fc_query_test import SAMPLE as FC_SAMPLE  # noqa: E402
from eval.scripts.build_eval.refute_verify import w7  # noqa: E402

OUT_DIR = SRC / "eval/data/urn_runs/synth_expt"
C2 = SRC / "eval/data/urn_runs/c2_false"
TL = SRC / "eval/data/urn_runs/true_timeline/scores.jsonl"
FC_DOMS = set(FACT_CHECK_DOMAINS) | {d for d in TRUSTED_FACTCHECKERS if "/" not in d}


def orig_queries() -> dict:
    q = {}
    for p in (C2 / "scores.jsonl", C2 / "scores_ext.jsonl", TL):
        for line in p.open():
            r = json.loads(line)
            if r.get("query"):
                q[r["review_url"]] = r["query"]
    return q


def shift_ceiling(ceil: str, days: int) -> str:
    d = dt.datetime.strptime(ceil[:10], "%Y-%m-%d") + dt.timedelta(days=days)
    today = dt.datetime.strptime("2026-08-27", "%Y-%m-%d")
    return min(d, today).strftime("%Y-%m-%d")


def run_one(row: dict, query: str, days: int) -> dict:
    ceil = shift_ceiling(row["ceiling"], days)
    stats: dict = {}
    hits = None
    for attempt in range(2):
        _serper_gate()
        try:
            hits = search(query, 10, date_ceiling=ceil,
                          exclude_domains=[row["publisher_site"]],
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
    docs, cost = [], 0.0
    for h in (hits or [])[:10]:
        url = h.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        dom = _domain_of(url)
        fc = any(dom == f or dom.endswith("." + f) for f in FC_DOMS)
        entry = {"url": url, "domain": dom, "fc_domain": fc, "date": h.get("date")}
        if fc and not (h.get("date") or "").strip():
            entry["direction"] = None
            docs.append(entry)
            continue
        text = h.get("content") or scrape(url)
        if not text or len(text) < 200:
            text = h.get("snippet") or ""
        regions, _ = select_regions(text, row["claim"], query)
        if not regions:
            entry["direction"] = None
            docs.append(entry)
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
        entry["direction"] = aggregate_reads(rr).get("direction")
        docs.append(entry)
    return {"review_url": row["review_url"], "population": row["population"],
            "s7_base": row["s7"], "ceiling_was": row["ceiling"], "ceiling_now": ceil,
            "query": query, "n_new": len(docs), "docs": docs, "cost": cost}


def run(days: int, workers: int, smoke: int) -> None:
    out_path = OUT_DIR / f"ceiling_shift_{days}.jsonl"
    rows = [json.loads(l) for l in FC_SAMPLE.open()]
    qs = orig_queries()
    rows = [r for r in rows if r["review_url"] in qs and r.get("ceiling")]
    done = set()
    if out_path.exists():
        done = {json.loads(l)["review_url"] for l in out_path.open()}
    todo = [r for r in rows if r["review_url"] not in done]
    if smoke:
        rng = random.Random(20260827); rng.shuffle(todo); todo = todo[:smoke]
    print(f"{len(todo)} claims, ceiling +{days}d, workers={workers}", flush=True)
    lock = threading.Lock(); n = [0]; cost = [0.0]; t0 = time.time()
    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, r, qs[r["review_url"]], days) for r in todo]
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
    print(f"done: {n[0]} claims, ${cost[0]:.2f}", flush=True)


def report(days: int) -> None:
    w = w7()
    path = OUT_DIR / f"ceiling_shift_{days}.jsonl"
    res = [json.loads(l) for l in path.open()]
    per = collections.defaultdict(list)
    for r in res:
        per[r["population"]].append(r)
    for pop, rows in sorted(per.items()):
        n = len(rows)
        fc = sum(1 for r in rows for d in r["docs"] if d["fc_domain"])
        dirs = collections.Counter(d["direction"] for r in rows for d in r["docs"]
                                   if d.get("direction"))
        gained = sum(1 for r in rows for d in r["docs"] if d.get("direction") in ("1", "2"))
        s_new = [r["s7_base"] + sum(w.get(d["direction"], 0.0) for d in r["docs"]
                                    if d.get("direction")) for r in rows]
        cross = sum(1 for r, s in zip(rows, s_new) if r["s7_base"] > -4.05 and s <= -4.05)
        print(f"\n=== {pop} (n={n}), ceiling +{days}d ===")
        print(f"  new docs {sum(r['n_new'] for r in rows)/n:.1f}/claim | "
              f"fact-check domains {fc/n:.2f}/claim")
        print(f"  new reads: " + " ".join(f"{k}:{dirs[k]}" for k in
                                          ("5","4","3","2","1","X","I") if dirs[k]))
        print(f"  refute reads gained {gained} ({gained/n:.2f}/claim)")
        print(f"  mean s7 {sum(r['s7_base'] for r in rows)/n:+.2f} -> {sum(s_new)/n:+.2f}")
        print(f"  claims newly crossing into the flag zone: {cross}/{n} ({cross/n:.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--shift", type=int, default=90)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if a.run:
        run(a.shift, a.workers, a.smoke)
    if a.report:
        report(a.shift)
