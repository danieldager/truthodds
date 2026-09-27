"""Arm 6 on the all-irrelevant block: Exa neural retrieval (clog/280826, 12:07).

The block: every retrieved document read I, so zero refuting documents, so under
the "never flag on silence" rule the claim can NEVER be flagged. Five Serper-side
interventions have now failed on it (ceiling +90d, fact-check-targeted query,
uncapped detailed query `spec`, date-token removal `nodate`, UGC unblock). Every
one of them reformulates a KEYWORD query against Google's top-10. This arm changes
the retrieval PARADIGM instead: one natural-language description of the evidence
that would refute the claim, sent to Exa's neural/auto search. Exa has never been
run on this population.

POPULATION. `retrieval_recovery_arms.load_corpora` + `_allirr`, verbatim — the
same block definition the five failed arms were scored on. 100 claims, 50 CN-false
+ 50 fc-gold, seed 20260828. Corpus recorded per claim.

DATE CEILING. NOT applied. The eval's ceiling exists to keep future evidence out
of the urn; here the question is whether Exa can reach the evidence AT ALL, and
the 27/08 arm showed a wider window helps rather than hurts. Every returned
document's publishedDate is recorded and every hit is tagged `future` against the
claim's saved ceiling, so the no-future-evidence guarantee stays CHECKABLE: the
report gives the moved count twice, once raw and once counting only documents
published on or before the ceiling.

READ. `evidence_urn_run.select_regions` / `read_doc` / `aggregate_reads`, the
production read-v5 chain at the production model, with the claim_block byte-copied
from `retrieval_recovery_arms.run_one`. No new rubric.

EXA IS A PROTECTED RESOURCE (1,000 free requests/month, house rule: never
live-test). Hard cap 100 requests for this probe, counted in a file that survives
a resume, taken BEFORE the request goes out, SystemExit on breach. Cache hits are
free and do not count. 4 workers, 0.75s global pacing.

    uv run python -m eval.scripts.build_eval.exa_probe --sample
    uv run python -m eval.scripts.build_eval.exa_probe --run --smoke 3
    uv run python -m eval.scripts.build_eval.exa_probe --run
    uv run python -m eval.scripts.build_eval.exa_probe --report
    uv run python -m eval.scripts.build_eval.exa_probe --show 8

Nothing under eval/data/urn_runs/ is read-modified or written. Output:
eval/data/exa_probe/.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from pipeline import disk_cache  # noqa: E402
from pipeline.config import FACT_CHECK_DOMAINS  # noqa: E402
from pipeline.credibility import TRUSTED_FACTCHECKERS  # noqa: E402
from pipeline.search import SearchError, cache_key, scrape, search  # noqa: E402
from pipeline.verify_tweet_claims import _domain_of  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
    aggregate_reads, llm, read_doc, select_regions)
from eval.scripts.build_eval.refute_verify import w7  # noqa: E402
from eval.scripts.build_eval.retrieval_recovery_arms import (  # noqa: E402
    FLAGS, SEED, _allirr, _base, load_corpora)

OUT_DIR = SRC / "eval/data/exa_probe"
SAMPLE = OUT_DIR / "sample.jsonl"
RESULTS = OUT_DIR / "results.jsonl"
COUNTER = OUT_DIR / "exa_requests.json"
FC_DOMS = set(FACT_CHECK_DOMAINS) | {d for d in TRUSTED_FACTCHECKERS if "/" not in d}

EXA_CAP = 100              # HARD stop on live Exa requests, for the life of the probe
EXA_PACE = 0.75            # seconds between live Exa requests, globally
LLM_CAP = 0.75             # hard USD stop on DeepInfra reads
N_PER_CORPUS = 50
TOP_K = 10
FLAG_EDGE = -4.05

# Neural search takes a description of the DOCUMENT you want, not keywords. The
# document we want is the one that would show the claim to be wrong. No claim is
# quoted back as a query and no dataset-derived example appears here.
EXA_QUERY_SYS = (
    "You write ONE search query for a NEURAL search engine that matches on meaning, not on "
    "keywords. The query must be a natural-language description of the kind of document that "
    "would show the following claim to be FALSE or would show what actually happened instead. "
    "Describe the document as a person would describe an article they are looking for: what it "
    "reports, about whom, where and when. Write one sentence, no boolean operators, no quotation "
    "marks, no site filters. Keep the specific people, places, numbers and organisations from the "
    "claim, because they are what the engine matches on. Do not ask whether the claim is true and "
    "do not describe a fact-check of it; describe the underlying reporting or record itself. "
    "Respond JSON only: {\"query\": \"...\"}")


# --------------------------------------------------------------------------- exa budget
class ExaBudget:
    """Live-request counter with a hard cap, persisted so a resume cannot re-spend.

    take() is called BEFORE the request leaves, so a crash over-counts rather than
    under-counts. Cache hits never call it."""

    def __init__(self, path: Path, cap: int):
        self.path, self.cap = path, cap
        self.lock = threading.Lock()
        self.last = 0.0
        self.n = json.loads(path.read_text())["n"] if path.exists() else 0

    def take(self) -> int:
        with self.lock:
            if self.n >= self.cap:
                raise SystemExit(f"EXA CAP {self.cap} REACHED ({self.n} spent) — stopping")
            self.n += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"n": self.n, "cap": self.cap}))
            wait = EXA_PACE - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            return self.n


BUDGET = ExaBudget(COUNTER, EXA_CAP)


def exa_search(query: str, xd: list[str]) -> tuple[list[dict], bool]:
    """The established Exa client (pipeline.search, provider='exa'), with the cache
    consulted first so a resume or a re-report costs nothing. NO date ceiling."""
    key = cache_key("exa", query, TOP_K, None, sorted(xd), 0)
    cached = disk_cache.get("exa", key)
    if cached is not None:
        for r in cached:
            r.setdefault("provider", "exa")
        return cached, True
    BUDGET.take()
    return search(query, TOP_K, date_ceiling=None, exclude_domains=xd,
                  min_results=0, provider="exa"), False


# --------------------------------------------------------------------------- sample
def build_sample() -> None:
    w = w7()
    rng = random.Random(SEED)
    cn, gold, _tl, _notes = load_corpora(w)
    rows = []
    for corpus, src, keyf in (("cn", cn, lambda r: r["post_id"]),
                              ("gold", gold, lambda r: r["review_url"])):
        block = [r for r in src.values() if _allirr(r)]
        block.sort(key=keyf)                      # deterministic frame before the draw
        rng2 = random.Random(SEED)
        rng2.shuffle(block)
        for r in block[:N_PER_CORPUS]:
            rows.append({"corpus": corpus, "key": r["review_url"],
                         "post_id": r.get("post_id"),
                         "claim": r.get("claim_resolved") or r.get("claim_text") or "",
                         "claim_date": r.get("claim_date_shown"),
                         "ceiling": r.get("ceiling"),
                         "publisher_site": r.get("publisher_site") or "x.com",
                         "query_prod": r.get("query") or "", **_base(r, w)})
        print(f"  {corpus}: all-irrelevant block {len(block)} -> drew "
              f"{min(N_PER_CORPUS, len(block))}")
    rng.shuffle(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLE.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    nb = sum(1 for r in rows if r["n_refute_base"])
    print(f"  TOTAL {len(rows)} claims | mean s7 "
          f"{sum(r['s7'] for r in rows)/len(rows):+.2f} | mean base docs "
          f"{sum(len(r['base_flags']) for r in rows)/len(rows):.2f} | "
          f"already >=1 refute {nb} (must be 0) -> {SAMPLE}")


# --------------------------------------------------------------------------- run
def gen_query(row: dict) -> tuple[str, float]:
    qblock = [f"CLAIM: {row['claim']}"]
    if row.get("claim_date"):
        qblock.append(f"CLAIM DATE: {row['claim_date']}")
    cost = 0.0
    for _ in range(3):
        try:
            q, c, _, _ = llm([{"role": "system", "content": EXA_QUERY_SYS},
                              {"role": "user", "content": "\n".join(qblock)}], max_tokens=160)
            cost += c
            out = (q.get("query") or "").strip()[:400]
            if out:
                return out, cost
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise RuntimeError("exa query-gen failed")


def is_future(date: str | None, ceiling: str | None) -> bool | None:
    if not date or not ceiling:
        return None
    return date[:10] > ceiling[:10]


def run_one(row: dict) -> dict:
    query, cost = gen_query(row)
    hits, cached = None, False
    for attempt in range(2):
        try:
            hits, cached = exa_search(query, [row["publisher_site"]])
            break
        except SearchError:
            if attempt == 0:
                time.sleep(30)
            else:
                raise

    seen = set(row["base_urls"])
    repeated = [h.get("url") for h in (hits or [])[:TOP_K] if h.get("url") in seen]
    claim_block = (f"CLAIM: {row['claim']}\n"
                   f"(claimed on {row.get('claim_date') or 'unknown date'}; judge the "
                   f"document's bearing on this exact proposition)")
    docs = []
    for h in (hits or [])[:TOP_K]:
        url = h.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        d = _domain_of(url)
        fc = any(d == f or d.endswith("." + f) for f in FC_DOMS)
        entry = {"url": url, "domain": d, "fc_domain": fc, "date": h.get("date"),
                 "future": is_future(h.get("date"), row.get("ceiling")),
                 "snippet": (h.get("snippet") or "")[:300]}
        if fc and not (h.get("date") or "").strip():
            entry.update(direction=None, read_status="fc-undated")   # urn policy, verbatim
            docs.append(entry)
            continue
        text = h.get("content") or scrape(url)
        prov = "exa-text" if h.get("content") else "scrape"
        if not text or len(text) < 200:
            text, prov = h.get("snippet") or "", "snippet"
        entry["provenance"] = prov
        regions, _ = select_regions(text, row["claim"], query)
        if not regions:
            entry.update(direction=None, read_status="empty-doc")
            docs.append(entry)
            continue
        rr = []
        for ids, sel in regions:
            got = None
            for _ in range(2):
                try:
                    got, c, _, _ = read_doc(claim_block, ids, sel, row["key"])
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
        entry["reason"] = agg.get("reason") or ""
        entry["sents"] = " ".join(s for _, sel in regions for s in sel)[:700]
        entry["read_status"] = "ok"
        docs.append(entry)
    return {"key": row["key"], "corpus": row["corpus"], "claim": row["claim"],
            "claim_date": row.get("claim_date"), "ceiling": row.get("ceiling"),
            "query_prod": row["query_prod"], "query_exa": query, "exa_cached": cached,
            "s7_base": row["s7"], "n_base_docs": len(row["base_flags"]),
            "n_refute_base": row["n_refute_base"], "n_hits": len(hits or []),
            "repeated_flags": [row["base_map"][u] for u in repeated],
            "n_new": len(docs), "docs": docs, "cost": cost}


def run(workers: int, smoke: int, verbose: bool) -> None:
    rows = [json.loads(l) for l in SAMPLE.open()]
    done = {json.loads(l)["key"] for l in RESULTS.open()} if RESULTS.exists() else set()
    todo = [r for r in rows if r["key"] not in done]
    if smoke:
        todo = todo[:smoke]
    if not todo:
        print("nothing to do")
        return
    print(f"exa probe: {len(todo)} claims | {workers} workers | exa spent so far "
          f"{BUDGET.n}/{EXA_CAP} | {dict(collections.Counter(r['corpus'] for r in todo))}",
          flush=True)
    lock = threading.Lock()
    n, cost, t0 = [0], [0.0], time.time()
    with RESULTS.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, r) for r in todo]
        for fut in as_completed(futs):
            try:
                out = fut.result()
            except SystemExit as e:
                print(f"  {e}", flush=True)
                continue
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                f.write(json.dumps(out) + "\n")
                f.flush()
                n[0] += 1
                cost[0] += out["cost"]
                if verbose:
                    print(f"\n--- {out['corpus']} {out['claim'][:110]}")
                    print(f"    Q_prod: {out['query_prod']}")
                    print(f"    Q_exa : {out['query_exa']}  (cached={out['exa_cached']})")
                    print(f"    hits {out['n_hits']} new {out['n_new']}")
                    for d in out["docs"]:
                        print(f"      [{d.get('direction') or d.get('read_status')}] "
                              f"{str(d.get('date'))[:10]:12s} fut={d.get('future')} "
                              f"{d['domain'][:34]:34s} {(d.get('reason') or '')[:60]}")
                if cost[0] > LLM_CAP:
                    raise SystemExit(f"LLM CAP ${LLM_CAP} breached at {n[0]} claims")
                if n[0] % 10 == 0 or n[0] == len(todo):
                    el = time.time() - t0
                    rate = n[0] / max(el, 0.001)
                    print(f"  {n[0]}/{len(todo)}  ${cost[0]:.3f}  exa {BUDGET.n}/{EXA_CAP}  "
                          f"{rate*60:.1f} claims/min  "
                          f"ETA {(len(todo)-n[0])/max(rate,1e-6)/60:.1f}m", flush=True)
    print(f"done: {n[0]} claims, LLM ${cost[0]:.4f}, exa {BUDGET.n}/{EXA_CAP}, "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


# --------------------------------------------------------------------------- report
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def report() -> None:
    w = w7()
    res = [json.loads(l) for l in RESULTS.open()]
    n = len(res)
    print(f"\n################ EXA PROBE  n={n} claims "
          f"({dict(collections.Counter(r['corpus'] for r in res))}), "
          f"exa requests spent {BUDGET.n}/{EXA_CAP}")

    def block(rows, label):
        m = len(rows)
        if not m:
            return
        read = [d for r in rows for d in r["docs"] if d.get("direction")]
        dirs = collections.Counter(d["direction"] for d in read)
        stat = collections.Counter(d.get("read_status") for r in rows for d in r["docs"])
        ref = [d for d in read if d["direction"] in ("1", "2")]
        moved = sum(1 for r in rows if any(d.get("direction") in ("1", "2") for d in r["docs"]))
        movedc = sum(1 for r in rows
                     if any(d.get("direction") in ("1", "2") and not d.get("future")
                            for d in r["docs"]))
        fut = [d for r in rows for d in r["docs"] if d.get("future")]
        undated = [d for r in rows for d in r["docs"] if not d.get("date")]
        nd = sum(r["n_new"] for r in rows)
        empty = sum(1 for r in rows if r["n_hits"] == 0)
        flag = 0
        for r in rows:
            add = [d for d in r["docs"] if d.get("direction")]
            s = r["s7_base"] + sum(w.get(d["direction"], 0.0) for d in add)
            nref = sum(1 for d in add if d["direction"] in ("1", "2"))
            flag += (s <= FLAG_EDGE and nref > 0)
        lo, hi = wilson(moved, m)
        lo2, hi2 = wilson(movedc, m)
        print(f"\n  --- {label}  n={m}")
        print(f"      hits/claim {sum(r['n_hits'] for r in rows)/m:.2f} | new docs {nd} "
              f"({nd/m:.2f}/claim) | read {len(read)} | repeats of base URLs "
              f"{sum(len(r['repeated_flags']) for r in rows)} | empty result sets {empty}")
        print("      new reads: " + " ".join(f"{k}:{dirs[k]}" for k in "54321XI" if dirs[k])
              + " | unread: " + " ".join(f"{k}:{v}" for k, v in stat.items() if k != "ok"))
        print(f"      REFUTING new docs {len(ref)} ({len(ref)/max(nd,1):.1%} of added)")
        print(f"      CLAIMS MOVED 0-refute -> >=1 refute: {moved}/{m} = {moved/m:.1%} "
              f"Wilson [{lo:.1%}, {hi:.1%}]")
        print(f"      ... ceiling-clean only          : {movedc}/{m} = {movedc/m:.1%} "
              f"Wilson [{lo2:.1%}, {hi2:.1%}]")
        print(f"      NEWLY FLAGGABLE (union, s7<=-4.05 & >=1 refute): {flag}/{m} "
              f"({flag/m:.1%})")
        print(f"      date hygiene: future-dated docs {len(fut)}/{nd} "
              f"({len(fut)/max(nd,1):.1%}), of which refuting "
              f"{sum(1 for d in fut if d.get('direction') in ('1','2'))} | undated "
              f"{len(undated)}/{nd}")

    block(res, "ALL")
    for c in sorted({r["corpus"] for r in res}):
        block([r for r in res if r["corpus"] == c], f"corpus={c}")
    per = collections.Counter(r["n_new"] for r in res)
    print("\n  per-claim NEW document counts: "
          + " ".join(f"{k}:{per[k]}" for k in sorted(per)))
    doms = collections.Counter(d["domain"] for r in res for d in r["docs"])
    print("  top domains: " + ", ".join(f"{k}({v})" for k, v in doms.most_common(12)))
    rdoms = collections.Counter(d["domain"] for r in res for d in r["docs"]
                                if d.get("direction") in ("1", "2"))
    print("  refuting-doc domains: " + ", ".join(f"{k}({v})" for k, v in rdoms.most_common(12)))


def show(k: int, misses: bool = False) -> None:
    shown = 0
    for l in RESULTS.open():
        r = json.loads(l)
        ref = [d for d in r["docs"] if d.get("direction") in ("1", "2")]
        if bool(ref) == misses:
            continue
        print("=" * 100)
        print(f"[{r['corpus']}] CLAIM:", r["claim"][:220])
        print("  claim date", r.get("claim_date"), "ceiling", r.get("ceiling"),
              "| base docs", r["n_base_docs"], "all I")
        print("  Q_prod:", r["query_prod"])
        print("  Q_exa :", r["query_exa"])
        print(f"  exa hits {r['n_hits']}, new {r['n_new']}")
        for d in (ref if not misses else r["docs"])[:6]:
            print(f"    [{d.get('direction') or d.get('read_status')}] "
                  f"{str(d.get('date'))[:10]:12s} future={d.get('future')} {d['domain']}")
            print(f"        {d['url'][:120]}")
            if d.get("reason"):
                print(f"        reason: {d['reason'][:150]}")
            if d.get("sents"):
                print(f"        {d['sents'][:300]}")
        shown += 1
        if shown >= k:
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.sample:
        build_sample()
    if a.run:
        run(min(a.workers, 4), a.smoke, a.verbose or bool(a.smoke))
    if a.report:
        report()
    if a.show:
        show(a.show, a.misses)
