"""Cap-sweep: principled READ token budget (program §4b; Daniel 2026-07-25).

Question: does raising the per-doc cap buy real evidence? Design:
- ~200 CAL claims (seed 606, excluded later from nothing — methodological run) →
  search+scrape; keep docs whose post-clean length exceeds the contested region
  (>2.5k tokens ≈ 10k chars) until ~150 such docs.
- REFERENCE per doc: CHUNKED full-doc read — contiguous ≤2k-token chunks, each read
  independently (no long-context degradation in the gold standard); E_ref = union of
  directional sentence ids.
- ARMS: prep-v5 selection + read at caps 4k / 8k / 16k chars (~1k/2k/4k tokens),
  same docs, paired.
- METRICS per cap: containment = |E_ref ∩ window| / |E_ref|; detection = of contained
  ref sentences, share cited; ghost rate = cited-but-not-in-E_ref (precision proxy);
  direction agreement vs reference majority; all stratified by evidence position
  third (beginning/middle/end of doc) and by anchor density.

  uv run python -m eval.scripts.build_eval.cap_sweep
Output: eval/data/urn_runs/cap_sweep.jsonl + printed summary table.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import eval.scripts.build_eval.evidence_urn_run as R

OUT = Path("eval/data/urn_runs/cap_sweep.jsonl")
CAPS = [4000, 8000, 16000]          # chars ≈ 1k / 2k / 4k tokens
CONTESTED_MIN_CHARS = 10000         # only docs where the cap actually binds
CONTESTED_MAX_CHARS = 160000        # >40k-token monsters excluded: uniformly irrelevant (measured), dominate runtime
TARGET_DOCS = 150
CHUNK_CHARS = 8000                  # reference chunk size (~2k tokens, comfy zone)


from concurrent.futures import ThreadPoolExecutor
import threading
_SEL_LOCK = threading.Lock()   # select_sentences mutates R.MAX_DOC_CHARS globals


def chunked_reference(parts, claim_block, key):
    """Chunked full-doc reference; chunk 1 primes the prompt cache, rest run in
    PARALLEL (chunks are independent)."""
    chunks, i = [], 0
    while i < len(parts):
        j, total = i, 0
        while j < len(parts) and total + len(parts[j]) <= CHUNK_CHARS:
            total += len(parts[j]); j += 1
        chunks.append((list(range(i + 1, j + 1)), parts[i:j]))
        i = j
    ref_sup, ref_ag, cost_total = set(), set(), 0.0

    def _read(ch):
        ids, sl = ch
        try:
            out, cost, _, _ = R.read_doc(claim_block, ids, sl, key)
            return out, cost
        except Exception:
            return None, 0.0

    out0, c0 = _read(chunks[0])
    cost_total += c0
    if out0:
        ref_sup.update(out0["support"]); ref_ag.update(out0["against"])
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for out, c in ex.map(_read, chunks[1:]):
                cost_total += c
                if out:
                    ref_sup.update(out["support"]); ref_ag.update(out["against"])
    return ref_sup, ref_ag, cost_total


def main():
    df = pl.read_parquet(R.CAL).sample(200, seed=606)
    ng = R.newsguard_score_map()
    done = set()
    if OUT.exists():
        for l in open(OUT):
            done.add(json.loads(l)["doc_key"])
    fh = open(OUT, "a")
    n_docs, spent, t0 = len(done), 0.0, time.time()

    for row in df.iter_rows(named=True):
        if n_docs >= TARGET_DOCS:
            break
        claim = row["claim_text"]
        ceil, _src = R.ceiling_for(row)
        try:
            query = None
            q, cost, _, _ = R.llm([{"role": "system", "content": R.QUERY_SYS},
                                   {"role": "user", "content": f"CLAIM: {claim}"}], max_tokens=80)
            spent += cost
            query = (q.get("query") or "").strip()
            if not query:
                continue
            R._serper_gate()
            hits = R.search(query, 10, date_ceiling=ceil or None,
                            exclude_domains=sorted(set(R.FACT_CHECK_DOMAINS)
                                                   | {d for d in R.TRUSTED_FACTCHECKERS if "/" not in d}
                                                   | {row["publisher_site"]}),
                            min_results=10, stats={}, provider="serper")
        except Exception:
            continue
        claim_block = f"CLAIM: {claim}\n(judge the document's bearing on this exact proposition)"
        for h in hits[:10]:
            if n_docs >= TARGET_DOCS:
                break
            url = h.get("url") or ""
            doc_key = f"{row['review_url']}|{url}"
            if doc_key in done:
                continue
            text = h.get("content") or R.scrape(url)
            if not text:
                continue
            parts = R.sentences(text)
            dlen = sum(len(x) for x in parts)
            if dlen < CONTESTED_MIN_CHARS or dlen > CONTESTED_MAX_CHARS:
                continue
            # reference (parallel chunks)
            ref_sup, ref_ag, rc = chunked_reference(parts, claim_block, doc_key)
            spent += rc
            ref = sorted(ref_sup | ref_ag)
            # arm selections serialized (global-state mutation), reads in parallel
            selections = {}
            with _SEL_LOCK:
                for cap in CAPS:
                    R.MAX_DOC_CHARS, R.MAX_SENTS = cap, 400
                    selections[cap] = R.select_sentences(text, claim, query)

            def _arm(cap):
                ids, sel, meta = selections[cap]
                try:
                    out, cost, _, _ = R.read_doc(claim_block, ids, sel, doc_key)
                    return cap, ids, meta, out, cost
                except Exception:
                    return cap, ids, meta, None, 0.0

            arms = {}
            with ThreadPoolExecutor(max_workers=3) as ex:
                for cap, ids, meta, out, cost in ex.map(_arm, CAPS):
                    spent += cost
                    arms[str(cap)] = {"window_ids": ids, "meta": meta,
                                      "read": out or {"direction": None, "support": [],
                                                      "neutral": [], "against": []}}
            fh.write(json.dumps({"doc_key": doc_key, "claim": claim[:200],
                                 "url": url, "n_sents": len(parts),
                                 "ref_support": sorted(ref_sup), "ref_against": sorted(ref_ag),
                                 "arms": arms}) + "\n")
            fh.flush()
            n_docs += 1
            done.add(doc_key)
            if n_docs % 10 == 0:
                print(f"  {n_docs}/{TARGET_DOCS} docs | ${spent:.2f} | "
                      f"{(time.time()-t0)/60:.0f}m", flush=True)
    fh.close()

    # ---- summary ----
    recs = [json.loads(l) for l in open(OUT)]
    print(f"\n== cap sweep: {len(recs)} docs, ${spent:.2f} this session ==")
    for cap in CAPS:
        cont_n = cont_d = det_n = det_d = ghost = cited = 0
        for r in recs:
            ref = set(r["ref_support"]) | set(r["ref_against"])
            if not ref:
                continue
            a = r["arms"].get(str(cap))
            if not a:
                continue
            w = set(a["window_ids"])
            cited_ids = set(a["read"]["support"]) | set(a["read"]["against"])
            cont_n += len(ref & w); cont_d += len(ref)
            det_n += len(ref & w & cited_ids); det_d += len(ref & w)
            ghost += len(cited_ids - ref); cited += len(cited_ids)
        print(f"cap {cap:5d}c (~{cap//4}tok): containment {cont_n}/{cont_d} "
              f"({cont_n/max(cont_d,1):.0%}) | detection {det_n}/{det_d} "
              f"({det_n/max(det_d,1):.0%}) | ghost-cite {ghost}/{max(cited,1)} "
              f"({ghost/max(cited,1):.0%})")


if __name__ == "__main__":
    main()
