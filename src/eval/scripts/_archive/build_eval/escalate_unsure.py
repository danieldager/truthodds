"""TH1c ESCALATION arm (logodds_sprint Workstream TH, Daniel 2026-08-25).

For claims the synthesis stage returned "unsure": ONE additional full pass —
new query (Exa, neural: content-rich natural-language phrasing, a different
angle than the original Serper query) → retrieve (Exa bundles page text; zero
Serper) → read the new docs (mode-routed prompt) → re-synthesize over the
combined old + new evidence.

Exa constraints are HARD (memory: never live-test Exa; 1k free req/month):
default mode is --fixtures (canned responses keyed by claim_id, built by hand
for development); the network path runs only under --live, once, sized and
ledgered first, guarded by --max-exa.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, ".")
import eval.scripts.build_eval.evidence_urn_run as eur  # noqa: E402
from eval.scripts.build_eval.read_v5_prompts import (  # noqa: E402
    READ_SYS_ATTRIB, READ_SYS_MODE)
from eval.scripts.build_eval.verdict_synthesis import synthesize  # noqa: E402
from pipeline.search import search  # noqa: E402
from pipeline.verify_tweet_claims import _domain_of, _rel_info, _voice_key  # noqa: E402

EXA_QUERY_SYS = """\
You write ONE web search query for a neural (semantic) search engine, to find
evidence about a claim. A previous keyword search with the query shown already
ran; your query must take a DIFFERENT angle: phrase what supposedly happened
the way a news article covering it would, in natural language. Prefer the
event's substance over rare names; never include a year, a verdict word
(hoax, debunked, fake, true), or constraints the claim does not state.

Reply with JSON only: {"query": "<the query>"}"""

SOCIAL_XD = ["x.com", "twitter.com", "t.co"]


def gen_exa_query(claim: str, prior_query: str) -> tuple[str | None, float]:
    for att in range(3):
        try:
            obj, cost, _, _ = eur.llm(
                [{"role": "system", "content": EXA_QUERY_SYS},
                 {"role": "user", "content": f"CLAIM: {claim}\nPREVIOUS QUERY: {prior_query}"}],
                max_tokens=80)
            q = (obj.get("query") or "").strip()[:300]
            if q:
                return q, cost
        except Exception:
            time.sleep(2 * (att + 1))
    return None, 0.0


def read_new_docs(rec: dict, hits: list[dict], read_sys: str, ng_scores: dict) -> tuple[list[dict], float]:
    """Replicates run_claim's per-doc prep+read for Exa hits; ranks continue at 11."""
    claim = rec.get("claim_resolved") or rec["claim_text"]
    claim_block = (f"CLAIM: {claim}\n"
                   f"(claimed on {rec.get('claim_date_shown') or 'unknown date'}; judge the "
                   f"document's bearing on this exact proposition)")
    ceil = rec.get("ceiling")
    seen_voices = {d["voice"] for d in rec.get("results") or []}
    old_urls = {d["url"] for d in rec.get("results") or []}
    out, cost_all = [], 0.0
    prev = eur.READ_SYS
    eur.READ_SYS = read_sys
    try:
        for i, h in enumerate(hits[:5]):
            url = h.get("url") or ""
            if not url or url in old_urls:
                continue
            rank = 11 + i
            dom = _domain_of(url)
            rel, ng = _rel_info(dom, url, ng_scores)
            text = h.get("content") or ""
            prov = "exa"
            if len(text) < eur.SNIPPET_MIN_TEXT:
                text, prov = h.get("snippet") or "", "snippet"
            regions, prep_meta = eur.select_regions(text, claim, "")
            smap = {i: s for ids, sel in regions for i, s in zip(ids, sel)}
            entry = {"rank": rank, "url": url, "domain": dom, "voice": _voice_key(dom),
                     "mirror": _voice_key(dom) in seen_voices, "rel": rel, "ng": ng,
                     "provenance": prov, "fc_domain": False,
                     "date": h.get("date"), "leak_flag": eur._is_leak(h.get("date"), ceil),
                     "n_sents": len(smap), "sent_ids": sorted(smap),
                     "sents": [smap[i] for i in sorted(smap)], "prep": prep_meta}
            if not smap:
                entry["read"] = {"direction": "I", "evidence": [], "reason": "",
                                 "qc_flag": "empty-doc-code"}
                entry["read_status"] = "empty-doc"
            else:
                reads = []
                for ids, sel in regions:
                    r = None
                    for _ in range(2):
                        try:
                            r, cost, _, _ = eur.read_doc(claim_block, ids, sel,
                                                         key=f"esc:{rec['claim_id']}")
                            cost_all += cost
                        except Exception:
                            time.sleep(2)
                            continue
                        if r:
                            break
                    reads.append(r or {"direction": "I", "evidence": [], "reason": "",
                                       "qc_flag": "read-failed"})
                entry["region_reads"] = reads
                entry["read"] = eur.aggregate_reads(reads)
                entry["read_status"] = ("failed" if all(
                    x.get("qc_flag") == "read-failed" for x in reads) else "ok")
            out.append(entry)
    finally:
        eur.READ_SYS = prev
    return out, cost_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="scores.jsonl-schema file")
    ap.add_argument("--synth", required=True, help="verdict_synthesis output jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", help="mode_labels.jsonl for read routing")
    ap.add_argument("--fixtures", help="canned Exa responses jsonl {claim_id, results:[...]}")
    ap.add_argument("--live", action="store_true", help="REAL Exa calls — one sized run only")
    ap.add_argument("--max-exa", type=int, default=60, help="hard live-request guard")
    ap.add_argument("--attrib-read", action="store_true",
                    help="route attribution claims to READ_SYS_ATTRIB (post-gate)")
    ap.add_argument("--cap", type=float, default=1.0)
    a = ap.parse_args()
    if not a.live and not a.fixtures:
        sys.exit("need --fixtures (dev) or --live (the one sized run)")

    modes = {}
    if a.modes:
        for line in open(a.modes):
            d = json.loads(line)
            if d.get("mode_llm"):
                modes[d["claim_id"]] = d["mode_llm"]

    unsure = {json.loads(l)["claim_id"] for l in open(a.synth)
              if json.loads(l)["verdict"] == "unsure"}
    recs = {}
    for line in open(a.input):
        r = json.loads(line)
        cid = r.get("claim_id") or r["review_url"]
        r.setdefault("claim_id", cid)
        if cid in unsure:
            recs[cid] = r

    fixtures = {}
    if a.fixtures:
        for line in open(a.fixtures):
            d = json.loads(line)
            fixtures[d["claim_id"]] = d["results"]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = {json.loads(l)["claim_id"] for l in open(out)} if out.exists() else set()
    jobs = [r for cid, r in sorted(recs.items()) if cid not in seen]
    if a.live:
        jobs = jobs[:a.max_exa]
    print(f"{len(unsure)} unsure | {len(jobs)} to escalate ({len(seen)} done) | "
          f"{'LIVE' if a.live else 'fixtures'} | cap ${a.cap}", flush=True)

    ng_scores = eur.newsguard_score_map()
    lock = threading.Lock()
    state = {"spent": 0.0, "done": 0, "exa": 0, "t0": time.time()}

    def one(rec):
        cid = rec["claim_id"]
        mode = modes.get(cid) or ("attribution" if rec.get("claim_type") == "attribution"
                                  else "assertion")
        read_sys = READ_SYS_ATTRIB if (a.attrib_read and mode == "attribution") else READ_SYS_MODE
        claim = rec.get("claim_resolved") or rec["claim_text"]
        q, qcost = gen_exa_query(claim, rec.get("query") or "")
        if not q:
            return cid, None, qcost
        if a.fixtures:
            hits = fixtures.get(cid, [])
        else:
            with lock:
                state["exa"] += 1
                if state["exa"] > a.max_exa:
                    return cid, None, qcost
            hits = search(q, 5, date_ceiling=rec.get("ceiling") or None,
                          exclude_domains=SOCIAL_XD, min_results=0, provider="exa")
        new_docs, rcost = read_new_docs(rec, hits, read_sys, ng_scores)
        directional = any((d.get("read") or {}).get("direction") in ("5", "4", "3", "2", "1")
                          for d in new_docs)
        if not directional:
            # No new information -> no new verdict: re-running synthesis on the same
            # evidence can flip a borderline unsure (observed in the fixture dry run,
            # unsure 0.1 -> false 0.95 on two junk I-docs). The verdict stands.
            return cid, {"review_url": rec["review_url"], "claim_id": cid,
                         "verdict": "unsure", "confidence": 0.0,
                         "reason": "escalation retrieved no directional evidence",
                         "key_evidence": [], "mode": mode, "exa_query": q,
                         "stage": "escalated-no-new-evidence",
                         "new_docs": [{k: d.get(k) for k in
                                       ("rank", "url", "domain", "rel", "date", "read",
                                        "read_status", "leak_flag", "sent_ids", "sents")}
                                      for d in new_docs]}, qcost + rcost
        row, scost = synthesize(rec, mode, extra_docs=new_docs)
        if row:
            row.update({"mode": mode, "exa_query": q, "stage": "escalated",
                        "new_docs": [{k: d.get(k) for k in
                                      ("rank", "url", "domain", "rel", "date", "read",
                                       "read_status", "leak_flag", "sent_ids", "sents")}
                                     for d in new_docs]})
        return cid, row, qcost + rcost + scost

    with open(out, "a") as f, ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(one, r) for r in jobs]
        for fu in as_completed(futs):
            cid, row, cost = fu.result()
            with lock:
                state["spent"] += cost
                state["done"] += 1
                if row:
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                print(f"  {state['done']}/{len(jobs)} {cid} "
                      f"{'ok' if row else 'FAIL'} ${state['spent']:.3f}", flush=True)
                if state["spent"] > a.cap:
                    print("CAP HIT — aborting", flush=True)
                    sys.exit(1)
    print(f"DONE {state['done']} ${state['spent']:.3f} exa_used={state['exa']}", flush=True)


if __name__ == "__main__":
    main()
