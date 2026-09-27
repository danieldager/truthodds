"""Three retrieval arms on ONE fixed population: does dropping the date ceiling help?

Daniel's decision (2026-08-28): production retrieval refuses any document published
after the claim, so the 166/211 post-dated refuting documents the Exa probe found are
unreachable. A genuinely novel claim has no evidence either way and must wait; most
claims are not novel. Measure what removing the ceiling buys.

  A  ceiling ON, production configuration. NOT re-run: the saved production reads
     (e1_ctx/results-00.jsonl, true_timeline/scores.jsonl) ARE arm A, so it
     reproduces the known numbers by construction rather than by luck.
  A' arm A's saved documents with the circularity filter applied post-hoc, so the
     A -> B comparison is not confounded by the filter itself.
  B  ceiling OFF, circular evidence excluded. THE HEADLINE.
  C  ceiling OFF, production exclusions only (origin outlet + social blocklist).
     Known-contaminated upper bound. NOT a real number.

CIRCULARITY. Gold labels come FROM fact-check articles and CN labels FROM community
notes, so for every labelled claim a post-dated document written about that exact
claim by the reviewer that produced our label exists BY CONSTRUCTION. Arm B excludes,
each rule recorded separately so the report can price it:
  own_review  the claim's own review_url and its host
  fc_domain   FACT_CHECK_DOMAINS + TRUSTED_FACTCHECKERS + every distinct host of an
              fc-gold review_url (11, all already inside those two lists)
  fc_path     syndicated fact-check SECTIONS on non-fc domains (URL-path rule)
  note_mirror community-note mirrors; x/twitter/t.co are already in SCRAPE_BLOCKLIST
  origin      publisher_site + exclude_extra, via search(exclude_domains=...)
Domain rules cannot catch a syndicated copy of a fact-check on an unlisted host, so
every READ document also gets a one-shot reader boolean, "is this document itself a
fact-check of THIS claim". Arm B is reported with and without those.

Serper's `-site:` budget is ~32 words and _SITE_OP_BUDGET is 16, so a fact-check
blocklist cannot go in the query: exclusion is POST-retrieval. Retrieve-k CANNOT be
raised to compensate — `num` above 10 is ignored on this plan (measured, see TOP_K) —
so arm B legitimately retrieves FEWER documents than arm A, and the only depth on
offer is a second billed page. Both are reported: B_p1 is the uncompensated
shortfall, B is page-2-backfilled to 10. The document-count difference is a CONFOUND
on the B-minus-A delta, not noise, and the report says so.

  uv run python -m eval.scripts.build_eval.no_ceiling_arms --sample
  uv run python -m eval.scripts.build_eval.no_ceiling_arms --run --smoke 15
  uv run python -m eval.scripts.build_eval.no_ceiling_arms --run
  uv run python -m eval.scripts.build_eval.no_ceiling_arms --report

Nothing under eval/data/urn_runs/ is written. Output: eval/data/no_ceiling/.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from pipeline.config import FACT_CHECK_DOMAINS, SCRAPE_BLOCKLIST  # noqa: E402
from pipeline.credibility import TRUSTED_FACTCHECKERS  # noqa: E402
from pipeline.search import (  # noqa: E402
    SearchError, _serper_page, scrape, search, serper_finalize)
from pipeline.verify_tweet_claims import _domain_of  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
    _serper_gate, aggregate_reads, llm, read_doc, select_regions)
from eval.scripts.build_eval import fit_urn  # noqa: E402

OUT_DIR = SRC / "eval/data/no_ceiling"
SAMPLE = OUT_DIR / "sample.jsonl"
RESULTS = OUT_DIR / "results.jsonl"
E1 = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
TL = SRC / "eval/data/urn_runs/true_timeline/scores.jsonl"
LADDER = SRC / "eval/data/urn_runs/e1_ctx/model_ladder/ladder.json"

FLAGS = ("5", "4", "3", "2", "1", "X", "I")
FLAGSET = set(FLAGS)
VOICE = {"5": "support", "4": "support", "1": "refute", "2": "refute",
         "3": "silent", "X": "silent", "I": "silent"}
SEED = 20260828
BOOT_REPS, BOOT_SEED = 2000, 707
# retrieve-k CANNOT be raised on this Serper plan: `num` above 10 is ignored, not
# billed — measured 2026-08-28 on our own key, num=10/20/30/100 all return ~9-10
# organic and the response body says {"credits": 1} every time. So k stays at the
# production 10 and the only depth available is the `page` parameter, one extra
# billed request per page. search()'s cache key omits `page`, so the page-2 pull
# bypasses the cache entirely (`_serper_page` direct) and cannot corrupt it.
TOP_K = 10
KEEP = 10           # surviving documents kept per arm (production urn size)
BUDGET_CAP = 2.60   # hard USD stop on LLM spend, checked after every claim
N_GOLD_PER_CLASS = 250
N_TL = 300

# --- circularity blocklist -------------------------------------------------
# The 11 distinct hosts of fc-gold review_urls (snopes, politifact, factcheck.afp,
# factuel.afp, verafiles, boomlive, factcheck.org, aap, fullfact, africacheck,
# ghanafact) are all already inside these two lists; the union is taken anyway so
# the rule survives a corpus change.
GOLD_REVIEW_HOSTS = set()   # filled by build_sample from the actual review_urls
FC_BLOCK = set(FACT_CHECK_DOMAINS) | {d for d in TRUSTED_FACTCHECKERS if "/" not in d} | {
    # the Feedback network publishes at *.feedback.org, which neither list carries
    # (they hold sciencefeedback.co / healthfeedback.org); the smoke's reader boolean
    # caught science.feedback.org surviving the domain rules.
    "feedback.org", "sciencefeedback.org", "usatoday.com/story/news/factcheck"}
NOTE_DOMAINS = {"communitynotes.x.com", "communitynotes.twitter.com", "birdwatch.x.com",
                "transparency.x.com", "communitynotes.info", "twittercommunitynotes.com"}
_FC_PATH = re.compile(
    r"fact[-_/]?check|factcheck|debunk|hoax|/rumou?r|misinformation|disinformation|"
    r"fake[-_]news|false[-_]claim|no[-_]evidence|verificacion|verificaci|faktencheck|"
    r"desmentido|bulo|birdwatch|community[-_]note", re.I)

# v2. v1 ("does it adjudicate the claim") read as "does it settle the claim" and fired on
# a 1937 NYT archive piece, a Medium history blog and four ordinary endorsement reports —
# it was measuring content, not genre. v2 asks for the GENRE and names the exempt classes.
# Offline A/B on the 149 smoke documents: 18 rescued, 11 newly caught, of which the
# science.feedback.org review was a genuine fact-check the domain list had missed.
FC_DETECT_SYS = (
    "You classify a document's GENRE, not its content. Question: is this document a "
    "FACT-CHECK ARTICLE about the claim below — a piece whose purpose is to rate, verify "
    "or debunk that claim and deliver a verdict on it — or a republication or summary of "
    "such a piece? Genre is what decides. Ordinary news reporting, archived articles, "
    "primary documents, official statements, encyclopedia entries, opinion and commentary "
    "are NOT fact-checks, even when what they report settles the claim completely. "
    "A document is a fact-check only if it presents itself as adjudicating a circulating "
    "claim: it names the claim as something being asserted or shared, and it issues a "
    "verdict on it. Respond JSON only: {\"factcheck_of_claim\": true or false}")


def host(u: str) -> str:
    return urlsplit(u or "").netloc.lower().removeprefix("www.").removeprefix("m.")


def in_list(d: str, doms: set) -> bool:
    return d in doms or any(d.endswith("." + x) for x in doms)


def circ_reason(url: str, review_url: str) -> str | None:
    """Why arm B drops this document, or None. First matching rule wins; the rules
    are recorded separately so the report can say what each one cost."""
    d = _domain_of(url)
    if url == review_url or (review_url.startswith("http") and d == host(review_url)):
        return "own_review"
    if in_list(d, FC_BLOCK) or in_list(d, GOLD_REVIEW_HOSTS):
        return "fc_domain"
    if in_list(d, NOTE_DOMAINS):
        return "note_mirror"
    if _FC_PATH.search(urlsplit(url).path or ""):
        return "fc_path"
    return None


# --------------------------------------------------------------------------- sample
def gold_rows() -> list[dict]:
    """model_ladder's headline population, verbatim: veracity 1-5, >=1 read document,
    media-provenance axis out, rating_subtype 'mixed' out. y=1 iff veracity>=4, so
    unprovable (3) is FALSE at eval; mid=True keeps it out of every fit."""
    axis = fit_urn.load_judged_axis()
    out = []
    for line in E1.open():
        r = json.loads(line)
        v = r.get("veracity")
        if v not in (1, 2, 3, 4, 5):
            continue
        if (r.get("rating_subtype") or "?") == "mixed":
            continue
        if (axis.get(r["review_url"]) or "untagged") == fit_urn.MEDIA_AXIS:
            continue
        docs = [d for d in (r.get("results") or [])
                if (d.get("read") or {}).get("direction") in FLAGSET]
        if not docs:
            continue
        out.append(_row(r, "gold", 1 if v >= 4 else 0, v == 3, docs))
    return out


def tl_rows() -> list[dict]:
    out = []
    for line in TL.open():
        r = json.loads(line)
        docs = [d for d in (r.get("results") or [])
                if (d.get("read") or {}).get("direction") in FLAGSET]
        if not docs:
            continue
        out.append(_row(r, "timeline", None, False, docs))
    return out


def _row(r: dict, corpus: str, y, mid: bool, docs: list[dict]) -> dict:
    return {"key": r["review_url"], "corpus": corpus, "y": y, "mid": mid,
            "veracity": r.get("veracity"),
            "claim": r.get("claim_resolved") or r.get("claim_text") or "",
            "claim_date": r.get("claim_date_shown") or "",
            "ceiling": r.get("ceiling") or "",
            "publisher_site": r.get("publisher_site") or "x.com",
            "exclude_extra": list(r.get("exclude_extra") or []),
            "query": r.get("query") or "",
            "post_id": r.get("post_id"),
            "fold": fit_urn.fold_of(r["review_url"], fit_urn.K_FOLDS),
            # arm A = the saved production reads, kept whole (url so A' can filter).
            # `txt` is the text the production reader actually saw, carried so a
            # REUSED document needs neither a re-scrape nor a re-read: a read is a
            # (claim, url) pair, and this record IS that pair's saved read.
            "armA": [{"url": d.get("url") or "", "flag": d["read"]["direction"],
                      "date": d.get("date"), "status": d.get("read_status"),
                      "txt": " ".join(d.get("sents") or [])[:1200]} for d in docs]}


def build_sample() -> None:
    rng = random.Random(SEED)
    g, t = gold_rows(), tl_rows()
    print(f"gold headline population {len(g)}  timeline population {len(t)}")
    pos = [r for r in g if r["y"] == 1]
    neg = [r for r in g if r["y"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    rng.shuffle(t)
    rows = pos[:N_GOLD_PER_CLASS] + neg[:N_GOLD_PER_CLASS] + t[:N_TL]
    rows = [r for r in rows if r["query"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLE.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    c = collections.Counter((r["corpus"], r["y"]) for r in rows)
    for k in sorted(c, key=str):
        sub = [r for r in rows if (r["corpus"], r["y"]) == k]
        print(f"  {str(k):22s} n={c[k]:4d}  mean armA docs "
              f"{sum(len(r['armA']) for r in sub)/len(sub):.2f}")
    print(f"  TOTAL {len(rows)} -> {SAMPLE}")


# --------------------------------------------------------------------------- run
def fc_detect(claim: str, dom: str, text: str) -> tuple[bool | None, float]:
    """One-shot reader boolean: is THIS document a fact-check of THIS claim.
    Catches syndicated/aggregated copies that domain filtering structurally cannot."""
    try:
        obj, cost, _, _ = llm(
            [{"role": "system", "content": FC_DETECT_SYS},
             {"role": "user", "content": f"CLAIM: {claim}\nSOURCE: {dom}\n\n"
                                         f"DOCUMENT:\n{text[:1500]}"}],
            max_tokens=24)
        v = obj.get("factcheck_of_claim")
        return (bool(v) if isinstance(v, bool) else None), cost
    except Exception:  # noqa: BLE001
        return None, 0.0


def read_one(row: dict, h: dict, claim_block: str, saved: dict) -> tuple[dict, float]:
    """One document through the production read chain, verbatim (scrape -> snippet
    fallback, prep-v7 regions, read-v5, fc-undated refusal), plus the fc boolean.

    REUSE (Daniel 2026-08-28): read only NEW evidence. A read is a (claim, document)
    pair, so a URL that already has a saved read UNDER THIS CLAIM is reused verbatim
    from the urn run and never re-sent to the reader. The same URL under a different
    claim is a different pair and gets read. The fc-detector still runs, off the saved
    text, so arm B_rd is defined on every arm-B document."""
    url = h.get("url") or ""
    d = _domain_of(url)
    fc_domain = in_list(d, FC_BLOCK) or in_list(d, GOLD_REVIEW_HOSTS)
    entry = {"url": url, "domain": d, "date": h.get("date"), "fc_domain": fc_domain,
             "circ": circ_reason(url, row["key"])}
    cost = 0.0
    if url in saved:
        s = saved[url]
        entry.update(flag=s["flag"], read_status="reused", reused=True,
                     saved_status=s.get("status"), sents=(s.get("txt") or "")[:400])
        det, c = (True, 0.0) if s.get("status") == "fc-undated" else \
            fc_detect(row["claim"], d, s.get("txt") or "")
        entry["fc_detect"] = det
        return entry, c
    entry["reused"] = False
    if fc_domain and not (h.get("date") or "").strip():
        entry.update(flag="I", read_status="fc-undated", fc_detect=True)
        return entry, cost
    text = h.get("content") or scrape(url)
    prov = "scrape"
    if not text or len(text) < 200:
        text, prov = h.get("snippet") or "", "snippet"
    entry["provenance"] = prov
    regions, _ = select_regions(text, row["claim"], row["query"])
    if not regions:
        entry.update(flag="I", read_status="empty-doc", fc_detect=None)
        return entry, cost
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
    entry["flag"] = agg["direction"]
    entry["reason"] = (agg.get("reason") or "")[:120]
    entry["read_status"] = "failed" if all(x.get("qc_flag") == "read-failed"
                                           for x in rr) else "ok"
    seen_text = " ".join(s for _, sel in regions for s in sel)
    entry["sents"] = seen_text[:400]
    det, c2 = fc_detect(row["claim"], d, seen_text)
    entry["fc_detect"] = det
    cost += c2
    return entry, cost


def run_one(row: dict) -> dict:
    xd = [row["publisher_site"]] + list(row["exclude_extra"])
    hits = None
    for attempt in range(2):
        _serper_gate()
        try:
            hits = search(row["query"], TOP_K, date_ceiling=None, exclude_domains=xd,
                          min_results=0, provider="serper")
            break
        except SearchError:
            if attempt == 0:
                time.sleep(120)
            else:
                raise
    hits = hits or []
    for h in hits:
        h["page"] = 1
    # arm C: production filtering only. arm B: + the circularity rules. Both capped
    # at KEEP so the urn size is the production one; the union is what we pay to read.
    c_hits = hits[:KEEP]
    b_hits = [h for h in hits if not circ_reason(h.get("url") or "", row["key"])][:KEEP]
    # Page-2 pull, ONLY when the filter starved arm B. It does not erase the
    # document-count confound (page-2 documents are lower-ranked than anything arm A
    # ever saw), so the report keeps BOTH: B_p1 = page 1 only, the honest shortfall,
    # and B = backfilled to KEEP, count-matched to A and C.
    n_p2, n_req = 0, 1
    if len(b_hits) < KEEP:
        n_req = 2
        _serper_gate()
        try:
            p2 = serper_finalize(_serper_page(row["query"], TOP_K, None, xd, 2), xd)
        except SearchError:
            p2 = []
        seen_p1 = {h.get("url") for h in hits}
        extra = [h for h in p2 if h.get("url") not in seen_p1
                 and not circ_reason(h.get("url") or "", row["key"])]
        for h in extra:
            h["page"] = 2
        extra = extra[:KEEP - len(b_hits)]
        n_p2 = len(extra)
        b_hits = b_hits + extra
    union, seen = [], set()
    for h in c_hits + b_hits:
        u = h.get("url") or ""
        if u and u not in seen:
            seen.add(u)
            union.append(h)
    b_urls = {h.get("url") for h in b_hits}
    c_urls = {h.get("url") for h in c_hits}

    claim_block = (f"CLAIM: {row['claim']}\n"
                   f"(claimed on {row.get('claim_date') or 'unknown date'}; judge the "
                   f"document's bearing on this exact proposition)")
    saved = {d["url"]: d for d in row["armA"] if d["url"]}
    docs, cost = [], 0.0
    for h in union:
        e, c = read_one(row, h, claim_block, saved)
        e["in_b"] = h.get("url") in b_urls
        e["in_c"] = h.get("url") in c_urls
        e["page"] = h.get("page", 1)
        docs.append(e)
        cost += c
    return {"key": row["key"], "corpus": row["corpus"], "y": row["y"], "mid": row["mid"],
            "fold": row["fold"], "claim": row["claim"], "ceiling": row["ceiling"],
            "query": row["query"], "n_hits": len(hits), "n_p2": n_p2,
            "n_requests": 2 if n_p2 or len(b_hits) < KEEP else 1,
            "docs": docs, "cost": cost}


def run(workers: int, smoke: int, limit: int = 0) -> None:
    rows = [json.loads(l) for l in SAMPLE.open()]
    _load_hosts(rows)
    done = set()
    if RESULTS.exists():
        done = {json.loads(l)["key"] for l in RESULTS.open()}
    todo = [r for r in rows if r["key"] not in done]
    # Class-mixed batch order, fixed seed. The tool call this runs in is capped at
    # 10 minutes, so the run is chunked; shuffling first means any partial result set
    # is balanced across gold-T / gold-F / timeline rather than truncated by stratum.
    random.Random(SEED + 1).shuffle(todo)
    if limit:
        todo = todo[:limit]
    if smoke:
        rng = random.Random(SEED)
        by = collections.defaultdict(list)
        for r in todo:
            by[(r["corpus"], r["y"])].append(r)
        todo = []
        for k in sorted(by, key=str):
            rng.shuffle(by[k])
            todo += by[k][:max(1, smoke // len(by))]
    if not todo:
        print("nothing to do")
        return
    print(f"{len(todo)} claims | {workers} workers | "
          f"{dict(collections.Counter((r['corpus'], r['y']) for r in todo))}", flush=True)
    lock = threading.Lock()
    n, cost, creds, t0 = [0], [0.0], [0], time.time()
    nre, nnew = [0], [0]
    with RESULTS.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, r) for r in todo]
        for fut in as_completed(futs):
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                f.write(json.dumps(out) + "\n")
                f.flush()
                n[0] += 1
                cost[0] += out["cost"]
                creds[0] += out["n_requests"]
                nre[0] += sum(1 for d in out["docs"] if d.get("reused"))
                nnew[0] += sum(1 for d in out["docs"] if not d.get("reused"))
                if cost[0] > BUDGET_CAP:
                    raise SystemExit(f"BUDGET CAP ${BUDGET_CAP} hit at {n[0]} claims")
                if n[0] % 10 == 0 or n[0] == len(todo):
                    el = time.time() - t0
                    rate = n[0] / max(el, 1e-3)
                    print(f"  {n[0]}/{len(todo)}  ${cost[0]:.3f}  {rate*60:.1f} claims/min  "
                          f"ETA {(len(todo)-n[0])/max(rate,1e-6)/60:.1f}m  "
                          f"reused {nre[0]}/{nre[0]+nnew[0]} docs", flush=True)
    print(f"done: {n[0]} claims, LLM ${cost[0]:.4f}, {creds[0]} serper requests "
          f"= {creds[0]} credits (Serper's own body says credits:1 per request, any num), "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


def agree(n: int, workers: int = 10) -> None:
    """Prompt-version agreement check on REUSED gold documents.

    Why it is needed. The saved reads are reused verbatim, so they must come from the
    same instrument as the new reads. read_v5_prompts.py changed twice since the gold
    run: 4018927 (08-27) only APPENDS READ_SYS_ATTRIB, which is gated and unwired, but
    b922a02 (08-05, the morning after e1_ctx finished) introduced READ_SYS_MODE. A
    byte-compare of today's READ_SYS_MODE against the READ_SYS_V5 that produced e1_ctx
    shows the WORDS are identical and only one line-break moved, inside a single
    paragraph. Identical words, different token stream: too small to reason about,
    so it is measured. The timeline corpus needs no check (its run is 08-27, after the
    change, so its saved reads and the new reads share a byte-identical prompt).

    Re-reads the saved sentences with the CURRENT prompt and reports agreement. The
    disagreement this finds is prompt drift AND temperature-0 nondeterminism together
    (the ledger's read_ab row measured ~11% temp-0 drift on its own), so it is an
    upper bound on the prompt's contribution, not an estimate of it.
    """
    keys = {json.loads(l)["key"] for l in SAMPLE.open()
            if json.loads(l)["corpus"] == "gold"}
    pool = []
    for line in E1.open():
        r = json.loads(line)
        if r["review_url"] not in keys:
            continue
        cb = (f"CLAIM: {r.get('claim_resolved') or r['claim_text']}\n"
              f"(claimed on {r.get('claim_date_shown') or 'unknown date'}; judge the "
              f"document's bearing on this exact proposition)")
        for d in r.get("results") or []:
            if d.get("read_status") == "ok" and d.get("sent_ids") and d.get("sents"):
                pool.append((cb, r["review_url"], d))
    random.Random(SEED).shuffle(pool)
    pool = pool[:n]
    print(f"re-reading {len(pool)} saved gold documents with the current prompt")

    def go(t):
        cb, key, d = t
        out, c, _, _ = read_doc(cb, d["sent_ids"], d["sents"], key)
        return d["read"]["direction"], (out or {}).get("direction"), c

    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = list(ex.map(go, pool))
    ok = [g for g in got if g[1]]
    same = sum(1 for a, b, _ in ok if a == b)
    print(f"  agreement {same}/{len(ok)} = {same/max(len(ok),1):.1%}  "
          f"(${sum(g[2] for g in got):.4f})")
    vsame = sum(1 for a, b, _ in ok if VOICE[a] == VOICE[b])
    print(f"  voice-level agreement {vsame}/{len(ok)} = {vsame/max(len(ok),1):.1%}")
    conf = collections.Counter((a, b) for a, b, _ in ok if a != b)
    print("  disagreements (saved -> re-read):",
          ", ".join(f"{a}->{b}:{k}" for (a, b), k in conf.most_common(12)) or "none")


def _load_hosts(rows: list[dict]) -> None:
    GOLD_REVIEW_HOSTS.clear()
    for r in rows:
        if r["corpus"] == "gold" and r["key"].startswith("http"):
            GOLD_REVIEW_HOSTS.add(host(r["key"]))
    # every distinct fc-gold review host in the WHOLE corpus, not just the sample
    for line in E1.open():
        u = json.loads(line)["review_url"]
        if u.startswith("http"):
            GOLD_REVIEW_HOSTS.add(host(u))


def smoke_inspect(k: int = 15) -> None:
    """Eyeball the blocklist: what survived, what was dropped and why."""
    res = [json.loads(l) for l in RESULTS.open()]
    print(f"\ninspecting {len(res)} claims")
    kept = collections.Counter()
    drop = collections.Counter()
    for r in res:
        for d in r["docs"]:
            (drop if d["circ"] else kept)[d["circ"] or "kept"] += 1
    print("  drop reasons:", dict(drop), " kept:", kept["kept"])
    shown = 0
    for r in res:
        dd = [d for d in r["docs"] if d["circ"] or d.get("fc_detect")]
        if not dd:
            continue
        print("=" * 96)
        print("CLAIM:", r["claim"][:150])
        for d in dd[:6]:
            print(f"   [{d['circ'] or '-':11s}] fc_detect={str(d.get('fc_detect')):5s} "
                  f"[{d.get('flag')}] {d['url'][:100]}")
        shown += 1
        if shown >= k:
            break


# --------------------------------------------------------------------------- score
def pinned_weights() -> dict:
    j = json.loads(LADDER.read_text())
    return {"7-flag": j["models"]["7-flag"]["weights"],
            "3-voice": j["models"]["3-voice"]["weights"]}


def arm_flags(rec: dict, arm: str, base: dict) -> list[str]:
    """The flag multiset each arm scores this claim on."""
    if arm == "A":
        return [d["flag"] for d in base["armA"]]
    if arm == "A_filt":
        return [d["flag"] for d in base["armA"]
                if not circ_reason(d["url"], base["key"])]
    if arm == "B_p1":       # page-1 only: the document-count shortfall, uncompensated
        return [d["flag"] for d in rec["docs"] if d["in_b"] and d.get("page", 1) == 1]
    if arm == "B":          # page-2 backfilled to KEEP: count-matched to A and C
        return [d["flag"] for d in rec["docs"] if d["in_b"]]
    if arm == "B_rd":       # B minus reader-detected fact-checks
        return [d["flag"] for d in rec["docs"] if d["in_b"] and not d.get("fc_detect")]
    if arm == "C":
        return [d["flag"] for d in rec["docs"] if d["in_c"]]
    raise ValueError(arm)


ARMS = ("A", "A_filt", "B_p1", "B", "B_rd", "C")


def count_matrix(flag_lists: list[list[str]], model: str) -> np.ndarray:
    chans = list(FLAGS) if model == "7-flag" else ["support", "silent", "refute"]
    idx = {c: i for i, c in enumerate(chans)}
    C = np.zeros((len(chans), len(flag_lists)))
    for j, fl in enumerate(flag_lists):
        for f in fl:
            C[idx[f if model == "7-flag" else VOICE[f]], j] += 1
    return C


def fit_w(C, y, mid, idx):
    keep = idx[~mid[idx]]
    p, n = keep[y[keep] == 1], keep[y[keep] == 0]
    cT, cF = C[:, p].sum(1), C[:, n].sum(1)
    K = C.shape[0]
    return np.log(((cT + 1) / (cT.sum() + K)) / ((cF + 1) / (cF.sum() + K)))


def oof_scores(C, y, mid, fold):
    s = np.empty(C.shape[1])
    allidx = np.arange(C.shape[1])
    for k in range(fit_urn.K_FOLDS):
        tr, te = allidx[fold != k], allidx[fold == k]
        if len(te):
            s[te] = fit_w(C, y, mid, tr) @ C[:, te]
    return s


def auc_np(s_pos, s_neg):
    neg = np.sort(s_neg)
    lo = np.searchsorted(neg, s_pos, side="left")
    hi = np.searchsorted(neg, s_pos, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(s_pos) * len(neg)))


def recall_at_fpr(s, y, budget=0.02):
    return fit_urn.recall_at_fpr(sorted(zip(s.tolist(), y.tolist())), budget)


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _iso(d: str) -> str | None:
    """Serper's human date ("Mar 3, 2024", "2 days ago") -> ISO, or None. Relative
    dates resolve against today, which is on the late side of the true publication
    date and so can only make the post-dated count CONSERVATIVE, never inflated."""
    d = (d or "").strip()
    if m := re.match(r"^(\d{4})-(\d{2})-(\d{2})", d):
        return m.group(0)
    if m := re.match(r"^([A-Za-z]{3})\w*\s+(\d{1,2}),?\s+(\d{4})$", d):
        mo = _MONTHS.get(m.group(1).lower())
        return f"{m.group(3)}-{mo:02d}-{int(m.group(2)):02d}" if mo else None
    if m := re.match(r"^([A-Za-z]{3})\w*\s+(\d{4})$", d):
        mo = _MONTHS.get(m.group(1).lower())
        return f"{m.group(2)}-{mo:02d}-01" if mo else None
    if re.match(r"^\d+\s+(second|minute|hour|day|week|month|year)s?\s+ago$", d, re.I):
        import datetime
        n = int(d.split()[0])
        unit = d.split()[1].rstrip("s").lower()
        days = {"second": 0, "minute": 0, "hour": 0, "day": 1, "week": 7,
                "month": 30, "year": 365}[unit] * n
        return (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
    return None


def ci(v):
    a = np.sort(np.asarray(v))
    return float(a[int(0.025 * len(a))]), float(a[int(0.975 * len(a))])


def report() -> None:
    rows = {json.loads(l)["key"]: json.loads(l) for l in SAMPLE.open()}
    _load_hosts(list(rows.values()))
    res = [json.loads(l) for l in RESULTS.open()]
    res = [r for r in res if r["key"] in rows]
    W = pinned_weights()

    gold = [r for r in res if r["corpus"] == "gold"]
    tlr = [r for r in res if r["corpus"] == "timeline"]
    print(f"\npopulation: gold {len(gold)} (T {sum(1 for r in gold if r['y']==1)} / "
          f"F {sum(1 for r in gold if r['y']==0)}) | timeline {len(tlr)}")

    # ---- per-arm document accounting and flag distribution
    print("\n### documents per claim and flag distribution")
    hdr = f"{'arm':7s} {'corpus':9s} {'docs/claim':>10s}  " + "  ".join(f"{f:>5s}" for f in FLAGS)
    print(hdr)
    dist = {}
    for arm in ARMS:
        for name, sub in (("gold", gold), ("timeline", tlr)):
            fls = [arm_flags(r, arm, rows[r["key"]]) for r in sub]
            n = sum(len(f) for f in fls)
            c = collections.Counter(f for fl in fls for f in fl)
            dist[(arm, name)] = (n / max(len(sub), 1), c, n)
            print(f"{arm:7s} {name:9s} {n/max(len(sub),1):10.2f}  " +
                  "  ".join(f"{100*c[f]/max(n,1):5.1f}" for f in FLAGS))

    # ---- what the circularity filter removed, rule by rule
    print("\n### arm B exclusions (of the ceiling-off retrieved lists)")
    dr = collections.Counter()
    ndocs = 0
    for r in res:
        for d in r["docs"]:
            ndocs += 1
            dr[d["circ"] or "kept"] += 1
    print("  " + "  ".join(f"{k}={v}" for k, v in dr.most_common()))
    det = collections.Counter()
    for r in res:
        for d in r["docs"]:
            if d["in_b"]:
                det[d.get("fc_detect")] += 1
    print(f"  reader fc_detect on arm-B documents: true={det[True]} false={det[False]} "
          f"unknown={det[None]} ({100*det[True]/max(sum(det.values()),1):.1f}% of B)")

    # ---- reuse: how much of the ceiling-OFF set the ceiling-ON run had already read.
    # This is a finding in its own right: it says whether the ceiling reorders the top
    # of the list or fetches a different list.
    print("\n### read reuse (documents already read under the ceiling, reused verbatim)")
    print(f"  {'arm':7s} {'corpus':9s} {'docs':>6s} {'reused':>7s} {'new':>6s} {'reuse %':>8s}")
    for arm in ("B", "C"):
        for name, sub in (("gold", gold), ("timeline", tlr)):
            k = "in_b" if arm == "B" else "in_c"
            dd = [d for r in sub for d in r["docs"] if d[k]]
            ru = sum(1 for d in dd if d.get("reused"))
            print(f"  {arm:7s} {name:9s} {len(dd):6d} {ru:7d} {len(dd)-ru:6d} "
                  f"{100*ru/max(len(dd),1):7.1f}%")
    alld = [d for r in res for d in r["docs"]]
    ru = sum(1 for d in alld if d.get("reused"))
    print(f"  union   ALL       {len(alld):6d} {ru:7d} {len(alld)-ru:6d} "
          f"{100*ru/max(len(alld),1):7.1f}%   <- documents actually paid for: {len(alld)-ru}")

    # ---- gold AUC, pinned and refit, paired bootstrap
    print("\n### gold: AUC out of fold and recall at 2% FPR")
    y = np.array([r["y"] for r in gold])
    mid = np.array([bool(r["mid"]) for r in gold])
    fold = np.array([r["fold"] for r in gold])
    Cs = {(arm, m): count_matrix([arm_flags(r, arm, rows[r["key"]]) for r in gold], m)
          for arm in ARMS for m in ("7-flag", "3-voice")}
    chans = {"7-flag": list(FLAGS), "3-voice": ["support", "silent", "refute"]}
    pts = {}
    for m in ("7-flag", "3-voice"):
        wv = np.array([W[m][c] for c in chans[m]])
        print(f"\n  -- {m}")
        print(f"     {'arm':8s} {'AUC pinned':>11s} {'R@2%':>7s} | {'AUC refit':>10s} {'R@2%':>7s}")
        for arm in ARMS:
            C = Cs[(arm, m)]
            sp = wv @ C
            sr = oof_scores(C, y, mid, fold)
            ap, ar = auc_np(sp[y == 1], sp[y == 0]), auc_np(sr[y == 1], sr[y == 0])
            rp = recall_at_fpr(sp, y)[0]
            rr = recall_at_fpr(sr, y)[0]
            pts[(arm, m)] = (ap, ar, rp, rr)
            print(f"     {arm:8s} {ap:11.4f} {rp:7.3f} | {ar:10.4f} {rr:7.3f}")

    print(f"\n  paired bootstrap, {BOOT_REPS} reps, seed {BOOT_SEED}, resampled within gold class")
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    reps = {k: [] for k in [(a, m, w) for a in ARMS for m in ("7-flag", "3-voice")
                            for w in ("pin", "refit")]}
    for _ in range(BOOT_REPS):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy, mm, ff = y[idx], mid[idx], fold[idx]
        for m in ("7-flag", "3-voice"):
            wv = np.array([W[m][c] for c in chans[m]])
            for arm in ARMS:
                Cm = Cs[(arm, m)][:, idx]
                sp = wv @ Cm
                reps[(arm, m, "pin")].append(auc_np(sp[yy == 1], sp[yy == 0]))
                s = np.empty(len(idx))
                for k in range(fit_urn.K_FOLDS):
                    tr, te = np.where(ff != k)[0], np.where(ff == k)[0]
                    if len(te):
                        s[te] = fit_w(Cm, yy, mm, tr) @ Cm[:, te]
                reps[(arm, m, "refit")].append(auc_np(s[yy == 1], s[yy == 0]))
    for m in ("7-flag", "3-voice"):
        for w in ("pin", "refit"):
            base = np.array(reps[("A", m, w)])
            print(f"\n     {m} / {w}   arm A AUC {np.mean(base):.4f} "
                  f"[{ci(base)[0]:.4f}, {ci(base)[1]:.4f}]")
            for arm in ARMS[1:]:
                d = np.array(reps[(arm, m, w)]) - base
                lo, hi = ci(d)
                print(f"       d({arm} - A) {np.mean(d):+.4f}  [{lo:+.4f}, {hi:+.4f}]"
                      f"{'  *' if lo > 0 or hi < 0 else ''}")

    # ---- timeline: no gold labels, so the transfer test is the false-alarm side
    print("\n### timeline transfer (no labels there: gold-FALSE vs timeline)")
    print("    positives = timeline claims (the ~95%-true production-like draw),")
    print("    negatives = the gold-FALSE claims of the same run.")
    gf = [r for r in gold if r["y"] == 0]
    for m in ("7-flag", "3-voice"):
        wv = np.array([W[m][c] for c in chans[m]])
        print(f"\n  -- {m}  (pinned weights)")
        print(f"     {'arm':8s} {'AUC':>8s} {'R@2%FPR':>9s} {'tl flag rate @A thr':>21s}")
        thrA = None
        for arm in ARMS:
            st = wv @ count_matrix([arm_flags(r, arm, rows[r["key"]]) for r in tlr], m)
            sf = wv @ count_matrix([arm_flags(r, arm, rows[r["key"]]) for r in gf], m)
            s = np.concatenate([st, sf])
            yy = np.concatenate([np.ones(len(st)), np.zeros(len(sf))])
            a = auc_np(st, sf)
            rec, _, thr = recall_at_fpr(s, yy)
            if arm == "A":
                thrA = thr
            rate = float((st <= thrA).mean())
            print(f"     {arm:8s} {a:8.4f} {rec:9.3f} {rate:21.3f}")

    # ---- paired bootstrap on the TRANSFER, resampled within group. The gold AUC
    # gain can be an artifact of how gold was built (both classes were selected
    # because a fact-checker wrote about them, so both are guaranteed post-dated
    # coverage); the timeline carries no such guarantee. If the gain does not
    # survive here it does not transfer, and the report has to say so.
    print(f"\n  paired bootstrap on the transfer, {BOOT_REPS} reps, seed {BOOT_SEED}")
    rng2 = np.random.default_rng(BOOT_SEED)
    Ct = {(a, m): count_matrix([arm_flags(r, a, rows[r["key"]]) for r in tlr], m)
          for a in ARMS for m in ("7-flag", "3-voice")}
    Cf = {(a, m): count_matrix([arm_flags(r, a, rows[r["key"]]) for r in gf], m)
          for a in ARMS for m in ("7-flag", "3-voice")}
    treps = {(a, m, k): [] for a in ARMS for m in ("7-flag", "3-voice")
             for k in ("auc", "rec")}
    for _ in range(BOOT_REPS):
        it = rng2.choice(len(tlr), len(tlr))
        if_ = rng2.choice(len(gf), len(gf))
        for m in ("7-flag", "3-voice"):
            wv = np.array([W[m][c] for c in chans[m]])
            for a in ARMS:
                st, sf = (wv @ Ct[(a, m)])[it], (wv @ Cf[(a, m)])[if_]
                treps[(a, m, "auc")].append(auc_np(st, sf))
                sc = np.concatenate([st, sf])
                yy = np.concatenate([np.ones(len(st)), np.zeros(len(sf))])
                treps[(a, m, "rec")].append(recall_at_fpr(sc, yy)[0])
    for m in ("7-flag", "3-voice"):
        for k, lab in (("auc", "AUC"), ("rec", "recall@2%FPR")):
            base = np.array(treps[("A", m, k)])
            print(f"\n     {m} / {lab}   arm A {np.mean(base):.4f} "
                  f"[{ci(base)[0]:.4f}, {ci(base)[1]:.4f}]")
            for a in ARMS[1:]:
                d = np.array(treps[(a, m, k)]) - base
                lo, hi = ci(d)
                print(f"       d({a} - A) {np.mean(d):+.4f}  [{lo:+.4f}, {hi:+.4f}]"
                      f"{'  *' if lo > 0 or hi < 0 else ''}")

    # ---- mechanism: the gain must come from documents the ceiling was refusing.
    print("\n### mechanism: are arm-B's refuting documents post-dated?")
    print(f"  {'corpus':9s} {'refuting B docs':>16s} {'new (not reused)':>17s} "
          f"{'dated > ceiling':>16s} {'undated':>8s}")
    for name, sub in (("gold", gold), ("timeline", tlr)):
        ref = [(r, d) for r in sub for d in r["docs"]
               if d["in_b"] and d.get("flag") in ("1", "2")]
        new = [(r, d) for r, d in ref if not d.get("reused")]
        post = sum(1 for r, d in new if (d.get("date") or "") and r["ceiling"]
                   and _iso(d["date"]) and _iso(d["date"]) > r["ceiling"])
        und = sum(1 for r, d in new if not (d.get("date") or "").strip())
        print(f"  {name:9s} {len(ref):16d} {len(new):17d} {post:16d} {und:8d}")

    # timeline refute mass, the quantity the ceiling is supposed to be protecting
    print("\n### refuting documents per claim (the thing a ceiling-off gain must come from)")
    print(f"  {'arm':8s} {'gold-F':>9s} {'gold-T':>9s} {'timeline':>9s}")
    for arm in ARMS:
        def refm(sub):
            v = [sum(1 for f in arm_flags(r, arm, rows[r["key"]]) if f in ("1", "2"))
                 for r in sub]
            return sum(v) / max(len(v), 1)
        print(f"  {arm:8s} {refm([r for r in gold if r['y']==0]):9.3f} "
              f"{refm([r for r in gold if r['y']==1]):9.3f} {refm(tlr):9.3f}")

    print(f"\ntotal LLM spend recorded in results: ${sum(r['cost'] for r in res):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--agree", type=int, default=0,
                    help="re-read N saved gold docs, report agreement")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="claims this batch (resumable)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.sample:
        build_sample()
    if a.run:
        run(a.workers, a.smoke, a.limit)
    if a.inspect:
        _load_hosts([json.loads(l) for l in SAMPLE.open()])
        smoke_inspect()
    if a.agree:
        agree(a.agree, a.workers)
    if a.report:
        report()
