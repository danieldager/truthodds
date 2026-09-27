"""Three retrieval arms aimed at the all-irrelevant block (clog/280826, 11:40).

The block: every retrieved document read I, so zero refuting documents, so under
the "never flag on silence" rule the claim can NEVER be flagged. 27.9% of fc-gold
FALSE (494) and 17.0% of the CN-false urn (335). `nonexistent_event` is out of
scope here (a separate probe owns it). These arms take the next two taxonomy
codes plus the UGC question.

  spec    over_generic. The production query prompt caps at 10 words, tells the
          model to "ignore secondary details", and discourages quotes. On a
          micro-incident that strips exactly the tokens that identify it. The arm
          inverts those three clauses and nothing else.
  nodate  wrong_date. The production query TEXT frequently carries a literal year
          or month (41.8% CN / 44.5% gold of all-irrelevant queries) which the
          model derives from CLAIM DATE — the POST's date, not the event's. That
          token is a hard lexical filter on top of a `tbs` ceiling that already
          bounds the window, and the taxonomy says it is often simply wrong. The
          arm reissues the SAME query with the date token deleted from the text.
          THE CEILING IS UNCHANGED, so the eval's no-future-evidence guarantee is
          untouched — this is the one date fix that cannot break it.
  ugc     the UGC blocklist. 111 of 335 CN all-irrelevant notes cite ONLY
          blocklisted UGC (x.com 67, youtube 22, instagram 21, twitter 17,
          tiktok 10), twice the control rate. The arm unblocks UGC but KEEPS the
          origin-source exclusion, at the only granularity that leaves anything
          to test: the disputed POST's own URL is dropped, so a post can never be
          evidence for itself, while other posts stay reachable. (Domain-level
          origin exclusion is what `search(exclude_domains=...)` does and is kept
          verbatim for every non-UGC origin; for a CN claim the origin domain IS
          x.com, so keeping it domain-wide would delete the arm.)

Every arm is scored on the ONE number that matters: not documents added, but
claims moved from zero-refute to at-least-one-refute. The two arms tested on
2026-08-27 both added documents and moved nothing (60/71 and 170/191 new
documents still read I).

    uv run python -m eval.scripts.build_eval.retrieval_recovery_arms --sample
    uv run python -m eval.scripts.build_eval.retrieval_recovery_arms --run --arm spec --smoke 10
    uv run python -m eval.scripts.build_eval.retrieval_recovery_arms --report

Nothing under eval/data/urn_runs/ is read-modified or written. Output:
eval/data/retrieval_recovery/.
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

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from pipeline.config import FACT_CHECK_DOMAINS, SCRAPE_BLOCKLIST  # noqa: E402
from pipeline.credibility import TRUSTED_FACTCHECKERS  # noqa: E402
from pipeline.search import (  # noqa: E402
    SERPER_ENDPOINT, SearchError, _drop_domains, _request, _tbs_date_ceiling,
    scrape, search, serper_headers, serper_parse)
from pipeline.verify_tweet_claims import _domain_of  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
    _serper_gate, aggregate_reads, llm, read_doc, select_regions)
from eval.scripts.build_eval.refute_verify import w7  # noqa: E402
from eval.scripts.build_eval import fit_urn  # noqa: E402

OUT_DIR = SRC / "eval/data/retrieval_recovery"
SAMPLE = OUT_DIR / "sample.jsonl"
TAX = SRC / "eval/data/retrieval_forensics/taxonomy.jsonl"
C2 = SRC / "eval/data/urn_runs/c2_false"
E1 = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
TL = SRC / "eval/data/urn_runs/true_timeline/scores.jsonl"
FC_DOMS = set(FACT_CHECK_DOMAINS) | {d for d in TRUSTED_FACTCHECKERS if "/" not in d}
FLAGS = set("54321XI")
SEED = 20260828
FLAG_EDGE = -4.05
BUDGET_CAP = 4.00          # hard USD stop, checked after every claim

# per-population target sizes for the full run (--smoke overrides)
N_CTRL = 100

# The production prompt's three specificity-losing clauses, inverted. Everything
# else about the task is held fixed, so the only moving part is how much of the
# claim survives into the query.
SPEC_QUERY_SYS = (
    "You write ONE web search query to check a factual claim. The claim describes a SPECIFIC "
    "incident, and the query must identify THAT incident rather than the general topic it "
    "belongs to. Carry over every proper noun, number, quantity, place and distinctive noun "
    "phrase in the claim: those details are the only thing separating this incident from the "
    "topic. Never abstract, generalise or summarise, and never replace a specific thing with "
    "its category. If the claim contains distinctive wording or a quotation, put that wording "
    "in double quotes. Prefer a longer, more specific query over a shorter, vaguer one; there "
    "is no word limit. The query should surface independent reporting or primary evidence, not "
    "commentary about whether the claim is true. "
    "Respond JSON only: {\"query\": \"...\"}")

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_ISO = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_MONTH = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b\.?", re.I)
_STATUS = re.compile(r"/status/(\d+)")
_URL_RE = re.compile(r'https?://[^\s<>"\')\]]+')


def has_date_token(q: str) -> bool:
    q = q or ""
    return bool(_ISO.search(q) or _YEAR.search(q) or _MONTH.search(q))


def strip_dates(query: str, claim: str) -> str:
    """Delete date tokens the query ADDED. A token that also appears in the claim
    text is part of the assertion ("Gorbachev was assured in 1990") and is kept —
    stripping it would change the proposition, not just the retrieval window."""
    claim_l = (claim or "").lower()
    out = query or ""
    out = _ISO.sub(lambda m: m.group(0) if m.group(0) in claim_l else " ", out)
    out = _MONTH.sub(lambda m: m.group(0) if m.group(0).lower().rstrip(".") in claim_l else " ", out)
    out = _YEAR.sub(lambda m: m.group(0) if m.group(0) in claim_l else " ", out)
    # an orphaned day number left behind by a removed month ("February 28" -> " 28")
    out = re.sub(r"\s+\d{1,2}(st|nd|rd|th)?\b(?!\s*[-/])", lambda m: m.group(0)
                 if m.group(0).strip() in claim_l else " ", out)
    return re.sub(r"\s{2,}", " ", out).strip(" ,-")


def dom(u: str) -> str:
    return urlsplit(u).netloc.lower().removeprefix("www.").removeprefix("m.")


def blocked(d: str) -> bool:
    return d in SCRAPE_BLOCKLIST or any(d.endswith("." + b) for b in SCRAPE_BLOCKLIST)


def band(s: float, n_refute: int) -> bool:
    """Flaggable = crosses the log-odds edge AND at least one refuting read.
    The second clause is the "never flag on silence" rule: 10 irrelevant reads
    score -4.02, a hair off the edge, so score alone would flag pure silence."""
    return s <= FLAG_EDGE and n_refute > 0


# --------------------------------------------------------------------------- sample
def _allirr(r: dict) -> bool:
    ds = [d["read"]["direction"] for d in r["results"] if d["read"]["direction"] in FLAGS]
    return bool(ds) and all(d == "I" for d in ds)


def _base(r: dict, w: dict) -> dict:
    docs = [d for d in r["results"] if d["read"]["direction"] in FLAGS]
    flags = [d["read"]["direction"] for d in docs]
    # url -> saved flag, so REPLACE semantics can reuse the old read for a URL the new
    # query returns again (read-v5 at temperature 0: re-reading it would cost money to
    # reproduce the same answer).
    return {"s7": round(sum(w.get(f, 0.0) for f in flags), 4),
            "base_flags": flags, "base_urls": [d.get("url") for d in docs],
            "base_map": {d.get("url"): d["read"]["direction"] for d in docs},
            "n_refute_base": sum(1 for f in flags if f in ("1", "2"))}


def load_corpora(w: dict):
    excl = {e["claim"][:80] for e in json.loads((C2 / "fit_exclusions.json").read_text())}
    notes = {}
    for p in ("match.jsonl", "match_ext.jsonl"):
        for line in (C2 / p).open():
            j = json.loads(line)
            notes[j["post_id"]] = j.get("note") or ""
    cn = {}
    for p in ("scores.jsonl", "scores_ext.jsonl"):
        for line in (C2 / p).open():
            r = json.loads(line)
            if ((r.get("claim_resolved") or r.get("claim_text") or "")[:80]) in excl:
                continue
            if not [d for d in r["results"] if d["read"]["direction"] in FLAGS]:
                continue
            cn[r["post_id"]] = r
    axis = fit_urn.load_judged_axis()
    gold = {}
    for line in E1.open():
        r = json.loads(line)
        if r.get("veracity") not in (1, 2, 3):
            continue
        if (r.get("rating_subtype") or "?") == "mixed":
            continue
        if (axis.get(r["review_url"]) or "untagged") == fit_urn.MEDIA_AXIS:
            continue
        if not [d for d in r["results"] if d["read"]["direction"] in FLAGS]:
            continue
        gold[r["review_url"]] = r
    tl = {}
    for line in TL.open():
        r = json.loads(line)
        if not [d for d in r["results"] if d["read"]["direction"] in FLAGS]:
            continue
        tl[r["post_id"]] = r
    return cn, gold, tl, notes


def row_of(r: dict, pop: str, arm: str, w: dict, corpus: str, extra: dict | None = None) -> dict:
    out = {"pop": pop, "arm": arm, "corpus": corpus,
           "key": r["review_url"], "post_id": r.get("post_id"),
           "claim": r.get("claim_resolved") or r.get("claim_text") or "",
           "claim_date": r.get("claim_date_shown"), "ceiling": r.get("ceiling"),
           "publisher_site": r.get("publisher_site") or "x.com",
           "query_prod": r.get("query") or "", **_base(r, w)}
    out.update(extra or {})
    return out


def build_sample() -> None:
    w = w7()
    rng = random.Random(SEED)
    cn, gold, tl, notes = load_corpora(w)
    tax = [json.loads(l) for l in TAX.open()]
    rows = []

    def lookup(t):
        arm, _, ident = t["id"].partition(":")
        if arm.startswith("cn"):
            return cn.get(ident), "cn"
        return gold.get(ident), "gold"

    # --- arms spec / nodate: the taxonomy-labelled claims for the two target codes
    for t in tax:
        rec, corpus = lookup(t)
        if rec is None:
            continue
        allirr = t["arm"].endswith("allirr")
        if t["code"] == "over_generic":
            rows.append(row_of(rec, "spec_target" if allirr else "spec_ctrl", "spec", w,
                               corpus, {"tax_code": t["code"]}))
        elif t["code"] == "wrong_date":
            if not has_date_token(rec.get("query") or ""):
                continue                       # nothing for this arm to remove
            stripped = strip_dates(rec["query"], rec.get("claim_resolved")
                                   or rec.get("claim_text") or "")
            if stripped == (rec.get("query") or "").strip():
                continue
            rows.append(row_of(rec, "nodate_target" if allirr else "nodate_ctrl", "nodate", w,
                               corpus, {"tax_code": t["code"], "query_stripped": stripped}))

    # --- arm ugc: CN all-irrelevant claims whose note cites ONLY blocklisted UGC
    for pid, rec in cn.items():
        urls = [u.rstrip(".,;:)") for u in _URL_RE.findall(notes.get(pid, ""))]
        if not urls or not all(blocked(dom(u)) for u in urls):
            continue
        if not _allirr(rec):
            continue
        rows.append(row_of(rec, "ugc_target", "ugc", w, "cn",
                           {"note_urls": urls,
                            "note_self_cite": any(_STATUS.search(u) and
                                                  _STATUS.search(u).group(1) == pid for u in urls)}))

    # --- controls: unchanged-baseline populations, to price the harm of each change
    cn_ctrl = [r for r in cn.values() if not _allirr(r)]
    rng.shuffle(cn_ctrl)
    for r in cn_ctrl[:N_CTRL]:
        rows.append(row_of(r, "ugc_ctrl", "ugc", w, "cn"))
    tl_all = list(tl.values())
    rng.shuffle(tl_all)
    for r in tl_all[:N_CTRL]:
        rows.append(row_of(r, "ugc_true", "ugc", w, "tl"))
    for r in tl_all[N_CTRL:2 * N_CTRL]:
        rows.append(row_of(r, "spec_true", "spec", w, "tl"))
    nd_true = [r for r in tl_all[2 * N_CTRL:] if has_date_token(r.get("query") or "")]
    for r in nd_true[:N_CTRL]:
        s = strip_dates(r["query"], r.get("claim_resolved") or r.get("claim_text") or "")
        if s == (r.get("query") or "").strip():
            continue
        rows.append(row_of(r, "nodate_true", "nodate", w, "tl", {"query_stripped": s}))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLE.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    c = collections.Counter(r["pop"] for r in rows)
    for pop in sorted(c):
        sub = [r for r in rows if r["pop"] == pop]
        print(f"  {pop:15s} n={c[pop]:4d}  mean s7 {sum(r['s7'] for r in sub)/len(sub):+7.2f}  "
              f"already >=1 refute {sum(1 for r in sub if r['n_refute_base']):3d}  "
              f"corpora {dict(collections.Counter(r['corpus'] for r in sub))}")
    print(f"  TOTAL {len(rows)} claims -> {SAMPLE}")


# --------------------------------------------------------------------------- run
def spec_query(row: dict) -> str:
    qblock = [f"CLAIM: {row['claim']}"]
    if row.get("claim_date"):
        qblock.append(f"CLAIM DATE: {row['claim_date']}")
    for _ in range(3):
        try:
            q, c, _, _ = llm([{"role": "system", "content": SPEC_QUERY_SYS},
                              {"role": "user", "content": "\n".join(qblock)}], max_tokens=120)
            out = (q.get("query") or "").strip()[:300]
            if out:
                return out, c
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise RuntimeError("spec query-gen failed")


def search_unblocked(query: str, top_k: int, ceil: str | None,
                     exclude_domains: list[str], drop_status: str | None) -> list[dict]:
    """Serper with the UGC blocklist OFF. Deliberately bypasses `search()` rather
    than editing it: pipeline behaviour must be untouched, and `search()` applies
    SCRAPE_BLOCKLIST server-side (`-site:` operators) AND client-side
    (`serper_finalize`) with no way to disable either. Caller exclusions still
    apply, plus a URL-level drop of the disputed post itself."""
    payload = {"q": query + "".join(f" -site:{d}" for d in exclude_domains[:3]), "num": top_k}
    if ceil and (tbs := _tbs_date_ceiling(ceil)):
        payload["tbs"] = tbs
    data = _request("POST", SERPER_ENDPOINT, label="serper-ugc", json=payload,
                    headers=serper_headers(), timeout=15)
    res = serper_parse(data, top_k)
    if exclude_domains:
        res = _drop_domains(res, set(exclude_domains))
    if drop_status:
        res = [r for r in res
               if not (_STATUS.search(r["url"]) and _STATUS.search(r["url"]).group(1) == drop_status)]
    return res


def run_one(row: dict) -> dict:
    arm = row["arm"]
    ceil = row.get("ceiling") or None
    cost = 0.0
    if arm == "spec":
        query, c = spec_query(row)
        cost += c
    elif arm == "nodate":
        query = row["query_stripped"]
    else:
        query = row["query_prod"]

    hits = None
    for attempt in range(2):
        _serper_gate()
        try:
            if arm == "ugc":
                # origin-source exclusion KEPT: for a CN/timeline claim the origin is a
                # specific POST, dropped by status id; the publisher DOMAIN exclusion is
                # kept for any origin that is not itself the UGC platform under test.
                xd = [] if row["publisher_site"] in ("x.com", "twitter.com") \
                    else [row["publisher_site"]]
                hits = search_unblocked(query, 10, ceil, xd, row.get("post_id"))
            else:
                hits = search(query, 10, date_ceiling=ceil,
                              exclude_domains=[row["publisher_site"]],
                              min_results=0, provider="serper")
            break
        except SearchError:
            if attempt == 0:
                time.sleep(60)
            else:
                raise

    seen = set(row["base_urls"])
    repeated = [h.get("url") for h in (hits or [])[:10] if h.get("url") in seen]
    claim_block = (f"CLAIM: {row['claim']}\n"
                   f"(claimed on {row.get('claim_date') or 'unknown date'}; judge the "
                   f"document's bearing on this exact proposition)")
    docs = []
    for h in (hits or [])[:10]:
        url = h.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        d = _domain_of(url)
        fc = any(d == f or d.endswith("." + f) for f in FC_DOMS)
        entry = {"url": url, "domain": d, "fc_domain": fc, "date": h.get("date"),
                 "ugc": blocked(dom(url))}
        if fc and not (h.get("date") or "").strip():
            entry.update(direction=None, read_status="fc-undated")
            docs.append(entry)
            continue
        text = h.get("content") or scrape(url)          # scrape() refuses UGC -> snippet
        prov = "scrape"
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
    return {"key": row["key"], "pop": row["pop"], "arm": arm, "corpus": row["corpus"],
            "claim": row["claim"], "query_prod": row["query_prod"], "query_new": query,
            "s7_base": row["s7"], "n_refute_base": row["n_refute_base"],
            "n_hits": len(hits or []), "repeated_flags": [row["base_map"][u] for u in repeated],
            "note_urls": row.get("note_urls"), "n_new": len(docs), "docs": docs,
            "cost": cost}


def out_path(arm: str) -> Path:
    return OUT_DIR / f"{arm}_results.jsonl"


def run(arm: str, workers: int, smoke: int, pops: list[str] | None) -> None:
    rows = [json.loads(l) for l in SAMPLE.open() if json.loads(l)["arm"] == arm]
    if pops:
        rows = [r for r in rows if r["pop"] in pops]
    path = out_path(arm)
    done = set()
    if path.exists():
        done = {json.loads(l)["key"] + "|" + json.loads(l)["pop"] for l in path.open()}
    todo = [r for r in rows if r["key"] + "|" + r["pop"] not in done]
    if smoke:                       # equal-sized draw from each population
        rng = random.Random(SEED)
        by = collections.defaultdict(list)
        for r in todo:
            by[r["pop"]].append(r)
        todo = []
        for pop in sorted(by):
            rng.shuffle(by[pop])
            todo += by[pop][:smoke]
    if not todo:
        print(f"arm {arm}: nothing to do")
        return
    print(f"arm {arm}: {len(todo)} claims | {workers} workers | "
          f"{dict(collections.Counter(r['pop'] for r in todo))}", flush=True)
    lock = threading.Lock()
    n, cost, t0 = [0], [0.0], time.time()
    with path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
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
                if cost[0] > BUDGET_CAP:
                    raise SystemExit(f"BUDGET CAP ${BUDGET_CAP} breached at {n[0]} claims")
                if n[0] % 10 == 0 or n[0] == len(todo):
                    el = time.time() - t0
                    rate = n[0] / max(el, 0.001)
                    print(f"  {n[0]}/{len(todo)}  ${cost[0]:.3f}  {rate*60:.1f} claims/min  "
                          f"ETA {(len(todo)-n[0])/max(rate,1e-6)/60:.1f}m", flush=True)
    print(f"done arm {arm}: {n[0]} claims, LLM ${cost[0]:.3f}, {n[0]} serper credits, "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


# --------------------------------------------------------------------------- report
def union_replace(r: dict, w: dict) -> tuple[float, int, float, int]:
    """(s7, n_refute) under UNION (the new query is ADDED to the production one) and
    under REPLACE (the new query IS the production one). Replace is the realistic
    shape for a prompt or blocklist change; union is what the 27/08 arms reported,
    kept so the numbers are comparable. Under replace a URL the new query returns
    again keeps its saved read."""
    add = [d for d in r["docs"] if d.get("direction")]
    su = r["s7_base"] + sum(w.get(d["direction"], 0.0) for d in add)
    nu = r["n_refute_base"] + sum(1 for d in add if d["direction"] in ("1", "2"))
    rep = r.get("repeated_flags") or []
    sr = sum(w.get(f, 0.0) for f in rep) + sum(w.get(d["direction"], 0.0) for d in add)
    nr = sum(1 for f in rep if f in ("1", "2")) + sum(1 for d in add
                                                      if d["direction"] in ("1", "2"))
    return su, nu, sr, nr


def report(arm: str | None) -> None:
    w = w7()
    for a in (["spec", "nodate", "ugc"] if arm is None else [arm]):
        p = out_path(a)
        if not p.exists():
            continue
        res = [json.loads(l) for l in p.open()]
        per = collections.defaultdict(list)
        for r in res:
            per[r["pop"]].append(r)
        print(f"\n################ ARM {a}  (n={len(res)})")
        for pop in sorted(per):
            rows = per[pop]
            n = len(rows)
            nd = sum(r["n_new"] for r in rows)
            read = [d for r in rows for d in r["docs"] if d.get("direction")]
            dirs = collections.Counter(d["direction"] for d in read)
            ref = [d for d in read if d["direction"] in ("1", "2")]
            ugcd = [d for r in rows for d in r["docs"] if d.get("ugc")]
            ugc_ref = [d for d in ugcd if d.get("direction") in ("1", "2")]
            s_u, s_r, moved, fl_u, fl_r, lost_r, empty = [], [], 0, 0, 0, 0, 0
            for r in rows:
                su, nu, sr, nr = union_replace(r, w)
                s_u.append(su)
                s_r.append(sr)
                if r["n_refute_base"] == 0 and nu > 0:
                    moved += 1
                was = band(r["s7_base"], r["n_refute_base"])
                fl_u += (not was) and band(su, nu)
                fl_r += (not was) and band(sr, nr)
                lost_r += was and (not band(sr, nr))
                empty += r.get("n_hits", 0) == 0
            print(f"\n  --- {pop}  n={n}")
            print(f"      new docs {nd} ({nd/n:.2f}/claim) | read {len(read)} | "
                  f"UGC docs {len(ugcd)} ({len(ugcd)/n:.2f}/claim) | empty result sets {empty}")
            print("      new reads: " + " ".join(f"{k}:{dirs[k]}" for k in
                                                 "54321XI" if dirs[k]))
            print(f"      REFUTING new docs {len(ref)} ({len(ref)/max(nd,1):.1%} of added)"
                  + (f" | of which UGC {len(ugc_ref)}" if ugcd else ""))
            print(f"      claims 0-refute -> >=1 refute: {moved}/{n} ({moved/n:.1%})")
            print(f"      mean s7 base {sum(r['s7_base'] for r in rows)/n:+.2f} -> "
                  f"union {sum(s_u)/n:+.2f} | replace {sum(s_r)/n:+.2f}")
            print(f"      NEWLY FLAGGABLE union {fl_u}/{n} ({fl_u/n:.1%}) | "
                  f"replace {fl_r}/{n} ({fl_r/n:.1%}) | flag LOST under replace {lost_r}")
        # the false-alarm ledger: flags gained on the TRUE population price the arm
        tgt = [k for k in per if k.endswith("_target")]
        tru = [k for k in per if k.endswith("_true")]
        if tgt and tru:
            def gained(pop, which):
                g = 0
                for r in per[pop]:
                    su, nu, sr, nr = union_replace(r, w)
                    s, nref = (su, nu) if which == "union" else (sr, nr)
                    g += (not band(r["s7_base"], r["n_refute_base"])) and band(s, nref)
                return g, len(per[pop])
            for which in ("union", "replace"):
                gt, nt = gained(tgt[0], which)
                gf, nf = gained(tru[0], which)
                print(f"  == {a} [{which}]: recall gain {gt}/{nt} = {gt/nt:.1%} on the block, "
                      f"false-alarm gain {gf}/{nf} = {gf/nf:.1%} on TRUE claims")


def show(arm: str, pop: str, k: int = 12) -> None:
    """Read the actual recovered documents — aggregates decide nothing here."""
    for l in out_path(arm).open():
        r = json.loads(l)
        if r["pop"] != pop:
            continue
        new = [d for d in r["docs"] if d.get("direction")]
        if not new:
            continue
        print("=" * 100)
        print("CLAIM:", r["claim"][:200])
        print("Q_old:", r["query_prod"])
        print("Q_new:", r["query_new"])
        if r.get("note_urls"):
            print("NOTE :", ", ".join(r["note_urls"][:3]))
        for d in new[:6]:
            print(f"  [{d['direction']}] {'UGC ' if d.get('ugc') else ''}{d['domain']:28s} "
                  f"{(d.get('reason') or '')[:70]}")
            print(f"      {(d.get('sents') or '')[:260]}")
        k -= 1
        if k <= 0:
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--show", nargs=2, metavar=("ARM", "POP"))
    ap.add_argument("--arm", default=None, choices=["spec", "nodate", "ugc"])
    ap.add_argument("--pops", default=None, help="comma-separated population filter")
    ap.add_argument("--smoke", type=int, default=0, help="claims per population")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.sample:
        build_sample()
    if a.run:
        assert a.arm, "--run needs --arm"
        run(a.arm, a.workers, a.smoke, a.pops.split(",") if a.pops else None)
    if a.report:
        report(a.arm)
    if a.show:
        show(a.show[0], a.show[1])
