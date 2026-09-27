"""The no-ceiling 2x2 on the CN-false urn: date ceiling x UGC block.

no_ceiling_arms.py answered "does dropping the date ceiling help?" on fc-gold and
the timeline. It gained +0.066 AUC on gold, did NOT transfer to the timeline, and
the driver was CLAIM AGE (median ceiling age 1,512 d on gold, 3 d on the timeline).
This script asks the same question on the CN-false urn, and adds Daniel's second
question: what if we stop blocking user-generated content?

A clean 2x2, so the two factors are not confounded:
  A   ceiling ON,  UGC blocked    production baseline = the SAVED urn reads, $0
  B   ceiling OFF, UGC blocked    Daniel's first question
  U   ceiling ON,  UGC unblocked  Daniel's second question
  BU  ceiling OFF, UGC unblocked  both
Plus contaminated upper bounds C (= B without the circularity filter) and
CU (= BU without it). Those are NOT real numbers; they are labelled as such.

THE TRUE SIDE. CN falses alone give no AUC, so the TRUE side is the timeline urn —
the existing two-urn evaluation (fit_two_urn / cross_ladder). BOTH corpora get the
same arm treatment in every arm; anything else makes the comparison meaningless.
The timeline is contaminated at eps, so every AUC is reported observed AND
contamination-corrected with cross_ladder's own correction,
AUC_true = (AUC_obs - eps/2)/(1 - eps), at the measured eps = 0.1022.

CIRCULARITY, worse here than on gold. CN labels come FROM community notes, so a
document written about this exact claim by the noter exists by construction, and
unblocking UGC puts x.com back in the result set — which makes the source post and
the note itself directly retrievable. Excluded, each rule counted separately:
  own_post    any URL carrying the claim's own post_id (the source post, quotes of
              it, and any mirror that keeps the status id)
  own_note    any URL carrying the claim's noteId
  note_cited  the note's OWN cited URLs, parsed out of the note summary
              (2,076 of 2,086 urn notes cite >=1 raw URL) — exact URL match after
              normalisation, NOT host match: notes cite reuters and wikipedia and
              blocking those hosts would be a different experiment
  note_mirror communitynotes/birdwatch mirror domains
  fc_domain   FACT_CHECK_DOMAINS + TRUSTED_FACTCHECKERS + the science.feedback.org
              gap no_ceiling_arms found (reused verbatim from that script)
  fc_path     syndicated fact-check SECTIONS on non-fc domains (URL-path rule)
  origin      publisher_site, server-side, in the UGC-BLOCKED arms only
ORIGIN IN THE UNBLOCKED ARMS. publisher_site is x.com for every CN and timeline
claim, so "exclude the origin" and "block UGC" are the same operator here. The
unblocked arms therefore admit the PLATFORM and exclude the POST: x.com is
retrievable, the claim's own post and note are not. Every drop is counted and the
report prints them.
Domain and id rules cannot catch a retweet of the rumour under a different status
id, or a syndicated fact-check on an unlisted host, so no_ceiling_arms' per-document
reader boolean ("is this document itself a fact-check of THIS claim") is kept and
every filtered arm is also reported with those documents removed (_rd), as a lower
bound.

READ REUSE. A read is a (claim, document) pair, so any URL with a saved read under
the same claim is reused verbatim from the urn run and never re-sent to the reader.

UGC SCRAPING. pipeline.scrape() refuses SCRAPE_BLOCKLIST domains and caches the
refusal, and its cache is production's. This script therefore NEVER lets a UGC URL
near disk_cache: blocklisted domains go through _ugc_fetch, which has its own cache
file under eval/data/no_ceiling_cn/. Non-UGC documents use production scrape().

NO PAGE-2 BACKFILL. Serper bills one credit per request and ignores `num` above 10,
so the circularity filter legitimately leaves the filtered arms a little short.
no_ceiling_arms measured the page-2 backfill and it moved gold AUC by 0.0000, so
the credits are not spent here; the document-count shortfall is reported instead.

  uv run python -m eval.scripts.build_eval.no_ceiling_cn --sample
  uv run python -m eval.scripts.build_eval.no_ceiling_cn --run --smoke 20
  uv run python -m eval.scripts.build_eval.no_ceiling_cn --inspect
  uv run python -m eval.scripts.build_eval.no_ceiling_cn --run --limit 60
  uv run python -m eval.scripts.build_eval.no_ceiling_cn --report

Nothing under eval/data/urn_runs/ is written. Output: eval/data/no_ceiling_cn/.
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

from pipeline.config import SCRAPE_BLOCKLIST  # noqa: E402
from pipeline.search import (  # noqa: E402
    SERPER_ENDPOINT, SearchError, _request, _tbs_date_ceiling, scrape,
    serper_headers, serper_parse)
from pipeline.verify_tweet_claims import _domain_of  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
    _serper_gate, aggregate_reads, read_doc, select_regions)
from eval.scripts.build_eval import fit_urn  # noqa: E402
from eval.scripts.build_eval import no_ceiling_arms as NCA  # noqa: E402

OUT_DIR = SRC / "eval/data/no_ceiling_cn"
SAMPLE = OUT_DIR / "sample.jsonl"
RESULTS = OUT_DIR / "results.jsonl"
UGC_CACHE = OUT_DIR / "ugc_scrape.jsonl"
C2 = SRC / "eval/data/urn_runs/c2_false"
TL = SRC / "eval/data/urn_runs/true_timeline"
CN_NOTES = SRC / "eval/data/community_notes/cn_gold.parquet"

FLAGS = NCA.FLAGS
FLAGSET = NCA.FLAGSET
VOICE = NCA.VOICE
SEED = 20260828
BOOT_REPS, BOOT_SEED = 2000, 707
TOP_K = 10
KEEP = 10
EPS = 0.1022          # timeline_eps_audit.py, folded estimator 38/372
LLM_CAP = 1.60        # hard USD stop on DeepInfra spend
SERPER_CAP = 1750     # hard stop on billed Serper requests (1 credit each)
N_PER_CORPUS = 250

_URL_RE = re.compile(r"https?://[^\s\)\]\},;\"']+")
_NOTE_PATH = re.compile(r"communitynotes|birdwatch|/notes?/", re.I)


def norm_url(u: str) -> str:
    """Scheme/host/path only, lowercased, no www/m, no query, no trailing slash."""
    try:
        p = urlsplit((u or "").strip())
    except ValueError:      # malformed url in a note summary; never matches anything
        return "\x00" + (u or "")
    h = p.netloc.lower().removeprefix("www.").removeprefix("m.")
    return h + "/" + (p.path or "").strip("/").lower()


def _path(u: str) -> str:
    try:
        return urlsplit(u or "").path or ""
    except ValueError:
        return ""


def is_ugc(dom: str) -> bool:
    return NCA.in_list(dom, SCRAPE_BLOCKLIST)


# --------------------------------------------------------------------------- sample
def note_cited_urls() -> dict[str, list[str]]:
    """noteId -> the URLs the note's own summary cites, normalised."""
    import polars as pl
    d = pl.read_parquet(CN_NOTES, columns=["noteId", "summary"])
    out = {}
    for nid, s in zip(d["noteId"].cast(pl.Utf8).to_list(), d["summary"].to_list()):
        us = _URL_RE.findall(s or "")
        if us:
            out[nid] = sorted({norm_url(u) for u in us})
    return out


def _docs(r: dict) -> list[dict]:
    return [d for d in (r.get("results") or [])
            if (d.get("read") or {}).get("direction") in FLAGSET]


def _row(r: dict, corpus: str, cited: list[str]) -> dict:
    docs = _docs(r)
    return {
        "key": f"{corpus}:{r['review_url']}",
        "corpus": corpus,
        "claim": r.get("claim_resolved") or r.get("claim_text") or "",
        "claim_date": r.get("claim_date_shown") or "",
        "ceiling": r.get("ceiling") or "",
        "publisher_site": r.get("publisher_site") or "x.com",
        "query": r.get("query") or "",
        "post_id": str(r.get("post_id") or ""),
        "note_id": str(r.get("noteId") or ""),
        "note_cited": cited,
        # fold on the CLAIM text, exactly as cross_ladder does for both urns
        "fold": NCA.fit_urn.fold_of(r.get("claim_resolved") or r.get("claim_text") or "",
                                    fit_urn.K_FOLDS),
        # arm A = the saved production reads, kept whole. `txt` is what the
        # production reader actually saw, so a REUSED document needs neither a
        # re-scrape nor a re-read.
        "armA": [{"url": d.get("url") or "", "flag": d["read"]["direction"],
                  "date": d.get("date"), "status": d.get("read_status"),
                  "txt": " ".join(d.get("sents") or [])[:1200]} for d in docs],
    }


def load_population() -> tuple[list[dict], list[dict]]:
    """The CURRENT default populations: CN with fit_exclusions + the media-provenance
    purge (majority rule, ON by default), timeline raw. Zero-doc claims dropped.
    Reproduces fit_two_urn.load_urn's membership exactly."""
    excl = {e["claim"][:80] for e in json.loads((C2 / "fit_exclusions.json").read_text())}
    for x in fit_urn.extra_exclusion_paths():
        excl |= {e["claim"][:80] for e in json.loads(Path(x).read_text())
                 if e.get("excluded", True)}
    cited = note_cited_urls()
    cn = []
    for p in (C2 / "scores.jsonl", C2 / "scores_ext.jsonl"):
        for line in p.open():
            r = json.loads(line)
            claim = r.get("claim_resolved") or r.get("claim_text") or ""
            if claim[:80] in excl or not _docs(r):
                continue
            cn.append(_row(r, "cn", cited.get(str(r.get("noteId") or ""), [])))
    tl = []
    for line in (TL / "scores.jsonl").open():
        r = json.loads(line)
        if not _docs(r):
            continue
        tl.append(_row(r, "tl", []))
    return cn, tl


def build_sample() -> None:
    cn, tl = load_population()
    print(f"CN-false urn {len(cn)}   timeline urn {len(tl)}")
    rng = random.Random(SEED)
    rng.shuffle(cn)
    rng.shuffle(tl)
    rows = [r for r in cn[:N_PER_CORPUS] + tl[:N_PER_CORPUS] if r["query"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SAMPLE.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for c in ("cn", "tl"):
        sub = [r for r in rows if r["corpus"] == c]
        nc = sum(1 for r in sub if r["note_cited"])
        print(f"  {c:3s} n={len(sub):4d}  mean armA docs "
              f"{sum(len(r['armA']) for r in sub)/max(len(sub),1):.2f}"
              f"  with note-cited urls {nc}")
    print(f"  TOTAL {len(rows)} -> {SAMPLE}")


# --------------------------------------------------------------------------- retrieval
def unblocked_page(query: str, ceiling: str | None, page: int = 1) -> list[dict]:
    """One Serper page with NO server-side social/UGC `-site:` operators and NO
    client-side blocklist drop. Bypasses search() entirely, so it can neither read
    nor write the production search cache (whose key folds in the blocklist tag)."""
    payload = {"q": query, "num": TOP_K}
    if page > 1:
        payload["page"] = page
    if ceiling and (tbs := _tbs_date_ceiling(ceiling)):
        payload["tbs"] = tbs
    data = _request("POST", SERPER_ENDPOINT, label="serper", json=payload,
                    headers=serper_headers(), timeout=15)
    return serper_parse(data, TOP_K)


def blocked_page(query: str, ceiling: str | None, xd: list[str]) -> list[dict]:
    """Production retrieval: socials/UGC excluded server-side within Google's
    operator budget, full blocklist enforced client-side, origin excluded."""
    from pipeline.search import search
    return search(query, TOP_K, date_ceiling=ceiling, exclude_domains=xd,
                  min_results=0, provider="serper")


def circ_reason(url: str, row: dict) -> str | None:
    """Why the circularity filter drops this document, or None. First rule wins;
    the rules are counted separately so the report can price each one."""
    n = norm_url(url)
    pid, nid = row.get("post_id") or "", row.get("note_id") or ""
    if pid and pid in url:
        return "own_post"
    if nid and nid in url:
        return "own_note"
    if n in set(row.get("note_cited") or []):
        return "note_cited"
    d = _domain_of(url)
    if NCA.in_list(d, NCA.NOTE_DOMAINS) or (_NOTE_PATH.search(_path(url))
                                            and d in ("x.com", "twitter.com")):
        return "note_mirror"
    if NCA.in_list(d, NCA.FC_BLOCK):
        return "fc_domain"
    if NCA._FC_PATH.search(_path(url)):
        return "fc_path"
    return None


# --------------------------------------------------------------------------- ugc fetch
_ugc_lock = threading.Lock()
_ugc_mem: dict[str, str | None] = {}


def _ugc_load() -> None:
    if UGC_CACHE.exists():
        for line in UGC_CACHE.open():
            try:
                o = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            _ugc_mem[o["url"]] = o.get("text")


def _ugc_fetch(url: str) -> str | None:
    """Fetch a BLOCKLISTED (UGC) url. Own cache file, own request — production's
    scrape() refuses these domains and caches the refusal in the SHARED disk cache,
    so it must never see them."""
    with _ugc_lock:
        if url in _ugc_mem:
            return _ugc_mem[url]
    text = None
    try:
        import random as _r

        import requests
        import trafilatura
        from pipeline.search import _USER_AGENTS
        r = requests.get(url, headers={"User-Agent": _r.choice(_USER_AGENTS)}, timeout=15)
        if r.ok:
            text = trafilatura.extract(r.text, include_comments=False, include_tables=True)
            text = (text or "").strip() or None
    except Exception:  # noqa: BLE001
        text = None
    with _ugc_lock:
        _ugc_mem[url] = text
        UGC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with UGC_CACHE.open("a") as f:
            f.write(json.dumps({"url": url, "text": text}) + "\n")
    return text


# --------------------------------------------------------------------------- read
def read_one(row: dict, h: dict, claim_block: str, saved: dict) -> tuple[dict, float]:
    """One document through the production read chain (scrape -> snippet fallback,
    prep-v7 regions, read-v5), plus no_ceiling_arms' fact-check reader boolean.
    A URL with a saved read UNDER THIS CLAIM is reused verbatim and never re-read."""
    url = h.get("url") or ""
    d = _domain_of(url)
    ugc = is_ugc(d)
    entry = {"url": url, "domain": d, "date": h.get("date"), "ugc": ugc,
             "circ": circ_reason(url, row)}
    cost = 0.0
    if url in saved:
        s = saved[url]
        entry.update(flag=s["flag"], read_status="reused", reused=True,
                     saved_status=s.get("status"), sents=(s.get("txt") or "")[:400])
        det, c = (True, 0.0) if s.get("status") == "fc-undated" else \
            NCA.fc_detect(row["claim"], d, s.get("txt") or "")
        entry["fc_detect"] = det
        return entry, c
    entry["reused"] = False
    fc_domain = NCA.in_list(d, NCA.FC_BLOCK)
    if fc_domain and not (h.get("date") or "").strip():
        # production's undated-fact-check refusal, verbatim
        entry.update(flag="I", read_status="fc-undated", fc_detect=True)
        return entry, cost
    text = h.get("content") or (_ugc_fetch(url) if ugc else scrape(url))
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
    seen = " ".join(s for _, sel in regions for s in sel)
    entry["sents"] = seen[:400]
    det, c2 = NCA.fc_detect(row["claim"], d, seen)
    entry["fc_detect"] = det
    cost += c2
    return entry, cost


def run_one(row: dict) -> dict:
    xd = [row["publisher_site"]]
    ceil = row["ceiling"] or None
    calls = 0

    def _try(fn, *a):
        nonlocal calls
        for attempt in range(2):
            _serper_gate()
            try:
                out = fn(*a)
                calls += 1
                return out or []
            except SearchError:
                calls += 1
                if attempt == 0:
                    time.sleep(60)
                else:
                    return []
            except Exception:  # noqa: BLE001
                calls += 1
                if attempt == 0:
                    time.sleep(5)
                else:
                    return []
        return []

    # B  ceiling OFF, UGC blocked  (production retrieval minus the ceiling)
    hb = _try(blocked_page, row["query"], None, xd)
    # U  ceiling ON,  UGC unblocked
    hu = _try(unblocked_page, row["query"], ceil)
    # BU ceiling OFF, UGC unblocked
    hbu = _try(unblocked_page, row["query"], None)

    sets = {}
    for name, hits in (("b", hb), ("u", hu), ("bu", hbu)):
        keep, drop = [], []
        for h in hits[:KEEP]:
            (drop if circ_reason(h.get("url") or "", row) else keep).append(h)
        sets[name] = {h.get("url") for h in keep}
        sets[name + "_raw"] = {h.get("url") for h in hits[:KEEP]}
        sets[name + "_drop"] = [(h.get("url"), circ_reason(h.get("url") or "", row))
                                for h in drop]

    union, seen = [], set()
    for h in hb[:KEEP] + hu[:KEEP] + hbu[:KEEP]:
        u = h.get("url") or ""
        if u and u not in seen:
            seen.add(u)
            union.append(h)

    claim_block = (f"CLAIM: {row['claim']}\n"
                   f"(claimed on {row.get('claim_date') or 'unknown date'}; judge the "
                   f"document's bearing on this exact proposition)")
    saved = {d["url"]: d for d in row["armA"] if d["url"]}
    docs, cost = [], 0.0
    for h in union:
        e, c = read_one(row, h, claim_block, saved)
        u = h.get("url")
        for k in ("b", "u", "bu"):
            e["in_" + k] = u in sets[k]
            e["in_" + k + "_raw"] = u in sets[k + "_raw"]
        docs.append(e)
        cost += c
    return {"key": row["key"], "corpus": row["corpus"], "fold": row["fold"],
            "claim": row["claim"], "ceiling": row["ceiling"], "query": row["query"],
            "n_hits": {"b": len(hb), "u": len(hu), "bu": len(hbu)},
            "n_requests": calls, "docs": docs, "cost": cost,
            "dropped": {k: sets[k + "_drop"] for k in ("b", "u", "bu")}}


def run(workers: int, smoke: int, limit: int = 0) -> None:
    _ugc_load()
    rows = [json.loads(l) for l in SAMPLE.open()]
    done = {json.loads(l)["key"] for l in RESULTS.open()} if RESULTS.exists() else set()
    todo = [r for r in rows if r["key"] not in done]
    random.Random(SEED + 1).shuffle(todo)   # corpus-mixed order: a partial run stays balanced
    if smoke:
        rng = random.Random(SEED)
        by = collections.defaultdict(list)
        for r in todo:
            by[r["corpus"]].append(r)
        todo = []
        for k in sorted(by):
            rng.shuffle(by[k])
            todo += by[k][:max(1, smoke // len(by))]
    elif limit:
        todo = todo[:limit]
    if not todo:
        print("nothing to do")
        return
    print(f"{len(todo)} claims | {workers} workers | "
          f"{dict(collections.Counter(r['corpus'] for r in todo))}", flush=True)
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
                if cost[0] > LLM_CAP or creds[0] > SERPER_CAP:
                    raise SystemExit(f"CAP HIT at {n[0]} claims: LLM ${cost[0]:.3f} "
                                     f"(cap {LLM_CAP}) / {creds[0]} credits (cap {SERPER_CAP})")
                if n[0] % 5 == 0 or n[0] == len(todo):
                    el = time.time() - t0
                    rate = n[0] / max(el, 1e-3)
                    print(f"  {n[0]}/{len(todo)}  ${cost[0]:.3f}  {creds[0]} cr  "
                          f"{rate*60:.1f} claims/min  "
                          f"ETA {(len(todo)-n[0])/max(rate,1e-6)/60:.1f}m  "
                          f"reused {nre[0]}/{nre[0]+nnew[0]} docs", flush=True)
    print(f"done: {n[0]} claims, LLM ${cost[0]:.4f}, {creds[0]} serper requests "
          f"= {creds[0]} credits, {(time.time()-t0)/60:.1f}m", flush=True)


# --------------------------------------------------------------------------- inspect
def inspect(k: int = 20) -> None:
    res = [json.loads(l) for l in RESULTS.open()]
    print(f"\ninspecting {len(res)} claims")
    drop = collections.Counter()
    for r in res:
        for arm, dd in r["dropped"].items():
            for _, why in dd:
                drop[(arm, why)] += 1
    print("  drops by (arm, rule):")
    for kk, v in sorted(drop.items()):
        print(f"    {kk[0]:3s} {kk[1]:12s} {v}")
    ugc = [d for r in res for d in r["docs"] if d["ugc"]]
    print(f"\n  UGC documents retrieved: {len(ugc)}  domains: "
          f"{dict(collections.Counter(d['domain'] for d in ugc).most_common(15))}")
    print(f"  UGC flags: {dict(collections.Counter(d.get('flag') for d in ugc))}")
    print(f"  UGC provenance: {dict(collections.Counter(d.get('provenance') for d in ugc))}")
    print("\n  --- filtered-out URLs (eyeball the blocklist) ---")
    shown = 0
    for r in res:
        allc = {u: w for arm in r["dropped"] for u, w in r["dropped"][arm]}
        det = [d for d in r["docs"] if d.get("fc_detect") and not d["circ"]]
        if not allc and not det:
            continue
        print("=" * 96)
        print(f"[{r['corpus']}] {r['claim'][:140]}")
        for u, w in list(allc.items())[:8]:
            print(f"   DROP [{w:11s}] {u[:110]}")
        for d in det[:4]:
            print(f"   KEPT-but-reader-says-factcheck [{d.get('flag')}] {d['url'][:100]}")
        shown += 1
        if shown >= k:
            break


# --------------------------------------------------------------------------- score
ARMS = ("A", "A_filt", "B", "B_rd", "U", "U_rd", "BU", "BU_rd", "C", "CU")
_RAW = {"C": "in_b_raw", "CU": "in_bu_raw"}
_KEY = {"B": "in_b", "U": "in_u", "BU": "in_bu"}


def arm_flags(rec: dict, arm: str, base: dict) -> list[str]:
    if arm == "A":
        return [d["flag"] for d in base["armA"]]
    if arm == "A_filt":
        return [d["flag"] for d in base["armA"] if not circ_reason(d["url"], base)]
    if arm in _RAW:
        return [d["flag"] for d in rec["docs"] if d[_RAW[arm]]]
    if arm.endswith("_rd"):
        k = _KEY[arm[:-3]]
        return [d["flag"] for d in rec["docs"] if d[k] and not d.get("fc_detect")]
    return [d["flag"] for d in rec["docs"] if d[_KEY[arm]]]


MODELS = ("7-flag", "3-voice")
CHANS = {"7-flag": list(FLAGS), "3-voice": ["support", "silent", "refute"]}


def count_matrix(flag_lists: list[list[str]], model: str) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CHANS[model])}
    C = np.zeros((len(CHANS[model]), len(flag_lists)))
    for j, fl in enumerate(flag_lists):
        for f in fl:
            C[idx[f if model == "7-flag" else VOICE[f]], j] += 1
    return C


def demix_w(cCN: np.ndarray, cTL: np.ndarray, eps: float) -> np.ndarray | None:
    """cross_ladder.demix_counts + laplace: log(p_TRUE / p_FALSE) with the
    timeline's false share removed at rate level."""
    k = len(cCN)
    pF = (cCN + 1) / (cCN.sum() + k)
    pM = (cTL + 1) / (cTL.sum() + k)
    pT = (pM - eps * pF) / (1 - eps)
    if (pT <= 0).any():
        return None
    return np.log(pT / pF)


def oof(Ccn, Ctl, fcn, ftl, eps):
    """Out-of-fold scores for both urns: the fit never sees the fold it scores."""
    scn, stl = np.empty(Ccn.shape[1]), np.empty(Ctl.shape[1])
    for k in range(fit_urn.K_FOLDS):
        w = demix_w(Ccn[:, fcn != k].sum(1), Ctl[:, ftl != k].sum(1), eps)
        if w is None:
            return None, None
        scn[fcn == k] = w @ Ccn[:, fcn == k]
        stl[ftl == k] = w @ Ctl[:, ftl == k]
    return scn, stl


def rec_at(s_cn, s_tl, eps, budget=0.02):
    """cross_ladder.roc_urn: low score means false. Recall over CN, false alarms
    over the timeline, corrected for the timeline's own false share."""
    thr = np.unique(np.concatenate([s_cn, s_tl]))
    rec = np.searchsorted(np.sort(s_cn), thr, side="right") / len(s_cn)
    fpr = np.searchsorted(np.sort(s_tl), thr, side="right") / len(s_tl)
    fpr = np.maximum.accumulate(np.clip((fpr - eps * rec) / (1 - eps), 0.0, 1.0))
    ok = fpr <= budget
    return float(rec[ok].max()) if ok.any() else 0.0


def corr(a_obs, eps):
    return (a_obs - eps / 2) / (1 - eps)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval — the own-post leak is a proportion, not an AUC, and it
    argues against unblocking UGC on its own, independently of any performance number."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def report() -> None:
    rows = {json.loads(l)["key"]: json.loads(l) for l in SAMPLE.open()}
    res = [json.loads(l) for l in RESULTS.open() if json.loads(l)["key"] in rows]
    cn = [r for r in res if r["corpus"] == "cn"]
    tl = [r for r in res if r["corpus"] == "tl"]
    print(f"\npopulation: CN-false {len(cn)}  timeline {len(tl)}  (eps {EPS})")

    print("\n### documents per claim and flag distribution")
    print(f"{'arm':8s} {'corpus':8s} {'docs/claim':>10s}  " +
          "  ".join(f"{f:>5s}" for f in FLAGS))
    for arm in ARMS:
        for name, sub in (("cn", cn), ("tl", tl)):
            fls = [arm_flags(r, arm, rows[r["key"]]) for r in sub]
            n = sum(len(f) for f in fls)
            c = collections.Counter(f for fl in fls for f in fl)
            print(f"{arm:8s} {name:8s} {n/max(len(sub),1):10.2f}  " +
                  "  ".join(f"{100*c[f]/max(n,1):5.1f}" for f in FLAGS))

    print("\n### RETRIEVED organic results per claim, per retrieval configuration")
    print("    (the count confound: retrieve-k cannot be equalised — `num` above 10 is")
    print("     ignored on this plan, `-site:` operators suppress Google's rich blocks")
    print("     and so free organic slots, and `tbs` shrinks the candidate pool)")
    print(f"  {'call':28s} {'cn':>8s} {'tl':>8s} {'all':>8s}")
    for k, lab in (("b", "B  -site: ops, no tbs"), ("u", "U  no ops, tbs"),
                   ("bu", "BU no ops, no tbs")):
        a = sum(r["n_hits"][k] for r in cn) / max(len(cn), 1)
        b = sum(r["n_hits"][k] for r in tl) / max(len(tl), 1)
        c = sum(r["n_hits"][k] for r in res) / max(len(res), 1)
        print(f"  {lab:28s} {a:8.2f} {b:8.2f} {c:8.2f}")

    print("\n### circularity drops, by arm and rule (of the retrieved top-10s)")
    drop = collections.Counter()
    tot = collections.Counter()
    for r in res:
        for arm, dd in r["dropped"].items():
            tot[arm] += r["n_hits"][arm]
            for _, why in dd:
                drop[(arm, why)] += 1
    for arm in ("b", "u", "bu"):
        d = {w: v for (a, w), v in drop.items() if a == arm}
        print(f"  {arm:3s} retrieved {tot[arm]:5d}  dropped {sum(d.values()):4d}  " +
              "  ".join(f"{k}={v}" for k, v in sorted(d.items(), key=lambda x: -x[1])))
    plat = collections.Counter()
    for r in res:
        for arm in ("u", "bu"):
            for u, w in r["dropped"][arm]:
                if w in ("own_post", "own_note", "note_mirror") and \
                        _domain_of(u) in ("x.com", "twitter.com", "t.co"):
                    plat[arm] += 1
    print(f"  of which dropped BECAUSE UGC was unblocked (x/twitter own-post, own-note,"
          f" note mirror): u={plat['u']} bu={plat['bu']}")

    det = collections.Counter()
    for r in res:
        for d in r["docs"]:
            if d["in_bu"]:
                det[d.get("fc_detect")] += 1
    print(f"  reader fc_detect on arm-BU documents: true={det[True]} false={det[False]} "
          f"unknown={det[None]} ({100*det[True]/max(sum(det.values()),1):.1f}%)")

    # ---- HEADLINE 1: the own-post leak. A proportion, reported on its own, because it
    # argues against unblocking UGC whatever the AUC does.
    print("\n### DIRECT CIRCULARITY: claims that retrieve their OWN source post once "
          "x.com is admitted")
    print(f"  {'arm':4s} {'corpus':8s} {'claims':>7s} {'own_post':>9s} {'rate':>7s} "
          f"{'Wilson 95%':>18s}  {'own_note':>9s} {'note_mirror':>12s}")
    for arm in ("b", "u", "bu"):
        for name, sub in (("cn", cn), ("tl", tl)):
            hit = sum(1 for r in sub
                      if any(w == "own_post" for _, w in r["dropped"][arm]))
            nn = sum(1 for r in sub
                     if any(w == "own_note" for _, w in r["dropped"][arm]))
            nm = sum(1 for r in sub
                     if any(w == "note_mirror" for _, w in r["dropped"][arm]))
            p, lo, hi = wilson(hit, len(sub))
            print(f"  {arm:4s} {name:8s} {len(sub):7d} {hit:9d} {p:7.3f} "
                  f"  [{lo:.3f}, {hi:.3f}]  {nn:9d} {nm:12d}")
    anyp = sum(1 for r in res if any(w == "own_post"
                                     for a in ("u", "bu") for _, w in r["dropped"][a]))
    p, lo, hi = wilson(anyp, len(res))
    print(f"  ANY unblocked arm, both corpora: {anyp}/{len(res)} = {p:.3f} "
          f"[{lo:.3f}, {hi:.3f}]")

    # ---- HEADLINE 2: UGC arrives snippet-grade
    ug = [d for r in res for d in r["docs"] if d["ugc"] and not d.get("reused")]
    sn = sum(1 for d in ug if d.get("provenance") == "snippet")
    p, lo, hi = wilson(sn, len(ug))
    print(f"\n### UGC evidence grade: {sn}/{len(ug)} newly fetched UGC documents are "
          f"SNIPPET-ONLY = {p:.3f} [{lo:.3f}, {hi:.3f}]")
    bd = collections.defaultdict(lambda: [0, 0])
    for d in ug:
        bd[d["domain"]][0] += 1
        bd[d["domain"]][1] += (d.get("provenance") == "snippet")
    for dom, (n, s) in sorted(bd.items(), key=lambda x: -x[1][0])[:12]:
        print(f"    {dom:22s} n={n:4d}  snippet-only {s:4d} ({100*s/n:3.0f}%)")

    print("\n### UGC documents (what unblocking actually admitted)")
    for name, sub in (("cn", cn), ("tl", tl)):
        u = [d for r in sub for d in r["docs"] if d["ugc"] and d["in_bu"]]
        c = collections.Counter(d.get("flag") for d in u)
        n = max(len(u), 1)
        print(f"  {name}: {len(u)} UGC docs in BU ({len(u)/max(len(sub),1):.2f}/claim)  "
              f"refute {100*(c['1']+c['2'])/n:.1f}%  support {100*(c['5']+c['4'])/n:.1f}%  "
              f"silent {100*(c['3']+c['X']+c['I'])/n:.1f}%")
        print(f"     top domains: "
              f"{dict(collections.Counter(d['domain'] for d in u).most_common(8))}")

    print("\n### read reuse (documents already read under production, reused verbatim)")
    print(f"  {'arm':8s} {'corpus':8s} {'docs':>6s} {'reused':>7s} {'new':>6s} {'reuse %':>8s}")
    for arm in ("B", "U", "BU"):
        for name, sub in (("cn", cn), ("tl", tl)):
            k = _KEY[arm]
            dd = [d for r in sub for d in r["docs"] if d[k]]
            ru = sum(1 for d in dd if d.get("reused"))
            print(f"  {arm:8s} {name:8s} {len(dd):6d} {ru:7d} {len(dd)-ru:6d} "
                  f"{100*ru/max(len(dd),1):7.1f}%")
    alld = [d for r in res for d in r["docs"]]
    ru = sum(1 for d in alld if d.get("reused"))
    print(f"  union    ALL      {len(alld):6d} {ru:7d} {len(alld)-ru:6d} "
          f"{100*ru/max(len(alld),1):7.1f}%   <- paid reads: {len(alld)-ru}")

    # ---- the two-urn AUC, observed and contamination-corrected
    fcn = np.array([r["fold"] for r in cn])
    ftl = np.array([r["fold"] for r in tl])
    Ccn = {(a, m): count_matrix([arm_flags(r, a, rows[r["key"]]) for r in cn], m)
           for a in ARMS for m in MODELS}
    Ctl = {(a, m): count_matrix([arm_flags(r, a, rows[r["key"]]) for r in tl], m)
           for a in ARMS for m in MODELS}
    # PINNED instrument: the arm-A fit, so every arm is scored by production's weights
    # and the delta is the retrieval change alone, not a refitted scale.
    pinned = {m: demix_w(Ccn[("A", m)].sum(1), Ctl[("A", m)].sum(1), EPS) for m in MODELS}

    def cell(a, m, mode):
        if mode == "pin":
            w = pinned[m]
            scn, stl = w @ Ccn[(a, m)], w @ Ctl[(a, m)]
        else:
            scn, stl = oof(Ccn[(a, m)], Ctl[(a, m)], fcn, ftl, EPS)
            if scn is None:
                return None
        obs = NCA.auc_np(stl, scn)          # timeline = positive class
        return obs, corr(obs, EPS), rec_at(scn, stl, EPS)

    print("\n### CN-false vs timeline: AUC (observed and eps-corrected) and recall@2%FPR")
    for m in MODELS:
        for mode in ("pin", "refit"):
            print(f"\n  -- {m} / {'pinned arm-A weights' if mode=='pin' else 'refit per arm, out of fold'}")
            print(f"     {'arm':8s} {'AUC obs':>9s} {'AUC corr':>9s} {'R@2%FPR':>9s}")
            for a in ARMS:
                c = cell(a, m, mode)
                print(f"     {a:8s} " + ("infeasible" if c is None else
                                         f"{c[0]:9.4f} {c[1]:9.4f} {c[2]:9.3f}"))

    print(f"\n  paired bootstrap, {BOOT_REPS} reps, seed {BOOT_SEED}, resampled within corpus")
    rng = np.random.default_rng(BOOT_SEED)
    reps = {(a, m, k): [] for a in ARMS for m in MODELS for k in ("auc", "rec")}
    for _ in range(BOOT_REPS):
        ic = rng.choice(len(cn), len(cn))
        it = rng.choice(len(tl), len(tl))
        for m in MODELS:
            w = pinned[m]
            for a in ARMS:
                scn, stl = (w @ Ccn[(a, m)])[ic], (w @ Ctl[(a, m)])[it]
                reps[(a, m, "auc")].append(corr(NCA.auc_np(stl, scn), EPS))
                reps[(a, m, "rec")].append(rec_at(scn, stl, EPS))
    # docs/claim carried onto every delta line: retrieve-k cannot be equalised across
    # the 2x2 (`num` above 10 is ignored; `-site:` operators suppress Google's rich
    # blocks and free organic slots, and `tbs` shrinks the candidate pool outright —
    # measured, see the retrieval-count section), so a count difference must never be
    # read as a performance difference.
    dpc = {}
    for a in ARMS:
        for name, sub in (("cn", cn), ("tl", tl)):
            fls = [arm_flags(r, a, rows[r["key"]]) for r in sub]
            dpc[(a, name)] = sum(len(f) for f in fls) / max(len(sub), 1)
    for m in MODELS:
        for k, lab in (("auc", "AUC corrected"), ("rec", "recall@2%FPR")):
            base = np.array(reps[("A", m, k)])
            print(f"\n     {m} / {lab}   arm A {np.mean(base):.4f} "
                  f"[{NCA.ci(base)[0]:.4f}, {NCA.ci(base)[1]:.4f}]"
                  f"   docs/claim cn {dpc[('A','cn')]:.2f} tl {dpc[('A','tl')]:.2f}")
            for a in ARMS[1:]:
                d = np.array(reps[(a, m, k)]) - base
                lo, hi = NCA.ci(d)
                print(f"       d({a:6s} - A) {np.mean(d):+.4f}  [{lo:+.4f}, {hi:+.4f}]"
                      f"{'  *' if lo > 0 or hi < 0 else '   '}"
                      f"  docs/claim cn {dpc[(a,'cn')]:5.2f} ({dpc[(a,'cn')]-dpc[('A','cn')]:+.2f}) "
                      f"tl {dpc[(a,'tl')]:5.2f} ({dpc[(a,'tl')]-dpc[('A','tl')]:+.2f})")

    # ---- mechanism
    print("\n### refuting documents per claim (what any gain must come from)")
    print(f"  {'arm':8s} {'CN-false':>10s} {'timeline':>10s}")
    for a in ARMS:
        def refm(sub):
            v = [sum(1 for f in arm_flags(r, a, rows[r["key"]]) if f in ("1", "2"))
                 for r in sub]
            return sum(v) / max(len(v), 1)
        print(f"  {a:8s} {refm(cn):10.3f} {refm(tl):10.3f}")

    print("\n### are the new refuting documents post-dated?")
    print(f"  {'arm':6s} {'corpus':8s} {'refuting':>9s} {'new':>6s} {'dated > ceiling':>16s} "
          f"{'undated':>8s}")
    for a in ("B", "BU"):
        for name, sub in (("cn", cn), ("tl", tl)):
            k = _KEY[a]
            ref = [(r, d) for r in sub for d in r["docs"]
                   if d[k] and d.get("flag") in ("1", "2")]
            new = [(r, d) for r, d in ref if not d.get("reused")]
            post = sum(1 for r, d in new if (d.get("date") or "") and r["ceiling"]
                       and NCA._iso(d["date"]) and NCA._iso(d["date"]) > r["ceiling"])
            und = sum(1 for r, d in new if not (d.get("date") or "").strip())
            print(f"  {a:6s} {name:8s} {len(ref):9d} {len(new):6d} {post:16d} {und:8d}")

    print(f"\ntotal LLM spend recorded in results: ${sum(r['cost'] for r in res):.4f}")
    print(f"total serper requests recorded: {sum(r['n_requests'] for r in res)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    if a.sample:
        build_sample()
    if a.run:
        run(a.workers, a.smoke, a.limit)
    if a.inspect:
        inspect()
    if a.report:
        report()
