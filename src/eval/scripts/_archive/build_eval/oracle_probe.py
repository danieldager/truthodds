"""Oracle probe: we KNOW the document that refutes the claim. Could we have found it?

Daniel's question (2026-08-28): "There are false claims for which we know there is
evidence, because the fact checkers and the community notes reviewers have confirmed
the claim is false with some evidence. And it seems like we cannot find the evidence.
Is there a way for us to find this evidence that we know exists?"

This works BACKWARDS from the reviewer's own cited source. Four stages, early-stop:

  S0  ceiling legality. Does the target predate the claim's ceiling? A target dated
      after the ceiling is unreachable BY CONSTRUCTION and is not a query failure.
      (The Exa probe found 166/211 refutations postdate the claim.)
  S1  is the target in Google's index AT ALL. Exact-title-in-quotes, then a
      site-restricted fallback. Binary. This is the fork: if the document is not
      indexed no query can reach it and the thread is closed.
  S2  is it reachable by a query built from the CLAIM ONLY (never from the target).
      A 5-rung ladder; record the rung of first appearance in the top 10.
        r1 the production query verbatim  r2 bare claim text  r3 entities+event noun,
        no date  r4 natural-language question  r5 entity + source-class routing hint
  S3  the diagnosis: for targets some rung reaches and r1 does not, name the
      difference between the production query and the winning rung.

POPULATION is imported from retrieval_recovery_arms (load_corpora + _allirr) so this
is the same all-irrelevant block the six previous arms were scored on: every retrieved
document read I, fc-gold media_authenticity already excluded.

Serper only. Nothing under eval/data/urn_runs/ is written. Output: eval/data/oracle_probe/.

    uv run python -m eval.scripts.build_eval.oracle_probe --sample
    uv run python -m eval.scripts.build_eval.oracle_probe --dates --workers 16
    uv run python -m eval.scripts.build_eval.oracle_probe --stage1 --workers 8
    uv run python -m eval.scripts.build_eval.oracle_probe --stage2 --workers 8
    uv run python -m eval.scripts.build_eval.oracle_probe --report
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
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

import requests  # noqa: E402

from pipeline import disk_cache  # noqa: E402
from pipeline.search import (  # noqa: E402
    SERPER_ENDPOINT, SearchError, _request, _serper_page, cache_key, search,
    serper_finalize, serper_headers, serper_parse)
from eval.scripts.build_eval import retrieval_forensics as RF  # noqa: E402
from eval.scripts.build_eval.retrieval_recovery_arms import _allirr, load_corpora  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import llm  # noqa: E402
from eval.scripts.build_eval.refute_verify import w7  # noqa: E402

OUT = SRC / "eval/data/oracle_probe"
SAMPLE = OUT / "sample.jsonl"
DATES = OUT / "target_dates.jsonl"
S1 = OUT / "stage1.jsonl"
S2 = OUT / "stage2.jsonl"
RUNGS = OUT / "rungs.jsonl"

SEED = 20260828
N_SAMPLE = 120
MAX_TARGETS = 3          # reviewer URLs probed per claim
SERPER_CAP = 2600        # hard stop, ~$2.60 at the conservative $1/1k credit rate
_URL_RE = re.compile(r'https?://[^\s<>"\')\]]+')
_STATUS = re.compile(r"/status/(\d+)")
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# --- global serper meter -----------------------------------------------------
_meter_lock = threading.Lock()
_credits = [0]


def spend(n: int = 1) -> None:
    with _meter_lock:
        _credits[0] += n
        if _credits[0] > SERPER_CAP:
            raise SystemExit(f"SERPER CAP {SERPER_CAP} breached")


_gate_lock = threading.Lock()
_gate_last = [0.0]
_GATE = 0.05             # 20 q/s, well under the plan's ceiling


def gate() -> None:
    while True:
        with _gate_lock:
            now = time.time()
            if now - _gate_last[0] >= _GATE:
                _gate_last[0] = now
                return
            w = _GATE - (now - _gate_last[0])
        time.sleep(w)


# --------------------------------------------------------------------------- sample
def _cn_targets(note: str, post_id: str) -> list[str]:
    out = []
    for u in _URL_RE.findall(note or ""):
        u = u.rstrip(".,;:)")
        d = RF.dom(u)
        if not d or d in RF.NAV or RF.WIDGET.search(u):
            continue
        if d in RF.ARCHIVE or any(d.endswith("." + a) for a in RF.ARCHIVE):
            continue
        m = _STATUS.search(u)
        if m and m.group(1) == post_id:      # the disputed post itself is never evidence
            continue
        out.append(u)
    return list(dict.fromkeys(out))


def _gold_survivors() -> dict[str, list[str]]:
    """The reviewer's outbound links minus self-domain, widgets, archives, nav and
    per-publisher chrome. Same four filters as retrieval_forensics.gold."""
    links = {}
    for line in RF.FCL.open():
        r = json.loads(line)
        links[r["review_url"]] = r["links"]
    surv = {}
    for ru, ls in links.items():
        self_d = RF.dom(ru)
        keep = []
        for l in ls:
            u = l["url"]
            d = RF.dom(u)
            if not d or d == self_d or RF.dmatch(d, self_d):
                continue
            if RF.WIDGET.search(u) or d in RF.NAV:
                continue
            if d in RF.ARCHIVE or any(d.endswith("." + a) for a in RF.ARCHIVE):
                continue
            keep.append((d, u))
        surv[ru] = keep
    pub_docs, pub_dom = collections.Counter(), collections.Counter()
    for ru, s in surv.items():
        pub = RF.dom(ru)
        pub_docs[pub] += 1
        for d in {d for d, _ in s}:
            pub_dom[(pub, d)] += 1
    chrome = {(p, d) for (p, d), k in pub_dom.items()
              if pub_docs[p] >= 5 and k / pub_docs[p] >= RF.CHROME_DF}
    return {ru: [u for d, u in s if (RF.dom(ru), d) not in chrome] for ru, s in surv.items()}


def build_sample() -> None:
    w = w7()
    cn, gold, _tl, notes = load_corpora(w)
    surv = _gold_survivors()
    pool = []
    for pid, r in cn.items():
        if not _allirr(r):
            continue
        t = _cn_targets(notes.get(pid, ""), pid)
        if t:
            pool.append(("cn", r, t))
    for ru, r in gold.items():
        if not _allirr(r):
            continue
        t = surv.get(ru) or []
        if t:
            pool.append(("gold", r, t))

    rows = []
    for corpus, r, targets in pool:
        ev = [u for u in targets if not RF.blocked(RF.dom(u))]
        ugc = [u for u in targets if RF.blocked(RF.dom(u))]
        rows.append({
            "corpus": corpus,
            "key": r["review_url"],
            "post_id": r.get("post_id"),
            "claim": r.get("claim_resolved") or r.get("claim_text") or "",
            "claim_date": r.get("claim_date_shown"),
            "ceiling": r.get("ceiling"),
            "publisher_site": r.get("publisher_site") or "x.com",
            "query_prod": r.get("query") or "",
            "claim_type": r.get("claim_type"),
            "targets_ev": ev[:MAX_TARGETS],
            "n_ev": len(ev), "n_ugc": len(ugc),
            "blocked_only": not ev,
        })
    print(f"population (all-irrelevant, >=1 reviewer URL): {len(rows)}  "
          f"{dict(collections.Counter(r['corpus'] for r in rows))}")
    bo = [r for r in rows if r["blocked_only"]]
    print(f"  blocked-only (every reviewer URL on the UGC blocklist): {len(bo)} "
          f"({len(bo)/len(rows):.1%})  {dict(collections.Counter(r['corpus'] for r in bo))}")

    rng = random.Random(SEED)
    rng.shuffle(rows)
    draw = rows[:N_SAMPLE]
    OUT.mkdir(parents=True, exist_ok=True)
    with SAMPLE.open("w") as f:
        for r in draw:
            f.write(json.dumps(r) + "\n")
    print(f"  DRAW {len(draw)}  {dict(collections.Counter(r['corpus'] for r in draw))}  "
          f"blocked-only in draw {sum(1 for r in draw if r['blocked_only'])}  -> {SAMPLE}")
    print(f"  probe targets in draw: {sum(len(r['targets_ev']) for r in draw)}")


# ------------------------------------------------------------------------- S0 dates
_META = [
    re.compile(r'property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)', re.I),
    re.compile(r'content=["\']([^"\']+)["\'][^>]*property=["\']article:published_time["\']', re.I),
    re.compile(r'name=["\'](?:pubdate|publish-date|publication_date|date|DC\.date\.issued|'
               r'article\.published|sailthru\.date|parsely-pub-date)["\'][^>]*content=["\']([^"\']+)', re.I),
    re.compile(r'itemprop=["\']datePublished["\'][^>]*content=["\']([^"\']+)', re.I),
    re.compile(r'"datePublished"\s*:\s*"([^"]+)"', re.I),
    re.compile(r'<time[^>]+datetime=["\']([^"\']+)', re.I),
]
_URLDATE = re.compile(r"/(20\d{2})[/-](\d{1,2})[/-](\d{1,2})(?:/|-|$)")
_URLYM = re.compile(r"/(20\d{2})[/-](\d{1,2})/")


def _iso(s: str) -> str | None:
    s = (s or "").strip()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return m.group(0)
    for fmt in ("%d %B %Y", "%B %d, %Y", "%d %b %Y", "%b %d, %Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s[:30].strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def url_date(u: str) -> str | None:
    m = _URLDATE.search(u)
    if m:
        y, mo, d = m.groups()
        try:
            return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
        except ValueError:
            return None
    m = _URLYM.search(u)
    if m:
        y, mo = m.groups()
        try:
            return datetime(int(y), int(mo), 28).strftime("%Y-%m-%d")   # month-end, conservative
        except ValueError:
            return None
    return None


def fetch_date(u: str) -> dict:
    """Free: URL path date, then the page's own meta tags. No paid API."""
    ud = url_date(u)
    out = {"url": u, "url_date": ud, "meta_date": None, "http": None, "title": None}
    try:
        r = requests.get(u, headers=_UA, timeout=12, allow_redirects=True)
        out["http"] = r.status_code
        if r.status_code < 400 and "html" in (r.headers.get("content-type") or "").lower():
            h = r.text[:400000]
            for rx in _META:
                m = rx.search(h)
                if m and (d := _iso(m.group(1))):
                    out["meta_date"] = d
                    break
            if (m := re.search(r"<title[^>]*>(.*?)</title>", h, re.S | re.I)):
                t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip()
                out["title"] = t[:160] or None
    except Exception as e:  # noqa: BLE001
        out["http"] = type(e).__name__
    out["date"] = out["meta_date"] or out["url_date"]
    out["src"] = "meta" if out["meta_date"] else ("url" if out["url_date"] else None)
    return out


def run_dates(workers: int) -> None:
    rows = [json.loads(l) for l in SAMPLE.open()]
    urls = sorted({u for r in rows for u in r["targets_ev"]})
    done = {json.loads(l)["url"] for l in DATES.open()} if DATES.exists() else set()
    todo = [u for u in urls if u not in done]
    print(f"dates: {len(todo)} urls ({len(done)} cached) | {workers} workers", flush=True)
    t0, n = time.time(), [0]
    lock = threading.Lock()
    with DATES.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(fetch_date, u) for u in todo]):
            d = fut.result()
            with lock:
                f.write(json.dumps(d) + "\n")
                f.flush()
                n[0] += 1
                if n[0] % 25 == 0 or n[0] == len(todo):
                    el = time.time() - t0
                    print(f"  {n[0]}/{len(todo)}  {n[0]/max(el,.001)*60:.0f}/min  "
                          f"ETA {(len(todo)-n[0])/max(n[0]/max(el,.001),1e-6)/60:.1f}m", flush=True)


def load_dates() -> dict:
    if not DATES.exists():
        return {}
    return {j["url"]: j for j in (json.loads(l) for l in DATES.open())}


def load_index() -> dict:
    if not S1.exists():
        return {}
    return {j["url"]: j for j in (json.loads(l) for l in S1.open())}


def legality(rows: list[dict], dates: dict, idx: dict | None = None) -> None:
    """Annotate each row in place: per-target legal / illegal / unknown. `idx` is the
    S1 output, whose Serper result carries a publication date for pages we could not
    fetch ourselves (403/404) — it resolves 20 of the 116 undated targets."""
    idx = idx or {}
    for r in rows:
        r["tgt"] = []
        for u in r["targets_ev"]:
            d = (dates.get(u) or {}).get("date")
            src = (dates.get(u) or {}).get("src")
            ttl = (dates.get(u) or {}).get("title")
            if d is None and (sd := (idx.get(u) or {}).get("serper_date")):
                if (d := _iso(sd)):
                    src = "serper"
            if d is None:
                cls = "unknown"
            elif r["ceiling"] and d <= r["ceiling"]:
                cls = "legal"
            else:
                cls = "illegal"
            r["tgt"].append({"url": u, "date": d, "title": ttl, "_fetched": u in dates,
                             "date_src": src, "legal": cls})
        cls = [t["legal"] for t in r["tgt"]]
        r["claim_legal"] = ("blocked_only" if not cls else
                            "legal" if "legal" in cls else
                            "unknown" if "unknown" in cls else "illegal")


# ------------------------------------------------------------------------ S1 index
def _title_of(u: str) -> str | None:
    try:
        r = requests.get(u, headers=_UA, timeout=12)
        if r.status_code >= 400:
            return None
        m = re.search(r"<title[^>]*>(.*?)</title>", r.text[:200000], re.S | re.I)
        if not m:
            return None
        t = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1))).strip()
        return t[:160] or None
    except Exception:  # noqa: BLE001
        return None


def _norm(u: str) -> tuple[str, str]:
    sp = urlsplit(u.split("#")[0].split("?")[0])
    return sp.netloc.lower().removeprefix("www."), sp.path.rstrip("/").lower()


def raw_serper(q: str, num: int = 10) -> list[dict]:
    """Serper with NO blocklist and NO ceiling: pure index-presence question."""
    gate()
    spend()
    data = _request("POST", SERPER_ENDPOINT, label="serper-oracle",
                    json={"q": q, "num": num}, headers=serper_headers(), timeout=20)
    return serper_parse(data, num)


def probe_index(t: dict) -> dict:
    """Is this exact document in Google's index. Title already came from the S0 fetch;
    it is only refetched when S0 never saw the page at all."""
    u = t["url"]
    title = t.get("title") or (None if t.get("_fetched") else _title_of(u))
    out = {"url": u, "title": title, "indexed": False, "how": None, "serper_date": None}
    dom = _norm(u)[0]
    slug = re.sub(r"[-_+]", " ", u.rstrip("/").rsplit("/", 1)[-1])[:90]
    tries = []
    if title:
        tries.append((f'"{title}"', "exact_title"))
        tries.append((f'site:{dom} {title[:90]}', "site_title"))
    else:
        tries.append((u, "raw_url"))          # Google resolves a pasted URL when it has it
    tries.append((f'site:{dom} {slug}', "site_slug"))
    for q, how in tries:
        try:
            res = raw_serper(q)
        except SearchError:
            continue
        for h in res:
            if _norm(h.get("url") or "") == _norm(u):
                out.update(indexed=True, how=how, serper_date=h.get("date"))
                return out
        # domain+title present but a different URL: the doc exists on an indexed site
        if how == "exact_title" and any(_norm(h.get("url") or "")[0] == dom for h in res):
            out["domain_visible"] = True
    return out


def run_stage1(workers: int) -> None:
    rows = [json.loads(l) for l in SAMPLE.open()]
    legality(rows, load_dates(), load_index())
    todo = []
    for r in rows:
        for t in r["tgt"]:
            if t["legal"] in ("legal", "unknown"):
                todo.append(t)
    done = {json.loads(l)["url"] for l in S1.open()} if S1.exists() else set()
    todo = [t for t in todo if t["url"] not in done]
    print(f"stage1: {len(todo)} targets | {workers} workers | cap {SERPER_CAP}", flush=True)
    t0, n, hit = time.time(), [0], [0]
    lock = threading.Lock()
    with S1.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(probe_index, t) for t in todo]):
            try:
                o = fut.result()
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                f.write(json.dumps(o) + "\n")
                f.flush()
                n[0] += 1
                hit[0] += o["indexed"]
                if n[0] % 20 == 0 or n[0] == len(todo):
                    el = time.time() - t0
                    print(f"  {n[0]}/{len(todo)}  indexed {hit[0]}  credits {_credits[0]}  "
                          f"{n[0]/max(el,.001)*60:.0f}/min  "
                          f"ETA {(len(todo)-n[0])/max(n[0]/max(el,.001),1e-6)/60:.1f}m", flush=True)


# ------------------------------------------------------------------------- S2 ladder
RUNG_SYS = (
    "You write web search queries to check a factual claim. You are given only the CLAIM. "
    "Produce three different queries, each a plain search-engine query string, no operators "
    "unless asked:\n"
    "r3: the claim's named entities plus the noun that names the EVENT or THING at issue. "
    "No date token of any kind, no year, no month. Keywords only.\n"
    "r4: the same check written as a natural-language question a person would type.\n"
    "r5: the claim's single most important entity plus a hint that ROUTES to the kind of "
    "source that would settle it - an official register or database, a statistical agency, "
    "a court or parliamentary record, the named actor's own newsroom or press release, a "
    "regulator's filing. Name that source type in the query. No date token.\n"
    'Respond JSON only: {"r3": "...", "r4": "...", "r5": "..."}')


def gen_rungs(row: dict) -> tuple[dict, float]:
    for _ in range(3):
        try:
            j, c, _, _ = llm([{"role": "system", "content": RUNG_SYS},
                              {"role": "user", "content": f"CLAIM: {row['claim']}"}],
                             max_tokens=220)
            if all((j.get(k) or "").strip() for k in ("r3", "r4", "r5")):
                return {k: j[k].strip()[:300] for k in ("r3", "r4", "r5")}, c
        except Exception:  # noqa: BLE001
            time.sleep(2)
    raise RuntimeError("rung gen failed")


def ladder_search(q: str, ceiling: str | None, xd: list[str]) -> list[dict]:
    """The PRODUCTION retrieval path (blocklist, ceiling, origin exclusion) so a hit
    means the production pipeline would really have seen the document. `xd` is the
    same [publisher_site] the original run passed, so r1 replays it exactly (and
    usually lands on the existing disk-cache entry, costing nothing)."""
    key = cache_key("serper", q, 10, ceiling or None, sorted(xd), 0)
    if disk_cache.get("serper", key) is None:      # meter LIVE fetches only
        gate()
        spend()
    return search(q, 10, date_ceiling=ceiling or None, exclude_domains=xd, min_results=0,
                  provider="serper")


def run_stage2(workers: int) -> None:
    rows = [json.loads(l) for l in SAMPLE.open()]
    idx = load_index()
    legality(rows, load_dates(), idx)
    rungs = {}
    if RUNGS.exists():
        rungs = {j["key"]: j for j in (json.loads(l) for l in RUNGS.open())}

    todo = []
    for r in rows:
        tg = [t for t in r["tgt"] if t["legal"] == "legal" and (idx.get(t["url"]) or {}).get("indexed")]
        if tg:
            r["probe"] = tg
            todo.append(r)
    done = {json.loads(l)["key"] for l in S2.open()} if S2.exists() else set()
    todo = [r for r in todo if r["key"] not in done]
    print(f"stage2: {len(todo)} claims | {workers} workers | credits so far {_credits[0]}",
          flush=True)

    lock = threading.Lock()
    rf = RUNGS.open("a")

    def one(r: dict) -> dict:
        cost = 0.0
        if r["key"] in rungs:
            g = rungs[r["key"]]
        else:
            gg, cost = gen_rungs(r)
            g = {"key": r["key"], **gg}
            with lock:
                rf.write(json.dumps(g) + "\n")
                rf.flush()
        ladder = [("r1", r["query_prod"]), ("r2", r["claim"][:280]),
                  ("r3", g["r3"]), ("r4", g["r4"]), ("r5", g["r5"])]
        want = {_norm(t["url"]) for t in r["probe"]}
        wantdom = {_norm(t["url"])[0] for t in r["probe"]}
        out = {"key": r["key"], "corpus": r["corpus"], "claim": r["claim"],
               "claim_type": r.get("claim_type"), "ceiling": r["ceiling"],
               "targets": r["probe"], "queries": dict(ladder), "cost": cost, "rungs": {}}
        first = None
        firstdom = None
        for name, q in ladder:
            if not q:
                out["rungs"][name] = {"hit": None, "n": 0}
                continue
            try:
                res = ladder_search(q, r["ceiling"], [r["publisher_site"]])
            except SearchError as e:
                out["rungs"][name] = {"hit": None, "err": str(e)[:80]}
                continue
            urls = [h.get("url") or "" for h in res]
            hit = [u for u in urls if _norm(u) in want]
            dhit = [u for u in urls if _norm(u)[0] in wantdom]
            out["rungs"][name] = {"hit": bool(hit), "hit_urls": hit[:3],
                                  "dom_hit": bool(dhit), "n": len(res),
                                  "top": urls[:10]}
            if hit and first is None:
                first = name
            if dhit and firstdom is None:
                firstdom = name
            if first and name != "r1":
                break                      # early stop: the rung question is answered
        out["first_hit"] = first
        out["first_dom_hit"] = firstdom
        out["r1_hit"] = bool((out["rungs"].get("r1") or {}).get("hit"))
        return out

    t0, n = time.time(), [0]
    with S2.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, r) for r in todo]):
            try:
                o = fut.result()
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                f.write(json.dumps(o) + "\n")
                f.flush()
                n[0] += 1
                el = time.time() - t0
                if n[0] % 5 == 0 or n[0] == len(todo):
                    print(f"  {n[0]}/{len(todo)}  credits {_credits[0]}  "
                          f"{n[0]/max(el,.001)*60:.1f} claims/min  "
                          f"ETA {(len(todo)-n[0])/max(n[0]/max(el,.001),1e-6)/60:.1f}m", flush=True)
    rf.close()


# ---------------------------------------------------------------- S2b why not top-10
# The ladder found almost nothing, so the next question is WHICH of the two remaining
# explanations holds for an indexed, ceiling-legal document that no claim-derived query
# surfaces:
#   depth   it IS retrievable, just not in the top 10 -> a ranking/depth problem
#   ceiling Google's `tbs` custom-date-range filter (EVAL-ONLY; production passes
#           date_ceiling=None) drops it, because Google does not know the page's date
#           -> an eval artifact, not a production failure
#   neither the query and the document share no retrievable surface at all
S2B = OUT / "stage2b.jsonl"


def _probe(q: str, ceiling: str | None, xd: list[str], num: int, want: set) -> dict:
    key = cache_key("serper", q, num, ceiling or None, sorted(xd), 0)
    if disk_cache.get("serper", key) is None:
        gate()
        spend(1 if num <= 10 else 2)          # Serper bills num>10 at 2 credits
    res = search(q, num, date_ceiling=ceiling or None, exclude_domains=xd, min_results=0,
                 provider="serper")
    urls = [h.get("url") or "" for h in res]
    rank = next((i + 1 for i, u in enumerate(urls) if _norm(u) in want), None)
    return {"n": len(urls), "rank": rank}


def run_stage2b(workers: int) -> None:
    res = [json.loads(l) for l in S2.open()]
    miss = [r for r in res if not r["first_hit"]]
    rows = {json.loads(l)["key"]: json.loads(l) for l in SAMPLE.open()}
    done = {json.loads(l)["key"] for l in S2B.open()} if S2B.exists() else set()
    miss = [r for r in miss if r["key"] not in done]
    print(f"stage2b: {len(miss)} unreached claims | {workers} workers | "
          f"credits so far {_credits[0]}", flush=True)

    def one(r: dict) -> dict:
        want = {_norm(t["url"]) for t in r["targets"]}
        xd = [rows[r["key"]]["publisher_site"]]
        out = {"key": r["key"], "corpus": r["corpus"], "claim": r["claim"],
               "targets": r["targets"], "queries": r["queries"], "probes": {}}
        for rung in ("r1", "r2"):
            q = r["queries"].get(rung)
            if not q:
                continue
            out["probes"][f"{rung}_deep100_ceil"] = _probe(q, r["ceiling"], xd, 100, want)
            out["probes"][f"{rung}_noceil_top10"] = _probe(q, None, xd, 10, want)
            out["probes"][f"{rung}_noceil_deep100"] = _probe(q, None, xd, 100, want)
        return out

    lock, n, t0 = threading.Lock(), [0], time.time()
    with S2B.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, r) for r in miss]):
            try:
                o = fut.result()
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                f.write(json.dumps(o) + "\n")
                f.flush()
                n[0] += 1
                el = time.time() - t0
                if n[0] % 5 == 0 or n[0] == len(miss):
                    print(f"  {n[0]}/{len(miss)}  credits {_credits[0]}  "
                          f"{n[0]/max(el,.001)*60:.1f} claims/min  "
                          f"ETA {(len(miss)-n[0])/max(n[0]/max(el,.001),1e-6)/60:.1f}m", flush=True)


# ------------------------------------------------------------------- S2c real depth
# S2b's num=100 was a dud: this Serper plan caps a request at 10 organic results and
# silently ignores num, so those probes were top-10 repeats. `page` DOES paginate, so
# depth is measured by walking pages 2-5 (ranks 11-50) of the two strongest rungs.
# NOTE: `search()`'s cache key does not include `page`, so paginated calls MUST bypass
# it (they would otherwise overwrite the production page-1 entry for that query).
S2C = OUT / "stage2c.jsonl"
DEPTH_PAGES = (2, 3, 4, 5)


def _page_probe(q: str, ceiling: str | None, xd: list[str], page: int, want: set) -> dict:
    gate()
    spend()
    res = serper_finalize(_serper_page(q, 10, ceiling or None, xd, page), xd)
    urls = [h.get("url") or "" for h in res]
    return {"n": len(urls), "hit": [u for u in urls if _norm(u) in want],
            "dom_hit": [u for u in urls if _norm(u)[0] in {d for d, _ in want}]}


def run_stage2c(workers: int) -> None:
    res = [json.loads(l) for l in S2.open()]
    miss = [r for r in res if not r["first_hit"]]
    rows = {json.loads(l)["key"]: json.loads(l) for l in SAMPLE.open()}
    done = {json.loads(l)["key"] for l in S2C.open()} if S2C.exists() else set()
    miss = [r for r in miss if r["key"] not in done]
    print(f"stage2c: {len(miss)} claims x 2 rungs x {len(DEPTH_PAGES)} pages | "
          f"{workers} workers | credits so far {_credits[0]}", flush=True)

    def one(r: dict) -> dict:
        want = {_norm(t["url"]) for t in r["targets"]}
        xd = [rows[r["key"]]["publisher_site"]]
        out = {"key": r["key"], "corpus": r["corpus"], "claim": r["claim"],
               "targets": r["targets"], "pages": {}}
        for rung in ("r1", "r2"):
            q = r["queries"].get(rung)
            if not q:
                continue
            for pg in DEPTH_PAGES:
                try:
                    out["pages"][f"{rung}_p{pg}"] = _page_probe(q, r["ceiling"], xd, pg, want)
                except SearchError as e:
                    out["pages"][f"{rung}_p{pg}"] = {"err": str(e)[:80]}
        out["depth_hit"] = any(v.get("hit") for v in out["pages"].values())
        out["depth_dom_hit"] = any(v.get("dom_hit") for v in out["pages"].values())
        return out

    lock, n, t0 = threading.Lock(), [0], time.time()
    with S2C.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, r) for r in miss]):
            try:
                o = fut.result()
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                f.write(json.dumps(o) + "\n")
                f.flush()
                n[0] += 1
                el = time.time() - t0
                if n[0] % 5 == 0 or n[0] == len(miss):
                    print(f"  {n[0]}/{len(miss)}  credits {_credits[0]}  depth-hits "
                          f"{sum(1 for _ in [0]):d}  {n[0]/max(el,.001)*60:.1f} claims/min  "
                          f"ETA {(len(miss)-n[0])/max(n[0]/max(el,.001),1e-6)/60:.1f}m", flush=True)


# ------------------------------------------------------------------ S3 the diagnosis
# The ladder is flat and depth is empty, so the "which rung wins" table cannot carry the
# diagnosis. The question becomes: WHAT KIND OF DOCUMENT is the reviewer's source, and is
# it the kind a claim-derived query could ever rank? Two things are asked of the model,
# both from the CLAIM and the TARGET's title/url only (no page fetch, no verdict text):
#   relation  is this a document ABOUT the claim (a debunk, a report of the event) or a
#             document that merely CONTAINS the contradicting fact (a primary record, a
#             register, background) -- the latter shares no surface with the claim
#   mode      does it refute by SUBSTITUTION (the claim names a slot, the record shows a
#             different occupant), by ABSENCE (the record would list it and does not), or
#             by DIRECT CONTRADICTION (the document asserts the claim is false)
S3 = OUT / "stage3.jsonl"

S3_SYS = (
    "You are shown a CLAIM that a fact-checker or a Community Notes reviewer rated false, "
    "and the TITLE and URL of one source that reviewer cited. Judge the SOURCE's relation to "
    "the CLAIM from the title and url alone. Do not guess at page contents you cannot see.\n"
    "relation, exactly one of:\n"
    "  direct_debunk       the source is itself about the claim being false or disputed\n"
    "  event_report        the source reports the same event the claim is about, in the "
    "claim's own terms\n"
    "  primary_record      an official register, database, filing, statistic, transcript, "
    "ruling or press release that CONTAINS the settling fact but is not about the claim\n"
    "  background          general context on the topic, not about this event\n"
    "  not_evidence        a profile page, section index, navigation, product page, or a "
    "link that could not settle anything\n"
    "mode, exactly one of:\n"
    "  substitution        the claim names a slot and the source shows a different occupant "
    "(different person, place, date, number, cause)\n"
    "  absence             the source is where the claimed thing would appear and it is not there\n"
    "  contradiction       the source states directly that the claim is untrue\n"
    "  none                the source settles nothing\n"
    "shares_naming: true if the source's title uses the same name for the event/thing that the "
    "claim uses, false if it names it differently or does not name it.\n"
    'Respond JSON only: {"relation": "...", "mode": "...", "shares_naming": true|false, '
    '"why": "one short sentence"}')


def run_stage3(workers: int) -> None:
    res = [json.loads(l) for l in S2.open()]
    done = {json.loads(l)["key"] + "|" + json.loads(l)["url"]
            for l in S3.open()} if S3.exists() else set()
    jobs = []
    for r in res:
        for t in r["targets"]:
            if r["key"] + "|" + t["url"] not in done:
                jobs.append((r, t))
    print(f"stage3: {len(jobs)} (claim, target) pairs | {workers} workers", flush=True)

    def one(job):
        r, t = job
        msg = (f"CLAIM: {r['claim']}\nSOURCE TITLE: {t.get('title') or '(none)'}\n"
               f"SOURCE URL: {t['url']}")
        for _ in range(3):
            try:
                j, c, _, _ = llm([{"role": "system", "content": S3_SYS},
                                  {"role": "user", "content": msg}], max_tokens=200)
                if j.get("relation"):
                    return {"key": r["key"], "url": t["url"], "corpus": r["corpus"],
                            "claim": r["claim"], "title": t.get("title"),
                            "reached": bool(r["first_hit"]), "first_hit": r["first_hit"],
                            "cost": c, **{k: j.get(k) for k in
                                          ("relation", "mode", "shares_naming", "why")}}
            except Exception:  # noqa: BLE001
                time.sleep(2)
        return None

    lock, n, cost = threading.Lock(), [0], [0.0]
    with S3.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, j) for j in jobs]):
            o = fut.result()
            if not o:
                continue
            with lock:
                f.write(json.dumps(o) + "\n")
                f.flush()
                n[0] += 1
                cost[0] += o["cost"]
                if n[0] % 20 == 0 or n[0] == len(jobs):
                    print(f"  {n[0]}/{len(jobs)}  ${cost[0]:.4f}", flush=True)


# --------------------------------------------------------- S3b named failure patterns
# Read from the S3 examples (clog/280826): the reasons a reviewer-cited document cannot be
# ranked by a claim-derived query are NOT the query-shape faults the taxonomy predicted
# (date token, over-generic). They are properties of the DOCUMENT's relation to the claim.
# Every pattern below is decidable at production time from the claim alone or from the
# retrieved set, which is what makes them actionable.
S3B = OUT / "stage3b.jsonl"

S3B_SYS = (
    "You are shown a CLAIM rated false, and the TITLE and URL of one source the reviewer "
    "cited. Choose the ONE pattern that best describes why a web search built from the CLAIM "
    "would fail to rank this SOURCE.\n"
    "  direct_match        the source is about exactly this claim, using the claim's own "
    "entities and wording - a query from the claim SHOULD rank it\n"
    "  origin_source       the source is where the claim came from: the article, post or "
    "satire piece being fact-checked. It states the claim, it does not refute it\n"
    "  class_level_debunk  the source debunks the general kind of claim but never names this "
    "claim's specific people, places or numbers\n"
    "  different_framing   the source is about the same underlying fact but names a different "
    "principal actor, object or event than the claim does\n"
    "  other_language      the source is written in a different language from the claim\n"
    "  adjacent_event      the source is about a different but similar event - other date, "
    "other place, other person\n"
    "  not_evidence        a profile, section index, tag page, navigation or product page\n"
    'Respond JSON only: {"pattern": "...", "why": "one short sentence"}')


def run_stage3b(workers: int) -> None:
    s2 = {json.loads(l)["key"]: json.loads(l) for l in S2.open()}
    s3 = [json.loads(l) for l in S3.open()]
    done = {json.loads(l)["key"] + "|" + json.loads(l)["url"]
            for l in S3B.open()} if S3B.exists() else set()
    jobs = [r for r in s3 if r["key"] + "|" + r["url"] not in done]
    print(f"stage3b: {len(jobs)} pairs | {workers} workers", flush=True)

    def one(r):
        msg = (f"CLAIM: {r['claim']}\nSOURCE TITLE: {r.get('title') or '(none)'}\n"
               f"SOURCE URL: {r['url']}")
        for _ in range(3):
            try:
                j, c, _, _ = llm([{"role": "system", "content": S3B_SYS},
                                  {"role": "user", "content": msg}], max_tokens=160)
                if j.get("pattern"):
                    hits = {u for v in s2[r["key"]]["rungs"].values()
                            for u in (v.get("hit_urls") or [])}
                    return {"key": r["key"], "url": r["url"], "corpus": r["corpus"],
                            "claim": r["claim"], "title": r.get("title"),
                            "relation": r["relation"], "mode": r["mode"],
                            "pair_reached": any(_norm(u) == _norm(r["url"]) for u in hits),
                            "query_prod": s2[r["key"]]["queries"]["r1"],
                            "first_hit": s2[r["key"]]["first_hit"],
                            "pattern": j["pattern"], "why": j.get("why"), "cost": c}
            except Exception:  # noqa: BLE001
                time.sleep(2)
        return None

    lock, n, cost = threading.Lock(), [0], [0.0]
    with S3B.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(one, j) for j in jobs]):
            o = fut.result()
            if not o:
                continue
            with lock:
                f.write(json.dumps(o) + "\n")
                f.flush()
                n[0] += 1
                cost[0] += o["cost"]
                if n[0] % 20 == 0 or n[0] == len(jobs):
                    print(f"  {n[0]}/{len(jobs)}  ${cost[0]:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--dates", action="store_true")
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--stage2b", action="store_true")
    ap.add_argument("--stage2c", action="store_true")
    ap.add_argument("--stage3", action="store_true")
    ap.add_argument("--stage3b", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.sample:
        build_sample()
    if a.dates:
        run_dates(a.workers)
    if a.stage1:
        run_stage1(a.workers)
    if a.stage2:
        run_stage2(a.workers)
    if a.stage2b:
        run_stage2b(a.workers)
    if a.stage2c:
        run_stage2c(a.workers)
    if a.stage3:
        run_stage3(a.workers)
    if a.stage3b:
        run_stage3b(a.workers)








