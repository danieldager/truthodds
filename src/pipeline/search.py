"""Web search (Serper or Tavily) + URL scrape utilities. Infrastructure for Tier 3.

Provider is selected by `config.SEARCH_PROVIDER` (or the per-call `provider=` arg):
  - "serper": Google SERP JSON; results carry snippets only → the verifier scrapes.
  - "tavily": agent-native search; results carry snippet (`content`) AND full cleaned page
    text (`raw_content`), so the verifier can skip the scrape step.
Genuine empty results (HTTP 200, no hits) return [] and are cached. API failures raise
SearchError so they surface as counted errors instead of looking like "no evidence".
"""
from __future__ import annotations

import functools
import hashlib
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

import requests
import trafilatura

from config import (
    BRAVE_API_KEY,
    BRAVE_ENDPOINT,
    EXA_API_KEY,
    EXA_ENDPOINT,
    JINA_API_KEY,
    JINA_READER_ENDPOINT,
    SEARCH_PROVIDER,
    SEARXNG_ENDPOINT,
    SERPER_API_KEY,
    SERPER_ENDPOINT,
    TAVILY_API_KEY,
    TAVILY_ENDPOINT,
)
from pipeline import disk_cache
from pipeline.config import (
    DOMAIN_DELAY_MAX,
    DOMAIN_DELAY_MIN,
    SCRAPE_BLOCKLIST,
    SCRAPE_TIMEOUT,
)


# Social/login-wall domains excluded server-side at each provider (same SCRAPE_BLOCKLIST the
# post-retrieval rerank/scrape already drop). Doing it at the source reclaims result slots so
# all SEARCH_RETRIEVE_K come back as usable, non-social candidates. _BLOCKLIST_TAG folds the
# list's contents into the cache key, so editing SCRAPE_BLOCKLIST invalidates stale entries.
#
# Google counts each `-site:` operator against its ~32-word query budget and SILENTLY drops
# the excess. The 2026-07-25 E1 pilot proved the failure mode: the list was sent sorted
# alphabetically, so tiktok/twitter/x/youtube — the highest-traffic UGC — sorted past the
# budget and leaked back in (334 results, 8.3% of the urn). Server-side operators are now
# (1) PRIORITY-ORDERED by observed leak frequency and (2) capped at _SITE_OP_BUDGET; the
# FULL blocklist is additionally enforced client-side in _fetch_serper (guaranteed drop —
# a leaked social costs a result slot, but can never become evidence again).
_SOCIAL_PRIORITY = [
    "youtube.com", "tiktok.com", "x.com", "twitter.com", "facebook.com", "reddit.com",
    "instagram.com", "scribd.com", "medium.com", "quora.com", "threads.com", "pinterest.com",
    "linkedin.com", "fandom.com", "tumblr.com", "researchgate.net",
]
_SITE_OP_BUDGET = 16  # server-side -site: operators reserved for socials (query words + caller xd use the rest)
_SOCIAL_EXCLUDES = [d for d in _SOCIAL_PRIORITY if d in SCRAPE_BLOCKLIST] + \
    sorted(SCRAPE_BLOCKLIST - set(_SOCIAL_PRIORITY))
_SERPER_SITE_EXCLUDE = "".join(f" -site:{d}" for d in _SOCIAL_EXCLUDES[:_SITE_OP_BUDGET])
_BLOCKLIST_TAG = "soc-" + hashlib.sha1(",".join(_SOCIAL_EXCLUDES).encode()).hexdigest()[:8]
# Bump when the RESULT-SHAPING policy changes (what gets dropped, PDFs in/out, backfill) —
# distinct from _BLOCKLIST_TAG, which only tracks the list's contents. "rp2" (2026-07-28)
# rotates every Serper entry written before the pools.py/search.py consolidation, when the
# two paths applied different policies to the same cache key. Exa/Tavily deliberately have
# no tag: their entries are paid/metered and their result policy did not change here.
_RESULT_POLICY_TAG = {"serper": "rp2"}
# A caller exclusion of at most this many domains is ALSO sent server-side (origin outlet +
# linked article). Larger lists (the eval fact-check block) stay client-side only.
_SERVER_XD_MAX = 3


class SearchError(Exception):
    """A search API call failed (HTTP error, quota, timeout) — distinct from 0 results."""

_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

_domain_locks: Dict[str, threading.Lock] = {}
_domain_last_seen: Dict[str, float] = {}
_global_lock = threading.Lock()

# --- SearXNG pacing: it scrapes Google/Brave/etc, which CAPTCHA/throttle under load.
# Enforce a minimum gap between SearXNG calls across all worker threads. ---
_SEARXNG_MIN_INTERVAL = 1.0
_searxng_lock = threading.Lock()
_searxng_last_call = 0.0

# Brave metered plan = 50 QPS (per the X-RateLimit headers). Pace just under that; the
# 429-retry in _request handles any overshoot. (The old "free" tier was 1 QPS — retired.)
_BRAVE_MIN_INTERVAL = 0.03  # ~33 QPS, safely under the 50 QPS limit
_brave_lock = threading.Lock()
_brave_last_call = 0.0

_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3


def _searxng_pace() -> None:
    global _searxng_last_call
    with _searxng_lock:
        wait = _SEARXNG_MIN_INTERVAL - (time.time() - _searxng_last_call)
        if wait > 0:
            time.sleep(wait)
        _searxng_last_call = time.time()


def _brave_pace() -> None:
    global _brave_last_call
    with _brave_lock:
        wait = _BRAVE_MIN_INTERVAL - (time.time() - _brave_last_call)
        if wait > 0:
            time.sleep(wait)
        _brave_last_call = time.time()


def _request(method: str, url: str, *, label: str, retry_empty=None, timeout: int = 30, **kw) -> dict:
    """HTTP with backoff retries on transient failures. Raises SearchError when exhausted.

    retry_empty(json) -> bool : optional predicate; if it returns True the response is
    treated as a (transient) throttle and retried — used for SearXNG, which returns 200 +
    empty results when its upstream engines are blocked.
    """
    last = ""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.request(method, url, timeout=timeout, **kw)
            if resp.status_code in _RETRY_STATUS:
                raise requests.RequestException(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            if retry_empty and retry_empty(data):
                raise requests.RequestException("throttled (empty + unresponsive engines)")
            return data
        except requests.RequestException as e:
            body = getattr(getattr(e, "response", None), "text", "") or ""
            last = f"{e} {body[:160]}".strip()
            if any(s in last for s in ("400", "401", "403")):  # permanent — don't retry
                break
            if attempt < _MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise SearchError(f"{label}: {last}")


def _tbs_date_ceiling(date_ceiling: str) -> str:
    """ISO YYYY-MM-DD -> Google `tbs` custom-date-range string with no upper-bound leakage.

    Google wants US M/D/YYYY (no zero-padding). Returns "" on an unparseable date so the
    caller simply omits the filter rather than erroring.
    """
    try:
        d = datetime.strptime(date_ceiling, "%Y-%m-%d")
    except ValueError:
        return ""
    cd_max = f"{d.month}/{d.day}/{d.year}"
    return f"cdr:1,cd_min:1/1/1900,cd_max:{cd_max}"


def cache_key(provider: str, query: str, top_k: int, date_ceiling: str | None,
              exclude_domains: list[str] | None, min_results: int = 0) -> str:
    """THE cache key for a search call — every caller (sync `search()` and the async
    pools) must build its key here, or the two paths silently share entries written
    under different result policies (postmortem: clog/270726, pools.py divergence)."""
    xd = sorted(exclude_domains or [])
    parts = [query, str(top_k), date_ceiling or "", _BLOCKLIST_TAG]
    if tag := _RESULT_POLICY_TAG.get(provider):
        parts.append(tag)
    if xd:  # only extend when excluding, so production reuses existing entries
        parts += ["xd-" + hashlib.sha1(",".join(xd).encode()).hexdigest()[:8],
                  f"min{min_results}",
                  "sxd1" if len(xd) <= _SERVER_XD_MAX else "cxd"]
    return disk_cache.make_key(*parts)


def search(query: str, top_k: int, *, date_ceiling: str | None = None,
           provider: str | None = None, exclude_domains: list[str] | None = None,
           min_results: int = 0, stats: dict | None = None) -> list[dict]:
    """Run a web search via the configured provider.

    Returns up to top_k results as dicts: {"url", "snippet", "date": str|None,
    "content": str|None}. `content` is the full cleaned page text when the provider
    supplies it (Tavily) — the verifier uses it instead of scraping; None otherwise.

    date_ceiling (ISO YYYY-MM-DD): restrict to pages published on/before that date
    (eval-time leakage control). Left None in production for the freshest evidence.

    exclude_domains: extra domains excluded on top of the social blocklist — the eval-only
    fact-check block (Serper filters these client-side since Google drops excess `-site:`
    operators; Tavily/Exa exclude them server-side). min_results: when excluding those starves a
    Serper query below this many results, a second page is pulled and merged in (so the verifier
    still sees ~5 sources). Both are no-ops in production (empty / 0), leaving cache keys unchanged.
    Raises SearchError on an API failure (so it isn't mistaken for "no evidence").
    """
    provider = provider or SEARCH_PROVIDER
    xd = sorted(exclude_domains or [])
    key = cache_key(provider, query, top_k, date_ceiling, xd, min_results)
    results = disk_cache.get(provider, key)
    if results is None:
        fetch = {"serper": _fetch_serper, "tavily": _fetch_tavily, "brave": _fetch_brave,
                 "searxng": _fetch_searxng, "exa": _fetch_exa}.get(provider)
        if fetch is None:
            raise SearchError(f"unknown search provider {provider!r}")
        results = fetch(query, top_k, date_ceiling, xd, min_results, stats)
        disk_cache.set_(provider, key, results)  # cache only on success (errors raised above)

    for r in results:
        r.setdefault("provider", provider)  # tag source provenance (covers cache hits too)
    return results


# ---------------------------------------------------------------------------
# Serper primitives — THE single definition of a Serper call and what its results
# mean. The async SerperPool (pipeline/pools.py) owns concurrency/retry/breaker and
# calls exactly these, so there is one place where the request shape, the PDF policy,
# the blocklist drop and the backfill rule live. Never re-implement them elsewhere:
# the 2026-07-28 audit found the pool had drifted into dropping PDFs and skipping the
# blocklist entirely, while sharing a cache key with this path.
# ---------------------------------------------------------------------------

def serper_payload(query: str, top_k: int, date_ceiling: str | None,
                   exclude_domains: list[str] | None = None, page: int = 1) -> dict:
    """The Serper request body: socials excluded server-side (within Google's operator
    budget), plus a SMALL caller exclusion so Google backfills the slot for us."""
    site_exclude = _SERPER_SITE_EXCLUDE
    xd = sorted(exclude_domains or [])
    if xd and len(xd) <= _SERVER_XD_MAX:
        site_exclude += "".join(f" -site:{d}" for d in xd)
    payload = {"q": query + site_exclude, "num": top_k}
    if page > 1:
        payload["page"] = page
    if date_ceiling and (tbs := _tbs_date_ceiling(date_ceiling)):
        payload["tbs"] = tbs
    return payload


def serper_headers() -> dict:
    return {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}


def serper_parse(data: dict, top_k: int) -> list[dict]:
    """Serper JSON -> our result dicts. PDFs are KEPT (since 2026-07-21 scrape() strips
    their text with pypdf, and PDFs are disproportionately primary sources)."""
    out: list[dict] = []
    for r in data.get("organic", []):
        link = r.get("link", "")
        if link:
            out.append({
                "url": link,
                "snippet": (r.get("snippet") or "").strip(),
                "date": (r.get("date") or "").strip() or None,
                "content": None,  # serper has no page body; verifier scrapes
            })
        if len(out) >= top_k:
            break
    return out


def serper_finalize(results: list[dict], exclude_domains: list[str] | None = None) -> list[dict]:
    """Guaranteed client-side drop: the FULL blocklist first (Google silently ignores
    -site: operators past its query-word budget, so server-side exclusion always leaks a
    tail), then the caller's exclusions."""
    results = _drop_domains(results, SCRAPE_BLOCKLIST)
    if exclude_domains:
        results = _drop_domains(results, set(exclude_domains))
    return results


def serper_merge(page1: list[dict], page2: list[dict]) -> list[dict]:
    """Append page-2 backfill, dropping URLs already present."""
    seen = {r["url"] for r in page1}
    return page1 + [r for r in page2 if r["url"] not in seen]


def _serper_page(query: str, top_k: int, date_ceiling: str | None,
                 exclude_domains: list[str], page: int) -> list[dict]:
    """One Serper page of organic results, synchronous (page>1 paginates)."""
    data = _request("POST", SERPER_ENDPOINT, label="serper",
                    json=serper_payload(query, top_k, date_ceiling, exclude_domains, page),
                    headers=serper_headers(), timeout=10)
    return serper_parse(data, top_k)


def _drop_domains(results: list[dict], domains: set[str]) -> list[dict]:
    """Filter out results whose host is (a subdomain of) any excluded domain."""
    out = []
    for r in results:
        host = urlparse(r["url"]).netloc.lower().removeprefix("www.")
        if host in domains or any(host.endswith("." + d) for d in domains):
            continue
        out.append(r)
    return out


def _fetch_serper(query: str, top_k: int, date_ceiling: str | None,
                  exclude_domains: list[str], min_results: int, stats: dict | None = None) -> list[dict]:
    # Socials/UGC are excluded SERVER-SIDE (`-site:` operators, within Google's budget).
    # Fact-check domains (a 25+ eval list) are filtered CLIENT-SIDE: Google silently ignores
    # excess `-site:` operators, so server-side exclusion of that many leaks (verified —
    # leadstories survived). If filtering FCs starves page 1 below min_results, pull page 2
    # and merge so the verifier keeps ~5.
    # 2026-07-21 (Daniel): a SMALL caller exclusion (the origin outlet + the article the post
    # links to, 1-2 domains) also goes server-side, so Google backfills the slot instead of us
    # deleting a result after the fact — those were a measurable slice of the sub-10 result
    # lists. The client-side filter still runs on everything: if Google ignores an operator,
    # the drop still happens, it just costs a slot as before.
    results = serper_finalize(_serper_page(query, top_k, date_ceiling, exclude_domains, 1),
                              exclude_domains)
    resampled = min_results and len(results) < min_results
    if resampled:  # FC block starved page 1 → page-2 backfill (retrieval-health signal)
        page2 = serper_finalize(_serper_page(query, top_k, date_ceiling, exclude_domains, 2),
                                exclude_domains)
        results = serper_merge(results, page2)
    if stats is not None:
        stats["resampled"] = bool(resampled)
    return results[: max(top_k, min_results)]


def _fetch_tavily(query: str, top_k: int, date_ceiling: str | None,
                  exclude_domains: list[str], min_results: int, stats: dict | None = None) -> list[dict]:
    payload = {
        "query": query,
        "max_results": top_k,
        "search_depth": "basic",       # 1 credit/search
        "include_raw_content": True,   # full cleaned page text -> skip scraping
        "exclude_domains": _SOCIAL_EXCLUDES + list(exclude_domains),  # socials + eval FC block
    }
    if date_ceiling:
        payload["end_date"] = date_ceiling  # YYYY-MM-DD, no upper-bound leakage
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    data = _request("POST", TAVILY_ENDPOINT, label="tavily", json=payload, headers=headers, timeout=30)
    results: list[dict] = []
    for r in data.get("results", []):
        link = r.get("url", "")
        if link:  # PDFs kept since 2026-07-21 — scrape() strips their text (pypdf)
            results.append({
                "url": link,
                "snippet": (r.get("content") or "").strip(),
                "date": r.get("published_date") or None,
                "content": (r.get("raw_content") or "").strip() or None,
            })
        if len(results) >= top_k:
            break
    return results


def _fetch_searxng(query: str, top_k: int, date_ceiling: str | None,
                   exclude_domains: list[str], min_results: int, stats: dict | None = None) -> list[dict]:
    # exclude_domains/min_results unsupported here (no server-side exclude); the eval runs on Serper.
    # No engine-level date filter; the verifier's post-filter handles date_ceiling.
    # Pace globally + retry: SearXNG returns 200 with empty results when upstream engines
    # (Google/Brave) throttle it, so treat empty+unresponsive as a transient throttle.
    _searxng_pace()
    params = {"q": query, "format": "json", "categories": "general", "language": "en"}
    data = _request("GET", SEARXNG_ENDPOINT, label="searxng", params=params, timeout=15,
                    retry_empty=lambda d: not d.get("results") and bool(d.get("unresponsive_engines")))
    results: list[dict] = []
    for r in data.get("results", []):
        link = r.get("url", "")
        if link:  # PDFs kept since 2026-07-21 — scrape() strips their text (pypdf)
            results.append({
                "url": link,
                "snippet": (r.get("content") or "").strip(),
                "date": (r.get("publishedDate") or "").strip() or None,
                "content": None,  # no page body; verifier scrapes
            })
        if len(results) >= top_k:
            break
    return results


def _fetch_exa(query: str, top_k: int, date_ceiling: str | None,
               exclude_domains: list[str], min_results: int, stats: dict | None = None) -> list[dict]:
    # Raw retrieval: per-result source url + highlights/text. NOT the synthesized output.
    payload = {
        "query": query,
        "type": "auto",
        "numResults": top_k,
        "excludeDomains": _SOCIAL_EXCLUDES + list(exclude_domains),  # socials + eval FC block
        "contents": {"highlights": True, "text": {"maxCharacters": 12000}},
    }
    if date_ceiling:
        payload["endPublishedDate"] = date_ceiling  # ISO YYYY-MM-DD
    headers = {"x-api-key": EXA_API_KEY, "Content-Type": "application/json"}
    data = _request("POST", EXA_ENDPOINT, label="exa", json=payload, headers=headers, timeout=30)
    results: list[dict] = []
    for r in data.get("results", []):
        link = r.get("url", "")
        if not link:  # PDFs kept since 2026-07-21 — scrape() strips their text (pypdf)
            continue
        highlights = [h.strip() for h in (r.get("highlights") or []) if h.strip()]
        text = (r.get("text") or "").strip()
        results.append({
            "url": link,
            "snippet": (" … ".join(highlights) or text[:300]).strip(),
            "date": r.get("publishedDate") or None,
            "content": text or None,
        })
        if len(results) >= top_k:
            break
    return results


def _fetch_brave(query: str, top_k: int, date_ceiling: str | None,
                 exclude_domains: list[str], min_results: int, stats: dict | None = None) -> list[dict]:
    # exclude_domains/min_results unsupported here (no server-side exclude); the eval runs on Serper.
    # Real keyed API (not scraped) → no IP blocking. Free tier = 1 query/sec, so pace globally.
    _brave_pace()
    params = {"q": query, "count": top_k}
    if date_ceiling:  # restrict to pages indexed on/before the ceiling (custom freshness range)
        params["freshness"] = f"1900-01-01to{date_ceiling}"
    headers = {"X-Subscription-Token": BRAVE_API_KEY, "Accept": "application/json"}
    data = _request("GET", BRAVE_ENDPOINT, label="brave", params=params, headers=headers, timeout=20)
    results: list[dict] = []
    for r in data.get("web", {}).get("results", []):
        link = r.get("url", "")
        if link:  # PDFs kept since 2026-07-21 — scrape() strips their text (pypdf)
            snippet = re.sub(r"<[^>]+>", "", r.get("description") or "").strip()  # strip <strong> highlights
            results.append({
                "url": link,
                "snippet": snippet,
                "date": (r.get("page_age") or r.get("age") or None),
                "content": None,  # web search returns snippets only; verifier scrapes
            })
        if len(results) >= top_k:
            break
    return results


# --- High-value Jina fallback ---------------------------------------------------------------
# When a normal scrape fails, we retry through the Jina Reader API (off-IP render, busts many
# paywalls/WAFs) — but ONLY for "high-value" domains, to conserve Jina credits. High value =
# government (.gov/.mil), academic (.edu + a curated science-publisher set), or NewsGuard-reliable
# (resolved per-domain score >= the reliable cutoff). Everything else stays None on a scrape miss.
NEWSGUARD_RELIABLE_CUTOFF = 60.0  # NewsGuard's "reliable" score threshold (0–100 scale)

# Below this a page is a paywall stub / lede teaser, not a readable article. Aligned with
# verify_tweet_claims's read floor so the two layers cannot disagree — see scrape() for why that matters.
THIN_TEXT_CHARS = 400

# Domains where Trafilatura essentially never returns a readable article (paywall/JS wall) but
# Jina Reader does — going Jina-FIRST skips a doomed round-trip. Seeded from the 2026-07-12
# scrape audit (wsj: 4/4 stubs). Grow it as evidence accumulates: a domain that repeatedly
# yields sub-THIN_TEXT_CHARS text is a candidate. Deliberately NOT here: nytimes.com — Jina
# fails there too (10/12 hard failures), so it would only burn credits. See roadmap: NYT API.
_JINA_FIRST = frozenset({"wsj.com"})


def _is_jina_first(domain: str) -> bool:
    domain = domain.lower().removeprefix("www.")
    return domain in _JINA_FIRST or any(domain.endswith("." + d) for d in _JINA_FIRST)

_NEWSGUARD_PARQUET = Path(__file__).resolve().parent.parent / "eval" / "data" / "newsguard" / "newsguard_scores.parquet"

# Curated paywalled science/academic publishers (authoritative but Trafilatura-hostile), on top
# of the .edu suffix rule and the NewsGuard-reliable set.
_SCIENCE_PUBLISHERS = frozenset({
    "nature.com", "science.org", "sciencemag.org", "sciencedirect.com", "thelancet.com",
    "nejm.org", "pnas.org", "cell.com", "springer.com", "link.springer.com",
    "wiley.com", "onlinelibrary.wiley.com", "jamanetwork.com", "bmj.com", "nih.gov",
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "who.int", "cdc.gov",
    "europepmc.org", "arxiv.org", "ssrn.com", "jstor.org",
    # Added 2026-07-27: the gate excluded exactly the publishers most likely to need a
    # Jina retry. 79% of the E1 pilot's "junk" reads were extraction failures, and the
    # top hosts were sciencedirect and university course/PDF pages. academic.oup.com,
    # jstor.org and researchgate.net were also unblocked from SCRAPE_BLOCKLIST today,
    # so they now reach this gate for the first time.
    "mdpi.com", "tandfonline.com", "taylorfrancis.com", "sagepub.com",
    "journals.sagepub.com", "frontiersin.org", "biomedcentral.com", "cambridge.org",
    "academic.oup.com", "oup.com", "researchgate.net", "plos.org", "journals.plos.org",
    "elifesciences.org", "acs.org", "pubs.acs.org", "iopscience.iop.org",
    "annualreviews.org", "karger.com", "biorxiv.org", "medrxiv.org", "osf.io",
})


@functools.lru_cache(maxsize=1)
def newsguard_score_map() -> dict:
    """domain -> NewsGuard score (0-100), loaded once from the local parquet. Empty dict if the
    file is missing (proprietary, gitignored — never crash). Used by verify_tweet_claims.rank_hits."""
    try:
        import pandas as pd
        df = pd.read_parquet(_NEWSGUARD_PARQUET, columns=["domain", "score"])
    except Exception:
        return {}
    return {d.lower().removeprefix("www."): float(s)
            for d, s in zip(df["domain"], df["score"]) if isinstance(d, str)}


@functools.lru_cache(maxsize=1)
def _newsguard_reliable() -> frozenset:
    """Domains NewsGuard rates >= NEWSGUARD_RELIABLE_CUTOFF, loaded once from the local parquet.
    Degrades to an empty set if the file is missing (proprietary, gitignored — never crash)."""
    try:
        import pandas as pd
        df = pd.read_parquet(_NEWSGUARD_PARQUET, columns=["domain", "score"])
    except Exception:
        return frozenset()
    hi = df[df["score"] >= NEWSGUARD_RELIABLE_CUTOFF]["domain"]
    return frozenset(d.lower().removeprefix("www.") for d in hi if isinstance(d, str))


_WALL_PATTERNS = (
    ("consent", re.compile(r"\b(accept (all )?cookies|cookie (policy|preferences|settings)|"
                           r"we (use|and our partners use) cookies|manage (your )?preferences|"
                           r"consent to the use of cookies|gdpr)\b", re.I)),
    ("botcheck", re.compile(r"\b(enable javascript|javascript is (required|disabled)|"
                            r"verify (you are|you're) (a )?human|are you a robot|checking your browser|"
                            r"unusual traffic|access denied|request blocked|cloudflare|captcha|"
                            r"403 forbidden|permission denied)\b", re.I)),
    ("login", re.compile(r"\b(sign in to (continue|read)|log ?in to (continue|read)|"
                         r"create (a free )?account to|subscribe to (continue|read)|"
                         r"this (content|article) is for subscribers|register to (continue|read))\b", re.I)),
    ("error", re.compile(r"\b(404 not found|page not found|this page (has moved|no longer exists)|"
                         r"the requested url was not found|service unavailable|"
                         r"we can't find the page)\b", re.I)),
    ("placeholder", re.compile(r"\blorem ipsum\b", re.I)),
)
# A page is "wall-like" when a pattern hits AND there is little else: real articles ABOUT
# cookie law or CAPTCHAs exist, so the pattern alone must not condemn a long document.
_WALL_MAX_CHARS = 2500


def no_content(text: str | None) -> str | None:
    """Reason code if `text` is not a readable document, else None.

    Success used to be judged by LENGTH alone, so an HTTP-200 consent banner or bot
    check sailed through as a successful scrape, was cached permanently, and never
    reached the Jina fallback. Measured in the E1 pilot: 79% of reads labelled "junk"
    were extraction failures (median 156 chars), concentrated on PDF and academic hosts,
    and 11.4% of pages fetched *via Jina* were themselves wall text. This judges the
    content instead (Daniel 2026-07-27)."""
    if not text:
        return "empty"
    t = text.strip()
    if len(t) < THIN_TEXT_CHARS:
        return "thin"
    if len(t) <= _WALL_MAX_CHARS:
        for reason, pat in _WALL_PATTERNS:
            if pat.search(t):
                return reason
    # nav/link shells: many short lines, almost no sentence punctuation
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 12:
        short = sum(1 for ln in lines if len(ln.strip()) < 45)
        if short / len(lines) > 0.85 and t.count(". ") < len(lines) * 0.08:
            return "nav_shell"
    return None


def _is_high_value(domain: str) -> bool:
    """True if `domain` is worth spending a Jina credit on (see the section comment above)."""
    domain = domain.lower().removeprefix("www.")
    if domain.endswith((".gov", ".mil", ".edu")):
        return True
    # the corroboration gate's PRIMARY set (non-US gov suffixes, IGOs, courts, stats agencies):
    # a source that can single-handedly close a claim is always worth a Jina credit. The old
    # check missed all of these — 2 measured PRIMARY losses (europarl, hansard), audit 2026-07-13.
    from pipeline.credibility import is_primary_source
    if is_primary_source(domain):
        return True
    if domain in _SCIENCE_PUBLISHERS or any(domain.endswith("." + d) for d in _SCIENCE_PUBLISHERS):
        return True
    ng = _newsguard_reliable()
    return domain in ng or any(domain.endswith("." + d) for d in ng)


def _scrape_jina(url: str) -> str | None:
    """Paywall/WAF fallback: fetch `url` off-IP via the Jina Reader API (r.jina.ai) and return
    its cleaned text. Returns None on any error/timeout/credit-exhaustion so the caller stays
    None-safe. Rendering is slower than a plain GET, so allow a longer timeout."""
    if not JINA_API_KEY:
        return None
    try:
        headers = {"Authorization": f"Bearer {JINA_API_KEY}", "X-Return-Format": "text"}
        resp = requests.get(JINA_READER_ENDPOINT + url, headers=headers, timeout=SCRAPE_TIMEOUT * 3)
        resp.raise_for_status()
        text = resp.text.strip()
    except Exception:
        return None
    # Jina output was accepted on length alone; 11.4% of cached Jina pages were themselves
    # wall text, which then pinned the cache permanently as a "successful" fetch.
    return None if no_content(text) else text


def _pdf_text(data: bytes, max_pages: int = 40) -> str | None:
    """Strip text from PDF bytes (pypdf). Page cap keeps 500-page reports from flooding
    the read cap — news-relevant substance fronts the document like everywhere else."""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [pg.extract_text() or "" for pg in reader.pages[:max_pages]]
        text = "\n".join(pages).strip()
        return text or None
    except Exception:
        return None


def scrape(url: str) -> str | None:
    """Fetch a URL and extract main content via Trafilatura.

    Returns None if the URL's domain is in SCRAPE_BLOCKLIST, the request fails
    or returns 403/404, or nothing readable survives. A page that Trafilatura renders as a
    sub-THIN_TEXT_CHARS stub counts as a MISS, not a hit: on any miss, high-value domains
    (_is_high_value) get ONE retry through the Jina Reader paywall fallback before giving up —
    low-value domains never spend a Jina credit. Known-paywalled domains (_is_jina_first) go
    straight to Jina and skip the doomed Trafilatura fetch.
    """
    cached = disk_cache.get("scrape", url)
    if cached is not None:
        ctext = cached.get("text")
        # Cache unpinning (audit 2026-07-13): entries written before the Jina fixes pin a
        # None/stub forever — 40/43 of the measured scrape-failed/too-short drops were still
        # dead in cache, so the fixes never fire on eval reruns. For high-value domains, retry
        # Jina ONCE per URL and overwrite; the attempt is marked so a truly dead URL costs
        # exactly one credit ever.
        stale = ctext is None or cached.get("thin") or len(ctext or "") <= THIN_TEXT_CHARS
        # PDF unpin (2026-07-21): entries cached before the pypdf extractor existed pin a
        # None forever for exactly the primary documents PDFs tend to be — retry ONCE
        if stale and not cached.get("pdf_retried") and cached.get("source") != "pdf" and \
                url.split("?")[0].lower().endswith(".pdf"):
            fresh = None
            try:
                r = requests.get(url, headers={"User-Agent": random.choice(_USER_AGENTS)},
                                 timeout=SCRAPE_TIMEOUT)
                if r.ok:
                    fresh = _pdf_text(r.content)
            except Exception:
                pass
            if fresh and len(fresh) > THIN_TEXT_CHARS:
                disk_cache.set_("scrape", url, {"text": fresh, "source": "pdf"})
                return fresh
            cached["pdf_retried"] = True
            disk_cache.set_("scrape", url, cached)
        if stale and not cached.get("jina_retried") and cached.get("source") != "jina":
            dom = urlparse(url).netloc.lower().removeprefix("www.")
            blocked = dom in SCRAPE_BLOCKLIST or any(dom.endswith("." + d) for d in SCRAPE_BLOCKLIST)
            if not blocked and _is_high_value(dom):
                jina = _scrape_jina(url)
                if jina is not None and len(jina) > len(ctext or ""):
                    disk_cache.set_("scrape", url, {"text": jina, "source": "jina"})
                    return jina
                cached["jina_retried"] = True
                disk_cache.set_("scrape", url, cached)
        return ctext

    domain = urlparse(url).netloc.lower().removeprefix("www.")
    if domain in SCRAPE_BLOCKLIST or any(domain.endswith("." + d) for d in SCRAPE_BLOCKLIST):
        disk_cache.set_("scrape", url, {"text": None})
        return None

    jina_tried = False
    if _is_jina_first(domain):   # Trafilatura is known to fail here — don't bother fetching
        jina_tried = True
        jina = _scrape_jina(url)
        if jina is not None:
            disk_cache.set_("scrape", url, {"text": jina, "source": "jina"})
            return jina

    _wait_for_domain(domain)

    transient = False  # network error (not a definitive miss) — don't cache None, let a run retry
    text = None
    try:
        headers = {"User-Agent": random.choice(_USER_AGENTS)}
        resp = requests.get(url, headers=headers, timeout=SCRAPE_TIMEOUT)
        if resp.status_code not in (403, 404):
            resp.raise_for_status()
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype == "application/pdf" or url.split("?")[0].lower().endswith(".pdf"):
                # PDFs (Daniel 2026-07-21): ~0.5% of Serper results, often exactly the
                # primary documents we want (reports, filings). Text-strip via pypdf;
                # READ consumes the result like any article text.
                text = _pdf_text(resp.content)
            else:
                # include_tables=True so vote tallies, stats blocks, and legislative data come through.
                # include_comments=False keeps comment threads out.
                # Default (include_formatting=False) flattens H1/H2/etc into prose; page chrome
                # (nav, footer, sidebar, ads) is stripped by Trafilatura's main-content detector.
                text = trafilatura.extract(resp.text, include_comments=False, include_tables=True)
    except Exception:
        transient = True

    if text:
        text = text.strip()
        # Content test, not a length test: a 200-response consent banner or bot check is
        # long enough to pass a length bar and would then be cached as a success forever.
        if not (reason := no_content(text)):
            disk_cache.set_("scrape", url, {"text": text})
            return text
        _miss_reason = reason

    # Trafilatura failed (403/404, transient) OR returned a paywall stub. The bar here used to be
    # >100 chars — but paywall lede teasers land at 141-350 chars, so they CLEARED that bar, were
    # returned as a "hit", and were then killed downstream by verify_tweet_claims's 400-char read floor.
    # The Jina fallback built for exactly this case therefore never fired: 531 such entries sat
    # stranded in the cache (measured 2026-07-12; wsj 4/4, reuters, nbcnews, sky, mprnews).
    # Aligning this bar with that floor closes the dead zone.
    if not jina_tried and _is_high_value(domain):
        jina = _scrape_jina(url)
        if jina is not None:                       # _scrape_jina already rejects walls
            disk_cache.set_("scrape", url, {"text": jina, "source": "jina"})
            return jina

    if text:  # a stub, and Jina couldn't beat it — keep it rather than lose it outright,
              # but record WHY it is degraded so the read step can tell a wall from a
              # genuinely empty document instead of both becoming "junk".
        disk_cache.set_("scrape", url, {"text": text, "thin": True,
                                        "no_content": locals().get("_miss_reason")})
        return text
    if not transient:
        disk_cache.set_("scrape", url, {"text": None})  # definitive miss — cache the None
    return None


def _wait_for_domain(domain: str) -> None:
    """Block until the per-domain delay has elapsed since the last request to this host."""
    with _global_lock:
        lock = _domain_locks.setdefault(domain, threading.Lock())
    with lock:
        now = time.time()
        last = _domain_last_seen.get(domain, 0.0)
        delay = random.uniform(DOMAIN_DELAY_MIN, DOMAIN_DELAY_MAX)
        wait = delay - (now - last)
        if wait > 0:
            time.sleep(wait)
        _domain_last_seen[domain] = time.time()
