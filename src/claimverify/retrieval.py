"""Serper request shape, result policy, and page scraping — copied from pipeline/search.py
(2026-07-28 consolidation) with two additions: the RAW Serper JSON is cached and returned
alongside the parsed list, and scrape() reports HOW it got the text.

Date ceiling: ISO YYYY-MM-DD -> Google `tbs` custom range `cdr:1,cd_min:1/1/1900,
cd_max:M/D/YYYY` (US order, no zero padding). Probed 2026-09-07: every result dated on or
before the ceiling. (ClaimCheck's own code sends DD/MM/YYYY, which Google ignores for days
> 12 and mis-reads for days <= 12 — we send the intended ceiling, not their string.)
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
from urllib.parse import urlparse

import requests
import trafilatura

from claimverify import disk_cache
from claimverify.config import (
    DOMAIN_DELAY_MAX, DOMAIN_DELAY_MIN, JINA_API_KEY, JINA_READER_ENDPOINT, SCRAPE_BLOCKLIST,
    SCRAPE_TIMEOUT, SERPER_API_KEY, SRC,
)
from claimverify.credibility import is_primary_source

# --- Serper: server-side social exclusion (within Google's ~32-word operator budget) ------
_SOCIAL_PRIORITY = [
    "youtube.com", "tiktok.com", "x.com", "twitter.com", "facebook.com", "reddit.com",
    "instagram.com", "scribd.com", "medium.com", "quora.com", "threads.com", "pinterest.com",
    "linkedin.com", "fandom.com", "tumblr.com", "researchgate.net",
]
_SITE_OP_BUDGET = 16
_SOCIAL_EXCLUDES = [d for d in _SOCIAL_PRIORITY if d in SCRAPE_BLOCKLIST] + \
    sorted(SCRAPE_BLOCKLIST - set(_SOCIAL_PRIORITY))
_SERPER_SITE_EXCLUDE = "".join(f" -site:{d}" for d in _SOCIAL_EXCLUDES[:_SITE_OP_BUDGET])
_BLOCKLIST_TAG = "soc-" + hashlib.sha1(",".join(_SOCIAL_EXCLUDES).encode()).hexdigest()[:8]
_RESULT_POLICY_TAG = {"serper": "rp2"}
_SERVER_XD_MAX = 3


def blocklist_tag(blocklist: bool = True) -> str:
    """Cache-key component. With the blocklist off the Serper query carries no `-site:`
    suffix, so the key MUST differ or a no-blocklist run would replay a blocklisted call.
    """
    return _BLOCKLIST_TAG if blocklist else "soc-off"


def tbs_date_ceiling(date_ceiling: str) -> str:
    """ISO YYYY-MM-DD -> Google tbs string; "" on an unparseable date (filter omitted)."""
    try:
        d = datetime.strptime(date_ceiling, "%Y-%m-%d")
    except ValueError:
        return ""
    return f"cdr:1,cd_min:1/1/1900,cd_max:{d.month}/{d.day}/{d.year}"


_HIT_DATE_FORMATS = ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y")


def parse_hit_date(raw: str | None) -> str | None:
    """Serper's free-text result date -> ISO, or None when it does not parse (relative
    dates like "3 days ago" never do)."""
    s = (raw or "").strip()
    for f in _HIT_DATE_FORMATS:
        try:
            return datetime.strptime(s, f).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def post_ceiling_counts(hits: list[dict], date_ceiling: str | None) -> tuple[int, int]:
    """(hits dated strictly after the ceiling, hits with no parsable date). Measurement
    only — the caller filters nothing."""
    post = undated = 0
    for h in hits:
        d = parse_hit_date(h.get("date"))
        if d is None:
            undated += 1
        elif date_ceiling and d > date_ceiling[:10]:
            post += 1
    return post, undated


def cache_key(provider: str, query: str, top_k: int, date_ceiling: str | None,
              exclude_domains: list[str] | None, min_results: int = 0,
              blocklist: bool = True) -> str:
    """Identical to pipeline/search.cache_key so entries are shared with the production
    loop and the urn runs (same query + top_k + ceiling + exclusions => one Serper call)."""
    xd = sorted(exclude_domains or [])
    parts = [query, str(top_k), date_ceiling or "", blocklist_tag(blocklist)]
    if tag := _RESULT_POLICY_TAG.get(provider):
        parts.append(tag)
    if xd:
        parts += ["xd-" + hashlib.sha1(",".join(xd).encode()).hexdigest()[:8],
                  f"min{min_results}",
                  "sxd1" if len(xd) <= _SERVER_XD_MAX else "cxd"]
    return disk_cache.make_key(*parts)


def serper_payload(query: str, top_k: int, date_ceiling: str | None,
                   exclude_domains: list[str] | None = None, page: int = 1,
                   blocklist: bool = True) -> dict:
    site_exclude = _SERPER_SITE_EXCLUDE if blocklist else ""
    xd = sorted(exclude_domains or [])
    if xd and len(xd) <= _SERVER_XD_MAX:
        site_exclude += "".join(f" -site:{d}" for d in xd)
    payload = {"q": query + site_exclude, "num": top_k}
    if page > 1:
        payload["page"] = page
    if date_ceiling and (tbs := tbs_date_ceiling(date_ceiling)):
        payload["tbs"] = tbs
    return payload


def serper_headers() -> dict:
    return {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}


def serper_parse(data: dict, top_k: int) -> list[dict]:
    """Serper JSON -> result dicts. PDFs are KEPT (scrape() strips their text)."""
    out: list[dict] = []
    for r in data.get("organic", []):
        link = r.get("link", "")
        if link:
            out.append({"url": link, "snippet": (r.get("snippet") or "").strip(),
                        "date": (r.get("date") or "").strip() or None, "content": None})
        if len(out) >= top_k:
            break
    return out


def _drop_domains(results: list[dict], domains: set[str]) -> list[dict]:
    out = []
    for r in results:
        host = urlparse(r["url"]).netloc.lower().removeprefix("www.")
        if host in domains or any(host.endswith("." + d) for d in domains):
            continue
        out.append(r)
    return out


def serper_finalize(results: list[dict], exclude_domains: list[str] | None = None,
                    blocklist: bool = True) -> list[dict]:
    """Guaranteed client-side drop: full blocklist (unless off), then the caller's
    exclusions."""
    if blocklist:
        results = _drop_domains(results, SCRAPE_BLOCKLIST)
    if exclude_domains:
        results = _drop_domains(results, set(exclude_domains))
    return results


def serper_merge(page1: list[dict], page2: list[dict]) -> list[dict]:
    seen = {r["url"] for r in page1}
    return page1 + [r for r in page2 if r["url"] not in seen]


# --- Scraping ---------------------------------------------------------------------------
_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]
_domain_locks: dict[str, threading.Lock] = {}
_domain_last_seen: dict[str, float] = {}
_global_lock = threading.Lock()

NEWSGUARD_RELIABLE_CUTOFF = 60.0
THIN_TEXT_CHARS = 400
_JINA_FIRST = frozenset({"wsj.com"})
_NEWSGUARD_PARQUET = SRC / "eval" / "data" / "newsguard" / "newsguard_scores.parquet"
_SCIENCE_PUBLISHERS = frozenset({
    "nature.com", "science.org", "sciencemag.org", "sciencedirect.com", "thelancet.com",
    "nejm.org", "pnas.org", "cell.com", "springer.com", "link.springer.com",
    "wiley.com", "onlinelibrary.wiley.com", "jamanetwork.com", "bmj.com", "nih.gov",
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "who.int", "cdc.gov",
    "europepmc.org", "arxiv.org", "ssrn.com", "jstor.org",
    "mdpi.com", "tandfonline.com", "taylorfrancis.com", "sagepub.com",
    "journals.sagepub.com", "frontiersin.org", "biomedcentral.com", "cambridge.org",
    "academic.oup.com", "oup.com", "researchgate.net", "plos.org", "journals.plos.org",
    "elifesciences.org", "acs.org", "pubs.acs.org", "iopscience.iop.org",
    "annualreviews.org", "karger.com", "biorxiv.org", "medrxiv.org", "osf.io",
})


def _is_jina_first(domain: str) -> bool:
    domain = domain.lower().removeprefix("www.")
    return domain in _JINA_FIRST or any(domain.endswith("." + d) for d in _JINA_FIRST)


@functools.lru_cache(maxsize=1)
def newsguard_score_map() -> dict:
    """domain -> NewsGuard score, from the local parquet (proprietary, gitignored; {} if
    missing — then every outlet is UNRATED and the closing bar is nearly unreachable)."""
    try:
        import pandas as pd
        df = pd.read_parquet(_NEWSGUARD_PARQUET, columns=["domain", "score"])
    except Exception:
        return {}
    return {d.lower().removeprefix("www."): float(s)
            for d, s in zip(df["domain"], df["score"]) if isinstance(d, str)}


@functools.lru_cache(maxsize=1)
def _newsguard_reliable() -> frozenset:
    return frozenset(d for d, s in newsguard_score_map().items() if s >= NEWSGUARD_RELIABLE_CUTOFF)


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
_WALL_MAX_CHARS = 2500


def no_content(text: str | None) -> str | None:
    """Reason code if `text` is not a readable document, else None."""
    if not text:
        return "empty"
    t = text.strip()
    if len(t) < THIN_TEXT_CHARS:
        return "thin"
    if len(t) <= _WALL_MAX_CHARS:
        for reason, pat in _WALL_PATTERNS:
            if pat.search(t):
                return reason
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 12:
        short = sum(1 for ln in lines if len(ln.strip()) < 45)
        if short / len(lines) > 0.85 and t.count(". ") < len(lines) * 0.08:
            return "nav_shell"
    return None


def _is_high_value(domain: str) -> bool:
    domain = domain.lower().removeprefix("www.")
    if domain.endswith((".gov", ".mil", ".edu")):
        return True
    if is_primary_source(domain):
        return True
    if domain in _SCIENCE_PUBLISHERS or any(domain.endswith("." + d) for d in _SCIENCE_PUBLISHERS):
        return True
    ng = _newsguard_reliable()
    return domain in ng or any(domain.endswith("." + d) for d in ng)


def _scrape_jina(url: str) -> str | None:
    if not JINA_API_KEY:
        return None
    try:
        headers = {"Authorization": f"Bearer {JINA_API_KEY}", "X-Return-Format": "text"}
        resp = requests.get(JINA_READER_ENDPOINT + url, headers=headers, timeout=SCRAPE_TIMEOUT * 3)
        resp.raise_for_status()
        text = resp.text.strip()
    except Exception:
        return None
    return None if no_content(text) else text


def _pdf_text(data: bytes, max_pages: int = 40) -> str | None:
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = [pg.extract_text() or "" for pg in reader.pages[:max_pages]]
        text = "\n".join(pages).strip()
        return text or None
    except Exception:
        return None


def _wait_for_domain(domain: str) -> None:
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


def scrape_detail(url: str, blocklist: bool = True) -> dict:
    """Fetch + extract main content. Returns {"text": str|None, "source": cache|requests|
    jina|pdf|blocklist|miss, "reason": no_content code or None, "thin": bool}.
    Same policy as pipeline/search.scrape(): blocklist -> Jina-first domains -> per-domain
    delay -> requests+trafilatura (or pypdf) -> content test -> Jina retry for high-value
    domains -> keep a stub rather than lose it. Cache entries are shared with the
    production loop; stale None/stub entries get the one-time PDF/Jina unpin retries."""
    cached = disk_cache.get("scrape", url)
    if cached is not None and (blocklist or cached.get("source") != "blocklist"):
        ctext = cached.get("text")
        stale = ctext is None or cached.get("thin") or len(ctext or "") <= THIN_TEXT_CHARS
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
                return {"text": fresh, "source": "pdf", "reason": None, "thin": False}
            cached["pdf_retried"] = True
            disk_cache.set_("scrape", url, cached)
        if stale and not cached.get("jina_retried") and cached.get("source") != "jina":
            dom = urlparse(url).netloc.lower().removeprefix("www.")
            blocked = dom in SCRAPE_BLOCKLIST or any(dom.endswith("." + d) for d in SCRAPE_BLOCKLIST)
            if not blocked and _is_high_value(dom):
                jina = _scrape_jina(url)
                if jina is not None and len(jina) > len(ctext or ""):
                    disk_cache.set_("scrape", url, {"text": jina, "source": "jina"})
                    return {"text": jina, "source": "jina", "reason": None, "thin": False}
                cached["jina_retried"] = True
                disk_cache.set_("scrape", url, cached)
        return {"text": ctext, "source": "cache:" + (cached.get("source") or "requests"),
                "reason": cached.get("no_content") if ctext else "cached-miss",
                "thin": bool(cached.get("thin"))}

    domain = urlparse(url).netloc.lower().removeprefix("www.")
    if blocklist and (domain in SCRAPE_BLOCKLIST or
                      any(domain.endswith("." + d) for d in SCRAPE_BLOCKLIST)):
        disk_cache.set_("scrape", url, {"text": None, "source": "blocklist"})
        return {"text": None, "source": "blocklist", "reason": "blocklist", "thin": False}

    jina_tried = False
    if _is_jina_first(domain):
        jina_tried = True
        jina = _scrape_jina(url)
        if jina is not None:
            disk_cache.set_("scrape", url, {"text": jina, "source": "jina"})
            return {"text": jina, "source": "jina", "reason": None, "thin": False}

    _wait_for_domain(domain)
    transient = False
    text = None
    status = None
    source = "requests"
    try:
        resp = requests.get(url, headers={"User-Agent": random.choice(_USER_AGENTS)},
                            timeout=SCRAPE_TIMEOUT)
        status = resp.status_code
        if resp.status_code not in (403, 404):
            resp.raise_for_status()
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype == "application/pdf" or url.split("?")[0].lower().endswith(".pdf"):
                text = _pdf_text(resp.content)
                source = "pdf"
            else:
                text = trafilatura.extract(resp.text, include_comments=False, include_tables=True)
    except Exception:
        transient = True

    miss_reason = None
    if text:
        text = text.strip()
        if not (miss_reason := no_content(text)):
            disk_cache.set_("scrape", url, {"text": text, "source": source})
            return {"text": text, "source": source, "reason": None, "thin": False, "status": status}

    if not jina_tried and _is_high_value(domain):
        jina = _scrape_jina(url)
        if jina is not None:
            disk_cache.set_("scrape", url, {"text": jina, "source": "jina"})
            return {"text": jina, "source": "jina", "reason": None, "thin": False, "status": status}

    if text:
        disk_cache.set_("scrape", url, {"text": text, "thin": True, "no_content": miss_reason})
        return {"text": text, "source": source, "reason": miss_reason, "thin": True, "status": status}
    if not transient:
        disk_cache.set_("scrape", url, {"text": None})
    return {"text": None, "source": "miss", "reason": "transient" if transient else (miss_reason or "empty"),
            "thin": False, "status": status}


def scrape(url: str) -> str | None:
    return scrape_detail(url)["text"]
