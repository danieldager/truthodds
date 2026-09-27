"""Generic ClaimReview JSON-LD harvesting — shared by the JSON-LD-emitting publishers
(Lead Stories, Snopes, 20 Minutes, …).

`fetch()` does a browser-UA GET with a Wayback fallback (archive.org preserves the original
JSON-LD that some origins 403 or that reader proxies strip). `extract_claimreview()` pulls the
embedded ClaimReview into the shared record fields. The claim's source URL is checked across the
several spots publishers stash it (`itemReviewed.author.sameAs` for Lead Stories, `itemReviewed.
sameAs`/`url`, `appearance[].url` for others).
"""
from __future__ import annotations

import json
import re
import time
from html import unescape as _unescape  # imported by-name: `html` is shadowed by `html` params below

import requests

from config import JINA_API_KEY

UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


def fetch_jina(url: str, target_selector: str | None = None, return_format: str = "html",
               timeout: int = 60, retries: int = 3) -> str | None:
    """Fetch page content via the Jina Reader proxy (r.jina.ai) — fetches from Jina's own IP (and
    renders JS), bypassing Akamai/DataDome edge-blocks that 403 both plain requests and headless
    Chrome (e.g. AFP). A JINA_API_KEY raises the limit to 500 RPM. None on failure; backs off on 429.

    COST: Jina bills on the OUTPUT tokens only. `X-Target-Selector` returns ONLY the matching
    elements, so a tight selector (e.g. just the links you need) cuts the bill ~97% vs full-page
    html (~1-2k vs ~50k tok/article — clog 240626). `X-Retain-Images: none` trims further. Use
    `return_format="markdown"` for clean LLM-ready body text; keep "html" when you need attributes
    (markdown drops `rel`/`href` structure)."""
    headers = {"X-Return-Format": return_format, "X-Retain-Images": "none",
               "User-Agent": UA["User-Agent"]}
    if target_selector:
        headers["X-Target-Selector"] = target_selector
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    for attempt in range(retries):
        try:
            r = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r.text
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            return None  # incl. 402 (zero balance) → treat as a miss, no retry/spend
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


def fetch(url: str, timeout: int = 30, wayback: bool = True) -> str | None:
    """Browser-UA GET; on failure/block, fall back to the Wayback raw snapshot."""
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code < 400 and r.text:
            return r.text
    except requests.RequestException:
        pass
    return _wayback(url, timeout) if wayback else None


def _wayback(url: str, timeout: int) -> str | None:
    try:
        api = requests.get(
            "http://archive.org/wayback/available", params={"url": url}, timeout=timeout
        ).json()
        snap = (api.get("archived_snapshots") or {}).get("closest") or {}
        if snap.get("available") and snap.get("url"):
            # the `id_` modifier returns the ORIGINAL bytes (keeps the JSON-LD intact)
            raw = re.sub(r"/web/(\d+)/", r"/web/\1id_/", snap["url"], count=1)
            r = requests.get(raw, headers=UA, timeout=timeout)
            if r.status_code < 400:
                return r.text
    except (requests.RequestException, ValueError):
        pass
    return None


def _nodes(html: str) -> list[dict]:
    """All JSON-LD objects on the page, flattening @graph containers."""
    out: list[dict] = []
    for blk in re.findall(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S):
        try:
            obj = json.loads(blk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            out += [o for o in obj if isinstance(o, dict)]
        elif isinstance(obj, dict):
            graph = obj.get("@graph")
            out += [o for o in graph if isinstance(o, dict)] if isinstance(graph, list) else [obj]
    return out


def _source_url(item: dict) -> str | None:
    """The claim's originating URL — checked across the ClaimReview spots publishers use."""
    author = item.get("author") if isinstance(item.get("author"), dict) else {}
    for v in (item.get("sameAs"), author.get("sameAs"), item.get("url"), author.get("url")):
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]
        if isinstance(v, str) and v.startswith("http"):
            return v
    for app in item.get("appearance") or []:
        u = app.get("url") if isinstance(app, dict) else app
        if isinstance(u, str) and u.startswith("http"):
            return u
    return None


_SOCIAL = re.compile(
    r'https?://(?:www\.)?(?:twitter|x)\.com/[^/"\s]+/status/\d+'
    r'|https?://(?:www\.)?facebook\.com/[^"\s]+/(?:posts|videos)/\d+'
    r'|https?://(?:www\.)?instagram\.com/(?:p|reel)/[\w-]+'
    r'|https?://(?:www\.)?tiktok\.com/@[^/"\s]+/video/\d+'
)


def body_source_url(html: str) -> str | None:
    """First embedded social-media post URL in the page body — the source-URL fallback for
    publishers (Snopes, Full Fact, …) that embed the original post instead of putting it in
    the ClaimReview markup."""
    m = _SOCIAL.search(html)
    return m.group(0) if m else None


def extract_claimreview(html: str) -> dict | None:
    """First ClaimReview node on the page → shared record fields (None if absent)."""
    for o in _nodes(html):
        if "ClaimReview" not in str(o.get("@type", "")):
            continue
        rr = o.get("reviewRating") or {}
        item = o.get("itemReviewed") or {}
        author = item.get("author") if isinstance(item.get("author"), dict) else {}
        pub = (o.get("author") or o.get("publisher")) or {}

        def _u(s):  # decode HTML entities (Snopes JSON-LD ships &quot;/&amp;/&#39; — clog 250626)
            return _unescape(s) if isinstance(s, str) else s

        return {
            "normalized_claim": _u(o.get("claimReviewed") or item.get("name")),
            "verdict_raw": _u(rr.get("alternateName") or rr.get("ratingValue")),
            "rating_value": str(rr.get("ratingValue")) if rr.get("ratingValue") is not None else None,
            "review_url": o.get("url"),
            "review_date": (o.get("datePublished") or o.get("dateModified") or "")[:10] or None,
            "claimant": _u(author.get("name")),
            "claim_source_url": _source_url(item),
            "publisher_name": pub.get("name") if isinstance(pub, dict) else None,
        }
    return None
