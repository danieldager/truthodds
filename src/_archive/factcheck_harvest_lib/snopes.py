"""Snopes harvester — Google FCT API discovery + on-page ClaimReview parse.

Snopes ingests into the GFC API near-real-time (measured ~7h, ~complete — clog/230626), so we
discover review URLs from the API, then scrape each page for the cleaner inline ClaimReview
(claim + enum verdict) plus the bonus fields the API strips: the **source URL** (Snopes embeds the
original tweet in the body, not the ClaimReview markup → `body_source_url` fallback) and context
(`og:description` / the "KEY POINTS" box). Verdict is a clean enum → `harmonize.rule_map`.
"""
from __future__ import annotations

import re

import lxml.html

from eval.claimreview import body_source_url, extract_claimreview, fetch
from eval.harvest import paginate_claims

PUBLISHER_SITE = "snopes.com"

# Self / share-widget / ad noise to drop from the cited-sources list.
_SRC_SKIP = re.compile(
    r'snopes\.com|/cdn-cgi/|doubleclick|googlesyndication|amazon-adsystem|outbrain|taboola'
    r'|facebook\.com/(?:sharer|dialog)|twitter\.com/(?:share|intent)|mailto:|/privacy|/terms-', re.I)


def _cited_sources(html: str, checked_post: str | None) -> list[str]:
    """The sources the fact-check CITES (its evidence) — Snopes hyperlinks them inline in the
    article body (the dedicated 'Sources' list is JS-rendered, but the inline references are
    static). Excludes the post being checked (`checked_post`, kept separately in claim_source_url),
    Snopes' own links, share widgets and ad/cdn noise — so this column is the EVIDENCE side only."""
    try:
        tree = lxml.html.fromstring(html)
    except (lxml.etree.ParserError, ValueError):
        return []
    nodes = tree.xpath(
        '//*[contains(@class,"article-content") or contains(@id,"article-content")'
        ' or contains(@class,"single-body") or self::article]//a/@href')
    checked = (checked_post or "").split("?")[0]
    out: list[str] = []
    for h in nodes:
        if not h.startswith("http") or _SRC_SKIP.search(h):
            continue
        h0 = h.split("?")[0]
        if h0 != checked and h0 not in out:
            out.append(h0)
    return out


def discover(max_age_days: int, max_pages: int = 8) -> list[str]:
    """Recent Snopes review URLs from the Fact Check Tools API."""
    seen: set[str] = set()
    for c in paginate_claims(
        review_publisher_site_filter="snopes.com",
        max_age_days=max_age_days, page_size=100, max_pages=max_pages,
    ):
        for r in c.get("claimReview", []):
            u = r.get("url")
            if u and u not in seen and "snopes.com" in u:
                seen.add(u)
    return list(seen)


def parse_article(url: str) -> dict | None:
    """Fetch + parse one Snopes page into the shared record (None if no ClaimReview)."""
    html = fetch(url)
    if not html:
        return None
    rec = extract_claimreview(html)
    if not rec or not rec.get("normalized_claim"):
        return None
    rec["review_url"] = re.sub(r"(?<!:)//", "/", rec.get("review_url") or url)  # snopes // → /
    rec["publisher_site"] = PUBLISHER_SITE
    rec["language_code"] = "en"
    if not rec.get("claim_source_url"):
        rec["claim_source_url"] = body_source_url(html)  # the post being checked (embedded tweet)
    og = re.search(r'<meta property="og:description" content="([^"]+)"', html)
    rec["context"] = (og.group(1).strip() if og else None) or None
    rec["sources"] = _cited_sources(html, rec.get("claim_source_url"))  # evidence cited (not the post)
    return rec
