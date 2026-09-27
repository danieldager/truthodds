"""FactCheck.org harvester — GFC API discovery + article-page scrape for cited sources.

claim + verdict + claimant + claimDate come from the GFC API (FactCheck.org has no on-page
ClaimReview). A page fetch (not WAF-blocked) adds the evidence it cites inline in the article body —
gov / academic / primary sources, very thorough. Long-form political fact-checks, so the claim is
usually a statement with no resolvable source post → minimal raw pairs; the value is the gold verdict
+ the cited-source set. Verdict is a discrete-ish rating → rule_map (factcheck.org table) + llm_map
fallback for the free-text minority. (audit clog 250626.)
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import lxml.html

from eval.claimreview import body_source_url, fetch
from eval.harvest import paginate_claims

PUBLISHER_SITE = "factcheck.org"
LANG = "en"

_SRC_SKIP = re.compile(
    r'factcheck\.org|annenberg|/cdn-cgi/|doubleclick|googlesyndication|amazon-adsystem|outbrain'
    r'|facebook\.com/(?:sharer|dialog)|twitter\.com/(?:share|intent)|linkedin\.com/share'
    r'|mailto:|/privacy|/donate', re.I)


def _norm(u: str) -> str:
    return (u or "").split("?")[0].split("#")[0].rstrip("/")


def _cited_sources(html: str, checked: str | None) -> list[str]:
    """Evidence FactCheck.org cites — external links in the article body (.entry-content), minus the
    checked post, its own/Annenberg links and share/ad noise. The EVIDENCE side."""
    try:
        tree = lxml.html.fromstring(html)
    except (lxml.etree.ParserError, ValueError):
        return []
    nodes = (tree.xpath('//div[contains(@class,"entry-content")]//a/@href')
             or tree.xpath('//article//a/@href'))
    ch = _norm(checked or "")
    out: list[str] = []
    for h in nodes:
        if not h.startswith("http") or _SRC_SKIP.search(h):
            continue
        h0 = _norm(h)
        if h0 and h0 != ch and h0 not in out:
            out.append(h0)
    return out


def harvest_records(max_age_days: int, max_pages: int = 8) -> list[dict]:
    """GFC API discovery for the core fields, then a threaded page scrape for the cited sources."""
    api: list[dict] = []
    seen: set[str] = set()
    for c in paginate_claims(review_publisher_site_filter=PUBLISHER_SITE, max_age_days=max_age_days,
                             language_code=LANG, page_size=100, max_pages=max_pages):
        claim = (c.get("text") or "").strip()
        if not claim:
            continue
        claimant = (c.get("claimant") or "").strip() or None
        cdate = (c.get("claimDate") or "")[:10] or None
        for r in c.get("claimReview", []):
            url = r.get("url")
            if not url or PUBLISHER_SITE not in url or url in seen:
                continue
            seen.add(url)
            api.append({
                "normalized_claim": claim, "verdict_raw": r.get("textualRating"), "rating_value": None,
                "claimant": claimant, "publisher_site": PUBLISHER_SITE,
                "publisher_name": (r.get("publisher") or {}).get("name") or "FactCheck.org",
                "review_url": url, "review_date": (r.get("reviewDate") or "")[:10] or None,
                "claim_date": cdate, "language_code": r.get("languageCode") or LANG, "context": None,
            })

    def _with_sources(rec: dict) -> dict:
        html = fetch(rec["review_url"])
        cp = body_source_url(html) if html else None  # rare for FactCheck.org (statements)
        srcs = _cited_sources(html, cp) if html else None
        return {**rec, "claim_source_url": cp, "sources": srcs}

    with ThreadPoolExecutor(max_workers=5) as ex:
        return list(ex.map(_with_sources, api))
