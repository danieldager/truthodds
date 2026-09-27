"""20 Minutes "Fake Off" (FR) harvester — GFC API discovery + on-page ClaimReview.

Major French free daily; "Fake Off" debunks viral claims in French (our bilingual coverage). Emits
ClaimReview with a NUMERIC verdict (`ratingValue` 1–5, "Faux"=1 → `harmonize.numeric_map`) and, like
Snopes, embeds the original post in the body (→ `body_source_url`) rather than the markup.
"""
from __future__ import annotations

import re

import lxml.html

from eval.claimreview import body_source_url, extract_claimreview, fetch
from eval.harvest import paginate_claims

PUBLISHER_SITE = "20minutes.fr"

_SRC_SKIP = re.compile(
    r'20minutes\.fr|/cdn-cgi/|doubleclick|googlesyndication|amazon-adsystem|outbrain|taboola'
    r'|facebook\.com/(?:sharer|dialog)|twitter\.com/(?:share|intent)|linkedin\.com/share'
    r'|mailto:|/privacy|/cgu', re.I)


def _cited_sources(html: str, checked: str | None) -> list[str]:
    """Evidence 20 Minutes cites inline in the article body, minus the checked post, its own links
    and share/ad noise — the EVIDENCE side."""
    try:
        tree = lxml.html.fromstring(html)
    except (lxml.etree.ParserError, ValueError):
        return []
    nodes = tree.xpath('//article//a/@href') or tree.xpath('//main//a/@href')

    def _norm(u: str) -> str:
        return (u or "").split("?")[0].split("#")[0].rstrip("/")

    ch = _norm(checked or "")
    out: list[str] = []
    for h in nodes:
        if not h.startswith("http") or _SRC_SKIP.search(h):
            continue
        h0 = _norm(h)
        if h0 and h0 != ch and h0 not in out:
            out.append(h0)
    return out


def discover(max_age_days: int, max_pages: int = 4) -> list[str]:
    seen: set[str] = set()
    for c in paginate_claims(
        review_publisher_site_filter="20minutes.fr",
        max_age_days=max_age_days, language_code="fr", page_size=100, max_pages=max_pages,
    ):
        for r in c.get("claimReview", []):
            u = r.get("url")
            if u and u not in seen and "20minutes.fr" in u:
                seen.add(u)
    return list(seen)


def parse_article(url: str) -> dict | None:
    html = fetch(url)
    if not html:
        return None
    rec = extract_claimreview(html)
    if not rec or not rec.get("normalized_claim"):
        return None
    rec["review_url"] = rec.get("review_url") or url
    rec["publisher_site"] = PUBLISHER_SITE
    rec["language_code"] = "fr"
    if not rec.get("claim_source_url"):
        rec["claim_source_url"] = body_source_url(html)
    og = re.search(r'<meta property="og:description" content="([^"]+)"', html)
    rec["context"] = (og.group(1).strip() if og else None) or None
    rec["sources"] = _cited_sources(html, rec.get("claim_source_url"))  # evidence cited (not the post)
    return rec
