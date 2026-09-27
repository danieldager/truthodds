"""Full Fact harvester — GFC API discovery + on-page ClaimReview + body cited sources.

Full Fact ingests into the GFC API, so we discover review URLs from the API, then fetch each page
(not WAF-blocked) for the inline ClaimReview (claim + free-text verdict → llm_map), the embedded
checked-post URL (body_source_url — the Instagram/X post being checked), and the evidence Full Fact
cites inline in the article body. Mirrors eval.snopes; the difference is the free-text verdict.

Notes: Full Fact's "What was claimed" block == the ClaimReview claim (not a distinct raw), so the
raw side comes from resolving the checked post (X posts resolve via syndication; IG/FB mostly don't).
ClaimReview carries no claim date (itemReviewed is empty); claim_date is the resolved tweet's date
where available. Verdict is a free-text sentence ("False. The DHS said …") → harmonize.llm_map.
"""
from __future__ import annotations

import re

import lxml.html

from eval.claimreview import body_source_url, extract_claimreview, fetch
from eval.harvest import paginate_claims

PUBLISHER_SITE = "fullfact.org"

# Self / share-widget / ad noise to drop from the cited-sources list.
_SRC_SKIP = re.compile(
    r'fullfact\.org|/cdn-cgi/|doubleclick|googlesyndication|amazon-adsystem|outbrain|taboola'
    r'|facebook\.com/(?:sharer|dialog)|twitter\.com/(?:share|intent)|linkedin\.com/share'
    r'|mailto:|/privacy|/cookie|/donate', re.I)


def discover(max_age_days: int, max_pages: int = 8) -> list[str]:
    """Recent Full Fact review URLs from the Fact Check Tools API."""
    seen: set[str] = set()
    for c in paginate_claims(
        review_publisher_site_filter=PUBLISHER_SITE,
        max_age_days=max_age_days, page_size=100, max_pages=max_pages,
    ):
        for r in c.get("claimReview", []):
            u = r.get("url")
            if u and PUBLISHER_SITE in u and u not in seen:
                seen.add(u)
    return list(seen)


def _cited_sources(html: str, checked: str | None) -> list[str]:
    """Evidence Full Fact cites — external links in the article body, minus the checked post,
    Full Fact's own links and share/ad noise. The EVIDENCE side; the checked post stays in
    claim_source_url."""
    try:
        tree = lxml.html.fromstring(html)
    except (lxml.etree.ParserError, ValueError):
        return []
    nodes = tree.xpath('//main//a/@href') or tree.xpath('//article//a/@href')

    def _norm(u: str) -> str:  # strip query + trailing slash so the checked post matches reliably
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


def parse_article(url: str) -> dict | None:
    """Fetch + parse one Full Fact page into the shared record (None if no ClaimReview)."""
    html = fetch(url)
    if not html:
        return None
    rec = extract_claimreview(html)
    if not rec or not rec.get("normalized_claim"):
        return None
    rec["publisher_site"] = PUBLISHER_SITE
    rec["language_code"] = "en"
    if not rec.get("claim_source_url"):
        rec["claim_source_url"] = body_source_url(html)  # the post being checked (embedded)
    rec["sources"] = _cited_sources(html, rec.get("claim_source_url"))  # evidence cited (not the post)
    rec["context"] = None
    return rec
