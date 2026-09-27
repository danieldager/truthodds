"""Lead Stories harvester — sitemap discovery + generic ClaimReview JSON-LD parse.

Lead Stories (IFCN signatory, Meta/TikTok partner) emits clean per-article ClaimReview, so it
reuses `eval.claimreview`. The Atom feed only reaches ~5 days, so we enumerate the 3-month window
from `sitemap.txt` (dates are in the URL path, `/YYYY/MM/`) and post-filter on the exact
`datePublished` from each article's JSON-LD.
"""
from __future__ import annotations

import datetime
import re

import lxml.html

from eval.claimreview import UA, extract_claimreview, fetch  # noqa: F401  (UA re-exported)

PUBLISHER_SITE = "leadstories.com"
SITEMAP = "https://leadstories.com/sitemap.txt"
_URL_DATE = re.compile(r"/(\d{4})/(\d{2})/")

_SRC_SKIP = re.compile(
    r'leadstories\.com|/cdn-cgi/|doubleclick|googlesyndication|amazon-adsystem|outbrain|taboola'
    r'|trendolizer|facebook\.com/(?:sharer|dialog)|twitter\.com/(?:share|intent)|linkedin\.com/share'
    r'|mailto:|/privacy|/cookie', re.I)


def _cited_sources(html: str, checked: str | None) -> list[str]:
    """Evidence Lead Stories cites inline in the article body, minus the checked post (kept in
    claim_source_url), its own links and share/ad noise — the EVIDENCE side."""
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


def sitemap_urls(since: datetime.date) -> list[str]:
    """Hoax-alert article URLs whose path month is >= `since`'s month (coarse; the runner
    post-filters on the exact review date). Newest-month first."""
    txt = fetch(SITEMAP, wayback=False) or ""
    cutoff = (since.year, since.month)
    rows = []
    for line in txt.splitlines():
        u = line.strip()
        m = _URL_DATE.search(u)
        if not m or "hoax-alert" not in u:
            continue
        ym = (int(m.group(1)), int(m.group(2)))
        if ym >= cutoff:
            rows.append((ym, u))
    rows.sort(reverse=True)  # newest month first
    return [u for _, u in rows]


def parse_article(url: str) -> dict | None:
    """Fetch + parse one Lead Stories article into the shared record (None if no ClaimReview)."""
    html = fetch(url)
    if not html:
        return None
    rec = extract_claimreview(html)
    if not rec or not rec.get("normalized_claim"):
        return None
    rec["review_url"] = rec.get("review_url") or url
    rec["publisher_site"] = PUBLISHER_SITE
    rec["language_code"] = "en"
    m = re.search(r'<meta[^>]+property="og:description"[^>]+content="([^"]*)"', html)
    rec["context"] = (m.group(1).strip() if m else None) or None
    rec["sources"] = _cited_sources(html, rec.get("claim_source_url"))  # evidence cited (not the post)
    return rec
