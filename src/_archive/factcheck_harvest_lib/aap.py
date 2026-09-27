"""AAP FactCheck harvester — GFC API discovery + article-page scrape for the source side.

claim + verdict + claimant + claimDate come from the GFC API claim dicts. AAP's site is NOT
WAF-blocked, so we fetch each article page to add the two source-side fields the API strips:
  - claim_source_url : the CHECKED post (ClaimReview itemReviewed.appearance[].url — recovered by
                       eval.claimreview.extract_claimreview), i.e. the thing being debunked.
  - sources          : the EVIDENCE AAP cites, hyperlinked inline in the article body (its dedicated
                       <ol class="sources-content"> is usually empty); mostly gov/news primary
                       sources, kept separate from the checked post.
English-only (aap.com.au; non-EN probes returned 0). Verdicts are free-text sentences → llm_map.
(recon: clog 240626.)
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import lxml.html

from eval.claimreview import extract_claimreview, fetch
from eval.harvest import paginate_claims

# AAP FactCheck: publisher_site → language_code. One English-only vertical.
VERTICALS = {"aap.com.au": "en"}

# Sources-list noise: AAP's own / IFCN hosts, and decorative image-wrapped anchors.
_BOILER_HOST = re.compile(r'(?:^|\.)(?:aap\.com\.au|aapnews\.com\.au|poynter\.org)$', re.I)
_IMG_FILE = re.compile(r'\.(?:jpe?g|png|gif|webp|svg)(?:$|\?)', re.I)


def _cited_sources(tree, checked: str | None) -> list[str]:
    """Evidence AAP cites — inline <a> in the article body (+ the usually-empty sources-content ol),
    minus the IFCN 'follow us' trailer paragraph (AAP socials + a per-page YouTube playlist), AAP/
    IFCN self-hosts, decorative image anchors, and the checked post itself."""
    anchors = list(tree.xpath('//ol[contains(@class,"sources-content")]//a'))
    anchors += tree.xpath('//*[contains(@class,"article-content")]'
                          '//p[not(.//a[contains(@href,"poynter.org/ifcn")])]//a')
    out: list[str] = []
    for a in anchors:
        h = (a.get("href") or "").strip()
        if not h.startswith("http") or _IMG_FILE.search(h):
            continue
        if _BOILER_HOST.search(urlparse(h).netloc.lower()):
            continue
        if not (a.text_content() or "").strip():  # image/logo-wrapped anchor
            continue
        if h != checked and h not in out:
            out.append(h)
    return out


def harvest_records(site: str, lang: str, max_age_days: int, max_pages: int = 8) -> list[dict]:
    """GFC API discovery for the core fields, then a threaded article-page scrape for the
    checked-post URL + cited evidence sources (the API strips both)."""
    api: list[dict] = []
    seen: set[str] = set()
    for c in paginate_claims(review_publisher_site_filter=site, max_age_days=max_age_days,
                             language_code=lang, page_size=100, max_pages=max_pages):
        claim = (c.get("text") or "").strip()
        if not claim:
            continue
        claimant = (c.get("claimant") or "").strip() or None
        cdate = (c.get("claimDate") or "")[:10] or None
        for r in c.get("claimReview", []):
            url = r.get("url")
            if not url or site not in url or url in seen:
                continue
            seen.add(url)
            api.append({
                "normalized_claim": claim, "verdict_raw": r.get("textualRating"),
                "rating_value": None, "claimant": claimant, "publisher_site": site,
                "publisher_name": (r.get("publisher") or {}).get("name") or "AAP",
                "review_url": url, "review_date": (r.get("reviewDate") or "")[:10] or None,
                "claim_date": cdate, "language_code": r.get("languageCode") or lang,
                "context": None,
            })

    def _with_sources(rec: dict) -> dict:
        html = fetch(rec["review_url"])
        cp, srcs = None, None
        if html:
            cr = extract_claimreview(html)
            cp = (cr or {}).get("claim_source_url")  # checked post = appearance[].url
            try:
                srcs = _cited_sources(lxml.html.fromstring(html), cp)
            except (lxml.etree.ParserError, ValueError):
                srcs = None
        return {**rec, "claim_source_url": cp, "sources": srcs}

    with ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(_with_sources, api))
