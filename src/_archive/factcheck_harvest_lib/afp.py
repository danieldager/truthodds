"""AFP harvester — Google FCT API only (the AFP website is Akamai/DataDome WAF-blocked).

Unlike the scrape-based harvesters, AFP's claim + verdict + claimant all come straight from the
GFC API claim dicts, so there's no article fetch — AFP is the one publisher where the API is the
*only* viable channel (clog/230626). We pull both verticals: factcheck.afp.com (EN) and
factuel.afp.com (FR). The API strips source URLs, so the raw side is reconstructed later by
source_fetch.assemble_raw_context (attributed tier — AFP names its claimants). Verdict is a clean
enum → harmonize.rule_map (a per-vertical table; FR uses the French strings).
"""
from __future__ import annotations

import lxml.html

from eval.harvest import paginate_claims

# AFP verticals: publisher_site → language_code.
VERTICALS = {"factcheck.afp.com": "en", "factuel.afp.com": "fr"}

# Jina X-Target-Selector: return ONLY the rel-tagged source anchors → ~1-2k tokens/article (vs ~50k
# for full-page html), keeping the evidence/appearance split. The cost fix (clog 240626).
AFP_SELECTOR = "a[rel~=evidence], a[rel~=appearance]"


def afp_sources(html: str) -> tuple[str | None, list[str]]:
    """Parse jina-fetched AFP article HTML → (checked_post, cited_sources). AFP tags every body link
    by `rel`: rel~="appearance" = the post being checked, rel~="evidence" = the sources AFP cited —
    a clean built-in differentiation. Returns the first appearance URL + the deduped evidence list."""
    try:
        tree = lxml.html.fromstring(html)
    except (lxml.etree.ParserError, ValueError):
        return None, []

    def rel(name: str) -> list[str]:
        return tree.xpath(
            f'//a[contains(concat(" ",normalize-space(@rel)," ")," {name} ")]/@href')

    appearance = [h for h in rel("appearance") if h.startswith("http")]
    evidence, seen = [], set()
    for h in rel("evidence"):
        if h.startswith("http") and h not in seen:
            seen.add(h)
            evidence.append(h)
    return (appearance[0] if appearance else None), evidence


def harvest_records(site: str, lang: str, max_age_days: int, max_pages: int = 8) -> list[dict]:
    """Build shared-schema records directly from the GFC API claim dicts for one AFP vertical.

    Same internal record shape as the generic extractor (eval.claimreview.extract_claimreview) so
    the harvest script's `_row` projection is reused. Deduped by review URL within the pull.
    """
    recs: list[dict] = []
    seen: set[str] = set()
    for c in paginate_claims(
        review_publisher_site_filter=site,
        max_age_days=max_age_days, language_code=lang, page_size=100, max_pages=max_pages,
    ):
        claim = (c.get("text") or "").strip()
        if not claim:  # ~9% of the FR feed carries no claim text → unusable
            continue
        claimant = (c.get("claimant") or "").strip() or None
        for r in c.get("claimReview", []):
            url = r.get("url")
            if not url or site not in url or url in seen:
                continue
            seen.add(url)
            recs.append({
                "normalized_claim": claim,
                "verdict_raw": r.get("textualRating"),
                "rating_value": None,  # AFP carries a textual enum, no numeric ratingValue
                "claimant": claimant,
                "publisher_site": site,
                "publisher_name": (r.get("publisher") or {}).get("name") or "AFP",
                "review_url": url,
                "review_date": (r.get("reviewDate") or "")[:10] or None,
                "claim_date": (c.get("claimDate") or "")[:10] or None,  # original-claim date (API)
                "language_code": r.get("languageCode") or lang,
                "claim_source_url": None,  # API strips source URLs → no raw pair (verdict+date only)
                "context": None,
            })
    return recs
