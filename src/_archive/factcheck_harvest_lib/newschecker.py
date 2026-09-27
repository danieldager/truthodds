"""Newschecker harvester — Google FCT API only (record built straight from the API claim dicts).

Mirrors eval.afp: Newschecker's claim + verdict + claimant all come from the GFC API claim dicts,
so there's no article fetch. The verdict is a clean enum → harmonize.rule_map (the existing
RULES["newschecker.in"] table covers the live 90-day EN vocab with 0 misses; recon 240626). The
API strips source URLs and the claimant is the generic "Social Media Post", so the raw side is not
distinct enough to pair from the API alone (a later scrape of the on-page ClaimReview
appearance[].url — escaped in the Next.js __next_f payload — could recover X/Twitter source URLs).

Newschecker also publishes high volume in Hindi (hi ~101/92d), Tamil (ta ~69), and Bengali
(bn ~34) — a multilingual follow-up; this draft harvests the EN vertical only.
"""
from __future__ import annotations

from eval.harvest import paginate_claims

# Single EN vertical for now (hi/ta/bn are a multilingual follow-up): publisher_site → language_code.
VERTICALS = {"newschecker.in": "en"}


def harvest_records(site: str, lang: str, max_age_days: int, max_pages: int = 8) -> list[dict]:
    """Build shared-schema records directly from the GFC API claim dicts for one Newschecker vertical.

    Same internal record shape as eval.afp.harvest_records so the harvest script's `_row` projection
    is reused. Deduped by review URL within the pull.
    """
    recs: list[dict] = []
    seen: set[str] = set()
    for c in paginate_claims(
        review_publisher_site_filter=site,
        max_age_days=max_age_days, language_code=lang, page_size=100, max_pages=max_pages,
    ):
        claim = (c.get("text") or "").strip()
        if not claim:
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
                "rating_value": None,  # Newschecker carries a textual enum, no numeric ratingValue
                "claimant": claimant,
                "publisher_site": site,
                "publisher_name": (r.get("publisher") or {}).get("name") or "Newschecker",
                "review_url": url,
                "review_date": (r.get("reviewDate") or "")[:10] or None,
                "language_code": r.get("languageCode") or lang,
                "claim_source_url": None,  # API strips it; on-page appearance[].url is a later scrape
                "context": None,
            })
    return recs
