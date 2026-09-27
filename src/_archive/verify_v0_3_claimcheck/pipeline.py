"""Orchestrator: glue extract + Tier 1 + Tier 2 + Tier 3 in order.

Tier 0 (encoder ensemble) lives outside this package and is passed in as
`risk_score` if available.
"""
from __future__ import annotations
from datetime import datetime

from pipeline import cache, extract, fact_api, verify
from pipeline.models import AtomicClaim, ClaimVerdict, PipelineResult


def run(
    post: str,
    *,
    risk_score: float | None = None,
    source: str | None = None,
    skip_cache: bool = False,
    skip_fact_api: bool = False,
    date_ceiling: str | None = None,
) -> PipelineResult:
    """End-to-end pipeline on a single post.

    Args:
        post: raw post text.
        risk_score: Tier 0 output, if computed upstream (advisory only).
        source: anonymized post/platform id stored with each cached claim (observatory).
        skip_cache: **fully** bypass Tier 1 — no cache read AND no cache write (so no DB
            is needed). Used by eval runs to force fresh verification with zero cache
            side effects. NOTE: the standalone Tier-3 eval (`verify_run.py`) calls
            `verify.verify()` directly and never touches this orchestrator at all.
        skip_fact_api: bypass Tier 2. Used by eval runs to measure pure Tier 3.
        date_ceiling: mm/dd/yyyy upper bound for evidence publication dates. Default: today.

    Returns one ClaimVerdict per atomic claim extracted from the post.
    """
    claims = extract.extract(post)
    if not claims:
        return PipelineResult(post=post, risk_score=risk_score, verdicts=[])

    if date_ceiling is None:
        date_ceiling = datetime.today().strftime("%m/%d/%Y")

    verdicts = [
        _resolve_claim(claim, date_ceiling, source, skip_cache, skip_fact_api)
        for claim in claims
    ]
    return PipelineResult(post=post, risk_score=risk_score, verdicts=verdicts)


def _resolve_claim(
    claim: AtomicClaim,
    date_ceiling: str,
    source: str | None,
    skip_cache: bool,
    skip_fact_api: bool,
) -> ClaimVerdict:
    # Tier 1 — cache read (folds an equivalent claim into its cluster on a hit)
    if not skip_cache:
        hit = cache.lookup(claim, source=source)
        if hit is not None:
            return hit

    # Tier 2
    if not skip_fact_api:
        hit = fact_api.lookup(claim)
        if hit is not None:
            if not skip_cache:
                cache.write(hit, source=source)
            return hit

    # Tier 3
    verdict = verify.verify(claim, date_ceiling=date_ceiling)
    if not skip_cache:
        cache.write(verdict, source=source)
    return verdict
