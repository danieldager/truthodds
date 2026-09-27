"""Tier 2: Google Fact Check Tools API + harmonisation.

Query FCT API with the claim text. If a trusted-publisher review is returned,
map its `textualRating` to one of the 4 verdict classes via the rule table
already built for `eval_v1.parquet`.
"""
from __future__ import annotations

from pipeline.models import AtomicClaim, ClaimVerdict


def lookup(claim: AtomicClaim) -> ClaimVerdict | None:
    """Query FCT API. Returns a ClaimVerdict if a trusted publisher rates it, else None.

    TODO:
      1. GET FCTAPI_ENDPOINT with params {query: claim.text, key: FCTAPI_KEY}.
      2. Filter results to publishers whose site is in TRUSTED_PUBLISHERS.
      3. Pick the most recent review (max `reviewDate`).
      4. Harmonise its `textualRating` via `eval.harmonize.rule_map(publisher_site, rating)`;
         fall back to `eval.harmonize.llm_map(...)` if rule_map returns None.
      5. Build ClaimVerdict(verdict=..., tier_resolved=2, source_publisher=...,
                            likert=Likert(...)).  Likert can be set from the verdict
         (e.g. {Refuted: 5, others: 1}); we trust FCT publishers strongly.
    """
    raise NotImplementedError
