"""Data models for the v0.2 pipeline. Kept deliberately flat."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class AtomicClaim(BaseModel):
    text: str
    embedding: list[float]


class EvidenceDoc(BaseModel):
    url: str
    provider: str = ""                      # which search provider supplied this result
    snippet: str = ""                       # search-result snippet — ALWAYS retained
    scraped: bool = False                   # True if the full page was fetched + summarised
    relevant: bool = True                   # summariser relevance; snippet-only docs kept True
    publication_date: str | None = None     # ISO date, extracted by the summariser
    summary: str | None = None              # set only when scraped + relevant
    quotes: list[str] = []                  # direct quotes pulled from the source


class VerdictScores(BaseModel):
    """Per-dimension Likert 1-5. Each dimension is independent."""
    veracity: int               # 1-5; truth of the claim given the evidence
    evidence_sufficiency: int   # 1-5; how directly/fully the evidence addresses the claim
    evidence_agreement: int     # 1-5; whether the evidence pieces point the same way
    source_reliability: int     # 1-5; how trustworthy the cited sources are


class ClaimVerdict(BaseModel):
    claim: AtomicClaim
    scores: VerdictScores
    tier_resolved: Literal[1, 2, 3]
    cap_hit: bool = False                   # True if Tier 3 used all MAX_ROUNDS search slots
    redundant_exit: bool = False            # True if Tier 3 stopped early via _is_redundant
    stopped_reason: str = ""                # "confident" | "redundant" | "cap"
    resolving_provider: str = ""            # cascade provider active when it became confident
    providers_used: list[str] = []          # cascade providers actually queried, in order
    evidence_urls: list[str] = []           # Tier 3 only — URLs of docs in final pool
    source_publisher: str | None = None     # Tier 2 only
    justification: str = ""

    # --- Tier 3 trace + diagnostics (populated only when tier_resolved == 3) ---
    rounds_used: int = 0                    # search rounds executed (1-MAX_ROUNDS)
    past_queries: list[str] = []            # every query issued, in order
    analysis: str = ""                      # final Synthesis output (3-6 sentences)
    n_urls_seen: int = 0                    # total new (deduped) URLs across rounds
    n_scraped: int = 0                      # URLs whose full page was scraped + summarised
    n_blocked_or_failed: int = 0            # scrape failed AND snippet empty (truly no content)
    n_snippet_used: int = 0                 # scrape failed but snippet retained as evidence
    n_irrelevant: int = 0                   # scraped docs marked irrelevant by summariser
    n_search_errors: int = 0                # search API calls that errored (quota/HTTP) — NOT 0-results
    elapsed_seconds: float = 0.0            # wall-clock time
    llm_calls: int = 0                      # count of LLM calls made

    # --- ClaimCheck-style 4-class verdict (parallel evaluator, see verify._evaluate_4class) ---
    verdict_4class: str = ""                # one of {Supported, Refuted, Conflicting Evidence/Cherrypicking, Not Enough Evidence}
    verdict_4class_justification: str = ""

    # --- Post-level flag (design §3b; populated only when verify(post_context=...) is given) ---
    post_flag: bool | None = None           # True = flag the POST as potential misinformation
    post_flag_reason: str = ""


class PipelineResult(BaseModel):
    post: str
    risk_score: float | None = None         # Tier 0 output (advisory)
    verdicts: list[ClaimVerdict] = []       # one per atomic claim
