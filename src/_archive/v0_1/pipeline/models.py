from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel


class AtomicClaim(BaseModel):
    text: str
    queries: List[str]
    questions: List[str] = []


class ClaimExtractionResult(BaseModel):
    is_checkable: bool
    claims: List[AtomicClaim]


class EvidenceItem(BaseModel):
    url: str
    text: str
    credibility: float  # CRED-1 score, or 0.5 if domain unknown


class ClaimEvidence(BaseModel):
    claim: AtomicClaim
    evidence: List[EvidenceItem]
    blocked_count: int = 0
    scraped_count: int = 0


class LikertScores(BaseModel):
    Supported: int           # 1–5
    Refuted: int             # 1–5
    NotEnoughEvidence: int   # 1–5
    ConflictingEvidence: int # 1–5


class VerdictResult(BaseModel):
    claim: str
    reasoning: str
    verdict: str  # Supported | Refuted | Not Enough Evidence | Conflicting Evidence
    likert: LikertScores
    justification: str
    key_sources: List[str]

    @property
    def confidence(self) -> float:
        scores = {
            "Supported": self.likert.Supported,
            "Refuted": self.likert.Refuted,
            "Not Enough Evidence": self.likert.NotEnoughEvidence,
            "Conflicting Evidence": self.likert.ConflictingEvidence,
        }
        return scores.get(self.verdict, 1) / 5.0


class MisleadingnessResult(BaseModel):
    original_claim: str
    is_misleading: bool
    misleading_type: Optional[str] = None   # "scope" | "implication" | "omission"
    derived_claim: Optional[str] = None     # Ã, B, or A⁺ that was refuted
    derived_verdict: Optional[VerdictResult] = None


class PipelineResult(BaseModel):
    post: str
    is_checkable: bool
    verdicts: List[VerdictResult]
    post_verdict: Optional[str] = None
    post_confidence: Optional[float] = None
