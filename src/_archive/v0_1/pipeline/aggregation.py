from __future__ import annotations

from typing import List, Tuple

from config import VERDICT_CONFIDENCE_THRESHOLD
from pipeline.models import VerdictResult


def _confidence(v: VerdictResult) -> float:
    scores = {
        "Supported": v.likert.Supported,
        "Refuted": v.likert.Refuted,
        "Not Enough Evidence": v.likert.NotEnoughEvidence,
        "Conflicting Evidence": v.likert.ConflictingEvidence,
    }
    return scores.get(v.verdict, 1) / 5.0


def aggregate_verdicts(verdicts: List[VerdictResult]) -> Tuple[str, float]:
    """
    Collapse per-claim verdicts into a single post-level verdict.

    Priority: Refuted > Conflicting Evidence > NEI > Supported
    A higher-priority verdict wins if confidence >= VERDICT_CONFIDENCE_THRESHOLD.
    Supported requires ALL claims to be Supported.
    """
    if not verdicts:
        return "Not Enough Evidence", 0.0
    if len(verdicts) == 1:
        return verdicts[0].verdict, _confidence(verdicts[0])

    threshold = VERDICT_CONFIDENCE_THRESHOLD

    for verdict_type in ("Refuted", "Conflicting Evidence"):
        matching = [v for v in verdicts if v.verdict == verdict_type and _confidence(v) >= threshold]
        if matching:
            return verdict_type, max(_confidence(v) for v in matching)

    if all(v.verdict == "Supported" for v in verdicts):
        return "Supported", sum(_confidence(v) for v in verdicts) / len(verdicts)

    return "Not Enough Evidence", sum(_confidence(v) for v in verdicts) / len(verdicts)
