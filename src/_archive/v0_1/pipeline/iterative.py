from __future__ import annotations

from typing import List, Optional

import numpy as np
from openai import OpenAI

from config import (
    FIRE_CONFIDENCE_THRESHOLD,
    FIRE_REDUNDANCY_THRESHOLD,
    MAX_RETRIEVAL_ROUNDS,
    VERIFICATION_API_KEY,
    VERIFICATION_BASE_URL,
    VERIFICATION_MODEL,
)
from pipeline.models import AtomicClaim, ClaimEvidence, VerdictResult
from pipeline.retrieval import _get_embedding_model, retrieve_evidence
from pipeline.verification import verify_claim

_client = OpenAI(base_url=VERIFICATION_BASE_URL, api_key=VERIFICATION_API_KEY)


def _label_margin(result: VerdictResult) -> int:
    """Gap between the top two Likert scores. Large gap = model is decided; small gap = torn."""
    scores = sorted(
        [result.likert.Supported, result.likert.Refuted,
         result.likert.NotEnoughEvidence, result.likert.ConflictingEvidence],
        reverse=True,
    )
    return scores[0] - scores[1]


def _is_redundant(new_query: str, past_queries: List[str]) -> bool:
    if not past_queries:
        return False
    model = _get_embedding_model()
    embs = model.encode([new_query] + past_queries, normalize_embeddings=True)
    sims = embs[1:] @ embs[0]
    return float(sims.max()) > FIRE_REDUNDANCY_THRESHOLD


def _generate_followup_query(claim_text: str, reasoning: str, verdict: str, score: int) -> str:
    prompt = (
        f"Claim: {claim_text}\n\n"
        f"Reasoning so far:\n{reasoning}\n\n"
        f"Current verdict: {verdict} (confidence {score}/5 — uncertain)\n\n"
        "Generate ONE targeted search query to resolve the remaining uncertainty. "
        "Respond with ONLY the search query."
    )
    if "qwen" in VERIFICATION_MODEL.lower():
        prompt += "\n\n/no_think"
    resp = _client.chat.completions.create(
        model=VERIFICATION_MODEL,
        messages=[
            {"role": "system", "content": "You are a research assistant helping fact-check a claim."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip().strip('"').strip("'")


def _merge(base: ClaimEvidence, new: ClaimEvidence) -> ClaimEvidence:
    seen = {e.url for e in base.evidence}
    added = [e for e in new.evidence if e.url not in seen]
    return ClaimEvidence(
        claim=base.claim,
        evidence=base.evidence + added,
        blocked_count=base.blocked_count + new.blocked_count,
        scraped_count=base.scraped_count + new.scraped_count,
    )


def iterative_verify(
    claim: AtomicClaim,
    initial_evidence: ClaimEvidence,
    date_cutoff: Optional[str] = None,
    verbose: bool = False,
) -> VerdictResult:
    evidence = initial_evidence
    past_queries: List[str] = list(claim.queries)

    for round_num in range(1, MAX_RETRIEVAL_ROUNDS + 1):
        result = verify_claim(evidence)
        margin = _label_margin(result)
        is_nei = result.verdict == "Not Enough Evidence"

        if verbose:
            print(f"       [FIRE {round_num}/{MAX_RETRIEVAL_ROUNDS}]  {result.verdict}  margin={margin}  nei={is_nei}")

        if not is_nei and margin >= FIRE_CONFIDENCE_THRESHOLD:
            if verbose:
                print(f"       [FIRE] confident — early exit")
            return result

        if round_num == MAX_RETRIEVAL_ROUNDS:
            if verbose:
                print(f"       [FIRE] max rounds reached")
            return result

        followup = _generate_followup_query(claim.text, result.reasoning, result.verdict, margin)
        if verbose:
            print(f"       [FIRE] follow-up: {followup}")

        if _is_redundant(followup, past_queries):
            if verbose:
                print(f"       [FIRE] redundant query — stopping")
            return result

        past_queries.append(followup)
        followup_claim = AtomicClaim(text=claim.text, queries=[followup], questions=claim.questions)
        new_evidence = retrieve_evidence(followup_claim, date_cutoff=date_cutoff, verbose=verbose)
        evidence = _merge(evidence, new_evidence)

    return result  # unreachable but satisfies type checker
