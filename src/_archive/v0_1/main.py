from __future__ import annotations

import json
import sys
from typing import Optional

from pipeline.aggregation import aggregate_verdicts
from pipeline.claim_extraction import extract_claims
from pipeline.misleadingness import check_misleadingness
from pipeline.retrieval import retrieve_evidence
from pipeline.verification import verify_claim
from pipeline.models import PipelineResult, VerdictResult


def run(post: str, date_cutoff: Optional[str] = None) -> PipelineResult:
    """
    Run the full fact-checking pipeline on a social media post.

    Args:
        post: The text of the post to fact-check.
        date_cutoff: MM/DD/YYYY — only surface evidence published before this date.
                     Defaults to today if not provided.

    Returns:
        PipelineResult with verdicts for each atomic claim found.
    """
    print(f"\n{'='*60}")
    print(f"Post: {post[:100]}")
    print(f"{'='*60}\n")

    # Stage 1 — claim detection, decomposition, query generation
    print("[1/3] Extracting claims...")
    extraction = extract_claims(post)

    if not extraction.is_checkable:
        print("  → No checkable claims found. Exiting.\n")
        return PipelineResult(post=post, is_checkable=False, verdicts=[])

    print(f"  → {len(extraction.claims)} atomic claim(s) found:")
    for c in extraction.claims:
        print(f"     • {c.text}")

    verdicts: list[VerdictResult] = []

    for i, claim in enumerate(extraction.claims, 1):
        print(f"\n[2/3] Retrieving evidence for claim {i}/{len(extraction.claims)}...")
        claim_evidence = retrieve_evidence(claim, date_cutoff=date_cutoff)

        print(f"[3/3] Verifying claim {i}/{len(extraction.claims)}...")
        verdict = verify_claim(claim_evidence)
        verdicts.append(verdict)

        print(f"  → Verdict: {verdict.verdict} (confidence: {verdict.confidence:.2f})")
        print(f"     {verdict.justification}")

        if verdict.verdict == "Supported":
            print(f"\n[misleadingness] Checking claim {i} for misleadingness...")
            m = check_misleadingness(claim, verdict, date_cutoff=date_cutoff)
            if m.is_misleading:
                print(f"  → MISLEADING ({m.misleading_type}): {m.derived_claim}")
                print(f"     Derived verdict: {m.derived_verdict.verdict}")
            else:
                print(f"  → Not misleading")

    post_verdict, post_confidence = aggregate_verdicts(verdicts)
    result = PipelineResult(
        post=post,
        is_checkable=True,
        verdicts=verdicts,
        post_verdict=post_verdict,
        post_confidence=post_confidence,
    )

    print(f"\n{'='*60}")
    print(f"POST VERDICT: {post_verdict} (confidence: {post_confidence:.2f})")
    print("RESULT (JSON):")
    print(result.model_dump_json(indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default test post if none provided
        test_post = (
            "BREAKING: Scientists confirm that drinking bleach cures COVID-19. "
            "The WHO has officially endorsed this treatment as of last week."
        )
    else:
        test_post = " ".join(sys.argv[1:])

    run(test_post)
