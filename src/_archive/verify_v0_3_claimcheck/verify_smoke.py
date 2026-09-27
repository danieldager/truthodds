"""Verbose single-claim smoke test for the Tier-3 verifier.

Usage:
    python scripts/verify_smoke.py "Some claim to verify"
    python scripts/verify_smoke.py            # uses a default claim

Prints the full trace (queries, evidence funnel, both verdicts). Hits the live Serper +
LLM backends, so it needs SERPER_API_KEY + DEEPINFRA_API_KEY set.
"""
from __future__ import annotations

import sys

from pipeline.embedding import embed
from pipeline.models import AtomicClaim
from pipeline.verify import get_cache_stats, reset_cache_stats, verify

DEFAULT_CLAIM = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or DEFAULT_CLAIM
    claim = AtomicClaim(text=text, embedding=embed(text).tolist())

    print(f"CLAIM: {text}\n")
    reset_cache_stats()
    v = verify(claim, verbose=True)

    print("\n--- VERDICTS ---")
    print(f"4-class : {v.verdict_4class}")
    print(f"  why   : {v.verdict_4class_justification}")
    print(f"Likert  : veracity={v.scores.veracity} sufficiency={v.scores.evidence_sufficiency} "
          f"agreement={v.scores.evidence_agreement} reliability={v.scores.source_reliability}")
    print(f"  why   : {v.justification}")
    print("\n--- ANALYSIS ---")
    print(v.analysis)
    print("\n--- TRACE ---")
    print(f"stopped_reason : {v.stopped_reason}  (rounds_used={v.rounds_used}, cap_hit={v.cap_hit})")
    print(f"queries        : {v.past_queries}")
    print(f"funnel         : seen={v.n_urls_seen} scraped={v.n_scraped} "
          f"snippet_only={v.n_snippet_used} irrelevant={v.n_irrelevant} "
          f"blocked={v.n_blocked_or_failed}")
    print(f"llm_calls={v.llm_calls}  elapsed={v.elapsed_seconds}s")
    cs = get_cache_stats()
    print(f"prompt cache   : {cs['cached_tokens']}/{cs['prompt_tokens']} prompt tokens cached "
          f"({cs['cached_pct']}%)")
    print(f"evidence_urls  : {v.evidence_urls}")


if __name__ == "__main__":
    main()
