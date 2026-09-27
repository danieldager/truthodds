"""Single-claim smoke test for the Tier 3 verification loop.

Usage:
    uv run python -m scripts.tier3_smoke --claim "<claim text>"
    uv run python -m scripts.tier3_smoke --claim "..." --date-ceiling "10/27/2020"
    uv run python -m scripts.tier3_smoke --claim "..." --verbose
    uv run python -m scripts.tier3_smoke --claim "..." --eval-sees-raw
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

from pipeline.embedding import embed
from pipeline.models import AtomicClaim
from pipeline.verify import verify


def main() -> None:
    ap = argparse.ArgumentParser(description="Tier 3 smoke test")
    ap.add_argument("--claim", required=True, help="claim text to verify")
    ap.add_argument(
        "--date-ceiling",
        default=datetime.today().strftime("%m/%d/%Y"),
        help="Evidence date ceiling MM/DD/YYYY (default: today)",
    )
    ap.add_argument("--verbose", action="store_true", help="print per-round trace")
    ap.add_argument(
        "--eval-sees-raw",
        action="store_true",
        help="pass raw evidence to Evaluation (default: analysis only)",
    )
    args = ap.parse_args()

    claim = AtomicClaim(text=args.claim, embedding=embed(args.claim).tolist())
    print(f"claim: {claim.text}")
    print(f"date_ceiling: {args.date_ceiling}")
    print(f"eval_sees_raw: {args.eval_sees_raw}")
    print()

    t0 = time.time()
    verdict = verify(
        claim,
        date_ceiling=args.date_ceiling,
        verbose=args.verbose,
        eval_sees_raw=args.eval_sees_raw,
    )
    elapsed = time.time() - t0

    print()
    print("=" * 64)
    print(f"scores  (Likert 1-5)")
    print(f"  veracity              : {verdict.scores.veracity}")
    print(f"  evidence_coverage     : {verdict.scores.evidence_coverage}")
    print(f"  evidence_consistency  : {verdict.scores.evidence_consistency}")
    print(f"  source_quality        : {verdict.scores.source_quality}")
    print()
    print(f"justification: {verdict.justification}")
    print()
    print(f"rounds_used    : {verdict.rounds_used}")
    print(f"cap_hit        : {verdict.cap_hit}")
    print(f"redundant_exit : {verdict.redundant_exit}")
    print()
    print(f"queries issued ({len(verdict.past_queries)}):")
    for i, q in enumerate(verdict.past_queries, 1):
        print(f"  {i}. {q}")
    print()
    print(f"retrieval funnel:")
    print(f"  urls seen           : {verdict.n_urls_seen}")
    print(f"  blocked / failed    : {verdict.n_blocked_or_failed}")
    print(f"  scraped irrelevant  : {verdict.n_irrelevant}")
    print(f"  kept (relevant)     : {len(verdict.evidence_urls)}")
    for u in verdict.evidence_urls:
        print(f"    - {u}")
    print()
    print(f"analysis:\n  {verdict.analysis}")
    print()
    print(f"elapsed: {verdict.elapsed_seconds:.1f}s ({elapsed:.1f}s wall) | llm_calls: {verdict.llm_calls}")


if __name__ == "__main__":
    main()
