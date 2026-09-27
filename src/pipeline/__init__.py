"""Fact-checking pipeline package.

Canonical verification loop: `pipeline/verify_tweet_claims.py` (see docs/system_snapshot_2026-07-13.md).
The old tiered orchestrator (pipeline.py, v0.2) and the claim-level Tier-3 loop live in
src/_archive/verify_v0_3_claimcheck/. This init deliberately imports nothing: submodules are
imported directly (e.g. `from pipeline.verify_tweet_claims import verify_post`).
"""
