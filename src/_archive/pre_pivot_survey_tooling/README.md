# Archived 2026-07-13 — pre-pivot survey tooling (RSS/ACE era + feed-cascade relics)

Superseded by the 2026-07-06 pivot to harvesting outlets' own X posts
(`harvest_outlet_tweets.py` → two-pass `extract_tweet_claims.py` + `normalize_claims.py`).

- `ace.py` + `claim_extractor_prompt.md` — the Article Claim Extractor (RSS → article →
  claims chain, issues #13/#14). The .md was misleadingly named like a live prompt doc;
  the LIVE extraction prompts are the SYSTEM strings inside the two claim_sourcing scripts.
- `harvest_shortlist.py`, `filter_and_dedup_claims.py`, `make_verify_sample.py`,
  `verify_survey_run.py`, `build_verification_html.py`, `build_source_summary.py` —
  the RSS-era harvest → claim_pool → verify chain and its reports.
- `analyze_claims.py`, `cluster_claims.py` — feed-cascade era (862-post) claim analysis.

NOTE: `ace.py` was externally linked from GitHub issues at its old path (`src/eval/ace.py`);
those links now point at history. Link to
`src/_archive/pre_pivot_survey_tooling/ace.py` if it comes up again.
