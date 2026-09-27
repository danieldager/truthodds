# `verification_grading/` — Tier 3 verification distribution analysis

Mirror of `extraction_grading/` for the Tier 3 verification loop (`pipeline/verify.py`).

Same shape: data collection on a known-claim corpus → distribution analysis of our model's Likert scores + diagnostics. No comparison to gold labels in v1 — gold labels are *saved* alongside our outputs so future runs can derive 4-class labels and compute macro-F1 against ClaimCheck / PASS-FC / HerO benchmarks.

## Pipeline

```
load_claims.py   chenxwh/AVeriTeC dev → data/claims_n{N}.parquet
verify_run.py    pipeline.verify(claim) → data/verdicts_n{N}.parquet
```

## Schemas

`claims_n{N}.parquet` — one row per claim:
- `claim_id`           — stable sha1[:16] hash of original_claim_url
- `claim_text`         — the claim string
- `gold_label`         — AVeriTeC verdict (Supported / Refuted / NEI / Conflicting Evidence/Cherrypicking)
- `claim_date`         — date the claim was published (used for `date_ceiling`)
- `speaker`            — claim's speaker if known
- `fact_checking_article` — URL of the fact-check article
- `gold_justification` — AVeriTeC's human justification
- `original_claim_url` — source URL

`verdicts_n{N}.parquet` — one row per claim, joinable by `claim_id`:
- Scores (Likert 1-5): `veracity`, `evidence_coverage`, `evidence_consistency`, `source_quality`
- Output text: `justification`, `analysis`
- Evidence: `evidence_urls` (list[str]), `past_queries` (list[str])
- Diagnostics: `rounds_used`, `n_urls_seen`, `n_blocked_or_failed`, `n_irrelevant`, `elapsed_seconds`, `llm_calls`, `cap_hit`, `redundant_exit`, `tier_resolved`
- `error` — null on success, "<ExcType>: <msg>" on failure

## Notes

- AVeriTeC version: dataset is `chenxwh/AVeriTeC` (used by FEVER 2024/2025 shared tasks). ClaimCheck (76.4 dev accuracy) reportedly used AVeriTeC 1.0 dev, but the schemas/labels are compatible.
- All runs are resumable via `claim_id` dedup on the output parquet.
- Date ceiling for evidence filtering: parsed from `claim_date` (handles both `MM/DD/YYYY` and `YYYY-MM-DD`); falls back to today if missing.
