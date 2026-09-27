# `docs/` — design docs and plans

Navigation for the repo as a whole lives in `../../REPO_MAP.md`; what is true today lives in
`../../HUB.md`; what is open lives in `../clog/ROADMAP.md`. This file only says what is in
*this* directory and which of it is still authoritative.

## Authoritative

| Doc | Purpose |
|---|---|
| `logodds_sprint.md` | **plan of record** for the current sprint — read its "WHERE WE ACTUALLY ARE" section before quoting any number |
| `../CLAUDE.md` | pipeline architecture, Key Architectural Decisions, IDEA-001…016, Open Questions |
| `eval_framework_plan.md` | workstreams, NEXT UP, decisions |
| `datasets.md` + `dataset_improvements.md` | the claim piles (fc-gold, CN false urn, timeline urn) — funnels, labels, audits, which n each number uses; the critique + queued fixes (issue #27) |
| `truth_odds.md` | the evidence-as-signal veracity framework (the urn) |
| `timeline_urn.md` · `cn_eval_design.md` · `true_urn_handoff.md` | one per pile: what is in it, what is deliberately out, the caveats |
| `verify_loop_versions.md` · `verify_tweet_claims_loop_design.md` | the loop's changelog and design |
| `survey_claim_selection.md` · `claim_selection_status.md` | the survey pool and the paused Stage-1 gate |
| `run_checklist.md` | walk it before any paid run; the ledger is `../eval/data/run_ledger.md` |
| `annotation_guideline.md` · `claim_extraction_procedure.md` · `query_best_practices.md` | procedures |
| `dev500_full_review.html` | collaborator-facing pipeline walkthrough |

## Reference and background
`factchecking_with_LLMs.pdf` (the source theory paper — do not modify) ·
`llm_judge_lit_brief.md` · `image_authentication_research.md` · `image_pipeline_brief.md` ·
`source_ratings_datasheet.md` + `open_source_rating_dataset_design.md` ·
`survey_5k_roadmap.md` · `tier1_cache_design.md` · `enrichment_audit_spec.md` ·
`mechanisms_{extraction,verification}.yaml`.

## Results notes (dated, still cited)
`query_gen_plan.md` (parked) · `read_v5_polarity_bug.md` · `attribution_read_status.md` ·
`truth_odds_experiments.md` · `tweet_urn_plan.md` · `tweet_fit_corpus.md` ·
`tweet_corpus_true_stratum_research.md` · `cn_dated_true_scoping.md` ·
`communityfact_assessment.md` · `eval_verification_notes.md` · `harvest_sources.md` ·
`verdict_eval_plan.md` · `verify_adversarial_design.md` · `read_review_design.md`.

## `archive/`
Superseded design docs and dated audits, including `repo_layout.md` (2026-07-28, the old
second repo map — superseded by `REPO_MAP.md` on 2026-09-12), the pre-v0.3 pipeline specs,
`system_snapshot_2026-07-13.md` and `conversation_log.md`.

**Convention:** when a doc here goes stale, `git mv` it to `archive/`, prepend a
`> **HISTORICAL** — superseded by …` line, and update this README.
