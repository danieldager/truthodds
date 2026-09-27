# `eval/scripts/_archive/` — retired eval scripts

Frozen, **not runnable as-is**: an archived script that imports another archived script has a
dead import path (`eval.scripts.build_eval.<name>`). To re-run one, `git mv` it back first.
Their *outputs* are tracked where they mattered — see `REPO_MAP.md` › run directories.

## Top level (pre-2026-08 eras)
The eval-set build line (`build_eval.py`, `survey_publishers.py`, `survey_report.py`,
`probe_feed_publishers.py`, `add_binary_label.py`, `mine_corrections*.py`,
`harvest_incremental.py`), the AllSides-era harvesters (`harvest_*.py`), source-accessibility
probes, and the `claim_cascade/`, `feed_study/`, `verdict_confidence/`, `cache_eval/`,
`searxng_probe/`, `checkthat_t2/`, `extraction_grading/`, `verify_v0_3_satellites/` studies.
All superseded by the fc-gold line and the urn instrument.

## `build_eval/` — archived 2026-09-12
Closed investigations whose conclusions live in `src/clog/` and `src/clog/ROADMAP.md`.
Nothing in the live tree imports any of them.

**Retrieval forensics (Aug 2026, closed — "retrieval is the bottleneck", oracle probe).**
`refute_verify` · `fc_query_test` · `ceiling_test` · `new_refute_check` · `silent_audit` ·
`oracle_probe` · `retrieval_recovery_arms` · `retrieval_forensics` · `exa_probe` · `two_hop` ·
`slot_substitution_probe` · `no_ceiling_arms` · `no_ceiling_cn`

**Reader A/B relics — superseded 2026-09-09 by `reader_lab.py`.**
`read_ab` · `read_polarity_ab` · `build_read_suite` · `build_read_label_ui` ·
`build_read_flag_ui` · `score_bakeoff`

**The 2026-08-28 ladder report** (its deliverable `urn_runs/e1_ctx/model_ladder/model_ladder.html`
and the figures are tracked). `model_ladder_html` · `model_ladder_figs` ·
`claim_type_ladder` · `claim_type_side` · `reportability` · `reportability_ladder` ·
`coverage_audit` · `coverage_audit_html` · `coverage_audit_timeline`

**Synthetic / band / stratum one-offs (Aug 2026).**
`synth_band_expt` · `gold_synth` · `false_mix` · `true_arms` · `rel_urn` · `urn_band_router` ·
`error_census` · `cf_probe` · `acquitted_funnel` · `acquitted_yield` · `true_stratum_smoke` ·
`screen_graded_post` · `synth_expt/` (six scripts that were living inside the data tree at
`eval/data/urn_runs/synth_expt/`)

**Superseded by a named successor.**
`wire_true_screens`, `outlet_urn_run` → `wire_urn_run` + `wire_true_fit` (2026-09-10) ·
`escalate_unsure`, `verdict_synthesis` → the code verdict in `pipeline/verify_tweet_claims.py` ·
`resolve_claims` → `contextualize_claims` · `compare_runs` → `refit_report` ·
`mode_manipulation_check`, `cap_sweep`, `run_health` → one-shot checks, results in the clog

## `build_eval/` — archived 2026-09-21 (reduce-to-frozen-tool; last live commit `390e45d`)
Analysis / figure / ladder / audit scripts and the superseded E2/E3 tweet-corpus stack, archived when
the repo was cut down to the frozen read-v5 seven-flag instrument. None is on any reported-result
lineage path (fc_gold headline, AVeriTeC transfer, two-urn transfer, prodregime, CN survey) and nothing
live imports them. Their run OUTPUTS stay tracked where they mattered (see `REPO_MAP.md`).

- **Analysis / figures around the fit** (not dataset builders): `prodregime_analysis`,
  `prodregime_date_ceiling`, `e2_figures`, `e1_veracity_bands`, `urn_seven_vs_six`, `urn_fit_stability`,
  `refit_report`, `cross_ladder`, `transfer_ladder`, `query_lab`, `timeline_eps_audit`. (The results
  PRODUCERS `prodregime_run`/`prodregime_score` stay live; `model_ladder`/`quality_urn` stay live because
  `claimverify/fcgold_urn_oof` and `model_ladder` import them.)
- **the PI's 2×2** (closed 2026-09-08): `issue24_2x2`, `issue24_cross_12cell`.
- **E2/E3 tweet + key-claim corpus** (superseded by the CN survey + two-urn pivot; off all reported
  paths): `tweet_urn_run`, `tweet_screen_claims`, `tweet_urn_report`, `claim_prominence`,
  `key_claim_select`. (`keyclaim_urn_run` stays live: `wire_urn_run` imports its query/read functions.)
- **General-pool descriptive report**: `general_pool_build` (its screen library `general_pool_screen`
  stays live — `ingest_timeline_capture` imports it for the x_feed chain).
- **fc-gold build audits** (human-review HTML, no dataset output; the ruling record for the kept fc_gold
  chain): `build_fc_gold_audit_html`, `build_fci_join_audit`, `build_fc_gold_filter_review`; plus the
  Phase-0 GFC API smoke `fctapi_smoke`.
- **`bucket_opt/`** — the bucket-optimisation experiment (`bucketopt_core/run/structs`, `diag28`,
  `extra_axes`, `make_tables`).

## `claim_sourcing/` — archived 2026-09-21 (reduce-to-frozen-tool; last live commit `390e45d`)
The collaborator-facing review/dashboard builders and the old outlet-tweet (dev-500) verify chain,
superseded by the Community-Notes survey path. None feeds a reported dataset. The LIVE claim_sourcing
survey chain kept in place = `community_notes_api`, `harvest_note_eligible_posts`,
`select_cn_survey_examples`, plus the production extraction chain `extract_tweet_claims` /
`normalize_tweet_claims` / `build_verify_input` and `ingest_timeline_capture` (these feed x_feed + cn_false),
and `harvest_outlet_tweets` (kept, lineage uncertain — wire_true corpus provenance).
- **Old outlet-tweet / dev-500 verify + scoring**: `run_tweet_verify`, `compute_scorecard`,
  `evidence_profile_run`, `survey_screen_claims`, `sample_survey_subsets`, `draw_dev500b`,
  `sample_handle_tweets`, `audit_checkworthy_gate`, `check_extraction_gold`, `unrated_pool_census`,
  `divine_outlet_lean`.
- **Collaborator report / doc builders** (HTML deliverables, outputs tracked): `build_full_review`,
  `build_review_doc`, `build_script_doc`, `build_source_dashboard`, `build_truthodds_figures`,
  `apply_review_edits`, `apply_script_edits`.
