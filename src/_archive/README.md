# `_archive/` — what is frozen, and where

Archived ≠ deleted. Everything here is still in git (`git log --follow` works), it is simply
**not part of the live tree and not runnable as-is** (module paths assume the pre-archive
layout). Nothing live imports anything under an `_archive/`.

There are two archive roots, by area:

| Root | Holds |
|---|---|
| `src/_archive/` (this directory) | whole retired **subsystems** — a pipeline generation, a study, a tooling era |
| `src/eval/scripts/_archive/` | retired **eval scripts**, incl. `build_eval/` (see its own README for the per-script index) |
| `src/docs/archive/` | superseded design docs and dated audits (see `src/docs/README.md`) |

## Subsystems in this directory

| Bundle | What it was | Superseded |
|---|---|---|
| `v0_1/` | pipeline v0.1 (report, slides, eval checkpoints, plots) | 2026-05-18, by v0.2 |
| `verify_v0_2/` | second-generation verification stack | 2026-06-04 |
| `verify_v0_3_claimcheck/` | claim-level Tier-3 loop (`verify.py`, `verify_prompts.py`, `pipeline.py` orchestrator) + its runners | 2026-07-13, by the post-level loop `pipeline/verify_tweet_claims.py` |
| `pre_pivot_survey_tooling/` | RSS/ACE-era survey tooling, shortlist chain, feed-cascade relics | 2026-07-13, by the outlet-X-post harvest |
| `extraction_superseded/` | pass-1/pass-2 progression + bakeoff review docs | 2026-07-13, by the v4.7 two-pass chain |
| `claim_selection_gate/` | the Stage-1 scope gate study (prompt v1–v8, audits) | paused, then superseded by the claim-category code gate |
| `feed_image_study/` | image-inclusive feed study | 2026-06-26 |
| `loop_audits_v3_v4/` | verify-loop v3/v4 audit tooling | 2026-07 |
| `tier1_tier2_unbuilt/` | Tier-1 cache / Tier-2 FCT lookup sketches, never built out | spec'd only |
| `verdict_dataset/` | first verdict-dataset build | 2026-07 |
| `searxng/` | self-hosted SearXNG config; free tier-1 search, demoted | 2026-06 |
| `reports_html/` | superseded published HTML reports (AVeriTeC verification, dev50 verify reviews ×3, feed-study results, Monday deliverables, tweet-claim extraction review, verification dev200) — dated 2026-06/07, from the pre-frozen-tool era | 2026-09-21 |
| `factcheck_harvest_lib/` | the AllSides / per-publisher fact-check scrape library — the ClaimReview extractor (`claimreview.py`), per-publisher parsers (snopes/politifact/leadstories/twentyminutes/aap/afp/factcheckorg/fullfact/newschecker), `source_fetch.py`, the `scripts/` scrapers (`harvest_aap/afp/factcheckorg/fullfact`, `_pool`, `audit_*`, `build_dataset_splits`, `clean_harvest_text`, `combine_dataset`, `fetch_afp_sources`) and the `harvesters/` dir. Built the legacy `*_harvest.parquet` → `dataset_12mo`/`factcheck_textonly`/`eval_v1` line. **NOT on the fc-gold spine** — fc_gold pulls fresh from the Google Fact Check Tools API via `eval/scripts/build_eval/harvest_full.py` (which, with `eval/harvest.py` + `eval/harmonize.py`, stays LIVE). | 2026-09-21, by the GFC/fc-gold line |
