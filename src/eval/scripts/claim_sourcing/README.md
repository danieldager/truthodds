# claim_sourcing/

Builds the **survey-experiment claim pool** — real news outlets → checkable claims → verified veracities.
Distinct from the eval-dataset build. Full methodology: `src/docs/claim_extraction_procedure.md`.

Run order (all: `cd src && uv run --with feedparser --with trafilatura python eval/scripts/claim_sourcing/<script>`):

0. **Outlet selection** — NewsGuard reliability score × NewsGuard orientation (Left/Right), balanced
   2×2 (reliable/unreliable × left/right), ranked by Tranco traffic and curated to news outlets.
   Shortlist at `survey_claims/source_shortlist.csv`.
1. `harvest_shortlist.py` — systematic per-outlet harvest (RSS → homepage scrape → Jina for
   bot-blocked), a federal-politics keyword scope gate, body-only NEUTRAL extraction via `eval/ace.py`,
   per-outlet caps toward ~250/cell → `survey_claims/shortlist_claims.parquet` (published as
   `claims_pull.csv`).
2. `filter_and_dedup_claims.py` — semantic dedup + check-worthiness filter over the pool.
3. `make_verify_sample.py` → `verify_survey_run.py` — sample N per cell, shape to the `verify_text`
   schema, run Stage-3 verification (fact-check-blocked, date-limited) →
   `survey_claims/shortlist_verdicts.parquet` (published as `verification_results.csv`).
4. `build_source_summary.py` — per-source summary → `sources_pull_summary.csv`.

Outputs live under `eval/data/survey_claims/` (mostly gitignored — X ToS + NewsGuard-derived cells;
only the published deliverables are tracked). Retired scripts live in `eval/scripts/_archive/`.
