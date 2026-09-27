"""Central registry of eval data paths. Import this instead of hardcoding locations.

    from eval import paths
    df = pl.read_parquet(paths.VERDICT_DATASET)

Checkout-relative (no absolute /Users/... paths), so scripts run in any clone. When a dataset moves,
update ONLY this file. Big blobs live gitignored under DATA; small manifests + published deliverables
are tracked (see .gitignore).
"""
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]      # .../src
DATA = SRC / "eval" / "data"                    # gitignored data root

# --- survey-experiment claim sourcing (docs/claim_sourcing_pipeline.md) ---
SURVEY = DATA / "survey_claims"
CLAIM_POOL          = SURVEY / "claim_pool.csv"                    # tracked deliverable
CLAIM_POOL_AUDIT    = SURVEY / "claim_pool_audit.csv"             # tracked (keep/drop record)
CLAIM_POOL_VERIFY   = SURVEY / "claim_pool_verify.parquet"        # verify_text INPUT (regenerable from CSV)
CLAIM_POOL_VERDICTS = SURVEY / "claim_pool_verdicts.parquet"      # verify_text OUTPUT
SOURCES_WORKBOOK    = SURVEY / "baseline_sources_and_claims.xlsx"  # tracked (sources + claims tabs)
ACE_ATTEMPTS        = DATA / "ace_attempts.parquet"               # every ACE extraction attempt
CAPTURES            = SURVEY / "captures"                          # X NDJSON (X ToS — never commit)

# --- factcheck eval-dataset build (docs/dataset_methodology.md) ---
EVAL_V1            = DATA / "eval_v1.parquet"                     # compiled gold (tracked)
DATASET_12MO       = DATA / "dataset_12mo.parquet"
FACTCHECK_TEXTONLY = DATA / "factcheck_textonly.parquet"
IMAGES             = DATA / "images"                             # per-publisher post images

# --- stage-2 extraction eval ---
STAGE2_DATASET = DATA / "stage2_dataset.parquet"

# --- stage-3 verdict eval (docs/verdict_eval_plan.md) ---
VERDICT_DATASET     = DATA / "verdict_dataset.parquet"
VERDICT_AUDIT       = DATA / "verdict_audit.parquet"
VERDICT_SCOPE_FLOOR = DATA / "verdict_scope_floor.parquet"

# --- source ratings (internal / gitignored) ---
NEWSGUARD = DATA / "newsguard"
