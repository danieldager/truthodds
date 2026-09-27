# Eval / testing pipeline — single entrypoint.
#
# Run from src/ (the uv project root). The end-to-end smoke:
#     make test-pipeline            # adapt -> run model -> score, on a SMALL sample
#     make harvest                  # pull recent fact-checks (FCT API, free) into the eval set
#
# Defaults are small + frugal — `run` calls the model (paid Groq + Serper/Exa per claim).
# Raise N for a real run:  make test-pipeline N=50 W=3
#
# Override the source set to score a fresh harvest snapshot:
#     make test-pipeline EVAL=eval/data/snapshots/eval_<YYYYMMDD>.parquet

PY = uv run python

EVAL     ?= eval/data/eval_v1.parquet          # source eval set (gold) — or a harvest snapshot
CLAIMS   ?= eval/data/claims_demo.parquet      # adapter output = model input
VERDICTS ?= eval/data/verdicts_demo.parquet    # model output
N        ?= 5                                   # claims to run (smoke size)
W        ?= 2                                   # verify_run workers

.PHONY: harvest eval-to-claims run score test-pipeline

## harvest — RETIRED 2026-07-28: drove harvest_incremental (38-day rolling window, 8 publishers),
## which could not even import (GROQ_* constants removed from eval/harmonize.py) and is now archived.
## The gold line is built by eval/scripts/build_eval/harvest_full.py — one exhaustive GFC pull over
## 38 sites — which is NOT a drop-in replacement for the rolling window, so this is retired, not repointed.
harvest:
	@echo "RETIRED: harvest_incremental is archived (eval/scripts/_archive/) and did not import." ; \
	echo "Full-history pull: uv run python -m eval.scripts.build_eval.harvest_full" ; exit 1

## eval-to-claims — adapt an eval parquet -> the claims schema the verifier harness consumes
eval-to-claims:
	$(PY) -m eval.scripts.build_eval.eval_to_claims -i $(EVAL) -o $(CLAIMS)

## run / test-pipeline — RETIRED 2026-07-13: they drove the old claim-level Tier-3 loop, now in
## src/_archive/verify_v0_3_claimcheck/. The canonical loop is pipeline/verify_tweet_claims.py (driven by
## scripts/run_dev50_verify_review.py); an fc-gold/AVeriTeC adapter for it is roadmap WS1.
run test-pipeline:
	@echo "RETIRED: the old-loop runner is archived (src/_archive/verify_v0_3_claimcheck/)." ; \
	echo "Canonical loop: pipeline/verify_tweet_claims.py — see docs/system_snapshot_2026-07-13.md" ; exit 1

## score — 4-class accuracy + macro-F1 of a verdicts parquet vs the harmonized gold (saved to eval/reports/)
score:
	@mkdir -p eval/reports
	$(PY) -m eval.scripts.verification_grading.score -c $(CLAIMS) -v $(VERDICTS) 2>&1 | tee eval/reports/last_score.txt
