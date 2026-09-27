#!/bin/bash
# Re-read the AVeriTeC dev urn run (eval/data/urn_runs/averitec_dev/scores.jsonl, the
# 7 Sept issue-28 leg-2 run: 491 claims / 4,640 flagged docs of which 4,573 have a
# stored region) with the same production reader as reread.sh and reread_wire_true.sh:
# v7qa + map_qa on gpt-oss-120b reasoning low @ DeepInfra. Zero retrieval, resumable.
# 4,573 reads at the measured $0.000088/read => ~$0.40 for a cold pass.
#
# WHY: v7qa lost at claim level on fc_gold after a reader-consistent refit (AUC
# 0.856 -> 0.816, recall@2% 0.40 -> 0.23). This is the second test before deciding
# whether to revert: recompute the issue-28 AVeriTeC 7-flag AUC (0.868, frozen-ladder
# transfer) on v7qa flags over the same stored documents.
#
# There is no frozen parquet for this population, so the claim list is derived from the
# run file itself (all non-excluded review_urls) -> claims_averitec.json.
#
# WAITS for the wire_true leg to print its own anchored END line: the DeepInfra pool is
# shared and reader jobs run strictly one at a time. Anchored, and nothing echoed here
# can match the pattern being polled (08:51 lesson).
set -u
cd "$(dirname "$0")/../../../.."          # -> src/
OUT=eval/data/reader_lab/refit_v7qa
MODEL="${1:-${READ_MODEL:-openai/gpt-oss-120b}}"
COMMON="--prompt v7qa --model $MODEL --reasoning low --no-oss-json --max-tokens 1000 --workers 64 --gate-start 8 --gate-cap 64"
echo "=== $(date +%H:%M:%S) averitec parked, waiting for the outlet leg to finish" >> $OUT/run_all.log
until grep -qE "^=== [0-9:]+ END wire_true rc=0" $OUT/run_all.log; do sleep 30; done
echo "=== $(date +%H:%M:%S) CHAIN START model=$MODEL (averitec)" >> $OUT/run_all.log
run () {   # name  run-file  claims  outdir
  echo "=== $(date +%H:%M:%S) START $1" >> $OUT/run_all.log
  uv run python -m eval.scripts.build_eval.reader_lab --run "$2" --claims "$3" --out "$4" $COMMON >> $OUT/run_all.log 2>&1
  rc=$?          # capture BEFORE any command substitution: $(date) clobbers $?
  echo "=== $(date +%H:%M:%S) END $1 rc=$rc" >> $OUT/run_all.log
}
run averitec eval/data/urn_runs/averitec_dev/scores.jsonl $OUT/claims_averitec.json $OUT/averitec
echo "=== $(date +%H:%M:%S) AVERITEC DONE" >> $OUT/run_all.log
