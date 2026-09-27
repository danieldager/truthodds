#!/bin/bash
# Re-read the WIRE-TRUE population (eval/data/urn_runs/true_outlet/scores.jsonl, the
# 27 Aug outlet run, 1,217 claims / 11,098 flagged docs of which 11,038 have a stored
# region) with the same production reader as reread.sh: v7qa + map_qa on
# gpt-oss-120b reasoning low @ DeepInfra. Zero retrieval, resumable per read.
# 11,038 reads at the measured $0.000088/read => ~$1.0 for a cold pass.
#
# There is no frozen parquet for this population, so the claim list is derived from the
# run file itself (all non-excluded review_urls) -> claims_wire_true.json.
#
# WAITS for the 2026-09-11 four-leg chain (PID 44018) to print ALL DONE: the DeepInfra
# pool is shared and the two must not run concurrently.
set -u
cd "$(dirname "$0")/../../../.."          # -> src/
OUT=eval/data/reader_lab/refit_v7qa
MODEL="${1:-${READ_MODEL:-openai/gpt-oss-120b}}"
COMMON="--prompt v7qa --model $MODEL --reasoning low --no-oss-json --max-tokens 1000 --workers 64 --gate-start 8 --gate-cap 64"
# Match ONLY the chain's own end line, anchored. A loose `grep -q "ALL DONE"` matched the
# waiting message this script had just written to the same log (08:51, killed at 50 reads).
echo "=== $(date +%H:%M:%S) wire_true parked, waiting for the chain to finish" >> $OUT/run_all.log
until grep -qE "^=== [0-9:]+ ALL DONE$" $OUT/run_all.log; do sleep 30; done
echo "=== $(date +%H:%M:%S) CHAIN START model=$MODEL (wire_true)" >> $OUT/run_all.log
run () {   # name  run-file  claims  outdir
  echo "=== $(date +%H:%M:%S) START $1" >> $OUT/run_all.log
  uv run python -m eval.scripts.build_eval.reader_lab --run "$2" --claims "$3" --out "$4" $COMMON >> $OUT/run_all.log 2>&1
  rc=$?          # capture BEFORE any command substitution: $(date) clobbers $?
  echo "=== $(date +%H:%M:%S) END $1 rc=$rc" >> $OUT/run_all.log
}
run wire_true eval/data/urn_runs/true_outlet/scores.jsonl $OUT/claims_wire_true.json $OUT/wire_true
echo "=== $(date +%H:%M:%S) WIRE TRUE DONE" >> $OUT/run_all.log
