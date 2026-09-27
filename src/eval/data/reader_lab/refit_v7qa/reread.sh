#!/bin/bash
# Re-read the three frozen populations with the production reader (v7qa two-question
# prompt + map_qa, gpt-oss-120b reasoning low @ DeepInfra). Zero retrieval: every read
# replays a stored document region. Resumable — each read is cached per
# (prompt hash, model, claim, rank) under <pop>/cache/, so a relaunch pays only for what
# is missing. 65,465 reads, $0.000093/read measured => $6.07 for a cold pass.
#
# Blocked 2026-09-10 22:47: DeepInfra's gpt-oss pool answers 429 "Model busy"
# intermittently at concurrency ONE. Re-probe before relaunching.
#
#   bash eval/data/reader_lab/refit_v7qa/reread.sh
# then, once all four finish:
#   for p in fc_gold cn_false x_feed; do ... substitute_reader_flags substitute ... ; done
#   uv run python -m eval.scripts.build_eval.refit_v7qa --sub-dir <dir>
set -u
cd "$(dirname "$0")/../../../.."          # -> src/
OUT=eval/data/reader_lab/refit_v7qa
# $1 (or $READ_MODEL) overrides the reader model, so the Turbo pool can be tried without
# editing this file: `bash reread.sh openai/gpt-oss-120b-Turbo`. Default is the base pool.
# NOTE the cache key includes the model, so a different model re-reads from scratch and
# writes its own output file (<prompt>__<model tag>.jsonl) — the two never mix.
MODEL="${1:-${READ_MODEL:-openai/gpt-oss-120b}}"
COMMON="--prompt v7qa --model $MODEL --reasoning low --no-oss-json --max-tokens 1000 --workers 64 --gate-start 8 --gate-cap 64"
echo "=== $(date +%H:%M:%S) CHAIN START model=$MODEL" >> $OUT/run_all.log
run () {   # name  run-file  claims  outdir
  echo "=== $(date +%H:%M:%S) START $1" >> $OUT/run_all.log
  uv run python -m eval.scripts.build_eval.reader_lab --run "$2" --claims "$3" --out "$4" $COMMON >> $OUT/run_all.log 2>&1
  rc=$?          # capture BEFORE any command substitution: $(date) clobbers $?
  echo "=== $(date +%H:%M:%S) END $1 rc=$rc" >> $OUT/run_all.log
}
run fc_gold      eval/data/urn_runs/e1_ctx/results-00.jsonl       $OUT/claims_fc_gold.json  $OUT/fc_gold
run cn_false_c2  eval/data/urn_runs/c2_false/scores.jsonl         $OUT/claims_cn_false.json $OUT/cn_false_c2
run cn_false_ext eval/data/urn_runs/c2_false/scores_ext.jsonl     $OUT/claims_cn_false.json $OUT/cn_false_ext
run x_feed       eval/data/urn_runs/true_timeline/scores.jsonl    $OUT/claims_x_feed.json   $OUT/x_feed
echo "=== $(date +%H:%M:%S) ALL DONE" >> $OUT/run_all.log
