#!/bin/bash
# PROMPT-vs-MODEL control on AVeriTeC (Daniel 2026-09-11, redirect): re-read the same
# AVeriTeC dev regions with the PRODUCTION read prompt (PROMPTS["v5"] = READ_SYS_MODE,
# byte-for-byte, no mapper) on the SAME model as the v7qa leg, openai/gpt-oss-120b
# reasoning low @ DeepInfra. Zero retrieval, resumable. 4,573 reads, ~$0.43.
#
# WHY: v7qa lost on AVeriTeC too (frozen-ladder transfer 0.868 -> 0.804, oof self-fit
# 0.861 -> 0.809), but the PROMPT and the MODEL moved together — production is read-v5
# on DeepSeek-V4-Flash. Holding the model at gpt-oss and swapping only the prompt back
# separates the two. AVeriTeC replaces the fc_gold version of this control: same
# question, 4,573 reads instead of 30,776.
#
# The read cache key includes the prompt hash, so this shares nothing with the v7qa
# AVeriTeC reads and writes its own out dir.
set -u
cd "$(dirname "$0")/../../../.."          # -> src/
OUT=eval/data/reader_lab/refit_v7qa
MODEL="${1:-${READ_MODEL:-openai/gpt-oss-120b}}"
COMMON="--prompt v5 --model $MODEL --reasoning low --no-oss-json --max-tokens 1000 --workers 64 --gate-start 8 --gate-cap 64"
echo "=== $(date +%H:%M:%S) CHAIN START model=$MODEL prompt=v5 (averitec_v5oss)" >> $OUT/run_all.log
run () {   # name  run-file  claims  outdir
  echo "=== $(date +%H:%M:%S) START $1" >> $OUT/run_all.log
  uv run python -m eval.scripts.build_eval.reader_lab --run "$2" --claims "$3" --out "$4" $COMMON >> $OUT/run_all.log 2>&1
  rc=$?          # capture BEFORE any command substitution: $(date) clobbers $?
  echo "=== $(date +%H:%M:%S) END $1 rc=$rc" >> $OUT/run_all.log
}
run averitec_v5oss eval/data/urn_runs/averitec_dev/scores.jsonl $OUT/claims_averitec.json $OUT/averitec_v5oss
echo "=== $(date +%H:%M:%S) AVERITEC_V5OSS DONE" >> $OUT/run_all.log
