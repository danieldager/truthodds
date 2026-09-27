#!/bin/bash
# Money-ladder smokes: 3 claims (~28 reads) per candidate, read-v5 byte-identical to production.
set -u
cd "$(dirname "$0")/../../../.."          # -> src/
OUT=eval/data/reader_lab/money_ladder
run () {  # tag  model  extra-flags
  echo "=== $(date +%H:%M:%S) SMOKE $1 $2 $3"
  uv run python -m eval.scripts.build_eval.reader_lab \
     --run eval/data/urn_runs/averitec_dev/scores.jsonl \
     --claims $OUT/../refit_v7qa/claims_averitec.json \
     --prompt v5 --model "$2" --out $OUT/${1}_smoke \
     --limit 3 --workers 16 --gate-start 8 --gate-cap 16 $3 2>&1 | tail -22
  echo "=== $(date +%H:%M:%S) END SMOKE $1 rc=$?"
}
run oss_med   openai/gpt-oss-120b                                  "--reasoning medium --no-oss-json --max-tokens 1600"
run oss_high  openai/gpt-oss-120b                                  "--reasoning high --no-oss-json --max-tokens 2500"
run qwen235   Qwen/Qwen3-235B-A22B-Instruct-2507                   ""
run maverick  meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8    ""
run kimi_k2   moonshotai/Kimi-K2-Instruct-0905                     ""
echo "=== $(date +%H:%M:%S) ALL SMOKES DONE"
