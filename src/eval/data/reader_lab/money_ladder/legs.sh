#!/bin/bash
# Money-ladder full AVeriTeC legs, SEQUENTIAL (never two readers on one DeepInfra pool).
set -u
cd "$(dirname "$0")/../../../.."          # -> src/
OUT=eval/data/reader_lab/money_ladder
CLAIMS=eval/data/reader_lab/refit_v7qa/claims_averitec.json
RUN=eval/data/urn_runs/averitec_dev/scores.jsonl
LOG=$OUT/legs.log
leg () {  # tag  model  extra-flags
  echo "=== $(date +%H:%M:%S) START $1  model=$2  flags=$3" >> $LOG
  uv run python -m eval.scripts.build_eval.reader_lab --run $RUN --claims $CLAIMS \
     --prompt v5 --model "$2" --out $OUT/$1 --workers 64 --gate-start 8 --gate-cap 64 $3 >> $LOG 2>&1
  rc=$?
  echo "=== $(date +%H:%M:%S) END $1 rc=$rc" >> $LOG
}
leg oss_med  openai/gpt-oss-120b                               "--reasoning medium --max-tokens 3000"
leg qwen235  Qwen/Qwen3-235B-A22B-Instruct-2507                ""
leg maverick meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 ""
leg oss_high openai/gpt-oss-120b                               "--reasoning high --max-tokens 4000"
# gpt-oss at high returns an empty body on ~8% of reads and those parse failures ARE cached,
# so sweep them and re-run twice; a cached good read replays free.
for pass in 1 2; do
  n=$(python3 - <<'PY'
import json,glob,os
k=0
for f in glob.glob('eval/data/reader_lab/money_ladder/oss_high/cache/*.json'):
    d=json.load(open(f))
    if d.get('qc_flag')=='read-failed':
        os.remove(f); k+=1
print(k)
PY
)
  echo "=== $(date +%H:%M:%S) oss_high sweep $pass: purged $n failed reads" >> $LOG
  [ "$n" = "0" ] && break
  leg oss_high openai/gpt-oss-120b "--reasoning high --max-tokens 4000"
done
echo "=== $(date +%H:%M:%S) ALL LEGS DONE" >> $LOG
