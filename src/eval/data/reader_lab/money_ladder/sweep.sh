#!/bin/bash
# gpt-oss on DeepInfra returns an empty body on a few percent of reads; reader_lab CACHES that
# parse failure (no `err` key) so a plain re-run will not retry it. Purge and re-read, twice.
set -u
cd "$(dirname "$0")/../../../.."
OUT=eval/data/reader_lab/money_ladder
LOG=$OUT/legs.log
sweep () {   # tag  flags
  for pass in 1 2 3; do
    n=$(python3 -c "
import json,glob,os,sys
k=0
for f in glob.glob('eval/data/reader_lab/money_ladder/$1/cache/*.json'):
    d=json.load(open(f))
    if d.get('qc_flag')=='read-failed':
        os.remove(f); k+=1
print(k)")
    echo \"=== $(date +%H:%M:%S) $1 sweep $pass: purged $n failed reads\" >> $LOG
    [ "$n" = "0" ] && return
    echo "=== $(date +%H:%M:%S) START $1 sweep $pass" >> $LOG
    uv run python -m eval.scripts.build_eval.reader_lab \
      --run eval/data/urn_runs/averitec_dev/scores.jsonl \
      --claims $OUT/../refit_v7qa/claims_averitec.json \
      --prompt v5 --model openai/gpt-oss-120b --out $OUT/$1 \
      --workers 64 --gate-start 8 --gate-cap 64 $2 >> $LOG 2>&1
    echo "=== $(date +%H:%M:%S) END $1 sweep $pass rc=$?" >> $LOG
  done
}
sweep oss_med  "--reasoning medium --max-tokens 3000"
sweep oss_high "--reasoning high --max-tokens 4000"
echo "=== $(date +%H:%M:%S) SWEEPS DONE" >> $LOG
