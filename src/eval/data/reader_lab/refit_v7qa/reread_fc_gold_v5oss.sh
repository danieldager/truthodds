#!/bin/bash
# CONTROL for the v7qa regression: re-read the fc_gold population with the PRODUCTION
# read prompt (PROMPTS["v5"] = READ_SYS_MODE, byte-for-byte) on the SAME model as v7qa,
# openai/gpt-oss-120b reasoning low @ DeepInfra. Zero retrieval, resumable.
#
# WHY: v7qa lost at claim level on fc_gold after a reader-consistent refit
# (AUC 0.8559 -> 0.8164, recall@2%FPR 0.3998 -> 0.2304), but the PROMPT and the MODEL
# changed together (production is read-v5 on DeepSeek-V4-Flash). This leg holds the
# model fixed at gpt-oss and swaps only the prompt back to v5, so the loss can be
# attributed to one or the other.
#
# v5 has no entry in MAPPERS, so the raw {"direction","evidence","reason"} JSON is
# parsed by reader_lab._parse_json — nothing else to wire. `--no-oss-json` is kept:
# DeepInfra serves gpt-oss json mode from a separate, busy pool, and read_one then
# omits `response_format` entirely (verified on a 3-claim probe before launch).
#
# The read cache key includes the prompt hash, so this shares nothing with the v7qa
# fc_gold reads and writes its own out dir: 31,273 reads, ~$2.75 at $0.000088/read.
#
# WAITS for the AVeriTeC leg's own anchored END line: the DeepInfra pool is shared and
# reader jobs run strictly one at a time. Anchored, and nothing echoed here can match
# the pattern being polled (08:51 lesson).
set -u
cd "$(dirname "$0")/../../../.."          # -> src/
OUT=eval/data/reader_lab/refit_v7qa
MODEL="${1:-${READ_MODEL:-openai/gpt-oss-120b}}"
COMMON="--prompt v5 --model $MODEL --reasoning low --no-oss-json --max-tokens 1000 --workers 64 --gate-start 8 --gate-cap 64"
echo "=== $(date +%H:%M:%S) fc_gold_v5oss parked, waiting for the AVeriTeC leg to finish" >> $OUT/run_all.log
until grep -qE "^=== [0-9:]+ END averitec rc=0" $OUT/run_all.log; do sleep 30; done
echo "=== $(date +%H:%M:%S) CHAIN START model=$MODEL prompt=v5 (fc_gold_v5oss)" >> $OUT/run_all.log
run () {   # name  run-file  claims  outdir
  echo "=== $(date +%H:%M:%S) START $1" >> $OUT/run_all.log
  uv run python -m eval.scripts.build_eval.reader_lab --run "$2" --claims "$3" --out "$4" $COMMON >> $OUT/run_all.log 2>&1
  rc=$?          # capture BEFORE any command substitution: $(date) clobbers $?
  echo "=== $(date +%H:%M:%S) END $1 rc=$rc" >> $OUT/run_all.log
}
run fc_gold_v5oss eval/data/urn_runs/e1_ctx/results-00.jsonl $OUT/claims_fc_gold.json $OUT/fc_gold_v5oss
echo "=== $(date +%H:%M:%S) FC_GOLD_V5OSS DONE" >> $OUT/run_all.log
