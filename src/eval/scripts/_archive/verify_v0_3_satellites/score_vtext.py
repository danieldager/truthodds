"""Score verify_text output against the hand-audited misinfo labels (nudge-level + type)."""

# NOTE (2026-07-28 audit): this module has NO `if __name__ == "__main__"` guard —
# importing it RUNS the verify_text scoring and prints to stdout. Left as-is rather than refactored:
# it consumes output from the archived claim-level loop
# (src/_archive/verify_v0_3_claimcheck/), so it is a record, not live tooling.
# If you ever import it programmatically, wrap the body first.

import sys, json
import polars as pl
DATA = "src/eval/scripts/verification_grading/data"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/vtext_verdicts_full.parquet"

D = pl.read_parquet(OUT)
D = D.with_columns([(pl.col("v_veracity") <= 3).alias("pred_nudge"),
                    (pl.col("audit_veracity") <= 3).alias("gold_nudge")])
D = D.with_columns((pl.col("pred_nudge") == pl.col("gold_nudge")).alias("nudge_ok"))
N = D.height
print(f"=== verify_text vs AUDITED labels — N={N} ===\n")

# 1. Headline nudge accuracy + confusion
tp = D.filter(pl.col("pred_nudge") & pl.col("gold_nudge")).height
tn = D.filter(~pl.col("pred_nudge") & ~pl.col("gold_nudge")).height
fp = D.filter(pl.col("pred_nudge") & ~pl.col("gold_nudge")).height   # nudged a non-misinfo (over-nag)
fn = D.filter(~pl.col("pred_nudge") & pl.col("gold_nudge")).height   # missed misinfo (under-flag)
rec = tp/(tp+fn) if tp+fn else 0; prec = tp/(tp+fp) if tp+fp else 0
print(f"NUDGE accuracy: {D['nudge_ok'].mean():.3f}  ({D['nudge_ok'].sum()}/{N})")
print(f"  recall(catch misinfo)={rec:.3f}  precision={prec:.3f}  | under-flags(FN)={fn}  over-nags(FP)={fp}")

# 2. Type agreement (only where both nudge)
print("\nTYPE match (audit_type vs v_type, on agreed-nudge claims):")
both = D.filter(pl.col("pred_nudge") & pl.col("gold_nudge"))
print(f"  exact type match: {(both['audit_type']==both['v_type']).sum()}/{both.height}")
ct = both.group_by(["audit_type", "v_type"]).len().sort("len", descending=True)
for r in ct.head(10).to_dicts():
    print(f"    audit {r['audit_type']:11s} -> v {r['v_type']:11s}  {r['len']}")

# 3. Loop behavior / cost
print("\nLOOP: round distribution:", sorted(D["n_search_rounds"].to_list()))
print(f"  used READ: {D.filter(pl.col('n_read')>0).height}  used Exa: {D.filter(pl.col('provider_sequence').str.contains('exa')).height}"
      f"  cap-hit(>=10q): {D.filter(pl.col('n_search_rounds')>=10).height}")
sr = [json.loads(t) for t in D["trace_json"].to_list()]
from collections import Counter
stops = Counter(t.get("stopped") for t in sr)
reps = sum(t.get("counters", {}).get("json_repairs", 0) for t in sr)
fails = sum(t.get("counters", {}).get("parse_fails", 0) for t in sr)
print(f"  stop reasons: {dict(stops)}  | json_repairs={reps}  parse_fails={fails}")
print(f"  mean elapsed: {D['elapsed_s'].mean():.0f}s  mean llm_calls: {D['llm_calls'].mean():.1f}  max: {D['elapsed_s'].max():.0f}s")

# 4. The 3 with-post divergences — did seeing the post catch them?
print("\nWITH-POST DIVERGENCES (gold!=audit; the weaponization acid test):")
for r in D.filter(~pl.col("matches_gold")).to_dicts():
    ok = "✓" if (r["v_veracity"] <= 3) == (r["audit_veracity"] <= 3) else "✗"
    print(f"  {ok} audit={r['audit_veracity']}/{r['audit_type']} -> v={r['v_veracity']}/{r['v_type']}: {r['main_claim'][:60]}")

# 5. The misses (under-flags) — the true-but-misleading problem
print("\nUNDER-FLAGS (missed misinfo — audit nudge, model pass):")
for r in D.filter(~pl.col("pred_nudge") & pl.col("gold_nudge")).to_dicts():
    print(f"  audit={r['audit_veracity']}/{r['audit_type']} -> v={r['v_veracity']}/{r['v_type']} ({r['n_search_rounds']}q): {r['main_claim'][:60]}")
