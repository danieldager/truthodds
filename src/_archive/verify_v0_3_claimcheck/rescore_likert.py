"""Re-score ONLY the Likert evaluator over the cached cascade analyses (no search/synthesis).

Isolates the Likert-prompt change: feeds each claim's EXISTING synthesized analysis back
through _evaluate_likert with the new rubric, then recomputes the flag-if-veracity<=3 headline.
Lets us see — cheaply, one LLM call per claim — whether the polarity / misleadingness->3 /
unconfirmed-detail->4 fixes move the numbers before committing to a full re-run.

  uv run --directory <src> python -m eval.scripts.verification_grading.rescore_likert
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl

from config import VERIFICATION_MODEL
from pipeline.verify import _evaluate_likert

D = "eval/scripts/verification_grading/data"
def cid(u): return hashlib.md5((u or "").encode()).hexdigest()[:16]

claims = pl.read_parquet(f"{D}/claims_dev_content.parquet")
verd = pl.read_parquet(f"{D}/verdicts_dev_content_cascade.parquet").rename({"veracity": "veracity_old"})
sf = pl.read_parquet("eval/data/verdict_scope_floor.parquet").with_columns(
    pl.col("review_url").map_elements(cid, return_dtype=pl.Utf8).alias("claim_id")).select(
    "claim_id", "stage3_in_scope")
df = (claims.join(verd, on="claim_id", how="inner").join(sf, on="claim_id", how="left")
      .with_columns(pl.col("stage3_in_scope").fill_null(True)))
df = df.filter(pl.col("error").is_null() & pl.col("stage3_in_scope")
               & (pl.col("judged_axis") == "content"))
rows = df.to_dicts()
print(f"re-scoring {len(rows)} in-scope analyses with the new Likert rubric ({VERIFICATION_MODEL})")

def work(r):
    out = _evaluate_likert(r["claim_text"], r["analysis"] or "", VERIFICATION_MODEL,
                           image_context=(r.get("image_serialization") or None))
    return r["claim_id"], out["veracity"]

new = {}
with ThreadPoolExecutor(max_workers=4) as ex:
    for i, (k, v) in enumerate(ex.map(work, rows), 1):
        new[k] = v
        if i % 40 == 0:
            print(f"  {i}/{len(rows)}")

df = df.with_columns(pl.col("claim_id").replace_strict(new, default=None).alias("veracity_new"))

g = df["gold_veracity"].to_numpy()
po = df["veracity_old"].to_numpy(); pn = df["veracity_new"].to_numpy()

def report(p, tag):
    pred_flag = p <= 3; gold_flag = g <= 3
    acc = (pred_flag == gold_flag).mean()
    tp = int((gold_flag & pred_flag).sum()); fp = int((~gold_flag & pred_flag).sum())
    fn = int((gold_flag & ~pred_flag).sum()); tn = int((~gold_flag & ~pred_flag).sum())
    print(f"\n{tag}: flag-if<=3 ACC={acc:.3f}  recall={tp/(tp+fn):.3f} precision={tp/(tp+fp):.3f}  "
          f"MAE={np.abs(p-g).mean():.3f}  | misses(FN)={fn} over-flags(FP)={fp}")

report(po, "OLD Likert")
report(pn, "NEW Likert")

# movement on the 50 prior errors + regressions among prior-correct
old_flag = po <= 3; new_flag = pn <= 3; gold_flag = g <= 3
old_err = old_flag != gold_flag; new_err = new_flag != gold_flag
fixed = int((old_err & ~new_err).sum()); broke = int((~old_err & new_err).sum())
print(f"\nprior errors fixed by new Likert: {fixed}/{int(old_err.sum())}")
print(f"prior-correct newly broken (regressions): {broke}/{int((~old_err).sum())}")

df.select("claim_id", "claim_text", "gold_veracity", "veracity_old", "veracity_new",
          "rating_subtype").write_parquet("/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/767f3979-8040-4dcd-a252-87212d87cf8b/scratchpad/rescore.parquet")
print("\nsaved rescore.parquet")
