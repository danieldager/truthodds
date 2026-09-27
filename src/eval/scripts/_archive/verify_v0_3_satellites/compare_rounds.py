"""Compare the serper-2/exa-2 baseline vs the serper-3/exa-2 re-run (same dev claims).

  uv run --directory <src> python -m eval.scripts.verification_grading.compare_rounds

Reports: how much the 3rd Serper round cut Exa escalation, and whether headline accuracy held.
"""

# NOTE (2026-07-28 audit): this module has NO `if __name__ == "__main__"` guard —
# importing it RUNS the serper-3 round comparison and prints to stdout. Left as-is rather than refactored:
# it consumes output from the archived claim-level loop
# (src/_archive/verify_v0_3_claimcheck/), so it is a record, not live tooling.
# If you ever import it programmatically, wrap the body first.

import glob, hashlib, json
from pathlib import Path
import numpy as np
import polars as pl

D = Path("eval/scripts/verification_grading/data")
def cid(u): return hashlib.md5((u or "").encode()).hexdigest()[:16]

def load(vf):
    claims = pl.read_parquet(D / "claims_dev_content.parquet")
    verd = pl.read_parquet(D / vf)
    sf = pl.read_parquet("eval/data/verdict_scope_floor.parquet").with_columns(
        pl.col("review_url").map_elements(cid, return_dtype=pl.Utf8).alias("claim_id")).select(
        "claim_id", "stage3_in_scope")
    df = claims.join(verd, on="claim_id", how="inner").join(sf, on="claim_id", how="left")
    df = df.filter(pl.col("error").is_null()).with_columns(pl.col("stage3_in_scope").fill_null(True))
    return df.filter((pl.col("judged_axis") == "content") & pl.col("stage3_in_scope"))

def escalation(tag):
    n = exa = 0
    for fp in glob.glob(str(D / f"verdicts_dev_content_{tag}_trace/*.json")):
        t = json.load(open(fp)); n += 1
        if "exa" in (t.get("providers_used") or []): exa += 1
    return exa, n

def stats(df):
    g = df["gold_veracity"].to_numpy(); p = df["veracity"].to_numpy()
    def nb(v): return "flag" if v <= 2 else ("soft" if v == 3 else "pass")
    nudge = np.mean([nb(a) == nb(b) for a, b in zip(g, p)])
    binacc = np.mean([(a <= 3) == (b <= 3) for a, b in zip(g, p)])  # "nudge unless confident-true"
    return dict(n=len(g), mae=np.abs(p - g).mean(), off1=(np.abs(p - g) <= 1).mean(),
                nudge3=nudge, binflag=binacc,
                rounds=df["rounds_used"].mean() if "rounds_used" in df.columns else float("nan"))

for label, vf, tag in [("serper2/exa2 (baseline)", "verdicts_dev_content_cascade.parquet", "cascade"),
                       ("serper3/exa2 (new)", "verdicts_dev_content_32.parquet", "32")]:
    if not (D / vf).exists():
        print(f"{label}: MISSING ({vf})"); continue
    s = stats(load(vf)); exa, n = escalation(tag)
    print(f"\n{label}  (n={s['n']})")
    print(f"  exa escalation: {exa}/{n} = {exa/max(n,1):.0%}   avg rounds={s['rounds']:.2f}")
    print(f"  MAE={s['mae']:.3f}  off-by-1={s['off1']:.2f}  nudge3={s['nudge3']:.3f}  "
          f"binary-flag(<=3)={s['binflag']:.3f}")
