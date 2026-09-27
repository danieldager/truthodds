"""Shape a 50-per-cell sample of the new shortlist harvest into the verify_text input schema
(claim_id, raw_context, claim_date, side, reliability, ...) for verify_survey_run.py. Writes to a
SEPARATE input/output so the other session's claim_pool_verdicts.parquet is untouched."""
import pandas as pd, hashlib
BASE = "src/eval/data/survey_claims"
ATTEMPTS = "src/eval/data/ace_attempts.parquet"
SIDE = {"Left": "L", "Right": "R"}
REL = {"Reliable": "R", "Unreliable": "U"}

d = pd.read_parquet(f"{BASE}/shortlist_claims.parquet")
at = pd.read_parquet(ATTEMPTS)[["url", "attempted_at"]].drop_duplicates("url")
hdate = dict(zip(at.url, at.attempted_at.astype(str).str[:10]))

samp = pd.concat([g.sample(min(50, len(g)), random_state=0) for _, g in d.groupby("cell")])
rows = []
for _, r in samp.iterrows():
    claim = str(r["claim"]).strip()
    rows.append({
        "claim_id": hashlib.md5(claim.encode("utf-8")).hexdigest()[:12],
        "raw_context": claim,
        "claim_date": hdate.get(r["url"], "2026-07-06"),
        "side": SIDE.get(r["orientation"], "C"),
        "reliability": REL.get(r["reliability"], "U"),
        "outlet": r["outlet"], "domain": r["domain"], "cell": r["cell"],
        "source_url": r["url"], "headline": r.get("headline", ""),
        "batch": "shortlist", "ng_score": r["ng_score"],
    })
out = pd.DataFrame(rows).drop_duplicates("claim_id").reset_index(drop=True)
out.to_parquet(f"{BASE}/shortlist_verify_input.parquet", index=False)
print("wrote", len(out), "->", out.groupby(["reliability", "side"]).size().to_dict())
