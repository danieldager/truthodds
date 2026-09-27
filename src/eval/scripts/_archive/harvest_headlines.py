"""Harvest HEADLINE claims from every article we've already body-processed (no re-fetch).

We have ~300 articles in ace_attempts with their headline stored. Headlines yield partly-different
claims than bodies (per the headline test), so running ACE on each headline grows the pool cheaply.
Writes claims_headline.csv (extraction_source=headline). Dedup vs bodies happens later in the filter.
"""
import sys, json
import pandas as pd

sys.path.insert(0, "src")
from eval import ace

REPO = "src"
OUTDIR = REPO + "/eval/data/survey_claims"
ATTEMPTS = REPO + "/eval/data/ace_attempts.parquet"

bs = pd.read_excel(OUTDIR + "/source_accessibility.xlsx", sheet_name="by_source")
cellmap = dict(zip(bs["outlet"], bs["cell"]))
a = pd.read_parquet(ATTEMPTS)
a = a[a["status"].isin(["claims", "none"]) & a["headline"].astype(str).str.len().gt(15)].drop_duplicates("url")
print(f"headline-harvesting {len(a)} already-fetched articles", flush=True)

claims = []
for i, (_, r) in enumerate(a.iterrows(), 1):
    hl = str(r["headline"])
    try:
        res = ace.extract(hl, headline=None)
        cl = res.get("claims", [])
    except Exception:
        cl = []
    for c in cl:
        claims.append({"outlet": r["source"], "cell": cellmap.get(r["source"], "?"), "url": r["url"],
                       "headline": hl, "claim": c.get("claim"), "risk_reason": c.get("risk_reason"),
                       "quote": c.get("quote"), "extraction_source": "headline"})
    if i % 40 == 0:
        print(f"  {i}/{len(a)} ({len(claims)} claims)", flush=True)

cl = pd.DataFrame(claims)
cl.to_csv(OUTDIR + "/claims_headline.csv", index=False)
print(f"\nwrote {len(cl)} headline claims -> {OUTDIR}/claims_headline.csv", flush=True)
if len(cl):
    print("by cell:", dict(cl.groupby("cell").size()), flush=True)
