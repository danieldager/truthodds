"""Controlled headline-only vs full-article ACE test.

For a sample of articles we already ran full-body ACE on (ace_attempts.parquet has the headline +
the body's claim count), re-run ACE on the HEADLINE ALONE and compare hit rates. Answers a collaborator's
headlines-vs-article question with a real number.
"""
import sys, random
import pandas as pd

sys.path.insert(0, "src")
from eval import ace

ATTEMPTS = "src/eval/data/ace_attempts.parquet"
PER_SRC = int(sys.argv[1]) if len(sys.argv) > 1 else 6
ONLY = set(sys.argv[2].split("|")) if len(sys.argv) > 2 else None

a = pd.read_parquet(ATTEMPTS)
# articles whose body was actually processed (so a body result exists) + a real headline
a = a[a["status"].isin(["claims", "none"]) & a["headline"].astype(str).str.len().gt(15)].copy()
# deterministic spread: first PER_SRC per source
sample = a.sort_values("url").groupby("source", group_keys=False).head(PER_SRC)
if ONLY:
    sample = sample[sample["source"].isin(ONLY)]
print(f"testing {len(sample)} articles across {sample.source.nunique()} outlets", flush=True)

rows = []
for i, (_, r) in enumerate(sample.iterrows(), 1):
    hl = str(r["headline"])
    try:
        res = ace.extract(hl, headline=None)   # HEADLINE ONLY as the content
        hn = res["n"]
        hclaims = [c.get("claim") for c in res["claims"]]
    except Exception as e:
        hn, hclaims = 0, [f"ERR {e}"]
    rows.append({"source": r["source"], "headline": hl, "body_n": int(r["n_claims"]),
                 "headline_n": hn, "headline_claims": " | ".join(hclaims)})
    if i % 20 == 0:
        print(f"  {i}/{len(sample)}", flush=True)

d = pd.DataFrame(rows)
d["body_hit"] = d["body_n"] > 0
d["headline_hit"] = d["headline_n"] > 0
n = len(d)
print("\n=== HEADLINE-ONLY vs FULL-ARTICLE (same articles) ===", flush=True)
print(f"n = {n} articles", flush=True)
print(f"headline-only hit rate: {d.headline_hit.mean():.0%}  ({d.headline_hit.sum()}/{n})", flush=True)
print(f"full-article hit rate:  {d.body_hit.mean():.0%}  ({d.body_hit.sum()}/{n})", flush=True)
print(f"body found a claim, headline did NOT: {int((d.body_hit & ~d.headline_hit).sum())}", flush=True)
print(f"headline found a claim, body did NOT: {int((~d.body_hit & d.headline_hit).sum())}", flush=True)
d.to_csv("src/eval/data/survey_claims/headline_vs_article_test.csv", index=False)
print("\n-- sample headline hits (what ACE pulls from a headline alone) --", flush=True)
for _, r in d[d.headline_hit].head(6).iterrows():
    print(f"  [{r.source}] HL: {r.headline[:70]}", flush=True)
    print(f"      -> {str(r.headline_claims)[:90]}", flush=True)
print("\n-- sample: body-hit but headline-miss (claim was in the body) --", flush=True)
for _, r in d[d.body_hit & ~d.headline_hit].head(5).iterrows():
    print(f"  [{r.source}] HL: {r.headline[:80]} (body found {r.body_n})", flush=True)
