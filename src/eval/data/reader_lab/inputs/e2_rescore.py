import json, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.cwd()))
from eval.scripts.build_eval import fit_urn, graded_urn

SCR = Path("/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/3a18c030-7dc2-46a0-9eaa-15e9ef2d42f6/scratchpad")
pop = fit_urn.load_population(Path("eval/data/populations/fc_gold.parquet"))
rows = fit_urn.load_headline(Path("eval/data/urn_runs/e1_ctx/results-00.jsonl"), None, population=pop)
w = graded_urn.fit_graded(rows)
PAD = 10
def rating(s):  # project scale
    if s < -4.079: return 1
    if s < -2.075: return 2
    if s < 0.034: return 3
    if s < 5.560: return 4
    return 5

recs = []
excl = Counter()
for line in Path("eval/data/urn_runs/e2_tweets/results-00.jsonl").open():
    r = json.loads(line)
    if r.get("excluded"):
        excl[r["excluded"]] += 1; continue
    flags = [d["read"]["direction"] for d in r["results"] if d.get("read") and d["read"].get("direction")]
    flags = flags[:PAD] + ["I"] * max(0, PAD - len(flags))
    sc = sum(w.get(f, 0.0) for f in flags)
    recs.append({"claim_id": r["review_url"], "post_id": r["post_id"], "claim": r["claim_resolved"] or r["claim_text"],
                 "domain": r["publisher_site"], "ng_score": r["ng_score"], "bin": r["bin"], "lean": r["lean"],
                 "topic": r["topic"], "date": r["claim_date_shown"], "score": round(sc, 2), "rating": rating(sc),
                 "flags": ",".join(flags), "n_read": sum(1 for f in flags if f != "I")})
print("excluded", dict(excl))
df = pd.DataFrame(recs)
print("scored", len(df), "posts", df.post_id.nunique())
df["tier"] = np.where(df.ng_score < 30, "low", np.where(df.ng_score < 70, "mid", "high"))
print("\nclaims by tier x lean")
print(pd.crosstab(df.tier, df.lean, margins=True))
print("\nrating by tier (row %)")
print((pd.crosstab(df.tier, df.rating, normalize="index") * 100).round(1))
print("\nrating counts, tier x lean, ratings 1 and 2")
sub = df[df.rating <= 2]
print(pd.crosstab([sub.tier, sub.lean], sub.rating, margins=True))
print("\nrating 5 counts, tier x lean")
sub5 = df[df.rating == 5]
print(pd.crosstab(sub5.tier, sub5.lean, margins=True))
print("\nper outlet, low+mid: n, r1, r2, r5, mean score")
g = df[df.tier != "high"].groupby(["domain", "lean", "ng_score"])
t = g.agg(n=("score", "size"), r1=("rating", lambda x: (x == 1).sum()), r2=("rating", lambda x: (x == 2).sum()),
          r5=("rating", lambda x: (x == 5).sum()), mean=("score", "mean")).sort_values("ng_score")
print(t.to_string())
df.to_parquet(SCR / "e2_rescored.parquet")
print("\nlowest 12 in low+mid")
for _, r in df[df.tier != "high"].nsmallest(12, "score").iterrows():
    print(f"{r.score:+7.2f} {r.domain:22s} {r.lean:6s} {r.date} | {r.claim[:110]}")
print("\nhighest 8 in low+mid")
for _, r in df[df.tier != "high"].nlargest(8, "score").iterrows():
    print(f"{r.score:+7.2f} {r.domain:22s} {r.lean:6s} {r.date} | {r.claim[:110]}")
