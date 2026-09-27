import json, sys, math, re
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.cwd()))
from eval.scripts.build_eval import fit_urn, graded_urn

SCR = Path("/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/3a18c030-7dc2-46a0-9eaa-15e9ef2d42f6/scratchpad")
pop = fit_urn.load_population(Path("eval/data/populations/fc_gold.parquet"))
rows = fit_urn.load_headline(Path("eval/data/urn_runs/e1_ctx/results-00.jsonl"), None, population=pop)
w = graded_urn.fit_graded(rows)
print("weights", {k: round(v,4) for k,v in w.items()})
PAD=10
def rating(s):
    if s < -4.079: return 5
    if s < -2.075: return 4
    if s < 0.034: return 3
    if s < 5.560: return 2
    return 1

recs = {}   # claim_id -> dict(score, rating, flagstr, rec)
allscores=[]
for line in Path("eval/data/urn_runs/true_timeline/scores.jsonl").open():
    r = json.loads(line)
    flags=[d["read"]["direction"] for d in r["results"] if d.get("read") and d["read"].get("direction")]
    flags = flags[:PAD] + ["I"]*max(0, PAD-len(flags))
    sc = sum(w.get(f,0.0) for f in flags)
    allscores.append(sc)
    recs[r["review_url"]] = {"score": sc, "rating": rating(sc), "flags": ",".join(flags), "rec": r}
s=np.array(allscores)
dist = {str(k): round(float((np.array([rating(x) for x in s])==k).mean()*100),1) for k in [5,4,3,2,1]}
print("n scored", len(s), "dist", dist)

df = pd.read_parquet("eval/data/tweet_corpus/timeline_urn.parquet")
matched = df["claim_id"].isin(recs.keys()).sum()
print("parquet claims", len(df), "claim_id matched to scores", matched, "score recs", len(recs),
      "unmatched score keys", len(set(recs) - set(df["claim_id"])))

# local images index
imgidx = defaultdict(list)
for p in Path("eval/data/images").rglob("*.webp"):
    imgidx[p.stem.split("_")[0]].append(str(p.resolve()))

BP = re.compile(r"skip (directly )?to|cookie|enable javascript|official website|here's how you know|sign in|subscribe|log in|privacy policy", re.I)
def snippet(d):
    ev = (d.get("read") or {}).get("evidence") or []
    sents = d.get("sents") or []
    ids = [i for i in ev if isinstance(i,int) and 1 <= i <= len(sents)]
    picked = [sents[i-1] for i in ids] if ids else []
    picked = [x for x in picked if not BP.search(x)]
    if not picked:
        picked = [x for x in sents if len(x.strip()) > 50 and not BP.search(x)][:2]
    txt = " ".join(picked)
    txt = re.sub(r"^by [A-Z][\w.]* [A-Z][\w-]+( · \d{4})?( · Cited by \d+)? [—-] ", "", txt)
    txt = txt.replace("\xa0", " ").replace("Â ", " ")
    return txt[:200]

def clean(v):
    if v is None: return None
    if isinstance(v, float) and math.isnan(v): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, np.ndarray): return [str(x) for x in v.tolist()]
    if isinstance(v, (np.bool_,)): return bool(v)
    return v

out=[]
img_hits=0
for pid, g in df.groupby("post_id"):
    claims=[]
    for _, r in g.iterrows():
        m = recs.get(r["claim_id"])
        claims.append({"claim_id": r["claim_id"], "claim": r["claim"],
                       "checkworthy": bool(r["checkworthy"]),
                       "score": round(m["score"],2) if m else None,
                       "rating": m["rating"] if m else None,
                       "flags": m["flags"] if m else None})
    scored=[c for c in claims if c["score"] is not None]
    if not scored: continue
    low = min(scored, key=lambda c: c["score"])
    pr = low["rating"]
    if pr not in (5,4): continue
    # evidence for lowest-scoring claim
    ev=[]
    for d in recs[low["claim_id"]]["rec"]["results"]:
        if (d.get("read") or {}).get("direction") in ("1","2"):
            ev.append({"domain": d.get("domain"), "url": d.get("url"),
                       "date": d.get("date"), "direction": d["read"]["direction"],
                       "snippet": snippet(d)})
        if len(ev)==3: break
    low["evidence"]=ev
    r0 = g.iloc[0]
    li = imgidx.get(str(pid), [])
    if li: img_hits+=1
    out.append({"post_id": str(pid), "url": clean(r0["url"]), "handle": clean(r0["handle"]),
        "followers": clean(r0["followers"]), "created_at": clean(r0["created_at"]),
        "post_text": clean(r0["post_text"]), "is_quote": bool(r0["is_quote"]),
        "quoted_handle": clean(r0["quoted_handle"]), "quoted_text": clean(r0["quoted_text"]),
        "image_urls": clean(r0["image_urls"]) or [], "n_images": clean(r0["n_images"]),
        "n_videos": clean(r0["n_videos"]), "like_count": clean(r0["like_count"]),
        "retweet_count": clean(r0["retweet_count"]), "reply_count": clean(r0["reply_count"]),
        "view_count": clean(r0["view_count"]), "topic": clean(r0["topic"]), "cell": clean(r0["cell"]),
        "post_rating": pr, "post_score": round(low["score"],2),
        "local_images": li, "claims": claims})

out.sort(key=lambda p: (-p["post_rating"], p["post_score"]))
n5=sum(1 for p in out if p["post_rating"]==5); n4=len(out)-n5
c5=sum(1 for m in recs.values() if m["rating"]==5); c4=sum(1 for m in recs.values() if m["rating"]==4)
doc={"n_posts_r5": n5, "n_posts_r4": n4, "n_claims_r5": c5, "n_claims_r4": c4,
     "n_claims_scored": len(recs), "rating_share_pct": dist,
     "bands": {"5":"<-4.079","4":"[-4.079,-2.075)","3":"[-2.075,0.034)","2":"[0.034,5.560)","1":">=5.560"},
     "weights": {k: round(v,4) for k,v in w.items()},
     "join": "scores.jsonl review_url == parquet claim_id",
     "posts": out}
p = SCR/"posts_r5_r4.json"
p.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
print("wrote", p, "posts", len(out), "r5", n5, "r4", n4, "claims r5", c5, "r4", c4)
print("local image hits", img_hits, "/", len(out))
for rt in (5,4):
    print(f"--- rating {rt} examples ---")
    for p_ in [x for x in out if x["post_rating"]==rt][:3]:
        print(" ", p_["handle"], "|", (p_["post_text"] or "").replace("\n"," ")[:80])
