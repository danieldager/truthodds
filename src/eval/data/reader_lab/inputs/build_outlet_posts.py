import json, sys, math, re
from pathlib import Path
import numpy as np, pandas as pd

SCR = Path("/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/3a18c030-7dc2-46a0-9eaa-15e9ef2d42f6/scratchpad")
sc = pd.read_parquet(SCR / "e2_rescored.parquet")          # from e2_rescore.py, project numbering
vi = pd.read_parquet("eval/data/survey_claims/e2_verify_input_v2.parquet")
recs = {}
for line in Path("eval/data/urn_runs/e2_tweets/results-00.jsonl").open():
    r = json.loads(line)
    if not r.get("excluded"): recs[r["review_url"]] = r

BP = re.compile(r"skip (directly )?to|cookie|enable javascript|official website|here's how you know|sign in|subscribe|log in|privacy policy", re.I)
def snippet(d):
    ev = (d.get("read") or {}).get("evidence") or []
    sents = d.get("sents") or []
    ids = [i for i in ev if isinstance(i, int) and 1 <= i <= len(sents)]
    picked = [sents[i-1] for i in ids] if ids else []
    picked = [x for x in picked if not BP.search(x)]
    if not picked:
        picked = [x for x in sents if len(x.strip()) > 50 and not BP.search(x)][:2]
    txt = " ".join(picked)
    txt = re.sub(r"^by [A-Z][\w.]* [A-Z][\w-]+( · \d{4})?( · Cited by \d+)? [—-] ", "", txt)
    return txt.replace("\xa0", " ").replace("Â ", " ")[:200]

def clean(v):
    if v is None: return None
    if isinstance(v, float) and math.isnan(v): return None
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.ndarray): return list(dict.fromkeys(str(x) for x in v.tolist()))
    if isinstance(v, np.bool_): return bool(v)
    return v

roster = pd.read_csv("eval/data/survey_claims/source_roster.csv").set_index("domain")["lean"].to_dict()
def lean_of(r):
    v = r["cell"]
    return v if v in ("Left", "Right", "Center") else roster.get(r["domain"], v)
score = sc.set_index("claim_id")[["score", "rating", "flags"]].to_dict("index")
out = []
for pid, g in vi.groupby("post_id"):
    claims = []
    for _, r in g.iterrows():
        m = score.get(r["claim_id"])
        claims.append({"claim_id": r["claim_id"], "claim": r["claim"], "checkworthy": bool(r["checkworthy"]),
                       "score": m["score"] if m else None, "rating": int(m["rating"]) if m else None,
                       "flags": m["flags"] if m else None})
    scored = [c for c in claims if c["score"] is not None]
    if not scored: continue
    r0 = g.iloc[0]
    if float(r0["ng_score"]) >= 70: continue
    low = min(scored, key=lambda c: c["score"])
    high = max(scored, key=lambda c: c["score"])
    pr = low["rating"]
    if pr in (1, 2): key, want = low, ("1", "2")
    elif pr == 5: key, want = high, ("5", "4")
    else: continue
    ev = []
    for d in recs[key["claim_id"]]["results"]:
        if (d.get("read") or {}).get("direction") in want:
            ev.append({"domain": d.get("domain"), "url": d.get("url"), "date": d.get("date"),
                       "direction": d["read"]["direction"], "snippet": snippet(d)})
        if len(ev) == 3: break
    key["evidence"] = ev
    out.append({"post_id": str(pid), "url": clean(r0["url"]), "handle": clean(r0["handle"]), "domain": clean(r0["domain"]),
        "ng_score": clean(r0["ng_score"]), "lean": lean_of(r0), "bin": clean(r0["bin"]),
        "followers": None, "created_at": clean(r0["created_at"]), "post_text": clean(r0["post_text"]),
        "is_quote": bool(r0["is_quote"]), "quoted_handle": None, "quoted_text": None,
        "image_urls": clean(r0["image_urls"]) or [], "n_images": clean(r0["n_images"]), "n_videos": clean(r0["n_videos"]),
        "like_count": clean(r0["like_count"]), "retweet_count": clean(r0["retweet_count"]),
        "reply_count": clean(r0["reply_count"]), "view_count": clean(r0["view_count"]), "topic": clean(r0["topic"]),
        "post_rating": pr, "post_score": key["score"], "key_claim_id": key["claim_id"], "claims": claims})

lowmid = sc[sc.ng_score < 70]
cnt = {str(k): int((lowmid.rating == k).sum()) for k in (1, 2, 3, 4, 5)}
n = {str(k): sum(1 for p in out if p["post_rating"] == k) for k in (1, 2, 5)}
doc = {"numbering": "project", "n_posts": n, "n_claims": cnt, "n_claims_scored": int(len(lowmid)),
       "n_posts_scored": int(lowmid.post_id.nunique()),
       "rating_share_pct": {k: round(v / len(lowmid) * 100, 1) for k, v in cnt.items()},
       "posts": out}
(SCR / "outlet_posts.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
print("posts", n, "claims", cnt, "scored", len(lowmid), "posts scored", doc["n_posts_scored"])
