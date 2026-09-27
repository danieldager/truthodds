import os, json
import polars as pl
SCR=os.environ["SCR"]
OUT="eval/data/community_notes/survey_posts_sep_2026-09-21.parquet"
tbl=pl.read_parquet(f"{SCR}/posts_sep_table.parquet")
hyd={}
for line in open(f"{SCR}/hydrated_sep.jsonl"):
    r=json.loads(line); hyd[r["post_id"]]=r
# lean map (15 Sep flash pass) author -> (lean, reason)
cand=pl.read_parquet("eval/data/community_notes/survey_candidates_2026-09-15.parquet")
lean={r["author"]:(r["lean"],r["lean_reason"]) for r in cand.select(["author","lean","lean_reason"]).unique(subset="author").iter_rows(named=True)}
# author meta: description+followers. eligible by handle, author_lean_bio by author.
elig=pl.read_parquet("eval/data/community_notes/eligible_posts.parquet")
alb=pl.read_parquet("eval/data/community_notes/author_lean_bio_2026-09-16.parquet")
desc={}; folw={}
for r in alb.iter_rows(named=True):
    if r.get("bio"): desc[r["author"]]=r["bio"]
    if r.get("followers") is not None: folw[r["author"]]=r["followers"]
for r in elig.iter_rows(named=True):  # eligible fuller -> overwrite
    if r.get("author_description"): desc[r["handle"]]=r["author_description"]
    if r.get("author_followers") is not None: folw[r["handle"]]=r["author_followers"]
impr={r["post_id"]:r["view_count"] for r in elig.iter_rows(named=True) if r.get("view_count") is not None}

rows=[]
for t in tbl.iter_rows(named=True):
    pid=t["post_id"]; h=hyd.get(pid,{})
    handle=h.get("handle")
    ln=lean.get(handle,(None,None)) if handle else (None,None)
    rows.append({
        "post_id":pid,
        "created_at":t["created_at"],
        "snapshot_day":"2026-09-21",
        "alive":bool(h.get("ok")),
        "dead_reason":h.get("dead_reason"),
        "hydrate_source":h.get("source"),
        "handle":handle,
        "author_name":h.get("name"),
        "author_description":desc.get(handle) if handle else None,
        "followers":folw.get(handle) if handle else None,
        "impressions":impr.get(pid),
        "likes":h.get("likes"),
        "replies":h.get("replies"),
        "retweets":None,
        "verified":h.get("verified"),
        "lang":h.get("lang"),
        "text":h.get("text"),
        "has_media":h.get("has_media"),
        "media_types":h.get("media_types"),
        "is_quote":h.get("is_quote"),
        "is_longform":h.get("is_longform"),
        "possibly_sensitive":h.get("possibly_sensitive"),
        "lean":ln[0],"lean_reason":ln[1],
        "n_notes":t["n_notes"],
        "note_classifications":t["note_classifications"],
        "note_statuses":t["note_statuses"],
        "n_helpful":t["n_helpful"],"n_nmr":t["n_nmr"],
        "n_not_helpful":t["n_not_helpful"],"n_missing":t["n_missing"],
        "n_misleading":t["n_misleading"],"n_not_misleading":t["n_not_misleading"],
        "best_note_summary":t["best_note_summary"],
        "best_note_classification":t["best_note_classification"],
        "best_note_status":t["best_note_status"],
        "quasi_label":t["quasi_label"],
    })
df=pl.DataFrame(rows, infer_schema_length=None)
df.write_parquet(OUT)
print("wrote",OUT,df.shape,flush=True)
# ---- report funnel ----
alive=df.filter(pl.col("alive"))
print("posts with notes:",df.height,flush=True)
print("alive:",alive.height,f"({alive.height/df.height:.1%})",flush=True)
eng=alive.filter(pl.col("lang")=="en")
print("alive & english:",eng.height,flush=True)
import collections
print("note-status presence (posts):", {
 "has_HELPFUL":(df['n_helpful']>0).sum(),"has_NMR":(df['n_nmr']>0).sum(),
 "has_NOT_HELPFUL":(df['n_not_helpful']>0).sum(),"has_MISSING":(df['n_missing']>0).sum()},flush=True)
print("quasi_label:",dict(collections.Counter(df['quasi_label'].to_list())),flush=True)
print("lang top (alive):",dict(collections.Counter(alive['lang'].to_list()).most_common(8)),flush=True)
# lean coverage
alive_h=alive.filter(pl.col("handle").is_not_null())
au=set(alive_h["handle"].to_list())
lab=set(k for k in au if k in {a for a in lean})
print("distinct alive authors:",len(au),"| already have lean:",len(lab),"| need lean:",len(au-lab),flush=True)
# impressions & followers distribution
def q(s,name):
    s=[x for x in s if x is not None]
    if not s: print(f"  {name}: none"); return
    ss=sorted(s); import math
    p50=ss[len(ss)//2]; p90=ss[min(len(ss)-1,int(len(ss)*0.9))]
    print(f"  {name}: n={len(s)} p50={p50} p90={p90}",flush=True)
print("distributions (alive):",flush=True)
q(alive["impressions"].to_list(),"impressions")
q(alive["followers"].to_list(),"followers")
q(alive["likes"].to_list(),"likes")
print("ASSEMBLE_DONE",flush=True)
