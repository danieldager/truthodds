import zipfile, io, csv, json, datetime, collections
from pathlib import Path
import polars as pl

RAW=Path("eval/data/community_notes/raw")
NOTES=sorted(RAW.glob("20260921-notes-*.zip"))
NSH=sorted(RAW.glob("20260921-noteStatusHistory-*.zip"))
TW_EPOCH=1288834974657
SEP1_MS=int(datetime.datetime(2026,9,1,tzinfo=datetime.timezone.utc).timestamp()*1000)

def rows(path, fields):
    with zipfile.ZipFile(path) as z:
        name=z.namelist()[0]
        with z.open(name) as f:
            rdr=csv.DictReader(io.TextIOWrapper(f,encoding="utf-8",errors="replace"),delimiter="\t")
            for r in rdr:
                yield {k:r.get(k) for k in fields}

# 1. status map
print("parsing NSH...",flush=True)
status={}
nsh_n=0
for p in NSH:
    for r in rows(p,["noteId","currentStatus","lockedStatus"]):
        nsh_n+=1
        status[r["noteId"]]=r["currentStatus"] or "MISSING"
print(f"  NSH rows {nsh_n}, distinct noteIds {len(status)}",flush=True)

STMAP={"CURRENTLY_RATED_HELPFUL":"HELPFUL","NEEDS_MORE_RATINGS":"NMR",
       "CURRENTLY_RATED_NOT_HELPFUL":"NOT_HELPFUL"}
def st(nid): return STMAP.get(status.get(nid,"MISSING"),"MISSING") if nid in status else "MISSING"

# 2. notes -> keep only notes on Sep1+ posts
print("parsing notes shards...",flush=True)
notes_n=0; sep_notes=0
byid=collections.defaultdict(list)  # tweetId -> list of dicts
for p in NOTES:
    for r in rows(p,["noteId","tweetId","classification","summary","createdAtMillis"]):
        notes_n+=1
        tid=r["tweetId"]
        try: post_ms=(int(tid)>>22)+TW_EPOCH
        except (ValueError,TypeError): continue
        if post_ms<SEP1_MS: continue
        sep_notes+=1
        byid[tid].append({
            "noteId":r["noteId"],
            "cls":"MISLEADING" if r["classification"]=="MISINFORMED_OR_POTENTIALLY_MISLEADING"
                  else ("NOT_MISLEADING" if r["classification"]=="NOT_MISLEADING" else (r["classification"] or "?")),
            "status":st(r["noteId"]),
            "summary":r["summary"] or "",
            "post_ms":post_ms,
        })
print(f"  notes total {notes_n}; notes on Sep1+ posts {sep_notes}; distinct Sep1+ posts {len(byid)}",flush=True)

SRANK={"HELPFUL":3,"NMR":2,"NOT_HELPFUL":1,"MISSING":0}
recs=[]
for tid,ns in byid.items():
    post_ms=ns[0]["post_ms"]
    created=datetime.datetime.fromtimestamp(post_ms/1000,tz=datetime.timezone.utc)
    cls=[n["cls"] for n in ns]; sts=[n["status"] for n in ns]
    sc=collections.Counter(sts)
    has_help_mis=any(n["cls"]=="MISLEADING" and n["status"]=="HELPFUL" for n in ns)
    has_notmis_ok=any(n["cls"]=="NOT_MISLEADING" and n["status"]!="NOT_HELPFUL" for n in ns)
    has_mis_nothelp=any(n["cls"]=="MISLEADING" and n["status"]=="NOT_HELPFUL" for n in ns)
    if has_help_mis: quasi="false"
    elif has_notmis_ok or has_mis_nothelp: quasi="true"
    else: quasi=None
    best=max(ns,key=lambda n:(SRANK[n["status"]], len(n["summary"])))
    recs.append({
        "post_id":tid,
        "created_at":created,
        "n_notes":len(ns),
        "note_classifications":cls,
        "note_statuses":sts,
        "n_helpful":sc["HELPFUL"],"n_nmr":sc["NMR"],"n_not_helpful":sc["NOT_HELPFUL"],"n_missing":sc["MISSING"],
        "n_misleading":sum(1 for c in cls if c=="MISLEADING"),
        "n_not_misleading":sum(1 for c in cls if c=="NOT_MISLEADING"),
        "quasi_label":quasi,
        "best_note_summary":best["summary"],
        "best_note_classification":best["cls"],
        "best_note_status":best["status"],
    })
df=pl.DataFrame(recs)
df.write_parquet(f"{__import__('os').environ['SCR']}/posts_sep_table.parquet")
print("\n== SEP POST TABLE ==",flush=True)
print("distinct Sep1+ posts:",df.height,flush=True)
print("date range:",df["created_at"].min(),"->",df["created_at"].max(),flush=True)
print("quasi_label:",dict(collections.Counter(df["quasi_label"].to_list())),flush=True)
print("post has >=1 HELPFUL note:",(df["n_helpful"]>0).sum(),flush=True)
print("post has >=1 NMR note:",(df["n_nmr"]>0).sum(),flush=True)
print("post has >=1 NOT_HELPFUL note:",(df["n_not_helpful"]>0).sum(),flush=True)
print("post has >=1 MISSING note:",(df["n_missing"]>0).sum(),flush=True)
print("post has >=1 MISLEADING note:",(df["n_misleading"]>0).sum(),flush=True)
print("post has >=1 NOT_MISLEADING note:",(df["n_not_misleading"]>0).sum(),flush=True)
print("TABLE_DONE",flush=True)
