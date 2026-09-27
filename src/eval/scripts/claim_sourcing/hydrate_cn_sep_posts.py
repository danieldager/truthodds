import os, sys, math, re, json, time, threading, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars as pl
import requests

SCR=os.environ["SCR"]
TBL=f"{SCR}/posts_sep_table.parquet"
CACHE=f"{SCR}/hydrated_sep.jsonl"
UA={"User-Agent":"Mozilla/5.0"}
DIGITS="0123456789abcdefghijklmnopqrstuvwxyz"

def x_token(tid):
    n=(int(tid)/1e15)*math.pi; i,f=int(n),n-int(n)
    s="" if i else "0"
    while i: s,i=DIGITS[i%36]+s,i//36
    frac,c="",0
    while f>0 and c<24:
        f*=36;k=int(f);frac+=DIGITS[k];f-=k;c+=1
    return re.sub(r"(0+|\.)","",s+("."+frac if frac else ""))

lock=threading.Lock()
stats={"ok":0,"dead":0,"err":0,"429":0,"done":0}

def media_of(j):
    md=j.get("mediaDetails")
    if md: return [m.get("type") for m in md]
    ent=(j.get("entities") or {}).get("media") or []
    return [m.get("type") for m in ent] if ent else []

def fetch(tid, max_try=5):
    for att in range(max_try):
        try:
            r=requests.get("https://cdn.syndication.twimg.com/tweet-result",
                params={"id":tid,"token":x_token(tid),"lang":"en"},headers=UA,timeout=15)
        except requests.RequestException:
            time.sleep(min(2**att,16)+random.random()); continue
        if r.status_code==429:
            with lock: stats["429"]+=1
            time.sleep(min(2**att,16)+random.random()); continue
        if r.status_code in (404,):
            return {"post_id":tid,"ok":False,"dead_reason":"notfound","source":"syndication"}
        if r.status_code in (403,):
            return {"post_id":tid,"ok":False,"dead_reason":"forbidden","source":"syndication"}
        if r.status_code>=500:
            time.sleep(min(2**att,16)+random.random()); continue
        if r.status_code==200 and "json" in r.headers.get("content-type",""):
            try: j=r.json()
            except ValueError: return {"post_id":tid,"ok":False,"dead_reason":"badjson","source":"syndication"}
            if j.get("tombstone") or not (j.get("text") or (j.get("note_tweet") or {})):
                return {"post_id":tid,"ok":False,"dead_reason":"tombstone","source":"syndication"}
            nt=(j.get("note_tweet") or {}).get("text")
            u=j.get("user") or {}
            return {"post_id":tid,"ok":True,"source":"syndication",
                "text":nt or j.get("text") or "",
                "lang":j.get("lang"),"created_at":j.get("created_at"),
                "likes":j.get("favorite_count"),"replies":j.get("conversation_count"),
                "possibly_sensitive":j.get("possibly_sensitive"),
                "verified":bool(j.get("user",{}).get("verified") or j.get("user",{}).get("is_blue_verified")),
                "handle":u.get("screen_name"),"name":u.get("name"),
                "media_types":media_of(j),"has_media":bool(media_of(j)),
                "is_quote":bool(j.get("quoted_tweet")),"is_longform":bool(nt)}
        # unexpected 200 non-json etc.
        return {"post_id":tid,"ok":False,"dead_reason":f"http{r.status_code}","source":"syndication"}
    return {"post_id":tid,"ok":False,"dead_reason":"maxretry","source":"syndication"}

def main():
    tbl=pl.read_parquet(TBL)
    ids=tbl["post_id"].to_list()
    # reuse maps
    cand=pl.read_parquet("eval/data/community_notes/survey_candidates_2026-09-15.parquet")
    elig=pl.read_parquet("eval/data/community_notes/eligible_posts.parquet")
    reuse={}
    for r in cand.iter_rows(named=True):
        reuse[r["id"]]={"post_id":r["id"],"ok":True,"source":"candidates",
            "text":r["text"],"lang":"en","created_at":None,
            "likes":r["likes"],"replies":r["replies"],"possibly_sensitive":r["possibly_sensitive"],
            "verified":None,"handle":r["author"],"name":r["author_name"],
            "media_types":([r["media"]] if r["media"] else []),"has_media":bool(r["media"]),
            "is_quote":r["is_quote"],"is_longform":None}
    for r in elig.iter_rows(named=True):  # eligible fuller; only fill if not already from candidates
        if r["post_id"] in reuse: continue
        reuse[r["post_id"]]={"post_id":r["post_id"],"ok":True,"source":"eligible",
            "text":r["text"],"lang":r["lang"],"created_at":r["created_at"],
            "likes":r["like_count"],"replies":r["reply_count"],"possibly_sensitive":r["possibly_sensitive"],
            "verified":r["author_verified"],"handle":r["handle"],"name":r["author_name"],
            "media_types":list(r["media_types"] or []),"has_media":r["has_media"],
            "is_quote":r["is_quote"],"is_longform":r["is_longform"]}
    # resume: what's already cached
    done_ids=set()
    if os.path.exists(CACHE):
        for line in open(CACHE):
            try: done_ids.add(json.loads(line)["post_id"])
            except Exception: pass
    print(f"total {len(ids)} | cached {len(done_ids)} | reuse avail {len(reuse)}",flush=True)
    fh=open(CACHE,"a")
    # write reuse records not yet cached
    nre=0
    for tid in ids:
        if tid in done_ids: continue
        if tid in reuse:
            fh.write(json.dumps(reuse[tid])+"\n"); done_ids.add(tid); nre+=1
    fh.flush()
    print(f"wrote {nre} reuse records",flush=True)
    todo=[t for t in ids if t not in done_ids]
    print(f"to fetch: {len(todo)}",flush=True)
    t0=time.time(); N=len(todo)
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs={ex.submit(fetch,t):t for t in todo}
        for i,fut in enumerate(as_completed(futs),1):
            rec=fut.result()
            with lock:
                if rec["ok"]: stats["ok"]+=1
                elif rec.get("dead_reason") in ("maxretry",): stats["err"]+=1
                else: stats["dead"]+=1
                stats["done"]+=1
                fh.write(json.dumps(rec)+"\n")
            if i%2000==0 or i==N:
                el=time.time()-t0; rate=i/el; eta=(N-i)/rate if rate else 0
                fh.flush()
                print(f"  {i}/{N} | {rate:.1f}/s | ok {stats['ok']} dead {stats['dead']} err {stats['err']} 429 {stats['429']} | ETA {eta/60:.1f}m",flush=True)
    fh.close()
    print("HYDRATE_DONE",flush=True)

if __name__=="__main__": main()
