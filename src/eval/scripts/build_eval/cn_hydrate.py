"""CN tweet hydration via fxtwitter (free, no auth) — text-only, resumable.

Pulls post text for the hydration list (cluster representatives + non-collapsible
members) in shuffled order (seeded — a time-boxed run yields a representative random
sample). Appends JSONL (one line per attempt) so later runs resume by skipping seen
ids. NO media downloads (Daniel 2026-07-23). Pace: --pace seconds between requests
(politeness); 429/5xx → 60s backoff, and the run aborts if two backoffs don't clear it.

  uv run python -m eval.scripts.build_eval.cn_hydrate --minutes 20 --pace 0.7
  uv run python -m eval.scripts.build_eval.cn_hydrate --workers 3 --pace 0.5   # full run

Concurrency model (2026-07-23): --workers threads absorb request latency while a
GLOBAL dispatcher releases one request every --pace seconds (aggregate rate = 1/pace).
Adaptive politeness: any 429/5xx → 60s cooldown AND the dispatch interval is
permanently multiplied 1.5× (the run seeks the service's comfort level rather than
assuming it); >5 cooldowns → abort. Measured baseline: 0.75 req/s × 900 requests with
zero pushback.
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from eval.textnorm import clean_text

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SRC = Path("eval/data/community_notes/cn_gold_clusters.parquet")
OUT = Path("eval/data/community_notes/hydrated.jsonl")
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(tid: str) -> dict:
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            f"https://api.fxtwitter.com/status/{tid}", headers=UA), timeout=15)
        d = json.loads(r.read())
        t = d.get("tweet") or {}
        return {"tweetId": tid, "code": d.get("code"),
                "text": clean_text(t.get("text")), "author": (t.get("author") or {}).get("screen_name"),
                "lang": t.get("lang"), "created_at": t.get("created_at")}
    except urllib.error.HTTPError as e:
        return {"tweetId": tid, "code": e.code}
    except Exception as e:
        return {"tweetId": tid, "code": f"err:{type(e).__name__}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=0, help="0 = run until the list is done")
    ap.add_argument("--pace", type=float, default=0.5, help="global dispatch interval (s)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ids-parquet", help="hydrate the tweetId column of this parquet instead of the cluster list")
    ap.add_argument("--out", help="output JSONL (default: the CN hydration store)")
    args = ap.parse_args()

    import threading
    from concurrent.futures import ThreadPoolExecutor

    global OUT
    if args.out:
        OUT = Path(args.out)
    if args.ids_parquet:
        ids = pl.read_parquet(args.ids_parquet)["tweetId"].to_list()
    else:
        df = pl.read_parquet(SRC)
        ids = df.filter(pl.col("is_rep") | ~pl.col("collapsible"))["tweetId"].to_list()
    random.Random(42).shuffle(ids)
    seen = set()
    if OUT.exists():
        for l in open(OUT):
            try:
                seen.add(json.loads(l)["tweetId"])
            except Exception:
                pass
    todo = [t for t in ids if t not in seen]
    print(f"hydration list {len(ids)}; already done {len(seen)}; todo {len(todo)}; "
          f"{args.workers} workers, dispatch every {args.pace}s "
          f"({'no deadline' if not args.minutes else f'{args.minutes:.0f}m'})", flush=True)

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    state = {"live": 0, "gone": 0, "other": 0, "backoffs": 0, "pace": args.pace,
             "done": 0, "stop": False}
    lock = threading.Lock()
    fh = open(OUT, "a")
    t0 = time.time()

    def handle(tid):
        rec = fetch(tid)
        rec["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        c = rec["code"]
        with lock:
            fh.write(json.dumps(rec) + "\n")
            state["done"] += 1
            if c == 200:
                state["live"] += 1
            elif c in (404, 401):
                state["gone"] += 1
            else:
                state["other"] += 1
                if c == 429 or (isinstance(c, int) and c >= 500):
                    state["backoffs"] += 1
                    state["pace"] *= 1.5
                    print(f"  [{c}] cooldown 60s; dispatch interval -> "
                          f"{state['pace']:.2f}s (backoff {state['backoffs']})", flush=True)
                    if state["backoffs"] > 5:
                        state["stop"] = True
                    state["cooldown_until"] = time.time() + 60
            if state["done"] % 500 == 0:
                fh.flush()
                el = time.time() - t0
                rate = state["done"] / el * 60
                lr = state["live"] / max(state["live"] + state["gone"], 1)
                eta_h = (len(todo) - state["done"]) / max(state["done"] / el, 1e-9) / 3600
                print(f"  {state['done']}/{len(todo)} | {rate:.0f}/min | live {lr:.0%} "
                      f"| ETA {eta_h:.1f}h", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for tid in todo:
            if state["stop"] or (deadline and time.time() > deadline):
                break
            cu = state.get("cooldown_until", 0)
            if cu > time.time():
                time.sleep(cu - time.time())
            ex.submit(handle, tid)
            time.sleep(state["pace"])
    fh.close()
    print(f"\nfinished: live {state['live']} | gone {state['gone']} | other {state['other']} "
          f"(this run {state['done']}, {(time.time()-t0)/3600:.1f}h)", flush=True)


if __name__ == "__main__":
    main()
