"""How many strict acquitted TRUE claims does the L3 pool actually hold?

    uv run python -m eval.scripts.build_eval.acquitted_yield --notes      # once, ~min
    uv run python -m eval.scripts.build_eval.acquitted_yield --run --batches 1
    uv run python -m eval.scripts.build_eval.acquitted_yield --report

The 2026-08-25 smoke put the strict yield at 0.02 claims per post, but that rests
on ONE positive example (acquitted_clean 1 of 50 posts). The exact binomial 95%
interval on 1/50 runs [0.05%, 10.6%], so the all-L3 pool is somewhere in
[7, 1,440] claims. 35 and 400 are not distinguishable from that. This measures it.

No new hydration is needed: the full L3 pool is already hydrated, 13,271 live
English posts. No reads either -- extract, match-to-note and band-gate are all
LLM-only, so this can run alongside a search-phase job without touching the
Serper window.

Batched and RESUMABLE over a seeded shuffle of the whole pool, so a run can be
extended rather than restarted. Each batch appends to gate.jsonl and reprints the
funnel and the running yield with its interval, which is what the pool-size
decision actually needs (Daniel 2026-08-27: do not stop at 500, keep going if
1,000 looks reachable).

Bands are true_stratum_smoke's, unchanged, and only `acquitted_clean` counts.
`still_contested` is the same construct as fc-gold's rating_subtype=mixed, the
stratum we set aside from the eval population, so counting it here would
reintroduce on the fit side exactly what we removed on the eval side.
`off_target` claims were never accused at all and carry no label.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.scripts.build_eval.c2_audit import MATCH_SYS, _chat          # noqa: E402
from eval.scripts.build_eval.cn_false_stratum import screen, MODEL     # noqa: E402
from eval.scripts.build_eval.true_stratum_smoke import (               # noqa: E402
    GATE_SYS, _read_raw_notes, CRNH)

CORP = SRC / "eval/data/tweet_corpus"
OUT = SRC / "eval/data/urn_runs/acquitted_yield"
HYDRATED = ("acquitted_hydrated.jsonl", "acquitted_hydrated_0825_new.jsonl",
            "acquitted_hydrated_0825_rest.jsonl")
POOL = CORP / "acquitted_l3_pool.parquet"
NOTES = OUT / "rejected_notes.parquet"
GATES = OUT / "gate.jsonl"
MATCHES = OUT / "match.jsonl"
SEED = 20260827


def en_pool() -> dict[str, dict]:
    """Live English hydrated posts that are in the L3 acquitted pool."""
    l3 = set(pl.read_parquet(POOL)["tweetId"].cast(str).to_list())
    hyd = {}
    for f in HYDRATED:
        p = CORP / f
        if not p.exists():
            continue
        for line in p.open():
            d = json.loads(line)
            if (str(d.get("code")) == "200" and d.get("text") and d.get("lang") == "en"
                    and str(d["tweetId"]) in l3):
                hyd[str(d["tweetId"])] = d
    return hyd


def build_notes() -> None:
    """One pass over the raw CN dump for every post in the pool. Cache it."""
    OUT.mkdir(parents=True, exist_ok=True)
    ids = set(en_pool())
    print(f"parsing raw dump for rejected notes on {len(ids):,} posts ...", flush=True)
    notes = _read_raw_notes(ids)
    rej = notes.filter((pl.col("currentStatus") == CRNH) | (pl.col("lockedStatus") == CRNH))
    rej = rej.sort("createdAtMillis", descending=True).unique(subset="tweetId", keep="first")
    rej.write_parquet(NOTES)
    print(f"rejected-note coverage: {rej.height:,}/{len(ids):,} posts "
          f"({rej.height/len(ids)*100:.1f}%)", flush=True)


def _done_ids() -> set[str]:
    if not MATCHES.exists():
        return set()
    return {json.loads(l)["post_id"] for l in MATCHES.open()}


def _chain(posts: pl.DataFrame, tag: str, concurrency: int) -> pl.DataFrame:
    """extract -> normalize -> verify_input -> screen, same chain as the smoke."""
    pp = CORP / f"{tag}_posts.parquet"
    posts.write_parquet(pp)
    run = lambda c: subprocess.run(c, cwd=SRC, check=True)
    run([sys.executable, "eval/scripts/claim_sourcing/extract_tweet_claims.py",
         "-i", str(pp), "-o", str(CORP / f"{tag}_extracted.parquet"),
         "--html", str(CORP / f"{tag}_extracted.html"), "--no-images",
         "--model", MODEL, "--voice", "user", "--concurrency", str(concurrency)])
    run([sys.executable, "eval/scripts/claim_sourcing/normalize_tweet_claims.py",
         "--payload", str(CORP / f"{tag}_extracted.html"),
         "-o", str(CORP / f"{tag}_normalized.html"), "--model", MODEL,
         "--voice", "user", "--concurrency", str(concurrency)])
    run([sys.executable, "eval/scripts/claim_sourcing/build_verify_input.py",
         "--normalized", str(CORP / f"{tag}_normalized.html"), "--posts", str(pp),
         "-o", str(CORP / f"{tag}_verify_input.parquet"),
         "--extract-model", MODEL, "--normalize-model", MODEL])
    tagged = posts.with_columns([
        pl.lit(None, dtype=pl.Utf8).alias("noteId"),
        pl.lit(None, dtype=pl.Utf8).alias("cluster_id"),
        pl.lit(None, dtype=pl.UInt32).alias("cluster_size"),
        pl.lit(None, dtype=pl.Utf8).alias("note_date"),
        pl.lit(False).alias("mixed_signal")])
    screen(CORP / f"{tag}_verify_input.parquet", tagged, CORP / f"{tag}_claims.parquet")
    return pl.read_parquet(CORP / f"{tag}_claims.parquet")


def wilson(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval. Behaves at k=0 and k=n, unlike the normal approximation."""
    if not n:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run_batches(n_batches: int, batch: int, workers: int, concurrency: int) -> None:
    hyd, done = en_pool(), _done_ids()
    notes = {r["tweetId"]: r["summary"] for r in pl.read_parquet(NOTES).iter_rows(named=True)}
    order = sorted(set(hyd) & set(notes))          # only posts that HAVE a rejected note
    random.Random(SEED).shuffle(order)
    todo = [t for t in order if t not in done]
    print(f"pool {len(hyd):,} EN-hydrated | with a rejected note {len(order):,} "
          f"| already done {len(done):,} | todo {len(todo):,}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    for b in range(n_batches):
        ids = todo[b * batch:(b + 1) * batch]
        if not ids:
            print("pool exhausted", flush=True)
            break
        tag = f"acqy_b{len(done)//batch + b:03d}"
        posts = pl.DataFrame([{
            "post_id": t, "cell": "acquitted", "domain": "x.com",
            "handle": hyd[t].get("author") or "",
            "url": f"https://x.com/{hyd[t].get('author') or '_'}/status/{t}",
            "created_at": hyd[t].get("created_at"), "lang": "en", "text": hyd[t]["text"],
            "n_images": 0, "image_urls": [], "n_videos": 0, "video_urls": [],
        } for t in ids])
        print(f"\n=== batch {b+1}/{n_batches}: {len(ids)} posts ({tag}) ===", flush=True)
        claims = _chain(posts, tag, concurrency)

        by_post = {p: g for p, g in claims.group_by("post_id")}
        cost = [0.0]

        def do_match(t):
            pc = by_post.get((t,))
            clist = pc["claim"].to_list() if pc is not None else []
            cids = pc["claim_id"].to_list() if pc is not None else []
            numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(clist)) \
                or "(no claims extracted)"
            o = _chat(MATCH_SYS, f"TWEET (@{hyd[t].get('author') or ''}):\n{hyd[t]['text']}\n\n"
                                 f"COMMUNITY NOTE:\n{notes[t]}\n\nEXTRACTED CLAIMS:\n{numbered}")
            cost[0] += o.pop("_cost", 0)
            idx = o.get("target_idx")
            ok = isinstance(idx, int) and 0 <= idx < len(clist)
            return {"post_id": t, "n_claims": len(clist),
                    "target_claim_id": cids[idx] if ok else None,
                    "target_claim": clist[idx] if ok else None,
                    "miss_kind": o.get("miss_kind"), "confidence": o.get("confidence"),
                    "note": notes[t], "tweet": hyd[t]["text"]}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            matched = list(ex.map(do_match, ids))
        with MATCHES.open("a") as fh:
            for m in matched:
                fh.write(json.dumps(m) + "\n")

        hits = [m for m in matched if m["target_claim"]]

        def do_gate(m):
            o = _chat(GATE_SYS, f"CLAIM:\n{m['target_claim']}\n\n"
                                f"REJECTED COMMUNITY NOTE:\n{m['note']}")
            cost[0] += o.pop("_cost", 0)
            return {"post_id": m["post_id"], "claim_id": m["target_claim_id"],
                    "claim": m["target_claim"], "band": o.get("band"), "why": o.get("why")}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            gated = list(ex.map(do_gate, hits))
        with GATES.open("a") as fh:
            for g in gated:
                fh.write(json.dumps(g) + "\n")

        bands = Counter(g["band"] for g in gated)
        print(f"batch: {len(ids)} posts -> {claims.height} claims -> {len(hits)} matched "
              f"-> {dict(bands)} | ${cost[0]:.3f}", flush=True)
        done |= set(ids)
        report()


def report() -> None:
    if not GATES.exists():
        print("nothing yet")
        return
    gates = [json.loads(l) for l in GATES.open()]
    matches = [json.loads(l) for l in MATCHES.open()]
    n_posts = len(matches)
    bands = Counter(g["band"] for g in gates)
    clean = bands.get("acquitted_clean", 0)
    lo, hi = wilson(clean, n_posts)
    notes_n = pl.read_parquet(NOTES).height if NOTES.exists() else 0
    print(f"\n--- acquitted yield ---")
    print(f"posts processed        {n_posts:,}")
    print(f"matched to their note  {len(gates):,} ({len(gates)/max(1,n_posts)*100:.1f}%)")
    for b in ("acquitted_clean", "still_contested", "off_target"):
        print(f"  {b:18s} {bands.get(b,0):,}")
    print(f"strict yield/post      {clean/max(1,n_posts):.4f}  "
          f"95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"=> pool over the {notes_n:,} noted L3 posts: "
          f"{clean/max(1,n_posts)*notes_n:,.0f}  [{lo*notes_n:,.0f}, {hi*notes_n:,.0f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", action="store_true", help="parse the raw dump once")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--batches", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=12)
    a = ap.parse_args()
    if a.notes:
        build_notes()
    if a.run:
        run_batches(a.batches, a.batch, a.workers, a.concurrency)
    if a.report:
        report()


if __name__ == "__main__":
    main()
