"""C2 prompt A/B (Log-Odds Sprint) — outlet vs ordinary-user extraction prompt, 200 CN posts.

Motivation: the C2 audit found the extraction chain's outlet-built system prompt returns
empty on ordinary-user posts (replies, rants, anecdotes) — 4-5 of 17 recovery misses.
Daniel's go 2026-08-24: build the user variant (--voice user in extract/normalize), test
recovery at 200 posts, both arms on the SAME posts, note text offline-only as ever.
Also: gate v2 with a "partly" band (the Chelsea-double-treble class must stop passing).

  uv run python -m eval.scripts.build_eval.c2_prompt_ab --draw     # $0 lock the 200
  uv run python -m eval.scripts.build_eval.c2_prompt_ab --extract  # paid: chain x2 arms
  uv run python -m eval.scripts.build_eval.c2_prompt_ab --match    # paid: matcher x2 arms
  uv run python -m eval.scripts.build_eval.c2_prompt_ab --gate     # paid: gate v2 x2 arms
  uv run python -m eval.scripts.build_eval.c2_prompt_ab --report   # $0

Outputs: eval/data/tweet_corpus/c2_ab_posts.parquet, c2_ab_{old,new}_* chain files,
eval/data/urn_runs/c2_audit/ab_match_{old,new}.jsonl, ab_gate_{old,new}.jsonl.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval.c2_audit import (MATCH_SYS, MATCH_HASH, _chat,  # noqa: E402
                                              _note_class, BOXES)
from eval.scripts.build_eval.cn_false_stratum import select, screen, MODEL  # noqa: E402

CN = SRC / "eval/data/community_notes"
CORP = SRC / "eval/data/tweet_corpus"
OUT = SRC / "eval/data/urn_runs/c2_audit"
SEED = 20260824
ARMS = {"old": "outlet", "new": "user"}

GATE_SYS = """You judge one claim extracted from a tweet against the Community Note attached to that tweet. Take the note as accurate. Decide what the note, if accurate, establishes about THIS claim's literal truth.

Bands, exactly one:
- "false": the claim's central proposition, as stated, is false. The note's correction contradicts what the claim itself asserts — not its context, not its framing, not something else in the tweet.
- "partly": the claim's literal core is true or substantially true, and the note corrects framing, adds missing context, disputes an implication, or contradicts only a secondary detail (a date, a count that is off but directionally right, an aside). True-but-missing-context always lands here, never in "false".
- "true_or_untouched": the note does not bear on this claim's truth at all — it targets an image or video, who posted it, a different claim in the tweet, or it actually confirms this claim.

Discipline: judge the claim EXACTLY as written, in isolation. If the tweet was misleading but this extracted claim is a defensible generic version, the note likely lands "partly" or "true_or_untouched" — that is the correct answer, not a reason to stretch "false". A note that corrects exclusivity, primacy, or scale — "not the only", "not the first", "others have too" — contradicts an implied superlative, not the event itself: unless the claim as written asserts being only, first, or biggest, that is "partly" or "true_or_untouched", never "false". "Directly contradicts" means the note asserts the claim's own proposition did not happen or is not so.

Return only JSON:
{"band": "false"|"partly"|"true_or_untouched", "why": "<one sentence citing the decisive note content>"}"""


GATE_HASH = prompt_hash(GATE_SYS)


def draw() -> None:
    smoke = pl.read_parquet(CORP / "cn_false_smoke_posts.parquet")
    pool = select(10**9, SEED)  # full EN pool, posts schema
    boxes = pl.read_parquet(CN / "cn_gold_clusters.parquet").select(["noteId"] + BOXES)
    pool = pool.join(boxes, on="noteId", how="left")
    smoke = smoke.join(boxes, on="noteId", how="left")
    cls = [_note_class(r) for r in pool.iter_rows(named=True)]
    pool = pool.with_columns(pl.Series("note_class", cls))
    smoke = smoke.with_columns(
        pl.Series("note_class", [_note_class(r) for r in smoke.iter_rows(named=True)]))
    rest = pool.filter(~pl.col("post_id").is_in(smoke["post_id"].implode()))

    mix = Counter(cls)
    n_new = 150
    want = {c: round(n_new * v / len(cls)) for c, v in mix.items()}
    want[max(want, key=want.get)] += n_new - sum(want.values())  # rounding drift
    parts = []
    for c, k in want.items():
        sub = rest.filter(pl.col("note_class") == c)
        if k > 0 and sub.height:
            parts.append(sub.sample(n=min(k, sub.height), seed=SEED))
    new = pl.concat(parts).head(n_new)
    ab = pl.concat([smoke.with_columns(pl.lit("smoke").alias("provenance")),
                    new.with_columns(pl.lit("new").alias("provenance"))])
    ab.write_parquet(CORP / "c2_ab_posts.parquet")
    print(f"draw locked: {ab.height} posts (smoke {smoke.height} + new {new.height}) | "
          f"class mix {Counter(ab['note_class'].to_list())} | pool mix "
          f"{ {c: f'{v/len(cls):.0%}' for c, v in mix.items()} }", flush=True)


def extract(concurrency: int) -> None:
    posts_path = CORP / "c2_ab_posts.parquet"
    posts = pl.read_parquet(posts_path)
    for arm, voice in ARMS.items():
        tag = f"c2_ab_{arm}"
        ex_html, ex_parq = CORP / f"{tag}_extracted.html", CORP / f"{tag}_extracted.parquet"
        nm_html, ck_parq = CORP / f"{tag}_normalized.html", CORP / f"{tag}_verify_input.parquet"
        run = lambda cmd: subprocess.run(cmd, cwd=SRC, check=True)
        print(f"== arm {arm} (voice={voice}) ==", flush=True)
        run([sys.executable, "eval/scripts/claim_sourcing/extract_tweet_claims.py",
             "-i", str(posts_path), "-o", str(ex_parq), "--html", str(ex_html),
             "--no-images", "--model", MODEL, "--voice", voice,
             "--concurrency", str(concurrency)])
        run([sys.executable, "eval/scripts/claim_sourcing/normalize_tweet_claims.py",
             "--payload", str(ex_html), "-o", str(nm_html), "--model", MODEL,
             "--voice", voice, "--concurrency", str(concurrency)])
        run([sys.executable, "eval/scripts/claim_sourcing/build_verify_input.py",
             "--normalized", str(nm_html), "--posts", str(posts_path), "-o", str(ck_parq),
             "--extract-model", MODEL, "--normalize-model", MODEL])
        screen(ck_parq, posts, CORP / f"{tag}_claims.parquet")


def match() -> None:
    posts = pl.read_parquet(CORP / "c2_ab_posts.parquet")
    notes = pl.read_parquet(CN / "cn_gold.parquet").select(["noteId", "summary"])
    posts = posts.join(notes, on="noteId", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        claims = pl.read_parquet(CORP / f"c2_ab_{arm}_claims.parquet")
        out = OUT / f"ab_match_{arm}.jsonl"
        done = {json.loads(l)["post_id"] for l in open(out)} if out.exists() else set()
        fh = open(out, "a")
        cost = 0.0
        for i, p in enumerate(posts.iter_rows(named=True)):
            if p["post_id"] in done:
                continue
            pc = claims.filter(pl.col("post_id") == p["post_id"])
            clist, cids = pc["claim"].to_list(), pc["claim_id"].to_list()
            numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(clist)) or "(no claims extracted)"
            o = _chat(MATCH_SYS, f"TWEET (@{p['handle']}):\n{p['text']}\n\n"
                                 f"COMMUNITY NOTE:\n{p['summary']}\n\nEXTRACTED CLAIMS:\n{numbered}")
            cost += o.pop("_cost", 0)
            idx = o.get("target_idx")
            ok = isinstance(idx, int) and 0 <= idx < len(clist)
            fh.write(json.dumps({
                "post_id": p["post_id"], "noteId": p["noteId"], "arm": arm,
                "note_class": p["note_class"], "provenance": p["provenance"],
                "n_claims": len(clist),
                "target_claim_id": cids[idx] if ok else None,
                "target_claim": clist[idx] if ok else None,
                "confidence": o.get("confidence"), "miss_kind": o.get("miss_kind"),
                "miss_detail": o.get("miss_detail"), "reason": o.get("reason"),
                "prompt_hash": MATCH_HASH,
                "tweet": p["text"], "note": p["summary"]}) + "\n")
            if (i + 1) % 25 == 0:
                fh.flush()
                print(f"  {arm} {i+1}/{posts.height} | ${cost:.4f}", flush=True)
        fh.close()
        print(f"match {arm} done | ${cost:.4f}", flush=True)


def gate() -> None:
    for arm in ARMS:
        recs = [json.loads(l) for l in open(OUT / f"ab_match_{arm}.jsonl")]
        out = OUT / f"ab_gate_{arm}.jsonl"
        done = {json.loads(l)["post_id"] for l in open(out)} if out.exists() else set()
        fh = open(out, "a")
        cost = 0.0
        for r in recs:
            if not r["target_claim"] or r["post_id"] in done:
                continue
            o = _chat(GATE_SYS, f"CLAIM:\n{r['target_claim']}\n\nCOMMUNITY NOTE:\n{r['note']}")
            cost += o.pop("_cost", 0)
            fh.write(json.dumps({"post_id": r["post_id"], "arm": arm,
                                 "note_class": r["note_class"], "claim": r["target_claim"],
                                 "band": o.get("band"), "why": o.get("why"),
                                 "prompt_hash": GATE_HASH}) + "\n")
        fh.close()
        print(f"gate {arm} done | ${cost:.4f}", flush=True)


def report() -> None:
    posts = pl.read_parquet(CORP / "c2_ab_posts.parquet")
    m = {arm: {json.loads(l)["post_id"]: json.loads(l)
               for l in open(OUT / f"ab_match_{arm}.jsonl")} for arm in ARMS}
    ids = posts["post_id"].to_list()
    rec = {arm: {pid: bool(m[arm].get(pid, {}).get("target_claim_id")) for pid in ids}
           for arm in ARMS}

    print(f"== note-target recovery, n={len(ids)} posts ==")
    for arm in ARMS:
        n = sum(rec[arm].values())
        print(f"  {arm}: {n}/{len(ids)} ({n/len(ids):.0%})")
    both = sum(rec["old"][p] and rec["new"][p] for p in ids)
    new_only = sum(rec["new"][p] and not rec["old"][p] for p in ids)
    old_only = sum(rec["old"][p] and not rec["new"][p] for p in ids)
    neither = len(ids) - both - new_only - old_only
    print(f"  discordant: new-only {new_only} | old-only {old_only} | both {both} | neither {neither}")

    print("\n== recovery by note class ==")
    cls_of = dict(zip(posts["post_id"].to_list(), posts["note_class"].to_list()))
    for c in ("factual_error", "unverified_as_fact", "mixed_true_core", "other"):
        sub = [p for p in ids if cls_of[p] == c]
        if sub:
            print(f"  {c} (n={len(sub)}): old {sum(rec['old'][p] for p in sub)} | "
                  f"new {sum(rec['new'][p] for p in sub)}")

    print("\n== volume / junk ==")
    for arm in ARMS:
        cl = pl.read_parquet(CORP / f"c2_ab_{arm}_claims.parquet")
        cats = Counter(cl["category"].to_list())
        n = cl.height
        print(f"  {arm}: {n} claims ({n/len(ids):.2f}/post) | checkable "
              f"{cats['checkable']} ({cats['checkable']/n:.0%}) | "
              f"artifact+dup {cats['artifact']+cats['duplicate']} | "
              f"opinion {cats['opinion']} | trivial {cats['trivial']} | "
              f"unresolved {cats['unresolved']}")

    print("\n== gate v2 bands (matched claims) ==")
    for arm in ARMS:
        g = [json.loads(l) for l in open(OUT / f"ab_gate_{arm}.jsonl")]
        bands = Counter(x["band"] for x in g)
        print(f"  {arm}: {dict(bands)} | fit-eligible (false): "
              f"{bands['false']}/{len(g)}")

    print("\n== known audit extraction-miss posts, new arm ==")
    known = ["2051774692248482077", "2030043430177443859", "1857066310007492713",
             "2062235126784840167", "2066644846710804878", "1840741385491583107"]
    for pid in known:
        if pid not in m["new"]:
            continue
        r = m["new"][pid]
        print(f"  {pid}: recovered={bool(r['target_claim_id'])} -> {r['target_claim'] or r['miss_kind']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for s in ("draw", "extract", "match", "gate", "report"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--concurrency", type=int, default=12)
    a = ap.parse_args()
    if a.draw:
        draw()
    if a.extract:
        extract(a.concurrency)
    if a.match:
        match()
    if a.gate:
        gate()
    if a.report:
        report()
