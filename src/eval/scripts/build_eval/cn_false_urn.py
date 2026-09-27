"""C2 FALSE URN — full build (two-urn pivot, Daniel go 2026-08-26).

The FALSE side of the tweet fit corpus: 2,000 CN gold-tier cluster-rep posts
(EN, hydrated, media/satire/recency-screened) -> production extraction chain
(--voice user) -> post-chain screens -> note-target matcher -> gate v2 ->
evidence reads (query-v3/read-v5) over band-"false" claims ONLY. Note text is
OFFLINE-ONLY judge input (matcher/gate); it never enters extraction, queries,
or reads. The draw EXCLUDES the 200 prompt-A/B posts (c2_ab_posts.parquet):
gate v2 and the user-voice prompt were developed against them.

  uv run python -m eval.scripts.build_eval.cn_false_urn --draw          # $0 lock 2,000
  uv run python -m eval.scripts.build_eval.cn_false_urn --smoke         # first 50 e2e
  uv run python -m eval.scripts.build_eval.cn_false_urn --chain         # paid: full 2,000
  uv run python -m eval.scripts.build_eval.cn_false_urn --match --gate  # paid: judges
  uv run python -m eval.scripts.build_eval.cn_false_urn --reads         # paid: FALSE claims
  uv run python -m eval.scripts.build_eval.cn_false_urn --report        # $0

Outputs: eval/data/tweet_corpus/cn_false_urn_* (chain files),
eval/data/urn_runs/c2_false/ (match.jsonl, gate.jsonl, scores.jsonl, fit parquet).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.scripts.build_eval.c2_audit import (  # noqa: E402
    MATCH_SYS, _chat, _note_class, _snowflake_date, BOXES, FLAGS7)
from eval.scripts.build_eval.c2_prompt_ab import GATE_SYS  # noqa: E402
from eval.scripts.build_eval.cn_false_stratum import select, screen, MODEL  # noqa: E402

CN = SRC / "eval/data/community_notes"
CORP = SRC / "eval/data/tweet_corpus"
OUT = SRC / "eval/data/urn_runs/c2_false"
SEED = 20260826
N_POSTS = 2000
READS_CAP = 2.50   # USD hard cap for the reads stage (est $1.30)
TAG = "cn_false_urn"
SMOKE_TAG = "cn_false_urn_smoke"
EXT_TAG = "cn_false_urn_ext"  # extension draw (Daniel 2026-08-26: target >=1,000 fit-eligible)


def draw(ext_n: int | None = None) -> None:
    """Lock the 2,000-post draw: full EN pool minus the A/B development posts.

    ext_n: extension mode — draw ext_n MORE posts, excluding the locked draw too."""
    pool = select(10**9, SEED)  # full EN pool in posts schema (funnel printed)
    ab = pl.read_parquet(CORP / "c2_ab_posts.parquet")
    pool = pool.filter(~pl.col("post_id").is_in(ab["post_id"].implode()))
    if ext_n is not None:
        prev = pl.read_parquet(CORP / f"{TAG}_posts.parquet")
        pool = pool.filter(~pl.col("post_id").is_in(prev["post_id"].implode()))
    boxes = pl.read_parquet(CN / "cn_gold_clusters.parquet").select(["noteId"] + BOXES)
    pool = pool.join(boxes, on="noteId", how="left")
    cls = [_note_class(r) for r in pool.iter_rows(named=True)]
    pool = pool.with_columns(pl.Series("note_class", cls))
    n, tag, seed = ((ext_n, EXT_TAG, SEED + 1) if ext_n is not None
                    else (N_POSTS, TAG, SEED))
    posts = pool.sample(n=min(n, pool.height), seed=seed)
    posts.write_parquet(CORP / f"{tag}_posts.parquet")
    print(f"draw locked [{tag}]: {posts.height}/{pool.height} posts (A/B {ab.height} "
          f"excluded, seed {seed}) | note-class mix "
          f"{Counter(posts['note_class'].to_list())}", flush=True)


def chain(posts_path: Path, tag: str, concurrency: int) -> None:
    """The production extraction chain, --voice user (the C2 default)."""
    posts = pl.read_parquet(posts_path)
    ex_html, ex_parq = CORP / f"{tag}_extracted.html", CORP / f"{tag}_extracted.parquet"
    nm_html, ck_parq = CORP / f"{tag}_normalized.html", CORP / f"{tag}_verify_input.parquet"
    run = lambda cmd: subprocess.run(cmd, cwd=SRC, check=True)
    run([sys.executable, "eval/scripts/claim_sourcing/extract_tweet_claims.py",
         "-i", str(posts_path), "-o", str(ex_parq), "--html", str(ex_html),
         "--no-images", "--model", MODEL, "--voice", "user",
         "--concurrency", str(concurrency)])
    run([sys.executable, "eval/scripts/claim_sourcing/normalize_tweet_claims.py",
         "--payload", str(ex_html), "-o", str(nm_html), "--model", MODEL,
         "--voice", "user", "--concurrency", str(concurrency)])
    run([sys.executable, "eval/scripts/claim_sourcing/build_verify_input.py",
         "--normalized", str(nm_html), "--posts", str(posts_path), "-o", str(ck_parq),
         "--extract-model", MODEL, "--normalize-model", MODEL])
    screen(ck_parq, posts, CORP / f"{tag}_claims.parquet")


def _judge_stage(name: str, sys_prompt: str, rows: list[dict], out_path: Path,
                 payload, record, workers: int) -> None:
    """Resumable parallel LLM-judge stage (matcher / gate share the shape)."""
    done = {json.loads(l)["key"] for l in open(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["key"] not in done]
    print(f"{name}: {len(todo)}/{len(rows)} to run", flush=True)
    lock, state, t0 = threading.Lock(), {"cost": 0.0, "n": 0}, time.time()
    fh = open(out_path, "a")

    def work(r):
        try:
            o = _chat(sys_prompt, payload(r))
        except Exception as e:
            with lock:
                state["n"] += 1
                print(f"  [{name}-failed] {r['key']}: {type(e).__name__}: {e}", flush=True)
            return
        with lock:
            state["cost"] += o.pop("_cost", 0)
            state["n"] += 1
            fh.write(json.dumps({"key": r["key"], **record(r, o)}) + "\n")
            if state["n"] % 100 == 0:
                fh.flush()
                el = (time.time() - t0) / 60
                print(f"  {name} {state['n']}/{len(todo)} | ${state['cost']:.3f} | "
                      f"{el:.1f}m {state['n']/max(el,.01):.0f}/min", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"{name} done {state['n']} | ${state['cost']:.4f}", flush=True)


def match(posts_path: Path, claims_path: Path, out_path: Path, workers: int) -> None:
    posts = pl.read_parquet(posts_path)
    notes = pl.read_parquet(CN / "cn_gold.parquet").select(["noteId", "summary"])
    posts = posts.join(notes, on="noteId", how="left")
    claims = pl.read_parquet(claims_path)
    by_post: dict[str, list] = {}
    for c in claims.iter_rows(named=True):
        by_post.setdefault(c["post_id"], []).append(c)
    rows = [{"key": p["post_id"], **p, "claims": by_post.get(p["post_id"], [])}
            for p in posts.iter_rows(named=True)]

    def payload(r):
        numbered = "\n".join(f"{i}. {c['claim']}" for i, c in enumerate(r["claims"])) \
            or "(no claims extracted)"
        return (f"TWEET (@{r['handle']}):\n{r['text']}\n\nCOMMUNITY NOTE:\n{r['summary']}"
                f"\n\nEXTRACTED CLAIMS:\n{numbered}")

    def record(r, o):
        idx = o.get("target_idx")
        ok = isinstance(idx, int) and 0 <= idx < len(r["claims"])
        c = r["claims"][idx] if ok else None
        return {"post_id": r["post_id"], "noteId": r["noteId"],
                "note_class": r["note_class"], "n_claims": len(r["claims"]),
                "target_claim_id": c["claim_id"] if c else None,
                "target_claim": c["claim"] if c else None,
                "confidence": o.get("confidence"), "miss_kind": o.get("miss_kind"),
                "reason": o.get("reason"), "note": r["summary"]}

    OUT.mkdir(parents=True, exist_ok=True)
    _judge_stage("match", MATCH_SYS, rows, out_path, payload, record, workers)


def gate(match_path: Path, out_path: Path, workers: int) -> None:
    recs = [json.loads(l) for l in open(match_path)]
    rows = [{"key": r["post_id"], **r} for r in recs if r.get("target_claim")]
    payload = lambda r: f"CLAIM:\n{r['target_claim']}\n\nCOMMUNITY NOTE:\n{r['note']}"
    record = lambda r, o: {"post_id": r["post_id"], "noteId": r["noteId"],
                           "note_class": r["note_class"],
                           "target_claim_id": r["target_claim_id"],
                           "claim": r["target_claim"],
                           "band": o.get("band"), "why": o.get("why")}
    _judge_stage("gate", GATE_SYS, rows, out_path, payload, record, workers)


def reads(claims_path: Path, gate_path: Path, out_path: Path, workers: int) -> None:
    """Evidence reads over band-"false" claims only (the FALSE urn members)."""
    from pipeline.search import newsguard_score_map
    from eval.scripts.build_eval.evidence_urn_run import run_claim
    gated = [json.loads(l) for l in open(gate_path)]
    false_ids = {g["target_claim_id"]: g for g in gated if g.get("band") == "false"}
    claims = pl.read_parquet(claims_path).filter(
        pl.col("claim_id").is_in(list(false_ids)))
    n0 = claims.height
    # urn membership also requires the post-chain screens (fit-eligible)
    claims = claims.filter(pl.col("checkworthy") & ~pl.col("attribution_form")
                           & ~pl.col("media_locus") & ~pl.col("claim_dup")
                           & ~pl.col("e1_leak"))
    print(f"band-false {n0} -> fit-eligible {claims.height} "
          f"(screens dropped {n0 - claims.height})", flush=True)
    rows = []
    for c in claims.iter_rows(named=True):
        g = false_ids[c["claim_id"]]
        rows.append({
            "review_url": c["claim_id"], "claim_text": c["claim"],
            "publisher_site": "x.com", "claim_date": _snowflake_date(c["post_id"]),
            "claim_type": c["type"], "topic": c.get("topic"),
            "x_context": None, "context_ok": False,
            "resolution_status": "native", "claim_resolved": None,
            "veracity": None, "rating_subtype": None,
            "review_date": None, "x_date": None, "yr": None,
            "screen_verdict": "unscreened", "screen_leak": False,
            "note_class": g["note_class"], "post_id": c["post_id"],
            "noteId": g["noteId"], "attribution_form": c["attribution_form"],
            "media_locus": c["media_locus"], "mixed_signal": c["mixed_signal"]})
    seen = {json.loads(l)["review_url"] for l in open(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["review_url"] not in seen]
    print(f"reads: {len(todo)}/{len(rows)} FALSE claims | cap ${READS_CAP}", flush=True)
    ng = newsguard_score_map()
    lock = threading.Lock()
    budget = {"spent": 0.0, "done": 0, "cached_tok": 0, "prompt_tok": 0, "stop": False}
    t0 = time.time()
    fh = open(out_path, "a")

    def work(row):
        if budget["stop"]:
            return
        try:
            rec = run_claim(row, ng, budget, {})
        except Exception as e:
            with lock:
                budget["done"] += 1
                print(f"  [claim-failed] {row['review_url']}: {type(e).__name__}: {e}",
                      flush=True)
            return
        for k in ("note_class", "post_id", "noteId", "attribution_form",
                  "media_locus", "mixed_signal"):
            rec[k] = row[k]
        with lock:
            budget["spent"] += rec["cost"]
            budget["done"] += 1
            fh.write(json.dumps(rec) + "\n")
            n, el = budget["done"], (time.time() - t0) / 60
            if n % 20 == 0 or n == len(todo):
                fh.flush()
                eta = el / n * (len(todo) - n)
                print(f"  {n}/{len(todo)} | ${budget['spent']:.3f} | {el:.1f}m "
                      f"{n/max(el,.01):.1f}/min | ETA {eta:.0f}m", flush=True)
            if budget["spent"] / n * len(todo) > READS_CAP and n >= 20:
                print("  BUDGET ABORT", flush=True)
                budget["stop"] = True

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"reads done {budget['done']} | ${budget['spent']:.4f} | "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


def report(tag: str, match_path: Path, gate_path: Path, scores_path: Path | None) -> None:
    posts = pl.read_parquet(CORP / f"{tag}_posts.parquet")
    claims = pl.read_parquet(CORP / f"{tag}_claims.parquet")
    m = [json.loads(l) for l in open(match_path)] if match_path.exists() else []
    g = [json.loads(l) for l in open(gate_path)] if gate_path.exists() else []
    rec = sum(1 for r in m if r.get("target_claim_id"))
    bands = Counter(x.get("band") for x in g)
    n_posts = posts.height
    print(f"== {tag} funnel ==")
    print(f"posts {n_posts} -> claims {claims.height} "
          f"({claims.height/n_posts:.2f}/post) | matched {rec}/{len(m)} "
          f"({rec/max(len(m),1):.0%}) | gate {dict(bands)} | "
          f"FALSE yield {bands['false']}/{n_posts} = {bands['false']/n_posts:.2f}/post")
    print("by note class (gated false / matched):")
    for cls in ("factual_error", "unverified_as_fact", "mixed_true_core", "other"):
        mm = [r for r in m if r["note_class"] == cls and r.get("target_claim_id")]
        gg = [x for x in g if x["note_class"] == cls and x.get("band") == "false"]
        print(f"  {cls}: {len(gg)}/{len(mm)}")
    if scores_path and scores_path.exists():
        recs = [json.loads(l) for l in open(scores_path)]
        flags = Counter()
        for r in recs:
            for d in r["results"]:
                f = (d.get("read") or {}).get("direction")
                if f in FLAGS7:
                    flags[f] += 1
        tot = sum(flags.values())
        sup = (flags["5"] + flags["4"]) / tot
        ref = (flags["1"] + flags["2"]) / tot
        print(f"\n== reads: {len(recs)} claims, {tot} docs ==")
        print(f"flag mix: {dict(flags)} | support {sup:.1%} | refute {ref:.1%} | "
              f"silent {1-sup-ref:.1%}")
        print("(E1 gold-FALSE reference: support 13.9% / refute 15.6% / silent 70.5%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    for s in ("draw", "smoke", "chain", "match", "gate", "reads", "report"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--concurrency", type=int, default=12, help="chain LLM concurrency")
    ap.add_argument("--workers", type=int, default=16, help="judge/reads workers")
    ap.add_argument("--ext", action="store_true",
                    help="extension build: +ext-n posts beyond the locked 2,000")
    ap.add_argument("--ext-n", type=int, default=4500)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = EXT_TAG if a.ext else TAG
    sfx = "_ext" if a.ext else ""
    mp, gp, sp_ = OUT / f"match{sfx}.jsonl", OUT / f"gate{sfx}.jsonl", OUT / f"scores{sfx}.jsonl"
    if a.draw:
        draw(a.ext_n if a.ext else None)
    if a.smoke:
        posts = pl.read_parquet(CORP / f"{TAG}_posts.parquet").head(50)
        sp = CORP / f"{SMOKE_TAG}_posts.parquet"
        posts.write_parquet(sp)
        chain(sp, SMOKE_TAG, a.concurrency)
        match(sp, CORP / f"{SMOKE_TAG}_claims.parquet", OUT / "smoke_match.jsonl", a.workers)
        gate(OUT / "smoke_match.jsonl", OUT / "smoke_gate.jsonl", a.workers)
        report(SMOKE_TAG, OUT / "smoke_match.jsonl", OUT / "smoke_gate.jsonl", None)
    if a.chain:
        chain(CORP / f"{tag}_posts.parquet", tag, a.concurrency)
    if a.match:
        match(CORP / f"{tag}_posts.parquet", CORP / f"{tag}_claims.parquet",
              mp, a.workers)
    if a.gate:
        gate(mp, gp, a.workers)
    if a.reads:
        reads(CORP / f"{tag}_claims.parquet", gp, sp_, a.workers)
    if a.report:
        report(tag, mp, gp, sp_)


if __name__ == "__main__":
    main()
