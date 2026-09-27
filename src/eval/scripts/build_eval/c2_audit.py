"""C2 audit smoke (Log-Odds Sprint) — Daniel's two-question protocol before the full build.

Q1 RECOVERY: for CN-noted posts, does our extraction+normalization recover the claim
the note targets? Note text is used OFFLINE ONLY as post-hoc judge input — it never
enters extraction, queries, or reads (leakage guard).
Q2 VERDICT MATCH: on recovered note-target claims, does the instrument's flag match
the note's "misleading" verdict? Split by note checkbox class; the misleading-but-
true-core cell (missing-context w/o factual-error/unverified) reported separately.

Stages (run in order):
  uv run python -m eval.scripts.build_eval.c2_audit --dist    # $0 checkbox distributions
  uv run python -m eval.scripts.build_eval.c2_audit --match   # ~$0.01 note-target matching
  uv run python -m eval.scripts.build_eval.c2_audit --run     # paid: verify recovered claims
  uv run python -m eval.scripts.build_eval.c2_audit --report  # verdict-match analysis

Inputs: the existing 50-post C2 smoke (eval/data/tweet_corpus/cn_false_smoke_*).
Outputs: eval/data/urn_runs/c2_audit/ (match.jsonl, verify.jsonl).
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from pipeline.search import newsguard_score_map  # noqa: E402
from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import run_claim  # noqa: E402

CN = SRC / "eval/data/community_notes"
CORP = SRC / "eval/data/tweet_corpus"
OUT = SRC / "eval/data/urn_runs/c2_audit"
# The shipped constants (7-flag since 2026-09-14) and the 3-voice fit on the same
# frozen population, kept beside them as the comparison arm.
E1_METRICS = SRC / "eval/data/urn_runs/e1_ctx/headline_metrics.json"
E1_METRICS_3VOICE = SRC / "eval/data/urn_runs/e1_ctx/headline_metrics_clustered.json"
PAD_TO = 10
SNAPSHOT = datetime.date(2026, 7, 23)
BASE = "https://api.deepinfra.com/v1/openai"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
_KEY = None


def _key() -> str:
    """Read the API key on first use, never at import time."""
    global _KEY
    if _KEY is None:
        _KEY = next(l.split("=", 1)[1].strip() for l in (SRC / ".env").read_text().splitlines()
                    if l.startswith("DEEPINFRA_API_KEY="))
    return _KEY


BUDGET_CAP = 0.30
FLAGS7 = ["5", "4", "3", "2", "1", "X", "I"]

BOXES = ["misleadingFactualError", "misleadingUnverifiedClaimAsFact",
         "misleadingMissingImportantContext", "misleadingManipulatedMedia",
         "misleadingOutdatedInformation", "misleadingSatire"]


def _b(col):
    return pl.col(col).cast(pl.Int64, strict=False).fill_null(0)


def _note_class(row: dict) -> str:
    """Exclusive class for analysis cells. Priority: hard-false boxes > mixed > other."""
    fe = int(row.get("misleadingFactualError") or 0)
    uv = int(row.get("misleadingUnverifiedClaimAsFact") or 0)
    mc = int(row.get("misleadingMissingImportantContext") or 0)
    if fe:
        return "factual_error"
    if uv:
        return "unverified_as_fact"
    if mc:
        return "mixed_true_core"          # missing-context without FE/UV
    return "other"                        # outdated / media / satire residue


# ---------------------------------------------------------------- stage 0: dist

def dist() -> None:
    reps = pl.read_parquet(CN / "cn_gold_clusters.parquet").filter(pl.col("is_rep"))
    tot = sum(_b(f) for f in BOXES)
    reps = reps.with_columns([
        (_b("misleadingManipulatedMedia").eq(1) & tot.eq(1)).alias("media_only"),
        (_b("misleadingSatire").eq(1) & tot.eq(1)).alias("satire_only"),
        (pl.col("note_date") > SNAPSHOT - datetime.timedelta(days=30)).alias("too_young"),
    ])
    sel = reps.filter(~pl.col("media_only") & ~pl.col("satire_only") & ~pl.col("too_young"))
    hyd_ids = set()
    with open(CN / "hydrated.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if str(d.get("code")) == "200" and d.get("text") and d.get("lang") == "en":
                hyd_ids.add(str(d["tweetId"]))
    pool = sel.filter(pl.col("tweetId").is_in(hyd_ids))
    smoke = pl.read_parquet(CORP / "cn_false_smoke_posts.parquet").select("noteId") \
        .join(reps, on="noteId", how="left")

    for name, df in (("EN pool", pool), ("smoke draw", smoke)):
        n = df.height
        print(f"== {name} (n={n}) checkbox marginals (notes tick multiple) ==")
        for b in BOXES:
            print(f"  {b}: {df[b].cast(pl.Int64, strict=False).fill_null(0).sum()} "
                  f"({df[b].cast(pl.Int64, strict=False).fill_null(0).sum()/n:.0%})")
        cls = Counter(_note_class(r) for r in df.iter_rows(named=True))
        print(f"  exclusive class: {dict(cls)}")
        print(f"  co-occurrence FE&MC: "
              f"{df.filter(_b('misleadingFactualError').eq(1) & _b('misleadingMissingImportantContext').eq(1)).height}")


# ---------------------------------------------------------------- stage 1: match

MATCH_SYS = """You match a Community Note to the claim it targets. You get a tweet, the note that was attached to it, and a numbered list of claims extracted from the tweet. Decide which extracted claim (if any) the note is disputing or correcting.

Rules:
- The note targets the specific proposition it argues against, not the tweet's topic.
- Pick a claim only if the note's correction bears on THAT claim's truth. Paraphrase differences are fine; a claim about an adjacent fact is not a match.
- If the note disputes something no extracted claim states (an image/video, the tweet's framing or implication, a claim the extractor missed), answer null and classify why.

Return only JSON:
{"target_idx": <int or null>, "confidence": "high"|"medium"|"low",
 "miss_kind": null or "extraction_miss"|"media_content"|"implicit_framing"|"other",
 "miss_detail": "<if null target: one sentence — if extraction_miss, quote the tweet span carrying the disputed claim>",
 "reason": "<one sentence>"}"""


MATCH_HASH = prompt_hash(MATCH_SYS)


def _chat(sys_prompt: str, user: str, retries: int = 4) -> dict:
    payload = {"model": MODEL, "temperature": 0, "max_tokens": 500,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user}]}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {_key()}",
                                          "Content-Type": "application/json"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode())
                m = re.search(r"\{.*\}", d["choices"][0]["message"]["content"], re.S)
                out = json.loads(m.group(0)) if m else {}
                out["_cost"] = (d.get("usage") or {}).get("estimated_cost") or 0
                return out
        except Exception:
            time.sleep(3 * 2 ** a)
    return {}


def match() -> None:
    posts = pl.read_parquet(CORP / "cn_false_smoke_posts.parquet")
    claims = pl.read_parquet(CORP / "cn_false_smoke_claims.parquet")
    notes = pl.read_parquet(CN / "cn_gold.parquet").select(["noteId", "summary"] + BOXES)
    posts = posts.join(notes, on="noteId", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    fh = open(OUT / "match.jsonl", "w")
    cost = 0.0
    for p in posts.iter_rows(named=True):
        pc = claims.filter(pl.col("post_id") == p["post_id"])
        clist = pc["claim"].to_list()
        cids = pc["claim_id"].to_list()
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(clist)) or "(no claims extracted)"
        user = (f"TWEET (@{p['handle']}):\n{p['text']}\n\nCOMMUNITY NOTE:\n{p['summary']}\n\n"
                f"EXTRACTED CLAIMS:\n{numbered}")
        out = _chat(MATCH_SYS, user)
        cost += out.pop("_cost", 0)
        idx = out.get("target_idx")
        rec = {"post_id": p["post_id"], "noteId": p["noteId"],
               "note_class": _note_class(p), "mixed_signal": p["mixed_signal"],
               "n_claims": len(clist),
               "target_idx": idx,
               "target_claim_id": cids[idx] if isinstance(idx, int) and 0 <= idx < len(cids) else None,
               "target_claim": clist[idx] if isinstance(idx, int) and 0 <= idx < len(clist) else None,
               "confidence": out.get("confidence"), "miss_kind": out.get("miss_kind"),
               "miss_detail": out.get("miss_detail"), "reason": out.get("reason"),
               "prompt_hash": MATCH_HASH,
               "tweet": p["text"], "note": p["summary"]}
        fh.write(json.dumps(rec) + "\n")
        print(f"  {p['post_id']}: target={rec['target_claim_id']} conf={rec['confidence']} "
              f"miss={rec['miss_kind']}", flush=True)
    fh.close()
    print(f"match done | ${cost:.4f}", flush=True)


# ---------------------------------------------------------------- stage 2: run

def _snowflake_date(tid: str) -> str | None:
    try:
        ms = (int(tid) >> 22) + 1288834974657
        return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None


def run(workers: int) -> None:
    matches = [json.loads(l) for l in open(OUT / "match.jsonl")]
    claims = pl.read_parquet(CORP / "cn_false_smoke_claims.parquet")
    rows = []
    for m in matches:
        if not m["target_claim_id"]:
            continue
        c = claims.filter(pl.col("claim_id") == m["target_claim_id"]).row(0, named=True)
        rows.append({
            "review_url": c["claim_id"], "claim_text": c["claim"],
            "publisher_site": "x.com",
            "claim_date": _snowflake_date(c["post_id"]),
            "claim_type": c["type"], "topic": c.get("topic"),
            "x_context": None, "context_ok": False,
            "resolution_status": "native", "claim_resolved": None,
            "veracity": None, "rating_subtype": None,
            "review_date": None, "x_date": None, "yr": None,
            "screen_verdict": "unscreened", "screen_leak": False,
            "note_class": m["note_class"], "mixed_signal": m["mixed_signal"],
            "post_id": c["post_id"], "noteId": m["noteId"],
            "attribution_form": c["attribution_form"], "media_locus": c["media_locus"],
            "match_confidence": m["confidence"]})
    out = OUT / "verify.jsonl"
    seen = set()
    if out.exists():
        seen = {json.loads(l)["review_url"] for l in open(out)}
    todo = [r for r in rows if r["review_url"] not in seen]
    print(f"{len(todo)}/{len(rows)} recovered claims to verify | cap ${BUDGET_CAP}", flush=True)
    ng = newsguard_score_map()
    lock = threading.Lock()
    budget = {"spent": 0.0, "done": 0, "cached_tok": 0, "prompt_tok": 0, "stop": False}
    t0 = time.time()
    fh = open(out, "a")

    def work(row):
        if budget["stop"]:
            return
        try:
            rec = run_claim(row, ng, budget, {})
        except Exception as e:
            with lock:
                budget["done"] += 1
                print(f"  [claim-failed] {row['review_url']}: {type(e).__name__}: {e}", flush=True)
            return
        for k in ("note_class", "mixed_signal", "post_id", "noteId",
                  "attribution_form", "media_locus", "match_confidence"):
            rec[k] = row[k]
        with lock:
            budget["spent"] += rec["cost"]
            budget["done"] += 1
            fh.write(json.dumps(rec) + "\n")
            n, el = budget["done"], (time.time() - t0) / 60
            print(f"  {n}/{len(todo)} {row['note_class']} | ${budget['spent']:.3f} "
                  f"| {el:.1f}m {n/max(el,.01):.1f}/min", flush=True)
            if budget["spent"] / n * len(todo) > BUDGET_CAP and n >= 5:
                print("  BUDGET ABORT", flush=True)
                budget["stop"] = True

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"done {budget['done']} | ${budget['spent']:.4f} | {(time.time()-t0)/60:.1f}m", flush=True)


# ---------------------------------------------------------------- stage 3: report

def report() -> None:
    m7 = json.load(open(E1_METRICS))["overall"]
    w7, thr7 = m7["weights"], m7["threshold"]
    m3 = json.load(open(E1_METRICS_3VOICE))["overall"]
    w3 = m3["weights"]
    thr3 = sum(m3["thresholds_by_fold"]) / len(m3["thresholds_by_fold"])
    recs = [json.loads(l) for l in open(OUT / "verify.jsonl")]
    print(f"thresholds: s3 {thr3:.3f} | s7 {thr7:.3f} (2% budget)")
    rows = []
    for r in recs:
        flags = Counter()
        cited = []
        for d in r["results"]:
            f = (d.get("read") or {}).get("direction")
            if f in FLAGS7:
                flags[f] += 1
            if f in ("5", "4", "2", "1"):
                sents = d.get("sents") or []
                ev = (d.get("read") or {}).get("evidence") or []
                ids = d.get("sent_ids") or []
                pick = [sents[ids.index(i)] for i in ev if i in ids][:1] or sents[:1]
                cited.append((f, d["domain"], (pick[0] if pick else "")[:160]))
        # Both fits pad every claim to PAD_TO slots with silent reads, so scoring
        # pads too — the weights are only valid under that convention. `flags` is
        # left unpadded because it is also what the row reports.
        pad = max(0, PAD_TO - sum(flags.values()))
        s3 = (flags["5"] + flags["4"]) * w3["n_t"] \
            + (flags["1"] + flags["2"]) * w3["n_f"] \
            + (flags["3"] + flags["X"] + flags["I"] + pad) * w3["n_e"]
        # The shipped fit is SIX-flag (read-v6.1, 2026-09-14): a read-v5 "3" folds
        # into "X". w7 has no "3" key.
        s7 = (sum(flags[f] * w7[f] for f in FLAGS7 if f != "3")
              + flags["3"] * w7["X"] + pad * w7["I"])
        rows.append({**{k: r[k] for k in ("review_url", "claim_text", "note_class",
                                          "mixed_signal", "match_confidence")},
                     "flags": dict(flags), "s3": round(s3, 2), "s7": round(s7, 2),
                     "flag3": s3 < thr3, "flag7": s7 < thr7, "cited": cited})
    print(f"\n== verdict match (note says misleading -> expect flag) n={len(rows)} ==")
    for cls in ("factual_error", "unverified_as_fact", "mixed_true_core", "other"):
        sub = [r for r in rows if r["note_class"] == cls]
        if not sub:
            continue
        f7 = sum(r["flag7"] for r in sub)
        f3 = sum(r["flag3"] for r in sub)
        print(f"  {cls}: n={len(sub)} | flagged s7 {f7} ({f7/len(sub):.0%}) | "
              f"s3 {f3} ({f3/len(sub):.0%}) | median s7 "
              f"{sorted(r['s7'] for r in sub)[len(sub)//2]:+.1f}")
    print("\n== per-claim ==")
    for r in sorted(rows, key=lambda r: r["s7"]):
        print(f"  s7={r['s7']:+6.1f} {'FLAG' if r['flag7'] else 'pass'} "
              f"[{r['note_class']}{'/mixed' if r['mixed_signal'] else ''}] {r['claim_text'][:100]}")
        if not r["flag7"]:
            for f, dom, sent in r["cited"][:2]:
                print(f"       flag {f} {dom}: {sent}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", action="store_true")
    ap.add_argument("--match", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.dist:
        dist()
    if a.match:
        match()
    if a.run:
        run(a.workers)
    if a.report:
        report()
