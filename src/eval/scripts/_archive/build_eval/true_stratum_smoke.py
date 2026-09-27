"""TRUE-stratum 50+50 smoke (Log-Odds Sprint C2) — mirror audit before the full TRUE build.

Acquitted arm (50 L3 posts): extraction (--voice user) -> REJECTED-note target
matching (note text OFFLINE ONLY, same matcher as c2_audit) -> acquitted gate
("if the rejected note WERE accurate, would this claim be false?") -> instrument
verification of acquitted-clean claims. Wire arm (50 fit-eligible claims): direct
verification. Design: docs/tweet_fit_corpus.md "TRUE-stratum draw".

  uv run python -m eval.scripts.build_eval.true_stratum_smoke --draw     # $0 + raw-zip notes
  uv run python -m eval.scripts.build_eval.true_stratum_smoke --extract  # chain, acq arm
  uv run python -m eval.scripts.build_eval.true_stratum_smoke --match    # rejected-note matcher
  uv run python -m eval.scripts.build_eval.true_stratum_smoke --gate     # acquitted gate
  uv run python -m eval.scripts.build_eval.true_stratum_smoke --register # $0 BoW check
  uv run python -m eval.scripts.build_eval.true_stratum_smoke --run      # paid verify (hold if
                                                                        # another search job runs)
  uv run python -m eval.scripts.build_eval.true_stratum_smoke --report

Outputs: eval/data/tweet_corpus/acq_smoke_* (chain files), eval/data/urn_runs/true_smoke/.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import subprocess
import sys
import threading
import time
import zipfile
from collections import Counter
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from pipeline.search import newsguard_score_map  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import run_claim  # noqa: E402
from eval.scripts.build_eval.c2_audit import MATCH_SYS, _chat, _snowflake_date  # noqa: E402
from eval.scripts.build_eval.cn_false_stratum import screen, MODEL  # noqa: E402

CN = SRC / "eval/data/community_notes"
CORP = SRC / "eval/data/tweet_corpus"
OUT = SRC / "eval/data/urn_runs/true_smoke"
E1_METRICS = SRC / "eval/data/urn_runs/e1_ctx/headline_metrics.json"
GRADED_METRICS = SRC / "eval/data/urn_runs/e1_ctx/graded_metrics.json"
E1_RESULTS = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
SEED = 20260824
BUDGET_CAP = 1.00
FLAGS7 = ["5", "4", "3", "2", "1", "X", "I"]
MIS = "MISINFORMED_OR_POTENTIALLY_MISLEADING"
CRNH = "CURRENTLY_RATED_NOT_HELPFUL"

GATE_SYS = """You judge one claim extracted from a tweet against a Community Note that was proposed on that tweet and REJECTED by raters. Hypothetical: take the note's content as if it were accurate, and decide what it would establish about THIS claim's literal truth.

Bands, exactly one:
- "acquitted_clean": if the note were accurate, the claim's central proposition, as stated, would be false. The note directly attacks what this claim asserts — so the note's rejection is an acquittal of THIS claim.
- "still_contested": the note bears on this claim but only partially — it would correct framing, missing context, an implication, or a secondary detail (a date, an aside), or it half-concedes the core. Rejection of such a note does not establish the claim's truth.
- "off_target": the note does not bear on this claim's truth at all — it targets an image or video, who posted it, a different claim in the tweet, tone, or something the claim does not say. This claim was never really accused.

Discipline: judge the claim EXACTLY as written, in isolation. "Directly attacks" means the note asserts that the claim's own proposition did not happen or is not so.

Return only JSON:
{"band": "acquitted_clean"|"still_contested"|"off_target", "why": "<one sentence citing the decisive note content>"}"""


# ------------------------------------------------------------------ stage: draw

def _read_raw_notes(want_ids: set[str]) -> pl.DataFrame:
    """Stream the raw CN dump; keep MIS-class notes on the wanted tweets, join status."""
    rows = []
    for zp in sorted((CN / "raw").glob("*-notes-*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    rd = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
                    for r in rd:
                        if r.get("tweetId") in want_ids and r.get("classification") == MIS:
                            rows.append({k: r.get(k) for k in
                                         ("noteId", "tweetId", "summary", "createdAtMillis")})
    notes = pl.DataFrame(rows)
    st = []
    for zp in sorted((CN / "raw").glob("*-noteStatusHistory-*.zip")):
        keep = set(notes["noteId"].to_list())
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    rd = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
                    for r in rd:
                        if r.get("noteId") in keep:
                            st.append({"noteId": r["noteId"],
                                       "currentStatus": r.get("currentStatus"),
                                       "lockedStatus": r.get("lockedStatus")})
    return notes.join(pl.DataFrame(st), on="noteId", how="left")


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = pl.read_parquet(CORP / "acquitted_draw_manifest.parquet")
    hyd = {}
    with open(CORP / "acquitted_hydrated.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if str(d.get("code")) == "200" and d.get("text") and d.get("lang") == "en":
                hyd[str(d["tweetId"])] = d
    pool = man.filter(pl.col("tweetId").is_in(list(hyd)))
    l4 = pool.filter(pl.col("l4_robust")).sample(n=min(8, pool.filter(pl.col("l4_robust")).height),
                                                 seed=SEED)
    rest = pool.filter(~pl.col("tweetId").is_in(l4["tweetId"].implode())) \
               .sample(n=50 - l4.height, seed=SEED)
    drawn = pl.concat([l4, rest])
    posts = pl.DataFrame([{
        "post_id": t, "cell": "acquitted", "domain": "x.com",
        "handle": hyd[t].get("author") or "",
        "url": f"https://x.com/{hyd[t].get('author') or '_'}/status/{t}",
        "created_at": hyd[t].get("created_at"), "lang": "en", "text": hyd[t]["text"],
        "n_images": 0, "image_urls": [], "n_videos": 0, "video_urls": [],
    } for t in drawn["tweetId"].to_list()])
    posts = posts.join(drawn.rename({"tweetId": "post_id"})
                       .select(["post_id", "n_mis_crnh", "n_notmis_any", "l4_robust"]),
                       on="post_id", how="left")
    posts.write_parquet(CORP / "acq_smoke_posts.parquet")
    print(f"acq draw: {posts.height} posts (l4_robust {posts['l4_robust'].sum()}) "
          f"from EN-hydrated pool {pool.height}", flush=True)

    print("parsing raw dump for rejected notes (one-time, ~min)...", flush=True)
    notes = _read_raw_notes(set(posts["post_id"].to_list()))
    rej = notes.filter((pl.col("currentStatus") == CRNH) | (pl.col("lockedStatus") == CRNH))
    rej = rej.sort("createdAtMillis", descending=True)
    primary = rej.unique(subset="tweetId", keep="first")
    rej.write_parquet(OUT / "acq_rejected_notes_all.parquet")
    primary.write_parquet(OUT / "acq_rejected_notes.parquet")
    cov = posts.filter(pl.col("post_id").is_in(primary["tweetId"].implode())).height
    print(f"rejected-note coverage: {cov}/50 posts have a CRNH misleading note "
          f"({rej.height} total rejected notes)", flush=True)

    # wire arm draw
    scr = pl.read_parquet(SRC / "eval/data/urn_runs/wire_audit/screens.parquet")
    vi = pl.read_parquet(CORP / "wire_true_verify_input.parquet")
    elig = scr.filter(pl.col("fit_eligible")).join(
        vi.select(["post_id", "claim", "claim_id", "checkworthy", "type", "created_at"]),
        on=["post_id", "claim"], how="inner").filter(pl.col("checkworthy"))
    wire = elig.sample(n=50, seed=SEED)
    wire.write_parquet(OUT / "wire_smoke_claims.parquet")
    print(f"wire draw: 50/{elig.height} fit-eligible+checkworthy claims", flush=True)


# ------------------------------------------------------------------ stage: extract

def extract(concurrency: int) -> None:
    posts_path = CORP / "acq_smoke_posts.parquet"
    run = lambda cmd: subprocess.run(cmd, cwd=SRC, check=True)
    tag = "acq_smoke"
    run([sys.executable, "eval/scripts/claim_sourcing/extract_tweet_claims.py",
         "-i", str(posts_path), "-o", str(CORP / f"{tag}_extracted.parquet"),
         "--html", str(CORP / f"{tag}_extracted.html"), "--no-images",
         "--model", MODEL, "--voice", "user", "--concurrency", str(concurrency)])
    run([sys.executable, "eval/scripts/claim_sourcing/normalize_tweet_claims.py",
         "--payload", str(CORP / f"{tag}_extracted.html"),
         "-o", str(CORP / f"{tag}_normalized.html"), "--model", MODEL,
         "--voice", "user", "--concurrency", str(concurrency)])
    run([sys.executable, "eval/scripts/claim_sourcing/build_verify_input.py",
         "--normalized", str(CORP / f"{tag}_normalized.html"), "--posts", str(posts_path),
         "-o", str(CORP / f"{tag}_verify_input.parquet"),
         "--extract-model", MODEL, "--normalize-model", MODEL])
    posts = pl.read_parquet(posts_path).with_columns([
        pl.lit(None, dtype=pl.Utf8).alias("noteId"),
        pl.lit(None, dtype=pl.Utf8).alias("cluster_id"),
        pl.lit(None, dtype=pl.UInt32).alias("cluster_size"),
        pl.lit(None, dtype=pl.Utf8).alias("note_date"),
        pl.lit(False).alias("mixed_signal")])
    screen(CORP / f"{tag}_verify_input.parquet", posts, CORP / f"{tag}_claims.parquet")


# ------------------------------------------------------------------ stage: match

def match() -> None:
    posts = pl.read_parquet(CORP / "acq_smoke_posts.parquet")
    notes = pl.read_parquet(OUT / "acq_rejected_notes.parquet") \
        .rename({"tweetId": "post_id", "summary": "note"})
    posts = posts.join(notes.select(["post_id", "note"]), on="post_id", how="left")
    claims = pl.read_parquet(CORP / "acq_smoke_claims.parquet")
    fh = open(OUT / "match.jsonl", "w")
    cost = 0.0
    for p in posts.iter_rows(named=True):
        if not p["note"]:
            fh.write(json.dumps({"post_id": p["post_id"], "no_note": True}) + "\n")
            continue
        pc = claims.filter(pl.col("post_id") == p["post_id"])
        clist, cids = pc["claim"].to_list(), pc["claim_id"].to_list()
        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(clist)) or "(no claims extracted)"
        o = _chat(MATCH_SYS, f"TWEET (@{p['handle']}):\n{p['text']}\n\n"
                             f"COMMUNITY NOTE:\n{p['note']}\n\nEXTRACTED CLAIMS:\n{numbered}")
        cost += o.pop("_cost", 0)
        idx = o.get("target_idx")
        ok = isinstance(idx, int) and 0 <= idx < len(clist)
        fh.write(json.dumps({
            "post_id": p["post_id"], "n_claims": len(clist),
            "target_claim_id": cids[idx] if ok else None,
            "target_claim": clist[idx] if ok else None,
            "confidence": o.get("confidence"), "miss_kind": o.get("miss_kind"),
            "miss_detail": o.get("miss_detail"), "reason": o.get("reason"),
            "l4_robust": p["l4_robust"], "tweet": p["text"], "note": p["note"]}) + "\n")
        print(f"  {p['post_id']}: {'HIT' if ok else o.get('miss_kind')}", flush=True)
    fh.close()
    print(f"match done | ${cost:.4f}", flush=True)


def gate() -> None:
    recs = [json.loads(l) for l in open(OUT / "match.jsonl")]
    fh = open(OUT / "gate.jsonl", "w")
    cost = 0.0
    for r in recs:
        if r.get("no_note") or not r.get("target_claim"):
            continue
        o = _chat(GATE_SYS, f"CLAIM:\n{r['target_claim']}\n\nREJECTED COMMUNITY NOTE:\n{r['note']}")
        cost += o.pop("_cost", 0)
        fh.write(json.dumps({"post_id": r["post_id"], "claim": r["target_claim"],
                             "band": o.get("band"), "why": o.get("why")}) + "\n")
    fh.close()
    print(f"gate done | ${cost:.4f} | "
          f"{Counter(json.loads(l)['band'] for l in open(OUT / 'gate.jsonl'))}", flush=True)


# ------------------------------------------------------------------ stage: register

def register() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    import numpy as np
    cn = [json.loads(l) for l in open(SRC / "eval/data/urn_runs/c2_audit/ab_gate_new.jsonl")]
    cn_false = [r["claim"] for r in cn if r["band"] == "false"]
    acq = pl.read_parquet(CORP / "acq_smoke_claims.parquet")
    acq_claims = acq.filter(pl.col("checkworthy"))["claim"].to_list()
    X_txt = acq_claims + cn_false
    y = np.array([1] * len(acq_claims) + [0] * len(cn_false))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(X_txt)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    print(f"register-blind AUC (acquitted vs CN-false, 5-fold oof): "
          f"{roc_auc_score(y, p):.3f} | n_acq {len(acq_claims)} n_false {len(cn_false)}",
          flush=True)


# ------------------------------------------------------------------ stage: run

def run(workers: int) -> None:
    rows = []
    gates = {json.loads(l)["post_id"]: json.loads(l) for l in open(OUT / "gate.jsonl")}
    matches = {json.loads(l)["post_id"]: json.loads(l) for l in open(OUT / "match.jsonl")}
    claims = pl.read_parquet(CORP / "acq_smoke_claims.parquet")
    # verify ALL matched claims tagged by band (n=1 acquitted_clean alone is
    # uninterpretable; the report cells split by band)
    for pid, g in gates.items():
        m = matches[pid]
        c = claims.filter(pl.col("claim_id") == m["target_claim_id"]).row(0, named=True)
        rows.append({"review_url": c["claim_id"], "claim_text": c["claim"],
                     "publisher_site": "x.com", "claim_date": _snowflake_date(pid),
                     "claim_type": c["type"], "topic": c.get("topic"),
                     "x_context": None, "context_ok": False,
                     "resolution_status": "native", "claim_resolved": None,
                     "veracity": None, "rating_subtype": None,
                     "review_date": None, "x_date": None, "yr": None,
                     "screen_verdict": "unscreened", "screen_leak": False,
                     "arm": "acquitted", "post_id": pid,
                     "band": g["band"], "l4_robust": m.get("l4_robust")})
    wire = pl.read_parquet(OUT / "wire_smoke_claims.parquet")
    for c in wire.iter_rows(named=True):
        rows.append({"review_url": c["claim_id"], "claim_text": c["claim"],
                     "publisher_site": "x.com", "claim_date": _snowflake_date(c["post_id"]),
                     "claim_type": c["type"], "topic": None,
                     "x_context": None, "context_ok": False,
                     "resolution_status": "native", "claim_resolved": None,
                     "veracity": None, "rating_subtype": None,
                     "review_date": None, "x_date": None, "yr": None,
                     "screen_verdict": "unscreened", "screen_leak": False,
                     "arm": "wire", "post_id": c["post_id"], "band": None, "l4_robust": None})
    out = OUT / "verify.jsonl"
    seen = {json.loads(l)["review_url"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["review_url"] not in seen]
    print(f"{len(todo)}/{len(rows)} claims to verify "
          f"(acq {sum(r['arm']=='acquitted' for r in todo)}, "
          f"wire {sum(r['arm']=='wire' for r in todo)}) | cap ${BUDGET_CAP}", flush=True)
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
        for k in ("arm", "post_id", "band", "l4_robust"):
            rec[k] = row[k]
        with lock:
            budget["spent"] += rec["cost"]
            budget["done"] += 1
            fh.write(json.dumps(rec) + "\n")
            n, el = budget["done"], (time.time() - t0) / 60
            print(f"  {n}/{len(todo)} {row['arm']} | ${budget['spent']:.3f} | "
                  f"{el:.1f}m {n/max(el,.01):.1f}/min", flush=True)
            if budget["spent"] / n * len(todo) > BUDGET_CAP and n >= 5:
                print("  BUDGET ABORT", flush=True)
                budget["stop"] = True

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"done {budget['done']} | ${budget['spent']:.4f} | {(time.time()-t0)/60:.1f}m", flush=True)


# ------------------------------------------------------------------ stage: report

def report() -> None:
    w3 = json.load(open(E1_METRICS))["overall"]
    g = json.load(open(GRADED_METRICS))
    w7, thr7 = g["weights"], g["recall_at_2pct_fpr"]["threshold"]
    thr3 = w3["threshold"]
    recs = [json.loads(l) for l in open(OUT / "verify.jsonl")]

    # E1 gold-TRUE comparison mix (fit population, media-axis screened as in headline)
    e1_flags = Counter()
    n_e1 = 0
    with open(E1_RESULTS) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("veracity") or 0) >= 4:
                n_e1 += 1
                for d in r["results"]:
                    fl = (d.get("read") or {}).get("direction")
                    if fl in FLAGS7:
                        e1_flags[fl] += 1
    e1_tot = sum(e1_flags.values())

    print(f"thresholds: s3 {thr3:.3f} | s7 {thr7:.3f}")
    for arm in ("acquitted", "wire"):
        sub = [r for r in recs if r["arm"] == arm]
        if not sub:
            continue
        flags = Counter()
        rows = []
        for r in sub:
            fl = Counter()
            for d in r["results"]:
                x = (d.get("read") or {}).get("direction")
                if x in FLAGS7:
                    fl[x] += 1
                    flags[x] += 1
            s3 = (fl["5"] + fl["4"]) * w3["weights"]["n_t"] \
                + (fl["1"] + fl["2"]) * w3["weights"]["n_f"] \
                + (fl["3"] + fl["X"] + fl["I"]) * w3["weights"]["n_e"]
            s7 = sum(fl[x] * w7[x] for x in FLAGS7)
            rows.append((s7, s3, r))
        tot = sum(flags.values())
        sup = (flags["5"] + flags["4"]) / tot
        ref = (flags["1"] + flags["2"]) / tot
        sil = 1 - sup - ref
        f7 = sum(s7 < thr7 for s7, _, _ in rows)
        f3 = sum(s3 < thr3 for _, s3, _ in rows)
        med = sorted(s7 for s7, _, _ in rows)[len(rows) // 2]
        print(f"\n== {arm} n={len(rows)} | docs {tot} | support {sup:.1%} refute {ref:.1%} "
              f"silent {sil:.1%} | median s7 {med:+.1f} | "
              f"below-thr s7 {f7} ({f7/len(rows):.0%}) s3 {f3} ({f3/len(rows):.0%})")
        for s7, s3, r in sorted(rows, key=lambda x: x[0]):
            mark = "FLAG" if s7 < thr7 else "pass"
            print(f"  s7={s7:+6.1f} {mark} {r['claim_text'][:110]}")
            if s7 < thr7 or s3 < thr3:
                for d in r["results"]:
                    x = (d.get("read") or {}).get("direction")
                    if x in ("1", "2"):
                        sents = d.get("sents") or []
                        ev = (d.get("read") or {}).get("evidence") or []
                        ids = d.get("sent_ids") or []
                        pick = [sents[ids.index(i)] for i in ev if i in ids][:1] or sents[:1]
                        print(f"       flag {x} {d['domain']}: {(pick[0] if pick else '')[:170]}")
    e1_sup = (e1_flags["5"] + e1_flags["4"]) / e1_tot
    e1_ref = (e1_flags["1"] + e1_flags["2"]) / e1_tot
    print(f"\nE1 gold-TRUE reference (n={n_e1} claims): support {e1_sup:.1%} "
          f"refute {e1_ref:.1%} silent {1-e1_sup-e1_ref:.1%} | "
          f"below-thr reference: s3 fpr {w3['fpr']:.1%} | s7 2% budget")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for s in ("draw", "extract", "match", "gate", "register", "run", "report"):
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.draw:
        draw()
    if a.extract:
        extract(a.concurrency)
    if a.match:
        match()
    if a.gate:
        gate()
    if a.register:
        register()
    if a.run:
        run(a.workers)
    if a.report:
        report()
