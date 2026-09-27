"""CommunityFact probe — run CF claims through the E1 instrument UNMODIFIED (sprint C5b).

Purpose (docs/logodds_sprint.md C5): three numbers before any use of CF —
  1. label stress: claims whose evidence contradicts the label -> Daniel's read list;
  2. transfer: per-class flag mix, CF vs E1;
  3. multilingual viability: per-language retrieval health (NOTE: pipeline/search.py
     has NO locale parameter — non-EN claims hit default-locale Google; that gap is
     itself a probe deliverable, do not paper over it here).

Adapter follows tweet_urn_run.py: imports run_claim from evidence_urn_run, supplies
rows in the E1 column contract. Ceiling = snowflake post date (fallback: noteTimeStamp
date); review/x dates null so ceiling_src degrades to "claim_date". CF's evidenceURLs
are note-cited ordinary sources: NOT excluded from retrieval, but overlap is RECORDED
per claim (retrieval-validity signal).

  uv run python -m eval.scripts.build_eval.cf_probe --smoke            # 24 claims
  uv run python -m eval.scripts.build_eval.cf_probe --report           # analyze only

Output: eval/data/urn_runs/cf_probe/smoke.jsonl (resumable by claimId).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline.search import newsguard_score_map
from eval.scripts.build_eval.evidence_urn_run import (
    QUERY_PROMPT_V, READ_PROMPT_V, run_claim)

CSV = Path("eval/data/communityfact_v1_test.csv")
OUTDIR = Path("eval/data/urn_runs/cf_probe")
E1_METRICS = Path("eval/data/urn_runs/e1_ctx/headline_metrics.json")
GRADED_METRICS = Path("eval/data/urn_runs/e1_ctx/graded_metrics.json")
SEED = 707
SMOKE_EN = 8          # per label
SMOKE_OTHER = 1       # per label per language (es, pt, fr, ja)
BUDGET_CAP = 0.20     # hard USD cap for the smoke
FULL_CAP = 0.75       # full probe: 3x the ~$0.25 quote
# full-probe stratification (per label): EN force-includes attribution-tagged
# claims so the Rule-6 suspect harvest has n (Daniel 2026-08-22)
FULL_PLAN = {"en": (60, 15), "fr": (20, 0), "es": (10, 0), "pt": (10, 0), "ja": (10, 0)}

# Attribution-form TAG (CF Rule 6 labels "X claims Y" True even when Y is false, so
# these are expected label-suspects for OUR axis). Crude multilingual surface regex —
# a tag for stratified analysis, never a gate.
ATTRIB_RE = re.compile(
    r"\b(said|says|saying|claims?|claimed|wrote|writes|stated|states|announced|"
    r"declared|posted|tweeted|argued|alleges?|alleged|according to|"
    r"dijo|afirm[oó]|declar[oó]|asegur[oó]|según|"
    r"disse|afirmou|declarou|segundo|"
    r"a d[ée]clar[ée]|a dit|a affirm[ée]|selon|affirme)\b"
    r"|と述べ|と主張|と発言|によると|と書い", re.IGNORECASE)

# Media-locus TAG: claims whose checkable core is media authenticity/provenance —
# the analogue of E1's media_authenticity axis, reported as a separate cell, never
# counted in headline suspects. Calibration note (2026-08-22): only 3/3,578 CF
# claims match; CF's claim extraction normalizes this axis away (the 25 media-WORD
# claims are about media as subject matter — copyright, video games — not "this
# video shows X"). Kept for honesty of the cell.
MEDIA_RE = re.compile(
    r"\b(video|photo|image|footage|clip|picture|audio|recording|screenshot)s?\b"
    r".{0,40}\b(shows?|showing|depicts?|real|fake|authentic|genuine|AI|generated|"
    r"doctored|edited|manipulated|old|recycled)\b"
    r"|\b(AI[- ]generated|deepfake|photoshopped|digitally (altered|created))\b"
    r"|\b(this|the) (video|photo|image|footage|clip|picture)\b"
    r"|\b(v[ií]d[ée]o|foto|imagen|imagem)\b.{0,40}\b(muestra|mostra|montre|montrant|"
    r"real|falso|falsa|truqu[ée]e?|antig[oa]|ancienne?)\b"
    r"|この(動画|画像|写真)|(動画|画像|写真)は(本物|偽物|加工|AI)", re.IGNORECASE)


def snowflake_date(tid: str) -> str | None:
    try:
        ms = (int(tid.split("_", 1)[1]) >> 22) + 1288834974657
        d = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
        return f"{d:%Y-%m-%d}" if d.year >= 2007 else None
    except Exception:
        return None


def note_date(ts: str) -> str | None:
    try:
        return f"{datetime.strptime(ts, '%a %b %d %H:%M:%S %z %Y'):%Y-%m-%d}"
    except Exception:
        return None


def load_cf() -> pl.DataFrame:
    df = pl.read_csv(CSV, schema_overrides={"tweetId": pl.Utf8, "noteId": pl.Utf8})
    return df.with_columns([
        pl.col("tweetId").map_elements(snowflake_date, return_dtype=pl.Utf8).alias("post_date"),
        pl.col("noteTimeStamp").map_elements(note_date, return_dtype=pl.Utf8).alias("note_date"),
        pl.col("claim").map_elements(lambda c: bool(ATTRIB_RE.search(c)),
                                     return_dtype=pl.Boolean).alias("attrib_tag"),
        pl.col("claim").map_elements(lambda c: bool(MEDIA_RE.search(c)),
                                     return_dtype=pl.Boolean).alias("media_tag"),
        pl.col("claim").str.split(" ").list.len().alias("wc"),
    ])


def smoke_draw(df: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for lang, per_label in [("en", SMOKE_EN)] + [(l, SMOKE_OTHER) for l in ("es", "pt", "fr", "ja")]:
        for lab in (True, False):
            sub = df.filter((pl.col("language") == lang) & (pl.col("label") == lab)
                            & pl.col("post_date").is_not_null()).sort("claimId")
            parts.append(sub.sample(min(per_label, sub.height), seed=SEED))
    return pl.concat(parts)


def full_draw(df: pl.DataFrame) -> pl.DataFrame:
    """~220 claims per FULL_PLAN, smoke claims excluded, EN attribution force-include."""
    smoke_ids = json.load(open(OUTDIR / "smoke.sample.json"))["claim_ids"]
    pool = df.filter(~pl.col("claimId").is_in(smoke_ids)
                     & pl.col("post_date").is_not_null())
    parts = []
    for lang, (per_label, n_attrib) in FULL_PLAN.items():
        for lab in (True, False):
            sub = pool.filter((pl.col("language") == lang)
                              & (pl.col("label") == lab)).sort("claimId")
            att = sub.filter(pl.col("attrib_tag")).sample(
                min(n_attrib, sub.filter(pl.col("attrib_tag")).height), seed=SEED)
            rest = sub.filter(~pl.col("claimId").is_in(att["claimId"].to_list())).sample(
                min(per_label - att.height, sub.height - att.height), seed=SEED)
            parts.append(pl.concat([att, rest]))
    return pl.concat(parts)


def adapt(r: dict) -> dict:
    """CF row -> the E1 column contract run_claim hard-indexes (tweet_urn_run pattern)."""
    return {**r,
            "review_url": r["claimId"],
            "claim_text": r["claim"],
            # origin = X itself; x.com is UGC-blocked anyway, this keeps the
            # origin-exclusion semantics of the contract explicit.
            "publisher_site": "x.com",
            "claim_date": r["post_date"] or r["note_date"],
            "claim_type": "attribution" if r["attrib_tag"] else "assertion",
            "topic": None,
            "x_context": None, "context_ok": False,
            "resolution_status": "native", "claim_resolved": None,
            "veracity": None, "rating_subtype": None,
            "review_date": None, "x_date": None, "yr": None,
            "screen_verdict": "unscreened", "screen_leak": False}


def run_batch(workers: int, full: bool) -> None:
    name = "probe" if full else "smoke"
    cap = FULL_CAP if full else BUDGET_CAP
    draw = full_draw(load_cf()) if full else smoke_draw(load_cf())
    rows = [adapt(r) for r in draw.iter_rows(named=True)]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{name}.jsonl"
    (OUTDIR / f"{name}.sample.json").write_text(json.dumps(
        {"seed": SEED, "n": len(rows), "claim_ids": [r["review_url"] for r in rows]}, indent=1))
    seen = set()
    if out.exists():
        for line in open(out):
            try:
                seen.add(json.loads(line)["review_url"])
            except Exception:
                pass
    todo = [r for r in rows if r["review_url"] not in seen]
    print(f"{len(todo)}/{len(rows)} claims to run -> {out} | cap ${cap} | "
          f"{workers} workers | prompts {QUERY_PROMPT_V}/{READ_PROMPT_V}", flush=True)

    ng_scores = newsguard_score_map()
    lock = threading.Lock()
    budget = {"spent": 0.0, "done": 0, "cached_tok": 0, "prompt_tok": 0, "stop": False}
    t0 = time.time()
    fh = open(out, "a")

    def work(row):
        if budget["stop"]:
            return
        try:
            rec = run_claim(row, ng_scores, budget, {})
        except Exception as e:
            with lock:
                budget["done"] += 1
                print(f"  [claim-failed] {row['review_url']}: {type(e).__name__}: {e}", flush=True)
            return
        for k in ("label", "language", "domain", "attrib_tag", "media_tag", "wc",
                  "post_date", "note_date", "evidenceURLs", "tweetId", "noteId"):
            rec[k] = row.get(k)
        with lock:
            budget["spent"] += rec["cost"]
            budget["done"] += 1
            fh.write(json.dumps(rec) + "\n")
            n = budget["done"]
            proj = budget["spent"] / n * len(todo)
            print(f"  {n}/{len(todo)} {row['language']} lab={row['label']} "
                  f"| spent ${budget['spent']:.3f} proj ${proj:.2f} "
                  f"| {(time.time()-t0)/60:.1f}m {n/max((time.time()-t0)/60,.01):.1f}/min", flush=True)
            if proj > cap and n >= 6:
                print(f"  BUDGET ABORT: ${proj:.2f} > ${cap}", flush=True)
                budget["stop"] = True

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"done {budget['done']} | ${budget['spent']:.4f} | {(time.time()-t0)/60:.1f}m", flush=True)


# ---------------------------------------------------------------- analysis

FLAGS7 = ["5", "4", "3", "2", "1", "X", "I"]


def _dom(u: str) -> str:
    d = urlparse(u).netloc.lower()
    return d[4:] if d.startswith("www.") else d


def report(name: str = "smoke") -> None:
    recs = [json.loads(l) for l in open(OUTDIR / f"{name}.jsonl")]
    w3 = json.load(open(E1_METRICS))["overall"]["weights"]
    g = json.load(open(GRADED_METRICS))
    w7 = g["weights"]

    rows = []
    for r in recs:
        if r.get("excluded"):
            continue
        flags = Counter()
        doms, dirn = [], []
        cited = []
        for d in r["results"]:
            f = (d.get("read") or {}).get("direction")
            if f in FLAGS7:
                flags[f] += 1
            doms.append(d["domain"])
            if f in ("5", "4", "2", "1"):
                sents = d.get("sents") or []
                ev = (d.get("read") or {}).get("evidence") or []
                ids = d.get("sent_ids") or []
                pick = [sents[ids.index(i)] for i in ev if i in ids][:1] or sents[:1]
                cited.append((f, d["domain"], (pick[0] if pick else "")[:140]))
                dirn.append(f)
        n = sum(flags.values())
        s3 = (flags["5"] + flags["4"]) * w3["n_t"] + (flags["1"] + flags["2"]) * w3["n_f"] \
            + (flags["3"] + flags["X"] + flags["I"]) * w3["n_e"]
        s7 = sum(flags[f] * w7[f] for f in FLAGS7)
        note_doms = {_dom(u) for u in re.findall(r"https?://\S+", r.get("evidenceURLs") or "")}
        rows.append({"id": r["review_url"], "lang": r["language"], "label": r["label"],
                     "attrib": r["attrib_tag"], "media": r.get("media_tag"),
                     "wc": r["wc"], "claim": r["claim_text"],
                     "n_docs": n, "flags": dict(flags), "s3": round(s3, 2), "s7": round(s7, 2),
                     "read_ok": sum(1 for d in r["results"]
                                    if (d.get("read_status") or "ok") == "ok"),
                     "n_res": len(r["results"]),
                     "note_url_hit": bool(note_doms & set(doms)),
                     "cited": cited})

    print("== per-language retrieval health ==")
    for lang in ("en", "es", "pt", "fr", "ja"):
        sub = [r for r in rows if r["lang"] == lang]
        if not sub:
            continue
        docs = sum(r["n_res"] for r in sub)
        fl = Counter()
        for r in sub:
            fl.update(r["flags"])
        n = sum(fl.values()) or 1
        d_rate = (fl["5"] + fl["4"] + fl["2"] + fl["1"]) / n
        print(f"  {lang}: {len(sub)} claims | {docs/len(sub):.1f} docs/claim | "
              f"read-ok {sum(r['read_ok'] for r in sub)}/{docs} | "
              f"I {fl['I']/n:.0%} | directional {d_rate:.0%}")

    print("\n== per-class flag mix (share of documents) ==")
    for lab in (True, False):
        fl = Counter()
        for r in rows:
            if r["label"] == lab:
                fl.update(r["flags"])
        n = sum(fl.values()) or 1
        print(f"  label={lab}: " + "  ".join(f"{f}:{fl[f]/n:.1%}" for f in FLAGS7) + f"  (n={n})")

    print("\n== label-suspect candidates (>=1 directional read OPPOSING the label) ==")
    for r in sorted(rows, key=lambda r: r["s7"]):
        n_ref = r["flags"].get("1", 0) + r["flags"].get("2", 0)
        n_sup = r["flags"].get("5", 0) + r["flags"].get("4", 0)
        suspect = (r["label"] and n_ref >= 1 and r["s7"] < 0) or \
                  (not r["label"] and n_sup >= 1 and r["s7"] > 0)
        if suspect:
            print(f"  [{r['lang']}] label={r['label']} s7={r['s7']:+.1f} s3={r['s3']:+.1f} "
                  f"attrib={r['attrib']} | {r['claim'][:110]}")
            for f, dom, sent in r["cited"][:3]:
                print(f"      flag {f} {dom}: {sent}")

    hits = sum(r["note_url_hit"] for r in rows)
    print(f"\n== note-cited URL independently retrieved: {hits}/{len(rows)} claims ==")
    print(f"scored claims: {len(rows)} | mean docs {sum(r['n_docs'] for r in rows)/max(len(rows),1):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.smoke or a.full:
        run_batch(a.workers, full=a.full)
    if a.report or a.smoke or a.full:
        report("probe" if a.full else "smoke")
