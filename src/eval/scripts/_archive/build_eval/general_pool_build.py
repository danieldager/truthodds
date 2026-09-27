"""General-pool (trivially-true candidate) stratum — screen ALL zeerover captures,
extract claims, score with the instrument, topics + descriptive stats (Daniel go
2026-08-24: "extract the claims, show me a random sample, score them, topics").

Stages (each resumable / skip-if-done):
  screen  — on-topic screen over ALL en/fr captures (reuses the 200-tweet smoke rows)
  chain   — qualifying tweets -> production extraction chain; --voice user, except a
            curated list of news-outlet handles routed --voice outlet

  screen + chain MOVED 2026-08-25 to claim_sourcing/ingest_timeline_capture.py (TH2:
  one ingest path shared with the production capture flow). This module imports them;
  the stratum-specific work below (screens / score / report) stays here.
  screens — attribution/media/dup tags + cross-corpus dedup (CN-false, wire, E1)
  score   — verify ALL checkable claims via the cf_probe adapter (snowflake ceiling,
            s3+s7). DESCRIPTIVE: this stratum has no labels (F-T4). 20-claim smoke
            sanity gate, then continues. Cap $3.
  report  — score distribution, topics across strata, review_draft.html

  uv run python -m eval.scripts.build_eval.general_pool_build          # all stages
  uv run python -m eval.scripts.build_eval.general_pool_build --report # analysis only
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from eval.scripts.claim_sourcing.ingest_timeline_capture import (  # shared ingest
    stage_chain, stage_screen,
)

SRC = Path(__file__).resolve().parents[3]
POOL = SRC / "eval/data/tweet_corpus/general_pool"
TC = SRC / "eval/data/tweet_corpus"
OUT = SRC / "eval/data/urn_runs/general_pool"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
# The shipped constants (7-flag since 2026-09-14) and the 3-voice fit on the same
# frozen population, kept beside them as the comparison arm.
E1_METRICS = SRC / "eval/data/urn_runs/e1_ctx/headline_metrics.json"
E1_METRICS_3VOICE = SRC / "eval/data/urn_runs/e1_ctx/headline_metrics_clustered.json"
PAD_TO = 10
E1_RESULTS = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
SEED = 20260824
SCORE_CAP = 3.00
SMOKE_N = 20

# same tag regexes as the sibling strata (cn_false_stratum / cf_probe)
MEDIA_RE = re.compile(
    r"\b(video|photo|image|footage|clip|picture|audio|recording|screenshot)s?\b.{0,40}"
    r"\b(show|shows|showing|depict|depicts|depicting|captur|of|from|is real|is fake|"
    r"is authentic|AI-generated|doctored|manipulated|edited)\b"
    r"|\b(vid[ée]o|photo|image)s?\b.{0,40}\b(montre|montrant|truqu[ée]e?|r[ée]elle?|"
    r"authentique|falsifi[ée]e?)\b", re.I)


# ---------------------------------------------------------------- stage: screens

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9À-ɏ]{3,}", (s or "").lower()))


def _jac(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def stage_screens():
    df = pl.read_parquet(TC / "general_verify_input.parquet")
    df = df.with_columns([
        (pl.col("type") == "attribution").alias("attribution_form"),
        pl.col("claim").map_elements(lambda c: bool(MEDIA_RE.search(c or "")),
                                     return_dtype=pl.Boolean).alias("media_locus"),
    ])
    toks = [_tokens(c) for c in df["claim"].to_list()]
    dup, kept = [False] * len(toks), []
    for i, t in enumerate(toks):
        for j in kept:
            if _jac(t, toks[j]) >= 0.85:
                dup[i] = True
                break
        else:
            kept.append(i)
    # cross-corpus near-dup (vs CN-false A/B claims, wire claims, E1 gold claims)
    other = []
    for p in (TC / "c2_ab_new_verify_input.parquet", TC / "wire_true_verify_input.parquet"):
        other += [_tokens(c) for c in pl.read_parquet(p)["claim"].to_list()]
    with open(E1_RESULTS) as f:
        other += [_tokens(json.loads(line).get("claim_text", "")) for line in f]
    xdup = [any(_jac(t, o) >= 0.6 for o in other) if t else False for t in toks]
    df = df.with_columns([pl.Series("claim_dup", dup), pl.Series("xcorpus_dup", xdup)])
    scoreable = df.filter(pl.col("checkworthy") & ~pl.col("claim_dup"))
    print(f"screens: {df.height} claims | checkworthy {df['checkworthy'].sum()} | "
          f"attribution {df['attribution_form'].sum()} | media {df['media_locus'].sum()} | "
          f"dup {sum(dup)} | xcorpus {sum(xdup)} | SCOREABLE {scoreable.height}", flush=True)
    df.write_parquet(TC / "general_claims.parquet")


# ---------------------------------------------------------------- stage: score

def snowflake_date(tid: str) -> str | None:
    try:
        ms = (int(tid) >> 22) + 1288834974657
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return None


def _adapt(r: dict) -> dict:
    return {**r,
            "review_url": f"gp:{r['claim_id']}",
            "claim_text": r["claim"],
            "publisher_site": "x.com",
            "claim_date": snowflake_date(r["post_id"]),
            "claim_type": "attribution" if r["attribution_form"] else "assertion",
            "topic": r.get("topic"),
            "x_context": None, "context_ok": False,
            "resolution_status": "native", "claim_resolved": None,
            "veracity": None, "rating_subtype": None,
            "review_date": None, "x_date": None, "yr": None,
            "screen_verdict": "unscreened", "screen_leak": False}


def stage_score(workers: int):
    from pipeline.search import newsguard_score_map
    from eval.scripts.build_eval.evidence_urn_run import run_claim
    df = pl.read_parquet(TC / "general_claims.parquet")
    todo_df = df.filter(pl.col("checkworthy") & ~pl.col("claim_dup"))
    rows = [_adapt(r) for r in todo_df.iter_rows(named=True)]
    out = OUT / "scores.jsonl"
    seen = set()
    if out.exists():
        for line in open(out):
            try:
                seen.add(json.loads(line)["review_url"])
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in rows if r["review_url"] not in seen]
    # smoke-first ordering: fixed-seed shuffle so the first 20 are a random sanity draw
    random.Random(SEED).shuffle(todo)
    print(f"score: {len(todo)}/{len(rows)} to run | cap ${SCORE_CAP} | "
          f"smoke gate after {SMOKE_N}", flush=True)
    ng = newsguard_score_map()
    lock = threading.Lock()
    budget = {"spent": 0.0, "done": 0, "cached_tok": 0, "prompt_tok": 0, "stop": False}
    smoke_gate = threading.Event()
    if len(seen) >= SMOKE_N:
        smoke_gate.set()
    t0 = time.time()
    fh = open(out, "a")

    def work(row):
        if budget["stop"]:
            return
        if not smoke_gate.is_set() and budget["done"] >= SMOKE_N:
            smoke_gate.wait()
        try:
            rec = run_claim(row, ng, budget, {})
        except Exception as e:  # noqa: BLE001
            with lock:
                budget["done"] += 1
                print(f"  [claim-failed] {row['review_url']}: {type(e).__name__}: {e}",
                      flush=True)
            return
        for k in ("post_id", "claim_id", "lang", "voice", "handle", "topic",
                  "attribution_form", "media_locus", "xcorpus_dup", "screen_category"):
            rec[k] = row.get(k)
        with lock:
            budget["spent"] += rec["cost"]
            budget["done"] += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n = budget["done"]
            proj = budget["spent"] / n * len(todo)
            rate = n / max((time.time() - t0) / 60, .01)
            print(f"  {n}/{len(todo)} {row.get('lang')} | spent ${budget['spent']:.3f} "
                  f"proj ${proj:.2f} | {rate:.1f}/min ETA {(len(todo)-n)/max(rate,1):.0f}m",
                  flush=True)
            if proj > SCORE_CAP and n >= 10:
                print(f"  BUDGET ABORT: ${proj:.2f} > ${SCORE_CAP}", flush=True)
                budget["stop"] = True
            if not smoke_gate.is_set() and n >= SMOKE_N:
                fh.flush()
                ok = _smoke_ok(out)
                print(f"  SMOKE GATE at {n}: {'PASS - continuing' if ok else 'FAIL - aborting'}",
                      flush=True)
                if not ok:
                    budget["stop"] = True
                smoke_gate.set()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"score done {budget['done']} | ${budget['spent']:.4f} | "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


def _smoke_ok(out: Path) -> bool:
    recs = [json.loads(l) for l in open(out)]
    n_docs = sum(len(r.get("results") or []) for r in recs)
    reads_ok = sum(1 for r in recs for d in (r.get("results") or [])
                   if (d.get("read") or {}).get("direction") in FLAGS7)
    return n_docs > 0 and reads_ok / n_docs >= 0.5


# ---------------------------------------------------------------- stage: report

FLAGS7 = ["5", "4", "3", "2", "1", "X", "I"]


def _scores(rec, w3, w7):
    flags = Counter()
    cited = []
    for d in rec.get("results") or []:
        f = (d.get("read") or {}).get("direction")
        if f in FLAGS7:
            flags[f] += 1
        if f in ("5", "4", "2", "1"):
            sents = d.get("sents") or []
            ev = (d.get("read") or {}).get("evidence") or []
            ids = d.get("sent_ids") or []
            pick = [sents[ids.index(i)] for i in ev if i in ids][:1] or sents[:1]
            cited.append((f, d.get("domain"), (pick[0] if pick else "")[:180]))
    # Both fits pad every claim to PAD_TO slots with silent reads, so scoring pads
    # too — the weights are only valid under that convention. `flags` is left
    # unpadded because the caller reports it as the documents actually read.
    pad = max(0, PAD_TO - sum(flags.values()))
    s3 = ((flags["5"] + flags["4"]) * w3["n_t"] + (flags["1"] + flags["2"]) * w3["n_f"]
          + (flags["3"] + flags["X"] + flags["I"] + pad) * w3["n_e"])
    # Shipped fit is SIX-flag (read-v6.1, 2026-09-14): a read-v5 "3" folds into "X";
    # w7 has no "3" key.
    s7 = (sum(flags[f] * w7[f] for f in FLAGS7 if f != "3")
          + flags["3"] * w7["X"] + pad * w7["I"])
    return flags, s3, s7, cited


def _js_div(p: Counter, q: Counter) -> float:
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    js = 0.0
    for k in keys:
        a, b = p[k] / sp, q[k] / sq
        m = (a + b) / 2
        for x in (a, b):
            if x > 0:
                js += 0.5 * x * math.log2(x / m)
    return js


def stage_report():
    m7 = json.load(open(E1_METRICS))["overall"]
    w7, thr7 = m7["weights"], m7["threshold"]
    m3 = json.load(open(E1_METRICS_3VOICE))["overall"]
    w3 = m3["weights"]
    thr3 = sum(m3["thresholds_by_fold"]) / len(m3["thresholds_by_fold"])

    recs = [json.loads(l) for l in open(OUT / "scores.jsonl")]
    scored = []
    for r in recs:
        flags, s3, s7, cited = _scores(r, w3, w7)
        n = sum(flags.values())
        scored.append({**{k: r.get(k) for k in
                          ("review_url", "claim_id", "post_id", "lang", "voice", "handle",
                           "topic", "attribution_form", "screen_category")},
                       "claim": r.get("claim_text"), "n_docs": n, "flags": dict(flags),
                       "s3": round(s3, 2), "s7": round(s7, 2), "cited": cited,
                       "silent": all(f in ("X", "I") for f in flags.elements()) or n == 0,
                       "flagged7": s7 < thr7, "flagged3": s3 < thr3})
    json.dump(scored, open(OUT / "scored_claims.json", "w"), ensure_ascii=False, indent=1)

    print(f"\n== score distribution ({len(scored)} claims) ==")
    for lang in ("en", "fr"):
        sub = [s for s in scored if s["lang"] == lang]
        if not sub:
            continue
        fl = sum(1 for s in sub if s["flagged7"])
        sil = sum(1 for s in sub if s["silent"])
        med = sorted(s["s7"] for s in sub)[len(sub) // 2]
        print(f"  {lang}: n {len(sub)} | median s7 {med:+.1f} | flagged7 {fl} "
              f"({fl/len(sub):.0%}) | all-silent {sil} ({sil/len(sub):.0%})")
    print("  s7 histogram:", Counter(
        "<-8" if s["s7"] < -8 else "-8..-4" if s["s7"] < -4 else "-4..0" if s["s7"] < 0
        else "0..4" if s["s7"] < 4 else "4..8" if s["s7"] < 8 else ">8"
        for s in scored))
    print("\n== flagged claims (candidate NOT-trivially-true) ==")
    for s in sorted([s for s in scored if s["flagged7"]], key=lambda x: x["s7"]):
        print(f"  {s['s7']:+.1f} [{s['lang']}] @{s['handle']}: {s['claim'][:110]}")
        for f, dom, sent in s["cited"][:3]:
            print(f"      flag {f} {dom}: {sent[:130]}")

    # topics across strata
    strata = {"general": Counter(s["topic"] for s in scored)}
    ab = pl.read_parquet(TC / "c2_ab_new_verify_input.parquet")
    gate = {(json.loads(l)["post_id"], json.loads(l)["claim"]): json.loads(l)["band"]
            for l in open(SRC / "eval/data/urn_runs/c2_audit/ab_gate_new.jsonl")}
    fal = ab.filter(pl.struct(["post_id", "claim"]).map_elements(
        lambda r: gate.get((r["post_id"], r["claim"])) == "false", return_dtype=pl.Boolean))
    strata["cn_false"] = Counter(fal["topic"].to_list())
    wire = pl.read_parquet(TC / "wire_true_verify_input.parquet")
    wscr = pl.read_parquet(SRC / "eval/data/urn_runs/wire_audit/screens.parquet")
    weli = wire.join(wscr.filter(pl.col("fit_eligible")).select(["post_id", "claim"]),
                     on=["post_id", "claim"], how="inner")
    strata["wire"] = Counter(weli["topic"].to_list())
    # NOTE: acquitted stratum hook — add its verify_input here once extracted.
    print("\n== topic mix by stratum (share) ==")
    topics = sorted(set().union(*strata.values()))
    for name, c in strata.items():
        tot = sum(c.values())
        print(f"  {name} (n={tot}):",
              {t: f"{c[t]/tot:.0%}" for t in topics if c[t] / tot >= 0.03})
    for a, b in (("cn_false", "wire"), ("cn_false", "general"), ("wire", "general")):
        print(f"  JS({a},{b}) = {_js_div(strata[a], strata[b]):.3f}")
    json.dump({k: dict(v) for k, v in strata.items()},
              open(OUT / "topic_mix.json", "w"), indent=1)
    return scored, strata, thr7


def stage_html(scored: list[dict], strata: dict, thr7: float):
    df = pl.read_parquet(TC / "general_claims.parquet")
    post_text = {r["post_id"]: r["post_text"] for r in
                 df.select(["post_id", "post_text"]).unique().iter_rows(named=True)}
    sample = random.Random(SEED).sample(scored, min(40, len(scored)))
    thr_note = f"flagged = 7-flag score below the E1 2%-FPR threshold ({thr7:.3f})"

    def esc(s):
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # s7 histogram (inline SVG, en vs fr)
    bins = [(-99, -8), (-8, -4), (-4, 0), (0, 4), (4, 8), (8, 99)]
    labels = ["<−8", "−8..−4", "−4..0", "0..4", "4..8", ">8"]
    counts = {lang: [sum(1 for s in scored if s["lang"] == lang and lo <= s["s7"] < hi)
                     for lo, hi in bins] for lang in ("en", "fr")}
    mx = max(max(counts["en"], default=1), max(counts["fr"], default=1), 1)
    bars = []
    for i, lab in enumerate(labels):
        for k, (lang, color) in enumerate((("en", "#356"), ("fr", "#9ab"))):
            h = 160 * counts[lang][i] / mx
            bars.append(f'<rect x="{40+i*110+k*46}" y="{200-h}" width="42" height="{h}" '
                        f'fill="{color}"/>'
                        f'<text x="{40+i*110+k*46+21}" y="{194-h}" text-anchor="middle" '
                        f'font-size="11" fill="#333">{counts[lang][i]}</text>')
        bars.append(f'<text x="{40+i*110+44}" y="218" text-anchor="middle" font-size="11" '
                    f'fill="#333">{lab}</text>')
    svg = (f'<svg viewBox="0 0 720 230" style="max-width:720px;width:100%">'
           f'{"".join(bars)}'
           f'<text x="40" y="14" font-size="12" fill="#356">■ en</text>'
           f'<text x="90" y="14" font-size="12" fill="#9ab">■ fr</text></svg>')

    topics = sorted(set().union(*strata.values()))
    head = "".join(f"<th>{t}</th>" for t in topics)
    trows = ""
    for name, c in strata.items():
        tot = sum(c.values()) or 1
        cells = "".join(f"<td>{c[t]/tot:.0%}</td>" if c[t] else "<td>–</td>" for t in topics)
        trows += f"<tr><td><b>{name}</b> (n={tot})</td>{cells}</tr>"

    cards = ""
    for s in sample:
        fl = " ".join(f"{f}:{n}" for f, n in sorted(s["flags"].items()))
        tags = [t for t, on in (("attribution", s["attribution_form"]),
                                ("flagged", s["flagged7"]), ("silent", s["silent"])) if on]
        cited = "".join(f'<div class="cite">flag {f} — {esc(d)}: “{esc(sent)}”</div>'
                        for f, d, sent in (s.get("cited") or [])[:2])
        cards += f"""<div class="card">
<div class="meta">@{esc(s['handle'])} · {s['lang']} · topic: {s['topic']} ·
 screen: {s.get('screen_category') or '–'} · {' · '.join(tags) if tags else 'no tags'}</div>
<div class="tweet">{esc(post_text.get(s['post_id'], ''))}</div>
<div class="claim">→ {esc(s['claim'])}</div>
<div class="score">s7 {s['s7']:+.2f} · s3 {s['s3']:+.2f} · {s['n_docs']} read docs · flags {fl or 'none'}</div>
{cited}</div>"""

    html = f"""<meta charset="utf-8"><title>General-pool claim review</title><style>
body{{font:15px/1.5 -apple-system,sans-serif;color:#222;max-width:1100px;margin:2rem auto;padding:0 1rem}}
.card{{border:1px solid #ddd;border-radius:6px;padding:.8rem 1rem;margin:.8rem 0}}
.meta{{color:#666;font-size:13px}}.tweet{{white-space:pre-wrap;margin:.4rem 0;color:#444}}
.claim{{font-weight:600;margin:.3rem 0}}.score{{font-size:13px;color:#356}}
.cite{{font-size:13px;color:#865;margin-top:.2rem}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}}td,th{{border:1px solid #ccc;padding:.3rem .5rem;font-size:13px;text-align:left}}
</style>
<h1>General-pool (trivially-true candidate) stratum — claim review</h1>
<p>{len(scored)} scored claims from Daniel's zeerover captures (no labels — descriptive only).
{thr_note}. Random sample of {len(sample)} below, fixed seed {SEED}.</p>
<h2>7-flag score distribution</h2>{svg}
<h2>Topic mix across strata</h2>
<table><tr><th>stratum</th>{head}</tr>{trows}</table>
<h2>Random sample</h2>{cards}"""
    out = POOL / "review_draft.html"
    out.write_text(html)
    print(f"review draft -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--workers", type=int, default=16, help="scoring workers")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.report:
        stage_screen(args.concurrency, screen="v1")  # the eval pool stays on the original screen
        stage_chain(args.concurrency, screen="v1")
        stage_screens()
        stage_score(args.workers)
    scored, strata, thr7 = stage_report()
    stage_html(scored, strata, thr7)


if __name__ == "__main__":
    main()
