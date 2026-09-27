"""Refute-verification pass: does each refute read survive a targeted re-check?

The flagged-trues diagnosis (clog/270826) found ~97% of refute reads on
flag-zone gold-true claims are read errors (polarity inversion, wrong entity,
adjacent fact). Before adopting a verification pass in the pipeline, this
experiment measures both sides on a neutral verifier (no truth label shown):

  spare rate  — CN-false urn refute reads that survive (genuine refutations
                must not be killed, or recall collapses)
  kill rate   — refute reads on flag/cliff-zone timeline claims and on the
                flag-zone gold-true claims (the FP drivers)

plus the first-order score impact: killed refutes flipped to silent (I),
claims re-scored with the frozen eps=0.10 weights, band migration counted.

    uv run python -m eval.scripts.build_eval.refute_verify --sample
    uv run python -m eval.scripts.build_eval.refute_verify --run --smoke 20
    uv run python -m eval.scripts.build_eval.refute_verify --run --workers 32
    uv run python -m eval.scripts.build_eval.refute_verify --report
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from eval.scripts.build_eval.fit_urn import load_judged_axis, MEDIA_AXIS  # noqa: E402

OUT_DIR = SRC / "eval/data/urn_runs/synth_expt"
SAMPLE = OUT_DIR / "refute_verify_sample.jsonl"
VERDICTS = OUT_DIR / "refute_verify_verdicts.jsonl"
RECLASS = OUT_DIR / "refute_reclass.jsonl"
FIT = SRC / "eval/data/urn_runs/true_timeline/two_urn_fit.json"
C2 = SRC / "eval/data/urn_runs/c2_false"
TL = SRC / "eval/data/urn_runs/true_timeline/scores.jsonl"
E1 = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
SEED = 20260827
EPS = 0.10
PRO_MODEL = "deepseek-ai/DeepSeek-V4-Pro"
CN_PER_BAND = {"flag": 70, "cliff": 40, "pass": 40}

VERIFY_SYS = """You verify whether a document genuinely refutes a specific claim. \
You are given one CLAIM (with its date) and one DOCUMENT (domain, date, extracted \
sentences). An earlier reader marked this document as refuting the claim. Your job \
is to check that call against the text, strictly.

The document REFUTES the claim only if it contradicts this exact proposition:
- Same subject. The document must be about the same entity, event, or figure the \
claim names, not a similarly-worded different one.
- Opposite polarity. A document that REPORTS or CONFIRMS the claimed fact does not \
refute it. Watch for inverted readings: text describing X happening is not evidence \
against "X happened".
- Same scope and time. A figure or state of affairs from a different period, region, \
or definition than the claim's does not refute it. Respect the claim date: the \
document must speak to the state of the world the claim refers to.
- Attribution: if the claim is that a person or outlet SAID something, only evidence \
they did not say it refutes the claim. Evidence that what they said is wrong does not.
- Silence is not refutation. A document that merely fails to mention or support the \
claim, or covers an adjacent fact, does not refute it.

Answer with JSON only:
{"refutes": true|false, "reason": "<one sentence, max 25 words>"}"""


def w7() -> dict:
    for f in json.loads(FIT.read_text())["fits7"]:
        if abs(f["eps"] - EPS) < 1e-9 and f.get("weights"):
            return f["weights"]
    raise SystemExit("no eps=0.10 fit")


def band_of(s: float) -> str:
    return "flag" if s <= -4.05 else ("cliff" if s <= -2.0 else "pass")


def claim_rows(rec: dict, w: dict) -> tuple[float, list[dict]]:
    """(s7, one row per refute read)."""
    docs = [d for d in rec.get("results") or []
            if d.get("read") and d["read"].get("direction")]
    s = sum(w.get(d["read"]["direction"], 0.0) for d in docs)
    rows = []
    for i, d in enumerate(docs):
        if d["read"]["direction"] not in ("1", "2"):
            continue
        rows.append({
            "review_url": rec["review_url"], "doc_idx": i,
            "flag": d["read"]["direction"],
            "domain": d.get("domain"), "date": d.get("date"),
            "sents": " ".join(d.get("sents") or [])[:900],
            "claim": rec.get("claim_resolved") or rec.get("claim_text") or "",
            "ceiling": rec.get("ceiling"),
            "all_flags": [x["read"]["direction"] for x in docs],
        })
    return s, rows


def build_sample() -> None:
    w = w7()
    rng = random.Random(SEED)
    out = []

    excl = {e["claim"][:80] for e in json.loads((C2 / "fit_exclusions.json").read_text())}
    cn = collections.defaultdict(list)
    for p in ("scores.jsonl", "scores_ext.jsonl"):
        for line in (C2 / p).open():
            r = json.loads(line)
            if (r.get("claim_text") or "")[:80] in excl:
                continue
            s, rows = claim_rows(r, w)
            if rows:
                cn[band_of(s)].append((s, rows))
    for band, n in CN_PER_BAND.items():
        pool = cn[band]
        rng.shuffle(pool)
        for s, rows in pool[:n]:
            for row in rows:
                out.append({**row, "population": "cn_false", "band": band, "s7": round(s, 4)})

    for line in TL.open():
        r = json.loads(line)
        s, rows = claim_rows(r, w)
        if rows and band_of(s) in ("flag", "cliff"):
            for row in rows:
                out.append({**row, "population": "tl_feed", "band": band_of(s), "s7": round(s, 4)})

    axis = load_judged_axis()
    for line in E1.open():
        r = json.loads(line)
        if r.get("veracity") not in (4, 5):
            continue
        if (axis.get(r["review_url"]) or "untagged") == MEDIA_AXIS:
            continue
        s, rows = claim_rows(r, w)
        if rows and band_of(s) == "flag":
            for row in rows:
                out.append({**row, "population": "gold_true", "band": "flag", "s7": round(s, 4)})

    with SAMPLE.open("w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    n = collections.Counter((r["population"], r["band"]) for r in out)
    nc = collections.Counter()
    for r in out:
        nc[(r["population"], r["review_url"])] = 1
    claims = collections.Counter(p for p, _ in nc)
    print(f"{len(out)} refute reads, {sum(claims.values())} claims")
    for k in sorted(n):
        print(f"  {k[0]:10s} {k[1]:6s} {n[k]:5d} reads")


def verify(row: dict) -> dict:
    import requests
    from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL
    user = (f"CLAIM: {row['claim']}\n"
            f"CLAIM DATE: {row.get('ceiling') or 'unknown'}\n\n"
            f"DOCUMENT [{row.get('domain')}] (date: {row.get('date') or 'unknown'})\n"
            f"{row['sents']}")
    r = requests.post(f"{EXTRACTION_BASE_URL}/chat/completions",
                      headers={"Authorization": f"Bearer {EXTRACTION_API_KEY}"},
                      json={"model": PRO_MODEL, "temperature": 0, "max_tokens": 200,
                            "response_format": {"type": "json_object"},
                            "messages": [{"role": "system", "content": VERIFY_SYS},
                                         {"role": "user", "content": user}]},
                      timeout=120)
    r.raise_for_status()
    j = r.json()
    obj = json.loads(j["choices"][0]["message"]["content"] or "{}")
    return {"review_url": row["review_url"], "doc_idx": row["doc_idx"],
            "population": row["population"], "band": row["band"],
            "flag": row["flag"], "refutes": bool(obj.get("refutes")),
            "reason": obj.get("reason", ""),
            "cost": (j.get("usage") or {}).get("estimated_cost") or 0.0}


def run(smoke: int, workers: int) -> None:
    rows = [json.loads(l) for l in SAMPLE.open()]
    done = set()
    if VERDICTS.exists():
        done = {(json.loads(l)["review_url"], json.loads(l)["doc_idx"])
                for l in VERDICTS.open()}
    todo = [r for r in rows if (r["review_url"], r["doc_idx"]) not in done]
    if smoke:
        rng = random.Random(SEED)
        rng.shuffle(todo)
        todo = todo[:smoke]
    print(f"{len(todo)} to verify ({len(done)} done), workers={workers}", flush=True)
    lock = threading.Lock()
    n = [0]
    cost = [0.0]
    t0 = time.time()
    with VERDICTS.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(verify, r): r for r in todo}
        for fut in as_completed(futs):
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {futs[fut]['review_url']}: {e}", flush=True)
                continue
            with lock:
                f.write(json.dumps(out) + "\n")
                f.flush()
                n[0] += 1
                cost[0] += out["cost"]
                if n[0] % 50 == 0 or n[0] == len(todo):
                    el = time.time() - t0
                    print(f"  {n[0]}/{len(todo)}  ${cost[0]:.2f}  "
                          f"{n[0]/el:.1f}/s  ETA {(len(todo)-n[0])/max(n[0]/el,.01):.0f}s",
                          flush=True)
    print(f"done: {n[0]} verified, ${cost[0]:.2f}", flush=True)


def report() -> None:
    w = w7()
    rows = {(r["review_url"], r["doc_idx"]): r for r in map(json.loads, SAMPLE.open())}
    verd = [json.loads(l) for l in VERDICTS.open()]
    seen = {}
    for v in verd:
        seen[(v["review_url"], v["doc_idx"])] = v

    per = collections.defaultdict(lambda: [0, 0])
    for v in seen.values():
        per[(v["population"], v["band"])][0] += 1
        if v["refutes"]:
            per[(v["population"], v["band"])][1] += 1
    print("=== survival of refute reads (refutes=true = read kept) ===")
    for k in sorted(per):
        n, kept = per[k]
        print(f"  {k[0]:10s} {k[1]:6s} kept {kept:4d}/{n:4d}  ({kept/n:5.1%})")
    tot = collections.defaultdict(lambda: [0, 0])
    for v in seen.values():
        tot[v["population"]][0] += 1
        if v["refutes"]:
            tot[v["population"]][1] += 1
    for k in sorted(tot):
        n, kept = tot[k]
        print(f"  {k:10s} TOTAL  kept {kept:4d}/{n:4d}  ({kept/n:5.1%})")

    # score impact: killed refute -> I, re-score, count band migration
    claims = collections.defaultdict(dict)
    for (url, idx), r in rows.items():
        claims[(r["population"], url)][idx] = r
    print("\n=== band migration after killed refutes -> silent ===")
    mig = collections.defaultdict(collections.Counter)
    for (pop, url), docs in claims.items():
        any_r = next(iter(docs.values()))
        flags = list(any_r["all_flags"])
        judged = 0
        for idx, r in docs.items():
            v = seen.get((url, idx))
            if v is None:
                continue
            judged += 1
            if not v["refutes"]:
                flags[idx] = "I"
        if judged < len(docs):        # incomplete claim: skip migration
            continue
        b0 = band_of(any_r["s7"])
        b1 = band_of(sum(w.get(f, 0.0) for f in flags))
        mig[pop][(b0, b1)] += 1
    for pop in sorted(mig):
        print(f"  {pop}:")
        for (b0, b1), c in sorted(mig[pop].items()):
            mark = "" if b0 == b1 else "  <-- moved"
            print(f"    {b0:6s} -> {b1:6s} {c:4d}{mark}")



RECLASS_SYS = """An earlier reader marked one document as REFUTING a claim, and a check \
found that call wrong. Your job is to give the document its correct bearing on the \
claim. Do not delete the read, reclassify it.

You are given the CLAIM (with its date) and one DOCUMENT (domain, date, extracted \
sentences). Choose exactly one flag:

  "5"  the document states or establishes the claim itself
  "4"  the document points toward the claim but does not establish it
  "3"  contested: real evidence both for and against the claim itself
  "2"  the document points against the claim but does not disprove it
  "1"  the document contradicts or disproves the claim itself
  "X"  on-claim context: information about the claim's subject that permits neither \
support nor refutation, including background, partial figures, or side details
  "I"  irrelevant: a different proposition, even if the topic is the same

Rules. A document that REPORTS or CONFIRMS the claimed fact supports it ("5" or "4"), \
it does not refute it. A document about a different entity, event, or period than the \
claim names is "I". A document on the claim's subject that never reaches the claimed \
proposition is "X". Respect the claim date. If the claim is that someone SAID \
something, judge whether they said it, not whether it is true. Only choose "1" or "2" \
if the document really does bear against this exact proposition.

Answer with JSON only:
{"flag": "<one of 5 4 3 2 1 X I>", "reason": "<one sentence, max 25 words>"}"""


def reclass(row: dict) -> dict:
    import requests
    from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL
    user = (f"CLAIM: {row['claim']}\n"
            f"CLAIM DATE: {row.get('ceiling') or 'unknown'}\n\n"
            f"DOCUMENT [{row.get('domain')}] (date: {row.get('date') or 'unknown'})\n"
            f"{row['sents']}")
    r = requests.post(f"{EXTRACTION_BASE_URL}/chat/completions",
                      headers={"Authorization": f"Bearer {EXTRACTION_API_KEY}"},
                      json={"model": PRO_MODEL, "temperature": 0, "max_tokens": 200,
                            "response_format": {"type": "json_object"},
                            "messages": [{"role": "system", "content": RECLASS_SYS},
                                         {"role": "user", "content": user}]},
                      timeout=120)
    r.raise_for_status()
    j = r.json()
    obj = json.loads(j["choices"][0]["message"]["content"] or "{}")
    f = obj.get("flag")
    return {"review_url": row["review_url"], "doc_idx": row["doc_idx"],
            "population": row["population"], "band": row["band"],
            "flag_was": row["flag"], "flag_now": f if f in ("5","4","3","2","1","X","I") else "I",
            "reason": obj.get("reason", ""),
            "cost": (j.get("usage") or {}).get("estimated_cost") or 0.0}


def run_reclass(workers: int, smoke: int = 0) -> None:
    """Reclassify every read the verifier said is not a refutation."""
    rows = {(r["review_url"], r["doc_idx"]): r for r in map(json.loads, SAMPLE.open())}
    kills = [(v["review_url"], v["doc_idx"])
             for v in map(json.loads, VERDICTS.open()) if not v["refutes"]]
    done = set()
    if RECLASS.exists():
        done = {(json.loads(l)["review_url"], json.loads(l)["doc_idx"]) for l in RECLASS.open()}
    todo = [rows[k] for k in kills if k not in done and k in rows]
    if smoke:
        todo = todo[:smoke]
    print(f"{len(todo)} reads to reclassify, workers={workers}", flush=True)
    lock = threading.Lock(); n = [0]; cost = [0.0]; t0 = time.time()
    with RECLASS.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(reclass, r) for r in todo]
        for fut in as_completed(futs):
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {e}", flush=True); continue
            with lock:
                f.write(json.dumps(out) + "\n"); f.flush()
                n[0] += 1; cost[0] += out["cost"]
                if n[0] % 100 == 0 or n[0] == len(todo):
                    el = time.time() - t0
                    print(f"  {n[0]}/{len(todo)}  ${cost[0]:.2f}  {n[0]/el:.1f}/s", flush=True)
    print(f"done: {n[0]} reclassified, ${cost[0]:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reclass", action="store_true")
    a = ap.parse_args()
    if a.sample:
        build_sample()
    if a.run:
        run(a.smoke, a.workers)
    if a.reclass:
        run_reclass(a.workers, a.smoke)
    if a.report:
        report()
