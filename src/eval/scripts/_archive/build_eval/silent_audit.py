"""Is the bottleneck the READ step or RETRIEVAL?

The refute-verification pass only checked reads already marked as refuting. This
audits the other direction: reads the reader called silent or context-only
("I", "X", "3") on claims the score MISSED. If a genuine refutation is sitting
in those documents, the reader lost it and the bottleneck is the read step. If
the documents really do not refute, retrieval never delivered the evidence and
the bottleneck is retrieval.

Control: the same audit on timeline claims the score also left alone. Those are
mostly true, so anything the auditor "finds" there is auditor noise, and the
CN-false rate has to beat it to mean anything.

    uv run python -m eval.scripts.build_eval.silent_audit --sample
    uv run python -m eval.scripts.build_eval.silent_audit --run --workers 32
    uv run python -m eval.scripts.build_eval.silent_audit --report
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

from eval.scripts.build_eval.refute_verify import VERIFY_SYS, PRO_MODEL, w7  # noqa: E402

OUT_DIR = SRC / "eval/data/urn_runs/synth_expt"
SAMPLE = OUT_DIR / "silent_audit_sample.jsonl"
VERDICTS = OUT_DIR / "silent_audit_verdicts.jsonl"
C2 = SRC / "eval/data/urn_runs/c2_false"
TL = SRC / "eval/data/urn_runs/true_timeline/scores.jsonl"
SEED = 20260827
N_CLAIMS = {"cn_false_missed": 120, "tl_pass": 60}
SILENT = ("I", "X", "3")


def build_sample() -> None:
    w = w7()
    rng = random.Random(SEED)
    pools: dict[str, list] = {"cn_false_missed": [], "tl_pass": []}

    def collect(rec, pop):
        docs = [d for d in rec.get("results") or []
                if d.get("read") and d["read"].get("direction")]
        if not docs:
            return
        s = sum(w.get(d["read"]["direction"], 0.0) for d in docs)
        if s <= -4.05:               # already flagged: not a miss
            return
        sil = [(i, d) for i, d in enumerate(docs) if d["read"]["direction"] in SILENT]
        if not sil:
            return
        pools[pop].append({
            "review_url": rec["review_url"], "s7": round(s, 4),
            "claim": rec.get("claim_resolved") or rec.get("claim_text") or "",
            "ceiling": rec.get("ceiling"),
            "docs": [{"doc_idx": i, "flag": d["read"]["direction"],
                      "domain": d.get("domain"), "date": d.get("date"),
                      "sents": " ".join(d.get("sents") or [])[:900]} for i, d in sil]})

    excl = {e["claim"][:80] for e in json.loads((C2 / "fit_exclusions.json").read_text())}
    for p in ("scores.jsonl", "scores_ext.jsonl"):
        for line in (C2 / p).open():
            r = json.loads(line)
            if (r.get("claim_text") or "")[:80] in excl:
                continue
            collect(r, "cn_false_missed")
    for line in TL.open():
        collect(json.loads(line), "tl_pass")

    rows = []
    for pop, pool in pools.items():
        rng.shuffle(pool)
        for c in pool[:N_CLAIMS[pop]]:
            for d in c["docs"]:
                rows.append({"population": pop, "review_url": c["review_url"],
                             "s7": c["s7"], "claim": c["claim"],
                             "ceiling": c["ceiling"], **d})
    with SAMPLE.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = collections.Counter(r["population"] for r in rows)
    nc = collections.Counter(r["population"] for r in {(r["population"], r["review_url"]): r
                                                       for r in rows}.values())
    for pop in n:
        print(f"  {pop:18s} {nc[pop]:4d} claims  {n[pop]:5d} silent reads")


def audit(row: dict) -> dict:
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
            "population": row["population"], "flag_was": row["flag"],
            "refutes": bool(obj.get("refutes")), "reason": obj.get("reason", ""),
            "cost": (j.get("usage") or {}).get("estimated_cost") or 0.0}


def run(workers: int, smoke: int) -> None:
    rows = [json.loads(l) for l in SAMPLE.open()]
    done = set()
    if VERDICTS.exists():
        done = {(json.loads(l)["review_url"], json.loads(l)["doc_idx"]) for l in VERDICTS.open()}
    todo = [r for r in rows if (r["review_url"], r["doc_idx"]) not in done]
    if smoke:
        todo = todo[:smoke]
    print(f"{len(todo)} silent reads to audit, workers={workers}", flush=True)
    lock = threading.Lock(); n = [0]; cost = [0.0]; t0 = time.time()
    with VERDICTS.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(audit, r) for r in todo]
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
                    print(f"  {n[0]}/{len(todo)}  ${cost[0]:.2f}  {n[0]/el:.1f}/s  "
                          f"ETA {(len(todo)-n[0])/max(n[0]/el,.01):.0f}s", flush=True)
    print(f"done: {n[0]} audited, ${cost[0]:.2f}", flush=True)


def report() -> None:
    rows = {(r["review_url"], r["doc_idx"]): r for r in map(json.loads, SAMPLE.open())}
    verd = {(v["review_url"], v["doc_idx"]): v for v in map(json.loads, VERDICTS.open())}
    per_read = collections.defaultdict(lambda: [0, 0])
    per_claim = collections.defaultdict(lambda: collections.defaultdict(int))
    for k, v in verd.items():
        per_read[v["population"]][0] += 1
        per_read[v["population"]][1] += v["refutes"]
        per_claim[v["population"]][k[0]] += v["refutes"]
    print("=== reads the reader called silent that DO refute ===")
    for pop, (n, hit) in sorted(per_read.items()):
        print(f"  {pop:18s} {hit:5d}/{n:5d} reads  ({hit/n:5.1%})")
    print("\n=== claims with at least one missed refutation hiding in a silent read ===")
    for pop, d in sorted(per_claim.items()):
        n = len(d); hit = sum(1 for v in d.values() if v)
        print(f"  {pop:18s} {hit:4d}/{n:4d} claims ({hit/n:5.1%})   "
              f"mean recoverable refutes/claim {sum(d.values())/n:.2f}")
    by_flag = collections.defaultdict(lambda: [0, 0])
    for k, v in verd.items():
        if v["population"] != "cn_false_missed":
            continue
        c = by_flag[rows[k]["flag"]]
        c[0] += 1; c[1] += v["refutes"]
    print("\n=== CN-false missed: by the flag the reader gave it ===")
    for f, (n, hit) in sorted(by_flag.items()):
        print(f"  flag {f}: {hit:4d}/{n:5d} ({hit/n:5.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.sample:
        build_sample()
    if a.run:
        run(a.workers, a.smoke)
    if a.report:
        report()
