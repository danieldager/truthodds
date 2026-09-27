"""Are the refute reads that retrieval NEWLY produced genuine?

Both retrieval arms move claims into the flag zone, but today's audits showed
refute reads are wrong most of the time on true claims. So a crossing only
counts if the read behind it survives the same strict verifier. Re-scrapes the
document (no new search spend), re-selects sentences, and re-judges.

    uv run python -m eval.scripts.build_eval.new_refute_check --arm fc_query
    uv run python -m eval.scripts.build_eval.new_refute_check --arm ceiling_shift_90
    uv run python -m eval.scripts.build_eval.new_refute_check --report
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from pipeline.search import scrape  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import _clean, select_sentences  # noqa: E402
from eval.scripts.build_eval.refute_verify import VERIFY_SYS, PRO_MODEL  # noqa: E402
from eval.scripts.build_eval.fc_query_test import SAMPLE as FC_SAMPLE  # noqa: E402

OUT_DIR = SRC / "eval/data/urn_runs/synth_expt"
VERD = OUT_DIR / "new_refute_verdicts.jsonl"


def rows_for(arm: str) -> list[dict]:
    claims = {r["review_url"]: r for r in map(json.loads, FC_SAMPLE.open())}
    out = []
    for line in (OUT_DIR / ("fc_query_results.jsonl" if arm == "fc_query" else f"{arm}.jsonl")).open():
        r = json.loads(line)
        c = claims.get(r["review_url"])
        if not c:
            continue
        for d in r["docs"]:
            if d.get("direction") in ("1", "2"):
                out.append({"arm": arm, "review_url": r["review_url"],
                            "population": r["population"], "url": d["url"],
                            "domain": d["domain"], "date": d.get("date"),
                            "fc_domain": d["fc_domain"], "flag": d["direction"],
                            "claim": c["claim"], "ceiling": c.get("ceiling")})
    return out


def check(row: dict) -> dict | None:
    text = _clean(scrape(row["url"]) or "")
    if not text:
        return {**{k: row[k] for k in ("arm", "review_url", "population", "domain",
                                       "fc_domain", "flag")},
                "url": row["url"], "refutes": None, "reason": "unscrapeable"}
    _, sents, _ = select_sentences(text, row["claim"])
    if not sents:
        return {**{k: row[k] for k in ("arm", "review_url", "population", "domain",
                                       "fc_domain", "flag")},
                "url": row["url"], "refutes": None, "reason": "no-sentences"}
    import requests
    from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL
    user = (f"CLAIM: {row['claim']}\nCLAIM DATE: {row.get('ceiling') or 'unknown'}\n\n"
            f"DOCUMENT [{row['domain']}] (date: {row.get('date') or 'unknown'})\n"
            f"{' '.join(sents)[:900]}")
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
    return {**{k: row[k] for k in ("arm", "review_url", "population", "domain",
                                   "fc_domain", "flag")},
            "url": row["url"], "refutes": bool(obj.get("refutes")),
            "reason": obj.get("reason", ""),
            "cost": (j.get("usage") or {}).get("estimated_cost") or 0.0}


def run(arm: str, workers: int) -> None:
    rows = rows_for(arm)
    done = set()
    if VERD.exists():
        done = {(json.loads(l)["arm"], json.loads(l)["review_url"], json.loads(l)["url"])
                for l in VERD.open()}
    todo = [r for r in rows if (arm, r["review_url"], r["url"]) not in done]
    print(f"{arm}: {len(todo)} new refute reads to check", flush=True)
    lock = threading.Lock(); n = [0]; t0 = time.time()
    with VERD.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(check, r) for r in todo]
        for fut in as_completed(futs):
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {e}", flush=True); continue
            with lock:
                f.write(json.dumps(out) + "\n"); f.flush()
                n[0] += 1
                if n[0] % 25 == 0 or n[0] == len(todo):
                    print(f"  {n[0]}/{len(todo)}  {n[0]/(time.time()-t0):.1f}/s", flush=True)
    print(f"done: {n[0]}", flush=True)


def report() -> None:
    per = collections.defaultdict(lambda: collections.Counter())
    for l in VERD.open():
        v = json.loads(l)
        k = (v["arm"], v["population"])
        per[k]["n"] += 1
        if v["refutes"] is None:
            per[k]["dead"] += 1
        elif v["refutes"]:
            per[k]["genuine"] += 1
        else:
            per[k]["error"] += 1
    print(f"{'arm':20s} {'population':18s} {'n':>4} {'genuine':>8} {'error':>6} {'dead':>5}  rate")
    for k in sorted(per):
        c = per[k]
        judged = c["genuine"] + c["error"]
        rate = c["genuine"] / judged if judged else 0
        print(f"{k[0]:20s} {k[1]:18s} {c['n']:4d} {c['genuine']:8d} {c['error']:6d} "
              f"{c['dead']:5d}  {rate:5.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.arm:
        run(a.arm, a.workers)
    if a.report:
        report()
