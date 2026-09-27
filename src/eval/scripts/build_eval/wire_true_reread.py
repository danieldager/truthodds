"""Re-read wire-true's stored documents with the PRODUCTION reader (read-v5 on
DeepSeek-V4-Flash, evidence_urn_run.read_doc), so all three datasets are read by one
reader and the 7-flag two-urn fit is comparable (Daniel, 2026-09-10). No retrieval:
every document keeps its stored sentences, only `read`, `region_reads`, `read_status`
and the read prompt/hash change. Resumable by claim.

    uv run python -m eval.scripts.build_eval.wire_true_reread --workers 96 --gate-start 48 --gate-cap 96
Output: eval/data/urn_runs/wire_true/results-00_readv5.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.scripts.build_eval import evidence_urn_run as eur  # noqa: E402
from eval.scripts.build_eval import reader_lab  # noqa: E402

SRC = Path("eval/data/urn_runs/wire_true/results-00.jsonl")
OUT = Path("eval/data/urn_runs/wire_true/results-00_readv5.jsonl")


def reread(rec):
    claim = rec.get("claim_resolved") or rec["claim_text"]
    claim_block = (f"CLAIM: {claim}\n"
                   f"(claimed on {rec.get('claim_date_shown') or 'unknown date'}; judge the "
                   f"document's bearing on this exact proposition)")
    key = rec["review_url"]
    cost = 0.0
    for d in rec["results"]:
        if d.get("read_status") in ("fc-undated", "empty-doc") or not d.get("sents"):
            continue
        out = None
        for _ in range(2):
            try:
                with reader_lab.GATE:
                    out, c, _cached, _ptok = eur.read_doc(claim_block, d["sent_ids"], d["sents"], key)
                reader_lab.GATE.success()
            except Exception as e:  # noqa: BLE001
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code in reader_lab.THROTTLE_CODES or code is None:   # 429/5xx or transport failure
                    reader_lab.GATE.throttle(str(code or type(e).__name__))
                time.sleep(2)
                continue
            cost += c
            if out:
                break
        reads = [out or {"direction": "I", "evidence": [], "reason": "", "qc_flag": "read-failed"}]
        d["region_reads"] = reads
        d["read"] = eur.aggregate_reads(reads)
        d["read_status"] = "failed" if out is None else "ok"
    rec["prompts"]["read"] = eur.READ_PROMPT_V
    rec["prompt_hash"]["read"] = eur.PROMPT_HASHES["read"]
    rec["reread_cost"] = round(cost, 6)
    rec["reread_model"] = eur.VERIFICATION_MODEL
    return rec, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--gate-start", type=int, default=48)
    ap.add_argument("--gate-cap", type=int, default=96)
    args = ap.parse_args()
    reader_lab.GATE.limit = reader_lab.GATE.peak = args.gate_start
    reader_lab.GATE.cap = args.gate_cap
    recs = [json.loads(l) for l in SRC.open()]
    seen = set()
    if OUT.exists():
        for l in OUT.open():
            seen.add(json.loads(l)["review_url"])
    todo = [r for r in recs if r["review_url"] not in seen]
    n_docs = sum(1 for r in todo for d in r["results"] if d.get("sents"))
    print(f"{len(todo)} claims / {n_docs} docs to re-read with {eur.READ_PROMPT_V} on {eur.VERIFICATION_MODEL} | "
          f"{args.workers} workers | gate {args.gate_start}..{args.gate_cap} | cap ${args.budget}", flush=True)
    lock = threading.Lock()
    st = {"done": 0, "spent": 0.0, "failed": 0, "stop": False}
    t0 = time.time()
    fh = OUT.open("a")

    def work(rec):
        if st["stop"]:
            return
        rec, cost = reread(rec)
        with lock:
            st["done"] += 1
            st["spent"] += cost
            st["failed"] += sum(1 for d in rec["results"] if d.get("read_status") == "failed")
            fh.write(json.dumps(rec) + "\n")
            n = st["done"]
            if n % 100 == 0 or n == len(todo):
                fh.flush()
                el = (time.time() - t0) / 60
                print(f"  {n}/{len(todo)} | spent ${st['spent']:.3f} | projected ${st['spent'] / n * len(todo):.2f} | "
                      f"{el:.1f}m | {n / max(el, .01):.0f} claims/min | ETA {(len(todo) - n) / max(n / max(el, .01), .01):.0f}m | "
                      f"failed reads {st['failed']} | gate {reader_lab.GATE.limit} inflight {reader_lab.GATE.inflight}", flush=True)
            if st["spent"] >= args.budget:
                st["stop"] = True
                print("  [budget cap reached, stopping]", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"done: {st['done']} claims, ${st['spent']:.3f}, {(time.time() - t0) / 60:.1f} min, "
          f"failed reads {st['failed']} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
