"""Synthesis verdicts over the fc-gold E1 population (two-stage flagging test).

Same score-blind synthesis stage as tl_band_audit.py, run on every headline-
population gold claim. Output feeds two evaluations: synthesis accuracy vs gold
(calibrates the band audit) and the two-stage flag rule (score gate + synthesis
confirm).

    uv run python -m eval.scripts.build_eval.gold_synth --smoke 5
    uv run python -m eval.scripts.build_eval.gold_synth --run
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from eval.scripts.build_eval.evidence_urn_run import llm  # noqa: E402
from eval.scripts.build_eval.tl_band_audit import SYNTH_SYS, dossier_block  # noqa: E402
from eval.scripts.build_eval.fit_urn import load_judged_axis, MEDIA_AXIS  # noqa: E402

E1 = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
OUT = SRC / "eval/data/urn_runs/e1_ctx/synth_verdicts.jsonl"


def headline_recs() -> list[dict]:
    axis = load_judged_axis()
    out = []
    for line in E1.open():
        r = json.loads(line)
        if r.get("veracity") not in (1, 2, 3, 4, 5):
            continue
        if (axis.get(r["review_url"]) or "untagged") == MEDIA_AXIS:
            continue
        if any(d.get("read") and d["read"].get("direction") for d in r.get("results") or []):
            out.append(r)
    return out


def synth(rec: dict) -> dict:
    claim = rec.get("claim_resolved") or rec.get("claim_text") or ""
    user = (f"CLAIM: {claim}\n"
            f"CLAIM DATE: {rec.get('ceiling') or 'unknown'}\n\n"
            f"DOSSIERS:\n{dossier_block(rec)}")
    obj, cost, _, _ = llm(
        [{"role": "system", "content": SYNTH_SYS},
         {"role": "user", "content": user}],
        cache_key="gold-synth", max_tokens=300)
    v = obj.get("verdict")
    return {"review_url": rec["review_url"], "veracity": rec["veracity"],
            "verdict": v if v in ("true", "false", "unsure") else "unsure",
            "confidence": obj.get("confidence"), "reason": obj.get("reason", ""),
            "cost": cost}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    recs = headline_recs()
    done = set()
    if args.run and OUT.exists():
        done = {json.loads(l)["review_url"] for l in OUT.open()}
    todo = [r for r in recs if r["review_url"] not in done]
    if args.smoke:
        todo = todo[:args.smoke]
    print(f"population {len(recs)} | todo {len(todo)}", flush=True)

    lock = threading.Lock()
    state = {"n": 0, "cost": 0.0}
    t0 = time.time()
    sink = OUT.open("a") if args.run else None
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(synth, r) for r in todo]
        for fut in as_completed(futs):
            o = fut.result()
            with lock:
                state["n"] += 1
                state["cost"] += o["cost"]
                if sink:
                    sink.write(json.dumps(o) + "\n")
                    sink.flush()
                if args.smoke:
                    print(json.dumps(o, indent=1), flush=True)
                elif state["n"] % 200 == 0:
                    el = (time.time() - t0) / 60
                    print(f"  {state['n']}/{len(todo)} | ${state['cost']:.3f} | "
                          f"{el:.1f}m | {state['n']/el:.0f}/min | "
                          f"ETA {(len(todo)-state['n'])/(state['n']/el):.1f}m", flush=True)
    if sink:
        sink.close()
    print(f"done {state['n']} | ${state['cost']:.3f}", flush=True)


if __name__ == "__main__":
    main()
