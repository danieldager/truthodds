"""Banded precision audit on the timeline urn (Workstream TH, TH1c/TH5).

Scores every timeline-urn claim with the two-urn 7-flag weights (eps=0.05),
shows the s7 distribution so bands can be fixed on sight (TH5), and runs the
SYNTHESIS stage over the lowest-scoring claims: claim + its read dossiers ->
{verdict true/false/unsure, confidence, reason, key_evidence}. Score-blind —
the model never sees flags, directions, or s7 — so the verdict is an
independent second reading of the same evidence, not an echo of the score.

    uv run python -m eval.scripts.build_eval.tl_band_audit --dist
    uv run python -m eval.scripts.build_eval.tl_band_audit --smoke 3
    uv run python -m eval.scripts.build_eval.tl_band_audit --run --below -3.2
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

RUN_DIR = SRC / "eval/data/urn_runs/true_timeline"
SCORES = RUN_DIR / "scores.jsonl"
FIT = RUN_DIR / "two_urn_fit.json"
OUT = RUN_DIR / "band_audit.jsonl"
EPS = 0.05

SYNTH_SYS = """You are a fact-checking analyst delivering a final verdict on a claim, \
given evidence dossiers collected from a web search. Each dossier is an excerpt from one \
source document, with its domain and publication date.

Weigh the dossiers JOINTLY, do not tally them:
- Independent corroboration from reputable, unrelated sources outweighs volume. Mirrors \
and syndicated copies of one report count once.
- A direct, specific refutation from a strong source can outweigh many loose confirmations.
- Distinguish the claim being REPORTED (someone said/claimed X) from the claim being TRUE. \
If the claim asserts a fact, coverage that merely attributes it to someone is weak support.
- Mind dates: evidence predating the claimed event cannot confirm it; later authoritative \
corrections supersede earlier reports.
- Absence of coverage is weak evidence either way for niche topics, but telling for claims \
that would be major news if true.

Verdict semantics:
- "true": the evidence establishes the claim is accurate as stated.
- "false": the evidence establishes it is inaccurate, fabricated, or misleading as stated \
(including materially wrong numbers, wrong attribution, or a real fact framed to assert \
something false).
- "unsure": the dossiers are insufficient, contradictory, or off-point.

Respond with JSON only:
{"verdict": "true"|"false"|"unsure", "confidence": 1-5, "reason": "<=40 words",
 "key_evidence": [dossier numbers that decided it]}"""


def load_fit_weights() -> dict:
    fit = json.loads(FIT.read_text())
    for f in fit["fits7"]:
        if abs(f["eps"] - EPS) < 1e-9 and f.get("weights"):
            return f["weights"]
    raise SystemExit(f"no fits7 entry at eps={EPS}")


def load_scored(w: dict) -> list[dict]:
    rows = []
    for line in SCORES.open():
        rec = json.loads(line)
        flags = [d["read"]["direction"] for d in rec["results"]
                 if d.get("read") and d["read"].get("direction")]
        if not flags:
            continue
        s7 = sum(w.get(f, 0.0) for f in flags)
        rows.append({"rec": rec, "s7": s7, "n_docs": len(flags)})
    rows.sort(key=lambda r: r["s7"])
    return rows


def dossier_block(rec: dict) -> str:
    parts = []
    for i, d in enumerate(rec["results"], 1):
        sents = " ".join(d.get("sents") or [])[:900]
        date = d.get("date") or "undated"
        parts.append(f"[{i}] {d['domain']} ({date})\n{sents}")
    return "\n\n".join(parts)


def synth(row: dict) -> dict:
    rec = row["rec"]
    ceiling = rec.get("ceiling") or ""
    user = (f"CLAIM: {rec['claim_text']}\n"
            f"CLAIM DATE: {ceiling or 'unknown'}\n\n"
            f"DOSSIERS:\n{dossier_block(rec)}")
    obj, cost, _, _ = llm(
        [{"role": "system", "content": SYNTH_SYS},
         {"role": "user", "content": user}],
        cache_key="tl-band-synth", max_tokens=300)
    v = obj.get("verdict")
    return {"review_url": rec["review_url"], "claim_text": rec["claim_text"],
            "frame": rec.get("frame"), "handle": rec.get("handle"),
            "s7": round(row["s7"], 4), "n_docs": row["n_docs"],
            "verdict": v if v in ("true", "false", "unsure") else "unsure",
            "confidence": obj.get("confidence"), "reason": obj.get("reason", ""),
            "key_evidence": obj.get("key_evidence", []), "cost": cost}


def dist(rows: list[dict], th: list[dict]) -> None:
    import statistics
    ss = [r["s7"] for r in rows]
    n = len(ss)
    print(f"n={n} claims scored (s7, two-urn eps={EPS} weights)")
    qs = [0, 1, 2, 5, 10, 25, 50, 75, 90, 100]
    for q in qs:
        i = min(n - 1, max(0, int(q / 100 * n)))
        print(f"  p{q:<3} {ss[i]:+8.3f}")
    print(f"  mean {statistics.mean(ss):+8.3f}")
    for t in th:
        below = sum(1 for s in ss if s <= t["threshold"])
        print(f"  transfer thr {t['threshold']:+7.3f} (fpr<={t['fpr_budget']:.1%}): "
              f"{below} claims below ({below / n:.1%} of urn)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--below", type=float, default=None,
                    help="synthesize claims with s7 <= this")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    w = load_fit_weights()
    rows = load_scored(w)
    th = json.loads(FIT.read_text())["thresholds"]["7-flag"]

    if args.dist:
        dist(rows, th)
        return

    if args.smoke:
        todo = rows[:args.smoke]
    elif args.run and args.below is not None:
        done = set()
        if OUT.exists():
            done = {json.loads(l)["review_url"] for l in OUT.open()}
        todo = [r for r in rows if r["s7"] <= args.below
                and r["rec"]["review_url"] not in done]
    else:
        ap.error("pick --dist, --smoke N, or --run --below T")

    lock = threading.Lock()
    spent, n_done, t0 = 0.0, 0, time.time()
    mode = "a" if (args.run and OUT.exists()) else "w"
    sink = OUT.open(mode) if args.run else None
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(synth, r): r for r in todo}
        for fut in as_completed(futs):
            out = fut.result()
            with lock:
                spent += out["cost"]
                n_done += 1
                if sink:
                    sink.write(json.dumps(out) + "\n")
                    sink.flush()
                if args.smoke:
                    print(json.dumps(out, indent=1))
                elif n_done % 25 == 0:
                    el = (time.time() - t0) / 60
                    print(f"  {n_done}/{len(todo)} | ${spent:.3f} | {el:.1f}m "
                          f"| {n_done / el:.0f}/min", flush=True)
    if sink:
        sink.close()
    print(f"done {n_done} | ${spent:.3f}")
    if args.run:
        import collections
        vc = collections.Counter(json.loads(l)["verdict"] for l in OUT.open())
        print("verdicts:", dict(vc))


if __name__ == "__main__":
    main()
