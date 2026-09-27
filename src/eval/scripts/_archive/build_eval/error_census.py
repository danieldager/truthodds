"""Error census over every mis-scored E1 claim (review-step design brief, step 0).

Mechanism vocabulary was locked 2026-08-21 after two independent 25-claim
open-coding passes converged (same top-3 classes, near-identical counts).
Code-detectable classes are tagged WITHOUT the LLM; the LLM judges only claims
that carry directional reads.

    uv run python -m eval.scripts.build_eval.error_census --smoke 25
    uv run python -m eval.scripts.build_eval.error_census

Population: gold-TRUE with oof score < 0, plus hard-FALSE (veracity 1/2) with
score > 0. Mixed (3) excluded -- labelling-convention question, not instrument.
Output: eval/data/urn_runs/e1_ctx/error_census.parquet + funnel.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval.evidence_urn_run import llm
from eval.scripts.build_eval.fit_urn import FLAG_TO_VOICE, load
from eval.scripts.build_eval.urn_band_router import oof_rows

E1 = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
OUT = Path("eval/data/urn_runs/e1_ctx/error_census.parquet")

LABELS = ["gold_axis", "origin_echo", "adjacent_mismatch", "time_window",
          "polarity_inversion", "modality", "hedge_stripped",
          "self_contradiction", "x_underuse", "hard_thin", "other"]

SYS = (
    "You audit why an evidence-based veracity score pointed the WRONG way for a "
    "claim. You get the claim, its gold verdict, and the retrieved documents: "
    "domain, date, reliability, the direction flag a reader assigned "
    "(5/4 support, 1/2 refute, 3/X/I none), and the sentences the reader cited.\n\n"
    "Return JSON only:\n"
    '{"mechanism": "<label>", "why": "<12 words>", "confidence": "high|medium|low"}\n\n'
    "mechanism -- the PRIMARY cause, one of:\n"
    "- gold_axis: the gold verdict rates a different proposition than the claim "
    "text states (typically the provenance of a photo, video, list or document, "
    "while the claim keeps only the underlying event, which the evidence "
    "correctly treats)\n"
    "- origin_echo: the deciding support comes from the claim's own source, the "
    "claimant, or multiple near-identical copies of one story\n"
    "- adjacent_mismatch: a deciding flag rests on a different event, entity, "
    "instrument, route or sub-proposition than the claim's\n"
    "- time_window: a deciding flag rests on a document whose reference period "
    "differs from the claim's (earlier snapshot of a moving quantity, different "
    "year, sub-interval of a cumulative window)\n"
    "- polarity_inversion: the claim asserts a failure/absence and documents "
    "reporting that same outcome were flagged as refuting it\n"
    "- modality: a proposal, plan, debate or possibility was flagged as "
    "establishing a completed action\n"
    "- hedge_stripped: a qualifier in the claim (scope, population, degree) was "
    "ignored by a flat directional flag\n"
    "- self_contradiction: a deciding flag contradicts its own cited sentence\n"
    "- x_underuse: documents that state the claim's content were flagged X/I as "
    "mere background, starving a true claim of support\n"
    "- hard_thin: the retrieved evidence genuinely cannot settle the claim; no "
    "reading error\n"
    "- other: none of the above fits; say what you see in `why`\n\n"
    "Judge only from the given record. Pick the mechanism that most changed the "
    "score, not every flaw present."
)


def population():
    rows = load(E1, None)
    keys = []
    for line in E1.open():
        r = json.loads(line)
        if r.get("veracity") not in (1, 2, 3, 4, 5):
            continue
        c = collections.Counter()
        for d in r.get("results") or []:
            v = FLAG_TO_VOICE.get((d.get("read") or {}).get("direction"))
            if v:
                c[v] += 1
        if not sum(c.values()):
            continue
        keys.append(r["review_url"])
    assert len(keys) == len(rows)
    for r, k in zip(rows, keys):
        r["review_url"] = k
    scored = oof_rows(rows)
    return [r for r in scored
            if (r["y"] == 1 and r["s"] < 0)
            or (r["y"] == 0 and not r.get("mid") and r["s"] > 0)]


def digest(rec: dict) -> str:
    lines = [f"CLAIM: {rec.get('claim_resolved') or rec.get('claim_text')}",
             f"claimed on: {rec.get('claim_date_shown')}",
             f"GOLD: {'TRUE' if rec['veracity'] >= 4 else 'FALSE'} (veracity {rec['veracity']})",
             "", "DOCUMENTS:"]
    for d in rec.get("results") or []:
        rd = d.get("read") or {}
        if not d.get("sent_ids"):
            continue
        m = {i: s for i, s in zip(d["sent_ids"], d["sents"])}
        cited = [m[i] for i in (rd.get("evidence") or []) if i in m][:3]
        lines.append(f"- {d['domain']} | {d.get('date') or 'undated'} | "
                     f"NG {d.get('ng')} | flag {rd.get('direction')}"
                     + (f" | qc {rd['qc_flag']}" if rd.get("qc_flag") else ""))
        for c in cited:
            lines.append(f"    \"{c[:260]}\"")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget", type=float, default=1.0)
    args = ap.parse_args()

    pop = population()
    n_true = sum(r["y"] for r in pop)
    print(f"mis-scored population: {len(pop)} (gold-TRUE pulled neg {n_true} / "
          f"hard-FALSE pushed pos {len(pop) - n_true})")

    # code-detected class: no directional read at all
    silent = [r for r in pop if r["n_t"] + r["n_f"] == 0]
    todo = [r for r in pop if r["n_t"] + r["n_f"] > 0]
    print(f"code-tagged no_evidence: {len(silent)}  |  LLM-judged: {len(todo)}")

    by_url = {}
    want = {r["review_url"] for r in todo}
    for line in E1.open():
        r = json.loads(line)
        if r.get("review_url") in want:
            by_url[r["review_url"]] = r
    if args.smoke:
        todo = todo[:args.smoke]

    meter = {"cost": 0.0, "done": 0, "stop": False}
    out = []
    t0 = time.time()

    def work(row):
        if meter["stop"]:
            return
        rec = by_url[row["review_url"]]
        try:
            obj, cost, _, _ = llm(
                [{"role": "system", "content": SYS},
                 {"role": "user", "content": digest(rec)}],
                cache_key="error-census-v1", max_tokens=160)
            mech = obj.get("mechanism")
            out.append({"review_url": row["review_url"],
                        "gold": "TRUE" if row["y"] else "FALSE",
                        "oof_score": row["s"],
                        "mechanism": mech if mech in LABELS else "other",
                        "why": obj.get("why"),
                        "confidence": obj.get("confidence")})
            meter["cost"] += cost
        except Exception as e:
            print(f"  [failed] {row['review_url'][:60]}: {type(e).__name__}", flush=True)
        meter["done"] += 1
        if meter["done"] % 100 == 0 or meter["done"] == len(todo):
            el = time.time() - t0
            proj = meter["cost"] / meter["done"] * len(todo)
            print(f"  {meter['done']}/{len(todo)} | ${meter['cost']:.3f} "
                  f"(proj ${proj:.2f}) | {meter['done']/max(el,1e-9)*60:.0f}/min", flush=True)
            if proj > args.budget and meter["done"] >= 25:
                print("  BUDGET ABORT", flush=True)
                meter["stop"] = True

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    df = pd.DataFrame(
        out + [{"review_url": r["review_url"],
                "gold": "TRUE" if r["y"] else "FALSE", "oof_score": r["s"],
                "mechanism": "no_evidence", "why": "no directional read",
                "confidence": "code"} for r in silent])
    print(f"\n${meter['cost']:.4f} | {(time.time()-t0)/60:.1f}m")
    print("\nmechanism x gold:")
    print(pd.crosstab(df.mechanism, df.gold, margins=True).to_string())
    if not args.smoke:
        df.to_parquet(OUT)
        print(f"\nwrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
