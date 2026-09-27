"""Infer political lean + outlet kind for NG swap candidates (Daniel 2026-08-05).

NewsGuard's Orientation field is populated for only ~58% of candidates and never
emits "Center" — it is Left/Right only. Every Center slot in the E2 swap therefore
has no label at all, so lean must be inferred.

Design: infer for ALL candidates, including the ones NG already labels. The
NG-labelled subset is then a held-out check — agreement there tells us whether the
inferred labels on the unlabelled subset are worth acting on. Inferring only the
missing ones would give us numbers with no way to judge them.

Also emits `kind`, because the candidate pool contains outlets that are national
and political on paper but are opinion shops, single-author blogs, or defunct —
none of which can supply 1,000 original hard-news posts.

  uv run python -m eval.scripts.claim_sourcing.divine_outlet_lean

Reads  eval/data/survey_claims/ng_swap_candidates.csv
Writes eval/data/survey_claims/ng_swap_candidates_lean.csv
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval.evidence_urn_run import llm

IN = Path("eval/data/survey_claims/ng_swap_candidates.csv")
OUT = Path("eval/data/survey_claims/ng_swap_candidates_lean.csv")

# Abstract criteria only — no worked examples, no outlet names (they would leak the
# answer for the very rows we use to validate, and bias the rest toward the shapes
# named).
SYS = (
    "You identify US news outlets from their domain, brand name and subject areas.\n"
    "Return JSON only:\n"
    '{"lean": "Left|Center|Right|Unknown", "kind": "hard_news|opinion|niche|defunct|unknown", '
    '"confidence": "high|medium|low"}\n\n'
    "lean = the outlet's editorial orientation in US political terms. Center means it "
    "does not consistently favour either side, OR its subject matter is not political. "
    "Use Unknown only when the outlet is genuinely unrecognisable to you — do not guess "
    "from the domain name's connotations alone.\n\n"
    "kind:\n"
    "- hard_news: reports events, publishes multiple original stories daily\n"
    "- opinion: predominantly commentary, columns or advocacy\n"
    "- niche: narrow single-subject, single-author, or very low publication volume\n"
    "- defunct: known to have shut down, gone dormant, or merged into another outlet\n\n"
    "confidence = how sure you are of the identification itself, not of the lean.\n"
    "An outlet you do not recognise is Unknown/unknown/low — that is a useful answer, "
    "not a failure."
)


def one(row: dict) -> tuple[dict, float]:
    u = (f"domain: {row['domain']}\n"
         f"brand name: {row.get('Brand Name') or '(none given)'}\n"
         f"subject areas: {row.get('Topics') or '(none given)'}")
    for _ in range(3):
        try:
            obj, cost, _, _ = llm([{"role": "system", "content": SYS},
                                   {"role": "user", "content": u}],
                                  cache_key="lean", max_tokens=60)
            if obj.get("lean") in ("Left", "Center", "Right", "Unknown"):
                return {"lean_llm": obj["lean"],
                        "kind_llm": obj.get("kind") or "unknown",
                        "conf_llm": obj.get("confidence") or "low"}, cost
        except Exception:
            time.sleep(2)
    return {"lean_llm": "Unknown", "kind_llm": "unknown", "conf_llm": "low"}, 0.0


def main() -> None:
    df = pl.read_csv(IN)
    rows = df.to_dicts()
    print(f"divining lean/kind for {len(rows)} candidates", flush=True)
    lock, state = threading.Lock(), {"n": 0, "cost": 0.0}
    out, t0 = [None] * len(rows), time.time()

    def work(i_row):
        i, r = i_row
        v, cost = one(r)
        with lock:
            out[i] = {**r, **v}
            state["n"] += 1
            state["cost"] += cost
            if state["n"] % 100 == 0 or state["n"] == len(rows):
                el = time.time() - t0
                print(f"  {state['n']}/{len(rows)} | ${state['cost']:.3f} | "
                      f"{el/60:.1f}m | ETA {(len(rows)-state['n'])/max(state['n']/el,1e-9)/60:.0f}m",
                      flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(work, enumerate(rows)))

    res = pl.DataFrame([o for o in out if o])
    res.write_csv(OUT)

    # Held-out check against NG's own Left/Right labels.
    lab = res.filter(pl.col("lean_ng").is_in(["Left", "Right"]))
    agree = lab.filter(pl.col("lean_llm") == pl.col("lean_ng")).height
    print(f"\nNG-labelled rows: {lab.height} | inferred == NG: {agree} "
          f"({agree/max(lab.height,1):.1%})")
    if lab.height:
        print(lab.group_by(["lean_ng", "lean_llm"]).agg(pl.len().alias("n"))
              .sort(["lean_ng", "n"], descending=[False, True]))
    print()
    print(res.group_by("lean_llm").agg(pl.len().alias("n")).sort("n", descending=True))
    print(res.group_by("kind_llm").agg(pl.len().alias("n")).sort("n", descending=True))
    print(f"total ${state['cost']:.3f} | {(time.time()-t0)/60:.1f}m -> {OUT}")


if __name__ == "__main__":
    main()
