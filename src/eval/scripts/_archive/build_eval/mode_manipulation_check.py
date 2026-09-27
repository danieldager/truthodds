"""Does the claim MODE field actually change QUERY and READ? (Daniel 2026-08-04)

Before adopting read-v5.1 / query-v4 we have to know the mode is load-bearing and
not decorative. This runs the SAME claims under BOTH modes and compares:

  QUERY  same claim, mode=attribution vs mode=assertion -> do the queries differ,
         and in the predicted direction (utterance/speaker vs substance)?
  READ   same claim AND the same saved documents (tranche 1's exact sentences, so
         no search and no scraping) -> do flags move in the predicted direction:
         under attribution, documents arguing the content's truth should fall to
         "X"; under assertion, documents that merely relay the claim should fall
         to "X".

Ambiguous claims are the informative ones: "X said Y" where Y is itself checkable.
Those are exactly where the two modes disagree about what to look for.

  uv run python -m eval.scripts.build_eval.mode_manipulation_check -n 12
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval.claim_modes import MODE_ASSERTION, MODE_ATTRIBUTION
from eval.scripts.build_eval.evidence_urn_run import QUERY_SYS, llm, read_doc

RUN = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
OUT = Path("eval/data/mode_manipulation_check.json")


def query_for(claim, mode, ctx, date):
    q = [f"CLAIM: {claim}", f"MODE: {mode}"]
    if ctx:
        q.append(f"CONTEXT (circulation of the claim, from a neutral description): {ctx}")
    if date:
        q.append(f"CLAIM DATE: {date}")
    obj, cost, _, _ = llm([{"role": "system", "content": QUERY_SYS},
                           {"role": "user", "content": "\n".join(q)}], max_tokens=80)
    return (obj.get("query") or "").strip(), cost


def block(claim, mode, date):
    return (f"CLAIM: {claim}\nMODE: {mode}\n(claimed on {date or 'unknown date'}; "
            f"judge the document's bearing on this exact proposition)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=12, help="ambiguous claims to test")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    recs = [json.loads(l) for l in RUN.open()]
    # Ambiguous = phrased as an utterance claim, with documents that took a
    # position. Those are the claims where mode should change the answer.
    cand = [r for r in recs
            if not r.get("excluded")
            and r.get("claim_type") == "quote_attribution"
            and sum(1 for d in r["results"]
                    if d["read_status"] == "ok"
                    and d["read"]["direction"] in ("5", "4", "3", "2", "1")) >= 2]
    cand.sort(key=lambda r: r["review_url"])
    cases = cand[:args.n]
    print(f"{len(cand)} ambiguous claims available; testing {len(cases)}", flush=True)

    lock, state, out = threading.Lock(), {"cost": 0.0, "n": 0}, []

    def work(r):
        claim, date = r["claim_resolved"], r.get("claim_date_shown")
        ctx = ""    # queries in the run had context; hold it constant across arms
        res = {"review_url": r["review_url"], "claim": claim,
               "recorded_mode": r.get("claim_mode"), "queries": {}, "reads": []}
        for m in (MODE_ATTRIBUTION, MODE_ASSERTION):
            try:
                q, c = query_for(claim, m, ctx, date)
            except Exception:
                q, c = "", 0.0
            with lock:
                state["cost"] += c
            res["queries"][m] = q
        # re-read the SAME saved documents under both modes
        for d in r["results"]:
            if d["read_status"] != "ok" or not d["sent_ids"]:
                continue
            if d["read"]["direction"] not in ("5", "4", "3", "2", "1"):
                continue
            row = {"rank": d["rank"], "domain": d["domain"], "run_flag": d["read"]["direction"]}
            for m in (MODE_ATTRIBUTION, MODE_ASSERTION):
                try:
                    o, c, _, _ = read_doc(block(claim, m, date), d["sent_ids"], d["sents"],
                                          key=f"modechk-{m}-{r['review_url']}")
                except Exception:
                    o, c = None, 0.0
                with lock:
                    state["cost"] += c
                row[m] = (o or {}).get("direction")
            res["reads"].append(row)
        with lock:
            out.append(res)
            state["n"] += 1
            print(f"  {state['n']}/{len(cases)} | ${state['cost']:.3f}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, cases))

    # --- did the field do anything? ---
    qdiff = sum(1 for r in out
                if r["queries"].get(MODE_ATTRIBUTION, "").lower()
                != r["queries"].get(MODE_ASSERTION, "").lower())
    print(f"\nQUERY: {qdiff}/{len(out)} claims produced a DIFFERENT query under the two modes")
    reads = [x for r in out for x in r["reads"]]
    moved = [x for x in reads if x.get(MODE_ATTRIBUTION) != x.get(MODE_ASSERTION)]
    print(f"READ: {len(moved)}/{len(reads)} documents changed flag between modes")
    # predicted direction: assertion should send RELAY documents to X more often
    to_x_assert = sum(1 for x in moved
                      if x.get(MODE_ASSERTION) == "X" and x.get(MODE_ATTRIBUTION) != "X")
    to_x_attrib = sum(1 for x in moved
                      if x.get(MODE_ATTRIBUTION) == "X" and x.get(MODE_ASSERTION) != "X")
    print(f"  -> X under assertion only: {to_x_assert} | -> X under attribution only: {to_x_attrib}")
    print("\nflag transitions (attribution -> assertion):",
          dict(collections.Counter(f"{x.get(MODE_ATTRIBUTION)}->{x.get(MODE_ASSERTION)}"
                                   for x in moved).most_common(8)))
    print("\nsample query pairs:")
    for r in out[:6]:
        print(f"  claim: {r['claim'][:80]}")
        print(f"    attribution: {r['queries'].get(MODE_ATTRIBUTION)}")
        print(f"    assertion  : {r['queries'].get(MODE_ASSERTION)}")
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT} | total ${state['cost']:.3f}")


if __name__ == "__main__":
    main()
