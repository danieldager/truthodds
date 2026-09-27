"""Score each claim's EXPECTED INDEPENDENT COVERAGE, before the urn run.

Daniel 2026-08-05: claim prominence is likely the largest single effect on whether
the urn finds evidence, larger than publisher reliability — so it must be measured,
shown, and controlled for, not discovered afterwards.

The smoke made the mechanism concrete. Three floridadaily claims about a state CFO's
routine announcements drew 30 documents and 29 "I" flags; three nytimes claims about
wildfires, a Berlin attack and Trump/Iran drew 30 documents and 25 directional reads.
Nothing about the instrument differed. What differed was whether the world reports
the underlying event at all.

This scores the construct that actually drives retrieval: **how many INDEPENDENT
outlets would be expected to have covered this**, judged from the claim alone, with
no web access and no knowledge of what our retrieval found. Scoring it blind to the
outcome is the point — a prominence label derived from the urn's own results could
not then explain them.

Deliberately NOT a proxy for engagement: like/retweet counts measure the post's
virality, which is a property of the tweet, not of the event's coverage. A viral post
about an obscure local incident still has no independent trace.

  uv run python -m eval.scripts.build_eval.claim_prominence            # the drawn set
  uv run python -m eval.scripts.build_eval.claim_prominence --smoke    # 50 rows

Writes eval/data/survey_claims/e2_claim_prominence.parquet
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval.evidence_urn_run import llm
from eval.scripts.build_eval.tweet_urn_run import (SCREEN_EXCLUDE, balanced_draw,
                                                   load_claims)

OUT = Path("eval/data/survey_claims/e2_claim_prominence.parquet")

# Abstract criteria only, no worked examples (they would anchor the scale to whatever
# shapes happened to be named).
SYS = (
    "You estimate how widely a factual claim's underlying subject matter would have been "
    "reported by news outlets OTHER than the one that published it.\n\n"
    "You are NOT judging whether the claim is true, important, or interesting. You are "
    "judging one thing: if someone searched the web for this, how many INDEPENDENT "
    "outlets would plausibly have written about it.\n\n"
    "Return JSON only:\n"
    '{"coverage": "none|few|many", "scope": "local|national|international", '
    '"actor": "obscure|regional|national|global"}\n\n'
    "coverage:\n"
    "- none: routine institutional business, local announcements, remarks made in a "
    "small venue or niche programme — the kind of thing only the originating outlet "
    "would bother to report\n"
    "- few: of interest to a specific region, sector or community; a handful of outlets "
    "or trade press would cover it\n"
    "- many: a nationally or internationally consequential event, figure or dispute that "
    "most major outlets would report\n\n"
    "scope = the geographic reach of the subject matter.\n"
    "actor = how widely known the principal person or organisation named in the claim is.\n\n"
    "Judge from the claim text alone. Do not speculate about the publisher."
)


def one(claim: str, date: str) -> tuple[dict, float]:
    u = f"CLAIM: {claim}" + (f"\nDATE: {date}" if date else "")
    for _ in range(3):
        try:
            obj, cost, _, _ = llm([{"role": "system", "content": SYS},
                                   {"role": "user", "content": u}],
                                  cache_key="prominence", max_tokens=50)
            if obj.get("coverage") in ("none", "few", "many"):
                return {"coverage": obj["coverage"],
                        "scope": obj.get("scope") or "unknown",
                        "actor": obj.get("actor") or "unknown"}, cost
        except Exception:
            time.sleep(2)
    return {"coverage": "unknown", "scope": "unknown", "actor": "unknown"}, 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    d = load_claims().filter(pl.col("verify_eligible"))
    drawn, _ = balanced_draw(d, 400, 707, 0.25)
    # Score the whole drawn set, INCLUDING screen-excluded and pair-collapsed rows:
    # the collapse direction is still open, so both members of every pair need a score.
    rows = drawn.to_dicts()
    if args.smoke:
        rows = rows[::max(1, len(rows) // 50)][:50]
    print(f"scoring {len(rows)} claims for expected coverage", flush=True)

    lock, st = threading.Lock(), {"n": 0, "c": 0.0}
    out, t0 = [None] * len(rows), time.time()

    def work(i_row):
        i, r = i_row
        v, cost = one(r["claim"], (r.get("created_at") or "")[:10])
        with lock:
            out[i] = {"claim_id": r["claim_id"], "bin": r["bin"], "domain": r["domain"],
                      "type": r["type"], "post_id": r["post_id"], **v}
            st["n"] += 1
            st["c"] += cost
            if st["n"] % 500 == 0 or st["n"] == len(rows):
                el = (time.time() - t0) / 60
                print(f"  {st['n']}/{len(rows)} | ${st['c']:.3f} | {el:.1f}m", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, enumerate(rows)))

    res = pl.DataFrame([o for o in out if o])
    if not args.smoke:
        res.write_parquet(OUT)
        print(f"\nwrote {OUT}")
    print(res.group_by("coverage").agg(pl.len().alias("n")).sort("n", descending=True))
    print()
    print("coverage x bin (the balance question):")
    print(res.pivot(on="coverage", index="bin", values="claim_id",
                    aggregate_function="len").sort("bin"))
    print()
    print("coverage x claim type:")
    print(res.pivot(on="coverage", index="type", values="claim_id",
                    aggregate_function="len"))
    print(f"total ${st['c']:.3f} | {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
