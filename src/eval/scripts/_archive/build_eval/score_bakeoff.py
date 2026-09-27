"""Score the Haiku-vs-Sonnet labeller bake-off against Daniel's gold labels.

Input: eval/data/read_suite/bakeoff_results.json — [{cid, model, direction,
evidence, reason, notes}] from the workflow fleet (blind, context-free agents).

Per model: direction exact / coarse agreement with Daniel, evidence-bucket recall
and Jaccard vs the union of his three id lists (recall is the score that matters —
highlighter rule says supersets are fine), per-failure-mode breakdown, and the
STOP-rule check: high cross-model agreement + low Daniel agreement = the 25-July
correlated-bias pattern, halt and rethink instead of trusting the labellers.

  uv run python -m eval.scripts.build_eval.score_bakeoff
"""
from __future__ import annotations

import collections
import json

D2F = {"supports": "5", "partially-supports": "4", "partially-refutes": "2",
       "refutes": "1", "context": "X", "irrelevant": "I"}
COARSE = {"5": "S", "4": "S", "3": "C", "2": "R", "1": "R", "X": "C", "I": "I"}


def main() -> None:
    res = json.load(open("eval/data/read_suite/bakeoff_results.json"))
    gold = {l["cid"]: l for l in
            json.load(open("eval/data/read_suite/calibration_gold.json"))["labels"]
            if l["status"] == "gold"}
    raw = json.load(open("eval/data/read_suite/calibration.json"))
    mode = {c["cid"]: c.get("_mode", "?") for c in
            (raw["calibration"] if isinstance(raw, dict) else raw)}

    by = collections.defaultdict(dict)          # cid -> model -> label
    for r in res:
        by[r["cid"]][r["model"]] = r

    stats = {m: collections.Counter() for m in ("haiku", "sonnet")}
    permode = collections.defaultdict(lambda: collections.Counter())
    cross_agree = cross_n = both_wrong_agree = 0
    disagreements = []

    for cid, g in gold.items():
        gf = D2F[g["direction"]]
        gset = set(g.get("supports_ids") or []) | set(g.get("refutes_ids") or []) \
            | set(g.get("context_ids") or [])
        labels = by.get(cid) or {}
        for m in ("haiku", "sonnet"):
            r = labels.get(m)
            if not r:
                stats[m]["missing"] += 1
                continue
            stats[m]["n"] += 1
            ex = r["direction"] == gf
            co = COARSE[r["direction"]] == COARSE[gf]
            stats[m]["exact"] += ex
            stats[m]["coarse"] += co
            permode[mode[cid]][f"{m}-n"] += 1
            permode[mode[cid]][f"{m}-coarse"] += co
            if not g.get("direction_only") and gset:
                ev = set(r.get("evidence") or [])
                stats[m]["bucket_n"] += 1
                stats[m]["recall_sum"] += len(ev & gset) / len(gset)
                stats[m]["jacc_sum"] += len(ev & gset) / max(1, len(ev | gset))
            if not co:
                disagreements.append((cid, m, gf, r["direction"], mode[cid]))
        h, s = labels.get("haiku"), labels.get("sonnet")
        if h and s:
            cross_n += 1
            agree = COARSE[h["direction"]] == COARSE[s["direction"]]
            cross_agree += agree
            if agree and COARSE[h["direction"]] != COARSE[gf]:
                both_wrong_agree += 1

    for m in ("haiku", "sonnet"):
        st = stats[m]
        n = st["n"] or 1
        bn = st["bucket_n"] or 1
        print(f"{m:7s} n={st['n']} exact {st['exact']}/{n} = {st['exact']/n:.1%}  "
              f"coarse {st['coarse']}/{n} = {st['coarse']/n:.1%}  "
              f"bucket recall {st['recall_sum']/bn:.2f}  jaccard {st['jacc_sum']/bn:.2f}"
              + (f"  MISSING {st['missing']}" if st["missing"] else ""))
    print(f"\ncross-model coarse agreement: {cross_agree}/{cross_n} = "
          f"{cross_agree/max(1,cross_n):.1%}; agree-but-both-wrong: {both_wrong_agree} "
          f"(STOP-rule watch: high agree + low Daniel-match = correlated bias)")

    print("\nper mode (coarse hits / n):")
    for md, c in sorted(permode.items()):
        print(f"  {md:34s} haiku {c['haiku-coarse']}/{c['haiku-n']}   "
              f"sonnet {c['sonnet-coarse']}/{c['sonnet-n']}")

    print("\ncoarse disagreements with Daniel:")
    for cid, m, gf, af, md in sorted(disagreements):
        print(f"  {cid[:8]} {m:6s} gold={gf} agent={af}  [{md}]")


if __name__ == "__main__":
    main()
