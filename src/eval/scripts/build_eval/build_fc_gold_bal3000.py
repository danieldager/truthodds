"""Build fc_gold_bal3000.parquet — a 3,000-claim balanced slice (1,500 true /
1,500 false) that SUPERSETS fc_gold_rep1500 for the seven-flag read-v5 ceiling
refit (2026-09-16 revert to the graded seven-flag urn).

Composition:
  - all 1,500 rep1500 claims (750 true / 750 false), verbatim;
  - + 750 true / 750 false drawn from the "other" 1,736 frozen claims, stratified
    by judged_axis_llm x veracity, numpy default_rng seed 20260916.

Source pool = frozen fc_gold (3,236) = fc_gold.parquet (3,280) minus the 44
claim-shape failures, reconstructed with the SAME disjoint order as
build_fc_gold_filter_review.py / graded_urn's clean subset:
  resolution_status in {needs-article, unresolvable-list, self-attributed} +
  claim_screen_unresolved_referent + claim_screen_misattributed_media +
  claim_screen_media_locus.

Binary gold (fc_gold.parquet `y`): true = veracity {4,5}, false = veracity
{1,2,3} (veracity-3 unprovable is y=0, mid=True). Frozen 3,236 = 1,534 T / 1,702 F.
Other = 1,736; drawing 750 T + 750 F drops 34 T + 202 F.

Writes fc_gold_bal3000.parquet (claim_id only, matching rep1500) and
fc_gold_bal3000.manifest.json next to it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(".")
POPDIR = SRC / "eval/data/populations"
FROZEN = POPDIR / "fc_gold.parquet"
REP1500 = POPDIR / "fc_gold_rep1500.parquet"
RESULTS = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
REGISTRY = SRC / "eval/data/exclusions/registry.parquet"
OUT = POPDIR / "fc_gold_bal3000.parquet"
MANIFEST = POPDIR / "fc_gold_bal3000.manifest.json"
SEED = 20260916


def shape_drops(frz: pd.DataFrame) -> set:
    """The 44 claim-shape review_urls removed to go from 3,280 to the frozen 3,236,
    in the same disjoint order as graded_urn's clean subset."""
    rs = {}
    for line in RESULTS.open():
        r = json.loads(line)
        rs[r["review_url"]] = r.get("resolution_status")
    resolution = pd.Series(rs)
    reg = pd.read_parquet(REGISTRY)

    def ids(rule):
        return set(reg[reg.rule == rule].claim_id)

    keep = set(frz.index)
    a1 = set(resolution[resolution.isin(
        ["needs-article", "unresolvable-list", "self-attributed"])].index) & keep
    keep -= a1
    a2 = ids("claim_screen_unresolved_referent") & keep
    keep -= a2
    a3 = (ids("claim_screen_misattributed_media") | ids("claim_screen_media_locus")) & keep
    return a1 | a2 | a3


def alloc(rng, sub: pd.DataFrame, n: int) -> list:
    """Draw n claim_ids from sub, stratified by (judged_axis_llm, veracity):
    allocate n across strata proportional to size (largest-remainder rounding),
    sample without replacement within each stratum."""
    strata = {k: list(v) for k, v in
              sub.groupby(["judged_axis_llm", "veracity"]).groups.items()}
    sizes = {k: len(v) for k, v in strata.items()}
    total = sum(sizes.values())
    raw = {k: n * s / total for k, s in sizes.items()}
    take = {k: int(np.floor(x)) for k, x in raw.items()}
    rem = n - sum(take.values())
    # distribute the remainder by largest fractional part, capped at stratum size
    order = sorted(strata, key=lambda k: (raw[k] - take[k]), reverse=True)
    i = 0
    while rem > 0:
        k = order[i % len(order)]
        if take[k] < sizes[k]:
            take[k] += 1
            rem -= 1
        i += 1
        if i > 10 * len(order) * (rem + 1):
            break
    picked = []
    for k in sorted(strata, key=lambda x: (str(x[0]), x[1])):
        ids_k = sorted(strata[k])
        chosen = rng.choice(ids_k, size=take[k], replace=False) if take[k] else []
        picked.extend(list(chosen))
    return picked


def main():
    frz = pd.read_parquet(FROZEN).set_index("claim_id")
    rep = set(pd.read_parquet(REP1500)["claim_id"])
    drops = shape_drops(frz)
    assert len(drops) == 44, f"expected 44 claim-shape drops, got {len(drops)}"
    frozen = frz.drop(index=list(drops))
    assert rep <= set(frozen.index), "rep1500 not fully inside frozen 3,236"
    assert len(frozen) == 3236, f"frozen != 3236: {len(frozen)}"

    other = frozen.drop(index=list(rep))
    assert len(other) == 1736, f"other != 1736: {len(other)}"

    rng = np.random.default_rng(SEED)
    other_t = other[other.y == 1]
    other_f = other[other.y == 0]
    draw_t = alloc(rng, other_t, 750)
    draw_f = alloc(rng, other_f, 750)
    assert len(draw_t) == 750 and len(draw_f) == 750
    assert len(set(draw_t)) == 750 and len(set(draw_f)) == 750

    bal = sorted(rep | set(draw_t) | set(draw_f))
    assert len(bal) == 3000, f"bal != 3000: {len(bal)}"
    balf = frozen.loc[bal]
    assert int((balf.y == 1).sum()) == 1500 and int((balf.y == 0).sum()) == 1500

    pd.DataFrame({"claim_id": bal}).to_parquet(OUT, index=False)

    dropped = other.drop(index=list(draw_t) + list(draw_f))
    dropped_t = sorted(dropped[dropped.y == 1].index)
    dropped_f = sorted(dropped[dropped.y == 0].index)
    assert len(dropped_t) == 34 and len(dropped_f) == 202

    def strata_counts(df):
        g = df.groupby(["judged_axis_llm", "veracity"]).size()
        return {f"{a}|v{int(v)}": int(c) for (a, v), c in g.items()}

    manifest = {
        "built_by": "eval/scripts/build_eval/build_fc_gold_bal3000.py",
        "rows": 3000,
        "n_true": 1500,
        "n_false": 1500,
        "seed": SEED,
        "rng": "numpy default_rng",
        "gold": "true = veracity {4,5} (y=1); false = veracity {1,2,3} (y=0, v3=mid)",
        "rules": [
            "SUPERSET of fc_gold_rep1500 (all 1,500 rep claims kept verbatim: 750 T / 750 F)",
            "+ 750 T / 750 F drawn from the other 1,736 frozen claims, stratified by "
            "judged_axis_llm x veracity, numpy default_rng seed 20260916, without replacement",
            "source pool = frozen fc_gold 3,236 = fc_gold.parquet 3,280 minus 44 claim-shape "
            "failures (resolution needs-article/unresolvable-list/self-attributed + "
            "claim_screen_unresolved_referent + claim_screen_misattributed_media/media_locus), "
            "same disjoint order as graded_urn's clean subset / build_fc_gold_filter_review.py",
            "PURPOSE: fitting/eval population for the read-v5 SEVEN-flag ceiling refit "
            "(2026-09-16 revert to the graded 7-flag urn); doubles rep1500 while staying balanced",
        ],
        "pool": {"frozen_3236": 3236, "rep1500": 1500, "other": 1736,
                 "other_true": int((other_t).shape[0]), "other_false": int((other_f).shape[0])},
        "composition": {
            "veracity": {int(k): int(v) for k, v in
                         balf.veracity.value_counts().sort_index().items()},
            "axis": {k: int(v) for k, v in balf.judged_axis_llm.value_counts().items()},
            "rating_subtype": {k: int(v) for k, v in balf.rating_subtype.value_counts().items()},
        },
        "drawn_strata_true": strata_counts(other_t.loc[draw_t]),
        "drawn_strata_false": strata_counts(other_f.loc[draw_f]),
        "dropped": {
            "n_true": 34, "n_false": 202,
            "true_claim_ids": dropped_t,
            "false_claim_ids": dropped_f,
        },
        "inputs": {
            "fc_gold.parquet": FROZEN.name,
            "fc_gold_rep1500.parquet": REP1500.name,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    print(f"WROTE {OUT} ({len(bal)} rows: {int((balf.y==1).sum())} T / {int((balf.y==0).sum())} F)")
    print(f"WROTE {MANIFEST}")
    print("veracity:", manifest["composition"]["veracity"])
    print("axis:", manifest["composition"]["axis"])
    print("dropped: 34 T / 202 F")


if __name__ == "__main__":
    main()
