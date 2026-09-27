"""Verifier / spec for populations/fc_gold_rep1500.parquet.

This file has NO committed builder — its original build script (a 2026-09-14
scratchpad, gone) could not be recovered, and the exact id set is only PARTLY
reproducible from the recipe + the frozen data (see REPRODUCIBILITY below). So
rather than ship an approximate builder as if it were the real one, this script
(a) documents the recipe, (b) asserts every invariant of the pinned file, and
(c) reports the reconstruction overlap so the gap is explicit and checkable.

RECIPE (populations/manifest.json, clog/140926.md):
  seeded numpy default_rng(20260914); BALANCED 750 true / 750 false draw of 1,500
  claims, stratified by (judged_axis_llm, veracity); source = frozen
  fc_gold.parquet (3,280) minus 44 claim-shape failures = 3,236; true = veracity
  {4,5}, false = veracity {1,2} (veracity-3 excluded by the T/F requirement).
  Purpose: the pinned fitting population for read-v6.1 six-flag (and later the
  superset fc_gold_bal3000). rep1500 is git-IGNORED (not tracked); bal3000, which
  supersets it verbatim, IS tracked and reads it.

REPRODUCIBILITY (measured 2026-09-21, $0):
  Given the recipe + frozen data, the largest-remainder proportional allocation
  across (judged_axis_llm, veracity) strata reproduces the pinned per-stratum
  COUNTS exactly (24/24 strata). The TRUE side (750 ids, veracity {4,5}) then
  reproduces EXACTLY id-for-id with a fresh default_rng(20260914), strata visited
  in (veracity, axis) order, ids sorted within stratum, rng.choice without
  replacement. The FALSE side (750 ids, veracity {1,2}) is NOT reproducible this
  way: across >250 principled configs (rng API, alloc order, single vs per-side
  generator, 5 selection idioms, id sort, 5 stratum orders, seed sweep, pool =
  3,236 vs 3,280) the false overlap never exceeds ~376/750 and sits at the
  ~354.7 chance floor. The false-side selection carries a quirk from that day's
  code (the same day the sibling clean1500 had a superseded proportional draw)
  that is not recoverable from the recipe alone. Closest whole-file overlap:
  1107/1500 (= 750 true exact + ~357 false at chance).
"""
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(".")
POPDIR = SRC / "eval/data/populations"
FROZEN = POPDIR / "fc_gold.parquet"
REP1500 = POPDIR / "fc_gold_rep1500.parquet"
RESULTS = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
REGISTRY = SRC / "eval/data/exclusions/registry.parquet"
SEED = 20260914

PINNED_SHA256 = "644f154c433fe5fa730980e0d491d8a3f6b33453ac5db46b7ef6443288acedc5"
PINNED_VERACITY = {1: 715, 2: 35, 4: 130, 5: 620}


def shape_drops(frz: pd.DataFrame) -> set:
    """The 44 claim-shape review_urls removed to go from 3,280 to the frozen
    3,236, same disjoint order as build_fc_gold_bal3000.shape_drops."""
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


def alloc_take(sub: pd.DataFrame, n: int) -> dict:
    """Largest-remainder proportional allocation of n across
    (judged_axis_llm, veracity) strata (the counts the draw realises)."""
    strata = {k: list(v) for k, v in
              sub.groupby(["judged_axis_llm", "veracity"]).groups.items()}
    sizes = {k: len(v) for k, v in strata.items()}
    total = sum(sizes.values())
    raw = {k: n * s / total for k, s in sizes.items()}
    take = {k: int(np.floor(x)) for k, x in raw.items()}
    rem = n - sum(take.values())
    order = sorted(strata, key=lambda k: (raw[k] - take[k]), reverse=True)
    i = 0
    while rem > 0:
        k = order[i % len(order)]
        if take[k] < sizes[k]:
            take[k] += 1
            rem -= 1
        i += 1
    return strata, take


def draw_ver_axis(rng, sub, n):
    """The reconstruction that reproduces the TRUE side exactly (and the false
    side only at chance). Strata in (veracity, axis) order, ids sorted, choice."""
    strata, take = alloc_take(sub, n)
    picked = []
    for k in sorted(strata, key=lambda x: (x[1], str(x[0]))):
        ids_k = sorted(strata[k])
        if take[k]:
            picked += list(rng.choice(ids_k, size=take[k], replace=False))
    return set(picked)


def main():
    assert REP1500.exists(), f"missing {REP1500}"
    pinned_ids = list(pd.read_parquet(REP1500)["claim_id"])
    pinned = set(pinned_ids)
    sha = hashlib.sha256(REP1500.read_bytes()).hexdigest()

    frz = pd.read_parquet(FROZEN).set_index("claim_id")
    drops = shape_drops(frz)
    frozen = frz.drop(index=list(drops))
    pin = frz.loc[pinned_ids]

    # --- invariants of the pinned file ---
    assert len(pinned_ids) == 1500, f"n != 1500: {len(pinned_ids)}"
    assert len(pinned) == 1500, "duplicate claim_ids in pinned file"
    assert list(pd.read_parquet(REP1500).columns) == ["claim_id"], "cols != [claim_id]"
    assert pinned_ids == sorted(pinned_ids), "pinned file is not sorted by claim_id"
    assert len(drops) == 44, f"claim-shape drops != 44: {len(drops)}"
    assert len(frozen) == 3236, f"frozen != 3236: {len(frozen)}"
    assert pinned <= set(frozen.index), "pinned not a subset of the frozen 3,236"
    vc = {int(k): int(v) for k, v in pin.veracity.value_counts().sort_index().items()}
    assert vc == PINNED_VERACITY, f"veracity marginals {vc} != {PINNED_VERACITY}"
    n_true = int(pin.veracity.isin([4, 5]).sum())
    n_false = int(pin.veracity.isin([1, 2]).sum())
    assert n_true == 750 and n_false == 750, f"balance {n_true}/{n_false} != 750/750"

    # per-(axis, veracity) counts must equal the largest-remainder allocation
    tt = frozen[frozen.veracity.isin([4, 5])]
    ff = frozen[frozen.veracity.isin([1, 2])]
    _, take_t = alloc_take(tt, 750)
    _, take_f = alloc_take(ff, 750)
    take_all = {**take_t, **take_f}
    pin_counts = pin.groupby(["judged_axis_llm", "veracity"]).size().to_dict()
    keys = set(take_all) | set(pin_counts)
    mism = {k: (take_all.get(k, 0), pin_counts.get(k, 0))
            for k in keys if take_all.get(k, 0) != pin_counts.get(k, 0)}
    assert not mism, f"strata allocation != pinned counts: {mism}"

    # --- reconstruction overlap (reported, not asserted for the false side) ---
    rng = np.random.default_rng(SEED)
    rec_t = draw_ver_axis(rng, tt, 750)
    rng2 = np.random.default_rng(SEED)
    rec_f = draw_ver_axis(rng2, ff, 750)
    pin_t = set(pin[pin.veracity.isin([4, 5])].index)
    pin_f = set(pin[pin.veracity.isin([1, 2])].index)
    ov_t = len(rec_t & pin_t)
    ov_f = len(rec_f & pin_f)
    assert rec_t == pin_t, f"TRUE side no longer reproduces exactly: {ov_t}/750"

    if sha != PINNED_SHA256:
        print(f"WARNING: pinned sha256 {sha} != recorded {PINNED_SHA256} "
              "(file changed since 2026-09-21)")

    print("fc_gold_rep1500 invariants: ALL PASS")
    print(f"  n=1500  true/false=750/750  veracity={vc}")
    print(f"  subset of frozen 3,236: yes   strata==allocation: yes")
    print(f"  sha256={sha}")
    print(f"  reconstruction: TRUE side EXACT ({ov_t}/750); "
          f"FALSE side {ov_f}/750 (chance floor ~355 — not reproducible)")


if __name__ == "__main__":
    main()
