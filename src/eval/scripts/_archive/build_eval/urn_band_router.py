"""Three-band router analysis on the Truth Odds urn — how much can be skipped, at what cost.

The deployed prefilter is one threshold: score <= -4.5885 FLAGS a claim for the
expensive verification loop, everything above passes. That wastes work at the top
of the score range, where the urn is confidently supportive and the loop almost
never overturns it. This asks whether a second, upper threshold can retire those
claims WITHOUT verification, and prices the mistake.

    uv run python -m eval.scripts.build_eval.urn_band_router \
        -i eval/data/urn_runs/e1_ctx/results-00.jsonl

Bands, low to high:
    FLAG    score <= t_low     -> nudge / full verification (unchanged)
    CHECK   t_low < s < t_high -> the expensive loop decides
    PASS    score >= t_high    -> retired unverified

Everything is OUT OF FOLD, reusing fit_urn's fold assignment and weights, so the
band boundaries are not read off the same rows that fitted them. Labels follow the
2026-08-19 convention: mixed (veracity 3) counts as FALSE at eval, and stays out
of the weight fit.

The number that decides the router is the PASS band's MISS RATE: of all FALSE
claims in the gold set, what fraction the router retires unverified. Its complement
on the other side -- the share of the corpus the router never has to pay for -- is
the saving. Both are reported with denominators at every candidate boundary.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from eval.scripts.build_eval.fit_urn import (
    E1_METRICS_PATH, K_FOLDS, fit, load_headline, score)

# The deployed lower boundary (E1 oof, recall@FPR<=2%). Fixed, not re-swept: the
# router question is only about the UPPER boundary.
T_LOW = json.loads(E1_METRICS_PATH.read_text())["overall"]["threshold"]


def oof_rows(rows: list[dict]) -> list[dict]:
    """Attach an out-of-fold score to every row, keeping the row's own fields."""
    out = []
    for f in range(K_FOLDS):
        train = [r for r in rows if r["fold"] != f]
        test = [r for r in rows if r["fold"] == f]
        if not train or not test:
            continue
        w = fit(train)
        for r in test:
            out.append({**r, "s": score(r, w)})
    return out


def band_table(scored: list[dict], t_low: float, t_high: float) -> dict:
    n = len(scored)
    n_true = sum(r["y"] for r in scored)
    n_false = n - n_true
    bands = collections.defaultdict(list)
    for r in scored:
        b = "FLAG" if r["s"] <= t_low else ("PASS" if r["s"] >= t_high else "CHECK")
        bands[b].append(r)
    d = {"t_low": t_low, "t_high": t_high, "n": n, "n_true": n_true, "n_false": n_false}
    for b in ("FLAG", "CHECK", "PASS"):
        rs = bands[b]
        t = sum(r["y"] for r in rs)
        d[b] = {"n": len(rs), "share": len(rs) / n if n else 0.0,
                "n_true": t, "n_false": len(rs) - t,
                # of ALL false claims, what fraction lands in this band
                "false_captured": (len(rs) - t) / n_false if n_false else float("nan"),
                # of ALL true claims, what fraction lands in this band
                "true_captured": t / n_true if n_true else float("nan"),
                "purity_false": (len(rs) - t) / len(rs) if rs else float("nan")}
    d["miss_rate"] = d["PASS"]["false_captured"]          # falses retired unverified
    d["work_saved"] = d["PASS"]["share"]                  # corpus never paid for
    d["check_load"] = d["CHECK"]["share"]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=Path)
    ap.add_argument("-o", "--out", type=Path, help="write the sweep as JSON")
    ap.add_argument("--t-low", type=float, default=None)
    ap.add_argument("--slots", type=int, default=None)
    args = ap.parse_args()

    rows = load_headline(args.input, args.slots)
    scored = oof_rows(rows)
    n = len(scored)
    n_true = sum(r["y"] for r in scored)
    n_mid = sum(1 for r in scored if r.get("mid"))
    print(f"n={n} (TRUE {n_true} / FALSE {n - n_true}, of which mixed-as-false {n_mid})")
    print(f"out of fold, {K_FOLDS} folds, slot convention "
          f"{'N=' + str(args.slots) if args.slots else 'N=returned'}")
    print(f"t_low = {args.t_low:.4f} (deployed FLAG boundary, held fixed)\n")

    # Sweep the upper boundary over achievable scores. Report the frontier a
    # router designer actually chooses from: for each miss-rate budget, the
    # lowest t_high (= most work saved) that stays inside it.
    cands = sorted({round(r["s"], 6) for r in scored if r["s"] > args.t_low})
    sweep = [band_table(scored, args.t_low, t) for t in cands]

    print(f"{'t_high':>9} {'PASS n':>7} {'saved':>7} {'miss':>7} "
          f"{'missed F':>9} {'CHECK n':>8} {'FLAG n':>7}")
    budgets = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
    frontier = {}
    for b in budgets:
        ok = [d for d in sweep if d["miss_rate"] <= b]
        if not ok:
            print(f"  miss<={b:.1%}: unreachable")
            continue
        best = min(ok, key=lambda d: d["t_high"])       # lowest bar = most retired
        frontier[f"{b:.3f}"] = best
        print(f"{best['t_high']:9.3f} {best['PASS']['n']:7d} {best['work_saved']:7.1%} "
              f"{best['miss_rate']:7.2%} {best['PASS']['n_false']:9d} "
              f"{best['CHECK']['n']:8d} {best['FLAG']['n']:7d}   (budget {b:.1%})")

    print("\nfull band table at each frontier point:")
    for b, d in frontier.items():
        print(f"\n  miss budget {float(b):.1%}  ->  t_high {d['t_high']:.3f}")
        for band in ("FLAG", "CHECK", "PASS"):
            x = d[band]
            print(f"    {band:6s} n={x['n']:5d} ({x['share']:5.1%})  "
                  f"T {x['n_true']:4d} / F {x['n_false']:4d}   "
                  f"holds {x['false_captured']:5.1%} of all FALSE, "
                  f"{x['true_captured']:5.1%} of all TRUE, "
                  f"purity(false) {x['purity_false']:5.1%}")

    summary = {"input": str(args.input), "n": n, "n_true": n_true,
               "n_false": n - n_true, "n_mixed_as_false": n_mid,
               "folds": K_FOLDS, "slots": args.slots or "returned",
               "t_low": args.t_low, "frontier": frontier,
               "sweep": [d for d in sweep if d["PASS"]["n"] % 25 == 0]}
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
