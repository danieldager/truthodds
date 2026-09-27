"""Deliverable 2: diagnose the 28 cells + calibrate the analytic SE."""
import json, numpy as np
from bucketopt_core import *


def main():
    bootstrap()
    rows = load_gold(extra=True)
    y = np.array([r["y"] for r in rows]); mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    C28 = cell_matrix(rows)
    n_t, n_f = int(y.sum()), int((y==0).sum())
    print(f"population {len(rows)} claims (T {n_t} / F {n_f}), docs {int(C28.sum()):,}")
    assert (len(rows), n_t, n_f) == (3274, 1502, 1772)

    allidx = np.arange(len(rows))
    cT, cF = counts(C28, y, mid, allidx)
    w = loglr(cT, cF)
    se = se_analytic(cT, cF)
    tot = cT.sum() + cF.sum()

    # --- calibrate analytic vs the pinned 2000-rep bootstrap weight CIs -----------
    pin = json.loads(LADDER.read_text())["models"]["28-cell"]
    boot_hw = np.array([ (pin["weight_ci"][c][1]-pin["weight_ci"][c][0])/2 for c in CELL_NAME ])
    ana_hw = 1.96*se
    ratio = boot_hw/ana_hw
    INFL = float(np.median(ratio))
    print(f"\nanalytic vs bootstrap half-width: median ratio {INFL:.3f} "
          f"(IQR {np.percentile(ratio,25):.3f}-{np.percentile(ratio,75):.3f}) -> inflation {INFL:.3f}")

    # --- table -------------------------------------------------------------------
    order = np.argsort(-(cT+cF))
    print(f"\n{'cell':16s} {'Tdocs':>7s} {'Fdocs':>7s} {'%docs':>6s} {'logLR':>7s} "
          f"{'boot 95% CI':>18s} {'hw':>6s} {'ana hw':>7s}")
    tab = []
    for i in order:
        lo, hi = pin["weight_ci"][CELL_NAME[i]]
        print(f"{CELL_NAME[i]:16s} {int(cT[i]):7d} {int(cF[i]):7d} "
              f"{100*(cT[i]+cF[i])/tot:6.2f} {w[i]:+7.3f} "
              f"[{lo:+7.3f},{hi:+7.3f}] {boot_hw[i]:6.3f} {ana_hw[i]*INFL:7.3f}")
        tab.append({"cell": CELL_NAME[i], "docs_true": int(cT[i]), "docs_false": int(cF[i]),
                    "pct_docs": 100*(cT[i]+cF[i])/tot, "loglr": float(w[i]),
                    "boot_ci": [lo, hi], "boot_hw": float(boot_hw[i]),
                    "analytic_hw_inflated": float(ana_hw[i]*INFL)})

    # --- overlapping-CI pairs (merge candidates) ---------------------------------
    print("\npairs with OVERLAPPING bootstrap 95% CIs on the log-LR (merge candidates):")
    ov = []
    for i in range(NCELL):
        for j in range(i+1, NCELL):
            li,hi_ = pin["weight_ci"][CELL_NAME[i]]; lj,hj = pin["weight_ci"][CELL_NAME[j]]
            if li <= hj and lj <= hi_:
                ov.append((CELL_NAME[i], CELL_NAME[j], abs(w[i]-w[j]),
                           FLAG_OF[i]==FLAG_OF[j]))
    print(f"  {len(ov)} of {NCELL*(NCELL-1)//2} pairs overlap; "
          f"{sum(o[3] for o in ov)} of them share a flag")
    G = g2_matrix(cT, cF)
    iu = np.triu_indices(NCELL,1)
    gv = G[iu]
    sm = np.argsort(gv)[:15]
    print("\n15 least-supported splits by G2 (chi2 1df crit 3.84):")
    for k in sm:
        i, j = iu[0][k], iu[1][k]
        print(f"  {CELL_NAME[i]:14s} + {CELL_NAME[j]:14s}  G2 {gv[k]:8.2f}  "
              f"dw {w[i]-w[j]:+.3f}  {'same flag' if FLAG_OF[i]==FLAG_OF[j] else ''}")

    thin = [CELL_NAME[i] for i in range(NCELL) if ana_hw[i]*INFL > 0.35]
    print(f"\ncells failing hw<=0.35 (inflated analytic): {len(thin)}/28 -> {thin}")

    # --- extra axes availability -------------------------------------------------
    import collections
    rk = collections.Counter(); pv = collections.Counter()
    for r in rows:
        for d in r["docs"]:
            rk[min(d[2],99)] += 1; pv[d[3]] += 1
    print("\nrank distribution (top):", sorted(rk.items())[:12], "... n_distinct", len(rk))
    print("provenance distribution:", pv.most_common())

    json.dump({"inflation": INFL, "cells": tab,
               "n_overlapping_pairs": len(ov), "thin_cells_hw035": thin,
               "provenance": dict(pv), "rank_hist": {str(k): v for k, v in sorted(rk.items())}},
              open(OUT/"bucket_opt_diag.json","w"), indent=1)
    print("\nwrote", OUT/"bucket_opt_diag.json")


if __name__ == "__main__":
    main()
