"""Which TRUE urn should the weights be fitted on? Three arms, one FALSE side.

    uv run python -m eval.scripts.build_eval.true_arms

$0, reuses saved reads on all four corpora. The FALSE side is the CN-false urn in
every arm, the instrument is identical everywhere (query-v3 / read-v5, context
off), and the eval is the pinned fc-gold population with NO refit, so the arms
differ only in where the TRUE-side rates come from.

  timeline   the raw X feed, de-mixed at eps. Right distribution, contaminated
             labels, and the de-mix assumes the contaminating falses score like
             CN falses -- the assumption the PI's objection targets.
  outlet     reputable-outlet tweets, presumed true by source. Clean labels, but
             a measured coverage confound (support 47.1% vs the timeline's 35.3%)
             and a register-blind AUC of 0.732 against CN-false.
  pooled     both, added at the document level.

A FOURTH arm was killed before it got here. CN acquitted trues (accused, note
rejected) audited blind at 44% FALSE [36%, 52%] over 160 claims, so note
rejection is not a truth label. See clog/270826.

The number that decides this is NOT the AUC. The arms are expected to rank
similarly and differ in level, so the weight tables side by side are the output
that matters; the AUC is the headline metric and is reported for it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval import model_ladder as ML
from eval.scripts.build_eval import transfer_ladder as TL
from eval.scripts.build_eval.cross_ladder import demix_counts, laplace

OUTLET = Path("eval/data/urn_runs/true_outlet/scores.jsonl")
# Env-overridable so the 2026-08-28 label-validity refit writes its own copy.
OUT = Path(__import__("os").environ.get("TRUE_ARMS_OUT",
                      "eval/data/urn_runs/e1_ctx/model_ladder"))
EPS = 0.10
BOOT_REPS, BOOT_SEED = 2000, 707
MODELS = ML.MODELS
ARMS = ("timeline", "outlet", "pooled")


def true_counts(arm, cCN, cTL, cOU, eps):
    if arm == "outlet":
        return cOU
    dm = demix_counts(cCN, cTL, eps)
    if dm is None:
        return None
    return dm if arm == "timeline" else dm + cOU


def main() -> None:
    cn = TL.load_urn([TL.C2 / "scores.jsonl", TL.C2 / "scores_ext.jsonl"],
                     TL.C2 / "fit_exclusions.json")
    tl = TL.load_urn([TL.TL / "scores.jsonl"])
    ou = TL.load_urn([OUTLET])
    gold = [r for r in ML.load_docs() if r["subtype"] != ML.SET_ASIDE]
    y = np.array([r["y"] for r in gold])

    print(f"FALSE  CN-false   {len(cn):,} claims")
    print(f"TRUE   timeline   {len(tl):,} | outlet {len(ou):,} | pooled {len(tl)+len(ou):,}")
    print(f"EVAL   fc-gold    {len(gold):,} (T {int(y.sum())} / F {int((y==0).sum())}), no refit\n")

    C = {m: TL.urn_matrix(cn, m) for m in MODELS}
    T = {m: TL.urn_matrix(tl, m) for m in MODELS}
    O = {m: TL.urn_matrix(ou, m) for m in MODELS}
    G = {m: ML.count_matrix(gold, m) for m in MODELS}

    F = ML.channels("7-flag")
    cols = {"CN false": C["7-flag"].sum(1), "timeline raw": T["7-flag"].sum(1),
            "timeline demixed": demix_counts(C["7-flag"].sum(1), T["7-flag"].sum(1), EPS),
            "outlet": O["7-flag"].sum(1)}
    print("flag rates, % of documents")
    print(f"{'flag':6s}" + "".join(f"{k:>19}" for k in cols))
    for i, f in enumerate(F):
        print(f"{f:6s}" + "".join(f"{v[i]/v.sum()*100:>18.2f}%" for v in cols.values()))
    print(f"{'docs':6s}" + "".join(f"{v.sum():>19,.0f}" for v in cols.values()))

    res = {}
    for arm in ARMS:
        for m in MODELS:
            cT = true_counts(arm, C[m].sum(1), T[m].sum(1), O[m].sum(1), EPS)
            if cT is None:
                res.setdefault(arm, {})[m] = None
                continue
            w = laplace(cT, C[m].sum(1))
            s = w @ G[m]
            a = ML.auc_np(s[y == 1], s[y == 0])
            r2, f2, t2 = ML.recall_at_fpr(s, y, 0.02)
            res.setdefault(arm, {})[m] = {
                "auc": a, "recall_2pct": r2, "fpr_2pct": f2, "threshold_2pct": t2,
                "weights": dict(zip(ML.channels(m), map(float, w))),
                "roc": [x.tolist() for x in ML.roc(s, y)]}

    print(f"\nfc-gold transfer, no refit ({len(gold):,} claims)")
    print(f"{'arm':12s}" + "".join(f"{m:>22}" for m in MODELS))
    for arm in ARMS:
        print(f"{arm:12s}" + "".join(
            f"{'AUC %.4f rec %4.1f%%' % (res[arm][m]['auc'], res[arm][m]['recall_2pct']*100):>22}"
            if res[arm][m] else f"{'infeasible':>22}" for m in MODELS))

    print(f"\nbootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}) ...")
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    a_reps = {a: {m: [] for m in MODELS} for a in ARMS}
    w_reps = {a: {m: [] for m in MODELS} for a in ARMS}
    for _ in range(BOOT_REPS):
        iC = rng.integers(0, len(cn), len(cn))
        iT = rng.integers(0, len(tl), len(tl))
        iO = rng.integers(0, len(ou), len(ou))
        iG = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy = y[iG]
        for m in MODELS:
            cC = C[m][:, iC].sum(1)
            for arm in ARMS:
                cT = true_counts(arm, cC, T[m][:, iT].sum(1), O[m][:, iO].sum(1), EPS)
                if cT is None:
                    a_reps[arm][m].append(np.nan)
                    continue
                w = laplace(cT, cC)
                w_reps[arm][m].append(w)
                s = w @ G[m][:, iG]
                a_reps[arm][m].append(ML.auc_np(s[yy == 1], s[yy == 0]))

    for arm in ARMS:
        for m in MODELS:
            if not res[arm][m]:
                continue
            lo, hi = np.nanpercentile(a_reps[arm][m], [2.5, 97.5])
            res[arm][m]["auc_ci"] = [float(lo), float(hi)]
            W = np.array(w_reps[arm][m])
            wl, wh = np.percentile(W, [2.5, 97.5], axis=0)
            res[arm][m]["weight_ci"] = {c: [float(a), float(b)]
                                        for c, a, b in zip(ML.channels(m), wl, wh)}
    for m in MODELS:
        for arm in ARMS[1:]:
            d = np.array(a_reps[arm][m]) - np.array(a_reps["timeline"][m])
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            res[arm][m]["dauc_vs_timeline"] = res[arm][m]["auc"] - res["timeline"][m]["auc"]
            res[arm][m]["dauc_ci"] = [float(lo), float(hi)]

    print("\nAUC with 95% intervals, and the paired delta against the timeline arm")
    for m in MODELS:
        print(f"  {m}")
        for arm in ARMS:
            r = res[arm][m]
            d = (f"   d {r['dauc_vs_timeline']:+.4f} [{r['dauc_ci'][0]:+.4f}, "
                 f"{r['dauc_ci'][1]:+.4f}]") if arm != "timeline" else ""
            print(f"    {arm:10s} {r['auc']:.4f} [{r['auc_ci'][0]:.4f}, {r['auc_ci'][1]:.4f}]"
                  f"  rec@2% {r['recall_2pct']*100:5.1f}%{d}")

    print("\n7-flag weights, the comparison that actually decides the arm")
    gw = json.loads((OUT / "ladder.json").read_text())["models"]["7-flag"]["weights"]
    print(f"{'flag':6s}" + "".join(f"{a:>26}" for a in ARMS) + f"{'gold-fitted':>14}")
    for f in F:
        row = f"{f:6s}"
        for arm in ARMS:
            r = res[arm]["7-flag"]
            lo, hi = r["weight_ci"][f]
            row += f"{'%+.2f [%+.2f,%+.2f]' % (r['weights'][f], lo, hi):>26}"
        print(row + f"{gw[f]:>+14.2f}")

    payload = {"eps": EPS, "boot": {"reps": BOOT_REPS, "seed": BOOT_SEED},
               "sizes": {"cn": len(cn), "timeline": len(tl), "outlet": len(ou),
                         "gold": len(gold)},
               "flag_rates": {k: {f: float(v[i]) for i, f in enumerate(F)}
                              for k, v in cols.items()},
               "arms": res}
    (OUT / "true_arms.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT / 'true_arms.json'}")


if __name__ == "__main__":
    main()
