"""How much fc-gold FALSE belongs in the FALSE side of the fit?

    uv run python -m eval.scripts.build_eval.false_mix

$0, refits saved reads. The TRUE side is FIXED at the de-mixed timeline urn
(settled 2026-08-27: it beats the outlet corpus by 0.0198 AUC with a paired
interval clear of zero). The only thing that moves here is the FALSE side.

    p_FALSE(lam) = (1 - lam) * CN-false rates + lam * fc-gold-FALSE rates

lam 0 is the pure CN urn, lam 1 is pure gold FALSE, and the mixture is taken on
the RATE scale then put back on a common document total so the smoothing
strength does not drift with lam. This matters because the two corpora differ by
roughly 3x in documents, so a document-level pool is not a 50/50 mixture of
evidence -- it is whichever corpus is bigger.

TWO EVALS, both reported, because neither alone answers Daniel's question.
  gold   the pinned fc-gold population, n = 3,274. Out of fold on the gold-FALSE
         contribution whenever lam > 0, so no gold claim ever scores under
         weights its own row helped fit.
  urn    CN-false as FALSE against the timeline as TRUE, contamination-corrected
         at eps. Out of fold on BOTH urns whenever lam < 1.

The headline is AUC (Daniel's colleagues, 2026-08-27). recall at a 2% false
alarm budget is reported as the secondary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval import fit_urn
from eval.scripts.build_eval import model_ladder as ML
from eval.scripts.build_eval import transfer_ladder as TL
from eval.scripts.build_eval.cross_ladder import (demix_counts, fold_of, laplace,
                                                  recall_at, roc_urn)

# Env-overridable so the 2026-08-28 label-validity refit writes its own copy.
OUT = Path(__import__("os").environ.get("FALSE_MIX_OUT",
                      "eval/data/urn_runs/e1_ctx/model_ladder"))
LAMS = (0.0, 0.25, 0.50, 0.75, 1.0)
EPS = 0.10
K = fit_urn.K_FOLDS
MODELS = ML.MODELS
BOOT_REPS, BOOT_SEED = int(__import__("os").environ.get("REPS", 2000)), 707
BOOT_MODEL = "7-flag"


def mix_false(cCN: np.ndarray, cGF: np.ndarray, lam: float) -> np.ndarray:
    """Rate-scale mixture, returned on the CN urn's document scale."""
    if lam == 0.0:
        return cCN
    k = len(cCN)
    pCN = (cCN + 1) / (cCN.sum() + k)
    pGF = (cGF + 1) / (cGF.sum() + k)
    return ((1 - lam) * pCN + lam * pGF) * cCN.sum()


def main() -> None:
    gold = [r for r in ML.load_docs() if r["subtype"] != ML.SET_ASIDE]
    cn = TL.load_urn([TL.C2 / "scores.jsonl", TL.C2 / "scores_ext.jsonl"],
                     TL.C2 / "fit_exclusions.json")
    tl = TL.load_urn([TL.TL / "scores.jsonl"])

    y = np.array([r["y"] for r in gold])
    mid = np.array([r["mid"] for r in gold])
    gfold = np.array([r["fold"] for r in gold])
    cfold = np.array([fold_of(r["claim"]) for r in cn])
    tfold = np.array([fold_of(r["claim"]) for r in tl])
    fitrow = ~mid

    G = {m: ML.count_matrix(gold, m) for m in MODELS}
    CN = {m: TL.urn_matrix(cn, m) for m in MODELS}
    TLM = {m: TL.urn_matrix(tl, m) for m in MODELS}

    print(f"TRUE side fixed  de-mixed timeline, {len(tl):,} claims, eps {EPS:.2f}")
    print(f"FALSE side mixed CN {len(cn):,} claims / {int(CN['7-flag'].sum()):,} docs  "
          f"with gold-FALSE {int((fitrow & (y == 0)).sum()):,} claims / "
          f"{int(G['7-flag'][:, fitrow & (y == 0)].sum()):,} docs")
    print(f"EVAL gold {len(gold):,} (T {int(y.sum())} / F {int((y == 0).sum())})   "
          f"EVAL urn {len(cn):,} CN vs {len(tl):,} timeline, corrected\n")

    def cell(m, lam, ev, gsel=None, csel=None, tsel=None):
        gi = np.arange(len(gold)) if gsel is None else gsel
        ci = np.arange(len(cn)) if csel is None else csel
        ti = np.arange(len(tl)) if tsel is None else tsel
        Gm, CNm, TLm = G[m][:, gi], CN[m][:, ci], TLM[m][:, ti]
        yy, ff, gf = y[gi], fitrow[gi], gfold[gi]
        cf, tf = cfold[ci], tfold[ti]
        uses_gold = lam > 0.0
        uses_urn = lam < 1.0

        def w_for(mask_g, mask_c, mask_t):
            cGF = Gm[:, mask_g & ff & (yy == 0)].sum(1)
            cCN = CNm[:, mask_c].sum(1)
            dm = demix_counts(cCN, TLm[:, mask_t].sum(1), EPS)
            if dm is None:
                return None
            return laplace(dm, mix_false(cCN, cGF, lam))

        if ev == "gold":
            s = np.empty(len(gi))
            for k in range(K):
                mg = (gf != k) if uses_gold else np.ones(len(gi), bool)
                w = w_for(mg, np.ones(len(ci), bool), np.ones(len(ti), bool))
                if w is None:
                    return None
                te = gf == k
                s[te] = w @ Gm[:, te]
            r2, f2, _ = ML.recall_at_fpr(s, yy, 0.02)
            return {"auc": ML.auc_np(s[yy == 1], s[yy == 0]), "recall_2pct": r2,
                    "fpr_2pct": f2}

        s_cn, s_tl = np.empty(len(ci)), np.empty(len(ti))
        for k in range(K):
            mc = (cf != k) if uses_urn else np.ones(len(ci), bool)
            mt = (tf != k) if uses_urn else np.ones(len(ti), bool)
            w = w_for(np.ones(len(gi), bool), mc, mt)
            if w is None:
                return None
            s_cn[cf == k] = w @ CNm[:, cf == k]
            s_tl[tf == k] = w @ TLm[:, tf == k]
        x, yv, _ = roc_urn(s_cn, s_tl, EPS)
        a_obs = ML.auc_np(s_tl, s_cn)
        r2, f2 = recall_at(x, yv, 0.02)
        return {"auc": (a_obs - EPS / 2) / (1 - EPS), "auc_observed": a_obs,
                "recall_2pct": r2, "fpr_2pct": f2}

    res = {}
    for lam in LAMS:
        for ev in ("gold", "urn"):
            for m in MODELS:
                res.setdefault(f"{lam}", {}).setdefault(ev, {})[m] = cell(m, lam, ev)

    ref = json.loads((OUT / "ladder.json").read_text())["models"]
    print("AUC.  lam 0 = pure CN false, lam 1 = pure gold false. TRUE side is the timeline throughout.")
    print(f"{'':10s}" + "".join(f"{'eval ' + e:>35}" for e in ("gold", "urn")))
    print(f"{'lam':10s}" + "".join(f"{m:>11}" for e in ("gold", "urn") for m in MODELS))
    for lam in LAMS:
        line = f"{lam:<10.2f}"
        for ev in ("gold", "urn"):
            for m in MODELS:
                c = res[f"{lam}"][ev][m]
                line += f"{c['auc']:>11.4f}" if c else f"{'infeas':>11}"
        print(line)
    print(f"{'gold fit':10s}" + "".join(f"{ref[m]['auc_oof']:>11.4f}" for m in MODELS)
          + "   (fc-gold both sides, for reference)")

    print("\nrecall at a 2% false alarm budget")
    for lam in LAMS:
        line = f"{lam:<10.2f}"
        for ev in ("gold", "urn"):
            for m in MODELS:
                c = res[f"{lam}"][ev][m]
                line += f"{c['recall_2pct']*100:>10.1f}%" if c else f"{'--':>11}"
        print(line)

    print("\n7-flag weights as the FALSE side moves (fitted on everything, no folds)")
    F = ML.channels("7-flag")
    wtab = {}
    for lam in LAMS:
        cGF = G["7-flag"][:, fitrow & (y == 0)].sum(1)
        cCN = CN["7-flag"].sum(1)
        dm = demix_counts(cCN, TLM["7-flag"].sum(1), EPS)
        wtab[lam] = laplace(dm, mix_false(cCN, cGF, lam))
    print(f"{'flag':6s}" + "".join(f"{'lam ' + format(l, '.2f'):>12}" for l in LAMS)
          + f"{'gold fit':>12}")
    for i, f in enumerate(F):
        print(f"{f:6s}" + "".join(f"{wtab[l][i]:>+12.3f}" for l in LAMS)
              + f"{ref['7-flag']['weights'][f]:>+12.3f}")

    print(f"\nbootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}, {BOOT_MODEL}, "
          f"all corpora resampled) ...", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    reps = {ev: {l: [] for l in LAMS} for ev in ("gold", "urn")}
    for _ in range(BOOT_REPS):
        gi = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ci = rng.integers(0, len(cn), len(cn))
        ti = rng.integers(0, len(tl), len(tl))
        for ev in ("gold", "urn"):
            for lam in LAMS:
                c = cell(BOOT_MODEL, lam, ev, gi, ci, ti)
                reps[ev][lam].append(c["auc"] if c else np.nan)

    print(f"\n{BOOT_MODEL} AUC with 95% intervals, and the PAIRED delta against lam 0 (pure CN)")
    for ev in ("gold", "urn"):
        print(f"  eval {ev}")
        base = np.array(reps[ev][0.0])
        for lam in LAMS:
            a = res[f"{lam}"][ev][BOOT_MODEL]
            lo, hi = np.nanpercentile(reps[ev][lam], [2.5, 97.5])
            a["auc_ci"] = [float(lo), float(hi)]
            d = np.array(reps[ev][lam]) - base
            dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
            a["dauc_vs_lam0"] = a["auc"] - res["0.0"][ev][BOOT_MODEL]["auc"]
            a["dauc_ci"] = [float(dlo), float(dhi)]
            tag = "" if lam == 0.0 else (
                f"   d {a['dauc_vs_lam0']:+.4f} [{dlo:+.4f}, {dhi:+.4f}]"
                + ("" if dlo <= 0 <= dhi else "  *"))
            print(f"    lam {lam:.2f}  {a['auc']:.4f} [{lo:.4f}, {hi:.4f}]"
                  f"  rec@2% {a['recall_2pct']*100:5.1f}%{tag}")

    payload = {"eps": EPS, "lams": list(LAMS), "true_side": "demixed timeline",
               "boot": {"reps": BOOT_REPS, "seed": BOOT_SEED, "model": BOOT_MODEL},
               "sizes": {"gold": len(gold), "cn": len(cn), "timeline": len(tl)},
               "cells": res,
               "weights_7flag": {str(l): dict(zip(F, map(float, wtab[l]))) for l in LAMS}}
    (OUT / "false_mix.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT / 'false_mix.json'}")


if __name__ == "__main__":
    main()
