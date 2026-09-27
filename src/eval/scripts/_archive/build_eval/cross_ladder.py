"""Every fit against every eval, with the timeline urn's contamination corrected.

    uv run python -m eval.scripts.build_eval.cross_ladder

$0. Closes the square opened by model_ladder (gold fit, gold eval) and
transfer_ladder (urn fit, gold eval). Three fits x two evals x three bucketings.

FITS
  gold   fc-gold labels, Laplace log-LR, mixed and unprovable out of the fit
  urn    CN-false urn vs the de-mixed timeline urn at eps, no gold label
  pool   both, added at the document level. The FALSE side is gold-FALSE plus
         CN, the TRUE side is gold-TRUE plus the de-mixed timeline. Pooling is
         therefore weighted by document count, which is roughly 32k gold against
         39k urn, so neither corpus dominates.

EVALS
  gold   the pinned fc-gold population (media axis out, mixed out), n = 3,274
  urn    CN-false claims as FALSE against timeline claims as TRUE

THE CORRECTION. The timeline urn is not a clean TRUE set, a share eps of it is
false, so a raw urn-side AUC is biased down. If the contaminating claims score
like the CN-false urn does, then at any threshold

    FPR_obs = (1-eps) FPR_true + eps * Recall      =>  FPR_true = (FPR_obs - eps*Recall)/(1-eps)

which corrects the whole ROC pointwise, and integrating it gives the closed form

    AUC_true = (AUC_obs - eps/2) / (1 - eps)

Both are reported. The correction assumes only that a false claim sitting in the
timeline scores like a false claim in the CN urn, which is the same assumption
the rate-level de-mixing already makes.

WHAT IS NOT DONE HERE. The band audit read the bottom 400 timeline claims and
called 192 of them false. Relabelling those would raise the urn-side AUC, but the
audit only ever looked BELOW a score cut, so relabelling exactly where the score
already said false inflates the number by construction. It is reported once, at
the bottom, as a clearly marked upper bound, and it is not the headline.

Folds. Out of fold whenever the fit and the eval share a corpus, on all of it
otherwise. Bootstrap resamples every corpus a cell depends on.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval import fit_urn
from eval.scripts.build_eval import model_ladder as ML
from eval.scripts.build_eval import transfer_ladder as TL

_ENV = __import__("os").environ
# EPS_HEAD/EPS_SWEEP/CROSS_OUT are env-overridable so the measured-eps re-run
# (timeline_eps_audit.py, 2026-08-28) can point the sweep at its own number and
# its own output file without touching the pinned cross.json.
OUT = Path(_ENV.get("CROSS_OUT", "eval/data/urn_runs/e1_ctx/model_ladder"))
BAND_AUDIT = Path("eval/data/urn_runs/true_timeline/band_audit.jsonl")
EPS_SWEEP = tuple(float(x) for x in _ENV["EPS_SWEEP"].split(",")) \
    if _ENV.get("EPS_SWEEP") else (0.0, 0.05, 0.10, 0.15)
EPS_HEAD = float(_ENV.get("EPS_HEAD", 0.10))
BOOT_REPS, BOOT_SEED = int(__import__("os").environ.get("REPS", 2000)), 707
K = fit_urn.K_FOLDS
MODELS = ML.MODELS
FITS = ("gold", "urn", "pool")
EVALS = ("gold", "urn")


def fold_of(text: str) -> int:
    return int(hashlib.blake2b(text.encode(), digest_size=8).hexdigest(), 16) % K


def laplace(cT: np.ndarray, cF: np.ndarray) -> np.ndarray:
    k = len(cT)
    return np.log(((cT + 1) / (cT.sum() + k)) / ((cF + 1) / (cF.sum() + k)))


def demix_counts(cCN: np.ndarray, cTL: np.ndarray, eps: float) -> np.ndarray | None:
    """Timeline document counts with the false share removed, kept on the
    timeline's own scale so they can be pooled with gold counts."""
    k = len(cCN)
    pF = (cCN + 1) / (cCN.sum() + k)
    pM = (cTL + 1) / (cTL.sum() + k)
    pT = (pM - eps * pF) / (1 - eps)
    if (pT <= 0).any():
        return None
    return pT * cTL.sum()


def weights(fit: str, gT, gF, cCN, cTL, eps):
    """Column sums in, one weight vector out. None if the de-mix is infeasible."""
    if fit == "gold":
        return laplace(gT, gF)
    dm = demix_counts(cCN, cTL, eps)
    if dm is None:
        return None
    if fit == "urn":
        return laplace(dm, cCN)
    return laplace(gT + dm, gF + cCN)


def roc_urn(s_cn: np.ndarray, s_tl: np.ndarray, eps: float):
    """Detection framing, low score means false. Recall over the CN urn, false
    alarms over the timeline urn, corrected for the timeline's false share."""
    thr = np.unique(np.concatenate([s_cn, s_tl]))
    rec = np.searchsorted(np.sort(s_cn), thr, side="right") / len(s_cn)
    fpr_obs = np.searchsorted(np.sort(s_tl), thr, side="right") / len(s_tl)
    fpr = np.clip((fpr_obs - eps * rec) / (1 - eps), 0.0, 1.0)
    fpr = np.maximum.accumulate(fpr)          # keep it a curve after clipping
    return np.r_[0.0, fpr, 1.0], np.r_[0.0, rec, 1.0], thr


def recall_at(x: np.ndarray, y: np.ndarray, budget: float) -> tuple[float, float]:
    ok = x <= budget
    return (float(y[ok].max()), float(x[ok].max())) if ok.any() else (0.0, 0.0)


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

    print(f"gold eval {len(gold):,} (T {int(y.sum())} / F {int((y == 0).sum())})")
    print(f"urn eval  {len(cn):,} CN-false vs {len(tl):,} timeline, "
          f"corrected at eps {EPS_HEAD:.2f}\n")

    G = {m: ML.count_matrix(gold, m) for m in MODELS}
    CN = {m: TL.urn_matrix(cn, m) for m in MODELS}
    TLM = {m: TL.urn_matrix(tl, m) for m in MODELS}
    # Gold fit rows only: mixed is already gone, unprovable stays out of fits.
    fitrow = ~mid

    def cell(m, fit, ev, eps, gsel=None, csel=None, tsel=None):
        """One (fit, eval) number. gsel/csel/tsel are bootstrap index arrays."""
        gi = np.arange(len(gold)) if gsel is None else gsel
        ci = np.arange(len(cn)) if csel is None else csel
        ti = np.arange(len(tl)) if tsel is None else tsel
        Gm, CNm, TLm = G[m][:, gi], CN[m][:, ci], TLM[m][:, ti]
        yy, ff, gf = y[gi], fitrow[gi], gfold[gi]
        cf, tf = cfold[ci], tfold[ti]
        uses_gold = fit in ("gold", "pool")
        uses_urn = fit in ("urn", "pool")

        def w_for(mask_g, mask_c, mask_t):
            gT = Gm[:, mask_g & ff & (yy == 1)].sum(1)
            gF = Gm[:, mask_g & ff & (yy == 0)].sum(1)
            return weights(fit, gT, gF, CNm[:, mask_c].sum(1), TLm[:, mask_t].sum(1), eps)

        if ev == "gold":
            s = np.empty(len(gi))
            for k in range(K):
                mg = gf != k if uses_gold else np.ones(len(gi), bool)
                w = w_for(mg, np.ones(len(ci), bool), np.ones(len(ti), bool))
                if w is None:
                    return None
                te = gf == k
                s[te] = w @ Gm[:, te]
            x, yv = ML.roc(s, yy)
            a = ML.auc_np(s[yy == 1], s[yy == 0])
            r2, f2, _ = ML.recall_at_fpr(s, yy, 0.02)
            return {"auc": a, "recall_2pct": r2, "fpr_2pct": f2,
                    "roc": [x.tolist(), yv.tolist()]}

        s_cn, s_tl = np.empty(len(ci)), np.empty(len(ti))
        for k in range(K):
            mc = cf != k if uses_urn else np.ones(len(ci), bool)
            mt = tf != k if uses_urn else np.ones(len(ti), bool)
            w = w_for(np.ones(len(gi), bool), mc, mt)
            if w is None:
                return None
            s_cn[cf == k] = w @ CNm[:, cf == k]
            s_tl[tf == k] = w @ TLm[:, tf == k]
        x, yv, _ = roc_urn(s_cn, s_tl, eps)
        a_obs = ML.auc_np(s_tl, s_cn)          # timeline as the positive class
        a_corr = (a_obs - eps / 2) / (1 - eps)
        r2, f2 = recall_at(x, yv, 0.02)
        return {"auc": a_corr, "auc_observed": a_obs, "recall_2pct": r2,
                "fpr_2pct": f2, "roc": [x.tolist(), yv.tolist()]}

    # ---- headline matrix + eps sweep ----------------------------------------
    res = {}
    for fit in FITS:
        for ev in EVALS:
            for m in MODELS:
                res.setdefault(f"{fit}->{ev}", {})[m] = cell(m, fit, ev, EPS_HEAD)

    print("AUC, rows are what the weights were fitted on, columns are what they were scored on")
    print(f"{'':16s}" + "".join(f"{'eval ' + e:>34}" for e in EVALS))
    print(f"{'':16s}" + "".join(f"{m:>11}" for e in EVALS for m in MODELS))
    for fit in FITS:
        line = f"fit {fit:12s}"
        for ev in EVALS:
            for m in MODELS:
                c = res[f"{fit}->{ev}"][m]
                line += f"{c['auc']:>11.4f}" if c else f"{'infeasible':>11}"
        print(line)
    print("\nrecall at a 2% false alarm budget")
    for fit in FITS:
        line = f"fit {fit:12s}"
        for ev in EVALS:
            for m in MODELS:
                c = res[f"{fit}->{ev}"][m]
                line += f"{c['recall_2pct']*100:>10.1f}%" if c else f"{'--':>11}"
        print(line)

    sweep = {}
    for eps in EPS_SWEEP:
        for fit in FITS:
            for m in MODELS:
                c = cell(m, fit, "urn", eps)
                sweep.setdefault(str(eps), {}).setdefault(fit, {})[m] = (
                    {k: v for k, v in c.items() if k != "roc"} if c else None)
    print(f"\nurn-side AUC by assumed false share (corrected / observed)")
    for eps in EPS_SWEEP:
        row = f"  eps {eps*100:>4.0f}%  "
        for fit in FITS:
            c = sweep[str(eps)][fit]["7-flag"]
            row += f"{fit} {c['auc']:.3f}/{c['auc_observed']:.3f}   " if c else f"{fit} -- "
        print(row)

    # ---- bootstrap ----------------------------------------------------------
    print(f"\nbootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}) ...")
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    reps = {f"{f}->{e}": {m: [] for m in MODELS} for f in FITS for e in EVALS}
    for _ in range(BOOT_REPS):
        gi = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ci = rng.integers(0, len(cn), len(cn))
        ti = rng.integers(0, len(tl), len(tl))
        for fit in FITS:
            for ev in EVALS:
                for m in MODELS:
                    c = cell(m, fit, ev, EPS_HEAD, gi, ci, ti)
                    reps[f"{fit}->{ev}"][m].append(c["auc"] if c else np.nan)
    for key in reps:
        for m in MODELS:
            if res[key][m] is None:
                continue
            lo, hi = np.nanpercentile(reps[key][m], [2.5, 97.5])
            res[key][m]["auc_ci"] = [float(lo), float(hi)]
            res[key][m]["boot_infeasible"] = int(np.isnan(reps[key][m]).sum())

    print("\nAUC with 95% intervals")
    for key in [f"{f}->{e}" for f in FITS for e in EVALS]:
        print(f"  {key}")
        for m in MODELS:
            c = res[key][m]
            if not c:
                print(f"    {m:9s} de-mix infeasible")
                continue
            print(f"    {m:9s} {c['auc']:.4f} [{c['auc_ci'][0]:.4f}, {c['auc_ci'][1]:.4f}]"
                  f"   recall@2% {c['recall_2pct']*100:5.1f}%"
                  + (f"   observed {c['auc_observed']:.4f}" if "auc_observed" in c else ""))

    # ---- band-audit relabel, reported as an upper bound only ----------------
    # NB `audit` is keyed on claim_text[:80], so `n_audited` / `n_called_false` below
    # are JOIN counts, not audit counts: three of the 400 audited claims collide on
    # that key, which is why they read 397 and 190. Quote band_audit.jsonl itself
    # (400 read, 192 called false) anywhere those figures are shown to a reader.
    audit = {}
    if BAND_AUDIT.exists():
        for line in BAND_AUDIT.open():
            r = json.loads(line)
            audit[r["claim_text"][:80]] = r["verdict"]
        called_false = {k for k, v in audit.items() if v == "false"}
        keep = np.array([r["claim"][:80] not in called_false for r in tl])
        n_rm = int((~keep).sum())
        # Removing identified falses also removes contamination, so the residual
        # share must come down or the correction is applied twice.
        eps_rest = max(0.0, (EPS_HEAD * len(tl) - n_rm) / max(1, len(tl) - n_rm))
        print(f"\nband-audit sensitivity (UPPER BOUND, score-conditional): "
              f"{len(audit)} timeline claims audited, {len(called_false)} called false, "
              f"{n_rm} matched and removed, residual eps {eps_rest:.3f}")
        band = {"n_removed": n_rm, "eps_residual": eps_rest}
        for fit in FITS:
            for m in MODELS:
                ti = np.where(keep)[0]
                c = cell(m, fit, "urn", eps_rest, None, None, ti)
                band.setdefault(fit, {})[m] = ({k: v for k, v in c.items() if k != "roc"}
                                               if c else None)
            print(f"  fit {fit:6s} " + "  ".join(
                f"{m} {band[fit][m]['auc']:.4f}" if band[fit][m] else f"{m} --"
                for m in MODELS))
    else:
        band = {}

    payload = {"eps_head": EPS_HEAD, "boot": {"reps": BOOT_REPS, "seed": BOOT_SEED},
               "sizes": {"gold": len(gold), "cn": len(cn), "timeline": len(tl)},
               "cells": res, "eps_sweep": sweep,
               "band_audit_upper_bound": {"n_audited": len(audit),
                                          "n_called_false": len(called_false) if audit else 0,
                                          "cells": band}}
    (OUT / "cross.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT / 'cross.json'}")


if __name__ == "__main__":
    main()
