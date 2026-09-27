"""Issue #24: cross-corpus transfer matrix extended with the 12-cell model.

Reproduces cross_ladder.py's 3 fits x 2 evals for 3-voice / 7-flag / 28-cell and
adds 12-cell (3-voice x the same 4 source tiers). Identical fitting, folds, eps
handling, eps_head and bootstrap (2,000 reps, seed 707) -- the pure functions are
IMPORTED from cross_ladder, only the model list changes. Read-only: writes one
JSON into eval/data/urn_runs/e1_ctx/model_ladder/issue24/, never over cross.json.

Sanity gate: 3-voice / 7-flag / 28-cell AUC must match the pinned cross.json to
+-0.001 in every one of the six cells, else exit(2).
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from eval.scripts.build_eval import cross_ladder as CL
from eval.scripts.build_eval import fit_urn
from eval.scripts.build_eval import model_ladder as ML
from eval.scripts.build_eval import transfer_ladder as TL
from eval.scripts.build_eval.quality_urn import VOICE

OUT = Path("eval/data/urn_runs/e1_ctx/model_ladder/issue24")
PINNED = Path("eval/data/urn_runs/e1_ctx/model_ladder/cross.json")

# The new model. ML.channels / ML.count_matrix / TL.urn_matrix all dispatch
# through ML.KEYS, so registering it here is enough.
ML.KEYS["12-cell"] = lambda fl, tr: f"{ML.VOICE_NAME[VOICE[fl]]} | {tr}"
MODELS = ("3-voice", "7-flag", "28-cell", "12-cell")
FITS, EVALS = CL.FITS, CL.EVALS
K = CL.K
EPS_HEAD = CL.EPS_HEAD
BOOT_REPS, BOOT_SEED = CL.BOOT_REPS, CL.BOOT_SEED

_S: dict = {}     # process-global corpus state, set by _setup()


def _setup() -> dict:
    gold = [r for r in ML.load_docs() if r["subtype"] != ML.SET_ASIDE]
    cn = TL.load_urn([TL.C2 / "scores.jsonl", TL.C2 / "scores_ext.jsonl"],
                     TL.C2 / "fit_exclusions.json")
    tl = TL.load_urn([TL.TL / "scores.jsonl"])
    y = np.array([r["y"] for r in gold])
    return {"gold": gold, "cn": cn, "tl": tl, "y": y,
            "mid": np.array([r["mid"] for r in gold]),
            "gfold": np.array([r["fold"] for r in gold]),
            "cfold": np.array([CL.fold_of(r["claim"]) for r in cn]),
            "tfold": np.array([CL.fold_of(r["claim"]) for r in tl]),
            "G": {m: ML.count_matrix(gold, m) for m in MODELS},
            "CN": {m: TL.urn_matrix(cn, m) for m in MODELS},
            "TLM": {m: TL.urn_matrix(tl, m) for m in MODELS}}


def thr_at(x: np.ndarray, thr: np.ndarray, budget: float):
    """Threshold of the operating point CL.recall_at picks (last x <= budget)."""
    ok = np.where(x <= budget)[0]
    if not len(ok):
        return None
    i = int(ok.max())
    return float(thr[i - 1]) if 1 <= i <= len(thr) else None


def cell(m, fit, ev, eps, gsel=None, csel=None, tsel=None, full=False):
    """Byte-for-byte CL.main().cell, plus the threshold when full=True."""
    S = _S
    gold, cn, tl = S["gold"], S["cn"], S["tl"]
    y, mid, gfold, cfold, tfold = S["y"], S["mid"], S["gfold"], S["cfold"], S["tfold"]
    gi = np.arange(len(gold)) if gsel is None else gsel
    ci = np.arange(len(cn)) if csel is None else csel
    ti = np.arange(len(tl)) if tsel is None else tsel
    Gm, CNm, TLm = S["G"][m][:, gi], S["CN"][m][:, ci], S["TLM"][m][:, ti]
    yy, ff, gf = y[gi], (~mid)[gi], gfold[gi]
    cf, tf = cfold[ci], tfold[ti]
    uses_gold = fit in ("gold", "pool")
    uses_urn = fit in ("urn", "pool")

    def w_for(mask_g, mask_c, mask_t):
        gT = Gm[:, mask_g & ff & (yy == 1)].sum(1)
        gF = Gm[:, mask_g & ff & (yy == 0)].sum(1)
        return CL.weights(fit, gT, gF, CNm[:, mask_c].sum(1), TLm[:, mask_t].sum(1), eps)

    if ev == "gold":
        s = np.empty(len(gi))
        for k in range(K):
            mg = gf != k if uses_gold else np.ones(len(gi), bool)
            w = w_for(mg, np.ones(len(ci), bool), np.ones(len(ti), bool))
            if w is None:
                return None
            te = gf == k
            s[te] = w @ Gm[:, te]
        a = ML.auc_np(s[yy == 1], s[yy == 0])
        if not full:
            return {"auc": a}
        r2, f2, t2 = ML.recall_at_fpr(s, yy, 0.02)
        return {"auc": a, "recall_2pct": r2, "fpr_2pct": f2, "threshold_2pct": t2}

    s_cn, s_tl = np.empty(len(ci)), np.empty(len(ti))
    for k in range(K):
        mc = cf != k if uses_urn else np.ones(len(ci), bool)
        mt = tf != k if uses_urn else np.ones(len(ti), bool)
        w = w_for(np.ones(len(gi), bool), mc, mt)
        if w is None:
            return None
        s_cn[cf == k] = w @ CNm[:, cf == k]
        s_tl[tf == k] = w @ TLm[:, tf == k]
    a_obs = ML.auc_np(s_tl, s_cn)              # timeline as the positive class
    a_corr = (a_obs - eps / 2) / (1 - eps)
    if not full:
        return {"auc": a_corr}
    x, yv, thr = CL.roc_urn(s_cn, s_tl, eps)
    r2, f2 = CL.recall_at(x, yv, 0.02)
    return {"auc": a_corr, "auc_observed": a_obs, "recall_2pct": r2,
            "fpr_2pct": f2, "threshold_2pct": thr_at(x, thr, 0.02)}


def _init():
    global _S
    _S = _setup()


def _rep(args):
    gi, ci, ti = args
    return [[[cell(m, fit, ev, EPS_HEAD, gi, ci, ti) for m in MODELS]
             for ev in EVALS] for fit in FITS]


def main() -> None:
    os.chdir(SRC)
    os.environ.setdefault("GOLD_EXCLUSIONS", "none")   # pinned pre-purge gold, n=3,274
    global _S
    _S = _setup()
    y = _S["y"]
    print(f"gold eval {len(_S['gold']):,} (T {int(y.sum())} / F {int((y == 0).sum())})")
    print(f"urn eval  {len(_S['cn']):,} CN-false vs {len(_S['tl']):,} timeline, "
          f"eps_head {EPS_HEAD:.2f}\n")
    assert (len(_S["gold"]), int(y.sum()), int((y == 0).sum())) == (3274, 1502, 1772)

    res = {}
    for fit in FITS:
        for ev in EVALS:
            for m in MODELS:
                res.setdefault(f"{fit}->{ev}", {})[m] = cell(m, fit, ev, EPS_HEAD, full=True)

    # ---- sanity gate against the pinned cross.json --------------------------
    pin = json.loads(PINNED.read_text())["cells"]
    bad = []
    for key in res:
        for m in MODELS[:3]:
            a, b = res[key][m]["auc"], pin[key][m]["auc"]
            d = abs(a - b)
            print(f"gate {key:12s} {m:8s} {a:.6f} vs {b:.6f}  |d|={d:.2e}")
            if d >= 1e-3:
                bad.append((key, m, a, b, d))
    if bad:
        print("SANITY CHECK FAILED", bad)
        sys.exit(2)
    print("sanity check passed (3-voice / 7-flag / 28-cell reproduce cross.json within 1e-3)\n")

    print("AUC, rows = fit corpus, columns = eval corpus")
    print(f"{'':16s}" + "".join(f"{'eval ' + e:>44}" for e in EVALS))
    print(f"{'':16s}" + "".join(f"{m:>11}" for e in EVALS for m in MODELS))
    for fit in FITS:
        line = f"fit {fit:12s}"
        for ev in EVALS:
            for m in MODELS:
                c = res[f"{fit}->{ev}"][m]
                line += f"{c['auc']:>11.4f}" if c else f"{'infeasible':>11}"
        print(line)

    # ---- bootstrap ----------------------------------------------------------
    print(f"\nbootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}) ...", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    ncn, ntl = len(_S["cn"]), len(_S["tl"])
    draws = []
    for _ in range(BOOT_REPS):     # same draw ORDER as cross_ladder.main
        gi = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ci = rng.integers(0, ncn, ncn)
        ti = rng.integers(0, ntl, ntl)
        draws.append((gi, ci, ti))

    reps = {f"{f}->{e}": {m: [] for m in MODELS} for f in FITS for e in EVALS}
    t0 = time.time()
    with Pool(int(os.environ.get("NPROC", 8)), initializer=_init) as pool:
        for i, out in enumerate(pool.imap(_rep, draws, chunksize=8)):
            for fi, fit in enumerate(FITS):
                for ei, ev in enumerate(EVALS):
                    for mi, m in enumerate(MODELS):
                        c = out[fi][ei][mi]
                        reps[f"{fit}->{ev}"][m].append(c["auc"] if c else np.nan)
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{BOOT_REPS}  {el:.0f}s  "
                      f"ETA {el/(i+1)*(BOOT_REPS-i-1):.0f}s", flush=True)

    for key in reps:
        for m in MODELS:
            if res[key][m] is None:
                continue
            lo, hi = np.nanpercentile(reps[key][m], [2.5, 97.5])
            res[key][m]["auc_ci"] = [float(lo), float(hi)]
            res[key][m]["boot_infeasible"] = int(np.isnan(reps[key][m]).sum())

    # CI drift vs the pinned file, reported but not gated (informational).
    print("\nCI check vs cross.json (max |d| over the 3 pinned models)")
    dmax = 0.0
    for key in res:
        for m in MODELS[:3]:
            if "auc_ci" in pin[key][m]:
                dmax = max(dmax, max(abs(res[key][m]["auc_ci"][i] - pin[key][m]["auc_ci"][i])
                                     for i in (0, 1)))
    print(f"  max CI-endpoint drift {dmax:.2e}")

    print("\nAUC with 95% intervals")
    for key in [f"{f}->{e}" for f in FITS for e in EVALS]:
        print(f"  {key}")
        for m in MODELS:
            c = res[key][m]
            if not c:
                print(f"    {m:9s} de-mix infeasible")
                continue
            print(f"    {m:9s} {c['auc']:.4f} [{c['auc_ci'][0]:.4f}, {c['auc_ci'][1]:.4f}]"
                  f"   recall@2% {c['recall_2pct']*100:5.1f}% (FPR {c['fpr_2pct']*100:.2f}%)"
                  + (f"   observed {c['auc_observed']:.4f}" if "auc_observed" in c else ""))

    payload = {
        "source": "scratchpad reproduction of cross_ladder.py + the 12-cell model",
        "eps_head": EPS_HEAD, "folds": fit_urn.K_FOLDS,
        "boot": {"reps": BOOT_REPS, "seed": BOOT_SEED},
        "gold_exclusions": os.environ.get("GOLD_EXCLUSIONS"),
        "models": list(MODELS),
        "n_weights": {m: len(ML.channels(m)) for m in MODELS},
        "channels": {m: ML.channels(m) for m in MODELS},
        "sizes": {"gold": len(_S["gold"]), "gold_true": int(y.sum()),
                  "gold_false": int((y == 0).sum()),
                  "cn": len(_S["cn"]), "timeline": len(_S["tl"])},
        "sanity": {key: {m: {"ours": res[key][m]["auc"], "pinned": pin[key][m]["auc"]}
                         for m in MODELS[:3]} for key in res},
        "cells": res}
    (OUT / "issue24_cross_12cell.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT / 'issue24_cross_12cell.json'}")


if __name__ == "__main__":
    main()
