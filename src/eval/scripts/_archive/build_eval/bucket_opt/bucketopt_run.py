"""Bucket-partition optimisation: all structures, one population, one protocol.

    cd src && GOLD_EXCLUSIONS=none PYTHONPATH=eval/scripts/build_eval/bucket_opt uv run python eval/scripts/build_eval/bucket_opt/bucketopt_run.py

$0. Pinned fc-gold (n=3,274, T 1,502 / F 1,772, 31,888 docs), 5 folds by blake2b
of review_url, 2,000-rep paired bootstrap resampled within gold class, seed 707.
Every selection (merge depth, ridge C, shrink M) happens inside the training
rows only; the outer test fold never informs it.
"""
from __future__ import annotations

import json
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bucketopt_core import (bootstrap, OUT, LADDER, NCELL, CELL_NAME, FLAG_OF, K_FOLDS,
                            BOOT_REPS, BOOT_SEED, load_gold, cell_matrix, counts,
                            loglr, se_analytic, auc_np, recall_at_fpr, load_urns,
                            cellw_from_labels)
import bucketopt_structs as BS
from bucketopt_structs import Ctx, INFL

NWORK = min(8, os.cpu_count() or 4)
_G = {}


def oof(fit, C, y, mid, fold, frozen=None, want_meta=False):
    s = np.empty(C.shape[1])
    allidx = np.arange(C.shape[1])
    metas = []
    for k in range(K_FOLDS):
        tr, te = allidx[fold != k], allidx[fold == k]
        v, params, keff, note = fit(Ctx(C, y, mid, fold, tr, FLAG_OF, frozen))
        s[te] = v @ C[:, te]
        if want_meta:
            metas.append({"fold": k, "k_eff": keff, "note": note})
    return (s, metas) if want_meta else s


# --------------------------------------------------------------- bootstrap
def _boot_chunk(args):
    lo, hi = args
    C, y, mid, fold = _G["C"], _G["y"], _G["mid"], _G["fold"]
    reg, frozen, plen = _G["reg"], _G["frozen"], _G["plen"]
    lab_full = _G["lab_full"]
    names = list(reg)
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(lo):                       # burn to keep replicates aligned
        rng.choice(pos, len(pos)); rng.choice(neg, len(neg))
    A = {n: [] for n in names}
    P = {n: [] for n in names}
    for _ in range(hi - lo):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        Cb, yy, mm, ff = C[:, idx], y[idx], mid[idx], fold[idx]
        allb = np.arange(len(idx))
        for n in names:
            fit = reg[n]
            fz = frozen.get(n, {})
            s = np.empty(len(idx))
            for k in range(K_FOLDS):
                tr, te = allb[ff != k], allb[ff == k]
                v = fit(Ctx(Cb, yy, mm, ff, tr, FLAG_OF, fz))[0]
                s[te] = v @ Cb[:, te]
            A[n].append(auc_np(s, yy))
            # Weight intervals are for the structure AS CHOSEN on the full data:
            # the partition (or design) is frozen and only the weights are refit,
            # otherwise replicate k's bucket 3 is not replicate k+1's bucket 3.
            if lab_full[n] is not None:
                cT, cF = counts(Cb, yy, mm, allb)
                lab = lab_full[n]
                K = lab.max() + 1
                P[n].append(loglr(np.bincount(lab, cT, minlength=K),
                                  np.bincount(lab, cF, minlength=K)))
            else:
                p = np.atleast_1d(np.asarray(fit(Ctx(Cb, yy, mm, ff, allb, FLAG_OF, fz))[1]))
                P[n].append(p if len(p) == plen[n] else None)
    return A, P


# ---------------------------------------------------------------- transfer
def demix_partition(lab, cF_raw, cM_raw, eps):
    """transfer_ladder's estimator, applied to an arbitrary partition of the
    base cells: aggregate raw urn doc counts to buckets, Laplace with K =
    n_buckets on each urn, de-mix the TRUE side, return per-cell weights."""
    K = lab.max() + 1
    aF = np.bincount(lab, cF_raw, minlength=K)
    aM = np.bincount(lab, cM_raw, minlength=K)
    pF = (aF + 1) / (aF.sum() + K)
    pM = (aM + 1) / (aM.sum() + K)
    pT = (pM - eps * pF) / (1 - eps)
    if (pT <= 0).any():
        return None
    return np.log(pT / pF)[lab]


def demix_cells(cF_raw, cM_raw, eps, K):
    pF = (cF_raw + 1) / (cF_raw.sum() + K)
    pM = (cM_raw + 1) / (cM_raw.sum() + K)
    pT = (pM - eps * pF) / (1 - eps)
    return (pF, pT) if (pT > 0).all() else (pF, None)


def main():
    bootstrap()
    t0 = time.time()
    rows = load_gold()
    y = np.array([r["y"] for r in rows]); mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    C = cell_matrix(rows)
    assert (len(rows), int(y.sum()), int((y == 0).sum())) == (3274, 1502, 1772)
    print(f"population {len(rows)} claims (T 1502 / F 1772), docs {int(C.sum()):,}")

    reg = BS.build_registry()
    names = list(reg)

    # ---- full-data fits: freeze hyper-parameters + record the chosen partition
    frozen, full = {}, {}
    allidx = np.arange(len(rows))
    for n in names:
        v, params, keff, note = reg[n](Ctx(C, y, mid, fold, allidx, FLAG_OF, {}))
        fz = {}
        for k, tag in (("C=", "C:"), ("M=", "M")):
            if note.startswith(k):
                val = float(note.split("=")[1])
                fz = {"C:flag": val, "C:add": val, "C:cell": val} if k == "C=" else {"M": val}
        frozen[n] = fz
        full[n] = {"v": v, "params": np.atleast_1d(np.asarray(params)),
                   "k_eff": keff, "note": note}
    plen = {n: len(full[n]["params"]) for n in names}
    lab_full = {n: _partition_labels(full[n]["v"], reg, n, C, y, mid, fold) for n in names}
    for n in names:                       # params reported = the frozen structure's
        if lab_full[n] is not None:
            lab = lab_full[n]; K = lab.max() + 1
            cT, cF = counts(C, y, mid, allidx)
            full[n]["params"] = loglr(np.bincount(lab, cT, minlength=K),
                                      np.bincount(lab, cF, minlength=K))
            plen[n] = K

    # ---- point estimates, nested selection --------------------------------
    res, S = {}, {}
    for n in names:
        s, metas = oof(reg[n], C, y, mid, fold, frozen={}, want_meta=True)
        S[n] = s
        a = auc_np(s, y)
        r2, f2, t2 = recall_at_fpr(s, y, 0.02)
        res[n] = {"auc_oof": a, "recall_2pct": r2, "fpr_2pct": f2,
                  "k_eff_full": full[n]["k_eff"], "note_full": full[n]["note"],
                  "fold_k_eff": [m["k_eff"] for m in metas],
                  "fold_note": [m["note"] for m in metas]}
        print(f"{n:34s} k={str(full[n]['k_eff'])[:5]:>5s}  AUC {a:.4f}  rec@2% {r2*100:5.1f}%")

    # ---- SANITY GATE -------------------------------------------------------
    pin = json.loads(LADDER.read_text())["models"]
    gate = {"3-voice": "3-voice", "7-flag": "7-flag", "28-cell (flag x tier)": "28-cell"}
    for n, p in gate.items():
        d = abs(res[n]["auc_oof"] - pin[p]["auc_oof"])
        print(f"gate {n} vs ladder {p}: {res[n]['auc_oof']:.6f} vs "
              f"{pin[p]['auc_oof']:.6f}  |d|={d:.2e}")
        assert d < 1e-3, (n, d)
    d12 = abs(res["12-cell (voice x tier)"]["auc_oof"] - 0.8619)
    print(f"gate 12-cell vs issue24 (b) 0.8619: |d|={d12:.2e}")
    assert d12 < 1e-3
    print("SANITY GATE PASSED\n")

    # ---- bootstrap ---------------------------------------------------------
    _G.update(C=C, y=y, mid=mid, fold=fold, reg=reg, frozen=frozen, plen=plen,
              lab_full=lab_full)
    edges = np.linspace(0, BOOT_REPS, NWORK + 1).astype(int)
    chunks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
    print(f"bootstrap {BOOT_REPS} reps, seed {BOOT_SEED}, {NWORK} workers ...", flush=True)
    A = {n: [] for n in names}; P = {n: [] for n in names}
    with ProcessPoolExecutor(NWORK, mp_context=mp.get_context("fork")) as ex:
        for i, (a, p) in enumerate(ex.map(_boot_chunk, chunks)):
            for n in names:
                A[n] += a[n]; P[n] += [x for x in p[n] if x is not None]
            print(f"  chunk {i+1}/{len(chunks)} done  {time.time()-t0:.0f}s", flush=True)

    ref = np.array(A["7-flag"])
    for n in names:
        arr = np.array(A[n])
        res[n]["auc_ci"] = [float(x) for x in np.percentile(arr, [2.5, 97.5])]
        d = arr - ref
        res[n]["dauc_vs_7flag"] = res[n]["auc_oof"] - res["7-flag"]["auc_oof"]
        res[n]["dauc_ci"] = [float(x) for x in np.percentile(d, [2.5, 97.5])]
        if P[n]:
            W = np.array(P[n])
            lo, hi = np.percentile(W, [2.5, 97.5], axis=0)
            hw = (hi - lo) / 2
            res[n]["n_param_reps"] = len(P[n])
            res[n]["hw_min"] = float(hw.min()); res[n]["hw_med"] = float(np.median(hw))
            res[n]["hw_max"] = float(hw.max())
            res[n]["params_full"] = [float(x) for x in full[n]["params"]]
            res[n]["params_ci"] = [[float(a), float(b)] for a, b in zip(lo, hi)]

    # ---- partition stability across outer folds ---------------------------
    for n in names:
        sigs = res[n]["fold_note"]
        if sigs and sigs[0] and ";" in sigs[0]:
            res[n]["partition_stable_across_folds"] = len(set(sigs)) == 1
            res[n]["fold_partitions"] = sigs
            res[n]["partition_full"] = full[n]["note"]
            res[n].pop("fold_note")

    # ---- transfer ----------------------------------------------------------
    print("\ntransfer: fit on the two urns, score the pinned gold, no refit "
          "(eps = 0.10, transfer_ladder's headline)")
    fr, tr_rows = load_urns()
    CF = cell_matrix(fr); CM = cell_matrix(tr_rows)
    cF_raw, cM_raw = CF.sum(1), CM.sum(1)
    print(f"  FALSE urn {len(fr):,} claims / {int(CF.sum()):,} docs;  "
          f"TIMELINE urn {len(tr_rows):,} / {int(CM.sum()):,} docs")
    tref = json.loads((LADDER.parent / "transfer.json").read_text())
    for eps in (0.0, 0.10):
        for n in names:
            lab = _partition_labels(full[n]["v"], reg, n, C, y, mid, fold)
            if lab is None:
                continue
            w = demix_partition(lab, cF_raw, cM_raw, eps)
            key = "transfer_auc" if eps == 0.10 else "transfer_auc_eps0"
            if w is None:
                res[n][key] = None
                continue
            s = w @ C
            res[n][key] = auc_np(s, y)
            if eps == 0.10:
                res[n]["transfer_recall_2pct"] = recall_at_fpr(s, y, 0.02)[0]
    for n, p in gate.items():
        got, want = res[n].get("transfer_auc"), tref["models"][p]["auc"]
        print(f"  gate {n:26s} transfer {got:.4f} vs transfer.json {want:.4f} "
              f"|d|={abs(got-want):.2e}")

    # shrinkage transfers: same de-mix, then the same shrink rule on urn support
    for n in ("shrink-EB toward flag", "shrink-soft M by inner CV",
              "shrink-hard M=200 (quality_urn)"):
        res[n]["transfer_auc"] = _shrink_transfer(n, cF_raw, cM_raw, 0.10, C, y, full)
        res[n]["transfer_auc_eps0"] = _shrink_transfer(n, cF_raw, cM_raw, 0.0, C, y, full)
    for eps, key in ((0.10, "transfer_auc"), (0.0, "transfer_auc_eps0")):
        res["additive log-LR (flag+tier)"][key] = _additive_transfer(
            cF_raw, cM_raw, eps, C, y)
    for n in names:
        res[n].setdefault("transfer_auc", None)
        if n.startswith("logit"):
            res[n]["transfer_note"] = ("not transferred: a discriminative fit needs "
                                       "claim-level labels, and the urns supply corpus "
                                       "identity rather than gold veracity")

    # ---- what the support constraint costs, swept ---------------------------
    print("\nsupport-constraint sweep (merge until every bucket's 95% half-width <= h)")
    sweep = {}
    for h in (0.20, 0.25, 0.35, 0.50, 0.75, 1.00):
        for within in (True, False):
            f = BS.make_merge_support(within, hw_max=h)
            s_ = oof(f, C, y, mid, fold)
            v, _, k, sig = f(Ctx(C, y, mid, fold, allidx, FLAG_OF, {}))
            lab = _partition_labels(v, reg, "merge", C, y, mid, fold)
            cT, cF = counts(C, y, mid, allidx)
            K = lab.max() + 1
            hw = 1.96 * INFL * se_analytic(np.bincount(lab, cT, minlength=K),
                                           np.bincount(lab, cF, minlength=K))
            key = f"h={h:.2f} {'within-flag' if within else 'any'}"
            sweep[key] = {"auc_oof": auc_np(s_, y),
                          "recall_2pct": recall_at_fpr(s_, y, 0.02)[0],
                          "k_full": int(k), "partition": sig,
                          "max_hw_achieved": float(hw.max())}
            r = sweep[key]
            print(f"  h<={h:.2f} {'within-flag' if within else 'any       '}  "
                  f"k={k:2d}  AUC {r['auc_oof']:.4f}  rec@2% {r['recall_2pct']*100:5.1f}%  "
                  f"max hw {r['max_hw_achieved']:.3f}")

    for n in names:
        r = res[n]
        print(f"{n:34s} AUC {r['auc_oof']:.4f} [{r['auc_ci'][0]:.4f},{r['auc_ci'][1]:.4f}] "
              f"dAUC {r['dauc_vs_7flag']:+.4f} [{r['dauc_ci'][0]:+.4f},{r['dauc_ci'][1]:+.4f}] "
              f"rec {r['recall_2pct']*100:4.1f}%  hw {r.get('hw_med', float('nan')):.3f}/"
              f"{r.get('hw_max', float('nan')):.3f}  transfer "
              f"{('%.4f' % r['transfer_auc']) if r.get('transfer_auc') else '  n/a '}")

    payload = {"population": {"n": len(rows), "n_true": 1502, "n_false": 1772,
                              "docs": int(C.sum())},
               "protocol": {"folds": K_FOLDS, "boot_reps": BOOT_REPS,
                            "boot_seed": BOOT_SEED, "se_inflation": INFL,
                            "hw_max": BS.HW_MAX, "g_crit": BS.G_CRIT,
                            "note": "ridge C and shrink M frozen at their "
                                    "full-data inner-CV value inside the bootstrap; "
                                    "merge depth and partition re-selected in every "
                                    "replicate and every fold"},
               "structures": res, "support_constraint_sweep": sweep}
    (OUT / "bucket_opt_results.json").write_text(json.dumps(payload, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o)))
    np.save(OUT / "bucket_opt_oof.npy", np.array([S[n] for n in names]))
    print(f"\nwrote {OUT/'bucket_opt_results.json'}  ({time.time()-t0:.0f}s)")


def _partition_labels(v, reg, n, C, y, mid, fold):
    """Recover the partition a structure induces on the 28 cells, if it is one.
    Structures whose weights are not a partition of the cells (logistic fits,
    the additive log-linear model, partial pooling) return None -- the urn
    de-mix estimator is defined on bucket rates, so they are not transferred."""
    if n.startswith(("logit", "additive", "shrink")):
        return None
    _, lab = np.unique(np.round(v, 9), return_inverse=True)
    return lab.astype(int)


def _additive_transfer(cF_raw, cM_raw, eps, C, y):
    """IPF the no-three-way-interaction model onto the urns' de-mixed TRUE side
    and raw FALSE side, then score gold with the resulting additive weights."""
    pF, pT = demix_cells(cF_raw, cM_raw, eps, NCELL)
    if pT is None:
        return None
    cT_eff = pT * cM_raw.sum(); cF_eff = pF * cF_raw.sum()
    nf, nt = FLAG_OF.max() + 1, 4
    N = np.zeros((nf, nt, 2))
    for i in range(NCELL):
        N[FLAG_OF[i], BS.TIER_OF[i], 0] = cT_eff[i] + 1.0
        N[FLAG_OF[i], BS.TIER_OF[i], 1] = cF_eff[i] + 1.0
    M = np.ones_like(N) * N.sum() / N.size
    for _ in range(200):
        M *= (N.sum(2) / M.sum(2))[:, :, None]
        M *= (N.sum(1) / M.sum(1))[:, None, :]
        M *= (N.sum(0) / M.sum(0))[None, :, :]
    W = np.log((M[:, :, 0] / M[:, :, 0].sum()) / (M[:, :, 1] / M[:, :, 1].sum()))
    return auc_np(W[FLAG_OF, BS.TIER_OF] @ C, y)


def _shrink_transfer(name, cF_raw, cM_raw, eps, C, y, full):
    """De-mixed 28-cell and 7-flag urn weights, combined by the structure's own
    shrink rule with lambda taken from the urns' own document support."""
    pF28, pT28 = demix_cells(cF_raw, cM_raw, eps, NCELL)
    if pT28 is None:
        return None
    w28 = np.log(pT28 / pF28)
    aF = np.bincount(BS.LAB_FLAG, cF_raw, minlength=7)
    aM = np.bincount(BS.LAB_FLAG, cM_raw, minlength=7)
    pF7, pT7 = demix_cells(aF, aM, eps, 7)
    if pT7 is None:
        return None
    par = np.log(pT7 / pF7)[BS.LAB_FLAG]
    cT_eff = pT28 * cM_raw.sum()                 # de-mixed effective TRUE support
    m = np.minimum(cT_eff, cF_raw)
    if name.startswith("shrink-EB"):
        se2 = (INFL * se_analytic(cT_eff, cF_raw)) ** 2
        lam = np.empty(NCELL)
        for f in range(7):
            k = BS.LAB_FLAG == f
            tau2 = max(((w28[k] - par[k]) ** 2).mean() - se2[k].mean(), 0.0)
            lam[k] = tau2 / (tau2 + se2[k])
    elif "hard" in name:
        lam = (m >= 200).astype(float)
    else:
        M = float(full[name]["note"].split("=")[1])
        lam = m / (m + M)
    w = lam * w28 + (1 - lam) * par
    return auc_np(w @ C, y)


if __name__ == "__main__":
    main()
