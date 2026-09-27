"""Deliverable 3e: are there finer RAW axes in the saved reads worth bucketing?

Two axes are already on every saved document and were never tested: the
retrieval `rank` (1-10) and `provenance` (snippet vs scraped full read).
NewsGuard numeric bins are NOT retested -- quality_urn.py already recorded the
verdict (ng covers ~37% of documents, UNRELIABLE holds ~249, so bins would be
mostly imputation).

Same population, folds, seed, bootstrap and estimator as bucketopt_run.py.
Each structure is a partition of an expanded base cell set; the data-driven one
re-selects its merge depth by inner CV inside every training fold.
"""
from __future__ import annotations

import json
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from bucketopt_core import (bootstrap, OUT, K_FOLDS, BOOT_REPS, BOOT_SEED, load_gold,
                            loglr, auc_np, recall_at_fpr, merge_path,
                            path_cellweights, cellw_from_labels)

NWORK = min(8, os.cpu_count() or 4)
RANKBIN = lambda r: "r1-3" if r <= 3 else ("r4-6" if r <= 6 else "r7+")
_G = {}


def build(rows, keyfn):
    keys = sorted({keyfn(d) for r in rows for d in r["docs"]})
    idx = {k: i for i, k in enumerate(keys)}
    C = np.zeros((len(keys), len(rows)))
    for j, r in enumerate(rows):
        for d in r["docs"]:
            C[idx[keyfn(d)], j] += 1
    return C, keys


def labels_from(keys, mapfn):
    _, lab = np.unique([repr(mapfn(k)) for k in keys], return_inverse=True)
    return lab.astype(int)


def cnt(C, y, mid, idx):
    keep = idx[~mid[idx]]
    p, n = keep[y[keep] == 1], keep[y[keep] == 0]
    return C[:, p].sum(1), C[:, n].sum(1)


def fit_fixed(lab):
    return lambda C, y, mid, fold, tr: cellw_from_labels(*cnt(C, y, mid, tr), lab)[0]


def fit_merge_cv(parent):
    def f(C, y, mid, fold, tr):
        nb = C.shape[0]
        ff = fold[tr]
        s, ys = [], []
        for k in range(K_FOLDS):
            itr, ite = tr[ff != k], tr[ff == k]
            cT, cF = cnt(C, y, mid, itr)
            labs, _ = merge_path(cT, cF, parent, False)
            W = path_cellweights(labs, cT, cF)
            V = np.full((nb, nb), np.nan)
            for lab, row in zip(labs, W):
                V[nb - (lab.max() + 1)] = row
            for i in range(nb):
                if np.isnan(V[i, 0]):
                    V[i] = W[-1]
            s.append(V @ C[:, ite]); ys.append(y[ite])
        S = np.concatenate(s, axis=1); yy = np.concatenate(ys)
        kbest = int(np.argmax([auc_np(row, yy) for row in S]))
        cT, cF = cnt(C, y, mid, tr)
        labs, _ = merge_path(cT, cF, parent, False)
        return cellw_from_labels(cT, cF, labs[min(kbest, len(labs) - 1)])[0]
    return f


def oof(fit, C, y, mid, fold):
    s = np.empty(C.shape[1]); a = np.arange(C.shape[1])
    for k in range(K_FOLDS):
        tr, te = a[fold != k], a[fold == k]
        s[te] = fit(C, y, mid, fold, tr) @ C[:, te]
    return s


def _chunk(args):
    lo, hi = args
    y, mid, fold, S = _G["y"], _G["mid"], _G["fold"], _G["S"]
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(lo):
        rng.choice(pos, len(pos)); rng.choice(neg, len(neg))
    A = {n: [] for n in S}
    for _ in range(hi - lo):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy, mm, ff = y[idx], mid[idx], fold[idx]
        for n, (C, fit) in S.items():
            A[n].append(auc_np(oof(fit, C[:, idx], yy, mm, ff), yy))
    return A


def main():
    bootstrap()
    t0 = time.time()
    rows = load_gold(extra=True)
    y = np.array([r["y"] for r in rows]); mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows])

    # base cell sets: (flag, tier, prov) and (flag, tier, rankbin)
    Cp, kp = build(rows, lambda d: (d[0], d[1], d[3]))
    Cr, kr = build(rows, lambda d: (d[0], d[1], RANKBIN(d[2])))
    print(f"flag x tier x provenance base {Cp.shape[0]} cells; "
          f"flag x tier x rankbin base {Cr.shape[0]} cells; docs {int(Cp.sum()):,}")

    S = {}
    S["7-flag (reference)"] = (Cp, fit_fixed(labels_from(kp, lambda k: k[0])))
    S["28-cell (reference)"] = (Cp, fit_fixed(labels_from(kp, lambda k: (k[0], k[1]))))
    S["7-flag x provenance (14)"] = (Cp, fit_fixed(labels_from(kp, lambda k: (k[0], k[2]))))
    S["7-flag x tier x prov (56)"] = (Cp, fit_fixed(labels_from(kp, lambda k: k)))
    S["7-flag x rankbin (21)"] = (Cr, fit_fixed(labels_from(kr, lambda k: (k[0], k[2]))))
    S["merge-any inner-CV over 56"] = (
        Cp, fit_merge_cv(labels_from(kp, lambda k: k[0])))
    S["merge-any inner-CV over 84"] = (
        Cr, fit_merge_cv(labels_from(kr, lambda k: k[0])))

    res = {}
    for n, (C, fit) in S.items():
        s = oof(fit, C, y, mid, fold)
        res[n] = {"auc_oof": auc_np(s, y), "recall_2pct": recall_at_fpr(s, y, 0.02)[0],
                  "n_base_cells": C.shape[0]}
        print(f"{n:32s} AUC {res[n]['auc_oof']:.4f}  rec@2% {res[n]['recall_2pct']*100:5.1f}%")

    _G.update(y=y, mid=mid, fold=fold, S=S)
    edges = np.linspace(0, BOOT_REPS, NWORK + 1).astype(int)
    print(f"bootstrap {BOOT_REPS} reps, {NWORK} workers ...", flush=True)
    A = {n: [] for n in S}
    with ProcessPoolExecutor(NWORK, mp_context=mp.get_context("fork")) as ex:
        for a in ex.map(_chunk, [(int(x), int(z)) for x, z in zip(edges[:-1], edges[1:])]):
            for n in S:
                A[n] += a[n]
    ref = np.array(A["7-flag (reference)"])
    for n in S:
        arr = np.array(A[n])
        res[n]["auc_ci"] = [float(v) for v in np.percentile(arr, [2.5, 97.5])]
        res[n]["dauc_vs_7flag"] = res[n]["auc_oof"] - res["7-flag (reference)"]["auc_oof"]
        res[n]["dauc_ci"] = [float(v) for v in np.percentile(arr - ref, [2.5, 97.5])]
        r = res[n]
        print(f"{n:32s} AUC {r['auc_oof']:.4f} [{r['auc_ci'][0]:.4f},{r['auc_ci'][1]:.4f}] "
              f"dAUC {r['dauc_vs_7flag']:+.4f} [{r['dauc_ci'][0]:+.4f},{r['dauc_ci'][1]:+.4f}]")
    (OUT / "bucket_opt_extra_axes.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT/'bucket_opt_extra_axes.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
