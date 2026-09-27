"""Issue #24: 2x2 of class count (3-voice vs 7-flag) x source nature (off / 4 tiers).

$0, local refit of the saved E1 reads. Same population, folds, seed, bootstrap and
Laplace log-LR weights as model_ladder.py; unshrunk at every level.
Cell (b) = 3-voice x 4 tiers has never been fitted before.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from eval.scripts.build_eval import fit_urn, model_ladder as ML
from eval.scripts.build_eval.quality_urn import CELLS, RATED, VOICE

OUT = Path("eval/data/urn_runs/e1_ctx/model_ladder/issue24")
VN = ML.VOICE_NAME
rated = lambda tr: "rated" if tr in RATED else "unrated"

KEYS = {
    "a": lambda fl, tr: VN[VOICE[fl]],
    "b": lambda fl, tr: f"{VN[VOICE[fl]]} | {tr}",
    "c": lambda fl, tr: fl,
    "d": lambda fl, tr: f"{fl} | {tr}",
    "b2": lambda fl, tr: f"{VN[VOICE[fl]]} | {rated(tr)}",
    "d2": lambda fl, tr: f"{fl} | {rated(tr)}",
}
MODELS = ["a", "b", "c", "d", "b2", "d2"]
LABEL = {"a": "3 classes", "b": "3 classes x source", "c": "7 classes",
         "d": "7 classes x source", "b2": "3 classes x rated/unrated",
         "d2": "7 classes x rated/unrated"}


def channels(m):
    seen = []
    for fl, tr in CELLS:
        ch = KEYS[m](fl, tr)
        if ch not in seen:
            seen.append(ch)
    return seen


def count_matrix(rows, m):
    chans = channels(m)
    idx = {c: i for i, c in enumerate(chans)}
    C = np.zeros((len(chans), len(rows)))
    for j, r in enumerate(rows):
        for fl, tr in r["docs"]:
            C[idx[KEYS[m](fl, tr)], j] += 1
    return C


def main():
    os.chdir(SRC)
    os.environ.setdefault("GOLD_EXCLUSIONS", "none")   # pinned pre-purge population (n=3,274)
    rows = ML.load_docs()
    kept = [r for r in rows if r["subtype"] != ML.SET_ASIDE]
    y = np.array([r["y"] for r in kept])
    mid = np.array([r["mid"] for r in kept])
    fold = np.array([r["fold"] for r in kept])
    n_t, n_f = int(y.sum()), int((y == 0).sum())
    print(f"population {len(kept)} claims (T {n_t} / F {n_f}), headline {len(rows)}")
    assert (len(kept), n_t, n_f) == (3274, 1502, 1772), (len(kept), n_t, n_f)

    C = {m: count_matrix(kept, m) for m in MODELS}
    docs = int(C["a"].sum())
    print(f"documents {docs:,}")

    # tier share of documents, for the thin-cell sentence
    tier_docs = {}
    for fl, tr in CELLS:
        pass
    for r in kept:
        for fl, tr in r["docs"]:
            tier_docs[tr] = tier_docs.get(tr, 0) + 1
    print("tier doc share", {k: round(100 * v / docs, 2) for k, v in tier_docs.items()})

    res = {}
    S = {}
    for m in MODELS:
        s = ML.oof_scores(C[m], y, mid, fold)
        S[m] = s
        a = ML.auc_np(s[y == 1], s[y == 0])
        r2, f2, t2 = ML.recall_at_fpr(s, y, 0.02)
        fx, tp = ML.roc(s, y)
        res[m] = {"label": LABEL[m], "n_weights": len(channels(m)),
                  "channels": channels(m), "auc_oof": a,
                  "recall_2pct": r2, "fpr_2pct": f2, "threshold_2pct": t2,
                  "roc": [fx.tolist(), tp.tolist()]}
        print(f"{m:3s} {LABEL[m]:28s} {len(channels(m)):2d}w  AUC {a:.6f}  "
              f"recall@2%FPR {r2:.4f} (realised FPR {f2:.4f})")

    # ---- SANITY GATE: a, c, d must match the pinned ladder.json to +-0.001 AUC
    pin = json.loads(Path("eval/data/urn_runs/e1_ctx/model_ladder/ladder.json").read_text())
    gate = {"a": "3-voice", "c": "7-flag", "d": "28-cell"}
    bad = []
    for m, pm in gate.items():
        ref = pin["models"][pm]["auc_oof"]
        d = abs(res[m]["auc_oof"] - ref)
        print(f"gate {m} vs {pm}: {res[m]['auc_oof']:.6f} vs {ref:.6f}  |d|={d:.2e}")
        if d >= 1e-3:
            bad.append((m, pm, res[m]["auc_oof"], ref, d))
    if bad:
        print("SANITY CHECK FAILED", bad)
        sys.exit(2)
    print("sanity check passed (a/c/d reproduce ladder.json within 1e-3)")

    # ---- paired bootstrap ---------------------------------------------------
    reps, seed = ML.BOOT_REPS, ML.BOOT_SEED
    print(f"bootstrap ({reps} reps, seed {seed}) ...", flush=True)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    a_reps = {m: [] for m in MODELS}
    t0 = time.time()
    for i in range(reps):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy, mm, ff = y[idx], mid[idx], fold[idx]
        for m in MODELS:
            Cm = C[m][:, idx]
            s = np.empty(len(idx))
            for k in range(fit_urn.K_FOLDS):
                tr, te = np.where(ff != k)[0], np.where(ff == k)[0]
                s[te] = ML.fit_w(Cm, yy, mm, tr) @ Cm[:, te]
            a_reps[m].append(ML.auc_np(s[yy == 1], s[yy == 0]))
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{reps}  {el:.0f}s  ETA {el/(i+1)*(reps-i-1):.0f}s", flush=True)

    for m in MODELS:
        res[m]["auc_ci"] = [float(x) for x in np.percentile(a_reps[m], [2.5, 97.5])]
        d = np.array(a_reps[m]) - np.array(a_reps["a"])
        res[m]["dauc_vs_a"] = res[m]["auc_oof"] - res["a"]["auc_oof"]
        res[m]["dauc_ci"] = [float(x) for x in np.percentile(d, [2.5, 97.5])]
        print(f"{m:3s} AUC {res[m]['auc_oof']:.4f} [{res[m]['auc_ci'][0]:.4f}, "
              f"{res[m]['auc_ci'][1]:.4f}]  dAUC {res[m]['dauc_vs_a']:+.4f} "
              f"[{res[m]['dauc_ci'][0]:+.4f}, {res[m]['dauc_ci'][1]:+.4f}]")

    payload = {"population": {"n": len(kept), "n_true": n_t, "n_false": n_f,
                              "headline": len(rows), "set_aside_subtype": ML.SET_ASIDE,
                              "set_aside_n": len(rows) - len(kept),
                              "n_unprovable_eval_only": int(mid.sum()),
                              "docs": docs, "tier_docs": tier_docs,
                              "tier_doc_pct": {k: 100 * v / docs for k, v in tier_docs.items()}},
               "boot": {"reps": reps, "seed": seed}, "folds": fit_urn.K_FOLDS,
               "tiers": ["PRIMARY", "RELIABLE", "UNRELIABLE", "UNRATED"],
               "sanity": {m: {"ours": res[m]["auc_oof"], "pinned": pin["models"][p]["auc_oof"]}
                          for m, p in gate.items()},
               "cells": res}
    (OUT / "issue24_2x2.json").write_text(json.dumps(payload, indent=1))

    with (OUT / "issue24_oof_scores.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["claim_id", "label", "score_a", "score_b", "score_c", "score_d"])
        for j, r in enumerate(kept):
            w.writerow([r["review_url"], int(y[j])] +
                       [f"{S[m][j]:.6f}" for m in ("a", "b", "c", "d")])
    print("wrote", OUT / "issue24_2x2.json", OUT / "issue24_oof_scores.csv")


if __name__ == "__main__":
    main()
