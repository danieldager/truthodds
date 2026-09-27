"""Monte Carlo reader-error correction counterfactual on the 7-flag graded urn.
Reproduces the two pinned oof AUCs as trust gates, then corrects refuting-on-true
and supporting-on-false reader errors by resampling deserved flags.

  uv run python -m eval.scripts.build_eval.reader_error_counterfactual

$0 -- rescores cached reads, no API calls. The deserved-flag distributions are the
37 reader_error records from the 2026-09-17 hand audit (clog/170926.md 14:30)."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.scripts.build_eval import fit_urn, graded_urn
from eval.scripts.build_eval import reader_error_lib as L

FLAGS7 = graded_urn.FLAGS7                 # ("5","4","3","X","I","2","1")
FIDX = {f: i for i, f in enumerate(FLAGS7)}
TRAIN = "eval/data/urn_runs/e1_ctx/results-00.jsonl"
PROD = "eval/data/urn_runs/e1_prodregime/cond_S_v5_full3000_v5.jsonl"
POPULATION = "eval/data/populations/fc_gold_bal3000.parquet"
OUT = "eval/data/urn_runs/e1_ctx/reader_error_counterfactual.json"
NDRAW, SEED = 200, 917
BOOT_REPS, BOOT_SEED = fit_urn.BOOT_REPS, fit_urn.BOOT_SEED

# ---- deserved-flag distributions from the 37 reader_error records -----------
# Each list = the flag(s) the judge said the doc deserved; a two-flag entry is a
# compound verdict, split 0.5/0.5 by drawing a record then a flag within it.
DESERVED_RT = [["5"], ["4"], ["4"], ["5"], ["4"], ["5"], ["4"], ["5", "4"], ["5"], ["I"],
               ["I"], ["5"], ["5"], ["4", "3"], ["5"], ["4"], ["5"], ["I"], ["I"], ["5"],
               ["5"], ["4"], ["5"]]                        # 23 refuting-on-true errors
DESERVED_SF = [["2"], ["1", "2"], ["2"], ["1", "2"], ["2"], ["X"], ["1", "2"], ["3", "I"],
               ["2"], ["I"], ["X"], ["1", "2"], ["2"], ["X"]]  # 14 supporting-on-false errors
ERR_RT, ERR_SF = 23 / 30, 14 / 30
WILSON_RT, WILSON_SF = 0.59, 0.30


def deserved_mass(pool):
    m = collections.Counter()
    for rec in pool:
        for f in rec:
            m[f] += 1.0 / len(rec)
    tot = sum(m.values())
    return {f: m[f] / tot for f in FLAGS7 if m[f] > 0}


def prep(doc_rows):
    """Precompute per-claim base counts (non-correctable flags + pad) and the
    original flags of the correctable docs (R = refuting-on-true, S = support-on-false)."""
    n = len(doc_rows)
    base = np.zeros((n, 7))
    corr_R, corr_S = [], []          # lists of original flag strings per claim
    y = np.zeros(n, int); mid = np.zeros(n, bool); fold = np.zeros(n, int)
    clusters = []
    for i, d in enumerate(doc_rows):
        y[i] = d["y"]; mid[i] = d["mid"]; fold[i] = d["fold"]; clusters.append(d["cluster"])
        rR, rS = [], []
        cnt = collections.Counter()
        for fl in d["real_flags"]:
            if d["y"] == 1 and fl in ("1", "2"):
                rR.append(fl)
            elif d["y"] == 0 and fl in ("5", "4"):
                rS.append(fl)
            else:
                cnt[fl] += 1
        n_pad = max(0, fit_urn.PAD_TO - len(d["real_flags"]))
        cnt["I"] += n_pad
        for f, c in cnt.items():
            base[i, FIDX[f]] += c
        corr_R.append(rR); corr_S.append(rS)
    return base, corr_R, corr_S, y, mid, fold, clusters


def draw_counts(base, corr_R, corr_S, pR, pS, doR, doS, rng):
    C = base.copy()
    for i in range(len(base)):
        for fl in corr_R[i]:
            if doR and rng.random() < pR:
                rec = DESERVED_RT[rng.integers(len(DESERVED_RT))]
                nf = rec[rng.integers(len(rec))]
            else:
                nf = fl
            C[i, FIDX[nf]] += 1
        for fl in corr_S[i]:
            if doS and rng.random() < pS:
                rec = DESERVED_SF[rng.integers(len(DESERVED_SF))]
                nf = rec[rng.integers(len(rec))]
            else:
                nf = fl
            C[i, FIDX[nf]] += 1
    return C.T          # 7 x n


def metrics(C, y, mid, fold):
    s = fit_urn.oof_np(C, y, mid, fold)
    auc = graded_urn._auc_np(s[y == 1], s[y == 0])
    nest = fit_urn.nested_threshold(C, y, mid, fold, 0.02)
    thr = float(np.mean(nest["thresholds_by_fold"]))
    idx = np.arange(C.shape[1]); flagged = np.zeros(len(idx), bool)
    for k in range(fit_urn.K_FOLDS):
        te = fold == k
        if not te.any() or te.all():
            continue
        tr = idx[~te]
        s_in = fit_urn.oof_np(C, y, mid, fold, tr)
        _, _, t = fit_urn.recall_at_fpr(list(zip(s_in.tolist(), y[tr].tolist())), 0.02)
        flagged[te] = (fit_urn.fit_np(C, y, mid, tr) @ C[:, idx[te]]) <= t
    return {"auc": float(auc), "recall": float(nest["recall"]), "fpr": float(nest["fpr"]),
            "threshold": thr, "false_flagged": int(flagged[y == 0].sum()),
            "true_flagged": int(flagged[y == 1].sum())}


def run_variant(base, corr_R, corr_S, y, mid, fold, pR, pS, doR, doS, ndraw, seed):
    rng = np.random.default_rng(seed)
    keys = ["auc", "recall", "fpr", "threshold", "false_flagged", "true_flagged"]
    acc = {k: [] for k in keys}
    Cs = []
    for _ in range(ndraw):
        C = draw_counts(base, corr_R, corr_S, pR, pS, doR, doS, rng)
        m = metrics(C, y, mid, fold)
        for k in keys:
            acc[k].append(m[k])
        Cs.append(C)
    out = {}
    for k in keys:
        a = np.array(acc[k], float)
        out[k] = {"mean": float(a.mean()),
                  "p2.5": float(np.percentile(a, 2.5)),
                  "p97.5": float(np.percentile(a, 97.5))}
    return out, acc, Cs


def cluster_ci_on_draw(C, y, mid, fold, clusters, reps, seed):
    """Cluster bootstrap (resample cluster_id) AUC + nested recall CI on one draw."""
    cl = np.unique([str(c) for c in clusters], return_inverse=True)[1]
    rng = np.random.default_rng(seed)
    aucs, recs = [], []
    for _ in range(reps):
        idx = fit_urn.boot_idx(rng, y, cl, True)
        s = fit_urn.oof_np(C, y, mid, fold, idx); yy = y[idx]
        aucs.append(graded_urn._auc_np(s[yy == 1], s[yy == 0]))
        # nested recall inside the replicate
        fl = np.zeros(len(idx), bool); ff = fold[idx]
        for k in range(fit_urn.K_FOLDS):
            te = ff == k
            if not te.any() or te.all():
                continue
            tr = idx[~te]
            s_in = fit_urn.oof_np(C, y, mid, fold, tr)
            _, _, t = fit_urn.recall_at_fpr(list(zip(s_in.tolist(), y[tr].tolist())), 0.02)
            fl[te] = (fit_urn.fit_np(C, y, mid, tr) @ C[:, idx[te]]) <= t
        recs.append(fl[yy == 0].mean())
    return {"auc_ci95": [float(x) for x in np.percentile(aucs, [2.5, 97.5])],
            "recall_ci95": [float(x) for x in np.percentile(recs, [2.5, 97.5])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=TRAIN)
    ap.add_argument("--prod", type=Path, default=PROD)
    ap.add_argument("--population", type=Path, default=POPULATION)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    bal = set(pd.read_parquet(args.population)["claim_id"].tolist())
    out = {"seed": SEED, "ndraws": NDRAW,
           "error_rates": {"refuting_on_true": ERR_RT, "supporting_on_false": ERR_SF,
                           "wilson_lower": {"refuting_on_true": WILSON_RT,
                                            "supporting_on_false": WILSON_SF}},
           "deserved_flag_dist": {"refuting_on_true": deserved_mass(DESERVED_RT),
                                  "supporting_on_false": deserved_mass(DESERVED_SF)},
           "deserved_records": {"refuting_on_true": DESERVED_RT,
                                "supporting_on_false": DESERVED_SF}}

    # ---------- PRODUCTION ----------
    dr = L.load_docs(args.prod, bal)
    base, cR, cS, y, mid, fold, clusters = prep(dr)
    C0 = base.copy()
    for i in range(len(base)):
        for fl in cR[i]:
            C0[i, FIDX[fl]] += 1
        for fl in cS[i]:
            C0[i, FIDX[fl]] += 1
    base_m = metrics(C0.T, y, mid, fold)
    out["baseline_production"] = base_m
    assert abs(base_m["auc"] - 0.9236577777777778) < 1e-9, base_m["auc"]
    print("TRUST GATE prod AUC", base_m["auc"], "recall", base_m["recall"],
          "flagged F/T", base_m["false_flagged"], base_m["true_flagged"])

    variants = {
        "i_both": dict(pR=ERR_RT, pS=ERR_SF, doR=True, doS=True),
        "ii_refuting_only": dict(pR=ERR_RT, pS=ERR_SF, doR=True, doS=False),
        "iii_supporting_only": dict(pR=ERR_RT, pS=ERR_SF, doR=False, doS=True),
        "iv_wilson_lower": dict(pR=WILSON_RT, pS=WILSON_SF, doR=True, doS=True),
    }
    out["production"] = {}
    for name, cfg in variants.items():
        res, acc, Cs = run_variant(base, cR, cS, y, mid, fold,
                                    cfg["pR"], cfg["pS"], cfg["doR"], cfg["doS"], NDRAW, SEED)
        out["production"][name] = res
        print(f"PROD {name:22s} AUC {res['auc']['mean']:.4f} "
              f"[{res['auc']['p2.5']:.4f},{res['auc']['p97.5']:.4f}]  "
              f"recall {res['recall']['mean']:.4f} "
              f"[{res['recall']['p2.5']:.4f},{res['recall']['p97.5']:.4f}]  "
              f"Fflag {res['false_flagged']['mean']:.0f} Tflag {res['true_flagged']['mean']:.0f} "
              f"thr {res['threshold']['mean']:+.3f}")
        if name == "i_both":
            aucs = np.array(acc["auc"])
            rep = int(np.argmin(np.abs(aucs - aucs.mean())))
            ci = cluster_ci_on_draw(Cs[rep], y, mid, fold, clusters, BOOT_REPS, BOOT_SEED)
            out["production"]["i_both_representative_draw"] = {
                "index": rep, "auc": float(aucs[rep]),
                "recall": acc["recall"][rep], "cluster_bootstrap": ci,
                "boot_reps": BOOT_REPS, "boot_seed": BOOT_SEED}
            print(f"   rep-draw {rep} AUC {aucs[rep]:.4f} cluster-CI {ci['auc_ci95']}  "
                  f"recall {acc['recall'][rep]:.4f} CI {ci['recall_ci95']}")

    # ---------- TRAINING (variant i, error rates assumed transferred) ----------
    drt = L.load_docs(args.train, bal)
    bt, cRt, cSt, yt, midt, foldt, clt = prep(drt)
    C0t = bt.copy()
    for i in range(len(bt)):
        for fl in cRt[i]:
            C0t[i, FIDX[fl]] += 1
        for fl in cSt[i]:
            C0t[i, FIDX[fl]] += 1
    base_mt = metrics(C0t.T, yt, midt, foldt)
    out["baseline_training"] = base_mt
    assert abs(base_mt["auc"] - 0.8569157777777778) < 1e-9, base_mt["auc"]
    print("TRUST GATE train AUC", base_mt["auc"], "recall", base_mt["recall"])
    rest, _, _ = run_variant(bt, cRt, cSt, yt, midt, foldt,
                             ERR_RT, ERR_SF, True, True, NDRAW, SEED)
    out["training_i_both"] = rest
    print(f"TRAIN i_both AUC {rest['auc']['mean']:.4f} "
          f"[{rest['auc']['p2.5']:.4f},{rest['auc']['p97.5']:.4f}]  "
          f"recall {rest['recall']['mean']:.4f} "
          f"[{rest['recall']['p2.5']:.4f},{rest['recall']['p97.5']:.4f}]")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
