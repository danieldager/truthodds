"""Frozen-instrument reader/retrieval ceilings. Applies the FROZEN training
weights (weights_v5_ceiling_bal3000.json) AS-IS (no refit) to training- and
production-mode evidence, with a nested 2%FPR threshold on the scored claims.
Reader errors corrected in ALL FOUR cells using the BLIND-audit rates and
blind deserved-flag distributions. Section C refits oof on the corrected reads
for comparison. 200 draws, seed 917.

  uv run python -m eval.scripts.build_eval.frozen_ceilings

$0 -- rescores cached reads, no API calls. Reuses reader_error_lib (doc loader)
and reader_error_counterfactual (RF.prep / RF.metrics)."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.scripts.build_eval import fit_urn, graded_urn
from eval.scripts.build_eval.e1_figures import _auc_np
from eval.scripts.build_eval import reader_error_lib as L
from eval.scripts.build_eval import reader_error_counterfactual as RF

FLAGS7 = graded_urn.FLAGS7
FIDX = {f: i for i, f in enumerate(FLAGS7)}
K = fit_urn.K_FOLDS
NDRAW, SEED = 200, 917
TRAIN = "eval/data/urn_runs/e1_ctx/results-00.jsonl"
PROD = "eval/data/urn_runs/e1_prodregime/cond_S_v5_full3000_v5.jsonl"
WEIGHTS = "eval/data/urn_runs/e1_ctx/weights_v5_ceiling_bal3000.json"
POPULATION = "eval/data/populations/fc_gold_bal3000.parquet"
BLIND_PATH = "eval/data/urn_runs/e1_ctx/blind120_summary.json"
OUT = "eval/data/urn_runs/e1_ctx/frozen_ceilings.json"

# Populated in main() (no file reads at import time, per the build_eval convention).
WV = None                          # 7-vector of frozen weights, FLAGS7 order
bal = None                         # fc_gold_bal3000 claim_ids
CELL_RATE, DESERVED = {}, {}       # blind four-cell error rates + deserved-flag lists


def load_weights(path):
    global WV
    wt = json.loads(Path(path).read_text())["weights"]
    WV = np.array([wt[f] for f in FLAGS7])


def load_blind(path):
    """BLIND four-cell error rates + deserved-flag distributions (blind re-judge)."""
    blind = json.loads(Path(path).read_text())
    for c in ("R-F", "R-T", "S-T", "S-F"):
        k, n = blind["cells"][c]["blind_reader_error"]
        CELL_RATE[c] = k / n
    des = collections.defaultdict(list)
    for r in blind["records"]:
        if r["blind_is_reader_error"]:
            des[r["cell"]].append(r["blind_flag"])
    for c in ("R-F", "R-T", "S-T", "S-F"):
        DESERVED[c] = des[c]
    assert CELL_RATE == {"R-F": 3 / 30, "R-T": 22 / 30, "S-T": 0 / 30, "S-F": 11 / 30}, CELL_RATE
    print("cell rates", {c: f"{int(round(CELL_RATE[c] * 30))}/30" for c in CELL_RATE})
    print("deserved  ", {c: dict(collections.Counter(DESERVED[c])) for c in DESERVED})


def cell_of(flag, y):
    if flag in ("1", "2"):
        return "R-F" if y == 0 else "R-T"
    if flag in ("5", "4"):
        return "S-F" if y == 0 else "S-T"
    return None


def prep4(doc_rows):
    """base = non-correctable flags (3/X/I + pad); corr[i] = list of (flag, cell)
    for every correctable doc (any 1/2/5/4); norefute_false[i] = original no-refute-false."""
    n = len(doc_rows)
    base = np.zeros((n, 7))
    corr = [[] for _ in range(n)]
    y = np.zeros(n, int); mid = np.zeros(n, bool); fold = np.zeros(n, int); clusters = []
    norefute_false = np.zeros(n, bool)
    for i, d in enumerate(doc_rows):
        y[i] = d["y"]; mid[i] = d["mid"]; fold[i] = d["fold"]; clusters.append(d["cluster"])
        cnt = collections.Counter()
        has_refute = False
        for fl in d["real_flags"]:
            cell = cell_of(fl, d["y"])
            if fl in ("1", "2"):
                has_refute = True
            if cell is not None:
                corr[i].append((fl, cell))
            else:
                cnt[fl] += 1
        n_pad = max(0, fit_urn.PAD_TO - len(d["real_flags"]))
        cnt["I"] += n_pad
        for f, c in cnt.items():
            base[i, FIDX[f]] += c
        norefute_false[i] = (d["y"] == 0 and not has_refute)
    return base, corr, y, mid, fold, clusters, norefute_false


def draw_counts4(base, corr, do_correct, base_inject, rng):
    C = base_inject.copy() if base_inject is not None else base.copy()
    for i in range(len(base)):
        for fl, cell in corr[i]:
            if do_correct and CELL_RATE[cell] > 0 and rng.random() < CELL_RATE[cell]:
                pool = DESERVED[cell]
                nf = pool[rng.integers(len(pool))]
            else:
                nf = fl
            C[i, FIDX[nf]] += 1
    return C.T  # 7 x n


def make_inject_base(base, norefute_false):
    """One refute (flag 1) per original no-refute-false claim: swap an I slot, else add."""
    b = base.copy(); n_swap = n_add = 0
    for i in range(len(base)):
        if norefute_false[i]:
            if b[i, FIDX["I"]] > 0:
                b[i, FIDX["I"]] -= 1; n_swap += 1
            else:
                n_add += 1
            b[i, FIDX["1"]] += 1
    return b, n_swap, n_add


def flag_counts_frozen(C, y, fold):
    s = WV @ C
    auc = float(_auc_np(s[y == 1], s[y == 0]))
    thrs = []; flagged = np.zeros(len(y), bool)
    for k in range(K):
        te = fold == k
        if not te.any() or te.all():
            continue
        _, _, thr = fit_urn.recall_at_fpr(list(zip(s[~te].tolist(), y[~te].tolist())), 0.02)
        flagged[te] = s[te] <= thr; thrs.append(float(thr))
    return {"auc": auc, "recall": float(flagged[y == 0].mean()),
            "fpr": float(flagged[y == 1].mean()), "threshold": float(np.mean(thrs)),
            "false_flagged": int(flagged[y == 0].sum()), "true_flagged": int(flagged[y == 1].sum())}


def summarize(dicts):
    keys = ["auc", "recall", "fpr", "threshold", "false_flagged", "true_flagged"]
    out = {}
    for k in keys:
        a = np.array([d[k] for d in dicts], float)
        out[k] = {"mean": float(a.mean()), "p2.5": float(np.percentile(a, 2.5)),
                  "p97.5": float(np.percentile(a, 97.5))}
    return out


def run_regime(path, label):
    dr = L.load_docs(path, bal)
    base, corr, y, mid, fold, clusters, nrf = prep4(dr)
    inj_base, ns, na = make_inject_base(base, nrf)
    n_inj = ns + na
    print(f"\n=== {label} === n={len(dr)} T={int((y==1).sum())} F={int((y==0).sum())} "
          f"no-refute-false injected={n_inj} (swap {ns}/add {na})")

    res = {"n": len(dr), "n_true": int((y == 1).sum()), "n_false": int((y == 0).sum()),
           "injected": n_inj}

    # ---- FROZEN (A/B) ----
    # baseline (deterministic)
    C0 = draw_counts4(base, corr, False, None, np.random.default_rng(SEED))
    res["frozen_baseline"] = flag_counts_frozen(C0, y, fold)
    # inject only (deterministic)
    Ci = draw_counts4(base, corr, False, inj_base, np.random.default_rng(SEED))
    res["frozen_inject"] = flag_counts_frozen(Ci, y, fold)
    # readerfix (200 draws)
    rng = np.random.default_rng(SEED)
    rf = [flag_counts_frozen(draw_counts4(base, corr, True, None, rng), y, fold) for _ in range(NDRAW)]
    res["frozen_readerfix"] = summarize(rf)
    # both (200 draws)
    rng = np.random.default_rng(SEED)
    bo = [flag_counts_frozen(draw_counts4(base, corr, True, inj_base, rng), y, fold) for _ in range(NDRAW)]
    res["frozen_both"] = summarize(bo)

    # ---- REFIT (C): corrected reads, weights refit oof ----
    rng = np.random.default_rng(SEED)
    rc = [RF.metrics(draw_counts4(base, corr, True, None, rng), y, mid, fold) for _ in range(NDRAW)]
    res["refit_readerfix"] = summarize(rc)

    # print
    def line(tag, m):
        g = lambda k: m[k]["mean"] if isinstance(m[k], dict) else m[k]
        print(f"  {tag:22s} AUC {g('auc'):.4f} rec {g('recall'):.4f} "
              f"F {g('false_flagged'):.0f} T {g('true_flagged'):.0f} thr {g('threshold'):+.3f}")
    line("frozen baseline", res["frozen_baseline"])
    line("frozen readerfix", res["frozen_readerfix"])
    line("frozen inject", res["frozen_inject"])
    line("frozen both", res["frozen_both"])
    line("refit readerfix (C)", res["refit_readerfix"])

    # two asked numbers (frozen readerfix vs frozen baseline)
    dfalse = res["frozen_readerfix"]["false_flagged"]["mean"] - res["frozen_baseline"]["false_flagged"]
    dtrue = res["frozen_baseline"]["true_flagged"] - res["frozen_readerfix"]["true_flagged"]["mean"]
    res["delta_false_flagged_more"] = float(dfalse)
    res["delta_true_flagged_fewer"] = float(dtrue)
    print(f"  --> more false flagged: +{dfalse:.0f}   fewer true flagged: -{dtrue:.0f}")
    return res


def main():
    global bal
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=TRAIN)
    ap.add_argument("--prod", type=Path, default=PROD)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--population", type=Path, default=POPULATION)
    ap.add_argument("--blind", type=Path, default=BLIND_PATH)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting the pinned frozen deliverable")
    args = ap.parse_args()
    if Path(args.out).resolve() == Path(OUT).resolve() and not args.force:
        raise SystemExit(f"refusing to overwrite the pinned frozen deliverable {OUT}: run with "
                         f"--out PATH to reproduce the trust asserts and write elsewhere, or "
                         f"--force to regenerate it in place.")

    load_weights(args.weights)
    load_blind(args.blind)
    bal = set(pd.read_parquet(args.population)["claim_id"].tolist())

    # trust gates
    drt = L.load_docs(args.train, bal)
    bt, cRt, cSt, yt, midt, foldt, clt = RF.prep(drt)
    C0t = bt.copy()
    for i in range(len(bt)):
        for fl in cRt[i]:
            C0t[i, FIDX[fl]] += 1
        for fl in cSt[i]:
            C0t[i, FIDX[fl]] += 1
    st = fit_urn.oof_np(C0t.T, yt, midt, foldt)
    auc_train_oof = float(_auc_np(st[yt == 1], st[yt == 0]))
    nf_t = fit_urn.nested_threshold(C0t.T, yt, midt, foldt, 0.02)
    assert abs(auc_train_oof - 0.8569157777777778) < 1e-9, auc_train_oof
    assert abs(nf_t["recall"] - 0.4166666666666667) < 1e-9, nf_t["recall"]
    print("TRUST training OOF refit", round(auc_train_oof, 4), round(nf_t["recall"], 4))

    out = {"seed": SEED, "ndraws": NDRAW, "weights_file": str(args.weights),
           "cell_rates": {c: CELL_RATE[c] for c in CELL_RATE},
           "deserved_dist": {c: dict(collections.Counter(DESERVED[c])) for c in DESERVED},
           "trust_gates": {"training_oof_refit_auc": auc_train_oof,
                           "training_oof_refit_recall": nf_t["recall"]}}
    out["A_training"] = run_regime(args.train, "A training-mode evidence (frozen weights)")
    # B baseline frozen must equal 0.9266 / 0.372
    out["B_production"] = run_regime(args.prod, "B production-mode evidence (frozen weights)")
    b = out["B_production"]["frozen_baseline"]
    assert abs(b["auc"] - 0.9266) < 2e-3 and abs(b["recall"] - 0.372) < 2e-3, b
    print("TRUST production frozen-as-is baseline reproduced 0.9266 / 0.372")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
