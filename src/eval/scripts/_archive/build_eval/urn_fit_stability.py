"""Figure 1 — six-flag weight stability vs n (read-v6.1, frozen 3,236).

Nested balanced random subsets; at each n refit the six weights and cluster-
bootstrap (>=200 reps) for weight CIs and oof AUC CI. Reuses fit_urn/graded_urn
fitters (no rewrite). Panel A: six weights + CI bands vs n (log x), shipped
rep1500 weights marked. Panel B: oof AUC + CI vs n, CI width on 2nd axis.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "eval/scripts/build_eval")
sys.path.insert(0, ".")
from eval.scripts.build_eval import fit_urn, graded_urn
from eval.scripts.build_eval.e1_figures import _auc_np

graded_urn.FLAGS = graded_urn.FLAGS6
FLAGS = graded_urn.FLAGS6                       # 5,4,X,I,2,1
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", required=True, help="scratchpad dir with intermediate inputs/outputs (pool3236_ids.json, fig1_results.json)")
SCR = Path(_ap.parse_args().out)
FULL = Path("eval/data/urn_runs/e1_ctx_v61/results-v6.1-full3236.jsonl")
FIGDIR = Path("eval/data/urn_runs/e1_ctx/figures")
NS = [250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3236]
REPS, SEED = 400, 707

pool3236 = set(json.load(open(SCR / "pool3236_ids.json")))
rows = fit_urn.load_headline(FULL, population=pool3236)
print("loaded rows:", len(rows), "(pool 3236)")
by = {r["review_url"]: r for r in rows}

def cnt6(fl):
    return [fl.get("X",0)+fl.get("3",0) if k=="X" else fl.get(k,0) for k in FLAGS]

def arrays(rws):
    C = np.array([cnt6(r["flags"]) for r in rws], float).T
    y = np.array([r["y"] for r in rws]); mid = np.array([r["mid"] for r in rws])
    fold = np.array([r["fold"] for r in rws]); cl = fit_urn.cluster_index(rws)
    return C, y, mid, fold, cl

def fit_w(rws):
    return graded_urn.fit_graded(rws)  # dict over FLAGS6

def oof_auc(rws):
    return fit_urn.auc(graded_urn.oof_graded(rws))

# nested balanced order: shuffle T and F once, take prefixes
trues = [r for r in rows if r["y"]==1]; falses=[r for r in rows if r["y"]==0]
rng = np.random.default_rng(20260916)
rng.shuffle(trues); rng.shuffle(falses)
nT, nF = len(trues), len(falses)
print(f"pool T {nT} / F {nF}")

def subset(n):
    half = n//2
    t = min(half, nT); f = min(n-t, nF)
    if t+f < n:  # top up from trues if falses short
        t = min(n-f, nT)
    return trues[:t] + falses[:f]

results = {}
shipped = json.load(open("eval/data/urn_runs/e1_ctx/headline_metrics.json"))["overall"]["weights"]
shipped = {k: (shipped[k]["weight"] if isinstance(shipped[k], dict) else shipped[k]) for k in shipped}
print("shipped weights:", shipped)

full_ci = None
for n in NS:
    rws = subset(n)
    w = fit_w(rws)
    auc_pt = oof_auc(rws)
    C, y, mid, fold, cl = arrays(rws)
    rb = np.random.default_rng(SEED)
    wreps = {k: [] for k in FLAGS}; aucreps = []
    for _ in range(REPS):
        idx = fit_urn.boot_idx(rb, y, cl, True)
        wv = fit_urn.fit_np(C, y, mid, idx)
        for i,k in enumerate(FLAGS): wreps[k].append(wv[i])
        s = fit_urn.oof_np(C, y, mid, fold, idx)
        yy = y[idx]
        aucreps.append(_auc_np(s[yy==1], s[yy==0]))
    wci = {k: [float(np.percentile(wreps[k],2.5)), float(np.percentile(wreps[k],97.5))] for k in FLAGS}
    aci = [float(np.percentile(aucreps,2.5)), float(np.percentile(aucreps,97.5))]
    results[n] = {"n": len(rws), "n_true": sum(r["y"] for r in rws),
                  "weights": {k: float(w[k]) for k in FLAGS}, "weight_ci": wci,
                  "oof_auc": float(auc_pt), "auc_ci": aci, "auc_ci_width": aci[1]-aci[0]}
    print(f"n={n:5d}({len(rws)}) AUC {auc_pt:.4f} [{aci[0]:.4f},{aci[1]:.4f}] w5 {w['5']:+.3f} w1 {w['1']:+.3f}")
    if n == 3236:
        full_ci = wci
        results["_full3236_weights"] = {k: float(w[k]) for k in FLAGS}
        results["_full3236_auc"] = float(auc_pt); results["_full3236_auc_ci"] = aci

# shipped-in-3236-CI check
inside = {k: (full_ci[k][0] <= shipped[k] <= full_ci[k][1]) for k in FLAGS}
results["_shipped_rep1500_weights"] = shipped
results["_shipped_inside_full3236_ci"] = inside
results["_all_shipped_inside"] = bool(all(inside.values()))
print("shipped inside 3236 CI:", inside, "ALL:", all(inside.values()))
json.dump(results, open(SCR/"fig1_results.json","w"), indent=1)

# ---- FIGURE ----
plt.rcParams.update({"font.family":"sans-serif","axes.grid":True,"grid.alpha":0.3})
ns = NS
greys = {"5":"0.0","4":"0.25","X":"0.45","I":"0.6","2":"0.35","1":"0.1"}
ls = {"5":"-","4":"-","X":"--","I":":","2":"-.","1":"-"}
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))
for k in FLAGS:
    pts = [results[n]["weights"][k] for n in ns]
    lo = [results[n]["weight_ci"][k][0] for n in ns]
    hi = [results[n]["weight_ci"][k][1] for n in ns]
    axA.plot(ns, pts, marker="o", ms=3.5, color=greys[k], ls=ls[k], label=f"flag {k}")
    axA.fill_between(ns, lo, hi, color=greys[k], alpha=0.12)
    axA.scatter([1500*1.0], [shipped[k]], marker="*", s=90, color=greys[k], zorder=5,
                edgecolor="white", linewidth=0.4)
axA.set_xscale("log"); axA.set_xticks(ns); axA.set_xticklabels([str(n) for n in ns], rotation=45)
axA.set_xlabel("n claims (log scale)"); axA.set_ylabel("fitted log-odds weight")
axA.set_title("A. Six flag weights vs n (star = shipped rep1500)")
axA.axhline(0, color="0.7", lw=0.7)
axA.legend(ncol=3, fontsize=8, loc="center right")

aucs = [results[n]["oof_auc"] for n in ns]
alo = [results[n]["auc_ci"][0] for n in ns]; ahi = [results[n]["auc_ci"][1] for n in ns]
widths = [results[n]["auc_ci_width"] for n in ns]
axB.plot(ns, aucs, marker="o", ms=4, color="0.0", label="oof AUC")
axB.fill_between(ns, alo, ahi, color="0.5", alpha=0.18)
axB.set_xscale("log"); axB.set_xticks(ns); axB.set_xticklabels([str(n) for n in ns], rotation=45)
axB.set_xlabel("n claims (log scale)"); axB.set_ylabel("out-of-fold AUC")
axB.set_title("B. Discrimination and its uncertainty vs n")
ax2 = axB.twinx()
ax2.plot(ns, widths, marker="s", ms=3.5, color="0.55", ls="--", label="95% CI width")
ax2.set_ylabel("AUC 95% CI width"); ax2.grid(False)
l1,la1 = axB.get_legend_handles_labels(); l2,la2 = ax2.get_legend_handles_labels()
axB.legend(l1+l2, la1+la2, fontsize=8, loc="lower right")
plt.tight_layout()
out = FIGDIR / "stability_weights_vs_n.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
