"""Figure 2 — seven-class read-v5 vs six-class read-v6.1.

Row 1: rep1500 (750T/750F). Row 2: full 3,236. Per row: overlaid ROC (oof AUC
+ CI in legend, recall@2%FPR annotated), weights side by side (bar, CIs), and
where the read-v5 "3" reads landed under read-v6.1 (per-doc transition counts).
Paired clustered bootstrap dAUC (v6.1_6 - v5_7). Reuses fit_urn fitters.
"""
import argparse, json, sys, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "eval/scripts/build_eval"); sys.path.insert(0, ".")
from eval.scripts.build_eval import fit_urn, graded_urn
from eval.scripts.build_eval.e1_figures import _auc_np

_ap = argparse.ArgumentParser()
_ap.add_argument("--out", required=True, help="scratchpad dir with intermediate inputs/outputs (pool3236_ids.json, fig2_results.json)")
SCR = Path(_ap.parse_args().out)
RESULTS00 = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")        # read-v5
FULL61 = Path("eval/data/urn_runs/e1_ctx_v61/results-v6.1-full3236.jsonl")  # read-v6.1
FIGDIR = Path("eval/data/urn_runs/e1_ctx/figures")
REPS, SEED = 400, 707
F7 = graded_urn.FLAGS7  # 5,4,3,X,I,2,1
F6 = graded_urn.FLAGS6  # 5,4,X,I,2,1

import pandas as pd
rep = set(pd.read_parquet("eval/data/populations/fc_gold_rep1500.parquet")["claim_id"])
pool3236 = set(json.load(open(SCR/"pool3236_ids.json")))

def cnt7(fl): return [fl.get(k,0) for k in F7]
def cnt6(fl): return [fl.get("X",0)+fl.get("3",0) if k=="X" else fl.get(k,0) for k in F6]

def load(path, pop): 
    return {r["review_url"]: r for r in fit_urn.load_headline(path, population=pop)}

def aligned(pop):
    v5 = load(RESULTS00, pop); v61 = load(FULL61, pop)
    urls = sorted(set(v5) & set(v61))
    r5 = [v5[u] for u in urls]; r6 = [v61[u] for u in urls]
    C7 = np.array([cnt7(r["flags"]) for r in r5], float).T
    C6 = np.array([cnt6(r["flags"]) for r in r6], float).T
    y = np.array([r["y"] for r in r5]); mid = np.array([r["mid"] for r in r5])
    fold = np.array([r["fold"] for r in r5]); cl = fit_urn.cluster_index(r5)
    return urls, C7, C6, y, mid, fold, cl

def oof_pairs(C, y, mid, fold):
    s = fit_urn.oof_np(C, y, mid, fold)
    return list(zip(s.tolist(), y.tolist()))

def roc_curve(pairs):
    # detector flags FALSE when score<=thr; TPR=recall_false, FPR=share true flagged
    tr = sorted(s for s,yy in pairs if yy==1); fa = sorted(s for s,yy in pairs if yy==0)
    import bisect
    xs=[0.0]; ys=[0.0]
    for thr in sorted({s for s,_ in pairs}):
        fpr = bisect.bisect_right(tr, thr)/len(tr)
        rec = bisect.bisect_right(fa, thr)/len(fa)
        xs.append(fpr); ys.append(rec)
    xs.append(1.0); ys.append(1.0)
    return xs, ys

def analyze(pop, name):
    urls, C7, C6, y, mid, fold, cl = aligned(pop)
    p7 = oof_pairs(C7, y, mid, fold); p6 = oof_pairs(C6, y, mid, fold)
    a7 = fit_urn.auc(p7); a6 = fit_urn.auc(p6)
    r7 = fit_urn.recall_at_fpr(p7, 0.02); r6 = fit_urn.recall_at_fpr(p6, 0.02)
    w7 = fit_urn.fit_np(C7, y, mid, np.arange(C7.shape[1]))
    w6 = fit_urn.fit_np(C6, y, mid, np.arange(C6.shape[1]))
    # paired cluster bootstrap
    rb = np.random.default_rng(SEED)
    A7=[]; A6=[]; D=[]; W7=[]; W6=[]
    for _ in range(REPS):
        idx = fit_urn.boot_idx(rb, y, cl, True)
        s7 = fit_urn.oof_np(C7, y, mid, fold, idx); s6 = fit_urn.oof_np(C6, y, mid, fold, idx)
        yy = y[idx]
        x7=_auc_np(s7[yy==1],s7[yy==0]); x6=_auc_np(s6[yy==1],s6[yy==0])
        A7.append(x7); A6.append(x6); D.append(x6-x7)
        W7.append(fit_urn.fit_np(C7,y,mid,idx)); W6.append(fit_urn.fit_np(C6,y,mid,idx))
    W7=np.array(W7); W6=np.array(W6)
    ci=lambda a:[float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]
    # "3" transition per-doc: v5 dir vs v6.1 dir on same (url,rank)
    v5recs={}
    for l in RESULTS00.open():
        r=json.loads(l)
        if r.get("review_url") in pop: v5recs[r["review_url"]]=r
    v61recs={}
    for l in FULL61.open():
        r=json.loads(l)
        if r.get("review_url") in pop: v61recs[r["review_url"]]=r
    trans=collections.Counter()
    for u in set(v5recs)&set(v61recs):
        d61={d["rank"]:(d.get("read") or {}).get("direction") for d in v61recs[u].get("results") or []}
        for d in v5recs[u].get("results") or []:
            dr=(d.get("read") or {}).get("direction")
            if dr=="3":
                trans[d61.get(d["rank"],"MISS")]+=1
    out={"name":name,"n":len(urls),"n_true":int(y.sum()),
         "v5_7flag":{"oof_auc":float(a7),"auc_ci":ci(A7),
                     "recall_at_2pct_fpr":float(r7[0]),"fpr":float(r7[1]),"thr":float(r7[2]),
                     "weights":{k:float(w7[i]) for i,k in enumerate(F7)},
                     "weight_ci":{k:ci(W7[:,i]) for i,k in enumerate(F7)}},
         "v61_6flag":{"oof_auc":float(a6),"auc_ci":ci(A6),
                      "recall_at_2pct_fpr":float(r6[0]),"fpr":float(r6[1]),"thr":float(r6[2]),
                      "weights":{k:float(w6[i]) for i,k in enumerate(F6)},
                      "weight_ci":{k:ci(W6[:,i]) for i,k in enumerate(F6)}},
         "dAUC_v61_minus_v5":float(a6-a7),"dAUC_ci":ci(D),
         "three_transition":dict(trans),"three_total":int(sum(trans.values()))}
    out["_roc7"]=roc_curve(p7); out["_roc6"]=roc_curve(p6)
    print(f"[{name}] n={len(urls)} v5-7 AUC {a7:.4f}{ci(A7)} rec {r7[0]:.3f} | v6.1-6 AUC {a6:.4f}{ci(A6)} rec {r6[0]:.3f} | dAUC {a6-a7:+.4f} {ci(D)}")
    print(f"   '3'->v6.1: {dict(trans)} (total {sum(trans.values())})")
    return out

res = {"rep1500": analyze(rep, "rep1500"), "full3236": analyze(pool3236, "full3236")}
# strip roc curves before json dump (keep separately)
dump = {k:{kk:vv for kk,vv in v.items() if not kk.startswith("_roc")} for k,v in res.items()}
json.dump(dump, open(SCR/"fig2_results.json","w"), indent=1)

# ---- FIGURE: 2 rows x 3 cols ----
plt.rcParams.update({"font.family":"sans-serif","axes.grid":True,"grid.alpha":0.3})
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ri,(key,r) in enumerate([("rep1500",res["rep1500"]),("full3236",res["full3236"])]):
    axR, axW, axT = axes[ri]
    # ROC
    x7,y7=r["_roc7"]; x6,y6=r["_roc6"]
    v5=r["v5_7flag"]; v61=r["v61_6flag"]
    axR.plot(x7,y7,color="0.55",ls="--",lw=1.6,
             label=f"read-v5 7-flag  AUC {v5['oof_auc']:.3f} [{v5['auc_ci'][0]:.3f},{v5['auc_ci'][1]:.3f}]")
    axR.plot(x6,y6,color="0.0",ls="-",lw=1.6,
             label=f"read-v6.1 6-flag  AUC {v61['oof_auc']:.3f} [{v61['auc_ci'][0]:.3f},{v61['auc_ci'][1]:.3f}]")
    axR.plot([0,1],[0,1],color="0.8",lw=0.8)
    axR.axvline(0.02,color="0.7",lw=0.7,ls=":")
    axR.set_xlim(0,1); axR.set_ylim(0,1)
    axR.set_xlabel("FPR (true claims flagged)"); axR.set_ylabel("recall (false claims flagged)")
    axR.set_title(f"{key}: ROC (dAUC {r['dAUC_v61_minus_v5']:+.4f} [{r['dAUC_ci'][0]:+.4f},{r['dAUC_ci'][1]:+.4f}])")
    axR.legend(fontsize=7.5, loc="lower right")
    axR.text(0.03,0.06,f"recall@2%FPR\nv5 {v5['recall_at_2pct_fpr']:.3f} / v6.1 {v61['recall_at_2pct_fpr']:.3f}",
             fontsize=7.5, transform=axR.transAxes, va="bottom",
             bbox=dict(boxstyle="round",fc="white",ec="0.7",alpha=0.9))
    # weights
    order = F7  # union display order 5,4,3,X,I,2,1
    xpos=np.arange(len(order)); wd=0.4
    w5v=[v5["weights"].get(k,np.nan) for k in order]
    w5lo=[v5["weights"].get(k,np.nan)-v5["weight_ci"].get(k,[np.nan,np.nan])[0] if k in v5["weights"] else 0 for k in order]
    w5hi=[v5["weight_ci"].get(k,[np.nan,np.nan])[1]-v5["weights"].get(k,np.nan) if k in v5["weights"] else 0 for k in order]
    w6v=[v61["weights"].get(k,np.nan) for k in order]
    w6lo=[v61["weights"].get(k,np.nan)-v61["weight_ci"].get(k,[np.nan,np.nan])[0] if k in v61["weights"] else 0 for k in order]
    w6hi=[v61["weight_ci"].get(k,[np.nan,np.nan])[1]-v61["weights"].get(k,np.nan) if k in v61["weights"] else 0 for k in order]
    axW.bar(xpos-wd/2, w5v, wd, yerr=[w5lo,w5hi], color="0.6", ecolor="0.3", capsize=2, label="read-v5 7-flag")
    axW.bar(xpos+wd/2, w6v, wd, yerr=[w6lo,w6hi], color="0.15", ecolor="0.0", capsize=2, label="read-v6.1 6-flag")
    axW.axhline(0,color="0.7",lw=0.7)
    axW.set_xticks(xpos); axW.set_xticklabels(order)
    axW.set_xlabel("reader flag"); axW.set_ylabel("fitted log-odds weight")
    axW.set_title(f"{key}: weights (v6.1 has no flag 3)")
    axW.legend(fontsize=7.5)
    # transition
    tkeys=[k for k in ("5","4","X","I","2","1","MISS") if k in r["three_transition"]]
    tvals=[r["three_transition"][k] for k in tkeys]
    axT.bar(range(len(tkeys)), tvals, color="0.3")
    axT.set_xticks(range(len(tkeys))); axT.set_xticklabels(tkeys)
    for i,v in enumerate(tvals): axT.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    axT.set_xlabel("read-v6.1 flag"); axT.set_ylabel("count of former read-v5 '3' reads")
    axT.set_title(f"{key}: where read-v5 '3' reads land (n={r['three_total']})")
plt.tight_layout()
out=FIGDIR/"seven_vs_six.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
