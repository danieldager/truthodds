import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
_ap=argparse.ArgumentParser()
_ap.add_argument("--out", required=True, help="scratchpad dir with intermediate inputs (taskA_out.json)")
SP=Path(_ap.parse_args().out)
FIG=Path("src/eval/data/urn_runs/e1_prodregime/figures")
A=json.load(open(SP/"taskA_out.json"))
plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.grid":True,
                     "grid.color":"0.9","grid.linewidth":0.6,"axes.axisbelow":True})
K="black"; G="0.55"  # shipped dark, S mid-gray -> distinguish by linestyle+marker instead
SHIP=dict(color="0.0",ls="--",marker="s",ms=3,lw=1.4,label="shipped (date ceiling)")
S=dict(color="0.0",ls="-",marker="o",ms=3,lw=1.4,label="Arm S (no ceiling)")

fig=plt.figure(figsize=(13.5,8.2))
gs=GridSpec(2,3,figure=fig,hspace=0.34,wspace=0.30)

# ---- Panel 1: ROC ----
ax=fig.add_subplot(gs[0,0])
p1=A["panel1_roc"]
ax.plot(p1["shipped"]["fpr"],p1["shipped"]["recall"],color="0.0",ls="--",lw=1.4)
ax.plot(p1["S"]["fpr"],p1["S"]["recall"],color="0.0",ls="-",lw=1.4)
ax.plot([0,1],[0,1],color="0.75",lw=0.8,ls=":")
ax.axvline(0.02,color="0.6",lw=0.8,ls=":")
sh=p1["shipped"]; ss=p1["S"]
ax.text(0.33,0.42,f"S AUC {ss['auc']:.3f} [{ss['auc_ci'][0]:.3f}, {ss['auc_ci'][1]:.3f}]",fontsize=8)
ax.text(0.33,0.33,f"shipped AUC {sh['auc']:.3f} [{sh['auc_ci'][0]:.3f}, {sh['auc_ci'][1]:.3f}]",fontsize=8)
ax.plot([],[],**SHIP);ax.plot([],[],**S);ax.legend(loc="lower right",fontsize=7.5,framealpha=0.9)
ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.set_xlabel("false-positive rate (true claims flagged)")
ax.set_ylabel("recall (false claims flagged)")
ax.set_title("(1) ROC — false-claim recall vs FPR",fontsize=9.5)

# ---- Panel 2: weights side by side ----
ax=fig.add_subplot(gs[0,1])
p2=A["panel2_weights"]; flags=["5","4","X","I","2","1"]
xpos=np.arange(len(flags)); w=0.38
for j,(cond,st) in enumerate((("shipped",dict(color="0.0",hatch="////")),("S",dict(color="0.55")))):
    ws=[p2[cond]["weights"][f] for f in flags]
    ci=[p2[cond]["weights_ci"][f] for f in flags]
    err=[[ws[i]-ci[i][0] for i in range(len(flags))],[ci[i][1]-ws[i] for i in range(len(flags))]]
    ax.bar(xpos+(j-0.5)*w,ws,w,yerr=err,capsize=2,edgecolor="black",linewidth=0.6,
           error_kw=dict(lw=0.8),label=("shipped" if cond=="shipped" else "Arm S"),
           color=("white" if cond=="shipped" else "0.6"),hatch=st.get("hatch"))
ax.axhline(0,color="black",lw=0.7)
ax.set_xticks(xpos);ax.set_xticklabels(flags)
ax.set_xlabel("flag  (5 supports … 1 contradicts)")
ax.set_ylabel("fitted log-odds weight")
ax.legend(fontsize=7.5);ax.set_title("(2) Six-flag weights (95% bootstrap CI)",fontsize=9.5)

# ---- Panel 3: recall vs FPR budget ----
ax=fig.add_subplot(gs[0,2])
p3=A["panel3_recall_vs_budget"]; bud=[x*100 for x in p3["budgets"]]
ax.plot(bud,p3["shipped"],color="0.0",ls="--",marker="s",ms=3,lw=1.3)
ax.plot(bud,p3["S"],color="0.0",ls="-",marker="o",ms=3,lw=1.3)
ax.axvline(2,color="0.6",lw=0.8,ls=":")
ax.text(2.15,0.05,"2% budget",fontsize=7,color="0.4",rotation=90,va="bottom")
ax.plot([],[],**SHIP);ax.plot([],[],**S);ax.legend(loc="lower right",fontsize=7.5)
ax.set_xlabel("FPR budget (%)");ax.set_ylabel("false-claim recall")
ax.set_ylim(0,1);ax.set_title("(3) Recall vs FPR budget",fontsize=9.5)

# ---- Panel 4: evidence mix ----
ax=fig.add_subplot(gs[1,0])
p4=A["panel4_evidence_mix"]
# grouped: 3 metrics x (shipped/S) x (false/true). Show shares as bars; mean docs annotated.
metrics=[("share_ge1_refute","≥1 refuting"),("share_all_silent","all-silent")]
groups=[("shipped","false"),("S","false"),("shipped","true"),("S","true")]
labels=["ship\nF","S\nF","ship\nT","S\nT"]
xpos=np.arange(len(groups)); w=0.38
ref=[p4[c][l]["share_ge1_refute"] for c,l in groups]
sil=[p4[c][l]["share_all_silent"] for c,l in groups]
ax.bar(xpos-w/2,ref,w,color="0.25",edgecolor="black",lw=0.5,label="≥1 refuting flag")
ax.bar(xpos+w/2,sil,w,color="white",edgecolor="black",lw=0.7,hatch="....",label="all-silent (all-I)")
for i,(c,l) in enumerate(groups):
    top=max(p4[c][l]["share_ge1_refute"],p4[c][l]["share_all_silent"])
    ax.text(i,top+0.015,f"{p4[c][l]['mean_docs']:.1f} docs",ha="center",fontsize=6.5,color="0.35")
ax.set_xticks(xpos);ax.set_xticklabels(labels,fontsize=8)
ax.set_ylabel("share of claims");ax.set_ylim(0,0.7)
ax.legend(fontsize=7.5,loc="upper right");ax.set_title("(4) Evidence mix by condition x gold",fontsize=9.5)

# ---- Panel 5: windowed ceiling (spans 2 cells) ----
ax=fig.add_subplot(gs[1,1:])
p5=A["panel5_windowed"]; wl=p5["windows"]; x=np.arange(len(wl))
axr=ax.twinx()
l1,=ax.plot(x,p5["keep_undated"]["auc"],color="0.0",ls="-",marker="o",ms=4,lw=1.4)
l2,=ax.plot(x,p5["drop_undated"]["auc"],color="0.0",ls="--",marker="^",ms=4,lw=1.4)
l3,=axr.plot(x,p5["keep_undated"]["recall"],color="0.55",ls="-",marker="s",ms=4,lw=1.4)
l4,=axr.plot(x,p5["drop_undated"]["recall"],color="0.55",ls="--",marker="D",ms=4,lw=1.4)
ax.set_xticks(x);ax.set_xticklabels([w if w=="unlimited" else f"+{w}d" for w in wl])
ax.set_xlabel("synthetic date ceiling  (claim date + window)")
ax.set_ylabel("oof AUC");axr.set_ylabel("recall @ 2% FPR",color="0.4")
axr.tick_params(axis="y",colors="0.4")
ax.set_ylim(0.70,0.95);axr.set_ylim(0.10,0.50)
ax.legend([l1,l2,l3,l4],["AUC keep-undated","AUC drop-undated","recall keep-undated","recall drop-undated"],
          fontsize=7.5,loc="lower right",ncol=2)
nfs=p5["nfs_unlimited_S"]
ax.set_title(f"(5) Windowed ceiling on S retrieval  (undated docs {A['undated_frac_kept_S']['frac']*100:.0f}% of kept)",fontsize=9.5)
fig.suptitle("Figure 3 — Date ceiling: shipped (ceiling) vs Arm S (production, no ceiling), fc_gold_rep1500 (750T/750F)",
             fontsize=11,y=0.985)
fig.savefig(FIG/"date_ceiling.png",dpi=150,bbox_inches="tight")
print("wrote",FIG/"date_ceiling.png")

# supplementary: standalone windowed panel with NFS markers, larger
figb,axb=plt.subplots(figsize=(7.5,5))
axc=axb.twinx()
axb.plot(x,p5["keep_undated"]["auc"],color="0.0",ls="-",marker="o",lw=1.5,label="AUC keep-undated")
axb.plot(x,p5["drop_undated"]["auc"],color="0.0",ls="--",marker="^",lw=1.5,label="AUC drop-undated")
axc.plot(x,p5["keep_undated"]["recall"],color="0.55",ls="-",marker="s",lw=1.5,label="recall keep-undated")
axc.plot(x,p5["drop_undated"]["recall"],color="0.55",ls="--",marker="D",lw=1.5,label="recall drop-undated")
axc.axhline(nfs["recall"],color="0.55",ls=":",lw=1.0)
axc.text(0.1,nfs["recall"]+0.005,f"S never-flag-on-silence recall {nfs['recall']:.3f} (AUC {nfs['auc']:.3f})",fontsize=7.5,color="0.35")
axb.set_xticks(x);axb.set_xticklabels([w if w=="unlimited" else f"+{w}d" for w in wl])
axb.set_xlabel("synthetic date ceiling (claim date + window)");axb.set_ylabel("oof AUC")
axc.set_ylabel("recall @ 2% FPR",color="0.4")
axb.set_title("Supp. — S retrieval under synthetic date ceilings")
h1,la1=axb.get_legend_handles_labels();h2,la2=axc.get_legend_handles_labels()
axb.legend(h1+h2,la1+la2,fontsize=7.5,loc="center right")
figb.savefig(FIG/"date_ceiling_windowed.png",dpi=150,bbox_inches="tight")
print("wrote",FIG/"date_ceiling_windowed.png")
