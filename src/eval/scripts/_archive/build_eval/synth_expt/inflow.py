import json, sys
from pathlib import Path
sys.path.insert(0,'.')
from eval.scripts.build_eval.fit_urn import load_headline, auc, recall_at_fpr
FLAGS7=("5","4","3","2","1","X","I")
FIT=json.load(open("eval/data/urn_runs/true_timeline/two_urn_fit.json"))
W_OLD=[f['weights'] for f in FIT['fits7'] if abs(f['eps']-0.10)<1e-9 and f.get('weights')][0]
W_NEW=json.load(open('/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/ec974c97-b7e9-48c5-a9d0-694b5c240611/scratchpad/w_reclass.json'))
sc=lambda fl,w: sum(w[k]*n for k,n in fl.items() if k in w)
gold=load_headline(Path("eval/data/urn_runs/e1_ctx/results-00.jsonl"))
TRUE=[r for r in gold if not r['mid'] and r['y']==1]
old=[sc(r['flags'],W_OLD) for r in TRUE]; new=[sc(r['flags'],W_NEW) for r in TRUE]
print(f"gold TRUE claims n={len(TRUE)}")
print(f"  in flag zone (<= -4.05) with OLD weights: {sum(1 for s in old if s<=-4.05)}")
print(f"  WORST CASE, NEW weights, gate applied to NOBODY: {sum(1 for s in new if s<=-4.05)}")
inflow=sum(1 for a,b in zip(old,new) if a>-4.05 and b<=-4.05)
print(f"  inflow (was safe, now flagged, no gate): {inflow}")
print(f"  of the inflow, how many carry NO refute read (gate cannot help them):",
      sum(1 for r,a,b in zip(TRUE,old,new) if a>-4.05 and b<=-4.05
          and (r['flags'].get('1',0)+r['flags'].get('2',0))==0))
pairs_o=[(sc(r['flags'],W_OLD), 0 if r['mid'] else r['y']) for r in gold]
pairs_n=[(sc(r['flags'],W_NEW), 0 if r['mid'] else r['y']) for r in gold]
for nm,p in (("OLD",pairs_o),("NEW (refit only, no gate)",pairs_n)):
    a=auc(p); rec,fpr,thr=recall_at_fpr(p,0.02)
    print(f"  {nm:26s} AUC {a:.3f}  recall@2%FPR {rec:.3f}  thr {thr:+.2f}")
