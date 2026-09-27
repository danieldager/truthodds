"""Refit the 7-flag weights after RECLASSIFYING (not killing) bad refute reads.
Transition matrix P(flag_now | pop, band, flag_was) measured on the sample is
applied to the full urn counts. No new calls beyond the reclass pass."""
import json, collections, math, sys
from pathlib import Path
sys.path.insert(0,'.')
FLAGS7=("5","4","3","2","1","X","I")
C2=Path("eval/data/urn_runs/c2_false"); TL=Path("eval/data/urn_runs/true_timeline/scores.jsonl")
FIT=json.load(open("eval/data/urn_runs/true_timeline/two_urn_fit.json"))
W_OLD=[f['weights'] for f in FIT['fits7'] if abs(f['eps']-0.10)<1e-9 and f.get('weights')][0]
def band_of(s): return "flag" if s<=-4.05 else ("cliff" if s<=-2.0 else "pass")

V={(x['review_url'],x['doc_idx']):x for x in map(json.loads,open('eval/data/urn_runs/synth_expt/refute_verify_verdicts.jsonl'))}
R={(x['review_url'],x['doc_idx']):x for x in map(json.loads,open('eval/data/urn_runs/synth_expt/refute_reclass.jsonl'))}
# transition counts
T=collections.defaultdict(collections.Counter)
for k,v in V.items():
    key=(v['population'],v['band'],v['flag'])
    if v['refutes']: T[key][v['flag']]+=1          # survived: unchanged
    elif k in R:     T[key][R[k]['flag_now']]+=1   # reclassified
print("transition matrix (flag_was -> distribution), cells with n>=20:")
for key in sorted(T):
    n=sum(T[key].values())
    if n<20: continue
    print(f"  {key[0]:9s} {key[1]:5s} {key[2]} n={n:4d}  " +
          " ".join(f"{f}:{T[key][f]/n:.2f}" for f in FLAGS7 if T[key][f]))

def trans(pop,band,flag):
    for key in ((pop,band,flag),(pop,'flag',flag),('tl_feed',band,flag)):
        if sum(T[key].values())>=15:
            n=sum(T[key].values()); return {f:T[key][f]/n for f in FLAGS7 if T[key][f]}
    return {flag:1.0}

def load(paths, excl_path=None):
    excl=set()
    if excl_path: excl={e["claim"][:80] for e in json.loads(Path(excl_path).read_text())}
    rows=[]
    for p in paths:
        for line in open(p):
            r=json.loads(line)
            claim=r.get("claim_resolved") or r.get("claim_text") or ""
            if claim[:80] in excl: continue
            fl=collections.Counter()
            for d in r.get("results") or []:
                dr=(d.get("read") or {}).get("direction")
                if dr in FLAGS7: fl[dr]+=1
            if sum(fl.values()):
                s=sum(W_OLD.get(k,0.)*n for k,n in fl.items())
                rows.append({"flags":dict(fl),"band":band_of(s),"s0":s})
    return rows

fr=load([C2/"scores.jsonl",C2/"scores_ext.jsonl"],C2/"fit_exclusions.json")
tr=load([TL])

def counts(rows,pop,reclassify=True):
    c=collections.Counter()
    for r in rows:
        for k,n in r["flags"].items():
            if reclassify and k in ("1","2"):
                for f,p in trans(pop,r["band"],k).items(): c[f]+=n*p
            else: c[k]+=n
    return c

def weights(cm,cf,eps=0.10):
    nm,nf=sum(cm.values()),sum(cf.values()); w={}
    for k in FLAGS7:
        pm=(cm.get(k,0)+1)/(nm+7); pf=(cf.get(k,0)+1)/(nf+7)
        pt=(pm-eps*pf)/(1-eps)
        if pt<=0: return None
        w[k]=math.log(pt/pf)
    return w

c0f,c0m=counts(fr,"cn_false",False),counts(tr,"tl_feed",False)
c1f,c1m=counts(fr,"cn_false",True), counts(tr,"tl_feed",True)
w0,w1=weights(c0m,c0f),weights(c1m,c1f)
print("\n            " + " ".join(f"{k:>7}" for k in FLAGS7))
print("BASELINE    " + " ".join(f"{w0[k]:+7.2f}" for k in FLAGS7))
print("RECLASSIFY  " + " ".join(f"{w1[k]:+7.2f}" for k in FLAGS7))
print("\nrate shift (FALSE urn | TRUE urn):")
for k in FLAGS7:
    print(f"  {k}: {c0f[k]/sum(c0f.values()):.4f}->{c1f[k]/sum(c1f.values()):.4f}   "
          f"{c0m[k]/sum(c0m.values()):.4f}->{c1m[k]/sum(c1m.values()):.4f}")
json.dump(w1,open('/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/ec974c97-b7e9-48c5-a9d0-694b5c240611/scratchpad/w_reclass.json','w'))
