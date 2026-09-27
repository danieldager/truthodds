"""Optimistic bound: apply the measured label-matched transition matrices to
every refute read (gold_true matrix on true claims, cn_false matrix on falses).
Label-aware, so it OVERSTATES; paired with the conservative bound it brackets."""
import json, collections, random, sys
from pathlib import Path
sys.path.insert(0,'.')
from eval.scripts.build_eval.fit_urn import load_headline
FLAGS7=("5","4","3","2","1","X","I")
FIT=json.load(open("eval/data/urn_runs/true_timeline/two_urn_fit.json"))
W_OLD=[f['weights'] for f in FIT['fits7'] if abs(f['eps']-0.10)<1e-9 and f.get('weights')][0]
W_NEW=json.load(open('/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/ec974c97-b7e9-48c5-a9d0-694b5c240611/scratchpad/w_reclass.json'))
V={(x['review_url'],x['doc_idx']):x for x in map(json.loads,open('eval/data/urn_runs/synth_expt/refute_verify_verdicts.jsonl'))}
R={(x['review_url'],x['doc_idx']):x for x in map(json.loads,open('eval/data/urn_runs/synth_expt/refute_reclass.jsonl'))}
T=collections.defaultdict(collections.Counter)
for k,v in V.items():
    T[(v['population'],v['flag'])][v['flag'] if v['refutes'] else R.get(k,{}).get('flag_now','I')]+=1
def draw(pop,flag,rng):
    c=T[(pop,flag)]
    if not sum(c.values()): return flag
    r=rng.random()*sum(c.values()); acc=0
    for f,n in c.items():
        acc+=n
        if r<=acc: return f
    return flag
sc=lambda fl,w: sum(w[k]*n for k,n in fl.items() if k in w)
gold=load_headline(Path("eval/data/urn_runs/e1_ctx/results-00.jsonl"))
TRUE=[r for r in gold if not r['mid'] and r['y']==1]
def gscore(r,pop,rng):
    s=0.
    for k,n in r['flags'].items():
        for _ in range(n):
            f=draw(pop,k,rng) if k in ("1","2") else k
            s+=W_NEW.get(f,0.)
    return s
FA_BASE=sum(1 for r in TRUE if sc(r['flags'],W_OLD)<=-4.05)
WPOP={'flag':581,'cliff':201,'pass':327}; tot=sum(WPOP.values())
CN=[]
for p in ("scores.jsonl","scores_ext.jsonl"):
    excl={e["claim"][:80] for e in json.loads(Path("eval/data/urn_runs/c2_false/fit_exclusions.json").read_text())}
    for line in open(f"eval/data/urn_runs/c2_false/{p}"):
        r=json.loads(line)
        if (r.get("claim_text") or "")[:80] in excl: continue
        fl=collections.Counter()
        for d in r.get("results") or []:
            dr=(d.get("read") or {}).get("direction")
            if dr in FLAGS7: fl[dr]+=1
        if sum(fl.values()) and (fl.get('1',0)+fl.get('2',0))>0: CN.append(dict(fl))
res=[]
for seed in range(5):
    rng=random.Random(seed)
    tn=sorted(gscore(r,'gold_true',rng) for r in TRUE)
    thr=tn[FA_BASE-1]
    hits=sum(1 for fl in CN if gscore({'flags':fl}if 0 else type('o',(),{'flags':fl})(),'cn_false',rng)<=thr) if 0 else \
         sum(1 for fl in CN if sum(W_NEW.get(draw('cn_false',k,rng) if k in ("1","2") else k,0.)
                                    for k,n in fl.items() for _ in range(n))<=thr)
    res.append(hits/len(CN))
base=sum(1 for fl in CN if sum(W_OLD.get(k,0.)*n for k,n in fl.items())<=-4.05)/len(CN)
print(f"CN claims carrying refutes: {len(CN)}")
print(f"  BASELINE reach at -4.05                : {base:.1%}")
print(f"  OPTIMISTIC bound (label-matched gate)  : {sum(res)/len(res):.1%}  at matched FPR ({FA_BASE}/{len(TRUE)})")
print(f"  CONSERVATIVE bound (measured earlier)  : 42.7%")
