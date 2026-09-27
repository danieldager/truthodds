"""Conservative matched-FPR comparison. Gate applied only where verified
(35 gold-true flag-zone claims); every other gold-true claim gets NO correction."""
import json, collections, sys
from pathlib import Path
sys.path.insert(0,'.')
from eval.scripts.build_eval.fit_urn import load_headline
FIT=json.load(open("eval/data/urn_runs/true_timeline/two_urn_fit.json"))
W_OLD=[f['weights'] for f in FIT['fits7'] if abs(f['eps']-0.10)<1e-9 and f.get('weights')][0]
W_NEW=json.load(open('/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/ec974c97-b7e9-48c5-a9d0-694b5c240611/scratchpad/w_reclass.json'))
sc=lambda fl,w: sum(w[k]*n for k,n in fl.items() if k in w)
V={(x['review_url'],x['doc_idx']):x for x in map(json.loads,open('eval/data/urn_runs/synth_expt/refute_verify_verdicts.jsonl'))}
R={(x['review_url'],x['doc_idx']):x for x in map(json.loads,open('eval/data/urn_runs/synth_expt/refute_reclass.jsonl'))}
rows=[json.loads(l) for l in open('eval/data/urn_runs/synth_expt/refute_verify_sample.jsonl')]
claims=collections.defaultdict(dict)
for r in rows: claims[(r['population'],r['review_url'])][r['doc_idx']]=r
def gated(pop,url,orig_flags):
    docs=claims.get((pop,url))
    if not docs: return orig_flags
    fl=list(orig_flags)
    for i in docs:
        v=V.get((url,i))
        if v is None: continue
        fl[i]= v['flag'] if v['refutes'] else R.get((url,i),{}).get('flag_now','I')
    return fl
gold=load_headline(Path("eval/data/urn_runs/e1_ctx/results-00.jsonl"))
TRUE=[r for r in gold if not r['mid'] and r['y']==1]
def expand(fl):
    out=[]
    for k,n in fl.items(): out+= [k]*n
    return out
tn=[]
for r in TRUE:
    fl=expand(r['flags'])
    g=gated('gold_true', r['review_url'], fl)
    tn.append(sum(W_NEW.get(f,0.) for f in g))
to=[sc(r['flags'],W_OLD) for r in TRUE]
FA_BASE=sum(1 for s in to if s<=-4.05)
tn_sorted=sorted(tn)
thr_new=tn_sorted[FA_BASE-1]
print(f"gold TRUE n={len(TRUE)}: baseline FA at -4.05 = {FA_BASE}  (FPR {FA_BASE/len(TRUE):.2%})")
print(f"matched-FPR threshold under reclassify+refit = {thr_new:+.3f}")
WPOP={'flag':581,'cliff':201,'pass':327}; tot=sum(WPOP.values())
cn=collections.defaultdict(list)
for (p,url),docs in claims.items():
    if p!='cn_false': continue
    any_r=next(iter(docs.values()))
    if any((url,i) not in V for i in docs): continue
    fl=list(any_r['all_flags'])
    s0=sum(W_OLD.get(f,0.) for f in fl)
    b='flag' if s0<=-4.05 else ('cliff' if s0<=-2.0 else 'pass')
    g=gated('cn_false',url,fl)
    cn[b].append((s0,sum(W_NEW.get(f,0.) for f in g)))
base=sum(WPOP[b]*sum(1 for s0,_ in v if s0<=-4.05)/len(v) for b,v in cn.items())/tot
newr=sum(WPOP[b]*sum(1 for _,s1 in v if s1<=thr_new)/len(v) for b,v in cn.items())/tot
print(f"\nAt MATCHED false-alarm count ({FA_BASE} gold-true claims flagged):")
print(f"  BASELINE            CN-refute claims reached {base:.1%}")
print(f"  RECLASSIFY + REFIT  CN-refute claims reached {newr:.1%}")
print(f"  => recall change {newr-base:+.1%} at identical FPR (conservative: gate applied to")
print(f"     the 35 verified gold-trues only, all other true claims uncorrected)")
