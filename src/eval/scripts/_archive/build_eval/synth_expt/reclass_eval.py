"""Exact trade on fully-verified claims: reclassify + refit vs baseline."""
import json, collections, sys
from pathlib import Path
sys.path.insert(0,'.')
FLAGS7=("5","4","3","2","1","X","I")
FIT=json.load(open("eval/data/urn_runs/true_timeline/two_urn_fit.json"))
W_OLD=[f['weights'] for f in FIT['fits7'] if abs(f['eps']-0.10)<1e-9 and f.get('weights')][0]
W_NEW=json.load(open('/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/ec974c97-b7e9-48c5-a9d0-694b5c240611/scratchpad/w_reclass.json'))
V={(x['review_url'],x['doc_idx']):x for x in map(json.loads,open('eval/data/urn_runs/synth_expt/refute_verify_verdicts.jsonl'))}
R={(x['review_url'],x['doc_idx']):x for x in map(json.loads,open('eval/data/urn_runs/synth_expt/refute_reclass.jsonl'))}
rows=[json.loads(l) for l in open('eval/data/urn_runs/synth_expt/refute_verify_sample.jsonl')]
claims=collections.defaultdict(dict)
for r in rows: claims[(r['population'],r['review_url'])][r['doc_idx']]=r

print("per-read expected score change (score RISES = less flagged):")
for pop,band in (("gold_true","flag"),("cn_false","flag"),("tl_feed","flag")):
    T=collections.Counter()
    for k,v in V.items():
        if v['population']!=pop or v['band']!=band or v['flag']!='1': continue
        T[v['flag'] if v['refutes'] else R.get(k,{}).get('flag_now','I')]+=1
    n=sum(T.values())
    if not n: continue
    exp=sum(T[f]/n*W_NEW[f] for f in T)
    print(f"  {pop:10s} flag-1 reads n={n:4d}   {W_OLD['1']:+.2f} -> {exp:+.2f}   delta {exp-W_OLD['1']:+.2f}")

new={}
for (pop,url),docs in claims.items():
    any_r=next(iter(docs.values())); flags=list(any_r['all_flags'])
    if any((url,i) not in V for i in docs): continue
    for i in docs:
        v=V[(url,i)]
        flags[i]= v['flag'] if v['refutes'] else R.get((url,i),{}).get('flag_now','I')
    s0=sum(W_OLD.get(f,0.) for f in any_r['all_flags'])
    s1=sum(W_NEW.get(f,0.) for f in flags)
    new[(pop,url)]=(s0,s1)

WPOP={'flag':581,'cliff':201,'pass':327}; tot=sum(WPOP.values())
cn=collections.defaultdict(list)
for (p,u),(s0,s1) in new.items():
    if p!='cn_false': continue
    b='flag' if s0<=-4.05 else ('cliff' if s0<=-2.0 else 'pass'); cn[b].append((s0,s1))
gt=[(s0,s1) for (p,u),(s0,s1) in new.items() if p=='gold_true']
base=sum(WPOP[b]*sum(1 for s0,_ in v if s0<=-4.05)/len(v) for b,v in cn.items())/tot
print(f"\nBASELINE thr -4.05: CN reach {base:.1%} | gold-true false alarms {len(gt)}")
print(f"\nRECLASSIFY + REFIT, sweeping threshold:")
print(f"  {'thr':>7} {'CN reach':>9} {'gold-true FA':>14} {'LR+ index':>10}")
for thr in [-8,-7,-6,-5.5,-5,-4.5,-4.05,-3.5,-3.0,-2.5]:
    c=sum(WPOP[b]*sum(1 for _,s1 in v if s1<=thr)/len(v) for b,v in cn.items())/tot
    fa=sum(1 for _,s1 in gt if s1<=thr)
    idx=(c/base)/(fa/len(gt)) if fa else float('inf')
    print(f"  {thr:7.2f} {c:8.1%} {fa:11d}/{len(gt)} {idx:9.2f}x")
