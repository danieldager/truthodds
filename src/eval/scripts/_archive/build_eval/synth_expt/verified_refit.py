"""Refit the 7-flag weights using only VERIFIED refutes. No new API calls:
survival rates measured on the sampled cells are applied to the full urns."""
import json, collections, math, sys
from pathlib import Path
sys.path.insert(0, '.')
from eval.scripts.build_eval.fit_urn import FLAG_TO_VOICE, auc, load_headline, recall_at_fpr

FLAGS7 = ("5","4","3","2","1","X","I")
C2 = Path("eval/data/urn_runs/c2_false")
TL = Path("eval/data/urn_runs/true_timeline/scores.jsonl")
FIT = json.load(open("eval/data/urn_runs/true_timeline/two_urn_fit.json"))
W_OLD = [f['weights'] for f in FIT['fits7'] if abs(f['eps']-0.10)<1e-9 and f.get('weights')][0]

def band_of(s): return "flag" if s<=-4.05 else ("cliff" if s<=-2.0 else "pass")

# measured survival: (pop, band, flag) -> rate
V = [json.loads(l) for l in open('eval/data/urn_runs/synth_expt/refute_verify_verdicts.jsonl')]
cell = collections.defaultdict(lambda:[0,0])
for x in V:
    c = cell[(x['population'], x['band'], x['flag'])]; c[0]+=1; c[1]+=x['refutes']
SURV = {k:(kp/n) for k,(n,kp) in cell.items()}

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
            if not sum(fl.values()): continue
            s=sum(W_OLD.get(k,0.)*n for k,n in fl.items())
            rows.append({"flags":dict(fl),"s0":s,"band":band_of(s),"claim":claim})
    return rows

false_rows = load([C2/"scores.jsonl", C2/"scores_ext.jsonl"], C2/"fit_exclusions.json")
true_rows  = load([TL])
print(f"FALSE urn {len(false_rows)} claims | TRUE urn {len(true_rows)} claims")

def flag_counts(rows):
    c=collections.Counter()
    for r in rows:
        for k,n in r["flags"].items(): c[k]+=n
    return c

def verified_counts(rows, pop, pass_surv=None):
    """Move killed refutes (1,2) into I using measured cell survival."""
    c=collections.Counter()
    for r in rows:
        for k,n in r["flags"].items():
            if k in ("1","2"):
                key=(pop, r["band"], k)
                sv = SURV.get(key)
                if sv is None: sv = pass_surv
                c[k]+=n*sv; c["I"]+=n*(1-sv)
            else: c[k]+=n
    return c

def weights_from(cm, cf, eps=0.10):
    nm, nf = sum(cm.values()), sum(cf.values())
    w={}
    for k in FLAGS7:
        pm=(cm.get(k,0)/nm*nm+1)/(nm+7); pf=(cf.get(k,0)/nf*nf+1)/(nf+7)
        pt=(pm-eps*pf)/(1-eps)
        if pt<=0: return None
        w[k]=math.log(pt/pf)
    return w

base_w = weights_from(flag_counts(true_rows), flag_counts(false_rows))
print("\nBASELINE weights (eps=0.10):", " ".join(f"{k}{base_w[k]:+.2f}" for k in FLAGS7))

print("\nVERIFIED-REFIT weights, sweeping the unmeasured timeline PASS-band survival:")
print(f"  {'tl_pass':>8} " + " ".join(f"{k:>7}" for k in FLAGS7))
out={}
for ps in [0.20,0.25,0.30,0.35,0.40,0.45]:
    cf=verified_counts(false_rows,"cn_false")
    cm=verified_counts(true_rows,"tl_feed",pass_surv=ps)
    w=weights_from(cm,cf)
    out[ps]=w
    print(f"  {ps:8.0%} " + " ".join(f"{w[k]:+7.2f}" for k in FLAGS7))
json.dump({str(k):v for k,v in out.items()}, open('/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/ec974c97-b7e9-48c5-a9d0-694b5c240611/scratchpad/refit_w.json','w'))

# rate detail at ps=0.30
cf=verified_counts(false_rows,"cn_false"); cm=verified_counts(true_rows,"tl_feed",pass_surv=0.30)
cf0, cm0 = flag_counts(false_rows), flag_counts(true_rows)
print("\nRefute/silent RATE shift (ps=30%):")
for k in ("1","2","I"):
    print(f"  flag {k}: FALSE urn {cf0[k]/sum(cf0.values()):.4f} -> {cf[k]/sum(cf.values()):.4f}   "
          f"TRUE urn {cm0[k]/sum(cm0.values()):.4f} -> {cm[k]/sum(cm.values()):.4f}")
