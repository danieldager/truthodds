"""Render Stage-3 verification verdicts as a self-contained, filterable HTML viewer.

Reusable for ANY verification run: point it at any verdicts parquet in the verify_survey_run output
schema (columns: cell, reliability, side, domain, main_claim, v_veracity 1-5, v_misinfo, v_type,
justification, n_search_rounds, n_read, source_url, headline). Browse each claim with its cell,
veracity, misinfo type, justification, and a link to the source article; filter by cell / veracity /
misinfo and free-text search.

  cd src && uv run python eval/scripts/claim_sourcing/build_verification_html.py \
      [-i verdicts.parquet] [-o out.html] [-t "Title"]

Defaults render survey_claims/shortlist_verdicts.parquet -> survey_claims/verification_results.html.
"""
import argparse, pandas as pd, json

BASE = "src/eval/data/survey_claims"
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("-i", "--input", default=f"{BASE}/shortlist_verdicts.parquet", help="verdicts parquet")
ap.add_argument("-o", "--output", default=f"{BASE}/verification_results.html", help="output HTML")
ap.add_argument("-t", "--title", default="Survey claim verification", help="page title")
args = ap.parse_args()

d = pd.read_parquet(args.input)
want = ["claim_id", "cell", "reliability", "side", "domain", "main_claim", "v_veracity", "v_misinfo",
        "v_type", "justification", "n_search_rounds", "n_read", "source_url", "headline"]
cols = [c for c in want if c in d.columns]
rows = d[cols].where(pd.notna(d[cols]), None).to_dict("records")

cells = [{"cell": c, "n": int(len(g)), "mean": round(float(g.v_veracity.mean()), 2),
          "misinfo": round(float((g.v_veracity <= 3).mean()), 2)}
         for c, g in d.groupby("cell")]
vdist = {int(k): int(v) for k, v in d.v_veracity.value_counts().sort_index().items()}
data = {"rows": rows, "cells": cells, "vdist": vdist, "total": int(len(d)),
        "mean": round(float(d.v_veracity.mean()), 2),
        "false": int((d.v_veracity <= 2).sum()), "misinfo": int((d.v_veracity <= 3).sum())}
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#fbfbfa;--fg:#1a1a19;--muted:#6b6b68;--card:#fff;--line:#e5e4e1;--accent:#3b5bdb;
 --v1:#c0392b;--v2:#e07b39;--v3:#c99a2e;--v4:#5a9367;--v5:#2f7d4f;
 --RL:#2f6feb;--RR:#c0392b;--UL:#7048b6;--UR:#b8860b;}
@media(prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e8e8e6;--muted:#9a9a97;--card:#202225;--line:#33353a;
 --v1:#e05a4d;--v2:#e08a4f;--v3:#d8b04a;--v4:#6fae7e;--v5:#4f9d6b;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:24px 18px 80px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);font-size:14px;margin:0 0 18px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:16px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:5px 10px 5px 0}
th{color:var(--muted);font-weight:600}
.vbar{display:flex;gap:2px;align-items:flex-end;height:44px;margin-top:6px}
.vbar>div{flex:1;display:flex;flex-direction:column;align-items:center;gap:3px}
.vbar .bar{width:100%;border-radius:3px 3px 0 0}
.vbar small{color:var(--muted);font-size:11px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:14px;position:sticky;top:0;
 background:var(--bg);padding:10px 0;z-index:5;border-bottom:1px solid var(--line)}
.controls input,.controls select{font:inherit;padding:6px 9px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--fg)}
.controls input[type=search]{flex:1;min-width:180px}
label.chk{color:var(--muted);font-size:13px;display:flex;align-items:center;gap:5px}
.count{color:var(--muted);font-size:13px;margin:0 0 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px 15px;margin-bottom:10px}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:7px;font-size:12px}
.badge{padding:2px 8px;border-radius:20px;font-weight:600;color:#fff}
.chip{padding:2px 8px;border-radius:6px;font-weight:600;font-size:12px;border:1px solid var(--line)}
.dom{color:var(--muted)}
.dom a{color:var(--accent);text-decoration:none}.dom a:hover{text-decoration:underline}
.type{color:var(--muted)}
.claim{font-weight:600;margin:2px 0 6px}
.just{color:var(--muted);font-size:13.5px}
.foot{margin-top:8px;font-size:12px;color:var(--muted);display:flex;gap:14px;flex-wrap:wrap}
.foot a{color:var(--accent);text-decoration:none}
</style></head><body><div class="wrap">
<h1>__TITLE__</h1>
<p class="sub" id="sub"></p>
<div class="panel" id="stats"></div>
<div class="controls">
 <input type="search" id="q" placeholder="Search claim / domain / justification…">
 <select id="fcell"><option value="">All cells</option></select>
 <select id="fver"><option value="">All veracities</option><option>5</option><option>4</option><option>3</option><option>2</option><option>1</option></select>
 <label class="chk"><input type="checkbox" id="fmis"> misinfo only (&le;3)</label>
</div>
<p class="count" id="count"></p>
<div id="list"></div>
</div>
<script type="application/json" id="data">__PAYLOAD__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const VC={1:'var(--v1)',2:'var(--v2)',3:'var(--v3)',4:'var(--v4)',5:'var(--v5)'};
const VL={1:'False',2:'Mostly false',3:'Mixed / unsupported',4:'Mostly true',5:'True'};
const CC={'Reliable-Left':'var(--RL)','Reliable-Right':'var(--RR)','Unreliable-Left':'var(--UL)','Unreliable-Right':'var(--UR)'};
const esc=s=>(s==null?'':String(s)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.getElementById('sub').textContent=`${D.total} claims across ${D.cells.length} cells · mean veracity ${D.mean} · ${D.misinfo} flagged (≤3) · ${D.false} false (≤2)`;
let sh='<table><tr><th>Cell</th><th>n</th><th>mean veracity</th><th>misinfo rate</th></tr>';
D.cells.forEach(c=>{sh+=`<tr><td><span class="badge" style="background:${CC[c.cell]||'#888'}">${esc(c.cell)}</span></td><td>${c.n}</td><td>${c.mean}</td><td>${Math.round(c.misinfo*100)}%</td></tr>`;});
sh+='</table><div class="vbar">';
const mx=Math.max(1,...Object.values(D.vdist));
for(let v=1;v<=5;v++){const n=D.vdist[v]||0;sh+=`<div><div class="bar" style="height:${Math.round(6+38*n/mx)}px;background:${VC[v]}"></div><small>${v}</small><small>${n}</small></div>`;}
sh+='</div>';document.getElementById('stats').innerHTML=sh;
const fc=document.getElementById('fcell');D.cells.forEach(c=>{const o=document.createElement('option');o.value=c.cell;o.textContent=c.cell;fc.appendChild(o);});
const q=document.getElementById('q'),fver=document.getElementById('fver'),fmis=document.getElementById('fmis'),list=document.getElementById('list'),count=document.getElementById('count');
function render(){
 const t=q.value.toLowerCase(),cell=fc.value,ver=fver.value,mis=fmis.checked;
 const out=D.rows.filter(r=>{
  if(cell&&r.cell!==cell)return false;
  if(ver&&String(r.v_veracity)!==ver)return false;
  if(mis&&!(r.v_veracity<=3))return false;
  if(t&&!((r.main_claim||'')+' '+(r.domain||'')+' '+(r.justification||'')).toLowerCase().includes(t))return false;
  return true;});
 count.textContent=`${out.length} of ${D.total} claims`;
 list.innerHTML=out.map(r=>`<div class="card">
  <div class="meta">
   <span class="badge" style="background:${CC[r.cell]||'#888'}">${esc(r.cell)}</span>
   <span class="chip" style="border-color:${VC[r.v_veracity]};color:${VC[r.v_veracity]}">veracity ${r.v_veracity} · ${VL[r.v_veracity]||''}</span>
   ${r.v_type?`<span class="type">${esc(r.v_type)}</span>`:''}
   <span class="dom">${r.source_url?`<a href="${esc(r.source_url)}" target="_blank" rel="noopener">${esc(r.domain)}</a>`:esc(r.domain)}</span>
  </div>
  <div class="claim">${esc(r.main_claim)}</div>
  <div class="just">${esc(r.justification)}</div>
  <div class="foot">${r.n_search_rounds!=null?`<span>${r.n_search_rounds} search round(s), ${r.n_read} doc(s) read</span>`:''}${r.source_url?`<a href="${esc(r.source_url)}" target="_blank" rel="noopener">source ↗</a>`:''}</div>
 </div>`).join('');
}
[q,fc,fver].forEach(e=>e.addEventListener('input',render));fmis.addEventListener('change',render);
render();
</script></body></html>"""

open(args.output, "w").write(HTML.replace("__TITLE__", args.title).replace("__PAYLOAD__", payload))
print(f"wrote {args.output} ({data['total']} claims from {args.input})")
