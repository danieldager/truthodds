"""Round-2 gold labelling page — the v5 flag schema (2026-08-03).

Differences from round 1 (`build_read_label_ui.py`, 3 buckets, 6 directions):
one flag from {5,4,3,2,1,X,I} (contested now has a button), ONE evidence set
(click a sentence to toggle it in/out), claim and document dates in the header,
and the claim text is the RESOLVED text — the same payloads the silver labellers
see, so Daniel's round-2 labels validate them like-for-like.

  uv run python -m eval.scripts.build_eval.build_read_flag_ui \
      --payload-dir <dir> --cases eval/data/read_suite/cases.json \
      --dual eval/data/read_suite/measure_labels_partial.json \
      -o <out.html> --n 40
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

FLAGS = [("5", "supports"), ("4", "partially supports"), ("3", "contested"),
         ("2", "partially refutes"), ("1", "refutes"),
         ("X", "context (on-claim, non-directional)"), ("I", "irrelevant")]
REASONS = ["off-claim", "adjacent-only"]


def draw(cases_meta, payload_dir, dual_cids, n):
    """Stratified draw over failure modes, dual-labelled cases first."""
    have = {p.stem.removeprefix("case_") for p in payload_dir.glob("case_*.json")}
    pool = [c for c in cases_meta if c["cid"] in have]
    picked, seen = [], set()
    for c in pool:                                    # all dual-labelled first
        if c["cid"] in dual_cids and c["cid"] not in seen:
            picked.append(c)
            seen.add(c["cid"])
    by_mode = collections.defaultdict(list)
    for c in pool:
        if c["cid"] not in seen:
            by_mode[(c.get("_modes") or ["?"])[0]].append(c)
    modes = sorted(by_mode)
    while len(picked) < n and any(by_mode.values()):
        for m in modes:
            if by_mode[m] and len(picked) < n:
                c = by_mode[m].pop(0)
                picked.append(c)
                seen.add(c["cid"])
    return picked[:max(n, len([c for c in picked if c["cid"] in dual_cids]))]


def build(picked, payload_dir):
    data = []
    for c in picked:
        p = json.loads((payload_dir / f"case_{c['cid']}.json").read_text())
        data.append({"cid": c["cid"], "mode": (c.get("_modes") or ["?"])[0],
                     "domain": p["document_domain"], "claim": p["claim"],
                     "claim_date": p["claim_date"], "doc_date": p["document_date"],
                     "context": p.get("claim_context") or "",
                     "sents": p["sentences"]})
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    flags = json.dumps(FLAGS)
    reasons = json.dumps(REASONS)
    return """<!doctype html><meta charset="utf-8"><title>READ gold v2</title>
<style>
body{font:15px/1.5 -apple-system,sans-serif;max-width:920px;margin:1.2rem auto;padding:0 1rem;color:#111}
#hdr{position:sticky;top:0;background:#fff;border-bottom:2px solid #111;padding:.5rem 0;z-index:9}
.meta{font-size:12px;color:#666}.dates{font-weight:600;color:#333}
#claim{font-size:16px;font-weight:600;margin:.3rem 0}
.s{padding:.15rem .5rem;border-left:3px solid #ddd;cursor:pointer;margin:.12rem 0}
.s.on{background:#e8f4e8;border-left-color:#171}
.s .n{color:#999;font-size:12px;margin-right:.5em}
button{font:14px -apple-system,sans-serif;padding:.35rem .7rem;margin:.15rem;cursor:pointer;
border:1px solid #888;background:#fff;border-radius:4px}
button.sel{background:#111;color:#fff}
#bar{display:flex;flex-wrap:wrap;align-items:center;gap:.3rem}
#notes{width:100%;font:14px -apple-system,sans-serif;margin-top:.3rem}
#prog{float:right;font-size:13px;color:#333}
kbd{background:#eee;border-radius:3px;padding:0 .3em;font-size:11px}
</style>
<div id="hdr">
 <div id="prog"></div>
 <div class="meta" id="meta"></div>
 <div id="claim"></div>
 <div class="meta" id="ctx"></div>
 <div id="bar"></div>
 <div id="reasonrow" style="display:none"></div>
 <input id="notes" placeholder="notes (only if borderline)">
</div>
<div id="doc"></div>
<div style="margin:2rem 0">
 <button onclick="nav(-1)">&#8592; prev</button>
 <button onclick="nav(1)">next &#8594;</button>
 <button onclick="exportJSON()" style="float:right">Export JSON</button>
</div>
<script>
const CASES=__DATA__, FLAGS=__FLAGS__, REASONS=__REASONS__;
const KEY="read_gold_v2";
let st=JSON.parse(localStorage.getItem(KEY)||"{}"); let i=st._i||0;
function cur(){return CASES[i]}
function lab(){const c=cur(); st[c.cid]=st[c.cid]||{direction:null,evidence:[],reason:"",notes:""}; return st[c.cid]}
function save(){st._i=i; localStorage.setItem(KEY,JSON.stringify(st))}
function render(){
 const c=cur(), l=lab();
 document.getElementById("prog").textContent=(i+1)+"/"+CASES.length+"  labelled "+
   CASES.filter(x=>st[x.cid]&&st[x.cid].direction).length;
 document.getElementById("meta").innerHTML=c.domain+" &middot; mode "+c.mode+
   ' &middot; <span class="dates">claim date: '+c.claim_date+" &middot; doc date: "+c.doc_date+"</span>";
 document.getElementById("claim").textContent=c.claim;
 document.getElementById("ctx").textContent=c.context?("context: "+c.context):"";
 const bar=document.getElementById("bar"); bar.innerHTML="";
 FLAGS.forEach(([f,desc],k)=>{const b=document.createElement("button");
   b.innerHTML="<kbd>"+f+"</kbd> "+desc; if(l.direction===f)b.classList.add("sel");
   b.onclick=()=>{l.direction=f; if(f!=="I")l.reason=""; save(); render()}; bar.appendChild(b)});
 const rr=document.getElementById("reasonrow"); rr.style.display=l.direction==="I"?"block":"none";
 rr.innerHTML=""; if(l.direction==="I") REASONS.forEach(r=>{const b=document.createElement("button");
   b.textContent=r; if(l.reason===r)b.classList.add("sel");
   b.onclick=()=>{l.reason=r; save(); render()}; rr.appendChild(b)});
 document.getElementById("notes").value=l.notes||"";
 const doc=document.getElementById("doc"); doc.innerHTML="";
 c.sents.forEach(([id,txt])=>{const d=document.createElement("div");
   d.className="s"+(l.evidence.includes(id)?" on":"");
   d.innerHTML='<span class="n">['+id+"]</span>"+txt.replace(/&/g,"&amp;").replace(/</g,"&lt;");
   d.onclick=()=>{const e=l.evidence, j=e.indexOf(id); j>=0?e.splice(j,1):e.push(id);
     e.sort((a,b)=>a-b); save(); render()}; doc.appendChild(d)});
}
document.getElementById("notes").addEventListener("input",e=>{lab().notes=e.target.value; save()});
function nav(d){i=Math.max(0,Math.min(CASES.length-1,i+d)); save(); render(); scrollTo(0,0)}
document.addEventListener("keydown",e=>{
 if(e.target.tagName==="INPUT")return;
 const f=FLAGS.map(x=>x[0]); const k=e.key.toUpperCase();
 if(f.includes(k)){lab().direction=k; if(k!=="I")lab().reason=""; save(); render()}
 if(e.key==="ArrowRight")nav(1); if(e.key==="ArrowLeft")nav(-1);
});
function exportJSON(){
 const out=CASES.map(c=>({cid:c.cid, ...(st[c.cid]||{direction:null,evidence:[],reason:"",notes:""})}));
 const a=document.createElement("a");
 a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));
 a.download="read_gold_v2.json"; a.click();
}
render();
</script>""".replace("__DATA__", payload).replace("__FLAGS__", flags).replace("__REASONS__", reasons)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload-dir", required=True, type=Path)
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--dual", type=Path, help="partial silver labels; their cids are drawn first")
    ap.add_argument("-o", "--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    meta = json.load(args.cases.open())["cases"]
    dual = set()
    if args.dual and args.dual.exists():
        dual = {r["cid"] for r in json.load(args.dual.open())}
    picked = draw(meta, args.payload_dir, dual, args.n)
    args.out.write_text(build(picked, args.payload_dir))
    md = collections.Counter((c.get("_modes") or ["?"])[0] for c in picked)
    print(f"wrote {args.out}: {len(picked)} cases "
          f"({len([c for c in picked if c['cid'] in dual])} dual-labelled), modes {dict(md)}")


if __name__ == "__main__":
    main()
