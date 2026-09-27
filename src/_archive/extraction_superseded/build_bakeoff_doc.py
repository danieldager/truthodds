"""Build a 4-modality bake-off review: extract x normalize over {Qwen, DeepSeek}. Tab 1 = summary
table (claims / type split / checkworthy / latency / projected cost per modality); Tab 2 = per
tweet, the four final claim sets side by side (sorted hardest-first), with the pre-normalize
extract shown on toggle. Self-contained, theme-aware HTML.

Joins six claims-review payloads on tweet url: two extracts (--qx / --dx) and four normalized
combos (--qq/--qd/--dq/--dd). Run stats (latency/cost) are baked in via STATS below.

  cd src && uv run python eval/scripts/claim_sourcing/build_bakeoff_doc.py \
      --qx qX.html --dx dX.html --qq QQ.html --qd QD.html --dq DQ.html --dd DD.html -o bakeoff.html
"""
import argparse, json, re, html
from pathlib import Path

# measured on the 100-tweet problematic slice, concurrency 12 (extract + normalize wall-clock, s;
# projected $ = full-corpus two-pass = extract + normalize, from each stage's DeepInfra estimate)
STATS = {
    "QQ": {"label": "Qwen → Qwen", "ex": "q", "nm": "q", "lat": 75 + 62, "cost": 1.30 + 1.52},
    "QD": {"label": "Qwen → DeepSeek", "ex": "q", "nm": "d", "lat": 75 + 68, "cost": 1.30 + 1.81},
    "DQ": {"label": "DeepSeek → Qwen", "ex": "d", "nm": "q", "lat": 66 + 57, "cost": 1.16 + 1.49},
    "DD": {"label": "DeepSeek → DeepSeek", "ex": "d", "nm": "d", "lat": 66 + 66, "cost": 1.16 + 1.77},
}
ORDER = ["QQ", "QD", "DQ", "DD"]


def load(path):
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', Path(path).read_text(), re.S)
    return {p["url"]: p for p in json.loads(m.group(1))["posts"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("qx", "dx", "qq", "qd", "dq", "dd"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    ex = {"q": load(args.qx), "d": load(args.dx)}
    nm = {"QQ": load(args.qq), "QD": load(args.qd), "DQ": load(args.dq), "DD": load(args.dd)}

    # tweet order = Qwen-extract input order (already sorted hardest-first in the parquet)
    urls = list(ex["q"].keys())
    posts = []
    for u in urls:
        base = ex["q"][u]
        cells = {}
        for k in ORDER:
            src = STATS[k]["ex"]
            cells[k] = {"pre": ex[src][u].get("claims") or [], "pre_nr": ex[src][u].get("nr"),
                        "fin": nm[k][u].get("claims") or [], "fin_nr": nm[k][u].get("nr")}
        posts.append({"handle": base.get("handle"), "cell": base.get("cell"), "domain": base.get("domain"),
                      "date": base.get("date"), "url": u, "text": base.get("text") or "",
                      "cells": cells})

    def agg(k):
        fin = [c for u in urls for c in (nm[k][u].get("claims") or [])]
        withc = sum(1 for u in urls if nm[k][u].get("claims"))
        cw = sum(1 for c in fin if c["cw"])
        at = sum(1 for c in fin if c["t"] == "attribution")
        return dict(n=len(fin), assertion=len(fin) - at, attribution=at, cw=cw,
                    withc=withc, per=len(fin) / max(1, len(urls)))

    esc = html.escape

    def srow(k):
        a = agg(k); s = STATS[k]
        return (f"<tr><td class='m'><span class='mx {s['ex']}'>{s['ex'].upper()}</span>"
                f"<span class='arr'>→</span><span class='mx {s['nm']}'>{s['nm'].upper()}</span>"
                f"&nbsp;{esc(s['label'])}</td>"
                f"<td class='n'>{a['n']}</td><td class='n'>{a['per']:.1f}</td>"
                f"<td class='n'>{a['assertion']}<span class='sub'> / {a['attribution']}</span></td>"
                f"<td class='n strong'>{a['cw']}<span class='sub'> ({100*a['cw']/max(1,a['n']):.0f}%)</span></td>"
                f"<td class='n'>{a['withc']}/100</td><td class='n'>{s['lat']}s</td>"
                f"<td class='n'>${s['cost']:.2f}</td></tr>")

    shead = ("<tr><th>modality (extract → normalize)</th><th class='n'>claims</th><th class='n'>/post</th>"
             "<th class='n'>assert<span class='sub'>/attrib</span></th><th class='n'>checkworthy</th>"
             "<th class='n'>posts w/claim</th><th class='n'>latency<span class='sub'> (100)</span></th>"
             "<th class='n'>corpus $</th></tr>")

    intro = ("Four extract&times;normalize modalities over Qwen3-235B-Instruct and DeepSeek-V4-Flash "
             "(both non-thinking, same prompts), on the <b>100 most problematic tweets</b> of the 500-tweet "
             "slice &mdash; selected for the churn they caused the two-pass (claim over-splitting, "
             "attribution/type flips, quote density). Latency is wall-clock at concurrency 12; corpus $ "
             "is the projected full two-pass on ~20,800 tweets. Tab 2 shows every tweet's four final claim "
             "sets side by side, hardest first; toggle the pre-normalize extract to see what each pass fixed.")

    payload = json.dumps({"posts": posts, "order": ORDER,
                          "labels": {k: STATS[k]["label"] for k in ORDER},
                          "mx": {k: {"ex": STATS[k]["ex"], "nm": STATS[k]["nm"]} for k in ORDER}},
                         ensure_ascii=False).replace("</", "<\\/")
    doc = (TEMPLATE.replace("__INTRO__", intro).replace("__SHEAD__", shead)
           .replace("__SROWS__", "".join(srow(k) for k in ORDER)).replace("__PAYLOAD__", payload))
    Path(args.output).write_text(doc)
    print(f"wrote {args.output} — {len(posts)} tweets x 4 modalities")
    for k in ORDER:
        a = agg(k)
        print(f"  {STATS[k]['label']:<24} {a['n']:>3} claims  {a['assertion']}/{a['attribution']} a/at  "
              f"cw {a['cw']} ({100*a['cw']/max(1,a['n']):.0f}%)")


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Extractor bake-off</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1c1a;--muted:#77776f;--card:#fff;--line:#e7e6e2;--accent:#4457c7;
 --L:#2f6feb;--R:#c0392b;--assert:#3f8a5c;--attrib:#7a56c0;--grp:#f2f1ee;--soft:#f6f5f2;
 --q:#4457c7;--d:#0f8f8f}
@media(prefers-color-scheme:dark){:root{--bg:#161719;--fg:#e9e9e6;--muted:#95958e;--card:#1f2123;
 --line:#31333799;--accent:#8894ec;--L:#5a8def;--R:#e0685b;--assert:#5aa877;--attrib:#a389df;
 --grp:#232527;--soft:#1c1e20;--q:#8894ec;--d:#2fc4c4}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:20px 18px 90px}
h1{font-size:19px;margin:0 0 12px;font-weight:650}
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin-bottom:18px}
.tab{padding:8px 16px;cursor:pointer;color:var(--muted);font-weight:600;font-size:14px;border-bottom:2px solid transparent;margin-bottom:-1px}
.tab.on{color:var(--fg);border-bottom-color:var(--accent)}
.pane{display:none}.pane.on{display:block}
.intro{font-size:14px;line-height:1.6;color:var(--fg);margin:0 0 16px}.intro b{font-weight:650}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:8px 11px;border-bottom:1px solid var(--line);vertical-align:baseline}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:0}
.m{font-weight:600}.sub{color:var(--muted);font-weight:400;font-size:.88em}.strong{font-weight:700}
.mx{display:inline-block;font-size:10px;font-weight:700;color:#fff;border-radius:4px;padding:1px 5px;letter-spacing:.02em}
.mx.q{background:var(--q)}.mx.d{background:var(--d)}
.arr{color:var(--muted);margin:0 4px}
/* tab 2 */
.filters{position:sticky;top:0;background:var(--bg);padding:6px 0 10px;z-index:5;border-bottom:1px solid var(--line);margin-bottom:14px}
.filters input[type=search]{font:inherit;font-size:14px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);width:100%;margin-bottom:9px}
.frow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:5px 0}
.flabel{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.fgroup{display:flex;flex-wrap:wrap;gap:4px}
.tog{font:inherit;font-size:12px;padding:3px 10px;border:1px solid var(--line);border-radius:14px;background:var(--card);color:var(--muted);cursor:pointer}
.tog:hover{color:var(--fg)}.tog.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tog.src.L.on{background:var(--L);border-color:var(--L)}.tog.src.R.on{background:var(--R);border-color:var(--R)}
.chk{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:13px}
.count{color:var(--muted);font-size:12.5px;margin:0 0 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px}
.chead{display:flex;flex-wrap:wrap;gap:9px;align-items:center;font-size:12px;color:var(--muted);margin-bottom:7px}
.chead .h{font-weight:600;color:var(--fg)}.chead .h .L{color:var(--L)}.chead .h .R{color:var(--R)}
.chead a{color:var(--accent);text-decoration:none}
.post{font-size:14px;white-space:pre-wrap;margin:0 0 11px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}
@media(max-width:980px){.grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:560px){.grid{grid-template-columns:1fr}}
.col{border:1px solid var(--line);border-radius:8px;padding:9px 10px;background:var(--soft)}
.col h4{margin:0 0 8px;font-size:11px;font-weight:700;display:flex;gap:5px;align-items:center;flex-wrap:wrap}
.col h4 .ct{color:var(--muted);font-weight:600;margin-left:auto}
.col h4 .ct b{color:var(--fg)}
.pre{border-top:1px dashed var(--line);margin-top:8px;padding-top:6px;display:none}
.showpre .pre{display:block}
.pre .pl{font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-bottom:4px}
.claim{display:flex;gap:7px;padding:4px 0}
.claim+.claim{border-top:1px solid var(--line)}
.pre .claim+.claim{border-top:1px solid var(--soft)}
.cdot{flex:none;width:6px;height:6px;border-radius:50%;margin-top:6px}
.cdot.assertion{background:var(--assert)}.cdot.attribution{background:var(--attrib)}
.cbody{flex:1;min-width:0}.ctext{font-size:12.5px;line-height:1.4}
.ctags{margin-top:1px}.tag{font-size:9.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.tag.warn{color:var(--R)}
.pre .ctext{color:var(--muted);font-size:12px}
.noclaim{color:var(--muted);font-size:12px;font-style:italic}
.pill{font-style:normal;font-size:10px;font-weight:600;color:var(--accent);border:1px solid var(--accent);border-radius:5px;padding:0 5px;margin-left:4px}
.leg{display:flex;gap:16px;flex-wrap:wrap;font-size:11.5px;color:var(--muted);margin:0 0 14px}
.leg span{display:inline-flex;align-items:center;gap:5px}.sw{width:10px;height:10px;border-radius:3px;display:inline-block}
</style></head><body><div class="wrap">
<h1>Extractor bake-off &mdash; Qwen &times; DeepSeek, extract &times; normalize</h1>
<div class="tabs"><div class="tab on" data-pane="p1">Summary</div><div class="tab" data-pane="p2">Tweets</div></div>

<div class="pane on" id="p1">
 <p class="intro">__INTRO__</p>
 <table>__SHEAD____SROWS__</table>
</div>

<div class="pane" id="p2">
 <div class="filters">
  <input type="search" id="q" placeholder="Search claim or post…">
  <div class="frow"><span class="flabel">sources</span><div class="fgroup" data-group="src" id="fsrc"></div></div>
  <div class="frow"><label class="chk"><input type="checkbox" id="fpre"> show pre-normalize extract</label></div>
 </div>
 <div class="leg">
  <span><span class="sw" style="background:var(--assert)"></span>assertion</span>
  <span><span class="sw" style="background:var(--attrib)"></span>attribution</span>
  <span><span class="mx q">Q</span> Qwen3-235B-Instruct</span>
  <span><span class="mx d">D</span> DeepSeek-V4-Flash</span>
  <span>header shows <b>extract&nbsp;n&nbsp;&rarr;&nbsp;final&nbsp;n</b></span>
 </div>
 <p class="count" id="count"></p><div id="list"></div>
</div>

<script type="application/json" id="data">__PAYLOAD__</script>
<script>
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{
 document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
 document.querySelectorAll('.pane').forEach(x=>x.classList.remove('on'));
 t.classList.add('on');document.getElementById(t.dataset.pane).classList.add('on');}));
const D=JSON.parse(document.getElementById('data').textContent);
const esc=s=>(s==null?'':String(s)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cls=c=>c&&c.toLowerCase().startsWith('l')?'L':'R';
const srcs=[...new Set(D.posts.map(p=>p.handle))].sort((a,b)=>a.localeCompare(b));
const cellOf={};D.posts.forEach(p=>cellOf[p.handle]=p.cell);
const srcWrap=document.getElementById('fsrc');
srcs.forEach(h=>{const b=document.createElement('button');b.className='tog src '+cls(cellOf[h]);b.dataset.v=h;b.textContent='@'+h;srcWrap.appendChild(b);});
const G={src:new Set()};
document.querySelectorAll('.tog').forEach(b=>b.addEventListener('click',()=>{const v=b.dataset.v;if(G.src.has(v)){G.src.delete(v);b.classList.remove('on');}else{G.src.add(v);b.classList.add('on');}render();}));
const q=document.getElementById('q'),list=document.getElementById('list'),count=document.getElementById('count');
const fpre=document.getElementById('fpre');
q.addEventListener('input',render);
fpre.addEventListener('change',()=>{document.getElementById('list').classList.toggle('showpre',fpre.checked);});
function claimsHtml(cs,nr,small){
 if(cs&&cs.length){
  return cs.map(c=>{
   const tags=[`<span class="tag">${esc(c.t)}</span>`];
   if(!c.cw)tags.push(' <span class="tag warn">not cw</span>');
   return `<div class="claim"><span class="cdot ${esc(c.t)}"></span><div class="cbody"><div class="ctext">${esc(c.c)}</div><div class="ctags">${tags.join('')}</div></div></div>`;
  }).join('');
 }
 return `<div class="noclaim">no claim<span class="pill">${esc(nr||'—')}</span></div>`;
}
function colHtml(k,cell){
 const mx=D.mx[k];
 const preN=cell.pre?cell.pre.length:0, finN=cell.fin?cell.fin.length:0;
 return `<div class="col"><h4><span class="mx ${mx.ex}">${mx.ex.toUpperCase()}</span><span class="arr">→</span><span class="mx ${mx.nm}">${mx.nm.toUpperCase()}</span>`
  +`<span class="ct">${preN}<span class="arr">→</span><b>${finN}</b></span></h4>`
  +claimsHtml(cell.fin,cell.fin_nr)
  +`<div class="pre"><div class="pl">extract (${mx.ex.toUpperCase()}): ${preN}</div>${claimsHtml(cell.pre,cell.pre_nr)}</div></div>`;
}
function render(){
 const t=q.value.toLowerCase();let shown=0;
 list.innerHTML=D.posts.filter(p=>{
  if(G.src.size&&!G.src.has(p.handle))return false;
  if(t){const hay=(p.text+' '+D.order.flatMap(k=>(p.cells[k].fin||[]).map(c=>c.c)).join(' ')).toLowerCase();if(!hay.includes(t))return false;}
  return true;
 }).map(p=>{shown++;
  const cols=D.order.map(k=>colHtml(k,p.cells[k])).join('');
  return `<div class="card"><div class="chead"><span class="h"><span class="${cls(p.cell)}">@${esc(p.handle)}</span></span><span>${esc(p.date)}</span><span>${esc(p.domain)}</span><a href="${esc(p.url)}" target="_blank" rel="noopener">tweet ↗</a></div><div class="post">${esc(p.text)}</div><div class="grid">${cols}</div></div>`;
 }).join('');
 count.textContent=`${shown} of ${D.posts.length} tweets`;}
render();
</script></div></body></html>"""


if __name__ == "__main__":
    main()
