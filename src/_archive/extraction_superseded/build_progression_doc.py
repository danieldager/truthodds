"""Build a two-pass progression review: Tab 1 = per-source summary of what the normalizer
changed (claims / checkworthy, pass1 -> pass2); Tab 2 = per-tweet, the POST with pass-1
(extract) and pass-2 (normalize) claims side by side so you can eyeball every correction.

Joins two claims-review HTML payloads on the tweet url: --pass1 (extract_tweet_claims.py --html)
and --pass2 (normalize_claims.py output). Self-contained, theme-aware HTML.

  cd src && uv run python eval/scripts/claim_sourcing/build_progression_doc.py \
      --pass1 <r5 extract>.html --pass2 <r5norm>.html -o progression.html
"""
import argparse, json, re, html
from pathlib import Path


def load_payload(path):
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', Path(path).read_text(), re.S)
    return json.loads(m.group(1))["posts"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass1", required=True, help="first-pass review HTML (extract)")
    ap.add_argument("--pass2", required=True, help="second-pass review HTML (normalize)")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    p1 = {p["url"]: p for p in load_payload(args.pass1)}
    p2 = {p["url"]: p for p in load_payload(args.pass2)}
    urls = [u for u in p1 if u in p2]

    posts = []
    for u in urls:
        a, b = p1[u], p2[u]
        ca, cb = a.get("claims") or [], b.get("claims") or []
        # changed if claim count differs, or any (text/type/cw) differs as a set
        sa = {(c["c"], c["t"], c["cw"]) for c in ca}
        sb = {(c["c"], c["t"], c["cw"]) for c in cb}
        posts.append({
            "handle": b.get("handle"), "cell": b.get("cell"), "domain": b.get("domain"),
            "url": u, "date": b.get("date"), "text": b.get("text"), "images": b.get("images") or [],
            "c1": ca, "nr1": a.get("nr"), "c2": cb, "nr2": b.get("nr"),
            "changed": sa != sb,
        })

    # per-source aggregates
    agg = {}
    for p in posts:
        hk = (p["handle"] or "").lower()
        s = agg.setdefault(hk, dict(handle=p["handle"], cell=p["cell"], posts=0,
                                    c1=0, c2=0, cw1=0, cw2=0, changed=0))
        s["posts"] += 1
        s["c1"] += len(p["c1"]); s["c2"] += len(p["c2"])
        s["cw1"] += sum(1 for c in p["c1"] if c["cw"]); s["cw2"] += sum(1 for c in p["c2"] if c["cw"])
        s["changed"] += 1 if p["changed"] else 0

    tot = dict(posts=len(posts), changed=sum(1 for p in posts if p["changed"]),
               c1=sum(len(p["c1"]) for p in posts), c2=sum(len(p["c2"]) for p in posts),
               cw1=sum(1 for p in posts for c in p["c1"] if c["cw"]),
               cw2=sum(1 for p in posts for c in p["c2"] if c["cw"]))
    esc = html.escape

    def delta(x, y):
        d = y - x
        col = "up" if d > 0 else "dn" if d < 0 else "flat"
        s = f"+{d}" if d > 0 else str(d) if d < 0 else "0"
        return f"{x}<span class='arr'>→</span>{y}<span class='d {col}'>{s}</span>"

    def row(s):
        lean = (s["cell"] or "R")[0].upper()
        return (f"<tr><td class='src'><span class='dot {lean}'></span>@{esc(s['handle'] or '')}</td>"
                f"<td class='n'>{s['posts']}</td>"
                f"<td class='n'>{delta(s['c1'], s['c2'])}</td>"
                f"<td class='n'>{delta(s['cw1'], s['cw2'])}</td>"
                f"<td class='n strong'>{s['changed']}</td></tr>")

    def section(lean):
        rs = sorted([s for s in agg.values() if (s["cell"] or "R")[0].upper() == lean[0]],
                    key=lambda s: -s["changed"])
        return f"<tr class='grp'><td colspan='5'>{lean}</td></tr>" + "".join(row(s) for s in rs)

    head = ("<tr><th>source</th><th class='n'>posts</th><th class='n'>claims (p1→p2)</th>"
            "<th class='n'>checkworthy (p1→p2)</th><th class='n'>tweets changed</th></tr>")

    intro = (f"Two-pass extraction on the fixed <b>{tot['posts']}-tweet</b> sample: a fast first pass "
             f"(<b>extract</b>) then a fast second pass (<b>normalize</b>) that merges over-splits, fixes "
             f"attribution vs assertion, strips pure opinion, and re-tags checkworthy. "
             f"The normalizer touched <b>{tot['changed']}/{tot['posts']}</b> tweets "
             f"({100*tot['changed']/tot['posts']:.0f}%). Claims <b>{tot['c1']} → {tot['c2']}</b> "
             f"({tot['c2']-tot['c1']:+d}); checkworthy <b>{tot['cw1']} → {tot['cw2']}</b> "
             f"({tot['cw2']-tot['cw1']:+d}, {100*tot['cw2']/max(1,tot['c2']):.0f}% of the final set). "
             f"Tab 2 shows every tweet with both passes side by side.")

    payload = json.dumps({"posts": posts}, ensure_ascii=False).replace("</", "<\\/")
    doc = (TEMPLATE.replace("__INTRO__", intro).replace("__HEAD__", head)
           .replace("__ROWS__", section("Left") + section("Right")).replace("__PAYLOAD__", payload))
    Path(args.output).write_text(doc)
    print(f"wrote {args.output} — {tot['posts']} posts, {tot['changed']} changed, "
          f"claims {tot['c1']}->{tot['c2']}, checkworthy {tot['cw1']}->{tot['cw2']}")


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Two-pass progression</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1c1a;--muted:#77776f;--card:#fff;--line:#e7e6e2;--accent:#4457c7;
 --L:#2f6feb;--R:#c0392b;--assert:#3f8a5c;--attrib:#7a56c0;--grp:#f2f1ee;--soft:#f6f5f2;
 --add:#3f8a5c;--del:#c0392b;--p1:#8a6d1f}
@media(prefers-color-scheme:dark){:root{--bg:#161719;--fg:#e9e9e6;--muted:#95958e;--card:#1f2123;
 --line:#31333799;--accent:#8894ec;--L:#5a8def;--R:#e0685b;--assert:#5aa877;--attrib:#a389df;
 --grp:#232527;--soft:#1c1e20;--add:#5aa877;--del:#e0685b;--p1:#c9a84a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:20px 18px 90px}
h1{font-size:19px;margin:0 0 12px;font-weight:650}
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin-bottom:18px}
.tab{padding:8px 16px;cursor:pointer;color:var(--muted);font-weight:600;font-size:14px;border-bottom:2px solid transparent;margin-bottom:-1px}
.tab.on{color:var(--fg);border-bottom-color:var(--accent)}
.pane{display:none}.pane.on{display:block}
.intro{font-size:14px;line-height:1.6;color:var(--fg);margin:0 0 16px}.intro b{font-weight:650}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:7px 11px;border-bottom:1px solid var(--line);vertical-align:baseline}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.grp td{background:var(--grp);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
tr:last-child td{border-bottom:0}
.src{font-weight:600}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;vertical-align:middle}
.dot.L{background:var(--L)}.dot.R{background:var(--R)}
.strong{font-weight:700}
.arr{color:var(--muted);margin:0 5px;font-weight:400}
.d{font-size:.82em;margin-left:6px;font-weight:600}.d.up{color:var(--add)}.d.dn{color:var(--del)}.d.flat{color:var(--muted)}
/* tab 2 */
.filters{position:sticky;top:0;background:var(--bg);padding:6px 0 10px;z-index:5;border-bottom:1px solid var(--line);margin-bottom:14px}
.filters input[type=search]{font:inherit;font-size:14px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);width:100%;margin-bottom:9px}
.frow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:5px 0}
.funit{display:flex;align-items:center;gap:7px}
.flabel{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.fgroup{display:flex;flex-wrap:wrap;gap:4px}
.tog{font:inherit;font-size:12px;padding:3px 10px;border:1px solid var(--line);border-radius:14px;background:var(--card);color:var(--muted);cursor:pointer}
.tog:hover{color:var(--fg)}
.tog.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tog.src.L.on{background:var(--L);border-color:var(--L)}.tog.src.R.on{background:var(--R);border-color:var(--R)}
.chk{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:13px}
.count{color:var(--muted);font-size:12.5px;margin:0 0 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:9px}
.card.chg{box-shadow:inset 3px 0 0 var(--accent)}
.chead{display:flex;flex-wrap:wrap;gap:9px;align-items:center;font-size:12px;color:var(--muted);margin-bottom:7px}
.chead .h{font-weight:600;color:var(--fg)}.chead .h .L{color:var(--L)}.chead .h .R{color:var(--R)}
.chead a{color:var(--accent);text-decoration:none}
.badge{font-size:10.5px;font-weight:600;border-radius:5px;padding:1px 6px;text-transform:uppercase;letter-spacing:.03em}
.badge.chg{color:var(--accent);border:1px solid var(--accent)}
.badge.same{color:var(--muted);border:1px solid var(--line)}
.post{font-size:14px;white-space:pre-wrap;margin:0 0 10px}
.imgs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}.imgs img{max-height:120px;max-width:100%;border-radius:6px;border:1px solid var(--line)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media(max-width:640px){.cols{grid-template-columns:1fr}}
.pcol{border:1px solid var(--line);border-radius:8px;padding:9px 11px;background:var(--soft)}
.pcol h4{margin:0 0 7px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;font-weight:700;display:flex;gap:7px;align-items:center}
.pcol.p1 h4{color:var(--p1)}.pcol.p2 h4{color:var(--accent)}
.pcol h4 .ct{color:var(--muted);font-weight:600}
.claim{display:flex;gap:8px;padding:5px 0}
.claim+.claim{border-top:1px solid var(--line)}
.cdot{flex:none;width:7px;height:7px;border-radius:50%;margin-top:7px}
.cdot.assertion{background:var(--assert)}.cdot.attribution{background:var(--attrib)}
.cbody{flex:1;min-width:0}.ctext{font-size:13px;line-height:1.45}
.ctags{margin-top:2px;display:flex;gap:7px;flex-wrap:wrap}
.tag{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.tag.warn{color:var(--del)}
.claim.new{background:color-mix(in srgb,var(--add) 12%,transparent);border-radius:5px;margin:0 -5px;padding-left:5px;padding-right:5px}
.claim.gone{opacity:.5;text-decoration:line-through solid var(--del)}
.noclaim{color:var(--muted);font-size:12.5px;font-style:italic;padding:2px 0}
.pill{font-style:normal;font-size:10.5px;font-weight:600;color:var(--accent);border:1px solid var(--accent);border-radius:5px;padding:1px 5px;margin-left:5px}
.leg{display:flex;gap:16px;flex-wrap:wrap;font-size:11.5px;color:var(--muted);margin:0 0 14px}
.leg span{display:inline-flex;align-items:center;gap:5px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.anno{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:12px}
.anno input[type=text]{font:inherit;font-size:13px;padding:6px 10px;border:1px solid var(--line);border-radius:7px;background:var(--card);color:var(--fg);min-width:190px}
.chk{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:13px}
.ct{color:var(--muted);font-size:12.5px}
.cmt{width:100%;margin-top:10px;font:inherit;font-size:13px;padding:7px 9px;border:1px solid var(--line);border-radius:7px;background:var(--soft);color:var(--fg);resize:vertical;min-height:32px;line-height:1.45}
.cmt::placeholder{color:var(--muted)}
.cmt:focus{border-color:var(--accent);background:var(--card);outline:none}
.card.commented{border-color:var(--accent);box-shadow:inset 3px 0 0 var(--accent)}
.fab{position:fixed;right:22px;bottom:22px;z-index:20;background:var(--accent);color:#fff;border:none;border-radius:22px;padding:11px 18px;font:inherit;font-weight:600;font-size:14px;cursor:pointer;box-shadow:0 3px 14px rgba(0,0,0,.28)}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:30;display:none;align-items:center;justify-content:center;padding:20px}
.modal.on{display:flex}
.sheet{background:var(--card);border:1px solid var(--line);border-radius:12px;max-width:640px;width:100%;max-height:82vh;display:flex;flex-direction:column;padding:16px}
.sheet h3{margin:0 0 9px;font-size:15px;font-weight:650}
.sheet textarea{flex:1;min-height:320px;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--soft);color:var(--fg);resize:none}
.sheet .row{display:flex;gap:8px;justify-content:flex-end;margin-top:11px}
.btn{font:inherit;font-size:13px;font-weight:600;padding:8px 15px;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
.btn.ghost{background:transparent;color:var(--accent)}
</style></head><body><div class="wrap">
<h1>Two-pass claim extraction &mdash; progression</h1>
<div class="tabs"><div class="tab on" data-pane="p1">Summary</div><div class="tab" data-pane="p2">Tweets</div></div>

<div class="pane on" id="p1">
 <p class="intro">__INTRO__</p>
 <table>__HEAD____ROWS__</table>
</div>

<div class="pane" id="p2">
 <div class="anno">
  <input type="text" id="anno" placeholder="Your name (labels the export)">
  <label class="chk"><input type="checkbox" id="fcmt"> commented only</label>
  <span class="ct" id="cmtct"></span>
 </div>
 <div class="filters">
  <input type="search" id="q" placeholder="Search claim or post…">
  <div class="frow"><span class="flabel">sources</span><div class="fgroup" data-group="src" id="fsrc"></div></div>
  <div class="frow"><label class="chk"><input type="checkbox" id="fchg"> changed only</label></div>
 </div>
 <div class="leg">
  <span><span class="sw" style="background:color-mix(in srgb,var(--add) 40%,transparent)"></span>added by normalize</span>
  <span><span class="sw" style="background:var(--del);opacity:.5"></span>removed by normalize</span>
  <span><span class="sw" style="background:var(--assert)"></span>assertion</span>
  <span><span class="sw" style="background:var(--attrib)"></span>attribution</span>
 </div>
 <p class="count" id="count"></p><div id="list"></div>
 <button class="fab" id="fab">Export comments</button>
 <div class="modal" id="modal"><div class="sheet"><h3 id="mtitle">Comments</h3><textarea id="mtext" readonly placeholder="No comments yet. Type in the box under any tweet."></textarea><div class="row"><button class="btn ghost" id="mclose">Close</button><button class="btn" id="mdl">Download JSON</button></div></div></div>
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
const fchg=document.getElementById('fchg');
q.addEventListener('input',render);fchg.addEventListener('change',render);
const keyOf=c=>c.c+''+c.t+''+(c.cw?1:0);
// --- manual annotation: per-tweet comments, persisted in the browser ---
const KEY='ddrev::';
let comments=JSON.parse(localStorage.getItem(KEY+'comments')||'{}');
const anno=document.getElementById('anno');anno.value=localStorage.getItem(KEY+'anno')||'';
anno.addEventListener('input',()=>localStorage.setItem(KEY+'anno',anno.value));
const fcmt=document.getElementById('fcmt');fcmt.addEventListener('change',render);
const cmtct=document.getElementById('cmtct'),fab=document.getElementById('fab');
const nCmt=()=>Object.values(comments).filter(v=>v&&v.trim()).length;
function updCt(){const n=nCmt();cmtct.textContent=n?n+' commented':'';fab.textContent='Export comments'+(n?' ('+n+')':'');}
list.addEventListener('input',e=>{const el=e.target;if(!el.classList||!el.classList.contains('cmt'))return;const u=el.dataset.url;if(el.value.trim())comments[u]=el.value;else delete comments[u];localStorage.setItem(KEY+'comments',JSON.stringify(comments));const card=el.closest('.card');if(card)card.classList.toggle('commented',!!el.value.trim());updCt();});
function buildExport(){const o={annotator:anno.value||null,exported:new Date().toISOString(),total_posts:D.posts.length,comments:[]};D.posts.forEach(p=>{const c=comments[p.url];if(c&&c.trim())o.comments.push({url:p.url,handle:p.handle,post:p.text,extract:(p.c1||[]).map(x=>({claim:x.c,type:x.t,checkworthy:x.cw})),normalize:(p.c2||[]).map(x=>({claim:x.c,type:x.t,checkworthy:x.cw})),comment:c});});return o;}
const modal=document.getElementById('modal'),mtext=document.getElementById('mtext'),mtitle=document.getElementById('mtitle');
fab.addEventListener('click',()=>{const o=buildExport();mtext.value=JSON.stringify(o,null,2);mtitle.textContent=o.comments.length+' comment'+(o.comments.length===1?'':'s')+(o.annotator?' — '+o.annotator:'');modal.classList.add('on');});
document.getElementById('mclose').addEventListener('click',()=>modal.classList.remove('on'));
modal.addEventListener('click',e=>{if(e.target===modal)modal.classList.remove('on');});
document.getElementById('mdl').addEventListener('click',()=>{const o=buildExport();const b=new Blob([JSON.stringify(o,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='dd-review-'+((o.annotator||'anon').replace(/[^a-z0-9]+/gi,'_'))+'.json';a.click();});
function colHtml(claims,other,nr,label,role){
 const oset=new Set((other||[]).map(keyOf));
 let inner;
 if(claims&&claims.length){
  inner=claims.map(c=>{
   const changed=!oset.has(keyOf(c));
   const mark=changed?(role==='p2'?' new':' gone'):'';
   const tags=[`<span class="tag">${esc(c.t)}</span>`];
   if(!c.cw)tags.push('<span class="tag warn">not checkworthy</span>');
   return `<div class="claim${mark}"><span class="cdot ${esc(c.t)}"></span><div class="cbody"><div class="ctext">${esc(c.c)}</div><div class="ctags">${tags.join('')}</div></div></div>`;
  }).join('');
 }else{
  inner=`<div class="noclaim">no claim<span class="pill">${esc(nr||'—')}</span></div>`;
 }
 return `<div class="pcol ${role}"><h4>${label}<span class="ct">${claims?claims.length:0}</span></h4>${inner}</div>`;
}
function render(){
 const t=q.value.toLowerCase();let shown=0;
 list.innerHTML=D.posts.filter(p=>{
  if(G.src.size&&!G.src.has(p.handle))return false;
  if(fchg.checked&&!p.changed)return false;
  if(fcmt.checked&&!(comments[p.url]&&comments[p.url].trim()))return false;
  if(t){const hay=(p.text+' '+[...(p.c1||[]),...(p.c2||[])].map(c=>c.c).join(' ')).toLowerCase();if(!hay.includes(t))return false;}
  return true;
 }).map(p=>{shown++;
  const imgs=(p.images||[]).map(u=>`<img src="${esc(u)}" loading="lazy">`).join('');
  const badge=p.changed?'<span class="badge chg">changed</span>':'<span class="badge same">unchanged</span>';
  const hasC=!!(comments[p.url]&&comments[p.url].trim());
  return `<div class="card${p.changed?' chg':''}${hasC?' commented':''}"><div class="chead"><span class="h"><span class="${cls(p.cell)}">@${esc(p.handle)}</span></span><span>${esc(p.date)}</span><span>${esc(p.domain)}</span>${badge}<a href="${esc(p.url)}" target="_blank" rel="noopener">tweet ↗</a></div>${imgs?`<div class="imgs">${imgs}</div>`:''}<div class="post">${esc(p.text)}</div><div class="cols">${colHtml(p.c1,p.c2,p.nr1,'Pass 1 · extract','p1')}${colHtml(p.c2,p.c1,p.nr2,'Pass 2 · normalize','p2')}</div><textarea class="cmt" data-url="${esc(p.url)}" placeholder="Comment on this tweet — a bad split, wrong type, wrong checkworthy call, missed/hallucinated claim…">${esc(comments[p.url]||'')}</textarea></div>`;
 }).join('');
 count.textContent=`${shown} of ${D.posts.length} posts`;}
render();updCt();
</script></div></body></html>"""


if __name__ == "__main__":
    main()
