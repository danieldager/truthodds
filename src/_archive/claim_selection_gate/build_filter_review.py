"""Generate the HTML review tool for the DECISIVE borderline FILTER rows (final-label decision).

  uv run python -m eval.scripts.build_filter_review

Decisive borderline = 235B and Gemma give a different recommend (positive/negative/drop) → the label
genuinely hangs on a human call. Writes a self-contained `eval/data/filter_review.html` (images embedded):
one card per row showing the post (text + image) + claim/rating + each model's call + note, with
Positive / Negative / Drop buttons defaulting to the 235B pick. Picks export a compact block
(`F001:positive F002:negative …`) → applied later via the provenance ledger.
Also writes `eval/data/filter_review_items.parquet` (id→key + the 235B default) for the apply step.
"""
from __future__ import annotations

import base64
import glob
import html
from pathlib import Path

import polars as pl

from eval.scripts.build_filter_dev import _recommend
from eval.textnorm import clean_text

OUT_HTML = Path("eval/data/filter_review.html")
OUT_ITEMS = Path("eval/data/filter_review_items.parquet")


def _rec(a, lab: str):
    oj = bool(a.get(f"A_{lab}_obvious_joke"))
    car = (a.get(f"B_{lab}_claim_matches_post") is True) or (a.get(f"B_{lab}_image_supports_claim") is True)
    return _recommend(a.get(f"A_{lab}_readable"), car, a.get(f"A_{lab}_political_implication"), a.get(f"A_{lab}_has_claim"), oj)


def _img_tag(image_paths) -> str:
    tags = []
    for p in (image_paths or [])[:2]:
        if p and Path(p).exists():
            b = base64.b64encode(Path(p).read_bytes()).decode()
            tags.append(f'<img src="data:image/webp;base64,{b}">')
    return "".join(tags)


def main() -> None:
    d = pl.read_parquet("eval/data/audit_results.parquet").filter(pl.col("A_lab_political_implication").is_not_null())
    look = {}
    for f in sorted(glob.glob("eval/data/*_harvest.parquet")):
        for r in pl.read_parquet(f).to_dicts():
            if r.get("review_url"):
                look[r["review_url"]] = {"raw_context": r.get("raw_context"), "image_paths": r.get("image_paths")}

    rows = []
    for a in d.to_dicts():
        (r, rr), (xr, xrr) = _rec(a, "lab"), _rec(a, "x")
        if r == xr:
            continue  # decisive only
        post = look.get(a["key"], {})
        rows.append({**a, "_r": r, "_rr": rr, "_xr": xr, "_xrr": xrr,
                     "_post": post.get("raw_context") or "", "_imgs": post.get("image_paths")})
    # order: the 47 negative→positive (Gemma-wants-keep) first, then the rest
    rows.sort(key=lambda a: (not (a["_r"] == "negative" and a["_xr"] == "positive"), a["source"]))

    cards, items = [], []
    for i, a in enumerate(rows):
        fid = f"F{i:03d}"
        items.append({"id": fid, "key": a["key"], "source": a["source"],
                      "claim_text": a.get("claim_text"), "default_235": a["_r"]})
        esc = lambda s: html.escape(clean_text(str(s or "")))
        cards.append(f"""
<div class="card" id="{fid}" data-default="{a['_r']}">
  <div class="hd"><b>{fid}</b> · {esc(a['source'])} · <span class="topic">{esc(a.get('A_lab_topic'))}</span>
       · rating: <i>{esc(a.get('original_rating'))[:60]}</i></div>
  <div class="post">{esc(a['_post'])[:600] or '<span class=dim>(no post text — image only)</span>'}</div>
  {_img_tag(a['_imgs'])}
  <div class="ctx"><b>claim (context):</b> {esc(a.get('claim_text'))[:300]}</div>
  <div class="split">235B → <b class="{a['_r']}">{a['_r']}</b> <span class=dim>({esc(a['_rr'])})</span>
       &nbsp;·&nbsp; Gemma → <b class="{a['_xr']}">{a['_xr']}</b> <span class=dim>({esc(a['_xrr'])})</span></div>
  <div class="note dim">235B: {esc(a.get('A_lab_note'))[:140]} | {esc(a.get('B_lab_note'))[:120]}</div>
  <div class="btns">
    <button class="btn" data-v="positive" onclick="choose('{fid}','positive')">positive</button>
    <button class="btn" data-v="negative" onclick="choose('{fid}','negative')">negative</button>
    <button class="btn" data-v="drop" onclick="choose('{fid}','drop')">drop</button>
  </div>
</div>""")

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Filter borderline review</title>
<style>
body{{font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:820px;margin:0 auto;padding:16px;color:#111}}
.card{{border:1px solid #ddd;border-radius:8px;padding:12px 14px;margin:14px 0}}
.hd{{color:#444;font-size:13px;margin-bottom:6px}} .topic{{background:#eef;padding:1px 6px;border-radius:4px}}
.post{{white-space:pre-wrap;background:#fafafa;padding:8px;border-radius:6px;margin:6px 0}}
img{{max-width:100%;max-height:420px;border-radius:6px;margin:6px 6px 6px 0}}
.ctx{{font-size:13px;color:#333;margin:6px 0}} .split{{margin:8px 0;font-size:13px}}
.note{{font-size:12px}} .dim{{color:#999}}
.positive{{color:#0a7d2e}} .negative{{color:#b00}} .drop{{color:#888}}
.btns{{margin-top:8px}} .btn{{padding:6px 16px;margin-right:8px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer;font-size:13px}}
.btn.sel{{background:#111;color:#fff;border-color:#111}}
#bar{{position:sticky;top:0;background:#fff;border-bottom:2px solid #111;padding:10px 0;z-index:9}}
textarea{{width:100%;height:70px;font-family:monospace;font-size:12px}}
</style></head><body>
<div id="bar">
  <b>Filter borderline review</b> — {len(rows)} decisive cards (label defaults to 235B; flip only what you disagree with).
  <button onclick="gen()" style="padding:6px 14px">Generate responses</button>
  <span id="count"></span>
  <textarea id="export" placeholder="click Generate — then copy this block back"></textarea>
</div>
{''.join(cards)}
<script>
function choose(id,v){{localStorage.setItem('fr_'+id,v);paint(id);}}
function paint(id){{const c=document.getElementById(id);const v=localStorage.getItem('fr_'+id)||c.dataset.default;
  c.querySelectorAll('.btn').forEach(b=>b.classList.toggle('sel',b.dataset.v===v));}}
window.onload=()=>{{document.querySelectorAll('.card').forEach(c=>{{if(!localStorage.getItem('fr_'+c.id))
  localStorage.setItem('fr_'+c.id,c.dataset.default);paint(c.id);}});gen();}};
function gen(){{let o=[],n=0,t=0;document.querySelectorAll('.card').forEach(c=>{{t++;const v=localStorage.getItem('fr_'+c.id);
  if(v){{o.push(c.id+':'+v);n++;}}}});document.getElementById('export').value=o.join(' ');
  document.getElementById('count').textContent=n+' / '+t+' decided';}}
</script></body></html>"""

    OUT_HTML.write_text(page)
    pl.DataFrame(items).write_parquet(OUT_ITEMS)
    flips = sum(1 for a in rows if a["_r"] == "negative" and a["_xr"] == "positive")
    print(f"wrote {OUT_HTML} — {len(rows)} decisive cards ({flips} negative→positive at the top)")
    print(f"wrote {OUT_ITEMS} (id→key + 235B default)")


if __name__ == "__main__":
    main()
