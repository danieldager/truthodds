"""WS2 — human-review HTML for the verdict-gold audit. Surfaces EVERY rating->veracity disagreement and
EVERY axis cross-model disagreement (+ the unprovable spot-check) from verdict_audit.parquet, so the human
adjudicates only the contested rows. Mirrors build_review_html.py (gate tool): localStorage persistence,
a Generate button emits an "ID:choice" block to paste back, and the ID->row map is saved for reproducible
application via the provenance ledger.

  uv run python -m eval.scripts.build_verdict_review
    -> eval/data/verdict_audit_review.html  +  eval/data/verdict_review_items.parquet

Buckets (priority order):
  A  rating->veracity disagreement (the headline label)   -> keep gold | 1 | 2 | 3 | 4 | 5
  C  unprovable spot-check (protects the gold-3 split)     -> unprovable(3) | mixed(3) | false(1) | true(5)
  B  judged_axis disagreement (headline include/exclude)   -> content | attribution | artifact
Default radio = the current gold value (so "no change" is one click / leaving it). --axis-cap N samples the
axis bucket (it is high-volume / low-stakes vs A+C); 0 = all.
"""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

import polars as pl

D = Path("eval/data")


def img_uri(paths) -> str:
    for p in (paths or [])[:1]:
        if p and Path(p).exists():
            return "data:image/webp;base64," + base64.b64encode(Path(p).read_bytes()).decode()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis-cap", type=int, default=120, help="max axis cards (hash-stable sample); 0 = all")
    args = ap.parse_args()

    df = pl.read_parquet(D / "verdict_audit.parquet").filter(pl.col("audited") & pl.col("needs_review"))
    ver = df.filter(pl.col("veracity_disagree")).sort(["veracity_both_disagree", "review_url"], descending=[True, False])
    unp = df.filter(pl.col("unprovable_suspect"))
    axis = df.filter(pl.col("axis_needs_human") & (~pl.col("veracity_disagree")) & (~pl.col("unprovable_suspect")))
    if args.axis_cap and axis.height > args.axis_cap:
        axis = axis.sort("review_url").head(args.axis_cap)  # review_url is hash-stable -> reproducible sample

    def card(bucket, idx, r, options, default):
        iid = f"{bucket}{idx:03d}"
        uri = img_uri(r.get("image_paths"))
        imgtag = f'<img loading="lazy" src="{uri}">' if uri else ""
        meta = (f"{r.get('publisher_site')} · rating=<b>{html.escape(str(r.get('original_rating'))[:60])}</b> · "
                f"gold v{r.get('gold_veracity')}/{r.get('rating_subtype')} · axis={r.get('axis_rule')}")
        models = (f"235B: agree={r.get('B_lab_veracity_agrees')} sugg=v{r.get('B_lab_suggested_veracity')} "
                  f"axis={r.get('B_lab_judged_axis')} — {html.escape((r.get('B_lab_note') or '')[:140])}<br>"
                  f"Gemma: agree={r.get('B_x_veracity_agrees')} sugg=v{r.get('B_x_suggested_veracity')} "
                  f"axis={r.get('B_x_judged_axis')} — {html.escape((r.get('B_x_note') or '')[:140])}")
        claim = html.escape((r.get("claim_text") or "")[:400])
        radios = "".join(
            f'<label class="opt"><input type="radio" name="{iid}" value="{v}"{" checked" if v==default else ""}>{lbl}</label>'
            for v, lbl in options)
        items.append({"id": iid, "review_url": r["review_url"], "bucket": bucket,
                      "gold_veracity": r.get("gold_veracity"), "rating_subtype": r.get("rating_subtype"),
                      "axis_rule": r.get("axis_rule")})
        return (f'<div class="card" data-id="{iid}"><div class="hd"><b>{iid}</b> <span class="muted">{meta}</span></div>'
                f'<div class="bd">{imgtag}<div class="txt"><b>claim:</b> {claim}'
                f'<div class="note">{models}</div></div></div><div class="opts">{radios}</div></div>')

    VER_OPTS = [("keep", "✓ keep gold"), ("1", "1 false"), ("2", "2 mostly-false"),
                ("3", "3 mixed/unprov"), ("4", "4 mostly-true"), ("5", "5 true")]
    UNP_OPTS = [("unprovable", "✓ unprovable (v3, no evidence)"), ("mixed", "mixed (v3, contested)"),
                ("1", "resolved → false (v1)"), ("5", "resolved → true (v5)")]
    AX_OPTS = [("content", "content"), ("attribution", "attribution"), ("artifact", "artifact")]

    items, cards = [], []
    cards.append(f'<h2>A. rating→veracity disagreement <span class="muted">({ver.height}) — the headline label; ⬆ both-model first</span></h2>')
    for i, r in enumerate(ver.to_dicts(), 1):
        cards.append(card("A", i, r, VER_OPTS, "keep"))
    cards.append(f'<h2>C. unprovable spot-check <span class="muted">({unp.height}) — genuine no-evidence vs mislabeled-resolved</span></h2>')
    for i, r in enumerate(unp.to_dicts(), 1):
        cards.append(card("C", i, r, UNP_OPTS, "unprovable"))
    cards.append(f'<h2>B. judged_axis disagreement <span class="muted">({axis.height} shown) — content stays in the headline; artifact/attribution are excluded</span></h2>')
    for i, r in enumerate(axis.to_dicts(), 1):
        cards.append(card("B", i, r, AX_OPTS, r.get("axis_rule") or "content"))

    pl.DataFrame(items).write_parquet(D / "verdict_review_items.parquet")
    n = len(items)
    page = """<!doctype html><html><head><meta charset="utf-8"><title>Verdict gold review</title><style>
body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#f5f5f5;color:#1a1a1a}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:9}
h2{margin:24px 16px 8px;font-size:16px} .muted{color:#888;font-weight:400}
.card{background:#fff;border:1px solid #e2e2e2;border-radius:8px;margin:8px 16px;padding:10px}
.card.done{border-left:4px solid #34a853} .hd{font-size:12px;margin-bottom:6px}
.bd{display:flex;gap:12px} .bd img{max-width:380px;max-height:300px;border-radius:6px;border:1px solid #eee;object-fit:contain}
.txt{flex:1;white-space:pre-wrap} .note{color:#666;font-style:italic;margin-top:6px;font-size:12px}
.opts{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.opt{padding:6px 10px;border:1px solid #ccc;border-radius:6px;cursor:pointer;user-select:none}
.opt input{margin-right:5px} .opt:has(:checked){background:#e6f4ea;border-color:#34a853}
button{padding:8px 14px;border:0;border-radius:6px;background:#1a73e8;color:#fff;cursor:pointer;font-size:14px}
#out{width:100%;height:90px;margin-top:8px;font-family:monospace;font-size:12px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
</style></head><body>
<header><div class="bar"><b>Verdict gold review</b> <span id="prog" class="muted"></span>
<button onclick="gen()">Generate</button><button onclick="dl()">Download</button><button onclick="copy()">Copy</button>
<span class="muted">A=veracity · C=unprovable · B=axis. Defaults = current gold (leave = no change). Saved in your browser.</span></div>
<textarea id="out" placeholder="Click Generate — copy this back to Claude (or Download)."></textarea></header>
__BODY__
<script>
const N=__N__;
function mark(){document.querySelectorAll('.card').forEach(c=>{const s=c.querySelector('input:checked');c.classList.toggle('done',!!s);});
 const d=document.querySelectorAll('input:checked').length;document.getElementById('prog').textContent=d+'/'+N+' set';}
function save(){const o={};document.querySelectorAll('input:checked').forEach(i=>o[i.name]=i.value);localStorage.setItem('verdictreview',JSON.stringify(o));}
function load(){try{const o=JSON.parse(localStorage.getItem('verdictreview')||'{}');for(const k in o){const el=document.querySelector(`input[name="${k}"][value="${o[k]}"]`);if(el)el.checked=true;}}catch(e){}}
function lines(){return [...document.querySelectorAll('.card')].map(c=>{const s=c.querySelector('input:checked');return c.dataset.id+':'+(s?s.value:'NONE');}).join(' ');}
function gen(){document.getElementById('out').value=lines();}
function dl(){const b=new Blob([lines()],{type:'text/plain'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='verdict_review_responses.txt';a.click();}
function copy(){gen();document.getElementById('out').select();document.execCommand('copy');}
document.addEventListener('change',()=>{save();mark();});
load();mark();
</script></body></html>"""
    page = page.replace("__BODY__", "\n".join(cards)).replace("__N__", str(n))
    (D / "verdict_audit_review.html").write_text(page)
    print(f"buckets: A veracity {ver.height} | C unprovable {unp.height} | B axis {axis.height} (shown) | total {n}")
    print(f"wrote eval/data/verdict_audit_review.html ({len(page)//1024} KB) + verdict_review_items.parquet")


if __name__ == "__main__":
    main()
