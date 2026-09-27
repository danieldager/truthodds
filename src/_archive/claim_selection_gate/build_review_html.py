"""Build a self-contained HTML review tool for the dataset-audit cases that need a HUMAN call:
  A) risky positive->negative relabels (carve-out / politics — the auditor's weak spot)
  B) negatives the auditor flagged check-worthy (mostly rhetoric errors — confirm the real ones)
  C) image-only positives the auditor found NO checkable claim in (out-of-context photo / lost caption)

Per item: FLAG (check-worthy -> positive) | PASS (ignore -> negative) | DROP (remove from dataset).
Selections persist in localStorage; a Generate button emits a compact "ID:choice" block to copy back
(and a Download button). The ID->row mapping is saved to gate_review_items.parquet so the responses can
be applied to the dataset reproducibly.

  uv run python -m eval.scripts.build_review_html  ->  eval/data/gate_review.html  +  gate_review_items.parquet
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import polars as pl

DATA = Path("eval/data")


def derive(r: dict) -> str:
    return "positive" if (r.get("in_scope") is True and r.get("has_claim") is True) else "negative"


def img_uri(p: str | None) -> str:
    if p and Path(p).exists():
        return "data:image/webp;base64," + base64.b64encode(Path(p).read_bytes()).decode()
    return ""


def main() -> None:
    rows = pl.read_parquet(DATA / "dataset_audit.parquet").filter(pl.col("recommend").is_not_null()).to_dicts()
    import re
    carve = re.compile(r"death|died|kill|crime|arrest|illness|accident|doping|fraud|recall", re.I)
    oos_topics = {"celebrity", "personal", "sports", "business", "product", "entertainment", "other"}

    p2n = [r for r in rows if r["current_label"] == "positive" and derive(r) == "negative"]
    risky = [r for r in p2n if not (r.get("topic") in oos_topics and not carve.search((r["text"] or "") + (r["note"] or "")))]
    n2p = [r for r in rows if r["current_label"] == "negative" and derive(r) == "positive"]

    def spec(uid, current_label, text, image_path, meta, note, default):
        return {"uid": uid, "current_label": current_label, "text": text, "image_path": image_path,
                "meta": meta, "note": note, "default": default}

    A = [spec(r["uid"], "positive", r.get("text") or "(no post text — image only)", r.get("image_path"),
              f"topic={r.get('topic')} · in_scope={r.get('in_scope')} · has_claim={r.get('has_claim')} · now=positive",
              "auditor: " + (r.get("note") or ""), "flag") for r in risky]
    B = [spec(r["uid"], "negative", r.get("text") or "", r.get("image_path"),
              f"topic={r.get('topic')} · now=negative", "auditor: " + (r.get("note") or ""), "pass") for r in n2p]
    C = []
    recp = DATA / "recovered_captions.parquet"
    for r in (pl.read_parquet(recp).to_dicts() if recp.exists() else []):
        leak = r.get("leakage")
        default = "drop" if leak else ("flag" if (r.get("re_in_scope") and r.get("re_has_claim")) else "pass")
        meta = (f"RECOVERED caption · topic={r.get('re_topic')} · in_scope={r.get('re_in_scope')} · "
                f"has_claim={r.get('re_has_claim')}" + (" · ⚠️ LEAKAGE — verify" if leak else ""))
        note = "original claim_text: " + (r.get("original_claim_text") or "")[:160] + "  |  re-audit: " + (r.get("re_note") or "")
        C.append(spec(r["uid"], "positive", r.get("recovered_caption") or "", r.get("image_path"), meta, note, default))

    buckets = [("A", "Risky positive→negative (carve-out / politics — the auditor's weak spot)", A),
               ("B", "Negatives the auditor flagged check-worthy (mostly rhetoric — confirm the few real)", B),
               ("C", "Image-only positives with RECOVERED caption (judge as normal posts; ⚠️ = possible verdict leakage)", C)]

    items, cards = [], []
    for prefix, title, lst in buckets:
        cards.append(f'<h2 id="sec{prefix}">{prefix}. {html.escape(title)} <span class="muted">({len(lst)})</span></h2>')
        for i, s in enumerate(lst, 1):
            iid = f"{prefix}{i:02d}"
            items.append({"id": iid, "uid": s["uid"], "current_label": s["current_label"],
                          "text": s["text"], "image_path": s["image_path"], "bucket": prefix})
            uri = img_uri(s["image_path"])
            imgtag = f'<img loading="lazy" src="{uri}">' if uri else ""
            txt = html.escape(s["text"] or "(no text)")
            note = html.escape(s["note"])
            radios = "".join(
                f'<label class="opt {v}"><input type="radio" name="{iid}" value="{v}"{" checked" if v==s["default"] else ""}>{lbl}</label>'
                for v, lbl in [("flag", "🚩 FLAG (check-worthy → positive)"), ("pass", "➡️ PASS (ignore → negative)"), ("drop", "🗑️ DROP (remove)")])
            cards.append(
                f'<div class="card" data-id="{iid}"><div class="hd"><b>{iid}</b> <span class="muted">{html.escape(s["meta"])}</span></div>'
                f'<div class="bd">{imgtag}<div class="txt">{txt}<div class="note">{note}</div></div></div>'
                f'<div class="opts">{radios}</div></div>')

    pl.DataFrame(items).write_parquet(DATA / "gate_review_items.parquet")
    body = "\n".join(cards)
    n = len(items)
    page = """<!doctype html><html><head><meta charset="utf-8"><title>Gate dataset review</title><style>
body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#f5f5f5;color:#1a1a1a}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:9}
h2{margin:24px 16px 8px;font-size:16px} .muted{color:#888;font-weight:400}
.card{background:#fff;border:1px solid #e2e2e2;border-radius:8px;margin:8px 16px;padding:10px}
.card.done{border-left:4px solid #34a853} .hd{font-size:12px;margin-bottom:6px}
.bd{display:flex;gap:12px} .bd img{max-width:420px;max-height:340px;border-radius:6px;border:1px solid #eee;object-fit:contain}
.txt{flex:1;white-space:pre-wrap} .note{color:#666;font-style:italic;margin-top:6px;font-size:13px}
.opts{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.opt{padding:6px 10px;border:1px solid #ccc;border-radius:6px;cursor:pointer;user-select:none}
.opt input{margin-right:5px} .opt.flag:has(:checked){background:#e6f4ea;border-color:#34a853}
.opt.pass:has(:checked){background:#e8f0fe;border-color:#4285f4} .opt.drop:has(:checked){background:#fce8e6;border-color:#ea4335}
button{padding:8px 14px;border:0;border-radius:6px;background:#1a73e8;color:#fff;cursor:pointer;font-size:14px}
#out{width:100%;height:90px;margin-top:8px;font-family:monospace;font-size:12px}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
</style></head><body>
<header><div class="bar"><b>Gate dataset review</b> <span id="prog" class="muted"></span>
<button onclick="gen()">Generate responses</button><button onclick="dl()">Download JSON</button>
<button onclick="copy()">Copy</button><span class="muted">FLAG=positive · PASS=negative · DROP=remove. Saved in your browser as you go.</span></div>
<textarea id="out" placeholder="Click Generate — then copy this back to Claude (or Download)."></textarea></header>
__BODY__
<script>
const N=__N__;
function mark(){document.querySelectorAll('.card').forEach(c=>{const s=c.querySelector('input:checked');c.classList.toggle('done',!!s);});
 const d=document.querySelectorAll('input:checked').length;document.getElementById('prog').textContent=d+'/'+N+' decided';}
function save(){const o={};document.querySelectorAll('input:checked').forEach(i=>o[i.name]=i.value);localStorage.setItem('gatereview',JSON.stringify(o));}
function load(){try{const o=JSON.parse(localStorage.getItem('gatereview')||'{}');for(const k in o){const el=document.querySelector(`input[name="${k}"][value="${o[k]}"]`);if(el)el.checked=true;}}catch(e){}}
function lines(){return [...document.querySelectorAll('.card')].map(c=>{const s=c.querySelector('input:checked');return c.dataset.id+':'+(s?s.value:'NONE');}).join(' ');}
function gen(){document.getElementById('out').value=lines();}
function dl(){const b=new Blob([lines()],{type:'text/plain'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='gate_review_responses.txt';a.click();}
function copy(){gen();document.getElementById('out').select();document.execCommand('copy');}
document.addEventListener('change',()=>{save();mark();});
load();mark();
</script></body></html>"""
    page = page.replace("__BODY__", body).replace("__N__", str(n))
    (DATA / "gate_review.html").write_text(page)
    print(f"buckets: A risky {len(A)} | B neg-flagged {len(B)} | C recovered-image-only {len(C)} | total {n}")
    print(f"wrote eval/data/gate_review.html ({len(page)//1024} KB) + gate_review_items.parquet")


if __name__ == "__main__":
    main()
