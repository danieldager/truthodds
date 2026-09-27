"""Hand-labelling UI for READ cases — the human anchor the LLM panels are scored against.

The READ A/B (read_ab.py) was retracted because both arms were graded by agent-generated
labels that over-demote (93% demotion vs Daniel's 63% on the same reads): two LLM panels
agreeing with each other while diverging from the human is correlated bias, not evidence.
So the yardstick has to be hand-made. This builds the instrument for that: one self-contained
HTML page, no server, no network (a strict CSP blocks external requests), that puts one case
per screen and asks a single question — what does THIS document say about THIS claim.

Anti-anchoring is the whole point of the design: the model's own answer is collapsed behind a
toggle and defaults to hidden, and the failure mode a case was drawn for (`_mode`) is stripped
from the payload entirely so it can never leak, not even through view-source. Labels autosave
to localStorage keyed by cid, so a refresh or a crash never costs a session.

  uv run python -m eval.scripts.build_eval.build_read_label_ui -i <cases.json> -o <out.html>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LIGHT = ("--paper:#fdfcfa;--ink:#1a1a1a;--sub:#5f6368;--line:#e8e6e1;--accent:#283593;"
         "--card:#ffffff;--zebra:#faf9f6;--mark:#fdf0bd;--btn:#ffffff;--shadow:#0000000a;"
         "--spbg:#eef0fa;--spline:#283593;--rfbg:#f9efeb;--rfline:#9c4227;"
         "--ctbg:#f2f1ec;--ctline:#a6a29a")
DARK = ("--paper:#16171a;--ink:#e8e6e1;--sub:#9aa0a6;--line:#2c2e33;--accent:#93a0e8;"
        "--card:#1c1e22;--zebra:#202227;--mark:#4a4321;--btn:#22242a;--shadow:#00000040;"
        "--spbg:#23283c;--spline:#93a0e8;--rfbg:#382622;--rfline:#e0997f;"
        "--ctbg:#26272b;--ctline:#6d6a63")

DIRECTIONS = ["supports", "partially-supports", "context",
              "partially-refutes", "refutes", "irrelevant"]
REASONS = ["off-claim", "adjacent-only", "stale", "opinion-only", "restates-claim", "no-content"]

# EN + FR — claims come in both languages. Kept deliberately short: this only decides
# which words get a background tint, so a miss costs nothing but a bit of extra colour.
STOPWORDS = """a about after all also an and any are as at be because been but by can could
did do does for from had has have he her him his how i if in into is it its just like may
me more most no not of on one only or other our out over said say she should so some such
than that the their them then there these they this those to too under up was we were what
when where which who will with would you your
au aux avec ce ces cet cette dans de des du elle en est et eux il ils je la le les leur lui
ma mais me meme mes moi mon ne nos notre nous on ou par pas peu plus pour qu que quel quelle
qui sa sans se ses son sont sur ta te tes toi ton tous tout tu un une vos votre vous y
etre avoir fait ete cela dont donc alors ainsi entre chez selon apres avant depuis
etait etaient sera seront avait avaient ont ait soit ete comme deja encore aussi
tres bien meme cette celui ceux lors"""

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>READ hand-labelling</title>
<style>
:root{__LIGHT__}
@media(prefers-color-scheme:dark){:root{__DARK__}}
:root[data-theme=dark]{__DARK__}
:root[data-theme=light]{__LIGHT__}
*{box-sizing:border-box}
html,body{background:var(--paper);color:var(--ink)}
body{margin:0;font:15px/1.6 "Avenir Next",Avenir,"Segoe UI",system-ui,-apple-system,
BlinkMacSystemFont,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:26px 28px 120px}
header{border-bottom:2px solid var(--ink);padding-bottom:12px;margin-bottom:10px;
display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
h1{font-size:19px;font-weight:700;letter-spacing:-.3px;margin:0;flex:1 1 auto}
.meta{color:var(--sub);font-size:12.5px;font-variant-numeric:tabular-nums}
.sticky{position:sticky;top:0;z-index:20;background:var(--paper);padding-top:10px}
.claimbox{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:6px;padding:12px 16px;box-shadow:0 1px 2px var(--shadow)}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.5px;font-weight:700;color:var(--sub)}
.claim{font-size:16px;line-height:1.5;margin:3px 0 6px;white-space:pre-wrap}
.docmeta{color:var(--sub);font-size:12px;word-break:break-all}
.docmeta b{color:var(--ink);font-weight:600}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:9px 0 7px;
border-bottom:1px solid var(--line);margin-bottom:4px}
button{font:inherit;font-size:12.5px;color:var(--ink);background:var(--btn);
border:1px solid var(--line);border-radius:5px;padding:5px 11px;cursor:pointer}
button:hover{border-color:var(--sub)}
button.on{background:var(--accent);border-color:var(--accent);color:var(--paper);font-weight:600}
button.sp.on{background:var(--spline);border-color:var(--spline)}
button.rf.on{background:var(--rfline);border-color:var(--rfline)}
button.ct.on{background:var(--ctline);border-color:var(--ctline)}
.spacer{flex:1 1 auto}
.doc{margin:10px 0 4px}
.s{display:flex;gap:10px;padding:4px 10px 4px 7px;border-left:5px solid transparent;
border-radius:3px;cursor:pointer;align-items:flex-start}
.s:nth-child(even){background:var(--zebra)}
.s:hover{outline:1px solid var(--line)}
.sn{flex:0 0 34px;text-align:right;color:var(--sub);font-size:11.5px;line-height:1.9;
font-variant-numeric:tabular-nums;user-select:none}
.st{flex:1 1 auto;white-space:pre-wrap;font-size:14.5px}
/* three states that survive a monochrome / colourblind read: the left rule differs in
   STYLE (solid / double / dotted) as well as hue, and each row carries a letter tag. */
.s.sp{background:var(--spbg);border-left:5px solid var(--spline)}
.s.rf{background:var(--rfbg);border-left:5px double var(--rfline)}
.s.ct{background:var(--ctbg);border-left:5px dotted var(--ctline)}
.tag{flex:0 0 auto;font-size:9.5px;text-transform:uppercase;letter-spacing:.6px;font-weight:700;
color:var(--sub);line-height:2.6;user-select:none;white-space:nowrap}
.tag b{font-size:11px;border:1px solid currentColor;border-radius:3px;padding:0 3px;margin-right:4px}
.s.sp .tag{color:var(--spline)}
.s.rf .tag{color:var(--rfline)}
mark{background:var(--mark);color:inherit;padding:0 1px;border-radius:2px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 16px;margin:14px 0;box-shadow:0 1px 2px var(--shadow)}
.row{display:flex;gap:7px;flex-wrap:wrap;margin:5px 0 2px}
textarea{width:100%;min-height:58px;font:inherit;font-size:13.5px;color:var(--ink);
background:var(--paper);border:1px solid var(--line);border-radius:5px;padding:7px 9px;resize:vertical}
details.model{margin:12px 0;font-size:13px;color:var(--sub)}
details.model summary{cursor:pointer;color:var(--accent);font-size:12.5px}
details.model .body{background:var(--zebra);border:1px solid var(--line);border-radius:6px;
padding:8px 12px;margin-top:6px;font-variant-numeric:tabular-nums}
.nav{display:flex;gap:10px;align-items:center;margin-top:16px}
.keys{color:var(--sub);font-size:11.5px;margin-top:18px;border-top:1px solid var(--line);
padding-top:9px;line-height:1.9}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:3px;padding:0 4px;
font:inherit;font-size:11px;background:var(--card)}
.hid{display:none}
#restored{color:var(--accent);font-size:12.5px}
</style></head><body><div class="wrap">

<header>
  <h1>READ hand-labelling</h1>
  <span class="meta" id="progress"></span>
  <span class="meta" id="done"></span>
  <span id="restored"></span>
  <button id="theme" title="toggle light/dark">theme</button>
  <button id="export">export json</button>
</header>

<div class="sticky">
  <div class="claimbox">
    <div class="lbl">Claim</div>
    <div class="claim" id="claim"></div>
    <div class="docmeta" id="docmeta"></div>
  </div>
  <div class="bar">
    <span class="lbl">Bucket</span>
    <button class="sp" id="bsp">supports</button>
    <button class="rf" id="brf">refutes</button>
    <button class="ct" id="bct">context</button>
    <span class="spacer"></span>
    <button id="hl">highlight: on</button>
    <button id="clear">clear selections</button>
  </div>
</div>

<div class="doc" id="doc"></div>

<div class="panel">
  <div class="lbl">Direction</div>
  <div class="row" id="dirs"></div>
  <div id="reasonwrap" class="hid">
    <div class="lbl" style="margin-top:9px">Reason code</div>
    <div class="row" id="reasons"></div>
  </div>
  <div class="lbl" style="margin-top:11px">Notes</div>
  <textarea id="notes" placeholder="anything the label alone does not capture"></textarea>
</div>

<details class="model"><summary>show what the model said</summary>
  <div class="body" id="modelbody"></div>
</details>

<div class="nav">
  <button id="prev">&#8592; prev</button>
  <button id="next">next &#8594;</button>
</div>

<div class="keys">
  <kbd>1</kbd>&#8211;<kbd>6</kbd> direction (__DIRLEGEND__) &#183;
  <kbd>s</kbd>/<kbd>r</kbd>/<kbd>c</kbd> active bucket (supports / refutes / context) &#183;
  <kbd>&#8592;</kbd>/<kbd>&#8594;</kbd> previous/next case &#183;
  click a sentence to toggle it into the active bucket (a sentence held by another
  bucket moves)
</div>
</div>

<script>
const CASES = __DATA__;
const DIRECTIONS = __DIRECTIONS__;
const REASONS = __REASONS__;
const STOP = new Set(__STOP__);
const KEY = "read_label_ui_v3";   /* v3 = three buckets; older two-bucket records are ignored */

/* bucket code -> {field on the record, row class, tag letter, label} */
const BUCKETS = {sp: {f: "supports_ids", k: "S", n: "supports"},
                 rf: {f: "refutes_ids",  k: "R", n: "refutes"},
                 ct: {f: "context_ids",  k: "C", n: "context"}};
const BCODES = ["sp", "rf", "ct"];

let idx = 0, bucket = "sp", highlight = true;
let state = {};

try { state = JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (e) { state = {}; }
const known = new Set(CASES.map(c => c.cid));
for (const k of Object.keys(state)) if (!known.has(k) || isEmpty(state[k])) delete state[k];
const nRestored = Object.keys(state).length;

function isEmpty(r) {
  return !r || (!r.direction && !r.notes
                && !BCODES.some(b => (r[BUCKETS[b].f] || []).length));
}
function rec(cid) {
  if (!state[cid]) state[cid] = {direction: null, supports_ids: [], refutes_ids: [],
                                 context_ids: [], reason_code: null, notes: "",
                                 labelled_at_index: null};
  return state[cid];
}
function save() {
  /* merely visiting a case must not look like work on reload: touched-but-blank
     records are dropped before writing. */
  const keep = {};
  for (const [k, v] of Object.entries(state)) if (!isEmpty(v)) keep[k] = v;
  try { localStorage.setItem(KEY, JSON.stringify(keep)); } catch (e) {}
  paintCounts();
}
function nDone() { return Object.values(state).filter(r => r.direction).length; }

/* ---- claim keywords ------------------------------------------------------ */
const WORD = /[\p{L}\p{N}]+/gu;
/* Compare accent-free: the stopword list is unaccented, and "election"/"élection"
   is the same word for the purpose of a background tint. */
function norm(w) {
  return w.toLowerCase().normalize("NFD").replace(/\p{Diacritic}/gu, "");
}
function keywords(claim) {
  const out = new Set();
  for (const m of (claim || "").matchAll(WORD)) {
    const w = norm(m[0]);
    if (STOP.has(w)) continue;
    if (w.length < 3 && !/^\d+$/.test(w)) continue;
    out.add(w);
  }
  return out;
}
function isKw(w, kws) {
  const l = norm(w);
  if (kws.has(l)) return true;
  if (l.length > 3 && l.endsWith("s") && kws.has(l.slice(0, -1))) return true;
  return kws.has(l + "s");
}

/* Text nodes only, never innerHTML: HTML metacharacters in the document can never
   become markup, and newlines survive via white-space:pre-wrap. */
function sentenceNodes(text, kws) {
  const frag = document.createDocumentFragment();
  if (!highlight || kws.size === 0) { frag.appendChild(document.createTextNode(text)); return frag; }
  let last = 0;
  for (const m of text.matchAll(WORD)) {
    if (!isKw(m[0], kws)) continue;
    if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
    const mk = document.createElement("mark");
    mk.textContent = m[0];
    frag.appendChild(mk);
    last = m.index + m[0].length;
  }
  frag.appendChild(document.createTextNode(text.slice(last)));
  return frag;
}

/* ---- buttons ------------------------------------------------------------- */
function mkButtons(host, values, onclick) {
  host.textContent = "";
  values.forEach(v => {
    const b = document.createElement("button");
    b.textContent = v;
    b.dataset.v = v;
    b.onclick = () => onclick(v);
    host.appendChild(b);
  });
}
mkButtons(document.getElementById("dirs"), DIRECTIONS, setDirection);
mkButtons(document.getElementById("reasons"), REASONS, setReason);

function setDirection(v) {
  const r = rec(CASES[idx].cid);
  r.direction = (r.direction === v) ? null : v;
  if (r.direction === null) r.labelled_at_index = null;
  else if (r.labelled_at_index === null) r.labelled_at_index = idx;
  if (r.direction !== "context" && r.direction !== "irrelevant") r.reason_code = null;
  save(); paintLabels();
}
function setReason(v) {
  const r = rec(CASES[idx].cid);
  r.reason_code = (r.reason_code === v) ? null : v;
  save(); paintLabels();
}

/* ---- rendering ----------------------------------------------------------- */
function paintCounts() {
  document.getElementById("progress").textContent = "case " + (idx + 1) + " of " + CASES.length;
  document.getElementById("done").textContent = nDone() + " / " + CASES.length + " labelled";
}
function paintLabels() {
  const r = rec(CASES[idx].cid);
  document.querySelectorAll("#dirs button").forEach(b =>
    b.classList.toggle("on", b.dataset.v === r.direction));
  const showReason = (r.direction === "context" || r.direction === "irrelevant");
  document.getElementById("reasonwrap").classList.toggle("hid", !showReason);
  document.querySelectorAll("#reasons button").forEach(b =>
    b.classList.toggle("on", b.dataset.v === r.reason_code));
  document.getElementById("notes").value = r.notes || "";
  BCODES.forEach(b => document.getElementById("b" + b).classList.toggle("on", bucket === b));
  paintSelection();
  paintCounts();
}
function paintSelection() {
  const r = rec(CASES[idx].cid);
  document.querySelectorAll("#doc .s").forEach(row => {
    const n = Number(row.dataset.n);
    const held = BCODES.find(b => r[BUCKETS[b].f].includes(n));
    BCODES.forEach(b => row.classList.toggle(b, b === held));
    const tag = row.querySelector(".tag");
    tag.textContent = "";
    if (held) {
      const k = document.createElement("b");
      k.textContent = BUCKETS[held].k;
      tag.append(k, document.createTextNode(BUCKETS[held].n));
    }
  });
}
/* one sentence lives in at most one bucket: clicking it under a different bucket MOVES it. */
function toggleSentence(n) {
  const r = rec(CASES[idx].cid);
  const held = BCODES.find(b => r[BUCKETS[b].f].includes(n));
  if (held) r[BUCKETS[held].f].splice(r[BUCKETS[held].f].indexOf(n), 1);
  if (held !== bucket) {
    const mine = r[BUCKETS[bucket].f];
    mine.push(n);
    mine.sort((a, b) => a - b);
  }
  save(); paintSelection();
}

function render() {
  const c = CASES[idx];
  document.getElementById("claim").textContent = c.claim || "";
  const dm = document.getElementById("docmeta");
  dm.textContent = "";
  const parts = [["domain", c.domain], ["reliability", c.rel], ["provenance", c.prov],
                 ["gold veracity (claim)", c.gold_veracity], ["cid", c.cid], ["url", c.doc_url]];
  parts.forEach(([k, v], i) => {
    if (v === undefined || v === null || v === "") return;
    if (dm.childNodes.length) dm.appendChild(document.createTextNode("  ·  "));
    const b = document.createElement("b");
    b.textContent = k + " ";
    dm.appendChild(b);
    dm.appendChild(document.createTextNode(String(v)));
  });

  const kws = keywords(c.claim);
  const doc = document.getElementById("doc");
  doc.textContent = "";
  const ids = c.sent_ids || [], sents = c.sents || [];
  for (let i = 0; i < Math.min(ids.length, sents.length); i++) {
    const row = document.createElement("div");
    row.className = "s";
    row.dataset.n = ids[i];
    const num = document.createElement("div");
    num.className = "sn";
    num.textContent = ids[i];
    const txt = document.createElement("div");
    txt.className = "st";
    txt.appendChild(sentenceNodes(String(sents[i]), kws));
    const tag = document.createElement("div");
    tag.className = "tag";
    row.append(num, txt, tag);
    row.onclick = () => toggleSentence(ids[i]);
    doc.appendChild(row);
  }

  const mb = document.getElementById("modelbody");
  mb.textContent = "answered " + (c.model_dir || "—")
    + "   |   support " + JSON.stringify(c.model_support || [])
    + "   against " + JSON.stringify(c.model_against || [])
    + "   context " + JSON.stringify(c.model_context || []);
  document.querySelector("details.model").open = false;
  window.scrollTo(0, 0);
  paintLabels();
}
function go(d) {
  const n = idx + d;
  if (n < 0 || n >= CASES.length) return;
  idx = n;
  render();
}

/* ---- chrome -------------------------------------------------------------- */
document.getElementById("notes").oninput = e => { rec(CASES[idx].cid).notes = e.target.value; save(); };
BCODES.forEach(b => document.getElementById("b" + b).onclick = () => {
  bucket = b; paintLabels();
});
document.getElementById("clear").onclick = () => {
  const r = rec(CASES[idx].cid);
  BCODES.forEach(b => r[BUCKETS[b].f] = []);
  save(); paintSelection();
};
document.getElementById("hl").onclick = e => {
  highlight = !highlight;
  e.target.textContent = "highlight: " + (highlight ? "on" : "off");
  render();
};
document.getElementById("prev").onclick = () => go(-1);
document.getElementById("next").onclick = () => go(1);
document.getElementById("theme").onclick = () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const dark = cur ? cur === "dark"
    : window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
};
document.getElementById("export").onclick = () => {
  const out = [];
  CASES.forEach(c => {
    const r = state[c.cid];
    if (!r || !r.direction) return;
    out.push({cid: c.cid, direction: r.direction, supports_ids: r.supports_ids,
              refutes_ids: r.refutes_ids, context_ids: r.context_ids,
              reason_code: r.reason_code || null, notes: r.notes || "",
              labelled_at_index: r.labelled_at_index});
  });
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "read_labels.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};
document.addEventListener("keydown", e => {
  const t = e.target.tagName;
  if (t === "TEXTAREA" || t === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "ArrowLeft") { go(-1); e.preventDefault(); return; }
  if (e.key === "ArrowRight") { go(1); e.preventDefault(); return; }
  const b = {s: "sp", r: "rf", c: "ct"}[e.key.toLowerCase()];
  if (b) { bucket = b; paintLabels(); return; }
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= DIRECTIONS.length) { setDirection(DIRECTIONS[n - 1]); e.preventDefault(); }
});

if (nRestored) document.getElementById("restored").textContent =
  "restored " + nRestored + " saved label" + (nRestored === 1 ? "" : "s");
render();
</script>
</body></html>
"""


def build(cases: list[dict]) -> str:
    # strip private keys (_mode above all): the failure mode a case was drawn for is the
    # expected answer, so it must not reach the page even via view-source.
    clean = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cases]
    data = json.dumps(clean, ensure_ascii=False)
    for ch, esc in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026")):
        data = data.replace(ch, esc)
    legend = ", ".join(f"{i}={d}" for i, d in enumerate(DIRECTIONS, 1))
    return (HTML.replace("__LIGHT__", LIGHT)
                .replace("__DARK__", DARK)
                .replace("__DIRLEGEND__", legend)
                .replace("__DIRECTIONS__", json.dumps(DIRECTIONS))
                .replace("__REASONS__", json.dumps(REASONS))
                .replace("__STOP__", json.dumps(sorted(set(STOPWORDS.split()))))
                .replace("__DATA__", data))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", required=True, type=Path,
                    help='cases JSON: {"calibration": [{cid, claim, sent_ids, sents, ...}]}')
    ap.add_argument("-o", "--output", required=True, type=Path, help="self-contained HTML page")
    args = ap.parse_args()

    raw = json.loads(args.input.read_text())
    cases = raw["calibration"] if isinstance(raw, dict) else raw
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(cases))
    print(f"wrote {args.output} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
