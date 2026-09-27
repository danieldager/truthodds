"""Build the fc_gold filter-review page: a blind, stratified hand-audit sample.

Daniel hand-reads dropped claims against their fact-check articles to judge whether
the three filters that take the frozen fc_gold population (3,280 claims) down to the
clean binary fitting subset (2,057) are trustworthy. None of the three has a human
audit; this page is that audit's instrument.

The frozen -> clean funnel (disjoint, in order of application):

    frozen fc_gold.parquet ......................... 3,280
      - claim-shape screens (44) ................... 3,236
          resolution_status needs-article/self-attr (19)
          unresolved referent (1)
          misattributed media + media locus (24)
      - attribution axis: judged_axis == quote_attribution (761) . 2,475
      - graded rating: not clear_true/clear_false (418) .......... 2,057
    clean binary subset ............................ 2,057  (831 T / 1,226 F)

The page draws a seeded, stratified sample of 150 rows: 50 attribution, 50
graded-rating (stratified by harmonised subtype), 25 claim-shape (all classes
represented), 25 KEPT controls (mixed true/false), shuffled. The filter that fired
is hidden behind a toggle so Daniel judges each claim blind. Verdicts persist in the
browser and export as JSON.

    uv run python -m eval.scripts.build_eval.build_fc_gold_filter_review

$0, no API calls. Writes review.html + review.csv under eval/data/fc_gold_filter_review/.
"""
from __future__ import annotations

import csv
import html
import json
import random
from pathlib import Path

import pandas as pd

FROZEN = Path("eval/data/populations/fc_gold.parquet")
RESULTS = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
REGISTRY = Path("eval/data/exclusions/registry.parquet")
V2 = Path("eval/data/fc_gold_v2.parquet")          # source of the original rating string
OUTDIR = Path("eval/data/fc_gold_filter_review")
SEED = 20260914

VERACITY_NAME = {1: "False", 2: "Mostly false", 3: "Unprovable",
                 4: "Mostly true", 5: "True"}
SUBTYPE_NAME = {"clear_true": "clear true", "clear_false": "clear false",
                "mostly_true": "mostly true", "mostly_false": "mostly false",
                "unprovable": "unprovable / no evidence"}


def build_frame() -> pd.DataFrame:
    """Frozen population joined with claim text, resolution status, original rating."""
    frz = pd.read_parquet(FROZEN).set_index("claim_id")
    rs, ct = {}, {}
    for line in RESULTS.open():
        r = json.loads(line)
        rs[r["review_url"]] = r.get("resolution_status")
        ct[r["review_url"]] = r.get("claim_resolved") or r.get("claim_text") or ""
    frz["resolution_status"] = pd.Series(rs)
    frz["claim_text"] = pd.Series(ct)
    v2 = (pd.read_parquet(V2)[["review_url", "original_rating"]]
          .dropna().drop_duplicates("review_url").set_index("review_url"))
    frz["original_rating"] = v2["original_rating"]
    return frz


def reconstruct(frz: pd.DataFrame) -> dict[str, set]:
    """Assign every claim to the FIRST filter that removes it (disjoint), in the
    same order graded_urn's clean subset is produced."""
    reg = pd.read_parquet(REGISTRY)

    def ids(rule: str) -> set:
        return set(reg[reg.rule == rule].claim_id)

    keep = set(frz.index)
    b: dict[str, set] = {}
    a1 = set(frz[frz.resolution_status.isin(
        ["needs-article", "unresolvable-list", "self-attributed"])].index) & keep
    b["shape_resolution"] = a1; keep -= a1
    a2 = ids("claim_screen_unresolved_referent") & keep
    b["shape_unresolved_referent"] = a2; keep -= a2
    a3 = (ids("claim_screen_misattributed_media") | ids("claim_screen_media_locus")) & keep
    b["shape_media"] = a3; keep -= a3
    att = set(frz[frz.judged_axis_llm == "quote_attribution"].index) & keep
    b["attribution"] = att; keep -= att
    gr = set(frz[~frz.rating_subtype.isin(["clear_true", "clear_false"])].index) & keep
    b["graded_rating"] = gr; keep -= gr
    b["KEPT"] = keep
    return b


def stratified(rng: random.Random, frz: pd.DataFrame, ids: list[str],
               key, want: int, floor: int = 1) -> list[str]:
    """Proportional allocation of `want` picks across the strata `key(id)`,
    each non-empty stratum guaranteed at least `floor`."""
    strata: dict = {}
    for cid in ids:
        strata.setdefault(key(cid), []).append(cid)
    for v in strata.values():
        rng.shuffle(v)
    total = len(ids)
    alloc = {k: max(floor, round(want * len(v) / total)) for k, v in strata.items()}
    # trim / pad to exactly `want`
    order = sorted(strata, key=lambda k: -len(strata[k]))
    while sum(alloc.values()) > want:
        for k in reversed(order):
            if alloc[k] > floor:
                alloc[k] -= 1
                if sum(alloc.values()) == want:
                    break
    while sum(alloc.values()) < want:
        for k in order:
            if alloc[k] < len(strata[k]):
                alloc[k] += 1
                if sum(alloc.values()) == want:
                    break
    picked = []
    for k in order:
        picked += strata[k][:alloc[k]]
    return picked


def reason_for(group: str, frz_row) -> tuple[str, str]:
    """(short label, full reason) for a filter group. Full reason is hidden until reveal."""
    if group == "attribution":
        return ("attribution axis",
                "Attribution axis. The fact-checker was judging who said something, "
                "not whether the underlying thing is true. Our system checks the "
                "content, so this is a mismatch it cannot be scored on.")
    if group == "graded_rating":
        st = SUBTYPE_NAME.get(frz_row.rating_subtype, frz_row.rating_subtype)
        return ("graded rating",
                f"Graded rating. Harmonised as {st}, not a clear true or clear false, "
                "so it has no clean binary answer to fit against.")
    if group == "shape_resolution":
        return ("claim shape",
                f"Claim shape. Could not be resolved to a standalone claim "
                f"(resolution status {frz_row.resolution_status}); the text does not "
                "stand on its own without the article.")
    if group == "shape_unresolved_referent":
        return ("claim shape",
                "Claim shape. Unresolved referent. The claim points at a this or that "
                "that is never pinned down, so it cannot be checked as written.")
    if group == "shape_media":
        return ("claim shape",
                "Claim shape. The real claim lives in an attached image or video, not "
                "in the words, so the text alone is not what the fact-check judged.")
    return ("kept", "Kept. No filter fired. This claim is in the clean fitting subset.")


def build_rows(frz: pd.DataFrame, buckets: dict[str, set]) -> list[dict]:
    rng = random.Random(SEED)

    def by_truth(cid):
        return "true" if frz.loc[cid].veracity >= 4 else "false"

    att = stratified(rng, frz, list(buckets["attribution"]), by_truth, 50)
    grd = stratified(rng, frz, list(buckets["graded_rating"]),
                     lambda c: frz.loc[c].rating_subtype, 50)
    shape_ids = (list(buckets["shape_resolution"]) + list(buckets["shape_unresolved_referent"])
                 + list(buckets["shape_media"]))
    shape_group = {}
    for g in ("shape_resolution", "shape_unresolved_referent", "shape_media"):
        for c in buckets[g]:
            shape_group[c] = g
    shp = stratified(rng, frz, shape_ids, lambda c: shape_group[c], 25)
    kept = stratified(rng, frz, list(buckets["KEPT"]), by_truth, 25)

    chosen = ([(c, "attribution") for c in att]
              + [(c, "graded_rating") for c in grd]
              + [(c, shape_group[c]) for c in shp]
              + [(c, "KEPT") for c in kept])
    rng.shuffle(chosen)

    rows = []
    for i, (cid, group) in enumerate(chosen, 1):
        fr = frz.loc[cid]
        short, full = reason_for(group, fr)
        rows.append({
            "id": f"R{i:03d}",
            "claim": fr.claim_text,
            "original_rating": str(fr.original_rating),
            "veracity": VERACITY_NAME[int(fr.veracity)],
            "judged_axis": fr.judged_axis_llm.replace("_", " "),
            "resolution_status": fr.resolution_status or "ok",
            "filter_group": "kept" if group == "KEPT" else (
                "attribution" if group == "attribution" else (
                    "graded rating" if group == "graded_rating" else "claim shape")),
            "reason_full": full,
            "url": cid,
        })
    return rows


HEADER = """We are cutting the frozen fc_gold population (3,280 claims) down to a clean
binary subset (2,057) to fit the Truth Odds weights on. Three filters do the cutting and
none of them has ever been checked by a person. This page is that check. Read each claim
against its fact-check article and say whether it belongs in the clean set. The reason a
claim was dropped is hidden so you judge it blind; reveal it once you have decided."""

FILTER_NOTES = [
    ("Attribution axis (761 dropped)",
     "The fact-checker was judging who said something, not whether the underlying "
     "thing is true. Our system checks the content, so an attribution claim is a "
     "mismatch it cannot be scored on."),
    ("Graded rating (418 dropped)",
     "We keep only claims rated clearly true or clearly false. Anything graded mostly "
     "true, mostly false, or unprovable is dropped because it has no clean binary "
     "answer to fit against."),
    ("Claim shape (44 dropped)",
     "We drop claims whose text does not stand on its own: it needs the article to make "
     "sense, points at an unresolved this or that, or its real claim lives in an image "
     "or video rather than the words."),
]


def render_html(rows: list[dict]) -> str:
    data = json.dumps(rows, ensure_ascii=False)
    notes = "\n".join(
        f'<div class="note"><div class="note-h">{html.escape(t)}</div>'
        f'<div class="note-b">{html.escape(b)}</div></div>'
        for t, b in FILTER_NOTES)
    return f"""<title>fc_gold Filter Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --ground:#ffffff; --paper:#fafaf9; --raise:#f2f1ef; --ink:#191817; --sub:#57544f;
  --faint:#8a867f; --line:#e4e2de; --line2:#d3d0ca; --pick:#191817; --pickink:#ffffff;
  --shadow:0 1px 2px rgba(20,18,16,.06),0 1px 1px rgba(20,18,16,.04);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#141312; --paper:#1b1a18; --raise:#252320; --ink:#ecebe8; --sub:#a8a49d;
    --faint:#7c7871; --line:#2c2a27; --line2:#3a3733; --pick:#ecebe8; --pickink:#141312;
    --shadow:0 1px 2px rgba(0,0,0,.4);
  }}
}}
:root[data-theme="dark"] {{
  --ground:#141312; --paper:#1b1a18; --raise:#252320; --ink:#ecebe8; --sub:#a8a49d;
  --faint:#7c7871; --line:#2c2a27; --line2:#3a3733; --pick:#ecebe8; --pickink:#141312;
  --shadow:0 1px 2px rgba(0,0,0,.4);
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif; line-height:1.5;
  -webkit-font-smoothing:antialiased; }}
a {{ color:inherit; }}
.mono {{ font-family:"IBM Plex Mono",ui-monospace,monospace; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:0 24px; }}

.bar {{ position:sticky; top:0; z-index:20; background:var(--ground);
  border-bottom:1px solid var(--line); }}
.bar .wrap {{ display:flex; align-items:center; gap:16px; flex-wrap:wrap;
  padding-block:12px; }}
.bar h1 {{ font-size:15px; font-weight:600; margin:0; letter-spacing:-.01em; }}
.bar h1 .tag {{ font-family:"IBM Plex Mono",monospace; color:var(--faint);
  font-weight:500; font-size:12px; margin-left:8px; }}
.spacer {{ flex:1 1 auto; }}
.count {{ font-family:"IBM Plex Mono",monospace; font-size:13px; color:var(--sub);
  font-variant-numeric:tabular-nums; }}
.count b {{ color:var(--ink); font-weight:600; }}
.btn {{ font:inherit; font-size:13px; border:1px solid var(--line2);
  background:var(--paper); color:var(--ink); padding:7px 13px; border-radius:7px;
  cursor:pointer; display:inline-flex; align-items:center; gap:7px; }}
.btn:hover {{ border-color:var(--faint); }}
.btn:focus-visible {{ outline:2px solid var(--ink); outline-offset:2px; }}
.btn[aria-pressed="true"] {{ background:var(--pick); color:var(--pickink);
  border-color:var(--pick); }}

.intro {{ padding-block:32px 8px; }}
.intro p {{ max-width:68ch; color:var(--sub); font-size:15px; margin:0 0 20px; }}
.legend {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:10px;
  overflow:hidden; margin-bottom:8px; }}
.note {{ background:var(--paper); padding:15px 16px; }}
.note-h {{ font-family:"IBM Plex Mono",monospace; font-size:12px; font-weight:600;
  letter-spacing:.02em; margin-bottom:6px; }}
.note-b {{ font-size:13px; color:var(--sub); line-height:1.45; }}

.rows {{ padding-block:20px 80px; display:flex; flex-direction:column; gap:14px; }}
.card {{ background:var(--paper); border:1px solid var(--line);
  border-radius:12px; box-shadow:var(--shadow); overflow:hidden; }}
.card.done {{ border-color:var(--line2); }}
.card.done::before {{ content:""; }}
.chead {{ display:flex; align-items:baseline; gap:12px; padding:14px 18px 0; }}
.rid {{ font-family:"IBM Plex Mono",monospace; font-size:13px; font-weight:600;
  color:var(--faint); }}
.status-dot {{ margin-left:auto; font-family:"IBM Plex Mono",monospace; font-size:11px;
  color:var(--faint); letter-spacing:.03em; }}
.claim {{ padding:8px 18px 4px; font-size:17px; line-height:1.45; }}
.meta {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
  background:var(--line); margin:12px 0 0; border-top:1px solid var(--line); }}
.mcell {{ background:var(--paper); padding:10px 18px; }}
.mk {{ font-family:"IBM Plex Mono",monospace; font-size:10.5px; letter-spacing:.05em;
  text-transform:uppercase; color:var(--faint); margin-bottom:3px; }}
.mv {{ font-size:13.5px; color:var(--ink); }}
.foot {{ display:flex; align-items:center; gap:16px; flex-wrap:wrap;
  padding:13px 18px; border-top:1px solid var(--line); }}
.src a {{ font-family:"IBM Plex Mono",monospace; font-size:12.5px; color:var(--sub);
  text-decoration:none; border-bottom:1px solid var(--line2); padding-bottom:1px; }}
.src a:hover {{ color:var(--ink); border-color:var(--ink); }}

.reason {{ padding:0 18px; max-height:0; overflow:hidden;
  transition:max-height .18s ease, padding .18s ease; }}
.reason.show {{ max-height:200px; padding:12px 18px; }}
.reason-in {{ background:var(--raise); border-radius:8px; padding:11px 14px;
  font-size:13px; color:var(--sub); border:1px solid var(--line); }}
.reason-in .rlab {{ font-family:"IBM Plex Mono",monospace; font-weight:600;
  color:var(--ink); font-size:11px; text-transform:uppercase; letter-spacing:.04em;
  margin-right:8px; }}

.verdict {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }}
.seg {{ display:inline-flex; border:1px solid var(--line2); border-radius:8px;
  overflow:hidden; }}
.seg label {{ font-size:12.5px; padding:7px 13px; cursor:pointer; color:var(--sub);
  border-right:1px solid var(--line2); user-select:none; white-space:nowrap; }}
.seg label:last-child {{ border-right:0; }}
.seg input {{ position:absolute; opacity:0; width:0; height:0; }}
.seg input:checked + label {{ background:var(--pick); color:var(--pickink); }}
.seg input:focus-visible + label {{ outline:2px solid var(--ink); outline-offset:-2px; }}
.notefield {{ flex:1 1 220px; min-width:180px; font:inherit; font-size:13px;
  background:var(--ground); color:var(--ink); border:1px solid var(--line2);
  border-radius:7px; padding:7px 10px; resize:vertical; min-height:36px; }}
.notefield:focus-visible {{ outline:2px solid var(--ink); outline-offset:1px; }}

.export {{ margin-top:14px; }}
.export textarea {{ width:100%; min-height:220px; font-family:"IBM Plex Mono",monospace;
  font-size:12px; background:var(--ground); color:var(--ink);
  border:1px solid var(--line2); border-radius:8px; padding:12px; }}
.hint {{ font-size:12.5px; color:var(--faint); margin:8px 0 0; }}

@media (max-width:720px) {{
  .meta {{ grid-template-columns:repeat(2,1fr); }}
  .legend {{ grid-template-columns:1fr; }}
}}
@media (prefers-reduced-motion: reduce) {{ .reason {{ transition:none; }} }}
</style>

<div class="bar">
  <div class="wrap">
    <h1>fc_gold Filter Review <span class="tag">150 claims</span></h1>
    <span class="spacer"></span>
    <span class="count" id="count"><b>0</b> / 150 judged</span>
    <button class="btn" id="reveal" aria-pressed="false">Reveal filter reasons</button>
    <button class="btn" id="exportBtn">Export JSON</button>
  </div>
</div>

<div class="wrap intro">
  <p>{html.escape(HEADER)}</p>
  <div class="legend">{notes}</div>
  <p class="hint">Judge each row on one question: should this claim be in the clean set we
  fit the weights on? Pick <b>filter right</b> if the drop (or, for a kept claim, the keep)
  was correct, <b>filter wrong</b> if it should be kept, or <b>unsure</b>.</p>
</div>

<div class="wrap"><div class="rows" id="rows"></div></div>

<div class="wrap export" id="exportBox" hidden>
  <textarea id="exportText" readonly></textarea>
  <p class="hint">Select all and copy, or use Copy. Paste this back to me and I will pull the verdicts in.
    <button class="btn" id="copyBtn" style="margin-left:8px;">Copy</button></p>
</div>

<script>
const DATA = {data};
const KEY = "fc_gold_filter_review_v1";
let store = {{}};
try {{ store = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ store = {{}}; }}

function save() {{
  try {{ localStorage.setItem(KEY, JSON.stringify(store)); }} catch (e) {{}}
}}
function judged() {{ return DATA.filter(r => store[r.id] && store[r.id].verdict).length; }}
function refreshCount() {{
  document.getElementById("count").innerHTML = "<b>" + judged() + "</b> / 150 judged";
}}
const VOPTS = [["right","filter right"],["wrong","filter wrong: should be kept"],["unsure","unsure"]];

function esc(s) {{ const d = document.createElement("div"); d.textContent = s == null ? "" : s; return d.innerHTML; }}

function card(r) {{
  const s = store[r.id] || {{}};
  const el = document.createElement("div");
  el.className = "card" + (s.verdict ? " done" : "");
  el.id = "card-" + r.id;
  const segs = VOPTS.map(([v,lab]) => {{
    const id = r.id + "-" + v;
    return '<input type="radio" name="' + r.id + '" id="' + id + '" value="' + v + '"'
      + (s.verdict === v ? " checked" : "") + '>'
      + '<label for="' + id + '">' + lab + '</label>';
  }}).join("");
  el.innerHTML =
    '<div class="chead"><span class="rid">' + r.id + '</span>'
      + '<span class="status-dot" id="dot-' + r.id + '">' + (s.verdict ? "judged" : "") + '</span></div>'
    + '<div class="claim">' + esc(r.claim) + '</div>'
    + '<div class="meta">'
      + mcell("fact-checker rating", r.original_rating)
      + mcell("harmonised", r.veracity)
      + mcell("judged axis", r.judged_axis)
      + mcell("resolution", r.resolution_status)
    + '</div>'
    + '<div class="reason" id="reason-' + r.id + '"><div class="reason-in">'
      + '<span class="rlab">' + esc(r.filter_group) + '</span>' + esc(r.reason_full) + '</div></div>'
    + '<div class="foot">'
      + '<span class="src"><a href="' + esc(r.url) + '" target="_blank" rel="noopener">open fact-check &#8599;</a></span>'
      + '<span class="spacer" style="flex:1"></span>'
      + '<div class="verdict"><div class="seg">' + segs + '</div>'
      + '<textarea class="notefield" id="note-' + r.id + '" placeholder="note (optional)">' + esc(s.note || "") + '</textarea>'
      + '</div>'
    + '</div>';
  return el;
}}
function mcell(k, v) {{
  return '<div class="mcell"><div class="mk">' + k + '</div><div class="mv">' + esc(v) + '</div></div>';
}}

const rowsEl = document.getElementById("rows");
DATA.forEach(r => rowsEl.appendChild(card(r)));

rowsEl.addEventListener("change", e => {{
  if (e.target.type === "radio") {{
    const id = e.target.name;
    store[id] = store[id] || {{}};
    store[id].verdict = e.target.value;
    save(); refreshCount();
    document.getElementById("card-" + id).classList.add("done");
    document.getElementById("dot-" + id).textContent = "judged";
  }}
}});
rowsEl.addEventListener("input", e => {{
  if (e.target.classList.contains("notefield")) {{
    const id = e.target.id.slice(5);
    store[id] = store[id] || {{}};
    store[id].note = e.target.value;
    save();
  }}
}});

const revealBtn = document.getElementById("reveal");
let revealed = false;
revealBtn.addEventListener("click", () => {{
  revealed = !revealed;
  revealBtn.setAttribute("aria-pressed", revealed ? "true" : "false");
  revealBtn.textContent = revealed ? "Hide filter reasons" : "Reveal filter reasons";
  document.querySelectorAll(".reason").forEach(x => x.classList.toggle("show", revealed));
}});

const exportBox = document.getElementById("exportBox");
document.getElementById("exportBtn").addEventListener("click", () => {{
  const out = DATA.map(r => ({{
    id: r.id, verdict: (store[r.id] || {{}}).verdict || null,
    note: (store[r.id] || {{}}).note || "", filter: r.filter_group,
    veracity: r.veracity, judged_axis: r.judged_axis,
    fact_checker_rating: r.original_rating, url: r.url, claim: r.claim
  }}));
  document.getElementById("exportText").value = JSON.stringify(
    {{judged: judged(), total: 150, verdicts: out}}, null, 2);
  exportBox.hidden = false;
  exportBox.scrollIntoView({{behavior: "smooth"}});
}});
document.getElementById("copyBtn").addEventListener("click", () => {{
  const t = document.getElementById("exportText");
  t.select();
  try {{ navigator.clipboard.writeText(t.value); }} catch (e) {{ document.execCommand("copy"); }}
}});

refreshCount();
</script>
"""


def main() -> None:
    frz = build_frame()
    buckets = reconstruct(frz)
    rows = build_rows(frz, buckets)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "review.html").write_text(render_html(rows), encoding="utf-8")
    with (OUTDIR / "review.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "id", "claim", "original_rating", "veracity", "judged_axis",
            "resolution_status", "filter_group", "reason_full", "url"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # provenance / sanity summary
    counts = {"attribution": 0, "graded rating": 0, "claim shape": 0, "kept": 0}
    for r in rows:
        counts[r["filter_group"]] += 1
    print(f"buckets (full): " + ", ".join(
        f"{k} {len(v)}" for k, v in buckets.items()))
    print(f"sample (150): {counts}")
    print(f"wrote {OUTDIR / 'review.html'} and {OUTDIR / 'review.csv'}")


if __name__ == "__main__":
    main()
