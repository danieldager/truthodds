"""Build the annotatable dev-50 verify-loop review HTML from the traced run
(scripts/run_dev50_verify_review.py -> eval/data/survey_claims/dev50_verify_trace/).

Output: reports/dev50_verify_review_<date>.html — one section per post: header
(verdict, flags), claims with per-claim comment boxes, then the full step timeline
(OPEN / search+ranked hits / TRIAGE / scrape+drops / READ / evidence / STEP) with
every LLM call's full prompt + raw response in collapsibles. Comments persist in
localStorage; the export button downloads them as JSON.

Usage (from src/): uv run python -m scripts.build_dev50_review
"""
from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from datetime import date
from pathlib import Path

from pipeline.verify_tweet_claims import rank_hits

_SRC = Path(__file__).resolve().parent.parent
TRACE_DIR = _SRC / "eval" / "data" / "survey_claims" / "dev50_verify_trace"
OUT = _SRC / "reports" / f"dev50_verify_review_{date.today().isoformat()}.html"

NS = ""

STATUS_CLS = {"supported": "s-sup", "refuted": "s-ref", "conflicting": "s-con",
              "unsupported": "s-uns", "open": "s-uns"}


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def load(trace_dir: Path) -> list[dict]:
    order = {p["post_id"]: i for i, p in enumerate(
        json.loads((trace_dir / "sample_posts.json").read_text()))}
    recs = []
    for shard in sorted(trace_dir.glob("results-*.jsonl")):
        for ln in shard.open():
            recs.append(json.loads(ln))
    recs.sort(key=lambda r: order.get(r["post_id"], 999))
    return recs


def mark_evidence(text: str) -> str:
    """<<cited>> -> <mark>, after escaping."""
    t = esc(text)
    t = t.replace("&lt;&lt;", "<mark>").replace("&gt;&gt;", "</mark>")
    return t


def collapsible(title: str, body_html: str, cls: str = "") -> str:
    return f'<details class="{cls}"><summary>{title}</summary><div class="dbody">{body_html}</div></details>'


def llm_details(t: dict) -> str:
    body = (f'<div class="plabel">USER PROMPT</div><pre>{esc(t["user"])}</pre>'
            f'<div class="plabel">RESPONSE (parsed)</div>'
            f'<pre>{esc(json.dumps(t["response"], ensure_ascii=False, indent=1))}</pre>')
    return collapsible(f'full prompt + response &middot; {t["secs"]}s', body, "raw")


def verdict_badge(v: dict | None) -> str:
    if not v:
        return '<span class="badge b-miss">NO VERDICT</span>'
    ver = v.get("veracity")
    nudge = v.get("nudge")
    cls = "b-nudge" if nudge else "b-pass"
    return (f'<span class="badge {cls}">veracity {ver} &middot; {esc(v.get("misinfo_type"))}'
            f' &middot; {"NUDGE" if nudge else "pass"}</span>')


def status_badge(st: str) -> str:
    return f'<span class="st {STATUS_CLS.get(st, "s-uns")}">{esc(st)}</span>'


# =============================================================================
# Per-post section
# =============================================================================

def round_trace_items(trace: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    """Split flat trace at search events: [pre-round items], [per-round items]."""
    pre, rounds_items, cur = [], [], None
    for t in trace:
        if t["kind"] in ("serper", "exa") or t.get("label") == "exa-query":
            if t.get("label") == "exa-query":
                cur = [t]
                rounds_items.append(cur)
                continue
            if cur is None or any(x["kind"] in ("serper", "exa") for x in cur):
                cur = []
                rounds_items.append(cur)
            cur.append(t)
        elif cur is None:
            pre.append(t)
        else:
            cur.append(t)
    return pre, rounds_items


def render_search(t: dict, rmeta: dict) -> str:
    picked = set()
    tri = rmeta.get("triage") or {}
    for i in (tri.get("read") or []):
        picked.add(i)
    rows = []
    # triage picks and evidence R-ids index the RANKED order, so display that order
    # (rank_hits is deterministic; re-ranking the recorded hits reproduces runtime order)
    for i, h in enumerate(rank_hits(t["hits"]), 1):
        pick = ' <span class="pick">READ PICK</span>' if i in picked else ""
        rows.append(f'<tr><td>{i}</td><td><a href="{esc(h.get("url"))}" target="_blank">'
                    f'{esc((h.get("url") or "")[:80])}</a>{pick}</td>'
                    f'<td>{esc(h.get("date") or "")}</td>'
                    f'<td class="snip">{esc((h.get("snippet") or "")[:300])}</td></tr>')
    prov = t["kind"].upper()
    return (f'<div class="ev-block"><span class="stepname">{prov} SEARCH</span> '
            f'<code class="q">{esc(t["query"])}</code> &middot; {len(t["hits"])} hits '
            f'&middot; {t["secs"]}s'
            + collapsible("ranked results (post-NewsGuard-tier order as fed to triage)",
                          f'<table class="hits">{"".join(rows)}</table>', "raw")
            + "</div>")


def render_round(rnd_items: list[dict], rmeta: dict, evidence: list[dict],
                 claims: list[dict], prior_ledger: dict) -> str:
    n_round = rmeta["round"]
    out = [f'<div class="round"><div class="rhead">ROUND {n_round} '
           f'<span class="prov">{esc(rmeta["provider"])}</span></div>']
    ev_this = [e for e in evidence if e["round"] == n_round]
    for t in rnd_items:
        if t["kind"] in ("serper", "exa"):
            out.append(render_search(t, rmeta))
        elif t["kind"] == "scrape":
            ok = "ok" if t["ok"] else '<span class="miss">FAILED</span>'
            out.append(f'<div class="mini">scrape {ok} &middot; {t["chars"]} chars &middot; '
                       f'{t["secs"]}s &middot; <a href="{esc(t["url"])}" target="_blank">'
                       f'{esc(t["url"][:90])}</a></div>')
        elif t["kind"] == "llm" and t["label"].startswith("triage"):
            r = t["response"]
            out.append(f'<div class="ev-block"><span class="stepname">TRIAGE</span> '
                       f'read picks: <code>{esc(r.get("read"))}</code> &middot; '
                       f'snippet evidence: <code>{esc(r.get("snippet_evidence"))}</code>'
                       + llm_details(t) + "</div>")
        elif t["kind"] == "llm" and t["label"] == "exa-query":
            out.append(f'<div class="ev-block"><span class="stepname">EXA QUERY COMPOSE</span> '
                       f'<code class="q">{esc(t["response"].get("query"))}</code>'
                       + llm_details(t) + "</div>")
        elif t["kind"] == "llm" and t["label"].startswith("read-"):
            doc_id = t["label"].split("-", 1)[1]
            evs = []
            for d in (t["response"].get("docs") or []):
                rep = ' <span class="flag">REPUBLICATION FLAG</span>' if d.get("republication") else ""
                for ev in (d.get("evidence") or []):
                    evs.append(f'<div class="mini">claim {esc(ev.get("claim_id"))} &middot; '
                               f'stance <b>{esc(ev.get("stance"))}</b> &middot; '
                               f'segs {esc(ev.get("segs"))}{rep}</div>')
            out.append(f'<div class="ev-block"><span class="stepname">READ {esc(doc_id)}</span>'
                       + ("".join(evs) or '<div class="mini">no evidence in this doc</div>')
                       + llm_details(t) + "</div>")
        elif t["kind"] == "llm" and t["label"].startswith("step"):
            led = t["response"].get("ledger") or []
            rows = "".join(
                f'<tr><td>{esc(l.get("claim_id"))}</td><td>{status_badge(l.get("status", "?"))}</td>'
                f'<td class="snip">{esc(claims[l["claim_id"] - 1]["c"][:120]) if isinstance(l.get("claim_id"), int) and 0 < l["claim_id"] <= len(claims) else ""}</td></tr>'
                for l in led)
            act = t["response"].get("action")
            nq = t["response"].get("query")
            v = t["response"].get("verdict")
            vs = ""
            if v:
                vs = (f'<div class="mini">verdict: {verdict_badge(v)}<br>'
                      f'<i>{esc(v.get("justification"))}</i></div>')
            out.append(f'<div class="ev-block"><span class="stepname">STEP</span> '
                       f'action <b>{esc(act)}</b>'
                       + (f' &middot; next query <code class="q">{esc(nq)}</code>' if nq and act == "search" else "")
                       + f'<table class="hits">{rows}</table>' + vs + llm_details(t) + "</div>")
    # dropped docs + resolved evidence for the round
    for d in (rmeta.get("dropped") or []):
        out.append(f'<div class="mini flag">DROPPED ({esc(d["reason"])}): '
                   f'{esc(d["url"][:90])}</div>')
    if ev_this:
        rows = []
        for e in ev_this:
            tag = ' <span class="flag">SNIPPET ONLY</span>' if e.get("snippet_only") else ""
            rep = ' <span class="flag">REPUB FLAG</span>' if e.get("republication") else ""
            rows.append(f'<div class="evrow"><b>{esc(e["src"])}</b> {esc(e["domain"])} &middot; '
                        f'{esc(e.get("date") or "no date")} &middot; claim {e["claim_id"]} &middot; '
                        f'stance {esc(e["stance"])}{tag}{rep}'
                        f'<div class="evtext">{mark_evidence(e["text"])}</div></div>')
        out.append(collapsible(f"resolved evidence entries fed to STEP ({len(ev_this)})",
                               "".join(rows), "evd"))
    out.append("</div>")
    return "".join(out)


def render_post(rec: dict, idx: int) -> str:
    res = rec["result"]
    pid = rec["post_id"]
    claims = res["claims"]
    ledger = res["ledger"]
    trace = res.get("trace") or []
    rounds = res["rounds"]
    flags = []
    if len(rounds) == 1:
        flags.append('<span class="flag ok">1-ROUND CONCLUDE</span>')
    if any(r["provider"] == "exa" for r in rounds):
        early = any(r["provider"] == "exa" and r["round"] <= 3 for r in rounds)
        flags.append(f'<span class="flag">EXA ESCALATION{" (redundant query)" if early else " (budget)"}</span>')
    if res.get("coerced_open"):
        flags.append(f'<span class="flag warn">COERCED OPEN -&gt; unsupported: {res["coerced_open"]}</span>')
    if any(s == "unsupported" for s in ledger.values()):
        flags.append('<span class="flag warn">HAS UNSUPPORTED</span>')

    ng = res.get("ng_score")
    hdr = (f'<div class="phead" id="post-{pid}"><span class="pnum">#{idx}</span> '
           f'<b>@{esc(res["handle"])}</b> ({esc(res["domain"])}) &middot; {esc(res["date"])} '
           f'&middot; NG {esc(ng if ng is not None else "?")} &middot; '
           f'<a href="{esc(res.get("url"))}" target="_blank">post</a> &middot; '
           f'{rec["elapsed_s"]}s &middot; {len(rounds)} round(s) {verdict_badge(res["verdict"])} '
           + " ".join(flags) + "</div>")

    post_text = f'<div class="ptext">{esc(res["text"])}</div>'

    v = res["verdict"] or {}
    just = (f'<div class="just"><b>Justification:</b> {esc(v.get("justification"))}</div>'
            if v.get("justification") else "")

    crows = []
    for i, c in enumerate(claims, 1):
        st = ledger.get(str(i), "?")
        crows.append(
            f'<div class="claim"><div class="cline">{i}. {status_badge(st)} {esc(c["c"])}</div>'
            f'<textarea class="cmt" data-key="dev50rev::{NS}{pid}::c{i}" '
            f'placeholder="comment on claim {i}..."></textarea></div>')
    crows.append(f'<textarea class="cmt pcmt" data-key="dev50rev::{NS}{pid}::post" '
                 f'placeholder="post-level comment..."></textarea>')

    pre, rnd_items = round_trace_items(trace)
    body = []
    for t in pre:
        if t["kind"] == "llm" and t["label"] == "open":
            r = t["response"]
            body.append(f'<div class="ev-block"><span class="stepname">OPEN</span> '
                        f'targets <code>{esc(r.get("targets"))}</code> &middot; query '
                        f'<code class="q">{esc(r.get("query"))}</code>' + llm_details(t) + "</div>")
    for i, items in enumerate(rnd_items):
        if i < len(rounds):
            body.append(render_round(items, rounds[i], res["evidence"], claims,
                                     rounds[i - 1]["ledger"] if i else {}))
    for t in trace:
        if t["kind"] == "llm" and t["label"] == "verdict-retry":
            body.append(f'<div class="ev-block"><span class="stepname">VERDICT RE-ASK</span> '
                        f'(STEP concluded without a verdict object)' + llm_details(t) + "</div>")

    return (f'<section class="post">{hdr}{post_text}{just}'
            f'<div class="claims">{"".join(crows)}</div>'
            + collapsible(f"step timeline ({len(trace)} events)", "".join(body), "timeline")
            + "</section>")


# =============================================================================
# Summary + concern panel
# =============================================================================

def build_summary(recs: list[dict]) -> str:
    results = [r["result"] for r in recs if r.get("ok")]
    n = len(results)
    vhist: dict = {}
    for res in results:
        v = (res["verdict"] or {}).get("veracity")
        vhist[v] = vhist.get(v, 0) + 1
    rhist: dict = {}
    for res in results:
        rhist[len(res["rounds"])] = rhist.get(len(res["rounds"]), 0) + 1
    nudges = sum(1 for res in results if (res["verdict"] or {}).get("nudge"))
    one_round = [res for res in results if len(res["rounds"]) == 1]
    coerced = [res for res in results if res.get("coerced_open")]
    unsup = [res for res in results if any(s == "unsupported" for s in res["ledger"].values())]
    exa_early = [res for res in results
                 if any(r["provider"] == "exa" and r["round"] <= 3 for r in res["rounds"])]
    exa_any = [res for res in results if any(r["provider"] == "exa" for r in res["rounds"])]

    lat: dict[str, list[float]] = {}
    for res in results:
        for t in res.get("trace") or []:
            if t["kind"] == "llm":
                key = re.sub(r"-.*", "", t["label"])
                lat.setdefault(key, []).append(t["secs"])
    lat_rows = "".join(
        f'<tr><td>{k}</td><td>{len(v)}</td><td>{statistics.median(v):.1f}s</td>'
        f'<td>{max(v):.1f}s</td></tr>'
        for k, v in sorted(lat.items()))

    def links(rs):
        return " ".join(f'<a href="#post-{res["id"].split("/")[-1] if "/" in str(res["id"]) else res["id"]}">'
                        f'@{esc(res["handle"])}</a>' for res in rs) or "&mdash;"

    vrow = " &middot; ".join(f"{k}: <b>{vhist[k]}</b>" for k in sorted(vhist, key=lambda x: (x is None, x)))
    rrow = " &middot; ".join(f"{k} round(s): <b>{rhist[k]}</b>" for k in sorted(rhist))

    return f"""
<div class="summary">
<h2>Run summary — {n} posts</h2>
<p>Veracity: {vrow} &nbsp;|&nbsp; nudges: <b>{nudges}</b>/{n}</p>
<p>Rounds: {rrow}</p>
<table class="hits"><tr><th>call</th><th>n</th><th>median</th><th>max</th></tr>{lat_rows}</table>
<h2>Review focus (your four concerns)</h2>
<ol class="concerns">
<li><b>Early exit.</b> {len(one_round)}/{n} posts concluded in a single round: {links(one_round)}</li>
<li><b>Unverifiable claims.</b> {len(unsup)} posts finished with an <i>unsupported</i> claim: {links(unsup)}.
Of these, {len(coerced)} were coerced from <i>open</i> at the round budget (system never chose
"unsupported" itself — it wanted to keep searching): {links(coerced)}</li>
<li><b>Query redundancy.</b> {len(exa_any)} posts escalated to Exa; {len(exa_early)} of those escalated
EARLY because STEP proposed a near-duplicate query (token overlap &ge; 0.75 with a prior query):
{links(exa_early)}. In each post's timeline, compare STEP's "next query" with the prior queries.</li>
<li><b>STEP load.</b> Median STEP latency vs the other calls is in the table above — STEP is doing
ledger re-judging + control + (on conclude) verdict + justification in one call. Look at the STEP
collapsibles for signs of overload (sloppy ledger reasoning next to verbose justifications).</li>
</ol>
<p class="mini">Comments persist in this browser (localStorage). Use the export button (bottom right)
to download them as JSON and send back.</p>
</div>"""


def build_index(recs: list[dict]) -> str:
    rows = []
    for i, r in enumerate(recs, 1):
        res = r["result"]
        v = res["verdict"] or {}
        rows.append(f'<tr><td>{i}</td><td><a href="#post-{r["post_id"]}">@{esc(res["handle"])}</a></td>'
                    f'<td>{esc(res["date"])}</td><td>{len(res["rounds"])}</td>'
                    f'<td>{esc(v.get("veracity"))}</td><td>{esc(v.get("misinfo_type"))}</td>'
                    f'<td>{"NUDGE" if v.get("nudge") else ""}</td>'
                    f'<td class="snip">{esc(res["text"][:110])}</td></tr>')
    return ('<details class="idx" open><summary>Post index</summary><table class="hits">'
            '<tr><th>#</th><th>handle</th><th>date</th><th>rounds</th><th>ver</th>'
            '<th>type</th><th></th><th>post</th></tr>' + "".join(rows) + "</table></details>")


CSS = """
body{font-family:-apple-system,'Segoe UI',sans-serif;margin:0 auto;max-width:1100px;
padding:20px 24px 120px;color:#1a1a1a;background:#fafafa;font-size:14px;line-height:1.45}
h1{font-size:20px}h2{font-size:16px;margin:18px 0 6px}
.summary{background:#fff;border:1px solid #ddd;padding:12px 16px;margin-bottom:16px}
.concerns li{margin-bottom:8px}
section.post{background:#fff;border:1px solid #ddd;margin:14px 0;padding:12px 16px}
.phead{font-size:14px;margin-bottom:6px}
.pnum{color:#888;font-weight:600;margin-right:4px}
.ptext{white-space:pre-wrap;background:#f4f4f4;border-left:3px solid #bbb;padding:8px 10px;margin:6px 0}
.just{margin:6px 0;color:#333}
.claims{margin:8px 0}
.claim{margin:6px 0}
.cline{margin-bottom:2px}
.cmt{width:100%;box-sizing:border-box;min-height:30px;font:12px/1.4 -apple-system,sans-serif;
border:1px dashed #bbb;padding:4px 6px;background:#fffef5;resize:vertical}
.pcmt{min-height:40px;margin-top:6px}
.badge{padding:1px 8px;border-radius:3px;font-size:12px;font-weight:600}
.b-nudge{background:#8b1a1a;color:#fff}.b-pass{background:#2e5e2e;color:#fff}
.b-miss{background:#666;color:#fff}
.st{padding:0 6px;border-radius:3px;font-size:12px;font-weight:600}
.s-sup{background:#e2efe2;color:#2e5e2e}.s-ref{background:#f5dede;color:#8b1a1a}
.s-con{background:#f0e8d8;color:#7a5c1e}.s-uns{background:#e8e8e8;color:#555}
.flag{background:#f0e8d8;color:#7a5c1e;padding:0 6px;border-radius:3px;font-size:11px;font-weight:600}
.flag.warn{background:#f5dede;color:#8b1a1a}.flag.ok{background:#e2efe2;color:#2e5e2e}
.pick{background:#1a1a1a;color:#fff;padding:0 5px;border-radius:3px;font-size:10px}
.round{border-left:3px solid #999;margin:10px 0;padding:4px 0 4px 12px}
.rhead{font-weight:700;font-size:13px;margin-bottom:4px}
.prov{color:#666;font-weight:400}
.stepname{font-weight:700;font-size:12px;letter-spacing:.4px}
.ev-block{margin:8px 0;padding:6px 8px;background:#f7f7f7;border:1px solid #e5e5e5}
.mini{font-size:12px;color:#444;margin:3px 0}
.miss{color:#8b1a1a;font-weight:700}
code.q{background:#eee;padding:1px 5px;font-size:12px}
details{margin:4px 0}summary{cursor:pointer;font-size:12px;color:#555}
details.raw pre{white-space:pre-wrap;background:#22262b;color:#d6d9dd;padding:8px;
font-size:11px;max-height:480px;overflow:auto}
.plabel{font-size:10px;font-weight:700;color:#888;margin-top:6px}
table.hits{border-collapse:collapse;width:100%;font-size:12px;margin:4px 0}
table.hits td,table.hits th{border:1px solid #e0e0e0;padding:3px 6px;vertical-align:top;text-align:left}
.snip{color:#555}
.evrow{margin:8px 0;padding:6px;border:1px solid #e5e5e5;background:#fff;font-size:12px}
.evtext{margin-top:4px;color:#333}
mark{background:#ffe9a8;padding:0 1px}
details.idx table{background:#fff}
#exportBtn{position:fixed;bottom:24px;right:24px;background:#1a1a1a;color:#fff;border:0;
padding:10px 16px;font-size:13px;cursor:pointer;border-radius:4px}
a{color:#0a4a8a}
"""

JS = """
document.querySelectorAll('textarea.cmt').forEach(t=>{
  const k=t.dataset.key;
  t.value=localStorage.getItem(k)||'';
  t.addEventListener('input',()=>{localStorage.setItem(k,t.value);});
});
document.getElementById('exportBtn').addEventListener('click',()=>{
  const out={};
  for(let i=0;i<localStorage.length;i++){
    const k=localStorage.key(i);
    if(k.startsWith('dev50rev::')&&localStorage.getItem(k).trim())out[k]=localStorage.getItem(k);
  }
  const blob=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='dev50_review_comments.json';a.click();
});
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(TRACE_DIR), help="trace dir (shards + sample_posts.json)")
    ap.add_argument("--name", default="", help="doc name suffix, e.g. main_v2 / lowng_v2")
    args = ap.parse_args()
    trace_dir = Path(args.dir)
    out = (_SRC / "reports" / f"dev50_verify_review_{args.name}_{date.today().isoformat()}.html"
           if args.name else OUT)
    global NS
    NS = f"{args.name}::" if args.name else ""
    recs = load(trace_dir)
    ok = [r for r in recs if r.get("ok")]
    print(f"{len(recs)} records, {len(ok)} ok")
    sections = "".join(render_post(r, i) for i, r in enumerate(ok, 1))
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>dev-50 verify-loop review {args.name} — {date.today().isoformat()}</title>
<style>{CSS}</style></head><body>
<h1>Verify-loop review {args.name or ""} — {len(ok)} posts ({date.today().isoformat()})</h1>
<p class="mini">Traced run: every LLM call (full prompt + raw response), search, scrape.
Loop config: max 3 Serper rounds + 1 Exa escalation, triage ON, DeepSeek-V4-Flash non-thinking.</p>
{build_summary(ok)}
{build_index(ok)}
{sections}
<button id="exportBtn">Export comments</button>
<script>{JS}</script></body></html>"""
    out.write_text(doc)
    print(f"wrote {out} ({len(doc) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
