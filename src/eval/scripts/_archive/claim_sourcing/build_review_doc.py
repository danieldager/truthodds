"""Build the combined source-review document: Tab 1 = roster + extraction summary + per-source
flag breakdown; Tab 2 = the filterable per-claim viewer. Self-contained, theme-aware HTML.

Reads the roster CSV (source metadata + harvest counts) and a claims-review HTML produced by
extract_tweet_claims.py --html (its embedded JSON payload carries every post with its claims,
flags, and no_claim_reason). Re-run any time.

  cd src && uv run python eval/scripts/claim_sourcing/build_review_doc.py \
      --payload <slice500 review>.html [--roster source_roster.csv] -o review.html
"""
import argparse, re, csv, json, re, html
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
BASE = SRC / "eval/data/survey_claims"


def load_payload(path):
    m = re.search(r'<script type="application/json" id="data">(.*?)</script>', Path(path).read_text(), re.S)
    return json.loads(m.group(1))["posts"]


def band(s):
    return "0-30" if s < 30 else "30-50" if s < 50 else "50-70" if s < 70 else "70-90" if s < 90 else "90-100"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", required=True, help="slice review HTML from extract_tweet_claims.py --html")
    ap.add_argument("--pass1", default="", help="optional raw-extraction payload HTML — each post then shows "
                                                "pass 1 (extract) under the final (normalized) claims")
    ap.add_argument("--claims", default="", help="handoff claims parquet (v3): final types (repairs applied), "
                                                 "gate checkworthy, trivial/unresolved flags, topics, removal ledger")
    ap.add_argument("--roster", default=str(BASE / "source_roster.csv"))
    ap.add_argument("-o", "--output", default=str(BASE / "source_review.html"))
    args = ap.parse_args()

    posts = load_payload(args.payload)
    if args.claims:
        import pandas as pd
        cdf = pd.read_parquet(args.claims)
        fin, rem = {}, {}
        for url, g in cdf.groupby("url", sort=False):
            kept, removed = [], []
            for _, r in g.iterrows():
                if isinstance(r["removed_reason"], str):
                    removed.append({"c": r["claim"], "reason": r["removed_reason"]})
                else:
                    kept.append({"c": r["claim"], "t": r["type"],
                                 "topic": r.get("topic") if isinstance(r.get("topic"), str) else None,
                                 "fals": bool(r["falsifiable"]) if "falsifiable" in g.columns else True,
                                 "triv": bool(r.get("trivial")), "unres": bool(r.get("unresolved")),
                                 "cw": bool(r["checkworthy"])})
            fin[url], rem[url] = kept, removed
        for p_ in posts:
            p_["claims"] = fin.get(p_["url"], [])
            p_["removed"] = rem.get(p_["url"], [])
    def _annotate_diffs(p1_claims, p2_claims, removed):
        """Match pass-1 claims to final claims by token overlap; annotate final claims with
        newc (no pass-1 source), diff (word-level segments when text changed substantially),
        fchg (which flags/type changed vs the primary source), mfrom/grp (merge groups);
        annotate pass-1 claims with grp (merge membership)."""
        import difflib
        from collections import defaultdict
        tok = lambda t: {w for w in re.findall(r"[a-z0-9][a-z0-9'\-]*", (t or "").lower()) if len(w) > 2}
        p1t = [tok(c.get("c")) for c in p1_claims]
        p2t = [tok(c.get("c")) for c in p2_claims]
        assign = {}
        for i, t1 in enumerate(p1t):
            best, bj = 0.0, None
            for j, t2 in enumerate(p2t):
                ov = len(t1 & t2) / max(1, len(t1))
                if ov > best:
                    best, bj = ov, j
            if bj is not None and best >= 0.45:
                assign[i] = bj
        # merged-away pass-1 claims (in the removal ledger) attach to their content's best final claim
        for r in removed or []:
            if r.get("reason") != "merged":
                continue
            rt = tok(r.get("c"))
            bi = max(range(len(p1t)), key=lambda i: len(rt & p1t[i]) / max(1, len(rt)), default=None)
            if bi is not None and bi not in assign and p2t:
                bj = max(range(len(p2t)), key=lambda j: len(p1t[bi] & p2t[j]) / max(1, len(p1t[bi])))
                if len(p1t[bi] & p2t[bj]) / max(1, len(p1t[bi])) >= 0.3:
                    assign[bi] = bj
        srcs = defaultdict(list)
        for i, j in assign.items():
            srcs[j].append(i)
        gid = 0
        strip = lambda w: w.lower().strip(".,;:!?\"'()")
        for j, ii in sorted(srcs.items()):
            c2 = p2_claims[j]
            if len(ii) >= 2:
                gid += 1
                c2["mfrom"] = len(ii)
                c2["grp"] = gid
                for i in ii:
                    p1_claims[i]["grp"] = gid
            prim = max(ii, key=lambda i: len(p1t[i] & p2t[j]) / max(1, len(p1t[i])))
            c1 = p1_claims[prim]
            w1, w2 = (c1.get("c") or "").split(), (c2.get("c") or "").split()
            sm = difflib.SequenceMatcher(a=[strip(w) for w in w1], b=[strip(w) for w in w2])
            segs, changed = [], 0
            for op, a0, a1, b0, b1 in sm.get_opcodes():
                if b1 > b0:
                    segs.append([0 if op == "equal" else 1, " ".join(w2[b0:b1])])
                    if op != "equal":
                        changed += b1 - b0
            if changed / max(1, len(w2)) > 0.15 and len(ii) < 2:  # substantial edit (merges shown by grouping)
                c2["diff"] = segs
            fchg = []
            if c1.get("t") != c2.get("t"):
                fchg.append("type")
            if bool(c1.get("fals", True)) != bool(c2.get("fals", True)):
                fchg.append("fals")
            if bool(c1.get("unres")) != bool(c2.get("unres")):
                fchg.append("unres")
            if fchg:
                c2["fchg"] = fchg
        for j, c2 in enumerate(p2_claims):
            if j not in srcs:
                c2["newc"] = True

    if args.pass1:
        p1 = {p["url"]: p for p in load_payload(args.pass1)}
        # "changed" badge: compare on normalized text (case/punctuation/whitespace collapsed) so
        # a period added at the end doesn't light the badge (Daniel's review note 2026-07-14)
        norm = lambda t: " ".join(re.sub(r"[^\w\s]", "", (t or "").lower()).split())
        sig = lambda cs: [(norm(c.get("c")), c.get("t")) for c in (cs or [])]
        for p in posts:
            r1p = p1.get(p["url"])
            if r1p is not None:
                p["r1"] = r1p.get("claims") or []
                p["r1nr"] = r1p.get("nr")
                p["chg"] = sig(p["r1"]) != sig(p.get("claims"))
                _annotate_diffs(p["r1"], p.get("claims") or [], p.get("removed"))
    roster = {r["handle"].lower(): r for r in csv.DictReader(open(args.roster))}

    # per-source aggregates from the sample
    agg = {}
    for p in posts:
        hk = (p.get("handle") or "").lower()
        a = agg.setdefault(hk, dict(handle=p.get("handle"), cell=p.get("cell"), posts=0, claims=0,
                                    assertion=0, attribution=0, cw=0, gated=0, removed=0,
                                    promo=0, opinion=0, none=0, empty=0))
        a["posts"] += 1
        a["removed"] += len(p.get("removed") or [])
        cs = p.get("claims") or []
        if not cs:
            a["empty"] += 1
            r = p.get("nr") or "other"
            a[{"promo": "promo", "opinion": "opinion"}.get(r, "none")] += 1
        for c in cs:
            a["claims"] += 1
            a[c["t"]] = a.get(c["t"], 0) + 1
            if c["cw"]: a["cw"] += 1; a["gated"] += 1

    tot = dict(posts=len(posts), claims=sum(a["claims"] for a in agg.values()),
               gated=sum(a["gated"] for a in agg.values()),
               withc=sum(1 for p in posts if p.get("claims")))
    esc = html.escape

    def kfmt(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k" if v >= 1e3 else str(int(v))

    def row(a):
        r = roster.get((a["handle"] or "").lower(), {})
        ng = r.get("ng_score", "")
        lean = ("O" if r.get("register") == "off_topic"
                else ((r.get("lean") or a["cell"] or "?")[0].upper()))
        harv = r.get("tweets", "")
        nc = f"{a['empty']} <span class='sub'>({a['promo']}/{a['opinion']}/{a['none']})</span>" if a["empty"] else "0"
        return (f"<tr><td class='src'><span class='dot {lean}'></span>@{esc(a['handle'] or '')}"
                f"<span class='dom'>{esc(r.get('domain',''))}</span></td>"
                f"<td class='n'>{esc(str(ng))}</td>"
                f"<td class='n'>{esc(kfmt(r.get('followers')))}</td>"
                f"<td class='n'>{esc(str(harv))}</td>"
                f"<td class='n'>{a['claims']}</td>"
                f"<td class='n'>{a['assertion']}<span class='sub'> / {a['attribution']}</span></td>"
                f"<td class='n strong'>{a['cw']}<span class='sub'> / {a['claims']-a['cw']}</span></td>"
                f"<td class='n'>{a['removed']}</td>"
                f"<td class='n'>{nc}</td></tr>")

    def ng_of(a):
        try:
            return float(roster.get((a["handle"] or "").lower(), {}).get("ng_score") or 0)
        except ValueError:
            return 0.0

    def section(bname):
        rs = sorted([a for a in agg.values() if band(ng_of(a)) == bname], key=lambda a: -ng_of(a))
        if not rs:
            return ""
        return f"<tr class='grp'><td colspan='9'>NewsGuard {bname}</td></tr>" + "".join(row(a) for a in rs)

    head = ("<tr><th>source</th><th class='n'>NG</th><th class='n'>followers</th>"
            "<th class='n'>harvest</th><th class='n'>claims</th>"
            "<th class='n'>assert<span class='sub'>/attrib</span></th>"
            "<th class='n'>checkworthy<span class='sub'>/no</span></th>"
            "<th class='n'>removed</th>"
            "<th class='n'>no-claim<span class='sub'>(p/o/other)</span></th></tr>")

    per = tot["posts"] / max(1, len(agg))
    intro = (f"Claim extraction ran on the locked <b>{tot['posts']}-tweet</b> dev sample "
             f"({per:.0f} per source across all {len(agg)} sources, including the new center and "
             f"off-topic probe outlets). It extracts every checkable claim, well-formed, and tags each with "
             f"<b>checkworthy</b> - whether a professional fact-checker would bother to check it (weighty, "
             f"concrete, and consequential). On the sample it "
             f"produced <b>{tot['claims']} claims</b> ({tot['claims']/tot['posts']:.1f} per tweet); "
             f"<b>{tot['withc']}/{tot['posts']}</b> tweets yielded at least one claim, and "
             f"<b>{tot['gated']}</b> are checkworthy — the set we would send to verification. "
             f"Density tracks each outlet's posting style, not its reliability.")
    caption = ("Harvest is the full corpus (tweets collected on X). The remaining columns are from the "
               f"{tot['posts']}-tweet tuning sample. No-claim counts empty posts (promo / opinion / none).")

    import hashlib
    keysrc = Path(args.claims if args.claims else args.payload)
    runkey = hashlib.sha1(keysrc.read_bytes()).hexdigest()[:10]
    payload = json.dumps({"posts": posts}, ensure_ascii=False).replace("</", "<\\/")
    doc = TEMPLATE.replace("__INTRO__", intro).replace("__CAPTION__", esc(caption)) \
        .replace("__HEAD__", head) \
        .replace("__ROWS__", "".join(section(b) for b in ("90-100", "70-90", "50-70", "30-50", "0-30"))) \
        .replace("__PAYLOAD__", payload).replace("__RUNKEY__", runkey)
    Path(args.output).write_text(doc)
    print(f"wrote {args.output} — {tot['posts']} posts, {tot['claims']} claims, {tot['gated']} gated")


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Source review</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1c1a;--muted:#77776f;--card:#fff;--line:#e7e6e2;--accent:#4457c7;
 --L:#2f6feb;--R:#c0392b;--assert:#3f8a5c;--attrib:#7a56c0;--grp:#f2f1ee;--soft:#f6f5f2;
 --add:#3f8a5c;--del:#c0392b;--p1:#8a6d1f}
@media(prefers-color-scheme:dark){:root{--bg:#161719;--fg:#e9e9e6;--muted:#95958e;--card:#1f2123;
 --line:#31333799;--accent:#8894ec;--L:#5a8def;--R:#e0685b;--assert:#5aa877;--attrib:#a389df;--grp:#232527;--soft:#1c1e20;
 --add:#5aa877;--del:#e0685b;--p1:#c9a84a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1020px;margin:0 auto;padding:20px 18px 90px}
h1{font-size:19px;margin:0 0 12px;font-weight:650}
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin-bottom:18px}
.tab{padding:8px 16px;cursor:pointer;color:var(--muted);font-weight:600;font-size:14px;border-bottom:2px solid transparent;margin-bottom:-1px}
.tab.on{color:var(--fg);border-bottom-color:var(--accent)}
.pane{display:none}.pane.on{display:block}
.intro{font-size:14px;line-height:1.6;color:var(--fg);margin:0 0 8px}.intro b{font-weight:650}
.cap{font-size:12.5px;color:var(--muted);margin:0 0 16px}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:7px 11px;border-bottom:1px solid var(--line);vertical-align:baseline}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.grp td{background:var(--grp);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
tr:last-child td{border-bottom:0}
.src{font-weight:600}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px;vertical-align:middle}
.dot.L{background:var(--L)}.dot.R{background:var(--R)}.dot.C{background:#8e8e86}.dot.O{background:#c78f2e}
.dom{color:var(--muted);font-weight:400;font-size:11.5px;margin-left:7px}
.sub{color:var(--muted);font-weight:400;font-size:.88em}.strong{font-weight:700}
/* tab 2 */
.filters{position:sticky;top:0;background:var(--bg);padding:6px 0 10px;z-index:5;border-bottom:1px solid var(--line);margin-bottom:14px}
.filters input[type=search]{font:inherit;font-size:14px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);width:100%;margin-bottom:9px}
.frow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:5px 0}
.srcdet{margin:5px 0}
.srcdet summary{cursor:pointer;list-style:none}
.srcdet summary::before{content:"\25B8";margin-right:5px;font-size:10px}
.srcdet[open] summary::before{content:"\25BE"}
.srcdet summary::-webkit-details-marker{display:none}
.srcdet .fgroup{margin-top:7px}
#srcsum{text-transform:none;letter-spacing:0;color:var(--accent);font-weight:600}
.frow.groups{gap:10px 22px}
.funit{display:flex;align-items:center;gap:7px}
.flabel{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.fgroup{display:flex;flex-wrap:wrap;gap:4px}
.tog{font:inherit;font-size:12px;padding:3px 10px;border:1px solid var(--line);border-radius:14px;background:var(--card);color:var(--muted);cursor:pointer}
.tog:hover{color:var(--fg)}
.tog.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tog.src.L.on{background:var(--L);border-color:var(--L)}.tog.src.R.on{background:var(--R);border-color:var(--R)}
.count{color:var(--muted);font-size:12.5px;margin:0 0 12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:9px}
.chead{display:flex;flex-wrap:wrap;gap:9px;align-items:center;font-size:12px;color:var(--muted);margin-bottom:7px}
.chead .h{font-weight:600;color:var(--fg)}.chead .h .L{color:var(--L)}.chead .h .R{color:var(--R)}
.chead a{color:var(--accent);text-decoration:none}
.post{font-size:14px;white-space:pre-wrap;margin:0 0 9px}
.imgs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:9px}.imgs img{max-height:120px;max-width:100%;border-radius:6px;border:1px solid var(--line)}
.claims{border-top:1px solid var(--line);padding-top:4px}
.claim{display:flex;gap:9px;padding:6px 0}
.claim+.claim{border-top:1px solid var(--soft)}
.cdot{flex:none;width:7px;height:7px;border-radius:50%;margin-top:7px}
.cdot.assertion{background:var(--assert)}.cdot.attribution{background:var(--attrib)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media(max-width:640px){.cols{grid-template-columns:1fr}}
.pcol{border:1px solid var(--line);border-radius:8px;padding:9px 11px;background:var(--soft)}
.pcol h4{margin:0 0 7px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;font-weight:700;display:flex;gap:7px;align-items:center}
.pcol.p1 h4{color:var(--p1)}.pcol.p2 h4{color:var(--accent)}
.pcol h4 .ct{color:var(--muted);font-weight:600}
.claim.new{background:color-mix(in srgb,var(--add) 12%,transparent);border-radius:5px;margin:0 -5px;padding-left:5px;padding-right:5px}
.claim.gone{opacity:.62}.claim.gone .ctext{text-decoration:line-through solid var(--del)}
.ctext mark{background:color-mix(in srgb,var(--p1) 28%,transparent);color:inherit;border-radius:3px;padding:0 2px}
.tag.chgd{color:var(--p1);font-weight:700}
.mtag{font-size:10.5px;font-weight:700;color:var(--accent);border:1px solid var(--accent);border-radius:5px;padding:0 6px}
.claim.ingrp{border-left:2px solid var(--accent);margin-left:-2px;padding-left:7px;border-radius:0}
.chgbadge{font-size:11px;color:var(--accent);font-weight:700}
.cbody{flex:1;min-width:0}
.ctext{font-size:13.5px}
.ctags{margin-top:2px;display:flex;gap:6px;flex-wrap:wrap}
.tag{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
.tag.warn{color:var(--R)}
.noclaim{color:var(--muted);font-size:13px;font-style:italic;padding:2px 0}
.pill{font-style:normal;font-size:11px;font-weight:600;color:var(--accent);border:1px solid var(--accent);border-radius:5px;padding:1px 6px;margin-left:5px}
.anno{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:14px}
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
<h1>Source roster &amp; claim extraction</h1>
<div class="tabs"><div class="tab on" data-pane="p1">Roster</div><div class="tab" data-pane="p2">Claims</div></div>

<div class="pane on" id="p1">
 <p class="intro">__INTRO__</p>
 <p class="cap">__CAPTION__</p>
 <table>__HEAD____ROWS__</table>
</div>

<div class="pane" id="p2">
 <div class="anno">
  <input type="text" id="anno" placeholder="Your name (labels the export)">
  <label class="chk"><input type="checkbox" id="fcmt"> commented only</label>
  <button class="tog" id="shuf" title="Randomize post order — review a mix of sources instead of one source at a time">&#128256; shuffle</button>
  <button class="tog" id="clr" title="Delete every comment stored in this browser for this run">clear comments</button>
  <span class="ct" id="cmtct"></span>
 </div>
 <div class="filters">
  <input type="search" id="q" placeholder="Search claim or post…">
  <details class="srcdet"><summary class="flabel">sources <span id="srcsum"></span></summary><div class="fgroup" data-group="src" id="fsrc"></div></details>
  <div class="frow groups">
   <div class="funit"><span class="flabel">type</span><div class="fgroup" data-group="type"><button class="tog" data-v="assertion">assertion</button><button class="tog" data-v="attribution">attribution</button></div></div>
   <div class="funit"><span class="flabel">checkworthy</span><div class="fgroup" data-group="cw"><button class="tog" data-v="1">yes</button><button class="tog" data-v="0">no</button></div></div>
   <div class="funit"><span class="flabel">no-claim</span><div class="fgroup" data-group="reason"><button class="tog" data-v="promo">promo</button><button class="tog" data-v="opinion">opinion</button><button class="tog" data-v="none">other</button></div></div>
   <div class="funit"><span class="flabel">flags</span><div class="fgroup" data-group="fl"><button class="tog" data-v="nfals">not falsifiable</button><button class="tog" data-v="unres">unresolved</button><button class="tog" data-v="rm">removed</button></div></div>
   <div class="funit"><span class="flabel">normalizer</span><div class="fgroup" data-group="chg"><button class="tog" data-v="1">changed</button><button class="tog" data-v="0">unchanged</button></div></div>
  </div>
 </div>
 <p class="count" id="count"></p><div id="list"></div>
 <button class="fab" id="fab">Export comments</button>
 <div class="modal" id="modal"><div class="sheet"><h3 id="mtitle">Comments</h3><textarea id="mtext" readonly placeholder="No comments yet. Type in the box under any tweet where you disagree with the extraction."></textarea><div class="row"><button class="btn ghost" id="mclose">Close</button><button class="btn ghost" id="mdl">Download JSON</button><button class="btn" id="mcopy">Copy JSON</button></div></div></div>
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
const G={src:new Set(),type:new Set(),cw:new Set(),reason:new Set(),chg:new Set(),fl:new Set()};
document.querySelectorAll('.tog').forEach(b=>b.addEventListener('click',()=>{const g=b.parentElement.dataset.group,v=b.dataset.v;if(G[g].has(v)){G[g].delete(v);b.classList.remove('on');}else{G[g].add(v);b.classList.add('on');}render();}));
const q=document.getElementById('q'),list=document.getElementById('list'),count=document.getElementById('count');
q.addEventListener('input',render);
// --- manual annotation: per-tweet comments, persisted in the browser ---
const KEY='srev::__RUNKEY__::';   // namespaced per run: a new run starts with a clean slate
let comments=JSON.parse(localStorage.getItem(KEY+'comments')||'{}');
const anno=document.getElementById('anno');anno.value=localStorage.getItem(KEY+'anno')||'';
anno.addEventListener('input',()=>localStorage.setItem(KEY+'anno',anno.value));
const fcmt=document.getElementById('fcmt');fcmt.addEventListener('change',render);
// --- shuffle: randomize post order (persisted) so reviewers see a mix of sources ---
let order=JSON.parse(localStorage.getItem(KEY+'order')||'null');
if(order){const idx={};order.forEach((u,i)=>idx[u]=i);D.posts.sort((a,b)=>((idx[a.url]??1e9)-(idx[b.url]??1e9)));}
document.getElementById('shuf').addEventListener('click',()=>{
 for(let i=D.posts.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[D.posts[i],D.posts[j]]=[D.posts[j],D.posts[i]];}
 order=D.posts.map(p=>p.url);localStorage.setItem(KEY+'order',JSON.stringify(order));render();});
const cmtct=document.getElementById('cmtct'),fab=document.getElementById('fab');
document.getElementById('clr').addEventListener('click',()=>{
 const n=Object.values(comments).filter(v=>v&&v.trim()).length;
 if(!n){alert('No comments to clear.');return;}
 if(!confirm('Delete all '+n+' comment(s) stored for this run? This cannot be undone.'))return;
 comments={};localStorage.setItem(KEY+'comments','{}');render();});
const nCmt=()=>Object.values(comments).filter(v=>v&&v.trim()).length;
function updCt(){const n=nCmt();cmtct.textContent=n?n+' commented':'';fab.textContent='Export comments'+(n?' ('+n+')':'');}
list.addEventListener('input',e=>{const el=e.target;if(!el.classList||!el.classList.contains('cmt'))return;const u=el.dataset.url;if(el.value.trim())comments[u]=el.value;else delete comments[u];localStorage.setItem(KEY+'comments',JSON.stringify(comments));const card=el.closest('.card');if(card)card.classList.toggle('commented',!!el.value.trim());updCt();});
function buildExport(){const o={annotator:anno.value||null,exported:new Date().toISOString(),total_posts:D.posts.length,comments:[]};D.posts.forEach(p=>{const c=comments[p.url];if(c&&c.trim())o.comments.push({url:p.url,handle:p.handle,post:p.text,claims:(p.claims||[]).map(x=>({claim:x.c,type:x.t,topic:x.topic,trivial:x.triv,unresolved:x.unres,checkworthy:x.cw})),removed:p.removed||[],comment:c});});return o;}
const modal=document.getElementById('modal'),mtext=document.getElementById('mtext'),mtitle=document.getElementById('mtitle');
fab.addEventListener('click',()=>{const o=buildExport();mtext.value=JSON.stringify(o,null,2);mtitle.textContent=o.comments.length+' comment'+(o.comments.length===1?'':'s')+(o.annotator?' — '+o.annotator:'');modal.classList.add('on');});
document.getElementById('mclose').addEventListener('click',()=>modal.classList.remove('on'));
modal.addEventListener('click',e=>{if(e.target===modal)modal.classList.remove('on');});
document.getElementById('mdl').addEventListener('click',()=>{const o=buildExport();const b=new Blob([JSON.stringify(o,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='extraction-review-'+((o.annotator||'anon').replace(/[^a-z0-9]+/gi,'_'))+'.json';a.click();});
// hosted-artifact sandbox blocks blob downloads — Copy is the path that always works
document.getElementById('mcopy').addEventListener('click',e=>{const b=e.target,done=ok=>{b.textContent=ok?'Copied ✓':'Copy failed';setTimeout(()=>b.textContent='Copy JSON',1600);};
 if(navigator.clipboard&&navigator.clipboard.writeText)navigator.clipboard.writeText(mtext.value).then(()=>done(true),()=>{mtext.select();done(document.execCommand('copy'));});
 else{mtext.select();done(document.execCommand('copy'));}});
function diffHtml(c){
 if(!c.diff)return esc(c.c);
 return c.diff.map(([m,t])=>m?`<mark>${esc(t)}</mark>`:esc(t)).join(' ');}
function colHtml(claims,other,nr,label,role){
 let inner;
 if(claims&&claims.length){
  inner=claims.map(c=>{
   // pass-2 semantics: text struck ONLY for dropped claims (rendered below from the removal
   // ledger); flag-only changes show as highlighted chips; substantial edits highlight the
   // changed words; merges show grouped arrows. Pass-1 claims are never struck.
   const mark=(role==='p2'&&c.newc)?' new':'';
   const grp=c.grp?' ingrp':'';
   const tags=[];
   if(c.grp)tags.push(role==='p2'
     ?`<span class="mtag">&#8656; merged ${esc(c.mfrom)} claims (&#9312;${c.grp>1?'…':''})</span>`.replace('&#9312;','#'+c.grp)
     :`<span class="mtag">merge #${c.grp} &#8658;</span>`);
   const chg=c.fchg||[];
   tags.push(`<span class="tag${chg.includes('type')?' chgd':''}">${esc(c.t)}${chg.includes('type')?' (retyped)':''}</span>`);
   if(c.topic)tags.push(`<span class="tag" style="color:var(--accent)">${esc(c.topic)}</span>`);
   if(c.fals===false)tags.push(`<span class="tag warn${chg.includes('fals')?' chgd':''}">not falsifiable${chg.includes('fals')?' (changed)':''}</span>`);
   else if(chg.includes('fals'))tags.push('<span class="tag chgd">falsifiable restored</span>');
   if(c.unres)tags.push(`<span class="tag warn${chg.includes('unres')?' chgd':''}">unresolved${chg.includes('unres')?' (changed)':''}</span>`);
   else if(chg.includes('unres'))tags.push('<span class="tag chgd">unresolved cleared</span>');
   if(c.cw===false)tags.push('<span class="tag warn">not checkworthy</span>');
   const text=(role==='p2')?diffHtml(c):esc(c.c);
   return `<div class="claim${mark}${grp}"><span class="cdot ${esc(c.t)}"></span><div class="cbody"><div class="ctext">${text}</div><div class="ctags">${tags.join('')}</div></div></div>`;
  }).join('');
 }else{
  inner=`<div class="noclaim">no claim<span class="pill">${esc(nr||'—')}</span></div>`;
 }
 let rmHtml='';
 if(role==='p2'&&window.__curRemoved&&window.__curRemoved.length){
  rmHtml=window.__curRemoved.filter(r=>r.reason!=='merged').map(r=>`<div class="claim gone"><span class="cdot"></span><div class="cbody"><div class="ctext">${esc(r.c)}</div><div class="ctags"><span class="tag warn">removed: ${esc(r.reason)}</span></div></div></div>`).join('');}
 return `<div class="pcol ${role}"><h4>${label}<span class="ct">${claims?claims.length:0}</span></h4>${inner}${rmHtml}</div>`;
}
const unit=c=>(!G.type.size||G.type.has(c.t))&&(!G.cw.size||G.cw.has(c.cw?'1':'0'))
 &&(!G.fl.size||[...G.fl].some(f=>f==='nfals'?c.fals===false:f==='unres'?c.unres:false));
const postFl=p=>!G.fl.size||[...G.fl].some(f=>f==='rm'?((p.removed||[]).length>0):((p.claims||[]).some(c=>f==='nfals'?c.fals===false:c.unres)));
const claimFilt=()=>G.type.size||G.cw.size;
function render(){
 const t=q.value.toLowerCase();let shown=0;
 list.innerHTML=D.posts.filter(p=>{
  if(G.src.size&&!G.src.has(p.handle))return false;
  if(G.chg.size&&!G.chg.has(p.chg?'1':'0'))return false;
  if(fcmt.checked&&!(comments[p.url]&&comments[p.url].trim()))return false;
  if(!postFl(p))return false;
  const u=(p.claims||[]).filter(unit);
  const claimPass=!G.reason.size&&(u.length>0||(G.fl.has('rm')&&(p.removed||[]).length>0));
  const emptyPass=(!p.claims||!p.claims.length)&&!claimFilt()&&(!G.reason.size||G.reason.has(p.nr||'none'));
  if(!(claimPass||emptyPass))return false;
  if(t&&!((p.text+' '+(p.claims||[]).map(c=>c.c).join(' ')).toLowerCase().includes(t)))return false;
  p._u=u;return true;
 }).map(p=>{shown++;
  const imgs=(p.images||[]).map(u=>`<img src="${esc(u)}" loading="lazy">`).join('');
  const body=(p.claims&&p.claims.length)?`<div class="claims">`+p._u.map(c=>{
    const tags=[`<span class="tag">${esc(c.t)}</span>`];
    if(c.topic)tags.push(`<span class="tag" style="color:var(--accent)">${esc(c.topic)}</span>`);
    if(c.fals===false)tags.push('<span class="tag warn">not falsifiable</span>');
    if(c.unres)tags.push('<span class="tag warn">unresolved</span>');
    if(c.cw===false)tags.push('<span class="tag warn">not checkworthy</span>');
    return `<div class="claim"><span class="cdot ${c.t}"></span><div class="cbody"><div class="ctext">${esc(c.c)}</div><div class="ctags">${tags.join('')}</div></div></div>`;
   }).join('')+`</div>`
   :`<div class="noclaim">no claim<span class="pill">${esc(p.nr||'—')}</span></div>`;
  const hasC=!!(comments[p.url]&&comments[p.url].trim());
  const badge=(p.r1!=null&&p.chg)?`<span class="chgbadge">normalizer changed this</span>`:'';
  window.__curRemoved=p.removed||[];
  const middle=(p.r1!=null)
    ?`<div class="cols">${colHtml(p.r1,p.claims,p.r1nr,'Pass 1 · extract','p1')}${colHtml(p.claims,p.r1,p.nr,'Final · audited + gated','p2')}</div>`
    :body;
  return `<div class="card${hasC?' commented':''}"><div class="chead"><span class="h"><span class="${cls(p.cell)}">@${esc(p.handle)}</span></span><span>${esc(p.date)}</span><span>${esc(p.domain)}</span>${badge}<a href="${esc(p.url)}" target="_blank" rel="noopener">tweet ↗</a></div>${imgs?`<div class="imgs">${imgs}</div>`:''}<div class="post">${esc(p.text)}</div>${middle}<textarea class="cmt" data-url="${esc(p.url)}" placeholder="Disagree here — missed claim, wrong flag, bad split, wrong drop…">${esc(comments[p.url]||'')}</textarea></div>`;
 }).join('');
 count.textContent=`${shown} of ${D.posts.length} posts`;
 const ss=document.getElementById('srcsum');if(ss)ss.textContent=G.src.size?`· ${G.src.size} selected`:`(${srcs.length})`;}
render();updCt();
</script></div></body></html>"""


if __name__ == "__main__":
    main()
