"""Truth Odds — figure set 1: what does the first Serper query return (10 slots)?

Reads evidence-profile run dirs (evidence_profile_run.py output) and renders an
aggregate-only HTML artifact (inline SVG, no libraries): the average urn per group,
per-rank slot profile, n_t/n_f/n_empty distributions, source-kind mix, and the
retrieval funnel. TARGETED claims only (Daniel 2026-07-21); nothing one-row-per-post.

Groups: labeled sets split by gold (averitec-sup/ref, fcgold-sup/ref) and tweet groups
split by round-1 arm-A outcome (resolved-r1 / refuted-r1 / continuing).

  cd src && uv run python eval/scripts/claim_sourcing/build_truthodds_figures.py \
      --averitec eval/data/survey_claims/truthodds_profile_averitec \
      --fcgold eval/data/survey_claims/truthodds_profile_fcgold \
      --tweets eval/data/survey_claims/truthodds_profile_dev1000_s150 \
      -o <out.html>
"""
import argparse, functools, glob, html, json, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[3]))
from collections import Counter, defaultdict
from pathlib import Path

SUP = {"supports", "partially-supports"}
REF = {"refutes", "partially-refutes"}


def load(dirpath):
    recs = []
    for f in glob.glob(str(Path(dirpath) / "*.jsonl")):
        for l in open(f):
            d = json.loads(l)
            if d.get("ok"):
                recs.append(d["result"])
    return recs


def result_fates(rec):
    """Per result slot (rank 1..10): its fate for the TARGETED claims of round 1.
    Returns list of (rank, fate, ng, kind, on_claim). Fates:
    support-read / refute-read / neutral-read / snippet-voice / read-no-evidence /
    dropped-<reason-family> / not-picked"""
    if not rec.get("rounds"):
        return []
    rd = rec["rounds"][0]
    targets = set(rd.get("targets") or [])
    ev = rec.get("evidence") or []
    doc_by_url = {d["url"]: d["id"] for d in rd.get("docs") or []}
    drops = {d["url"]: d["reason"] for d in rd.get("dropped") or []}
    picks = set((rd.get("triage") or {}).get("read") or [])
    sc = {s["i"]: s for s in (rec.get("profile") or {}).get("source_classes") or []}
    out = []
    for r in rd.get("results") or []:
        i, url = r["i"], r.get("url")
        s = sc.get(i) or {}
        if url in doc_by_url:
            did = doc_by_url[url]
            ents = [e for e in ev if e.get("src") == did and e.get("claim_id") in targets]
            stances = {e.get("stance") for e in ents}
            fate = ("support-read" if stances & SUP else
                    "refute-read" if stances & REF else
                    "neutral-read" if ents else "read-no-evidence")
        elif any(e.get("src") == f"R1.{i}" and e.get("claim_id") in targets
                 and e.get("snippet_only") for e in ev):
            fate = "snippet-voice"
        elif url in drops:
            reason = drops[url]
            fam = ("dup" if reason.startswith(("dup", "near-dup")) else
                   "scrape-fail" if reason in ("scrape-failed", "too-short", "too-thin") else
                   "junk-offtopic" if reason in ("junk-page", "off-topic") else
                   "origin" if reason in ("origin", "linked-source", "republication") else "other")
            fate = f"dropped-{fam}"
        else:
            fate = "not-picked"
        out.append((i, fate, r.get("ng"), s.get("kind"), s.get("on_claim")))
    return out


def claim_counts(rec):
    """Per TARGETED claim: (n_t, n_f, n_empty) over the 10 slots — support/refute voices
    (distinct result slots producing directional evidence, full reads and snippet voices
    counted at result level), empty = 10 - n_t - n_f."""
    if not rec.get("rounds"):
        return []
    rd = rec["rounds"][0]
    ev = rec.get("evidence") or []
    doc_src = {d["id"] for d in rd.get("docs") or []}
    out = []
    for cid in rd.get("targets") or []:
        ents = [e for e in ev if e.get("claim_id") == cid]
        nt = len({e["src"] for e in ents if e.get("stance") in SUP and e["src"] in doc_src})
        nf = len({e["src"] for e in ents if e.get("stance") in REF and e["src"] in doc_src})
        n = len(rd.get("results") or []) or 10
        out.append((nt, nf, max(0, n - nt - nf)))
    return out


FATES = ["support-read", "refute-read", "neutral-read", "snippet-voice", "read-no-evidence",
         "not-picked", "dropped-dup", "dropped-scrape-fail", "dropped-junk-offtopic",
         "dropped-origin", "dropped-other"]
FCOL = {"support-read": "#1b7f3b", "refute-read": "#c62828", "neutral-read": "#7e57c2",
        "snippet-voice": "#8bc34a", "read-no-evidence": "#26a69a",
        "not-picked": "#bdbdbd", "dropped-dup": "#8d6e63", "dropped-scrape-fail": "#ef6c00",
        "dropped-junk-offtopic": "#fdd835", "dropped-origin": "#5c6bc0", "dropped-other": "#607d8b"}

# reliability composition of the RAW 10 results (pre-triage, pre-read). "institutional"
# = the code tier historically named PRIMARY (gov-TLD patterns + curated allowlist);
# Daniel 2026-07-21: reserve "primary" for truly primary records found at READ.
RELS = ["institutional", "NG 90-100", "NG 80-89", "NG 70-79", "NG 60-69", "NG <60", "unrated"]
RCOL = {"institutional": "#7986cb", "NG 90-100": "#1b7f3b", "NG 80-89": "#7cb342",
        "NG 70-79": "#fdd835", "NG 60-69": "#ef6c00", "NG <60": "#c62828",
        "unrated": "#9e9e9e"}


@functools.lru_cache(maxsize=1)
def _ng_map():
    from pipeline.search import newsguard_score_map
    return newsguard_score_map()


def rel_bin(rel, ng, dom=None):
    """Reliability bin for a stored result. When a record carries no NG score, retry the
    lookup with the current alias/subdomain rules (records written before 2026-07-21 have
    ng=None for renamed publishers such as ms.now)."""
    if rel == "PRIMARY":
        return "institutional"
    if ng is None and dom:
        from pipeline.verify_tweet_claims import _ng_of
        ng = _ng_of(dom, _ng_map())
    if ng is None:
        return "unrated"
    ng = float(ng)
    return ("NG 90-100" if ng >= 90 else "NG 80-89" if ng >= 80 else
            "NG 70-79" if ng >= 70 else "NG 60-69" if ng >= 60 else "NG <60")


GLOSS = {
    "on-claim": "Classifier judgment from the snippet: does this result address the events or statements the post's claims are about (vs merely topically adjacent)?",
    "targeted": "Claims the round's ONE query actually aimed at. Untargeted claims in multi-claim posts have artificially empty result profiles, so all figures count targeted claims only.",
    "voice": "An independent source after dedup: wire reprints, mirrors, and republications of the origin outlet collapse into one voice.",
    "snippet voice": "A search-result excerpt that bears on a claim without the page being read. Direction-less corroboration; can never close a claim.",
    "institutional": "Government TLD patterns, .edu, and a curated allowlist of official-record producers (the code tier historically named PRIMARY). 'Primary' proper is reserved for truly primary records identified at READ.",
    "unrated": "No NewsGuard score and not institutional: Wikipedia, UGC platforms, academic sites, and non-US/EU outlets NewsGuard does not cover.",
    "date ceiling": "Retrieval capped at the claim's date, approximating what a fresh claim faces. Labeled calibration sets run ceiling ON; tweets run production mode (OFF).",
    "urn": "the PI's model: each search result is a draw from an urn whose composition depends on whether the claim is true or false. These figures show the measured compositions.",
    "n_t": "Number of the 10 results that became supporting voices for the claim.",
    "n_f": "Number that became refuting voices.",
    "n_e": "Empty slots: results that produced no directional signal for the claim (not picked, dropped, off-claim, or read without evidence).",
    "READ": "The per-document pointer step: an LLM marks which sentences bear on which claim and assigns a page-local stance.",
    "fate": "What each of the 10 result slots became by the end of the round: a directional voice, a snippet voice, or one of the drop/skip categories.",
    "averitec": "AVeriTeC: an academic fact-checking benchmark (dev split). Real claims, mostly political and statistical, each labeled by annotators who also recorded the evidence. We use the Supported and Refuted claims and skip the disputed labels (Not Enough Evidence, Conflicting) and media-authenticity claims, which text search cannot settle.",
    "fcgold": "fc-gold: claims collected from professional fact-checkers through the Google Fact Check API, with their published ratings harmonised to our four labels. Mostly viral misinformation, so it is closer to what the nudge tool will actually meet than AVeriTeC is. Same filters applied.",
    "tweets": "dev-1000: our own corpus of posts from rated news outlets on X (the dev-500-A and dev-500-B draws), with claims extracted by the v4.7 chain. No truth labels, so it shows the average claim profile rather than a true-versus-false comparison.",
}


def term(word, label=None):
    tip = GLOSS.get(word, "")
    return f'<span class="gloss" data-tip="{esc(tip)}">{esc(label or word)}</span>'


def donut(pairs, title, subtitle="", size=210):
    """pairs: [(cat, count)] -> a donut of shares. Slices >=6% get an inline % label;
    everything is in the hover title. Shares, not an invented per-10 scale."""
    import math
    total = sum(v for _, v in pairs) or 1
    cx = cy = size / 2
    r_out, r_in = size / 2 - 26, (size / 2 - 26) * 0.58
    svg = [f'<svg viewBox="0 0 {size} {size + 26}" xmlns="http://www.w3.org/2000/svg">']
    ang = -math.pi / 2
    for cat, v in pairs:
        if v <= 0:
            continue
        frac = v / total
        sweep = frac * 2 * math.pi
        a2 = ang + sweep
        large = 1 if sweep > math.pi else 0
        x1, y1 = cx + r_out * math.cos(ang), cy + r_out * math.sin(ang)
        x2, y2 = cx + r_out * math.cos(a2), cy + r_out * math.sin(a2)
        xi2, yi2 = cx + r_in * math.cos(a2), cy + r_in * math.sin(a2)
        xi1, yi1 = cx + r_in * math.cos(ang), cy + r_in * math.sin(ang)
        d = (f"M {x1:.1f} {y1:.1f} A {r_out:.1f} {r_out:.1f} 0 {large} 1 {x2:.1f} {y2:.1f} "
             f"L {xi2:.1f} {yi2:.1f} A {r_in:.1f} {r_in:.1f} 0 {large} 0 {xi1:.1f} {yi1:.1f} Z")
        svg.append(f'<path d="{d}" fill="{FCOL[cat]}" stroke="#fff" stroke-width="1">'
                   f'<title>{esc(cat)}: {frac:.0%} ({v})</title></path>')
        if frac >= 0.06:
            mid = ang + sweep / 2
            rl = (r_out + r_in) / 2
            svg.append(f'<text x="{cx + rl * math.cos(mid):.1f}" y="{cy + rl * math.sin(mid) + 4:.1f}" '
                       f'text-anchor="middle" font-size="11" font-weight="600" fill="#fff">{frac:.0%}</text>')
        ang = a2
    svg.append(f'<text x="{cx}" y="{cy + 2}" text-anchor="middle" font-size="12" font-weight="700" fill="#333">{esc(title)}</text>')
    svg.append(f'<text x="{cx}" y="{cy + 17}" text-anchor="middle" font-size="10.5" fill="#888">{esc(subtitle)}</text>')
    svg.append('</svg>')
    return f'<div class="donut">{"".join(svg)}</div>'


def esc(x):
    return html.escape(str(x))



def hbar_stack(rows, title, note="", cats=None, colors=None, lgnd=False):
    """rows: [(label, {cat: mean_count})] — stacked horizontal bars over 10 slots."""
    W, BH, LW = 860, 26, 210
    parts = [f'<div class="fig"><h3>{esc(title)}</h3>']
    if note:
        parts.append(f'<p class="note">{note}</p>')   # caller-controlled HTML (glossary spans)
    if lgnd:
        parts.append(legend(cats, colors))
    h = len(rows) * (BH + 10) + 30
    svg = [f'<svg viewBox="0 0 {W} {h}" xmlns="http://www.w3.org/2000/svg">']
    cats = cats or FATES
    colors = colors or FCOL
    scale = (W - LW - 60) / 10.0
    for j, (label, mix) in enumerate(rows):
        y = j * (BH + 10) + 6
        svg.append(f'<text x="{LW-8}" y="{y+BH/2+4}" text-anchor="end" font-size="12" fill="#333">{esc(label)}</text>')
        x = LW
        for f in cats:
            v = mix.get(f, 0.0)
            if v <= 0.005:
                continue
            w = v * scale
            svg.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{BH}" fill="{colors[f]}">'
                       f'<title>{esc(f)}: {v:.2f} of 10</title></rect>')
            if w > 26:
                svg.append(f'<text x="{x+w/2:.1f}" y="{y+BH/2+4}" text-anchor="middle" font-size="10" fill="#222">{v:.1f}</text>')
            x += w
        total = sum(mix.get(f, 0.0) for f in cats)
        svg.append(f'<text x="{x+8:.1f}" y="{y+BH/2+4}" font-size="11" font-weight="600" fill="#444">'
                   f'\u03a3 {total:.1f}</text>')
    ax = LW
    for k in range(11):
        svg.append(f'<line x1="{ax:.1f}" y1="{h-22}" x2="{ax:.1f}" y2="{h-18}" stroke="#999"/>')
        svg.append(f'<text x="{ax:.1f}" y="{h-6}" text-anchor="middle" font-size="10" fill="#777">{k}</text>')
        ax += scale
    svg.append('</svg>')
    parts.append("".join(svg))
    parts.append('</div>')
    return "".join(parts)


def legend(cats=None, colors=None):
    cats = cats or FATES
    colors = colors or FCOL
    items = "".join(f'<span class="li"><span class="sw" style="background:{colors[f]}"></span>{esc(f)}</span>'
                    for f in cats)
    return f'<div class="legend">{items}</div>'


def rank_profile(groups):
    """stacked column per rank 1..10, averaged over ALL groups' posts + on-claim line."""
    fate_by_rank = defaultdict(Counter)
    onclaim = Counter()
    tot = Counter()
    for recs in groups.values():
        for rec in recs:
            for (i, fate, ng, kind, oc) in result_fates(rec):
                fate_by_rank[i][fate] += 1
                tot[i] += 1
                if oc:
                    onclaim[i] += 1
    W, H, CW = 860, 258, 64
    svg = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    for i in range(1, 11):
        n = tot[i] or 1
        x = 40 + (i - 1) * (CW + 14)
        y = H - 48
        for f in FATES:
            v = fate_by_rank[i].get(f, 0) / n
            hgt = v * (H - 78)
            y -= hgt
            svg.append(f'<rect x="{x}" y="{y:.1f}" width="{CW}" height="{hgt:.1f}" fill="{FCOL[f]}">'
                       f'<title>rank {i}, {esc(f)}: {v:.0%}</title></rect>')
        svg.append(f'<text x="{x+CW/2}" y="{H-32}" text-anchor="middle" font-size="11" fill="#555">#{i}</text>')
        oc = onclaim[i] / n
        svg.append(f'<text x="{x+CW/2}" y="{H-16}" text-anchor="middle" font-size="11" font-weight="600" '
                   f'fill="#1565c0">{oc:.0%}</text>')
    svg.append(f'<text x="40" y="{H-16}" text-anchor="end" font-size="10.5" fill="#1565c0"> </text>')
    svg.append('</svg>')
    return ('<div class="fig"><h3>4 · Signal by rank position</h3>'
            + '<p class="note">Each column is one rank, averaged over all posts. Same ' + term("fate")
            + ' colors as figure 2. The blue number under each rank is the share of that slot judged '
            + term("on-claim") + '. Note these are our positions, not Google\'s: results are reranked by '
            + 'credibility tier before anything sees them, and triage tends to pick from the top, so the '
            + 'decline across ranks is partly produced by our own ordering.</p>'
            + legend() + "".join(svg) + '</div>')


def count_hists(groups):
    """n_t / n_f / n_empty distributions per group (bins 0,1,2,3,4,5+)."""
    parts = ['<div class="fig"><h3>5 · Signal counts per claim (n_t, n_f, n_&empty;)</h3>'
             '<p class="note">Share of ' + term("targeted") + ' claims by count of supporting voices (' + term("n_t") + '), refuting voices (' + term("n_f") + '), and empty slots (' + term("n_e", "n_\u2205") + ') out of 10.</p><div class="tablewrap"><table class="ht"><tr><th></th>'
             '<th>n_t (supporting voices)</th><th>n_f (refuting voices)</th><th>n_&empty; (empty slots)</th></tr>']
    BINS = ["0", "1", "2", "3", "4", "5+"]
    for label, recs in groups.items():
        cc = [c for rec in recs for c in claim_counts(rec)]
        if not cc:
            continue
        row = [f'<tr><td class="gl">{esc(label)}<div class="dim">{len(cc)} claims</div></td>']
        for idx, col in enumerate(("t", "f", "e")):
            vals = [c[idx] if idx < 2 else min(c[2], 10) for c in cc]
            binned = Counter()
            for v0 in vals:
                if col == "e":
                    b = "0" if v0 <= 5 else "1" if v0 <= 6 else "2" if v0 <= 7 else \
                        "3" if v0 <= 8 else "4" if v0 <= 9 else "5+"
                else:
                    b = str(v0) if v0 < 5 else "5+"
                binned[b] += 1
            n = len(vals)
            cells = []
            labels_e = ["<=5", "6", "7", "8", "9", "10"] if col == "e" else BINS
            for b, bl in zip(BINS, labels_e):
                frac = binned.get(b, 0) / n
                hgt = int(frac * 46)
                color = "#2e7d32" if col == "t" else "#c62828" if col == "f" else "#9e9e9e"
                cells.append(f'<div class="vb"><div class="vbar" style="height:{hgt}px;background:{color}" title="{frac:.0%}"></div><div class="vl">{bl}</div></div>')
            row.append(f'<td><div class="vrow">{"".join(cells)}</div></td>')
        parts.append("".join(row) + "</tr>")
    parts.append('</table></div></div>')
    return "".join(parts)


def kind_mix(groups):
    parts = ['<div class="fig"><h3>6 · Source kinds in the 10 results</h3><div class="tablewrap"><table class="ht"><tr><th></th>']
    # press-release, video and social are each a rounding error; folded into other so the
    # table stays readable (Daniel 2026-07-21)
    kinds = ["news-report", "primary-official", "fact-check", "reference", "aggregator",
             "opinion", "other"]
    _FOLD = {"press-release", "video", "social"}
    parts.append("".join(f'<th>{esc(k)}</th>' for k in kinds) + '<th>' + term("on-claim") + '</th></tr>')
    for label, recs in groups.items():
        ct = Counter(); n = 0; oc = 0
        for rec in recs:
            for s in (rec.get("profile") or {}).get("source_classes") or []:
                kind = s.get("kind") or "other"
                ct["other" if kind in _FOLD or kind not in kinds else kind] += 1; n += 1
                if s.get("on_claim"): oc += 1
        if not n:
            continue
        cells = "".join(f'<td>{ct.get(k,0)/n:.0%}</td>' for k in kinds)
        parts.append(f'<tr><td class="gl">{esc(label)}</td>{cells}<td><b>{oc/n:.0%}</b></td></tr>')
    parts.append('</table></div></div>')
    return "".join(parts)


def funnel(groups):
    parts = ['<div class="fig"><h3>8 · Retrieval funnel</h3><div class="tablewrap"><table class="ht">'
             '<tr><th></th><th>posts</th><th>returned</th><th>on-claim</th><th>picked</th><th>read OK</th>'
             '<th>gave evidence</th><th>directional</th></tr>']
    for label, recs in groups.items():
        ret = onc = pick = read = gave = direc = nposts = 0
        for rec in recs:
            fates = result_fates(rec)
            if fates:
                nposts += 1
            ret += len(fates)
            sc = {s["i"]: s for s in (rec.get("profile") or {}).get("source_classes") or []}
            onc += sum(1 for (i, f, ng, k, o) in fates if o)
            rd = rec["rounds"][0] if rec.get("rounds") else {}
            pick += len((rd.get("triage") or {}).get("read") or [])
            read += sum(1 for (i, f, *_ ) in fates if f in
                        ("support-read", "refute-read", "neutral-read", "read-no-evidence"))
            gave += sum(1 for (i, f, *_ ) in fates if f in ("support-read", "refute-read", "neutral-read"))
            direc += sum(1 for (i, f, *_ ) in fates if f in ("support-read", "refute-read"))
        if not ret:
            continue
        parts.append(f'<tr><td class="gl">{esc(label)}</td><td>{nposts}</td><td>{ret}</td>'
                     f'<td>{onc} ({onc/ret:.0%})</td><td>{pick} ({pick/ret:.0%})</td>'
                     f'<td>{read} ({read/ret:.0%})</td><td>{gave} ({gave/ret:.0%})</td>'
                     f'<td><b>{direc} ({direc/ret:.0%})</b></td></tr>')
    parts.append('</table></div></div>')
    return "".join(parts)


def _outlet_ng():
    """post_id -> the source outlet's NewsGuard score, joined from the verify-input parquets."""
    import pandas as pd
    out = {}
    for f in ("dev500a_verify_input_v47.parquet", "dev500b_verify_input_v47.parquet"):
        path = Path("eval/data/survey_claims") / f
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["post_id", "ng_score"]).drop_duplicates("post_id")
        for pid, ng in zip(df.post_id.astype(str), df.ng_score):
            out[pid] = None if ng != ng else float(ng)
    return out


def _ng_tier(ng):
    if ng is None:
        return "unrated outlet"
    return ("NG 90-100" if ng >= 90 else "NG 80-89" if ng >= 80 else "NG 70-79" if ng >= 70
            else "NG 60-69" if ng >= 60 else "NG <60")


_NG_TIER_ORDER = ["NG 90-100", "NG 80-89", "NG 70-79", "NG 60-69", "NG <60", "unrated outlet"]


def ng_tier_section(tweet_recs):
    """Section 7: the raw-result reliability composition and the signal-count table, split by
    the NewsGuard tier of the SOURCE outlet (tweets only — the labeled sets are not outlet-tied)."""
    ong = _outlet_ng()
    tiers = {t: [] for t in _NG_TIER_ORDER}
    for r in tweet_recs:
        tiers[_ng_tier(ong.get(str(r["id"])))].append(r)
    rows, tbl = [], []
    for t in _NG_TIER_ORDER:
        recs = tiers[t]
        rmix = Counter(); n = 0
        nt = nf = nsig = nclaims = 0
        for rec in recs:
            if not rec.get("rounds"):
                continue
            n += 1
            for x in rec["rounds"][0].get("results") or []:
                rmix[rel_bin(x.get("rel"), x.get("ng"), x.get("domain"))] += 1
            for a, b, _ in claim_counts(rec):
                nt += a; nf += b; nclaims += 1
                if a + b == 0:
                    nsig += 1
        if not n or not nclaims:
            continue
        rows.append((f"{t}  (n={n})", {f: rmix[f] / n for f in RELS}))
        tbl.append((t, nclaims, nt / nclaims, nf / nclaims, nsig / nclaims))
    table = ('<table class="ht"><tr><th>outlet tier</th><th>claims</th>'
             '<th>mean ' + term("n_t") + '</th><th>mean ' + term("n_f")
             + '</th><th>no signal</th></tr>'
             + "".join(f'<tr><td class="gl">{esc(t)}</td><td>{c}</td><td>{a:.2f}</td>'
                       f'<td>{b:.2f}</td><td>{ns:.0%}</td></tr>' for t, c, a, b, ns in tbl)
             + '</table>')
    return ('<div class="fig"><h3>2 · By outlet reliability</h3>'
            '<p class="note">The first-query results for our own tweets, grouped by the NewsGuard rating of the '
            'outlet that posted them. The composition of the raw ten results comes out fairly flat across tiers. '
            'The table below adds what those results became once read. Claims from lower-rated outlets draw '
            'slightly fewer supporting voices and far more silence, from 5% with no directional evidence at the '
            'top tier to 22% at the bottom.</p>'
            + legend(RELS, RCOL)
            + hbar_stack(rows, "", "", cats=RELS, colors=RCOL)
            + '<h4 style="margin-top:16px">What those results became</h4>'
            + '<div class="tablewrap">' + table + '</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--averitec", default="")
    ap.add_argument("--fcgold", default="")
    ap.add_argument("--tweets", default="")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    groups = {}
    for name, path in (("averitec", args.averitec), ("fcgold", args.fcgold)):
        if path and Path(path).is_dir():
            recs = load(path)
            groups[f"{name} gold TRUE"] = [r for r in recs if r.get("gold") == "Supported"]
            groups[f"{name} gold FALSE"] = [r for r in recs if r.get("gold") == "Refuted"]
    if args.tweets and Path(args.tweets).is_dir():
        recs = load(args.tweets)
        groups["tweets, all"] = recs

        def a_status(rec, cid):
            return (rec.get("profile") or {}).get("arms", {}).get("A", {}).get(str(cid))
        def targeted_status(rec):
            if not rec.get("rounds"):
                return []
            return [a_status(rec, cid) for cid in rec["rounds"][0].get("targets") or []]
        groups["tweets resolved in round 1"] = [r for r in recs if targeted_status(r) and
                                             all(s in ("supported", "refuted", "conflicting") for s in targeted_status(r))]
        groups["tweets with a refuted claim"] = [r for r in recs if "refuted" in targeted_status(r)]
        groups["tweets continuing (open)"] = [r for r in recs if "open" in targeted_status(r)]

    urn_rows, rel_rows = [], []
    for label, recs in groups.items():
        mix = Counter(); rmix = Counter(); n = 0
        for rec in recs:
            fates = result_fates(rec)
            if not fates:
                continue
            n += 1
            for (i, f, *_ ) in fates:
                mix[f] += 1
            for r in rec["rounds"][0].get("results") or []:
                rmix[rel_bin(r.get("rel"), r.get("ng"), r.get("domain"))] += 1
        if n:
            urn_rows.append((f"{label}  (n={n})", {f: mix[f] / n for f in FATES}))
            rel_rows.append((f"{label}  (n={n})", {f: rmix[f] / n for f in RELS}))

    # ---- unrated-pool analysis (Daniel 2026-07-21) ----
    dom_ct = Counter(); kind_un = Counter(); kind_rt = Counter()
    oc_un = [0, 0]; oc_rt = [0, 0]
    fate_un = Counter(); fate_rt = Counter(); n_posts = 0
    for recs in groups.values():
        for rec in recs:
            if not rec.get("rounds"):
                continue
            rd = rec["rounds"][0]
            sc = {c["i"]: c for c in (rec.get("profile") or {}).get("source_classes") or []}
            fates = {i: f for (i, f, ng, k, ocl) in result_fates(rec)}
            for r in rd.get("results") or []:
                unr = rel_bin(r.get("rel"), r.get("ng"), r.get("domain")) == "unrated"
                cls = sc.get(r["i"]) or {}
                (oc_un if unr else oc_rt)[0] += bool(cls.get("on_claim"))
                (oc_un if unr else oc_rt)[1] += 1
                (fate_un if unr else fate_rt)[fates.get(r["i"], "?")] += 1
                if unr:
                    dom_ct[r.get("domain") or "?"] += 1
                    kind_un[cls.get("kind") or "?"] += 1
                else:
                    kind_rt[cls.get("kind") or "?"] += 1
    # groups double-count posts across tweet subsets; scale is comparative, fine

    fate_pairs_un = [(f, fate_un.get(f, 0)) for f in FATES]
    fate_pairs_rt = [(f, fate_rt.get(f, 0)) for f in FATES]
    donuts = ('<div class="donuts">'
              + donut(fate_pairs_un, "unrated", f"{sum(fate_un.values())} results")
              + donut(fate_pairs_rt, "rated", f"{sum(fate_rt.values())} results")
              + '</div>')
    top_doms = "".join(
        f'<li><span class="dn">{esc(d)}</span><span class="dc">{c}</span></li>'
        for d, c in dom_ct.most_common(18))
    kinds_tbl = "".join(
        f'<tr><td>{esc(k)}</td><td class="num">{kind_un.get(k,0)/(sum(kind_un.values()) or 1):.0%}</td>'
        f'<td class="num">{kind_rt.get(k,0)/(sum(kind_rt.values()) or 1):.0%}</td></tr>'
        for k in ("news-report", "reference", "social", "other", "opinion", "aggregator",
                  "primary-official", "fact-check", "video", "press-release"))
    unrated_section = f"""
<div class="fig"><h3>7 · The unrated results</h3>
<p class="note">{sum(dom_ct.values())} unrated results across {len(dom_ct)} distinct domains. Most domains appear once.
{term("on-claim", "On-claim")} rate is {oc_un[0]/max(oc_un[1],1):.0%}, against {oc_rt[0]/max(oc_rt[1],1):.0%} for rated results. Triage skips them more often. The ones that do get read produce real evidence.</p>
<div class="u6">
<div class="u6a"><h4>What became of them</h4>
<p class="note">Share of each group's results by {term("fate")}. Rated results become supporting or refuting evidence about twice as often; the gap is mostly triage skipping unrated sources.</p>
{donuts}{legend()}</div>
<div class="u6b"><h4>Source kinds, unrated vs rated</h4><table class="ht slim"><tr><th>kind</th><th>unrated</th><th>rated</th></tr>{kinds_tbl}</table></div>
</div>
<h4>Most frequent unrated domains</h4>
<ul class="doms">{top_doms}</ul>
<h4>Three ways to estimate their reliability</h4>
<ol class="routes">
<li><b>Curated lists (code, free).</b> The most frequent unrated domains fall into a few families. Wikipedia counts as one corroborating voice and never closes a claim. Academic and data institutions (presidency.ucsb.edu, researchgate, statista, macrotrends, .edu) can join the institutional tier. UGC platforms (reddit, threads, quora, medium, scribd) are never evidence and are now blocked at the search level.</li>
<li><b>Third-party ratings for the international press.</b> Many unrated news domains are just non-US outlets that NewsGuard does not cover (Times of India, ABS-CBN, Philstar, GMA Network, The Quint). Candidate sources: Media Bias/Fact Check, Wikipedia's perennial sources list, and the Lin et al. domain credibility scores (about 11k domains, free).</li>
<li><b>Agreement-based trust (later, free).</b> As runs accumulate, a domain whose stances agree with NG 90+ and institutional voices on the same claims earns an empirical reliability score. No external rater needed.</li>
</ol></div>"""

    body = [
        '<header><h1>Truth Odds: what the first Serper query returns</h1>'
        '<p class="sub">One query per post, 10 results, first round only. Labeled claim sets ('
        + term("averitec", "AVeriTeC") + ' 119 true / 200 false and ' + term("fcgold", "fc-gold")
        + ' 32 / 98, both with the date ceiling on) beside '
        + term("tweets", "dev-1000") + ' tweet claims (482 posts, 832 claims, no ceiling). '
        'Only targeted claims are counted. Every figure aggregates over claims.</p></header>',
        '<div class="keybox"><b>Key findings.</b><ul>'
        '<li>All three assumptions of the PI\'s model hold:'
        '<ul class="sub">'
        '<li>True claims draw more supporting voices than false ones: 2.4 (AVeriTeC) and 3.6 (fc-gold) '
        'against 0.6 for false claims in both sets.</li>'
        '<li>False claims draw more refuting voices than true ones: 1.8 (AVeriTeC) and 0.9 (fc-gold) '
        'against 0.2 and 0.1 for true claims.</li>'
        '<li>False claims more often draw nothing at all: 38% of false fc-gold claims get no directional '
        'signal, against 6% of true ones (AVeriTeC: 10% against 3%).</li>'
        '</ul></li>'
        '<li>Silence penalty likely depends on claim type. Empty slots are worth about \u22120.3 log-odds '
        'for fc-gold and about 0 for AVeriTeC.</li>'
        '<li>About a third of raw results have no NewsGuard rating. Figure 7 shows what they are.</li>'
        '<li>Grouped by the posting outlet\'s rating (figure 2), the raw result mix is flat, but claims from lower-rated outlets end up with far more silence: 5% no directional evidence at the top tier, 22% at the bottom.</li>'
        '</ul></div>',
        hbar_stack(rel_rows, "1 · What Serper returns",
                   "Average count of the 10 results by reliability class, before any selection: "
                   + term("institutional") + " (includes .edu), the five NewsGuard bins, and " + term("unrated") + ".",
                   cats=RELS, colors=RCOL, lgnd=True),
        ng_tier_section(load(args.tweets)) if args.tweets and Path(args.tweets).is_dir() else '',
        hbar_stack(urn_rows, "3 · What each result became",
                   "Average " + term("fate") + " of the 10 results: a supporting or refuting " + term("voice")
                   + ", a " + term("snippet voice") + ", or nothing. Triage is instructed to prefer "
                   + term("institutional") + " and reliably rated sources when choosing what to read, so which "
                   "results become evidence reflects that instruction as much as the sources themselves. "
                   "This skews every figure below.", lgnd=True),
        rank_profile(groups),
        count_hists(groups),
        kind_mix(groups),
        unrated_section,
        funnel(groups),
        '<footer>Working doc: src/docs/truth_odds.md. Runs: truthodds_profile_averitec_v2 / _fcgold_v2 / _dev1000_s150_v2 '
        '(first round only, stock prompts). Institutional means gov TLDs, .edu, and the curated allowlist. '
        'Primary is reserved for actual primary records identified at READ.</footer>',
    ]
    head = """<title>Truth Odds: first-query evidence profile</title>
<style>
:root{--ink:#1a1a1a;--sub:#5f6368;--line:#e8e6e1;--paper:#fdfcfa;--accent:#283593}
html,body{background:var(--paper);color:var(--ink);color-scheme:light}
body{margin:0;font:15px/1.6 "Avenir Next","Segoe UI",-apple-system,BlinkMacSystemFont,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:44px 28px 110px}
header{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:8px}
h1{font-size:27px;font-weight:700;letter-spacing:-.3px;margin:0 0 6px}
.sub{color:var(--sub);font-size:14px;max-width:70ch}
.keybox{background:#fff;border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:6px;padding:14px 18px;margin:22px 0;font-size:13.5px}
.keybox b{color:var(--accent)}
.keybox li{margin:4px 0}
.keybox ul.sub{margin:6px 0 8px;padding-left:20px}
.keybox ul.sub li{margin:3px 0;color:#333}
h3{font-size:17px;font-weight:700;margin:44px 0 6px;letter-spacing:-.2px}
h4{font-size:13.5px;font-weight:700;margin:14px 0 6px;color:#333}
.fno{display:inline-block;background:var(--ink);color:#fff;border-radius:4px;font-size:12px;width:20px;height:20px;line-height:20px;text-align:center;margin-right:9px;vertical-align:2px}
.note{color:var(--sub);font-size:13px;margin:2px 0 12px}
.fig{margin:8px 0 12px;background:#fff;border:1px solid var(--line);border-radius:8px;padding:16px 20px 12px;box-shadow:0 1px 2px #0000000a}
.fig .fig{border:none;box-shadow:none;padding:0;margin:0}
.legend{margin:10px 0 4px}
.li{display:inline-block;font-size:11.5px;color:#444;margin:0 12px 4px 0}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px;border:1px solid #00000014}
table.ht{border-collapse:collapse;font-size:12.5px;width:100%}
.ht th,.ht td{border-bottom:1px solid var(--line);padding:5px 9px;text-align:center}
.ht th{background:none;border-bottom:2px solid var(--ink);font-weight:700;font-size:11.5px;text-transform:uppercase;letter-spacing:.5px;color:#333}
.ht tr:nth-child(even) td{background:#faf9f6}
.ht.slim{width:auto}
.gl{text-align:left!important;font-weight:600;min-width:170px}
.num{text-align:right!important;font-variant-numeric:tabular-nums}
.dim{color:#999;font-size:11px;font-weight:400}
.cols{display:grid;grid-template-columns:minmax(240px,1fr) 2fr;gap:26px}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
ol.routes{font-size:13.5px;padding-left:20px}
ol.routes li{margin:8px 0;max-width:82ch}
.donuts{display:flex;gap:14px;flex-wrap:wrap;margin-top:2px}
.u6{display:grid;grid-template-columns:minmax(420px,1.4fr) minmax(240px,1fr);gap:28px;align-items:start;margin-bottom:6px}
@media(max-width:820px){.u6{grid-template-columns:1fr}}
.u6a .legend{margin-top:2px}
ul.doms{list-style:none;padding:0;margin:4px 0 10px;columns:3;column-gap:26px;font-size:12.5px}
@media(max-width:700px){ul.doms{columns:2}}
ul.doms li{display:flex;justify-content:space-between;gap:10px;break-inside:avoid;border-bottom:1px solid var(--line);padding:2px 0}
.dn{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dc{color:var(--sub);font-variant-numeric:tabular-nums}
.donut{flex:0 0 210px}
.vrow{display:flex;align-items:flex-end;gap:4px;height:64px}
.vb{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:64px}
.vbar{width:16px;border-radius:2px 2px 0 0}
.vl{font-size:9.5px;color:#888}
svg{max-width:100%;height:auto}
footer{margin-top:40px;color:var(--sub);font-size:12px;border-top:1px solid var(--line);padding-top:12px}
.gloss{border-bottom:1px dotted #8a8a8a;cursor:help;position:relative}
.gloss:hover::after{content:attr(data-tip);position:absolute;left:0;top:calc(100% + 6px);z-index:30;background:#1a1a1a;color:#fff;font-size:12px;line-height:1.45;font-weight:400;padding:8px 11px;border-radius:6px;width:300px;max-width:70vw;white-space:normal;box-shadow:0 4px 14px #0003}
.legend{background:#faf9f6;border:1px solid var(--line);border-radius:6px;padding:7px 12px 4px;margin:10px 0 8px;display:inline-block}
.tablewrap{overflow-x:auto;max-width:100%}
</style><div class="wrap">"""
    Path(args.output).write_text(head + "".join(body) + "</div>")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
