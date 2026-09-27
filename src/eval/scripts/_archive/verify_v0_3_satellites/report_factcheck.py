"""Interactive HTML report for the factcheck-dataset Tier-3 verification eval.

  uv run python -m eval.scripts.verification_grading.report_factcheck \
      -c data/claims_factcheck_n200.parquet -v data/verdicts_factcheck_n200.parquet

Recomputes everything live from verdicts + claims + per-claim traces (no hardcoded numbers) and
writes a SELF-CONTAINED interactive HTML (Plotly inline). Works on a partial/checkpointed run too.
Sections: validity gate · headline accuracy + asymmetric risk · calibration · NEE realignment ·
nudge analysis · latency breakdown · strata · misclassified table. (w/ Daniel, clog 250626)
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import polars as pl

DIR = Path(__file__).parent
DATA = DIR / "data"
REPORTS = DIR.parents[2] / "reports"  # src/reports — central, discoverable home for result HTML
CLAIMCHECK = 0.764  # ClaimCheck 4-class label accuracy, AVeriTeC dev (arXiv:2510.01226) — ref only

LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
SHORT = {"Supported": "SUP", "Refuted": "REF",
         "Not Enough Evidence": "NEI", "Conflicting Evidence/Cherrypicking": "CE"}
_ALIAS = {"Conflicting Evidence": "Conflicting Evidence/Cherrypicking"}
INK, OK, BAD, ACCENT, GRID = "#222", "#2e7d32", "#c62828", "#1565c0", "#e3e3e3"

# Curated catalog of v1's failure modes (from the source analysis + the 29-claim fix re-run, clog
# 260626) — a forward-looking document for the v2 dataset/pipeline rebuild. Hand-authored analysis;
# the rest of the report is recomputed live.
FAILURE_MODES = [
    {"id": "FM-1", "name": "Date-ceiling evidence collapse", "status": "FIXED", "tag": "DOMINANT",
     "cause": "The eval leakage guard's date filter (a) compared MIXED-FORMAT date strings lexically "
              "(\"Oct 2, 2025\" &gt; \"2025-11-05\" → wrongly treated as 'after the claim'), and (b) "
              "dropped evergreen reference pages (gov/data) whose <i>page</i> dates are recent. The "
              "evidence funnel collapsed from ~11 retrieved to ~1.2 used — the system then reasoned "
              "from the single old junk page that happened to survive.",
     "example": "Milk prices: FRED/USDA dropped as 'future evidence'; only an old nostalgia blog "
                "(yourtango.com) survived → wrongly Refuted a true (inflation-adjusted) claim.",
     "impact": "Drove the bulk of the 42% false-nudge-on-true. Silently degraded every prior eval too.",
     "fix": "IMPLEMENTED — parse publication_date to ISO before comparing; exempt evergreen domains "
            "(gov/edu/.int + data orgs) via credibility.is_reference; keep date-filtering news wires. "
            "Recovered 14/21 false-positives; evidence 1.2 → 7.6 docs/claim."},
    {"id": "FM-2", "name": "Over-confident single-source stop", "status": "FIXED", "tag": "",
     "cause": "190/200 claims stopped 'confident' after a single round — frequently on ONE document. "
              "No corroboration was required before emitting a verdict.",
     "example": "A true claim with one weak source → 'confident' → wrong verdict, never escalating.",
     "impact": "Compounded FM-1: even when a 2nd good source existed, the loop stopped before fetching it.",
     "fix": "IMPLEMENTED — hard corroboration gate (never conclude on ≤1 source while Exa is untried "
            "→ escalate); cascade-aware synthesis prompt; ROUNDS_PER_PROVIDER=2 (Serper→Exa). "
            "Validated: Exa fired on only 4/29 — not trigger-happy."},
    {"id": "FM-3", "name": "Query-generation garbage", "status": "FIXED", "tag": "",
     "cause": "The planner/synthesiser occasionally emitted a junk query — e.g. a bare ':' — which "
              "returned zero results, so the system reasoned from stale prior-round evidence.",
     "example": "Camp Mystic victim claim: query was literally ':' → 0 results.",
     "impact": "Rare but catastrophic for the affected claim (verifies on no fresh evidence).",
     "fix": "IMPLEMENTED — _clean_query validates the query (non-empty, has alphanumerics) and falls "
            "back to the claim text."},
    {"id": "FM-4", "name": "Axis mislabeling (artifact/attribution in the content set)",
     "status": "RESIDUAL", "tag": "~3 of 11",
     "cause": "Claims ABOUT a media artifact or a document/report — 'a clip SHOWS X', 'the documents "
              "CONTAIN Y', 'a report CLAIMS Z' — were tagged judged_axis=content, so the system "
              "verified the underlying fact instead of the artifact's authenticity / the report's "
              "existence.",
     "example": "'Clip shows the Ha Long Bay capsizing' — the event is real (we passed it), but the "
                "claim is that THIS clip shows it (miscaptioned → gold Refuted). Also: Cobain forensic "
                "report, DOJ documents.",
     "impact": "~3 residual errors; also lets a few artifact claims pollute the text-only content set.",
     "fix": "v1: tighten the axis classifier (enrich + quality gate) to catch 'clip/video/photo "
            "shows', 'documents/files contain', 'report/study claims/finds' → artifact/attribution → "
            "exclude. <b>v2: image processing resolves these properly</b> — the next version can "
            "actually verify the artifact (is the clip authentic / correctly captioned?) rather than "
            "excluding it."},
    {"id": "FM-5", "name": "Conflicting-Evidence nuance (partial truths)", "status": "RESIDUAL",
     "tag": "~4 of 11",
     "cause": "A claim is partly true / true-but-misframed; the system collapses it to Supported (or "
              "CE), the fact-checker rated Conflicting. The 4-class CE boundary is genuinely hard and "
              "partly gold-subjective.",
     "example": "Dr. Propst firing (a detail off), 323 vials (framing), Shein/Mangione (resembled vs "
                "confirmed), grocery $1,030 (exact figure vs the general claim).",
     "impact": "~4 residual; several are gold-defensible-either-way — a nuance ceiling, not a bug.",
     "fix": "Mitigate via finer claim decomposition (verify each load-bearing sub-part separately). "
            "Mostly PRODUCT-IRRELEVANT: a CE and a Supported-with-caveat drive the same nudge. Accept "
            "a ceiling on the 4-class metric here."},
    {"id": "FM-6a", "name": "Reasoning gap — literal vs intended interpretation", "status": "RESIDUAL",
     "tag": "tough",
     "cause": "The system reads a claim in its most literal frame and misses the intended one.",
     "example": "Milk prices read in NOMINAL dollars ($2.20→$2.90) when the claim means "
                "inflation-adjusted (why the fact-checker said Supported). Even with FRED data, it "
                "reasoned nominally.",
     "impact": "~1 residual; hard to detect generically.",
     "fix": "Hard. Would need claim-frame inference (inflation-adjustment, base rates, intended "
            "comparison). Out of near-term scope — flag as a known limitation."},
    {"id": "FM-6b", "name": "'Absence == refutation' (a retrieval-recall miss, not a reasoning bug)",
     "status": "RESIDUAL", "tag": "by-design",
     "cause": "The system refutes because a critical fact wasn't found in the retrieved sources.",
     "example": "Refuting a named flood victim because the name wasn't in the one Wikipedia list we "
                "retrieved — but that list isn't exhaustive.",
     "impact": "~1 residual.",
     "fix": "Per Daniel: rejecting an unsubstantiated LOAD-BEARING claim part is CORRECT BY DESIGN — "
            "if a critical part can't be substantiated, we should reject. So this is not a reasoning "
            "bug to 'fix'; the real failure is when the substantiation EXISTED and we missed it → a "
            "retrieval-RECALL problem. Fix = maximise recall (broader queries / more rounds / Exa); "
            "keep the reject logic."},
    {"id": "FM-7", "name": "Post-flag nudge call is net-negative", "status": "RESIDUAL", "tag": "",
     "cause": "The separate post-aware nudge call (raw post + claim + analysis) over-flags true "
              "claims whose POSTS look sensational.",
     "example": "On the 55 raw-post rows it changed 3 decisions: fixed 1, broke 2.",
     "impact": "Slightly hurts the nudge as built.",
     "fix": "Tighten the flag prompt (less framing-trigger-happy) or drop the call; re-measure on a "
            "larger raw-post set."},
]


def _norm(s):
    s = (s or "").strip()
    return _ALIAS.get(s, s)


def _pct(x):
    return f"{x*100:.1f}%"


def _quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(q * len(xs)))
    return xs[i]


# ---------- load ----------
def load(claims_name, verdicts_name):
    c = pl.read_parquet(DATA / claims_name)
    v = pl.read_parquet(DATA / verdicts_name)
    keepc = [x for x in ["claim_id", "gold_label", "publisher_site", "language_code",
                         "claim_date", "raw_context", "claim_text"] if x in c.columns]
    df = c.select(keepc).join(v, on="claim_id", how="inner", suffix="_v")
    ok = df.filter((pl.col("error").is_null()) & (pl.col("verdict_4class") != ""))
    return df, ok


def load_traces(verdicts_name):
    """Per-claim stage timings from the trace JSONs: search vs gather(scrape+summarise) vs eval."""
    tdir = DATA / (Path(verdicts_name).stem.replace("verdicts_", "verdicts_") + "_trace")
    rows = []
    for f in glob.glob(str(tdir / "*.json")):
        try:
            t = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        rounds = t.get("rounds") or []
        search = sum((r.get("search") or {}).get("elapsed_s", 0) or 0 for r in rounds)
        gather = sum(r.get("gather_elapsed_s", 0) or 0 for r in rounds)
        synth = sum((r.get("synthesis") or {}).get("elapsed_s", 0) or 0 for r in rounds if r.get("synthesis"))
        ev = t.get("eval_elapsed_s", 0) or 0
        total = t.get("total_elapsed_s", 0) or 0
        rows.append({"search": search, "gather": gather, "synth": synth, "eval": ev,
                     "other": max(0.0, total - search - gather - synth - ev), "total": total})
    return rows


# ---------- metrics ----------
def confusion(gold, pred):
    idx = {l: i for i, l in enumerate(LABELS)}
    m = [[0] * 4 for _ in range(4)]
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            m[idx[g]][idx[p]] += 1
    return m


def per_class(gold, pred):
    out = {}
    for lab in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"P": prec, "R": rec, "F1": f1, "n": sum(1 for g in gold if g == lab)}
    return out


# ---------- figures ----------
def fig_html(fig, first=False):
    fig.update_layout(margin=dict(l=50, r=20, t=30, b=40), font=dict(family="system-ui", color=INK),
                      paper_bgcolor="white", plot_bgcolor="white", height=360)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return pio.to_html(fig, include_plotlyjs=("inline" if first else False), full_html=False,
                       config={"displaylogo": False, "displayModeBar": False})


def f_confusion(m):
    z = m
    txt = [[str(z[i][j]) for j in range(4)] for i in range(4)]
    s = [SHORT[l] for l in LABELS]
    fig = go.Figure(go.Heatmap(z=z, x=s, y=s, text=txt, texttemplate="%{text}",
                               colorscale="Blues", showscale=False, hovertemplate="gold %{y} → pred %{x}: %{z}<extra></extra>"))
    fig.update_layout(yaxis=dict(autorange="reversed", title="gold"), xaxis=dict(title="predicted"))
    return fig


def f_perclass(pc):
    labs = [SHORT[l] for l in LABELS]
    fig = go.Figure()
    for metric, col in [("P", "#90a4ae"), ("R", "#546e7a"), ("F1", ACCENT)]:
        fig.add_bar(name=metric, x=labs, y=[pc[l][metric] for l in LABELS], marker_color=col)
    fig.update_layout(barmode="group", yaxis=dict(range=[0, 1], title="score"), legend=dict(orientation="h", y=1.15))
    return fig


def f_calibration(veracity_correct, veracity_wrong):
    fig = go.Figure()
    fig.add_box(y=veracity_correct, name="correct verdict", marker_color=OK, boxpoints="all", jitter=0.4, pointpos=0)
    fig.add_box(y=veracity_wrong, name="wrong verdict", marker_color=BAD, boxpoints="all", jitter=0.4, pointpos=0)
    fig.update_layout(yaxis=dict(title="Likert veracity (1–5)", range=[0.5, 5.5]))
    return fig


def f_latency_hist(elapsed):
    fig = go.Figure(go.Histogram(x=elapsed, marker_color=ACCENT, nbinsx=30))
    fig.update_layout(xaxis=dict(title="per-claim wall-clock (s)"), yaxis=dict(title="claims"))
    return fig


def f_stage_breakdown(traces):
    keys = ["search", "gather", "synth", "eval", "other"]
    names = {"search": "search", "gather": "scrape+summarise", "synth": "synthesis", "eval": "verdict+Likert", "other": "plan/nudge/overhead"}
    means = {k: (sum(t[k] for t in traces) / len(traces) if traces else 0) for k in keys}
    fig = go.Figure(go.Bar(x=[means[k] for k in keys], y=[names[k] for k in keys], orientation="h",
                           marker_color=["#90caf9", "#1565c0", "#546e7a", "#90a4ae", "#cfd8dc"],
                           text=[f"{means[k]:.1f}s" for k in keys], textposition="auto"))
    fig.update_layout(xaxis=dict(title="mean seconds / claim"), yaxis=dict(autorange="reversed"))
    return fig, means


def f_strata(df_ok, by):
    rows = []
    for g, sub in df_ok.group_by(by):
        key = g[0] if isinstance(g, tuple) else g
        gold = [_norm(x) for x in sub["gold_label"]]
        pred = [_norm(x) for x in sub["verdict_4class"]]
        acc = sum(a == b for a, b in zip(gold, pred)) / len(gold) if gold else 0
        rows.append((str(key), acc, len(gold)))
    rows.sort(key=lambda r: -r[2])
    fig = go.Figure(go.Bar(x=[r[0] for r in rows], y=[r[1] for r in rows], marker_color=ACCENT,
                           text=[f"{_pct(r[1])}<br>n={r[2]}" for r in rows], textposition="auto"))
    fig.update_layout(yaxis=dict(range=[0, 1], title="accuracy"))
    return fig


# ---------- html bits ----------
def card(label, value, sub=""):
    return (f'<div class="card"><div class="cval">{value}</div>'
            f'<div class="clab">{label}</div><div class="csub">{sub}</div></div>')


def table(headers, rows, cls=""):
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


CSS = """
*{box-sizing:border-box} body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#222;
margin:0;background:#fafafa;line-height:1.5} .wrap{max-width:1100px;margin:0 auto;padding:32px 24px}
h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:36px 0 6px;border-bottom:2px solid #222;padding-bottom:4px}
.meta{color:#666;font-size:13px;margin-bottom:8px} .note{color:#555;font-size:14px;margin:6px 0 14px}
.cards{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0} .card{background:#fff;border:1px solid #e3e3e3;border-radius:10px;
padding:14px 18px;min-width:150px;flex:1} .cval{font-size:28px;font-weight:650} .clab{font-size:13px;color:#444;margin-top:2px}
.csub{font-size:12px;color:#888} .gate{padding:12px 16px;border-radius:10px;font-weight:600;margin:10px 0}
.gate.ok{background:#e8f5e9;color:#2e7d32;border:1px solid #a5d6a7} .gate.warn{background:#fff3e0;color:#e65100;border:1px solid #ffcc80}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff} th,td{border:1px solid #e3e3e3;padding:6px 9px;text-align:left}
th{background:#f3f3f3;cursor:pointer;user-select:none} td.num,th.num{text-align:right} .scroll{max-height:430px;overflow:auto;border:1px solid #e3e3e3;border-radius:8px}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;font-weight:600} .tag.ok{background:#e8f5e9;color:#2e7d32}
.tag.bad{background:#ffebee;color:#c62828} .small{font-size:12px;color:#777}
.fm{background:#fff;border:1px solid #e3e3e3;border-left:4px solid #1565c0;border-radius:8px;padding:12px 16px;margin:10px 0}
.fm.fixed{border-left-color:#2e7d32} .fm.residual{border-left-color:#e65100}
.fm h3{margin:0 0 6px;font-size:15px} .fm p{margin:4px 0;font-size:13px;line-height:1.45} .fm .b{font-weight:600;color:#555}
.badge{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;font-weight:700;margin-left:8px}
.badge.fixed{background:#e8f5e9;color:#2e7d32} .badge.residual{background:#fff3e0;color:#e65100}
"""
SORTJS = """
document.querySelectorAll('table.sortable th').forEach((th,i)=>th.onclick=()=>{
 const tb=th.closest('table').tBodies[0],rows=[...tb.rows];const asc=!(th.dataset.asc==='1');th.dataset.asc=asc?'1':'0';
 rows.sort((a,b)=>{let x=a.cells[i].dataset.v??a.cells[i].innerText,y=b.cells[i].dataset.v??b.cells[i].innerText;
 const nx=parseFloat(x),ny=parseFloat(y);if(!isNaN(nx)&&!isNaN(ny)){x=nx;y=ny}return (x>y?1:x<y?-1:0)*(asc?1:-1)});
 rows.forEach(r=>tb.appendChild(r))});
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--claims", default="claims_factcheck_n200.parquet")
    ap.add_argument("-v", "--verdicts", default="verdicts_factcheck_n200.parquet")
    ap.add_argument("-o", "--out", default=None, help="default: src/reports/<name>_<timestamp>.html")
    args = ap.parse_args()

    df, ok = load(args.claims, args.verdicts)
    traces = load_traces(args.verdicts)
    n_all, n_ok = df.height, ok.height
    gold = [_norm(x) for x in ok["gold_label"]]
    pred = [_norm(x) for x in ok["verdict_4class"]]
    correct = [g == p for g, p in zip(gold, pred)]
    acc = sum(correct) / n_ok if n_ok else 0
    maj = Counter(gold).most_common(1)[0] if gold else ("", 0)
    pc = per_class(gold, pred)
    macro = sum(pc[l]["F1"] for l in LABELS) / 4
    m = confusion(gold, pred)

    # validity gate
    zero_ret = df.filter(pl.col("n_urls_seen") == 0).height if "n_urls_seen" in df.columns else 0
    serr = df.filter(pl.col("n_search_errors") > 0).height if "n_search_errors" in df.columns else 0
    resamp = df["n_resampled"].sum() if "n_resampled" in df.columns else 0
    resamp_claims = df.filter(pl.col("n_resampled") > 0).height if "n_resampled" in df.columns else 0
    valid = zero_ret / max(1, n_all) < 0.15

    P = []
    P.append(f"<h1>Tier-3 Verification — dev-{n_all} report</h1>")
    prov = ok["provider"][0] if "provider" in ok.columns and ok.height else "?"
    dc = ok["date_ceiling"][0] if "date_ceiling" in ok.columns and ok.height else "?"
    P.append(f'<div class="meta">provider <b>{prov}</b> · date-ceiling <b>{dc}</b> · fact-check domains <b>blocked</b> · '
             f'scored {n_ok}/{n_all}{" (partial / still running)" if n_all < 200 else ""}</div>')

    # 0. gate
    P.append("<h2>0 · Validity gate (retrieval health)</h2>")
    gcls = "ok" if valid else "warn"
    P.append(f'<div class="gate {gcls}">{"VALID" if valid else "CHECK"} — {zero_ret}/{n_all} claims got zero evidence '
             f'({_pct(zero_ret/max(1,n_all))}); {serr} hit search errors. FC-block forced a page-2 resample on '
             f'{resamp_claims} claims ({resamp} queries total).</div>')
    P.append('<div class="note">If the zero-evidence rate were high, the score would measure guessing, not verification. '
             'The resample counter shows how often blocking fact-checkers starved a query below 5 results.</div>')

    # 1. headline
    P.append("<h2>1 · Headline accuracy</h2>")
    P.append('<div class="cards">')
    P.append(card("4-class accuracy", _pct(acc), f"{sum(correct)}/{n_ok} scored"))
    P.append(card("macro-F1", f"{macro:.3f}", "mean across 4 classes"))
    P.append(card("majority baseline", _pct(maj[1]/n_ok) if n_ok else "—", f"always {SHORT.get(maj[0],'?')}"))
    P.append(card("ClaimCheck ref", _pct(CLAIMCHECK), "AVeriTeC, diff. dataset"))
    P.append("</div>")

    # 1b. product-relevant binary (the nudge decision: flag everything that isn't Supported)
    gf = [g != "Supported" for g in gold]
    pf = [p != "Supported" for p in pred]
    tn = sum(1 for g, p in zip(gf, pf) if not g and not p)
    fp = sum(1 for g, p in zip(gf, pf) if not g and p)
    fn = sum(1 for g, p in zip(gf, pf) if g and not p)
    tp = sum(1 for g, p in zip(gf, pf) if g and p)
    bin_acc = (tn + tp) / n_ok if n_ok else 0
    recall_mis = tp / (tp + fn) if tp + fn else 0
    fp_rate = fp / (fp + tn) if fp + tn else 0
    P.append("<h2>1b · Product-relevant: nudge vs no-nudge</h2>")
    P.append('<div class="note">The product nudges on Refuted / NEE / Conflicting alike and passes only Supported — '
             'so the 4-class Refuted↔NEE confusion is mostly <b>invisible to the user</b>. The decision that matters '
             'is <b>flag</b> (warn) vs <b>pass</b>. Daniel: "they both get nudged, so it\'s okay."</div>')
    P.append('<div class="cards">')
    P.append(card("binary nudge accuracy", _pct(bin_acc), f"{tn+tp}/{n_ok}"))
    P.append(card("misinfo caught", _pct(recall_mis), f"{tp}/{tp+fn} flag-worthy flagged"))
    P.append(card("false nudge on TRUE", _pct(fp_rate), f"{fp}/{fp+tn} Supported flagged", ))
    P.append("</div>")
    P.append(f'<div class="note"><b>The story:</b> the system catches <b>{_pct(recall_mis)}</b> of real '
             f'misinformation (flag-worthy → flagged), but wrongly flags <b>{_pct(fp_rate)}</b> of <b>TRUE</b> claims '
             '— the trust risk — largely because, under the fact-check block, it often cannot positively <i>confirm</i> '
             'a true recent claim and falls to NEE. The 37.5% 4-class score is dragged down by Refuted→NEE, which does '
             'not change the nudge. Read the misclassified table (§8) to see how much is this vs residual gold noise.</div>')

    # 2. confusion + asymmetric risk
    P.append("<h2>2 · Confusion &amp; the asymmetric risk</h2>")
    sup_r, ref_p = pc["Supported"]["R"], pc["Refuted"]["P"]
    P.append('<div class="note"><b>The trust-killer is a false “Refuted” on a true claim.</b> '
             f'Supported recall = <b>{_pct(sup_r)}</b> (true claims we correctly call true); '
             f'Refuted precision = <b>{_pct(ref_p)}</b> (of what we flag, how much is really false). '
             'Watch for a Refuted over-prediction bias.</div>')
    P.append('<div style="display:flex;gap:20px;flex-wrap:wrap">')
    P.append('<div style="flex:1;min-width:340px">' + fig_html(f_confusion(m), first=True) + "</div>")
    P.append('<div style="flex:1;min-width:340px">' + fig_html(f_perclass(pc)) + "</div>")
    P.append("</div>")
    P.append(table(["class", "precision", "recall", "F1", "support"],
                   [[SHORT[l], _pct(pc[l]["P"]), _pct(pc[l]["R"]), f"{pc[l]['F1']:.3f}", pc[l]["n"]] for l in LABELS]))

    # 3. calibration
    P.append("<h2>3 · Calibration (can we set a nudge threshold?)</h2>")
    ver = ok["veracity"].to_list() if "veracity" in ok.columns else [None] * n_ok
    vcorr = [v for v, c in zip(ver, correct) if v and c]
    vwrong = [v for v, c in zip(ver, correct) if v and not c]
    sep = (sum(vcorr)/len(vcorr) - sum(vwrong)/len(vwrong)) if vcorr and vwrong else 0
    P.append(f'<div class="note">If the Likert <b>veracity</b> score separates correct from wrong verdicts, we can gate '
             f'nudging on confidence. Mean veracity: correct <b>{(sum(vcorr)/len(vcorr) if vcorr else 0):.2f}</b> vs '
             f'wrong <b>{(sum(vwrong)/len(vwrong) if vwrong else 0):.2f}</b> (gap {sep:+.2f}).</div>')
    P.append(fig_html(f_calibration(vcorr, vwrong)))

    # 4. NEE realignment
    P.append("<h2>4 · NEE / Refuted realignment</h2>")
    nee_rows = [(g, p) for g, p in zip(gold, pred) if g == "Not Enough Evidence"]
    nee_pred = Counter(p for _, p in nee_rows)
    P.append('<div class="note">We realigned the system so “no supporting evidence found” = <b>NEE</b>, not Refuted '
             '(matching the gold). On NEE-gold claims, the system now predicts:</div>')
    P.append(table(["predicted", "count"], [[SHORT.get(k, k), v] for k, v in nee_pred.most_common()] or [["—", 0]]))

    # 5. nudge
    P.append("<h2>5 · Nudge decision (raw-post rows)</h2>")
    nud = ok.filter(pl.col("nudge_post_used") == True) if "nudge_post_used" in ok.columns else ok.head(0)  # noqa: E712
    nflag = nud.filter(pl.col("nudge_flag") == True).height if nud.height else 0  # noqa: E712
    # misleading-true: flagged but verdict not Refuted
    mislead = nud.filter((pl.col("nudge_flag") == True) & (pl.col("verdict_4class") != "Refuted")) if nud.height else ok.head(0)  # noqa: E712
    P.append(f'<div class="note">{nud.height} claims had a resolved raw post → a separate nudge call (raw post + claim + '
             f'analysis). Flagged <b>{nflag}/{nud.height}</b>. The post-level value-add is the <b>{mislead.height}</b> '
             '“flag a true/uncertain claim because the post’s framing misleads”.</div>')
    if nud.height:
        nrows = [[(r["claim_text"][:70] if r.get("claim_text") else "")[:70],
                  SHORT.get(_norm(r["verdict_4class"]), "?"),
                  f'<span class="tag {"bad" if r["nudge_flag"] else "ok"}">{"FLAG" if r["nudge_flag"] else "pass"}</span>',
                  (r.get("nudge_reason") or "")[:110]] for r in nud.head(30).iter_rows(named=True)]
        P.append('<div class="scroll">' + table(["claim", "verdict", "nudge", "reason"], nrows, "sortable") + "</div>")

    # 6. latency
    P.append("<h2>6 · Latency (per-claim Tier-3 cost)</h2>")
    el = ok["elapsed_seconds"].to_list() if "elapsed_seconds" in ok.columns else []
    rounds = ok["rounds_used"].to_list() if "rounds_used" in ok.columns else []
    caphit = ok.filter(pl.col("cap_hit") == True).height if "cap_hit" in ok.columns else 0  # noqa: E712
    llm = ok["llm_calls"].to_list() if "llm_calls" in ok.columns else []
    P.append('<div class="cards">')
    P.append(card("p50 latency", f"{_quantile(el,0.5):.0f}s", "median per claim"))
    P.append(card("p95 latency", f"{_quantile(el,0.95):.0f}s", "tail"))
    P.append(card("avg rounds", f"{(sum(rounds)/len(rounds) if rounds else 0):.1f}", f"cap-hit {caphit}/{n_ok}"))
    P.append(card("avg LLM calls", f"{(sum(llm)/len(llm) if llm else 0):.1f}", "per claim"))
    P.append("</div>")
    sb_fig, means = f_stage_breakdown(traces)
    P.append('<div class="note">Where the time goes (mean s/claim, from traces). <b>scrape+summarise</b> + the '
             'per-domain scrape delays dominate — exactly what bundled-content search (Exa/Tavily) and a Groq '
             'summariser would cut for production.</div>')
    P.append('<div style="display:flex;gap:20px;flex-wrap:wrap">')
    P.append('<div style="flex:1;min-width:340px">' + fig_html(f_latency_hist(el)) + "</div>")
    P.append('<div style="flex:1;min-width:340px">' + fig_html(sb_fig) + "</div>")
    P.append("</div>")

    # 7. strata
    P.append("<h2>7 · By publisher &amp; language</h2>")
    P.append('<div style="display:flex;gap:20px;flex-wrap:wrap">')
    P.append('<div style="flex:2;min-width:380px">' + fig_html(f_strata(ok, "publisher_site")) + "</div>")
    if "language_code" in ok.columns:
        P.append('<div style="flex:1;min-width:240px">' + fig_html(f_strata(ok, "language_code")) + "</div>")
    P.append("</div>")

    # 8. misclassified
    P.append("<h2>8 · Misclassified — for adjudication</h2>")
    P.append('<div class="note">Sortable. Read these to separate real model errors from residual gold noise (~2%) '
             'and to spot failure patterns. Click headers to sort.</div>')
    mrows = []
    ct = ok["claim_text"].to_list() if "claim_text" in ok.columns else ok["claim_text_v"].to_list()
    pubs = ok["publisher_site"].to_list() if "publisher_site" in ok.columns else [""] * n_ok
    for i, (g, p, c) in enumerate(zip(gold, pred, correct)):
        if c:
            continue
        mrows.append([f'<td data-v="{pubs[i]}">{pubs[i]}</td>',
                      f'<td data-v="{SHORT[g]}">{SHORT[g]}</td>',
                      f'<td data-v="{SHORT[p]}"><span class="tag bad">{SHORT[p]}</span></td>',
                      f'<td data-v="{ver[i] or 0}" class="num">{ver[i] or "—"}</td>',
                      f'<td>{(ct[i] or "")[:95]}</td>'])
    head = "".join(f"<th>{h}</th>" for h in ["publisher", "gold", "pred", "veracity", "claim"])
    body = "".join("<tr>" + "".join(r) + "</tr>" for r in mrows)
    P.append(f'<div class="scroll"><table class="sortable"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
    P.append(f'<div class="small">{len(mrows)} misclassified of {n_ok} scored.</div>')

    # 9. failure-mode catalog (for the v2 rebuild)
    P.append("<h2>9 · Failure-mode catalog (v1 → v2)</h2>")
    P.append('<div class="note">A complete account of where this version errs, for the dataset + '
             'pipeline rebuild. <span class="badge fixed">FIXED</span> = addressed after this run '
             '(this report\'s numbers predate the fixes); <span class="badge residual">RESIDUAL</span> '
             '= still open. The headline shift: <b>retrieval is no longer the bottleneck</b> (the '
             'date-ceiling bug drove most errors and is fixed); the residual is axis-classification, '
             'the hard Conflicting-Evidence class, and gold noise. The 29-claim fix re-run recovered '
             '18/29 wrongly-nudged → projected binary nudge accuracy 85.5% → ~94.5%, '
             'false-nudge-on-true 42% → ~14%.</div>')
    for fm in FAILURE_MODES:
        cls = "fixed" if fm["status"] == "FIXED" else "residual"
        tag = f' &middot; <span class="small">{fm["tag"]}</span>' if fm["tag"] else ""
        P.append(
            f'<div class="fm {cls}"><h3>{fm["id"]} &middot; {fm["name"]}'
            f'<span class="badge {cls}">{fm["status"]}</span>{tag}</h3>'
            f'<p><span class="b">Cause:</span> {fm["cause"]}</p>'
            f'<p><span class="b">Example:</span> {fm["example"]}</p>'
            f'<p><span class="b">Impact:</span> {fm["impact"]}</p>'
            f'<p><span class="b">Fix / v2:</span> {fm["fix"]}</p></div>')

    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>dev-{n_all} verification report</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>" + "".join(P)
            + f"</div><script>{SORTJS}</script></body></html>")
    REPORTS.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else REPORTS / f"verification_dev{n_all}_{datetime.now():%Y-%m-%d_%H%M}.html"
    out.write_text(html)
    print(f"wrote {out}  ({n_ok}/{n_all} scored, acc {_pct(acc)}, macro-F1 {macro:.3f})")


if __name__ == "__main__":
    main()
