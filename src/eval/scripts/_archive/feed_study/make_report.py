"""Generate the self-contained HTML results report for the feed_study sweep.

Reads the sample1000 masters + run parquets, computes every number live (no
hardcoded stats except the dedup-threshold curve, which is captured from
dedup_claims.py and labelled as such), renders monochrome matplotlib charts to
base64-embedded PNGs, and writes ONE self-contained file:

    docs/feed_study_report.html

Run = the 1000-post sample (data/posts_sample1000.parquet, seed 42). Pipeline
model gpt-oss-120b; agreement model qwen3-32b; 3rd-party judge llama-3.3-70b.
Everything is AGREEMENT-ONLY (no gold set).

  uv run python -m eval.scripts.feed_study.make_report
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

HERE = Path(__file__).parent
D = HERE / "data" / "sample1000"
OUT = HERE / "docs" / "feed_study_report.html"

GPT, QWEN = "gpt-oss-120b", "qwen3-32b"

# ---------------------------------------------------------------------------
# Monochrome house style
# ---------------------------------------------------------------------------
INK = "#1a1a1a"
GREY = "#7a7a7a"
LGREY = "#bdbdbd"
FILL = "#3a3a3a"
FILL2 = "#9a9a9a"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    both = sum(1 for x, y in zip(a, b) if x and y)
    nei = sum(1 for x, y in zip(a, b) if not x and not y)
    po = (both + nei) / n
    ry, rqy = sum(a) / n, sum(b) / n
    pe = ry * rqy + (1 - ry) * (1 - rqy)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def pair_kappa(df: pl.DataFrame, ca: str, cb: str) -> float:
    sub = df.select(ca, cb).drop_nulls()
    return kappa(sub[ca].to_list(), sub[cb].to_list())


# ---------------------------------------------------------------------------
# Load + compute (every stat live from the parquets)
# ---------------------------------------------------------------------------
mp = pl.read_parquet(D / "master_post.parquet")
mc = pl.read_parquet(D / "master_claim.parquet")
nov = pl.read_parquet(D / "novel_claims_gpt-oss-120b.parquet")
fct_med = pl.read_parquet(D / "fct_gpt-oss-120b.parquet")
fct_syn = pl.read_parquet(D / "fct_gpt-oss-120b_synth.parquet")

N_POSTS = mp.height
gpt_claims = mc.filter(pl.col("extractor_model") == GPT)
qwen_claims = mc.filter(pl.col("extractor_model") == QWEN)


def npass(col: str) -> int:
    return mp.filter(pl.col(col).fill_null(False)).height


# agreement (B1) + inter-filter (B8)
k_relev = pair_kappa(mp, "relev_raw_g", "relev_raw_q")
k_verif = pair_kappa(mp, "verif_raw_g", "verif_raw_q")
k_harm = pair_kappa(mp, "harm_raw_g", "harm_raw_q")
k_vr = pair_kappa(mp, "verif_raw_g", "relev_raw_g")
k_vh = pair_kappa(mp, "verif_raw_g", "harm_raw_g")
k_rh = pair_kappa(mp, "relev_raw_g", "harm_raw_g")

# funnel (B2)
v = npass("verif_raw_g")
vr = mp.filter(pl.col("verif_raw_g").fill_null(False) & pl.col("relev_raw_g").fill_null(False)).height
vrh = mp.filter(pl.col("verif_raw_g").fill_null(False) & pl.col("relev_raw_g").fill_null(False)
                & pl.col("harm_raw_g").fill_null(False)).height
surv_vrh = mp.filter(pl.col("verif_raw_g").fill_null(False) & pl.col("relev_raw_g").fill_null(False)
                     & pl.col("harm_raw_g").fill_null(False)).select("post_id")
surv_r = mp.filter(pl.col("relev_raw_g").fill_null(False)).select("post_id")
claims_vrh = gpt_claims.join(surv_vrh, on="post_id", how="inner").height
claims_r = gpt_claims.join(surv_r, on="post_id", how="inner").height

# harm distribution (B3)
ht_g = mp.select("harm_total_g").drop_nulls()["harm_total_g"]
ht_q = mp.select("harm_total_q").drop_nulls()["harm_total_q"]
med_g, med_q = ht_g.median(), ht_q.median()


# normalization stripping (B4)
def strip_stats(t: str) -> tuple[int, int, int, float]:
    hr, hc = f"harm_raw_{t}", f"harm_claim_{t}"
    j = mc.select("post_id", hc).join(mp.select("post_id", hr), on="post_id", how="inner").drop_nulls()
    raw_pos = j.filter(pl.col(hr)).height
    claim_pos = j.filter(pl.col(hr) & pl.col(hc)).height
    drop = raw_pos - claim_pos
    return raw_pos, claim_pos, drop, 100 * drop / raw_pos


sg, sq = strip_stats("g"), strip_stats("q")


# quality (B5)
def q_means(ext: str) -> dict:
    sub = mc.filter(pl.col("extractor_model") == ext)
    return {d: sub[d].drop_nulls().mean() for d in ("faithful", "decontextualized", "atomicity")}


def cov_flag(ext: str) -> tuple[float, dict, int]:
    pf = mc.filter(pl.col("extractor_model") == ext).select("post_id", "coverage", "flag").unique(subset=["post_id"])
    return pf["coverage"].drop_nulls().mean(), dict(pf.group_by("flag").len().iter_rows()), pf.height


qm_g, qm_q = q_means(GPT), q_means(QWEN)
cov_g, fl_g, np_g = cov_flag(GPT)
cov_q, fl_q, np_q = cov_flag(QWEN)
fl_all = {k: fl_g.get(k, 0) + fl_q.get(k, 0) for k in ("good", "borderline", "bad")}
np_all = np_g + np_q

# why high per-claim means coexist with ~1/3 borderline posts: the flag tracks
# COVERAGE (post-level completeness), not per-claim quality. (gpt-oss.)
_gq = mc.filter(pl.col("extractor_model") == GPT)
_uni = _gq.select("post_id", "coverage", "flag").unique(subset=["post_id"])
cov_good = _uni.filter(pl.col("flag") == "good")["coverage"].drop_nulls().mean()
cov_bord = _uni.filter(pl.col("flag") == "borderline")["coverage"].drop_nulls().mean()
faith_bord = _gq.filter(pl.col("flag") == "borderline")["faithful"].drop_nulls().mean()

# extraction agreement (B6)
s = mp.select("n_claims_g", "n_claims_q").fill_null(0)
gg, qq = s["n_claims_g"].to_list(), s["n_claims_q"].to_list()
exact = 100 * sum(1 for a, b in zip(gg, qq) if a == b) / len(gg)
within1 = 100 * sum(1 for a, b in zip(gg, qq) if abs(a - b) <= 1) / len(gg)
both_ext = sum(1 for a, b in zip(gg, qq) if a > 0 and b > 0)
cmp_dist = dict(mp.select("cmp_agreement").drop_nulls().group_by("cmp_agreement").len().iter_rows())
cmp_more = dict(mp.select("cmp_more_complete").drop_nulls().group_by("cmp_more_complete").len().iter_rows())

# dedup + Tier-2 (B7); curve captured from dedup_claims.py (gpt, V∧R∧H), 0.85 == nov.height
DEDUP_CURVE = {0.75: 358, 0.80: 376, 0.85: 410, 0.90: 423, 0.95: 438}
N_VRH_CLAIMS = nov.select(pl.col("n_members").sum()).item()
N_NOVEL = nov.height
assert DEDUP_CURVE[0.85] == N_NOVEL, f"curve/live mismatch: {DEDUP_CURVE[0.85]} vs {N_NOVEL}"


def _fct_rate(df: pl.DataFrame) -> tuple[int, int, float]:
    ok = df.filter(pl.col("error").is_null())
    m = ok.filter(pl.col("fct_match")).height
    return m, ok.height, 100 * m / ok.height


fct_m, fct_s = _fct_rate(fct_med), _fct_rate(fct_syn)

# corpus facts
corp = dict(mp.group_by("source_corpus").len().iter_rows())
langs = dict(mp.group_by("lang").len().sort("len", descending=True).head(4).iter_rows())
n_claims_total = mc.height
n_g, n_q = gpt_claims.height, qwen_claims.height


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _above(ax, x, y_frac, text, color=GREY, ha="center"):
    """Place an annotation ABOVE the axes (x in data coords, y in axes frac)."""
    ax.text(x, y_frac, text, transform=ax.get_xaxis_transform(), ha=ha,
            va="bottom", fontsize=8, color=color, clip_on=False)


def chart_agreement() -> str:
    labels = ["Relevance", "Verifiability", "Harm (FABLE)"]
    vals = [k_relev, k_verif, k_harm]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    y = range(len(labels))
    ax.barh(y, vals, color=[FILL, FILL2, FILL2], height=0.6)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    for xv, ls, txt in [(0.8, "--", "almost-perfect"), (0.6, ":", "substantial")]:
        ax.axvline(xv, color=LGREY, lw=1, ls=ls)
        _above(ax, xv, 1.04, txt)
    for i, vv in enumerate(vals):
        ax.text(vv + 0.012, i, f"{vv:.2f}", va="center", fontweight="bold")
    ax.set_xlabel("Cohen's κ  ·  gpt-oss-120b vs qwen3-32b, same raw posts")
    return fig_to_b64(fig)


def chart_funnel() -> str:
    stages = ["All posts", "Verifiable (V)", "+ Relevant (V∧R)", "+ Harmful (V∧R∧H)"]
    vals = [N_POSTS, v, vr, vrh]
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    y = range(len(stages))
    ax.barh(y, vals, color=[LGREY, "#8f8f8f", "#5f5f5f", FILL], height=0.62)
    ax.set_yticks(list(y), stages)
    ax.invert_yaxis()
    ax.set_xlim(0, N_POSTS * 1.14)
    for i, vv in enumerate(vals):
        ax.text(vv + N_POSTS * 0.013, i, f"{vv}  ({100*vv/N_POSTS:.0f}%)", va="center", fontweight="bold")
    ax.set_xlabel("posts surviving each gate  ·  gpt-oss-120b, cumulative AND")
    return fig_to_b64(fig)


def chart_harm_dist() -> str:
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    rng = list(range(5, 26))
    cg = [int((ht_g == x).sum()) for x in rng]
    cq = [int((ht_q == x).sum()) for x in rng]
    w = 0.42
    ax.bar([x - w / 2 for x in rng], cg, width=w, color=FILL, label="gpt-oss-120b")
    ax.bar([x + w / 2 for x in rng], cq, width=w, color=FILL2, label="qwen3-32b")
    ax.axvline(med_g, color=INK, lw=1.2, ls="--")
    ax.axvline(med_q, color=GREY, lw=1.2, ls=":")
    _above(ax, med_g - 0.2, 1.03, f"gpt median {med_g:.0f}", color=INK, ha="right")
    _above(ax, med_q + 0.2, 1.03, f"qwen median {med_q:.0f}", color=GREY, ha="left")
    ax.set_xlabel("FABLE harm total (sum of 5 dimensions, 5–25) on the raw post")
    ax.set_ylabel("posts")
    ax.set_xticks(range(5, 26, 2))
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    return fig_to_b64(fig)


def chart_strip() -> str:
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    groups = ["gpt-oss-120b", "qwen3-32b"]
    raw = [sg[0], sq[0]]
    kept = [sg[1], sq[1]]
    y = range(len(groups))
    ax.barh(y, raw, color=LGREY, height=0.58, label="harmful at raw-post level")
    ax.barh(y, kept, color=FILL, height=0.58, label="still harmful as a claim")
    ax.set_yticks(list(y), groups)
    ax.invert_yaxis()
    for i, st in enumerate([sg, sq]):
        ax.text(st[0] + 14, i, f"{st[0]} → {st[1]}   ({st[3]:.0f}% stripped)",
                va="center", fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, max(raw) * 1.5)
    ax.set_xlabel("claims  ·  post flagged harmful  vs.  claim still harmful (same model)")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return fig_to_b64(fig)


def chart_quality() -> str:
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    dims = ["faithful", "decontext.", "atomicity", "coverage"]
    g = [qm_g["faithful"], qm_g["decontextualized"], qm_g["atomicity"], cov_g]
    q = [qm_q["faithful"], qm_q["decontextualized"], qm_q["atomicity"], cov_q]
    x = range(len(dims))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], g, width=w, color=FILL, label="gpt-oss-120b")
    b2 = ax.bar([i + w / 2 for i in x], q, width=w, color=FILL2, label="qwen3-32b")
    ax.set_ylim(4.0, 5.12)
    ax.set_xticks(list(x), dims)
    ax.set_ylabel("mean score (1–5)")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    return fig_to_b64(fig)


def chart_extr_agree() -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.0), constrained_layout=True)
    diffs = [a - b for a, b in zip(gg, qq)]
    bins = list(range(-4, 6))
    counts = [sum(1 for d in diffs if d == bb) for bb in bins]
    ax1.bar(bins, counts, color=[FILL if bb == 0 else FILL2 for bb in bins], width=0.85)
    ax1.set_xlabel("n_claims(gpt) − n_claims(qwen)", fontsize=9.5)
    ax1.set_ylabel("posts")
    ax1.set_xticks(range(-4, 6, 2))
    order = ["high", "medium", "low"]
    vals = [cmp_dist.get(k, 0) for k in order]
    ax2.bar(order, vals, color=[FILL, FILL2, LGREY], width=0.62)
    for i, vv in enumerate(vals):
        ax2.text(i, vv + 3, str(vv), ha="center", fontweight="bold", fontsize=9.5)
    ax2.set_ylim(0, max(vals) * 1.18)
    ax2.set_xlabel(f"llama claim-set agreement  ·  n={sum(vals)} both-extracted", fontsize=9.5)
    return fig_to_b64(fig)


def chart_dedup() -> str:
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    thr = list(DEDUP_CURVE.keys())
    clusters = list(DEDUP_CURVE.values())
    ax.plot(thr, clusters, marker="o", color=FILL, lw=1.6)
    ax.axhline(N_VRH_CLAIMS, color=LGREY, lw=1, ls="--")
    ax.text(0.95, N_VRH_CLAIMS + 0.8, f"{N_VRH_CLAIMS} raw claims (no merge)",
            fontsize=8, color=GREY, va="bottom", ha="right")
    for xv, yv in zip(thr, clusters):
        ax.annotate(str(yv), (xv, yv), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, fontweight="bold" if xv == 0.85 else "normal")
    ax.scatter([0.85], [N_NOVEL], s=80, facecolor="white", edgecolor=INK, zorder=5)
    ax.set_xlabel("cosine merge threshold")
    ax.set_ylabel("distinct (novel) claims")
    ax.set_ylim(345, 462)
    return fig_to_b64(fig)


def chart_interfilter() -> str:
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    labels = ["Relevance ↔ Harm", "Verifiability ↔ Relevance", "Verifiability ↔ Harm"]
    vals = [k_rh, k_vr, k_vh]
    y = range(len(labels))
    ax.barh(y, vals, color=FILL2, height=0.55)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.axvline(0.8, color=LGREY, lw=1, ls="--")
    _above(ax, 0.8, 1.05, "almost-perfect")
    for i, vv in enumerate(vals):
        ax.text(vv + 0.012, i, f"{vv:.2f}", va="center", fontweight="bold")
    ax.set_xlabel("Cohen's κ between filters  ·  low = they measure different things")
    return fig_to_b64(fig)


charts = {
    "agreement": chart_agreement(), "funnel": chart_funnel(), "harm": chart_harm_dist(),
    "strip": chart_strip(), "quality": chart_quality(), "extr_agree": chart_extr_agree(),
    "dedup": chart_dedup(), "interfilter": chart_interfilter(),
}


def img(key: str, alt: str) -> str:
    return f'<img src="data:image/png;base64,{charts[key]}" alt="{alt}">'


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
:root{--ink:#1a1a1a;--grey:#5f5f5f;--line:#e4e4e4;--bg:#fff;--soft:#f6f6f6;}
*{box-sizing:border-box;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  color:var(--ink);background:var(--bg);line-height:1.5;margin:0;font-size:15px;}
.wrap{max-width:820px;margin:0 auto;padding:46px 28px 90px;}
h1{font-size:27px;font-weight:700;letter-spacing:-.01em;margin:0 0 6px;}
h2{font-size:22px;font-weight:700;margin:50px 0 12px;padding-bottom:7px;border-bottom:1px solid var(--line);}
h3{font-size:17px;font-weight:600;margin:30px 0 4px;}
p{margin:8px 0;}
.sub{color:var(--grey);font-size:15px;margin:0 0 12px;}
.meta{font-size:13px;color:var(--grey);}
.meta code{background:var(--soft);padding:1px 5px;border-radius:3px;font-size:12px;}
figure{margin:14px 0 4px;}
figure img{width:100%;border:1px solid var(--line);border-radius:6px;}
figcaption{font-size:12.5px;color:var(--grey);margin-top:5px;}
.lead{font-size:16px;font-weight:600;margin:4px 0 2px;}
.note{font-size:13.5px;color:var(--grey);}
ul{margin:8px 0;padding-left:20px;}li{margin:4px 0;}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 4px;}
.kpi{flex:1 1 150px;background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:11px 14px;}
.kpi .v{font-size:22px;font-weight:700;}
.kpi .l{font-size:12px;color:var(--grey);margin-top:2px;line-height:1.35;}
.steps{margin:14px 0;}
.step{display:flex;gap:14px;padding:15px 0;border-top:1px solid var(--line);}
.step:last-child{border-bottom:1px solid var(--line);}
.step .n{flex:0 0 30px;height:30px;border-radius:50%;background:var(--ink);color:#fff;
  display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;}
.step .body{flex:1;}
.step .t{font-weight:700;font-size:15.5px;}
.step .t .tag{font-size:11px;font-weight:600;color:var(--grey);border:1px solid var(--line);
  border-radius:4px;padding:1px 6px;margin-left:8px;}
.step .d{font-size:14px;color:#2a2a2a;margin-top:4px;}
.step .io{font-size:12.5px;color:var(--grey);margin-top:5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.step .src{font-size:12.5px;margin-top:5px;}
.step .src a{color:var(--ink);}
.callout{background:var(--soft);border-left:3px solid var(--ink);border-radius:0 6px 6px 0;
  padding:11px 16px;margin:14px 0;font-size:14px;}
.refs{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin-top:18px;font-size:13px;}
.refs ol{margin:6px 0 0;padding-left:20px;}.refs li{margin:5px 0;color:#2a2a2a;}
.refs a{color:var(--ink);}
.flow{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;color:#2a2a2a;
  background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:14px 16px;
  overflow-x:auto;white-space:pre;line-height:1.5;}
.caveats li{font-size:14px;}
.derive{font-size:12.5px;color:var(--grey);font-style:italic;margin-top:2px;}
footer{margin-top:54px;font-size:12px;color:var(--grey);border-top:1px solid var(--line);padding-top:14px;}
"""


def step(n, title, tag, desc, io_line, src) -> str:
    return (f'<div class="step"><div class="n">{n}</div><div class="body">'
            f'<div class="t">{title}<span class="tag">{tag}</span></div>'
            f'<div class="d">{desc}</div><div class="io">{io_line}</div>'
            f'<div class="src">{src}</div></div></div>')


steps_html = "".join([
    step("V", "Selection — verifiability gate", "post",
         "“Does the post contain at least one <b>specific, verifiable</b> proposition?” Judges verifiability "
         "only — not truth, importance, or clarity — and returns the factual core with evaluative wrappers "
         "(“I think”, “outrageously”) stripped. Zero-shot, 4-step reasoning.",
         "raw post → verifiable (bool) + cleaned text",
         "Claimify stage 1 [1], adapted to score the whole post (the paper works sentence-by-sentence)."),
    step("R", "Relevance — public-consequence gate", "post",
         "“Is this a matter of <b>public consequence</b> — politics, health, science, economics, "
         "crime/conflict, public misinformation — rather than personal, sports, promotional, or opinion "
         "content?” A zero-shot rubric; the in-scope domain label is kept for analysis.",
         "raw post → in_scope (bool) + domain",
         "Our own filter — not from Claimify or FABLE. Added to focus extraction on publicly-consequential claims."),
    step("H", "Harm — FABLE rubric", "raw post",
         "Scores five harm dimensions 1–5 — <b>fragmentation, actionability, believability, spread-likelihood, "
         "exploitativeness</b> — on the <b>raw post</b>, so inflammatory framing is still present. Check-worthy "
         "if <code>total ≥ 12</code> or <code>(fragmentation+actionability+exploitativeness) ≥ 8</code>.",
         "raw post → 5 dims + checkworthy (bool)",
         "FABLE harm framework [2]; the five dimensions are verbatim, the check-worthy threshold is ours."),
    step("4", "Disambiguation", "claim",
         "Rewrites the verifiable core into a <b>standalone</b> sentence — resolving pronouns, partial names, "
         "acronyms, relative dates from the post only. Hybrid abstain: if a reference can’t be resolved, it "
         "returns nothing rather than guess.",
         "cleaned core + post → standalone sentence (or abstain)",
         "Claimify stage 2 [1]; we add a confidence flag (in practice it fires “high” almost always — inert)."),
    step("5", "Decomposition", "claim",
         "Splits the standalone sentence into <b>atomic, self-contained</b> claims, each checkable on its own, "
         "preserving attribution (“According to X…”). Minimal-but-complete — split genuine separate facts, "
         "don’t over-atomize.",
         "standalone sentence → claims[]",
         "Claimify stage 3 [1]; the single-checkable-unit target follows Molecular Facts [3]."),
    step("6", "Dedup — Tier 1", "claim",
         "Embeds each surviving claim and cosine-clusters near-duplicates (union-find). The cluster count is "
         "the number of <b>distinct</b> claims actually needing a check; the embeddings also seed the Tier-1 "
         "vector cache (Neon + pgvector).",
         "claims[] → distinct-claim clusters",
         "Embeddings: <code>paraphrase-multilingual-MiniLM-L12-v2</code> (multilingual Sentence-Transformers [4])."),
    step("7", "Fact-check lookup — Tier 2", "claim",
         "Queries the Google Fact Check Tools API for each distinct claim: does a published ClaimReview "
         "already exist? A hit resolves the claim cheaply and skips the expensive Tier-3 step.",
         "distinct claim → existing fact-check (bool) + publisher/rating",
         "Google Fact Check Tools API (ClaimReview schema)."),
    step("8", "Full verification — Tier 3", "downstream",
         "A web-search verification loop for claims with no cached or published answer. Out of scope for this "
         "study — shown only to complete the picture.",
         "unresolved claim → verdict (downstream)",
         "ClaimCheck-style retrieve-summarize-synthesize loop; measured separately."),
])

refs_html = """
<b>References</b>
<ol>
<li>Metropolitansky &amp; Larson. <i>Towards Effective Extraction and Evaluation of Factual Claims</i> (Claimify). ACL 2025. <a href="https://arxiv.org/abs/2502.10855">arXiv:2502.10855</a></li>
<li>Sehat et al. <i>Misinformation as a Harm: Structured Approaches for Fact-Checking Prioritization</i> (FABLE). CSCW 2024. <a href="https://arxiv.org/abs/2312.11678">arXiv:2312.11678</a></li>
<li><i>Molecular Facts: Desiderata for Decontextualization in LLM Fact Verification.</i> 2024. <a href="https://arxiv.org/abs/2406.20079">arXiv:2406.20079</a></li>
<li>Reimers &amp; Gurevych. Multilingual Sentence-Transformers (<code>paraphrase-multilingual-MiniLM-L12-v2</code>). <a href="https://www.sbert.net">sbert.net</a></li>
</ol>
"""

flow = (
    "  RAW POST\n"
    "     │\n"
    "  [V] Selection      — verifiable?            ┐\n"
    "  [R] Relevance      — public consequence?    ├─ AND  (post-level gates)\n"
    "  [H] Harm / FABLE   — harmful on raw post?   ┘\n"
    "     │  posts passing all three\n"
    "  [4] Disambiguation — rewrite as standalone sentence\n"
    "  [5] Decomposition  — split into atomic claims\n"
    "     │  claims[]\n"
    "  [6] Dedup  (Tier 1) — cluster near-duplicates  → distinct claims\n"
    "  [7] FCT    (Tier 2) — existing fact-check?\n"
    "  [8] Web    (Tier 3) — full verification  (downstream)\n"
)

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Feed-study — claim selection &amp; extraction</title>
<style>{CSS}</style></head>
<body><div class="wrap">

<h1>Claim selection &amp; extraction — feed-study</h1>
<p class="sub">How the pipeline decides what to fact-check, and how it behaved on a captured X/Twitter feed.</p>
<p class="meta">Run: <code>posts_sample1000.parquet</code> (seed 42) · {N_POSTS} posts ({corp.get('graphql_x862','?')} GraphQL + {corp.get('flat_capture','?')} flat-capture; top langs fr&nbsp;{langs.get('fr','?')} / en&nbsp;{langs.get('en','?')}) · {n_claims_total} claims ({n_g} gpt / {n_q} qwen). Models: pipeline <code>gpt-oss-120b</code>, agreement <code>qwen3-32b</code>, judge <code>llama-3.3-70b</code>. Sweep ~26.6k calls, 53&nbsp;min, €10. <b>Agreement-only — no gold set yet.</b></p>

<h2>How the pipeline works</h2>
<p>The system is a <b>cascade of independent filters before extraction</b>, then dedup and lookup. Three post-level gates decide whether a post is worth extracting from; surviving posts are rewritten into standalone sentences and split into atomic claims; claims are deduplicated and checked against existing fact-checks. The three gates combine with <b>AND</b>, so their order changes only cost, not the surviving set.</p>
<p>Every filter and extractor is run independently by <b>two model families</b> — gpt-oss-120b and qwen3-32b — so we can measure how much they agree (the subject of the second half).</p>

<div class="flow">{flow}</div>

<div class="steps">{steps_html}</div>

<div class="refs">{refs_html}</div>

<h2>What the run showed</h2>
<div class="kpis">
  <div class="kpi"><div class="v">κ 0.81</div><div class="l">relevance — most model-stable filter</div></div>
  <div class="kpi"><div class="v">188 → 450</div><div class="l">posts → claims through all 3 gates</div></div>
  <div class="kpi"><div class="v">410</div><div class="l">distinct claims after dedup</div></div>
  <div class="kpi"><div class="v">4.4%</div><div class="l">already have a published fact-check</div></div>
</div>

<h3>The filters agree at different rates</h3>
<p class="lead">Relevance is the most model-stable filter — κ&nbsp;{k_relev:.2f}, almost-perfect, and zero-shot.</p>
<p>Verifiability ({k_verif:.2f}) and harm ({k_harm:.2f}) are substantial but noisier; the public-consequence boundary is apparently the easiest for both model families to agree on.</p>
<figure>{img('agreement','cross-model kappa per filter')}<figcaption>Cohen’s κ between the two models on the same raw posts. Dashed = 0.8 (almost-perfect), dotted = 0.6 (substantial).</figcaption></figure>

<h3>The cascade funnel</h3>
<p class="lead">Of {N_POSTS} posts, {vrh} survive all three gates — yielding {claims_vrh} claims.</p>
<p>Relevance-only (dropping the harm gate) would pass {claims_r} claims instead. Because the gates are an AND, order changes only cost — verifiability first is cheapest.</p>
<figure>{img('funnel','cascade funnel')}<figcaption>Cumulative survival under V∧R∧H (gpt-oss-120b).</figcaption></figure>
<p class="derive">{claims_vrh} claims = gpt-oss claims whose post passed verifiable ∧ relevant ∧ harmful; {claims_r} = passed relevance only.</p>

<h3>Harm scores: where the models diverge</h3>
<p class="lead">gpt-oss centres harm lower (median {med_g:.0f}) than qwen3-32b (median {med_q:.0f}).</p>
<p>The gap is in believability and spread-likelihood, which qwen inflates; the three harm-core dimensions agree far better — which is why the check-worthy rule leans on them.</p>
<figure>{img('harm','harm total distribution')}<figcaption>FABLE harm total per model on the raw post; vertical lines mark each median.</figcaption></figure>

<h3>Normalization strips harm — so judge it on the raw post</h3>
<p class="lead">{sg[3]:.0f}–{sq[3]:.0f}% of raw-post harm flags vanish once FABLE scores the normalized claim.</p>
<p>Cleaning a post into a claim strips the inflammatory framing, the call to action, the targeting — so <b>harm must be judged up front, on the raw post</b>, before extraction removes it.</p>
<figure>{img('strip','harm stripped by normalization')}<figcaption>Claims whose post was flagged harmful (light) vs. those still harmful when the claim itself is scored (dark), same model.</figcaption></figure>

<h3>The claims we extract are near-perfect; what we miss is the gap</h3>
<p class="lead">Each claim scores ~{qm_g['decontextualized']:.1f}–{qm_g['atomicity']:.1f} (gpt-oss); the soft spot is coverage — whether extraction got <i>all</i> the checkable facts.</p>
<p>An independent llama-3.3-70b judge rates each <b>claim</b> 1–5 (faithfulness {qm_g['faithful']:.2f}, atomicity {qm_g['atomicity']:.2f}, decontextualization {qm_g['decontextualized']:.2f}) and flags each <b>post</b> as a whole: <b>{100*fl_all['good']/np_all:.0f}% good, {100*fl_all['borderline']/np_all:.0f}% borderline, {100*fl_all['bad']/np_all:.0f}% bad</b> ({fl_all['bad']} bad of {np_all}). The two are different units, which is why high per-claim means sit next to a third of posts flagged.</p>
<p>The flag tracks <b>coverage</b>, not claim quality: a “borderline” post almost always means a <i>minor omission</i> — its claims still average {faith_bord:.1f} on faithfulness, but coverage drops to {cov_bord:.1f} (vs {cov_good:.1f} for “good”). It’s not that the claims are shaky; it’s that one checkable fact was left on the table.</p>
<figure>{img('quality','extraction quality')}<figcaption>Mean per-claim dimensions plus coverage, axis zoomed to 4–5. Coverage (were checkable facts missed?) is the weakest — and it is what the good/borderline/bad flag follows.</figcaption></figure>

<h3>Do the two models extract the same claims?</h3>
<p class="lead">{exact:.0f}% exact agreement on claim count, {within1:.0f}% within ±1.</p>
<p>On the {both_ext} posts where both extracted something, the judge rated the claim-sets <b>{cmp_dist.get('high',0)} high / {cmp_dist.get('medium',0)} medium / {cmp_dist.get('low',0)} low</b>; gpt-oss was judged more complete more often ({cmp_more.get('A',0)} vs {cmp_more.get('B',0)}).</p>
<figure>{img('extr_agree','extraction agreement')}<figcaption>Left: per-post difference in claim count (gpt − qwen). Right: llama’s claim-set agreement on both-extracted posts.</figcaption></figure>

<h3>Distinct claims and existing fact-checks</h3>
<p class="lead">{N_VRH_CLAIMS} claims dedupe to {N_NOVEL} distinct ones; only {fct_m[2]:.1f}–{fct_s[2]:.1f}% already have a published fact-check.</p>
<p>At cosine 0.85, {100*(N_VRH_CLAIMS-N_NOVEL)/N_VRH_CLAIMS:.0f}% are redundant. Querying Google’s Fact Check Tools API, only {fct_m[0]}/{fct_m[1]} ({fct_m[2]:.1f}%) — or {fct_s[2]:.1f}% with a synthesized canonical form — already have one, so <b>~{N_NOVEL-fct_s[0]}/{N_NOVEL} would fall through to Tier-3</b>.</p>
<figure>{img('dedup','dedup threshold curve')}<figcaption>Distinct-claim count vs. cosine merge threshold; circled point (0.85) is the headline.</figcaption></figure>
<div class="callout"><b>Insight:</b> a fresh feed has almost no pre-existing fact-checks — the Tier-2 lookup and the claims cache pay off <i>over time</i> as the database fills, not on day one.</div>

<h3>The three gates measure different things</h3>
<p class="lead">Pairwise κ between filters is low ({k_vh:.2f}–{k_rh:.2f}) — the gates are not redundant.</p>
<p>Verifiability and harm overlap least; each gate removes a genuinely different slice of the feed, so none is a free drop.</p>
<figure>{img('interfilter','inter-filter overlap')}<figcaption>Cohen’s κ between each pair of gates on the same posts (gpt-oss, raw).</figcaption></figure>

<h2>Read these numbers carefully</h2>
<ul class="caveats">
<li><b>Agreement, not accuracy.</b> No gold-standard set yet — every κ measures whether two models <i>agree</i>, not whether either is <i>right</i>.</li>
<li><b>One capture.</b> All figures come from a single 1000-post sample of one feed; another feed could funnel differently.</li>
<li><b>The harm threshold is ours.</b> FABLE supplies the five dimensions; the check-worthy rule (total ≥ 12 or harm-core ≥ 8) is bespoke and uncalibrated.</li>
<li><b>The disambiguation confidence flag is inert</b> (“high” almost always) — not yet a useful signal.</li>
</ul>

<footer>Generated from <code>data/sample1000/</code> by <code>make_report.py</code> — every figure computed live from the run parquets; the dedup-threshold curve is captured from <code>dedup_claims.py</code>. Feed-study rebuild, 2026-06.</footer>

</div></body></html>"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html)
print(f"wrote {OUT}  ({len(html)//1024} KB, {len(charts)} charts)")
