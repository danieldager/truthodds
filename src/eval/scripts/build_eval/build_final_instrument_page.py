"""Regenerate the "Final Instrument for Survey Experiment" collaborator page.

Live page: https://claude.ai/code/artifact/54798fdf-92ba-4f68-9392-0e9796efa987

Every number on the page is enumerated in NUMBERS below. Each is either
DERIVED from a committed metric file (source != None -> the script recomputes
and --verify asserts it equals the page's displayed value) or carried verbatim
from the live page and tagged NO-COMMITTED-SOURCE (source is None -> the number
can only be reproduced from an uncommitted/untracked run output or a paid run;
--verify lists it but cannot assert it).

The page prose and structure are held verbatim in the HTML/MD templates; only
the ${id} tokens are substituted, so nothing in the prose is hand-edited.

Committed metric files this reads (all git-tracked):
  eval/data/urn_runs/e1_ctx/headline_metrics.json          frozen read-v5 7-flag fit
  eval/data/urn_runs/e1_ctx/weights_v5_ceiling_bal3000.json  frozen weights (sibling)
  eval/data/urn_runs/e1_ctx/blind120_summary.json          120-doc blind reader audit
  eval/data/urn_runs/e1_prodregime/prodregime_metrics.json Exa/Serper (rep1500, 6-flag)
Reused (committed computed numbers / helpers, not duplicated here):
  stronger_reader_scores.BASE_REF  frozen-instrument baseline rows (both regimes)

  cd src && uv run python -m eval.scripts.build_eval.build_final_instrument_page --verify
  cd src && uv run python -m eval.scripts.build_eval.build_final_instrument_page [--out DIR] [--force]

$0 -- reads committed JSON only, no API calls, no fits over the untracked reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from string import Template

from eval.scripts.build_eval.stronger_reader_scores import BASE_REF  # committed baseline rows

SRC = Path(__file__).resolve().parents[3]
E1 = SRC / "eval/data/urn_runs/e1_ctx"
PROD = SRC / "eval/data/urn_runs/e1_prodregime"
OUTDIR = SRC / "eval/data/urn_runs/final_instrument"
MINUS = "−"  # the page renders negatives with U+2212, not hyphen-minus


def sgn(x: float, d: int) -> str:
    """Signed number with the page's minus glyph, no leading + on positives."""
    s = f"{abs(x):.{d}f}"
    return (MINUS + s) if x < 0 else s


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


# ---------------------------------------------------------------- committed derivations
def derive() -> dict[str, str]:
    H = json.loads((E1 / "headline_metrics.json").read_text())
    O = H["overall"]
    S = H["silence_rule"]
    B = json.loads((E1 / "blind120_summary.json").read_text())["cells"]

    def err(cell: str, kind: str) -> str:
        k, n = B[cell][kind]
        return f"{k} of {n}"

    lo, hi = O["auc_ci95"]
    bt, bp = BASE_REF["training"], BASE_REF["production"]
    return {
        # frozen 7-flag training fit (headline_metrics.json)
        "train_auc": f'{O["auc_oof"]:.3f}',                 # 0.857
        "train_ci": f"{hi - lo:.3f}",                       # 0.024
        "tt_recall": pct(O["recall_at_2pct_fpr"]),          # 41.7%
        "boundary": sgn(O["threshold"], 1),                 # -4.0
        "wI_train": sgn(O["weights"]["I"], 2),              # -0.19
        "n3000": f'{O["n"]:,}',                             # 3,000
        "sil_train_true": str(S["n_silent_true"]),          # 333
        "sil_train_false": str(S["n_silent_false"]),        # 431
        # 120-doc blind reader audit (blind120_summary.json). judge one = earlier, blind = blind.
        "rf_j1": err("R-F", "earlier_reader_error"), "rf_bl": err("R-F", "blind_reader_error"),
        "st_j1": err("S-T", "earlier_reader_error"), "st_bl": err("S-T", "blind_reader_error"),
        "rt_j1": err("R-T", "earlier_reader_error"), "rt_bl": err("R-T", "blind_reader_error"),
        "sf_j1": err("S-F", "earlier_reader_error"), "sf_bl": err("S-F", "blind_reader_error"),
        "n_judged": str(json.loads((E1 / "blind120_summary.json").read_text())["n_judged"]),  # 120
        "per_cell": str(B["R-F"]["n"]),                     # 30
        # frozen-instrument baseline rows, both regimes (stronger_reader_scores.BASE_REF)
        "fz_tr_auc": f'{bt["auc"]:.3f}', "fz_tr_rec": pct(bt["recall"]),
        "fz_tr_false": str(bt["false_flagged"]), "fz_tr_true": str(bt["true_flagged"]),
        "fz_pr_auc": f'{bp["auc"]:.3f}', "fz_pr_rec": pct(bp["recall"]),
        "fz_pr_false": str(bp["false_flagged"]), "fz_pr_true": str(bp["true_flagged"]),
    }


# ---------------------------------------------------------------- number manifest
# Each row: id, page (the live page's displayed literal), source.
#   source is a committed file/derivation -> the id must be produced by derive();
#   source is None  -> NO COMMITTED SOURCE; page literal is carried verbatim.
COMMITTED = "headline_metrics.json"
BLIND = "blind120_summary.json"
BASEREF = "stronger_reader_scores.BASE_REF"
PRODREG = "prodregime_metrics.json (supports ordering only)"

NUMBERS: list[tuple[str, str, str | None]] = [
    # ---- Two ways to fit the weights (table)
    ("train_auc", "0.857", COMMITTED),
    ("train_ci", "0.024", COMMITTED),
    ("train_averitec", "0.869", None),
    ("train_averitec_ci", "0.074", None),
    ("prod_auc", "0.924", None),
    ("prod_ci", "0.020", None),
    ("prod_averitec", "0.858", None),
    ("prod_averitec_ci", "0.074", None),
    # ---- What we found in production mode
    ("hand60a", "60", None),
    ("sil_train_true", "333", COMMITTED),
    ("sil_train_false", "431", COMMITTED),
    ("wI_prod", MINUS + "0.59", None),
    ("wI_train", MINUS + "0.19", COMMITTED),
    ("sil_prod_true", "46", None),
    ("sil_prod_false", "305", None),
    ("score_ten_irr", MINUS + "5.9", None),
    ("bound_prod_ex", MINUS + "6.7", None),
    ("score_one_contra", MINUS + "7.6", None),
    # ---- Each weight set on the other evidence (cross table)
    ("tt_recall", "41.7%", COMMITTED),
    ("pt_auc", "0.840", None), ("pt_ci", "0.026", None), ("pt_recall", "40.5%", None),
    ("tp_auc", "0.927", None), ("tp_ci", "0.020", None), ("tp_recall", "37.2%", None),
    ("pp_auc", "0.924", None), ("pp_ci", "0.020", None), ("pp_recall", "38.9%", None),
    ("lose_016", "0.016", None),
    ("above_003", "0.003", None),
    # ---- The evidence production will actually see
    ("median_days", "29", None),
    ("pct_older", "82%", None),
    ("n_survey", "1,660", None),
    ("mix_fit_auc", "0.894", None), ("mix_fit_ci", "0.023", None),
    ("mix_fit_rec", "35.3%", None), ("mix_fit_bound", MINUS + "4.61", None),
    ("mix_tr_auc", "0.897", None), ("mix_tr_ci", "0.022", None),
    ("mix_tr_rec", "32.5%", None), ("mix_tr_bound", MINUS + "3.99", None),
    ("mix_pr_auc", "0.893", None), ("mix_pr_ci", "0.022", None),
    ("mix_pr_rec", "34.9%", None), ("mix_pr_bound", MINUS + "6.23", None),
    ("ahead_004", "0.004", None),
    ("margin_train", "2.1", None), ("margin_mix", "1.3", None), ("margin_prod", "0.3", None),
    # ---- Which weights we freeze
    ("boundary", MINUS + "4.0", COMMITTED),
    ("bound_mix_40", MINUS + "4.0", None),
    ("bound_prod_41", MINUS + "4.1", None),
    ("frozen_averitec", "0.869", None),
    ("frozen_prod_expected", "0.897", None),
    ("n3000", "3,000", COMMITTED),
    # ---- The reader
    ("n_judged", "120", BLIND),
    ("per_cell", "30", BLIND),
    ("rf_j1", "0 of 30", BLIND), ("rf_bl", "3 of 30", BLIND),
    ("st_j1", "0 of 30", BLIND), ("st_bl", "0 of 30", BLIND),
    ("rt_j1", "23 of 30", BLIND), ("rt_bl", "22 of 30", BLIND),
    ("sf_j1", "14 of 30", BLIND), ("sf_bl", "11 of 30", BLIND),
    ("true_doc_pct", "5%", None), ("false_doc_pct", "7%", None),
    ("wrong_ref500", "500", None), ("wrong_sup350", "350", None),
    ("refute60", "60", None),
    ("err_2237", "22 of 37", None), ("err_8", "8", None),
    # ---- What fixing the reader would be worth (training)
    ("fz_tr_auc", "0.860", BASEREF), ("fz_tr_rec", "38.2%", BASEREF),
    ("fz_tr_false", "573", BASEREF), ("fz_tr_true", "31", BASEREF),
    ("corr_tr_auc", "0.892", None), ("corr_tr_rec", "50.1%", None),
    ("corr_tr_false", "751", None), ("corr_tr_true", "13", None),
    # (production)
    ("fz_pr_auc", "0.927", BASEREF), ("fz_pr_rec", "37.2%", BASEREF),
    ("fz_pr_false", "558", BASEREF), ("fz_pr_true", "30", BASEREF),
    ("corr_pr_auc", "0.962", None), ("corr_pr_rec", "53.8%", None),
    ("corr_pr_false", "807", None), ("corr_pr_true", "12", None),
    ("more_false_tr", "178", None), ("fewer_true", "18", None),
    ("more_false_pr", "249", None),
    # ---- A stronger reader on the same claims (table + prose)
    ("fl500_fz", "0.873", None), ("fl500_own", "0.866", None),
    ("v4p500_fz", "0.863", None), ("v4p500_own", "0.851", None),
    ("v4p500_d", MINUS + "0.014 [" + MINUS + "0.037, +0.010]", None),
    ("v4p500b_fz", "0.853", None), ("v4p500b_own", "0.857", None),
    ("v4p500b_d", MINUS + "0.008 [" + MINUS + "0.031, +0.015]", None),
    ("fl1000_fz", "0.862", None), ("fl1000_own", "0.858", None),
    ("v4p1000b_fz", "0.851", None), ("v4p1000b_own", "0.849", None),
    ("v4p1000b_d", MINUS + "0.009 [" + MINUS + "0.026, +0.008]", None),
    ("kimi500_fz", "0.819", None), ("kimi500_own", "0.848", None),
    ("kimi500_d", MINUS + "0.019 [" + MINUS + "0.054, +0.015]", None),
    ("vote_wd270", "270", None), ("vote_r214", "214", None),
    ("vote_add112", "112", None), ("vote_r77", "77", None),
    ("v5b_add454", "454", None), ("v5b_wd416", "416", None),
    ("v5b_153", "153", None), ("v5b_518", "518", None),
    ("kimi_wd419", "419", None), ("kimi_r336", "336", None), ("kimi_add95", "95", None),
    # ---- Retrieval (oracle table)
    ("one_contra_tr", "50.0%", None), ("one_contra_pr", "40.9%", None),
    ("both_tr", "87.5%", None), ("both_pr", "84.1%", None),
    # ---- What a second search engine actually bought
    ("exa1500", "1,500", None),
    ("quarter", "quarter", None),
    ("serper_ratio", "seven", None),
    ("exa_ratio", "six", None),
    ("refute60b", "60", None),
]


def build_values(verify: bool) -> tuple[dict[str, str], list[dict]]:
    """Return the substitution map V and the per-number audit rows.

    Committed ids take their regenerated value; no-source ids take the page
    literal. In --verify a committed id whose regen != page literal is a MISMATCH.
    """
    d = derive()
    V: dict[str, str] = {}
    rows: list[dict] = []
    seen: set[str] = set()
    for nid, page, src in NUMBERS:
        if nid in seen:                       # id reused across the page: value already fixed
            rows.append({"id": nid, "page": page, "regen": V[nid], "src": src or "-",
                         "status": "committed(dup)" if src else "no-source(dup)"})
            continue
        seen.add(nid)
        if src:
            regen = d[nid]
            V[nid] = regen
            status = "OK" if regen == page else "MISMATCH"
            rows.append({"id": nid, "page": page, "regen": regen, "src": src, "status": status})
        else:
            V[nid] = page
            rows.append({"id": nid, "page": page, "regen": None, "src": "-", "status": "no-source"})
    return V, rows


# ---------------------------------------------------------------- templates
# Verbatim page prose/structure; only ${id} tokens are substituted. Figures are
# NOT regenerable from committed files, so each <figure> is a placeholder that
# names the untracked run outputs it would need (the generator fabricates nothing).
_FIG = ('<figure><div class="figph">[figure not regenerable from committed metric files &mdash; '
        'needs {need}]</div><figcaption>{cap}</figcaption></figure>')


def _fig(cap: str, need: str) -> str:
    return _FIG.format(cap=cap, need=need)


HTML_TMPL = Template("""<title>Final Instrument for Survey Experiment</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{--bg:#fbfaf7;--ink:#1c1b19;--muted:#5d5a54;--rule:#d9d5cc;--soft:#efece5;--accent:#2b2a27}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#171613;--ink:#ece8df;--muted:#a49f95;--rule:#3a3832;--soft:#23221e;--accent:#e6e2d8}}
:root[data-theme="dark"]{--bg:#171613;--ink:#ece8df;--muted:#a49f95;--rule:#3a3832;--soft:#23221e;--accent:#e6e2d8}
body{background:var(--bg);color:var(--ink);margin:0;font-family:"IBM Plex Sans",-apple-system,Helvetica,Arial,sans-serif;font-size:17px;line-height:1.6}
.doc{max-width:1080px;margin:0 auto;padding-block:56px 96px;padding-inline:24px}
h1{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:44px;line-height:1.1;margin:0 0 36px;text-wrap:balance}
h2{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:30px;margin:64px 0 14px;padding-top:22px;border-top:1px solid var(--rule);text-wrap:balance}
h3{font-size:17px;font-weight:600;letter-spacing:.01em;margin:30px 0 8px}
p{margin:12px 0}
ul{margin:10px 0 10px 22px}li{margin:6px 0}
.key{background:var(--soft);border-left:3px solid var(--accent);padding:14px 18px;margin:20px 0}
figure{margin:26px 0 30px}
.figph{width:100%;box-sizing:border-box;border:1px dashed var(--rule);background:var(--soft);color:var(--muted);padding:28px 18px;font-size:14px;text-align:center}
figcaption{font-size:15px;color:var(--muted);margin-top:8px}
.tbl{overflow-x:auto;margin:18px 0}
table{border-collapse:collapse;width:100%;font-size:15.5px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--rule);padding:9px 12px;text-align:right;vertical-align:top}
th:first-child,td:first-child,th.l,td.l{text-align:left}
thead th{font-weight:600;border-bottom:2px solid var(--ink)}
tbody tr.hl td{font-weight:600}
.note{font-size:15px;color:var(--muted)}
@media (max-width:700px){h1{font-size:34px}h2{font-size:26px}}
</style>
<div class="doc">

<h1>Final Instrument for Survey Experiment</h1>

<h2>Two ways to fit the weights</h2>

<p>The claims in fc-gold are old. Every one of them has a published fact-check and the web has had years to write about them. So we have to decide what evidence the search is allowed to see when we fit.</p>

<p>Training mode only lets the search see documents published before the claim. That is what a claim looks like on the day it appears. It also removes fact-check sites and articles that copy a fact-check.</p>

<p>Production mode lets the search see any date. It still removes fact-check sites and their copies. That one restriction stays only because every fc-gold claim has a fact-check by construction, and letting the search find it would make the fit circular. Real production will not remove them.</p>

<div class="tbl"><table>
<thead><tr><th class="l">Fit</th><th class="l">What the search sees</th><th>AUC</th><th>95% CI width</th><th>AVeriTeC AUC</th><th>95% CI width</th></tr></thead>
<tbody>
<tr class="hl"><td class="l">Training mode</td><td class="l">Documents dated before the claim. Fact-checks removed.</td><td>${train_auc}</td><td>${train_ci}</td><td>${train_averitec}</td><td>${train_averitec_ci}</td></tr>
<tr><td class="l">Production mode</td><td class="l">Any date. Fact-checks removed.</td><td>${prod_auc}</td><td>${prod_ci}</td><td>${prod_averitec}</td><td>${prod_averitec_ci}</td></tr>
</tbody></table></div>
<p class="note">Out of fold, five folds, claims about the same story kept in the same fold. CI width is the full width of the 95% interval. AVeriTeC is a 500 claim public benchmark with its own evidence, used as an outside check on each weight set.</p>

<h2>What we found in production mode</h2>

<p>Production mode scores seven points higher. We checked whether that gain is leaked fact-checks. We hand read ${hand60a} documents that production mode called refuting. None was a fact-check on an unlisted site and none was a news article copying one. They were primary sources, news reporting and reference pages. And the production weights do about as well on AVeriTeC as the training weights do. The gain is real evidence, not leakage.</p>

<p>The weights themselves barely move. The two decisive flags, a document that states the claim and a document that contradicts it, get the same weight in both modes. The one weight that changes is the irrelevant flag.</p>

${fig_weights}

<p>The reason is silence. Some claims come back with ten irrelevant documents. In training mode that happens to ${sil_train_true} true claims and ${sil_train_false} false ones, so silence tells you little and the fit gives an irrelevant document a weight near zero. In production mode a true claim almost always finds something on the open web. Silence drops to ${sil_prod_true} true claims and stays at ${sil_prod_false} false ones. The fit learns that and gives an irrelevant document a weight of ${wI_prod} instead of ${wI_train}.</p>

${fig_silence}

<p>A silent claim is never flagged in either mode. Ten irrelevant documents score ${score_ten_irr} with production weights and the boundary sits at ${bound_prod_ex}. To be flagged a claim needs a contradicting document. One contradiction among nine irrelevant documents scores ${score_one_contra} and is flagged. That is the behaviour we want. Silence alone passes, silence plus a refutation does not.</p>

${fig_roc}

<h2>Each weight set on the other evidence</h2>

<p>The two fits use different evidence, so their AUCs are not the same test. The fair test is to cross them. Take the production weights and apply them to training mode evidence, the evidence a claim has on the day it appears. Then take the training weights and apply them to production mode evidence. Same folds in both directions, so no claim is ever scored by weights that saw it.</p>

<div class="tbl"><table>
<thead><tr><th class="l">Weights</th><th class="l">Evidence</th><th>AUC</th><th>95% CI width</th><th>Recall at 2% FPR</th></tr></thead>
<tbody>
<tr><td class="l">Training mode</td><td class="l">Training mode</td><td>${train_auc}</td><td>${train_ci}</td><td>${tt_recall}</td></tr>
<tr><td class="l">Production mode</td><td class="l">Training mode</td><td>${pt_auc}</td><td>${pt_ci}</td><td>${pt_recall}</td></tr>
<tr><td class="l">Training mode</td><td class="l">Production mode</td><td>${tp_auc}</td><td>${tp_ci}</td><td>${tp_recall}</td></tr>
<tr><td class="l">Production mode</td><td class="l">Production mode</td><td>${pp_auc}</td><td>${pp_ci}</td><td>${pp_recall}</td></tr>
</tbody></table></div>
<p class="note">Recall counts every false claim, silent ones included. The boundary is always set on the evidence being scored.</p>

<p>Production weights on day zero evidence lose ${lose_016} against the training fit, and the paired test says that loss is real. Training weights on production evidence lose nothing. They come out ${above_003} above the production fit itself, and the paired test says even that small edge is real. They give up about two points of recall at the boundary. The lesson is that the evidence matters far more than which fit made the weights. Moving from day zero evidence to production evidence is worth seven points. Which weights you use is worth at most two.</p>

<p>The one loss has one cause. Production weights treat an irrelevant document as a mild sign of falsehood. On day zero evidence that sign is weak, so the weights over-read it. Training weights treat it as nothing, which costs a little recall at the boundary but nothing in ranking. The same thing explains the AVeriTeC gap. Swap the irrelevant weight and the two sets score the same there.</p>

<h2>The evidence production will actually see</h2>

<p>Neither end is what production looks like. Production will score posts that are days or weeks old. Some will have plenty written about them and some will have nothing yet. So we built an evaluation whose evidence matches that mix.</p>

<p>The September survey pool tells us how old a post is when we score it. The median is ${median_days} days and ${pct_older} are older than two weeks.</p>

${fig_age}

<p>For each fc-gold claim we drew an age from that distribution and hid every document published later than that. A claim drawn at 20 days sees only what the web had 20 days after it appeared. Undated documents are kept. This gives the same 3,000 labelled claims with the evidence availability production will have. We then fitted the weights on that mix, and we also applied the training weights and the production weights to it unchanged.</p>

${fig_mix}

<div class="tbl"><table>
<thead><tr><th class="l">Weights applied to the mixed evidence</th><th>AUC</th><th>95% CI width</th><th>Recall at 2% FPR</th><th>Boundary</th></tr></thead>
<tbody>
<tr><td class="l">Fitted on the mix</td><td>${mix_fit_auc}</td><td>${mix_fit_ci}</td><td>${mix_fit_rec}</td><td>${mix_fit_bound}</td></tr>
<tr><td class="l">Training weights</td><td>${mix_tr_auc}</td><td>${mix_tr_ci}</td><td>${mix_tr_rec}</td><td>${mix_tr_bound}</td></tr>
<tr><td class="l">Production weights</td><td>${mix_pr_auc}</td><td>${mix_pr_ci}</td><td>${mix_pr_rec}</td><td>${mix_pr_bound}</td></tr>
</tbody></table></div>
<p class="note">Five different random age draws give the same AUCs to the third decimal.</p>

<p>All three rank the claims the same way. The training weights come out ${ahead_004} ahead of the fit on the mix, and the paired test says that small edge is real. The production weights are indistinguishable from the fit. The training weights pay for that in recall at the boundary, about three points below the other two. With the irrelevant weight near zero, a single contradiction among nine irrelevant documents lands right on the boundary instead of clearly below it.</p>

<p>All three weight sets pass the two checks we care about. A claim with ten irrelevant documents is never flagged. A claim with one contradicting document among nine irrelevant ones is flagged. The margin above the silent pile is ${margin_train} for the training weights, ${margin_mix} for the mix fit and ${margin_prod} for the production weights.</p>

<p>Whichever weights we choose, the boundary has to be set on this mixed evidence, not on training mode evidence. The training boundary of ${boundary} was set where training mode scores 2% of true claims. On production evidence the scores spread wider and the same boundary would flag many more true claims than that.</p>


<h2>Which weights we freeze</h2>

<p>The training weights. They rank at least as well as any other set on every kind of evidence we tested, day zero, production and the mix, and they are the fit nobody can call inflated. They give up two or three points of recall at the boundary, and we accept that for a cleaner instrument. The boundary stays where the training fit put it, at ${boundary}. Set on the mixed evidence it comes out at ${bound_mix_40} as well, and on production evidence at ${bound_prod_41}, so the training weights and their own boundary ship as one object.</p>

<div class="key">Frozen instrument. Seven flags, read-v5 on DeepSeek-V4-Flash, Serper top ten, training-mode weights, boundary ${boundary}. Evaluation number, AUC ${train_auc} on fc-gold in training mode, ${frozen_averitec} on AVeriTeC. Expected on production-age evidence, AUC ${frozen_prod_expected}.</div>

<p>The rest of this page is about the two things we chose not to spend on, a stronger reader and a different search engine. For each we show how much a perfect version would gain, and what the version we could actually buy gained.</p>

<h2>The reader</h2>

<p>We hand read ${n_judged} documents from the production reads, ${per_cell} from each of four cells. Two judges did it, the second one blind to the gold label, the cell and the reader's flag. Both found the same thing.</p>

<div class="tbl"><table>
<thead><tr><th class="l">Document</th><th>Reader errors, judge one</th><th>Reader errors, blind judge</th></tr></thead>
<tbody>
<tr><td class="l">Refuting flag on a false claim</td><td>${rf_j1}</td><td>${rf_bl}</td></tr>
<tr><td class="l">Supporting flag on a true claim</td><td>${st_j1}</td><td>${st_bl}</td></tr>
<tr><td class="l">Refuting flag on a true claim</td><td>${rt_j1}</td><td>${rt_bl}</td></tr>
<tr><td class="l">Supporting flag on a false claim</td><td>${sf_j1}</td><td>${sf_bl}</td></tr>
</tbody></table></div>

<p>When the reader agrees with the gold it is almost always right. When it disagrees it is usually wrong. Those disagreeing cells are small, ${true_doc_pct} of the documents on true claims and ${false_doc_pct} of the documents on false claims, but across the 3,000 claims they add up to roughly ${wrong_ref500} wrong refutations of true claims and ${wrong_sup350} wrong supports of false claims. No fact-check leaks were found in the ${refute60} refuting documents.</p>

<p>The errors are one kind of mistake. In ${err_2237} cases the reader took a sentence about an adjacent fact, a different instance, a proposal instead of the enacted rule, a floor figure instead of the stated one, and graded the whole claim from it. In ${err_8} cases a debunk quoted the claim it was about to reject and the reader read the quote. No error was a matter of strength, a 1 that should have been a 2. Every error flipped the direction or graded an off-scope document.</p>

<h3>What fixing the reader would be worth</h3>

<p>We replaced the wrong flags with the flag the blind judge said the document deserved, at the rates above, rescored every claim with the frozen weights, and counted what changes at the 2% boundary. The weights are applied as they are, not refitted, so the baseline rows differ slightly from the out-of-fold numbers above.</p>

<div class="tbl"><table>
<thead><tr><th class="l">Training mode evidence</th><th>AUC</th><th>Recall at 2% FPR</th><th>False claims flagged</th><th>True claims flagged</th></tr></thead>
<tbody>
<tr><td class="l">Frozen instrument</td><td>${fz_tr_auc}</td><td>${fz_tr_rec}</td><td>${fz_tr_false}</td><td>${fz_tr_true}</td></tr>
<tr><td class="l">Reader errors corrected</td><td>${corr_tr_auc}</td><td>${corr_tr_rec}</td><td>${corr_tr_false}</td><td>${corr_tr_true}</td></tr>
</tbody></table></div>

<div class="tbl"><table>
<thead><tr><th class="l">Production mode evidence</th><th>AUC</th><th>Recall at 2% FPR</th><th>False claims flagged</th><th>True claims flagged</th></tr></thead>
<tbody>
<tr><td class="l">Frozen instrument</td><td>${fz_pr_auc}</td><td>${fz_pr_rec}</td><td>${fz_pr_false}</td><td>${fz_pr_true}</td></tr>
<tr><td class="l">Reader errors corrected</td><td>${corr_pr_auc}</td><td>${corr_pr_rec}</td><td>${corr_pr_false}</td><td>${corr_pr_true}</td></tr>
</tbody></table></div>

<p>In training mode the fix flags ${more_false_tr} more false claims, ${fz_tr_false} to ${corr_tr_false}, and ${fewer_true} fewer true claims, ${fz_tr_true} to ${corr_tr_true}. In production mode it flags ${more_false_pr} more false claims, ${fz_pr_false} to ${corr_pr_false}, and again ${fewer_true} fewer true claims. That is the ceiling for the reader, and it is large. The reason is not that the reader misses refutations. It is that its wrong refutations of true claims pin the boundary low, and once they are gone the boundary can rise and catch false claims that were already scored below it. A better reader would also change the fit a little. Refitting the weights on the corrected reads adds three to four points of recall on top of these numbers, so the weights barely move but the instrument would be refitted.</p>

<h3>A stronger reader on the same claims</h3>

<p>The cleanest test changes one thing. We took 500 fc-gold claims, 250 true and 250 false, kept the same documents the current reader saw in training mode, and had DeepSeek-V4-Pro read every one of them with the same prompt. We then scored the claims two ways, with the frozen weights and with weights refitted on V4-Pro's own reads, so a reader that spreads its flags differently gets a fair chance. Same claims, same documents, paired bootstrap.</p>

<div class="tbl"><table>
<thead><tr><th class="l">Reader</th><th class="l">Prompt</th><th>Claims</th><th>AUC, frozen weights</th><th>AUC, own weights</th><th>Own weights, difference vs current reader</th></tr></thead>
<tbody>
<tr class="hl"><td class="l">DeepSeek-V4-Flash, current</td><td class="l">v5, frozen</td><td>500</td><td>${fl500_fz}</td><td>${fl500_own}</td><td></td></tr>
<tr><td class="l">DeepSeek-V4-Pro</td><td class="l">v5, frozen</td><td>500</td><td>${v4p500_fz}</td><td>${v4p500_own}</td><td>${v4p500_d}</td></tr>
<tr><td class="l">DeepSeek-V4-Pro</td><td class="l">v5b, scope rule rewritten</td><td>500</td><td>${v4p500b_fz}</td><td>${v4p500b_own}</td><td>${v4p500b_d}</td></tr>
<tr class="hl"><td class="l">DeepSeek-V4-Flash, current</td><td class="l">v5, frozen</td><td>1,000</td><td>${fl1000_fz}</td><td>${fl1000_own}</td><td></td></tr>
<tr><td class="l">DeepSeek-V4-Pro</td><td class="l">v5b, scope rule rewritten</td><td>1,000</td><td>${v4p1000b_fz}</td><td>${v4p1000b_own}</td><td>${v4p1000b_d}</td></tr>
<tr><td class="l">Kimi K2.6, reasoning</td><td class="l">v5, frozen</td><td>500</td><td>${kimi500_fz}</td><td>${kimi500_own}</td><td>${kimi500_d}</td></tr>
</tbody></table></div>
<p class="note">Out of fold, five folds, on the sampled claims only. Brackets are the 95% interval of the paired difference. The frozen weights were fitted on all 3,000 claims, so the own-weights column is the fair comparison.</p>

<p>V4-Pro does not beat the current reader, and its own weights do not help it. The reason is simple. On the 500 claims it withdrew ${vote_wd270} of the current reader's directional votes, and ${vote_r214} of those had pointed the right way. It added ${vote_add112} new votes, ${vote_r77} of them right. The prompt tells the reader to abstain when a document does not settle the exact proposition. V4-Pro obeys that rule. The current reader reads the gist and votes, and the gist is right four times in five, so obeying the rule throws signal away.</p>

<p>v5b is the prompt with that rule replaced by one that lets a document refute a claim when the two cannot both be true. It brings the votes back, ${v5b_add454} added against ${v5b_wd416} withdrawn on the 1,000 claims, but the new votes are worse than the old ones. ${v5b_153} of its ${v5b_518} new strong refutations land on true claims. The abstentions became false alarms and the ranking did not move.</p>

<p>Kimi K2.6, a reasoning model at eleven times the price per read, does the same thing more strongly. On the 500 claims it moved a thousand reads to on-claim context, withdrew ${kimi_wd419} of the current reader's votes, ${kimi_r336} of them right, and added ${kimi_add95}. With the frozen weights it is significantly below the current reader. Its own weights recover most of that and it still lands below. The more faithfully a reader follows the scope rule, the more it costs.</p>

<h2>Retrieval</h2>

<p>The same question for search. First the ceiling, then what we could actually buy.</p>

<div class="tbl"><table>
<thead><tr><th class="l">Frozen weights, at the 2% boundary</th><th>Training mode evidence, recall</th><th>Production mode evidence, recall</th></tr></thead>
<tbody>
<tr><td class="l">Frozen instrument</td><td>${fz_tr_rec}</td><td>${fz_pr_rec}</td></tr>
<tr><td class="l">Reader errors corrected</td><td>${corr_tr_rec}</td><td>${corr_pr_rec}</td></tr>
<tr><td class="l">One contradicting document given to every false claim that had none</td><td>${one_contra_tr}</td><td>${one_contra_pr}</td></tr>
<tr><td class="l">Both</td><td>${both_tr}</td><td>${both_pr}</td></tr>
</tbody></table></div>

<p>The retrieval row is an oracle. It knows which claims are false and hands each one the document it needs. Production will not work like that. Some false claims have no refutation anywhere and some true claims have no support. It only says how much room there is. The room is about the same size as the reader's, and the two add up, because retrieval supplies the missing refutations and the reader fix stops true claims being dragged below the line. The refutation also has to be decisive. Give the same claims an undermining document instead of a contradicting one and almost nothing changes. With the frozen weights the oracle gains less on production evidence than on training evidence, because a lone contradiction among nine irrelevant documents lands right on the boundary and the other documents on those claims offset it. That is the recall we chose to give up with the training weights.</p>

<h3>What a second search engine actually bought</h3>

<p>Exa is a neural search engine. It matches on meaning rather than words, so it should find a refuting document even when the wording differs. We ran it on ${exa1500} claims in production mode, alone and combined with Serper, and fitted the weights on each set of reads.</p>

<p>Exa finds more evidence. On the false claims it returns a ${quarter} more refuting documents than Serper, and combined with Serper the count doubles. It also returns more supporting documents for false claims and more refuting documents for true claims. The extra evidence points the wrong way about as often as the evidence Serper already had. Serper's documents point the right way ${serper_ratio} times for every one that points the wrong way, Exa's ${exa_ratio} times. The ranking is set by that ratio, not by the number of documents, so the AUC does not move. Exa alone is a point below Serper, and both together tie Serper.</p>

${fig_exa}

<p>A hand check of ${refute60b} refuting documents per engine found both mostly genuine and neither leaking fact-checks. So a second engine with its own queries bought nothing. Reaching the retrieval ceiling would take more than ten documents, several queries per claim, whole pages instead of snippets, and primary sources. That is a different instrument, and we keep this one.</p>

<h2>What we freeze</h2>
<ul>
<li>Seven flags, read-v5 on DeepSeek-V4-Flash, Serper top ten.</li>
<li>Training-mode weights, fitted on ${n3000} balanced fc-gold claims, boundary ${boundary}.</li>
<li>Evaluation number, AUC ${train_auc} on fc-gold in training mode and ${frozen_averitec} on AVeriTeC.</li>
<li>No silence rule. Ten irrelevant documents never flag, one contradiction among nine does.</li>
<li>No stronger reader model, no second search engine. DeepSeek-V4-Pro and Kimi K2.6 on the same claims do not beat the current reader, with either prompt or their own weights. The prompt's scope rule is the first thing to work on after the survey, and the frontier readers are still untested.</li>
</ul>

</div>
""")


def render_html(V: dict[str, str]) -> str:
    figs = {
        "fig_weights": _fig("Fitted weights with 95% intervals. The shape is the same in both "
                            "modes. The irrelevant flag is the one that moves.",
                            "training weights (headline_metrics.json, committed) + production "
                            "weights (weights_v5_production_bal3000.json, untracked)"),
        "fig_silence": _fig("Claims where all ten documents were irrelevant. In training mode "
                            "silence tells you little. In production mode it leans false.",
                            "training + production reads over fc_gold_bal3000 (untracked)"),
        "fig_roc": _fig("The full trade-off for each fit. Higher and further left is better.",
                        "training + production oof scores (untracked reads)"),
        "fig_age": _fig("1,660 survey claims. Age is the scoring date minus the post date.",
                        "the September survey pool parquet (untracked)"),
        "fig_mix": _fig("Three weight sets scored on the same mixed-age evidence. Bars are AUC, "
                        "lines are 95% intervals.", "mixed-age fit outputs (untracked)"),
        "fig_exa": _fig("1,500 claims, production mode. A document points the right way when it "
                        "refutes a false claim or supports a true one. More documents, same "
                        "ratio, same AUC.",
                        "collab_exa_simple.png inputs = prodregime Exa/Serper reads (untracked)"),
    }
    return HTML_TMPL.substitute({**V, **figs})


MD_TMPL = Template("""# Final Instrument for Survey Experiment

## Two ways to fit the weights

| Fit | What the search sees | AUC | 95% CI width | AVeriTeC AUC | 95% CI width |
|---|---|--:|--:|--:|--:|
| **Training mode** | Documents dated before the claim. Fact-checks removed. | ${train_auc} | ${train_ci} | ${train_averitec} | ${train_averitec_ci} |
| Production mode | Any date. Fact-checks removed. | ${prod_auc} | ${prod_ci} | ${prod_averitec} | ${prod_averitec_ci} |

## What we found in production mode

Production mode scores seven points higher. We hand read ${hand60a} refuting documents; none leaked a fact-check. In training mode ten irrelevant documents happen to ${sil_train_true} true claims and ${sil_train_false} false ones; in production silence drops to ${sil_prod_true} true and stays at ${sil_prod_false} false, so the irrelevant weight moves to ${wI_prod} from ${wI_train}. Ten irrelevant documents score ${score_ten_irr} (boundary ${bound_prod_ex}); one contradiction among nine scores ${score_one_contra} and is flagged.

*Figure (weights, training vs production) — not regenerable from committed files (needs untracked production weights).*

## Each weight set on the other evidence

| Weights | Evidence | AUC | 95% CI width | Recall at 2% FPR |
|---|---|--:|--:|--:|
| Training mode | Training mode | ${train_auc} | ${train_ci} | ${tt_recall} |
| Production mode | Training mode | ${pt_auc} | ${pt_ci} | ${pt_recall} |
| Training mode | Production mode | ${tp_auc} | ${tp_ci} | ${tp_recall} |
| Production mode | Production mode | ${pp_auc} | ${pp_ci} | ${pp_recall} |

Production weights on day-zero evidence lose ${lose_016}; training weights on production evidence come out ${above_003} above the production fit.

## The evidence production will actually see

Median age ${median_days} days, ${pct_older} older than two weeks, ${n_survey} survey claims.

| Weights applied to the mixed evidence | AUC | 95% CI width | Recall at 2% FPR | Boundary |
|---|--:|--:|--:|--:|
| Fitted on the mix | ${mix_fit_auc} | ${mix_fit_ci} | ${mix_fit_rec} | ${mix_fit_bound} |
| Training weights | ${mix_tr_auc} | ${mix_tr_ci} | ${mix_tr_rec} | ${mix_tr_bound} |
| Production weights | ${mix_pr_auc} | ${mix_pr_ci} | ${mix_pr_rec} | ${mix_pr_bound} |

Training weights come out ${ahead_004} ahead of the mix fit. Margin above the silent pile: ${margin_train} (training), ${margin_mix} (mix), ${margin_prod} (production).

## Which weights we freeze

Training weights, boundary ${boundary} (${bound_mix_40} on the mix, ${bound_prod_41} on production).

> **Frozen instrument.** Seven flags, read-v5 on DeepSeek-V4-Flash, Serper top ten, training-mode weights, boundary ${boundary}. AUC ${train_auc} on fc-gold in training mode, ${frozen_averitec} on AVeriTeC. Expected on production-age evidence, AUC ${frozen_prod_expected}.

## The reader

${n_judged} documents hand read, ${per_cell} per cell, two judges.

| Document | Reader errors, judge one | Reader errors, blind judge |
|---|--:|--:|
| Refuting flag on a false claim | ${rf_j1} | ${rf_bl} |
| Supporting flag on a true claim | ${st_j1} | ${st_bl} |
| Refuting flag on a true claim | ${rt_j1} | ${rt_bl} |
| Supporting flag on a false claim | ${sf_j1} | ${sf_bl} |

Disagreeing cells are ${true_doc_pct} of documents on true claims and ${false_doc_pct} on false claims, ~${wrong_ref500} wrong refutations of true claims and ~${wrong_sup350} wrong supports of false claims; no leaks in the ${refute60} refuting documents. ${err_2237} errors were off-scope grading, ${err_8} were quoted-claim reads.

### What fixing the reader would be worth

| Training mode evidence | AUC | Recall at 2% FPR | False flagged | True flagged |
|---|--:|--:|--:|--:|
| Frozen instrument | ${fz_tr_auc} | ${fz_tr_rec} | ${fz_tr_false} | ${fz_tr_true} |
| Reader errors corrected | ${corr_tr_auc} | ${corr_tr_rec} | ${corr_tr_false} | ${corr_tr_true} |

| Production mode evidence | AUC | Recall at 2% FPR | False flagged | True flagged |
|---|--:|--:|--:|--:|
| Frozen instrument | ${fz_pr_auc} | ${fz_pr_rec} | ${fz_pr_false} | ${fz_pr_true} |
| Reader errors corrected | ${corr_pr_auc} | ${corr_pr_rec} | ${corr_pr_false} | ${corr_pr_true} |

Training: ${more_false_tr} more false (${fz_tr_false} to ${corr_tr_false}), ${fewer_true} fewer true (${fz_tr_true} to ${corr_tr_true}). Production: ${more_false_pr} more false (${fz_pr_false} to ${corr_pr_false}).

### A stronger reader on the same claims

| Reader | Prompt | Claims | AUC frozen | AUC own | Own-weights diff vs current |
|---|---|--:|--:|--:|--:|
| **Flash, current** | v5 | 500 | ${fl500_fz} | ${fl500_own} | |
| V4-Pro | v5 | 500 | ${v4p500_fz} | ${v4p500_own} | ${v4p500_d} |
| V4-Pro | v5b | 500 | ${v4p500b_fz} | ${v4p500b_own} | ${v4p500b_d} |
| **Flash, current** | v5 | 1,000 | ${fl1000_fz} | ${fl1000_own} | |
| V4-Pro | v5b | 1,000 | ${v4p1000b_fz} | ${v4p1000b_own} | ${v4p1000b_d} |
| Kimi K2.6 | v5 | 500 | ${kimi500_fz} | ${kimi500_own} | ${kimi500_d} |

V4-Pro withdrew ${vote_wd270} votes (${vote_r214} right), added ${vote_add112} (${vote_r77} right). v5b added ${v5b_add454} against ${v5b_wd416} withdrawn; ${v5b_153} of ${v5b_518} new strong refutations land on true claims. Kimi withdrew ${kimi_wd419} (${kimi_r336} right), added ${kimi_add95}.

## Retrieval

| Frozen weights, at the 2% boundary | Training, recall | Production, recall |
|---|--:|--:|
| Frozen instrument | ${fz_tr_rec} | ${fz_pr_rec} |
| Reader errors corrected | ${corr_tr_rec} | ${corr_pr_rec} |
| One contradicting document to every false claim that had none | ${one_contra_tr} | ${one_contra_pr} |
| Both | ${both_tr} | ${both_pr} |

### What a second search engine actually bought

Exa on ${exa1500} claims: a ${quarter} more refuting documents on false claims. Serper points the right way ${serper_ratio} times for every wrong one, Exa ${exa_ratio}. Same ratio, same AUC; Exa alone a point below Serper, both together tie Serper. ${refute60b} refuting documents hand checked per engine, no leaks.

*Figure (collab_exa_simple: right-way vs wrong-way documents per claim) — not regenerable from committed files (needs untracked prodregime reads).*

## What we freeze

- Seven flags, read-v5 on DeepSeek-V4-Flash, Serper top ten.
- Training-mode weights, fitted on ${n3000} balanced fc-gold claims, boundary ${boundary}.
- Evaluation number, AUC ${train_auc} on fc-gold in training mode and ${frozen_averitec} on AVeriTeC.
- No silence rule. Ten irrelevant documents never flag, one contradiction among nine does.
- No stronger reader model, no second search engine.
""")


def render_md(V: dict[str, str]) -> str:
    return MD_TMPL.substitute(V)


# ---------------------------------------------------------------- driver
def print_gate(rows: list[dict]) -> int:
    committed = [r for r in rows if r["status"] in ("OK", "MISMATCH")]
    dups = [r for r in rows if r["status"].endswith("(dup)")]
    nosrc = [r for r in rows if r["status"].startswith("no-source")]
    mism = [r for r in committed if r["status"] == "MISMATCH"]

    print("=" * 78)
    print("GATE: every number on the live page vs the regenerated value")
    print("=" * 78)
    print(f"{'id':<18}{'page':<26}{'regen':<26}status")
    print("-" * 78)
    print("[ DERIVED FROM COMMITTED METRIC FILES ]")
    for r in committed:
        print(f"{r['id']:<18}{r['page']:<26}{str(r['regen']):<26}{r['status']}  <- {r['src']}")
    print("\n[ NO COMMITTED SOURCE  (carried verbatim from the live page) ]")
    for r in nosrc:
        print(f"{r['id']:<18}{r['page']:<26}{'(page literal)':<26}no-source")
    print(f"\nunique committed {len({r['id'] for r in committed})}  "
          f"| unique no-source {len({r['id'] for r in nosrc})}  "
          f"| reused placements {len(dups)}  | mismatches {len(mism)}")
    if mism:
        print("\n!!! MISMATCHES (page value != committed metric file) !!!")
        for r in mism:
            print(f"  {r['id']}: page {r['page']!r} vs committed {r['regen']!r} ({r['src']})")
    return 1 if mism else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUTDIR,
                    help=f"output directory for the .md/.html (default {OUTDIR})")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing output files")
    ap.add_argument("--verify", action="store_true",
                    help="run the number-by-number gate and exit; write nothing")
    args = ap.parse_args()

    V, rows = build_values(args.verify)

    if args.verify:
        raise SystemExit(print_gate(rows))

    # gate always runs before writing; refuse to ship a page that disagrees with the files
    if print_gate(rows):
        raise SystemExit("refusing to write: a committed metric file disagrees with the page "
                         "(see MISMATCHES above). Fix the page prose or the source, do not paper over it.")

    args.out.mkdir(parents=True, exist_ok=True)
    md_path = args.out / "final_instrument.md"
    html_path = args.out / "final_instrument.html"
    for p in (md_path, html_path):
        if p.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {p}: pass --force to regenerate in place.")
    md_path.write_text(render_md(V))
    html_path.write_text(render_html(V))
    print(f"\nwrote {md_path}")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
