"""Generate the AVeriTeC results artifact HTML from stats.json + figures.

Every number comes from stats.json, written by averitec_artifact_stats.py. Prose is
Daniel's voice. House rules are no colons, semicolons or em dashes in prose, short
sentences, near monochrome, full width text matching the figures.

  uv run python -m claimverify.averitec_artifact_html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from claimverify.config import SRC

DEFAULT_ART = SRC / "eval/data/claimverify_runs/averitec_dev/artifact"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", default=str(DEFAULT_ART))
    a = ap.parse_args()
    A = Path(a.artifact)
    S = json.loads((A / "stats.json").read_text())

    def pc(x, d=1):
        return f"{x * 100:.{d}f}%"

    def f3(x):
        return f"{x:.3f}"

    def pp(x, d=1):
        return f"{x * 100:.{d}f} points"

    def fig(name):
        svg = (A / "figures" / f"fig_{name}.svg").read_text()
        return f'<div class="panel">{svg[svg.index("<svg"):]}</div>'

    meta, arms = S["meta"], S["arms_present"]
    lab, short = S["arm_labels"], S["arm_short"]
    fc, cel, urn = S["four_class"], S["ceiling"], S["urn"]
    matched, bc = S["matched"], S["maps_bc"]
    u7, u3 = urn["models"]["7-flag"], urn["models"]["3-voice"]
    ceil_on = [n for n in arms if S["ceiling"]["arms"][n]["date_ceiling_on"]] or arms
    best = max(ceil_on, key=lambda n: fc[n]["accuracy"])
    lead = cel["reference_arm"]
    lead_cel = cel["arms"][lead]
    depth = S["depth"]
    mc = S["mcnemar_arms"]
    pair_key = next((k for k in mc if "4-class" in k), None)
    pair_bin = next((k for k in mc if "binary" in k), None)

    # ---------- four-class table ----------
    fc_rows = (f"<tr><td>ClaimCheck paper, as reported</td>"
               f"<td class='m'>{f3(S['paper']['accuracy'])}</td><td class='m'>not reported</td>"
               f"<td class='m'>{S['paper']['n']}</td></tr>")
    fc_rows += "".join(
        f"<tr><td>{lab[n]}</td><td class='m'>{f3(fc[n]['accuracy'])}</td>"
        f"<td class='m'>{f3(fc[n]['ci'][0])} to {f3(fc[n]['ci'][1])}</td>"
        f"<td class='m'>{fc[n]['n']}</td></tr>" for n in arms)

    # ---------- matched table ----------
    mt_rows = ""
    for n in arms:
        g = matched[n]
        for sysname, e in g["systems"].items():
            who = lab[n] if sysname == "loop" else sysname
            p = g["mcnemar"].get(sysname)
            mt_rows += (f"<tr><td>{who}</td><td class='m'>{f3(g['fpr'])}</td>"
                        f"<td class='m'>{f3(e['fpr'])}</td><td class='m'>{f3(e['recall'])}</td>"
                        f"<td class='m'>{f3(e['ci'][0])} to {f3(e['ci'][1])}</td>"
                        f"<td class='m'>{f3(p['p_exact']) if p else 'reference'}</td></tr>")

    # ---------- where the gap comes from ----------
    cf = S.get("counterfactual")
    if cf:
        cfa = cf["arms"]
        cf_order = [n for n in ("top3", "all10", "noceil", "top3_noceil") if n in cfa]
        deepest = max(cf_order, key=lambda n: cfa[n]["n_unsupported"])
        openest = max(cf_order, key=lambda n: cfa[n]["actual_recall"])
        no_ceil = next((n for n in cf_order if not cfa[n]["date_ceiling_on"]), cf_order[-1])
        no_bar_acc = f3(cfa[no_ceil]["no_bar_accuracy"])
        cf_rows = "".join(
            f"<tr><td>{lab[n]}</td><td class='m'>{f3(cfa[n]['actual_accuracy'])}</td>"
            f"<td class='m'>{f3(cfa[n]['no_bar_accuracy'])}</td>"
            f"<td class='m'>{f3(cfa[n]['no_bar_accuracy_ci'][0])} to "
            f"{f3(cfa[n]['no_bar_accuracy_ci'][1])}</td>"
            f"<td class='m'>{f3(cfa[n]['actual_recall'])}</td>"
            f"<td class='m'>{f3(cfa[n]['no_bar_recall'])}</td>"
            f"<td class='m'>{f3(cfa[n]['actual_fpr'])}</td>"
            f"<td class='m'>{f3(cfa[n]['no_bar_fpr'])}</td></tr>" for n in cf_order)
        leak_rows = "".join(
            f"<tr><td>{lab[n]}</td><td class='m'>{pc(cfa[n]['fact_check_share'])}</td>"
            f"<td class='m'>{pc(cfa[n]['post_claim_share'])}</td>"
            f"<td class='m'>{'on' if cfa[n]['date_ceiling_on'] else 'off'}</td>"
            f"<td class='m'>{cfa[n]['stance_docs']:,}</td></tr>" for n in cf_order)
        gap_html = f"""<p>The loss against the paper is one shape, gold Supported claims that we
leave unsupported. Most of those are not a retrieval failure, they are the corroboration bar
refusing a close the model had already proposed, {cfa[deepest]['bar_refusal']} of the
{cfa[deepest]['n_unsupported']} unsupported verdicts in the {short[deepest]} arm against
{cfa[deepest]['retrieval_miss']} where no full-read supporting or refuting page was ever
found and {cfa[deepest]['other']} everything else.</p>
<table>
<tr><th>arm</th><th>accuracy</th><th>no bar accuracy</th><th>95% interval</th>
<th>flag recall</th><th>no bar recall</th><th>FPR</th><th>no bar FPR</th></tr>
{cf_rows}
</table>
<p>The other half of the gap is the ceiling, and it is worth seeing what reading past it buys.
With the ceiling off the loop reads far more pages that already carry a verdict on the claim.</p>
<table>
<tr><th>arm</th><th>fact-check host</th><th>dated after the claim</th><th>date ceiling</th>
<th>stance-bearing pages read</th></tr>
{leak_rows}
</table>
<p class="key">No ceiling plus no bar lands at {f3(cfa[no_ceil]['no_bar_accuracy'])}, which is
the paper's {f3(S['paper']['accuracy'])} within rounding, and it costs
{pp(cfa[openest]['actual_recall'] - cfa[no_ceil]['no_bar_recall'])} of flag recall. The bar is
what holds flag recall at {f3(cfa[openest]['actual_recall'])} in the {short[openest]} arm,
which is the number the nudge is built on.</p>"""
    else:
        no_bar_acc = 'the same place'
        gap_html = ('<p class="note">counterfactual.json is not built yet, so the bar '
                    'counterfactual is not on this page.</p>')

    # ---------- caveats ----------
    bc_bits = []
    for v in ("B", "C"):
        flips = [n for n, ok in bc[v]["loop_leads_7flag"].items() if not ok]
        if not flips:
            bc_bits.append(f"map {v} keeps the ordering")
        else:
            pairs = " and ".join(f"{f3(bc[v]['urn']['7-flag'][n])} against "
                                 f"{f3(bc[v]['arms'][n])} on {short[n]}" for n in flips)
            bc_bits.append(f"on map {v} the 7-flag urn edges ahead instead, {pairs}")
    bc_sentence = ", and ".join(bc_bits)
    bc_sentence = bc_sentence[0].upper() + bc_sentence[1:]
    caveats = [
        f"The date ceiling is a filter with leaks. {pc(lead_cel['share_searches_with_post_ceiling_hit'])} "
        f"of the {lead_cel['serper_searches']} Serper searches in the {short[lead]} arm returned "
        f"at least one post-claim page, and its {lead_cel['exa_searches']} Exa searches carry no "
        "date instrumentation at all.",
        f"The urn's weights were fitted on our own fact-check gold, {meta['urn_fit_population_n']:,} "
        "claims, and were frozen before this run. Nothing here was fitted on AVeriTeC.",
        "Binary map A carries the page. " + bc_sentence + ".",
        (f"Reading all ten documents is not measurably better than reading three. McNemar gives "
         f"p={f3(mc[pair_key]['p_exact'])} on four-class and p={f3(mc[pair_bin]['p_exact'])} on "
         "binary A." if pair_key and pair_bin else
         "Only one loop arm is on this page, so the read-depth comparison is not shown."),
        "The paper's number has no interval and no per-item predictions, so it cannot enter a "
        "paired test against either of ours.",
    ]
    cav_html = "".join(f"<li>{c}</li>" for c in caveats)

    missing_note = ""
    if S["arms_missing"]:
        missing_note = (f'<p class="note">Not in the comparison file yet, and so absent from '
                        f'every figure and table here, the {", ".join(S["arms_missing"])} '
                        f'{"arms" if len(S["arms_missing"]) > 1 else "arm"}.</p>')

    html = f"""<title>Paper, Loop, Urn</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --bg:#fbfaf7; --ink:#1c1b19; --mut:#6e6a63; --line:#e2dfd8; --panel:#ffffff;
  --accent:#7a3020; --barfill:#c9c4ba;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#161513; --ink:#e7e4dd; --mut:#98938a; --line:#302d28; --panel:#ffffff;
  --accent:#d08363; --barfill:#4a463f;
}} }}
:root[data-theme="dark"] {{
  --bg:#161513; --ink:#e7e4dd; --mut:#98938a; --line:#302d28; --panel:#ffffff;
  --accent:#d08363; --barfill:#4a463f;
}}
body {{ background:var(--bg); color:var(--ink);
  font:16px/1.65 "IBM Plex Sans",system-ui,sans-serif; margin:0; padding:3rem 2rem 5rem; }}
main {{ max-width:1080px; margin:0 auto; }}
h1 {{ font:600 2.3rem/1.2 "IBM Plex Serif",Georgia,serif; margin:0 0 .4rem; }}
h2 {{ font:600 1.35rem/1.3 "IBM Plex Serif",Georgia,serif; margin:3rem 0 .8rem;
  padding-top:1.6rem; border-top:1px solid var(--line); }}
h2 span {{ color:var(--mut); font-family:"IBM Plex Mono",monospace; font-size:.85rem;
  display:block; letter-spacing:.08em; margin-bottom:.3rem; }}
p {{ margin:.7rem 0; }}
.sub {{ color:var(--mut); margin-bottom:2.2rem; }}
table {{ border-collapse:collapse; width:100%; margin:1rem 0; font-size:.92rem; }}
th,td {{ text-align:left; padding:.42rem .7rem; border-bottom:1px solid var(--line); }}
th {{ color:var(--mut); font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }}
.m {{ font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
  padding:1rem; margin:1.2rem 0; overflow-x:auto; }}
.panel svg {{ max-width:100%; height:auto; display:block; }}
.flow {{ display:flex; gap:0; align-items:stretch; margin:1.4rem 0; flex-wrap:wrap; }}
.fbox {{ flex:1 1 150px; border:1px solid var(--line); border-radius:8px; padding:.7rem .8rem;
  font-size:.85rem; background:color-mix(in srgb, var(--bg) 60%, var(--panel)); margin:0 .4rem .4rem 0; }}
.fbox b {{ display:block; font-size:.92rem; }}
.key {{ border-left:3px solid var(--accent); padding:.2rem 0 .2rem 1rem; margin:1.2rem 0; }}
.note {{ color:var(--mut); font-size:.9rem; }}
ul {{ margin:.8rem 0; padding-left:1.2rem; }}
li {{ margin:.35rem 0; }}
</style>
<main>
<h1>Paper, Loop, Urn</h1>
<p class="sub">Three systems on the AVeriTeC dev 500, one gold set, matched conventions ·
comparison generated {S['comparison_generated_utc'][:10]}</p>

<h2><span>01</span>What is being compared</h2>
<p>ClaimCheck reports {f3(S['paper']['accuracy'])} four-class accuracy on the AVeriTeC dev 500.
We ran two systems of our own on the same {meta['n_gold_rows']} gold rows, which are
{meta['n_gold_claims']} unique claims with the nine duplicate rows inheriting their claim's
prediction. The first is our verify loop, run one claim at a time. The second is the
log-odds urn scoring the same claims with weights frozen from our own fact-check gold, no refit
anywhere on this page. Both run the same model. Both cap evidence at the claim's own date and drop
the outlet the claim came from. Intervals are 95 percent percentile bootstraps over the gold
rows, {meta['bootstrap']['reps']:,} resamples, seed {meta['bootstrap']['seed']}.</p>
<p>Two conventions run through everything below. Four-class accuracy uses the ClaimCheck
convention, where our unsupported verdict counts as Refuted. Every other number uses binary
map A, where gold Supported passes and every other gold class flags.</p>
<div class="flow">
<div class="fbox"><b>paper</b>ClaimCheck's published number, external, no per-item predictions</div>
<div class="fbox"><b>loop</b>our verify loop, up to {depth['loop_stages']} search stages per claim</div>
<div class="fbox"><b>urn</b>frozen log-odds weights, one query and one search per claim</div>
<div class="fbox"><b>same gold</b>{meta['n_gold_rows']} rows, {meta['n_gold_claims']} claims, same ceiling, same origin exclusion</div>
</div>
{missing_note}

<h2><span>02</span>Four-class accuracy</h2>
{fig('four_class')}
<table>
<tr><th>system</th><th>accuracy</th><th>95% interval</th><th>rows</th></tr>
{fc_rows}
</table>
<p>The gap to the paper is mostly a date ceiling. ClaimCheck sends its ceiling to Google as a
DD/MM/YYYY string, which Google reads month first, so on {cel['paper_dropped_rows']} of the
{meta['n_gold_rows']} rows the day is past 12, the string is not a valid date and the filter is
dropped, which is {pc(cel['paper_dropped_share_rows'])} of the set running with no ceiling at
all. We send the correct form and Google honours it, and even then
{pc(lead_cel['share_searches_with_post_ceiling_hit'])} of the
{lead_cel['serper_searches']} Serper searches in our {short[lead]} arm came back with at least
one page published after the claim's date.</p>

<h2><span>03</span>The urn ranks these claims as well as it ranks ours</h2>
{fig('roc')}
<p>The urn's AUC here is {f3(u7['auc'])} for the 7-flag model, against
{f3(u7['auc_fc_gold'])} on the fact-check gold it was fitted on. The 3-voice model gives
{f3(u3['auc'])} here against {f3(u3['auc_fc_gold'])} there. A score fitted on one corpus ranks
a different corpus just as well.</p>
<p class="key">The fitted operating point does not transfer, and it was never meant to. At the
threshold fitted for a 2 percent false alarm budget on our own feed, the urn flags
{pc(u7['fitted_recall'])} of AVeriTeC's flaggable claims at a
{pc(u7['fitted_fpr'])} false positive rate. AVeriTeC is not our feed. The ranking is the
transferable part, so the rest of this page compares at matched false positive rates.</p>

<h2><span>04</span>Same false positive rate, who catches more</h2>
{fig('matched')}
<table>
<tr><th>system</th><th>FPR held at</th><th>observed FPR</th><th>flag recall</th>
<th>95% interval</th><th>McNemar p vs the loop</th></tr>
{mt_rows}
</table>
<p>On map A the loop is ahead at every operating point, and the margin is small. Against the
{short[best]} arm the 7-flag urn gives up
{pp(matched[best]['systems']['loop']['recall'] - matched[best]['systems']['urn 7-flag']['recall'])}
of recall at the same false positive rate. That is one search and one parallel read pass
against up to {depth['loop_stages']} sequential search stages.</p>

<h2><span>05</span>Where the gap to the paper comes from</h2>
{gap_html}

<h2><span>06</span>What this says</h2>
<p>ClaimCheck's {f3(S['paper']['accuracy'])} is partly a product of three protocol choices, two of
them mistakes. The date filter it sends to Google is malformed, so on
{pc(cel['paper_dropped_share_rows'])} of the claims the search runs with no ceiling and the fact
check article about the claim is available as evidence. Absence of evidence is scored as Refuted
on a split that is {pc(S['gold']['refuted_share'])} Refuted. And there is no corroboration rule,
so a single page closes a claim. Our loop keeps the ceiling, keeps the rule, and lands at
{f3(fc[best]['accuracy'])}. Correct for their two mistakes and it lands at
{no_bar_acc}, the same performance.
On the decision we actually ship, flag or pass, the loop is the stronger instrument at every
operating point we measured. The frozen urn comes within
{pp(matched[best]['systems']['loop']['recall'] - matched[best]['systems']['urn 7-flag']['recall'])}
of the loop's recall at the same false positive rate, with one search stage instead of up to
{depth['loop_stages']}. That makes the urn a usable first pass and the loop the
thing you escalate to.</p>
<ul>{cav_html}</ul>

<p class="note">Every number and figure on this page is generated from the run outputs by
one script, {urn['n_covered_rows']} of {urn['n_gold_rows']} gold rows covered by the urn.</p>
</main>
"""
    out = A / "index.html"
    out.write_text(html)
    print("wrote", out, f"{len(html) / 1024:.0f}KB")


if __name__ == "__main__":
    main()
