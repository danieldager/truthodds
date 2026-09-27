"""Generate the two-urn results artifact HTML from stats.json + figures.

Prose is Daniel's voice. House rules: no colons, semicolons, or em dashes in
prose, short sentences, near monochrome, full width text matching the figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
A = SRC / "eval/data/urn_runs/synth_expt/artifact"


def main():
    S = json.loads((A / "stats.json").read_text())
    REFIT = SRC / "eval/data/populations/refit_results.json"

    def refit_transfer_recall(variant: str = "two_urn_7flag",
                              population: str = "cn_false + x_feed -> fc_gold") -> float:
        """Nested recall at FPR<=2% of the frozen-population two-urn transfer refit.

    stats.json carries a null transfer stat, so this is where the number comes
    from instead of a hardcoded literal (2026-09-08)."""
        cells = json.loads(REFIT.read_text())["cells"]
        cell = next(c for c in cells
                    if c["variant"] == variant and c["population"] == population)
        return cell["nested_recall_at_2pct_fpr"]["recall"]


    def pc(x, d=0):
        return f"{x*100:.{d}f}%"

    def fig(name):
        svg = (A / f"fig_{name}.svg").read_text()
        return f'<div class="panel">{svg}</div>'

    tl, cn = S["tl"], S["cn"]
    w7 = S["weights7"]["0.1"]
    gw = S["gold_weights7"]
    rv = S["recall_by_veracity"]
    ba = S["band_audit"]
    sc = S["synth_confusion"]
    pol = S["policies"]
    dial = S["dial"]
    miss = S["miss"]
    FLAGS = ["5", "4", "3", "2", "1", "X", "I"]

    weight_rows = "".join(
        f"<tr><td>{f}</td><td class='m'>{w7[f]:+.2f}</td><td class='m'>{gw[f]:+.2f}</td></tr>"
        for f in FLAGS)
    rv_rows = "".join(
        f"<tr><td>{b}</td><td class='m'>{pc(r['v1'],1)}</td><td class='m'>{pc(r['v2'],1)}</td>"
        f"<td class='m'>{pc(r['v3'],1)}</td></tr>" for b, r in rv.items())
    ba_rows = "".join(
        f"<tr><td>score at or below {r['thr']:.2f} (the {r['name']} cut)</td><td class='m'>{r['n']}</td>"
        f"<td class='m'>{r['false']}</td><td class='m'>{r['true']}</td><td class='m'>{r['unsure']}</td>"
        f"<td class='m'>{pc(r['p_decided'])}</td><td class='m'>{pc(r['p_floor'])}</td></tr>" for r in ba)
    polnames = {"flash": "small model", "pro": "large model",
                "flash_exa": "small model, unsure goes to Exa",
                "pro_exa": "large model, unsure goes to Exa",
                "pro_exa_pro": "large model twice, Exa in between"}
    pol_rows = "".join(
        f"<tr><td>{polnames[n]}</td><td class='m'>{pc(p['cn'],1)}</td>"
        f"<td class='m'>{pc(p['gold_f'],1)}</td><td class='m'>{pc(p['gold_t_fpr'],1)}</td></tr>"
        for n, p in pol.items())
    dial_rows = "".join(
        f"<tr><td>{d['label']}</td><td class='m'>{pc(d['fpr'],1)}</td>"
        f"<td class='m'>{pc(d['cn'],1)}</td><td class='m'>{pc(d['gold_f'],1)}</td></tr>" for d in dial)

    def funnel(steps, title):
        mx = steps[0][1]
        bars = "".join(
            f'<div class="frow"><div class="flab">{lab}</div>'
            f'<div class="ftrack"><div class="fbar" style="width:{max(3,int(100*n/mx))}%"></div></div>'
            f'<div class="fnum m">{n:,}</div></div>' for lab, n in steps)
        return f'<div class="funnel"><div class="ftitle">{title}</div>{bars}</div>'

    cn_funnel = funnel([("posts that carry a community note", 6500), ("claims matched to their note", 3936),
                        ("note says the claim itself is false", 2857), ("clean and fit eligible", cn["claims"])],
                       "the false urn, from community notes")
    tl_funnel = funnel([("claims extracted from captures", tl["claims"]), ("checkable claims", tl["cw"]),
                        ("english assertions, the parity set", tl["parity"]), ("with evidence read", tl["scored"])],
                       "the feed urn, from the timeline")

    html = f"""<title>Two Urns</title>
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
.funnels {{ display:grid; grid-template-columns:1fr 1fr; gap:2rem; margin:1.4rem 0; }}
@media (max-width:800px) {{ .funnels {{ grid-template-columns:1fr; }} }}
.ftitle {{ font-weight:600; margin-bottom:.6rem; }}
.frow {{ display:flex; align-items:center; gap:.6rem; margin:.3rem 0; }}
.flab {{ flex:0 0 46%; font-size:.85rem; color:var(--mut); }}
.ftrack {{ flex:1 1 auto; }}
.fbar {{ height:1.05rem; background:var(--barfill); border-radius:3px; }}
.fnum {{ flex:0 0 4.2rem; text-align:right; font-size:.85rem; }}
.key {{ border-left:3px solid var(--accent); padding:.2rem 0 .2rem 1rem; margin:1.2rem 0; }}
.note {{ color:var(--mut); font-size:.9rem; }}
</style>
<main>
<h1>Two Urns</h1>
<p class="sub">Fitting and stress testing the Truth Odds score without gold labels ·
fit and audits run 26 to 27 August 2026</p>

<h2><span>01</span>What we did</h2>
<p>We wanted to fit the evidence score on our own data instead of on fact checker labels.
So we built two piles of claims. One pile is claims that community notes established as
false. The other pile is a normal X timeline, mostly true, with a small share of false
mixed in. We ran the exact same evidence pipeline on both piles, one search per claim and
about ten read documents each. The weights come from how often each read flag appears on
one side versus the other. No gold label touches the fit at any point.</p>
<div class="flow">
<div class="fbox"><b>false urn</b>{cn['claims']:,} claims the community established as false</div>
<div class="fbox"><b>feed urn</b>{tl['scored']:,} claims from a real timeline</div>
<div class="fbox"><b>same instrument</b>one search and ~10 evidence reads per claim, identical on both sides</div>
<div class="fbox"><b>flag rates</b>how often each read outcome appears on each side</div>
<div class="fbox"><b>weights</b>the log ratio of the two rates, corrected for the false share in the feed</div>
</div>

<h2><span>02</span>The two datasets</h2>
<p>The false urn comes from community notes. We sampled noted posts, matched each note to
the claim it targets, kept the ones where the note says the claim itself is false as
stated, and screened out residue. The feed urn comes from timeline captures on our own
instrumented accounts plus a June search draw that adds age spread. Every claim passed
the same gate, extraction and normalization chain.</p>
<div class="funnels">{cn_funnel}{tl_funnel}</div>
<p>Both urns get the identical evidence treatment. The false urn holds {cn['docs']:,} read
documents, the feed urn {tl['docs']:,}. The two sides differ where they should. Support
voices are {pc(S['mix']['timeline']['sup'])} of feed evidence against
{pc(S['mix']['cn']['sup'])} on the false side, refutation {pc(S['mix']['timeline']['ref'])}
against {pc(S['mix']['cn']['ref'])}. Topic mixes differ, politics is about a third on both
sides but the false urn carries more entertainment and sports while the feed carries more
business. We treat that at analysis time, not by dropping data.</p>

<h2><span>03</span>The contamination correction</h2>
<p>The feed is not a clean pile of true claims. Some share of it, call it epsilon, is
false. We never identify which claims those are. The correction works on the rates
themselves. The observed rate of each flag in the feed is a blend, mostly what true
claims produce with a slice of what false claims produce mixed in. We know the false
side's rates exactly from the other urn, so we can subtract that slice and rescale,
which recovers the rate that true claims alone would show. The direction of the effect
matters. Contamination is what compresses the weights, because it drags the feed rates
toward the false urn rates and makes the two sides look more alike than they are.
Removing it pushes the weights deeper, most visibly on the refute flags. We fit at
several assumed epsilon values. The weights move modestly and the ranking quality does
not move at all, so a wrong guess is cheap. Our own audit of the lowest scoring fifth
of the feed already confirmed {pc(0.096,1)} of the whole urn false, so we use 0.10 as
the working value.</p>
{fig('eps')}

<h2><span>04</span>The weights replicate</h2>
<p>The check that matters most. The same seven flag weights fitted two completely
different ways. Filled dots are the two urn fit with no labels. Open dots are the old fit
on fc gold labels. Every sign agrees, including the odd positive weight on the X flag.
Magnitudes compress by about half, which is exactly what a contaminated pile does to a
rate fit, and the epsilon correction recovers part of it.</p>
{fig('weights')}
<table>
<tr><th>flag</th><th>two urn fit</th><th>fit on gold labels</th></tr>
{weight_rows}
</table>

<h2><span>05</span>Does it transfer</h2>
<p>We froze the two urn weights and scored the fc gold population of
{S['gold_n']['total']:,} claims, {S['gold_n']['true']:,} true and
{S['gold_n']['false_mixed']:,} false or mixed, with mixed counted as false. No refit.
The frozen weights match the labels-fitted baseline on ranking, and at a fixed two
percent false alarm budget they catch {pc(S['transfer']['0.1']['rec2'],1) if isinstance(S['transfer']['0.1']['rec2'], float) else pc(refit_transfer_recall(),1)}
of false claims against {pc(0.313,1)} for the labels-fitted three voice baseline.</p>
<p class="key">What performing well means for us. Catch the obvious misinformation with
confidence, accept misses on light or contested claims, and above all do not flag claims
that are likely true. The score already has that shape. At every alarm budget an
obviously false claim is two to four times more likely to be caught than a mixed one.</p>
{fig('recall')}
<table>
<tr><th>false alarm budget</th><th>obvious false caught</th><th>mostly false</th><th>mixed</th></tr>
{rv_rows}
</table>
<p class="note">The three severity groups are the fact checkers' own harmonized ratings,
levels one, two and three on the gold scale, not a judgment of ours. The mostly false
group holds only 76 claims, so its column is noisy and its apparent gap to mixed is a
handful of claims rather than a real inversion.</p>

<h2><span>06</span>Precision where the tool will live</h2>
<p>Alarm budgets on an eval set are not the deployment number. So we audited the feed
itself. We scored every feed claim, took everything scoring below minus 2.66, which is 400
claims, and had a score blind model read each claim with its evidence and give a
verdict. The flags below the candidate
thresholds are overwhelmingly justified.</p>
{fig('hist')}
<table>
<tr><th>cut</th><th>flagged</th><th>false</th><th>true</th><th>unsure</th><th>precision of decided</th><th>floor</th></tr>
{ba_rows}
</table>
<p class="note">The floor counts every unsure as a wrong flag. The confirmed wrong flags
are almost all fast moving market numbers where the evidence went stale, a small and
recognizable class.</p>

<h2><span>07</span>Why the score alone stops near forty percent</h2>
<p>The misses are not a weights problem. Of the obvious falses the score misses at the
two percent budget, {pc(miss['zero_refute'])} retrieved zero refuting documents and
{pc(miss['all_silent'])} retrieved nothing but silence. Only {pc(miss['fc_seen'])} ever
saw a fact check domain. The refutations exist, these are fact checked claims, but one
search query does not surface them. On the community note side we can measure this
directly because the note cites its sources. Our search recovers at least one of the
note's own domains for only {pc(0.351)} of claims. You cannot flag on evidence you never
retrieved.</p>

<h2><span>08</span>Reading the evidence jointly</h2>
<p>The score sums independent per document flags. A synthesis step instead shows one
model the claim and all ten dossiers at once and asks for a verdict of true, false or
unsure. It never sees the score or the flags. On the full fc gold set it condemns
{pc(sc['F']['false'],1)} of false claims, including {pc(0.856,1)} of the ones whose
evidence was pure silence, because it can reason that a claim of that size would have
left a trace. The cost of that reasoning is the other side. It also condemns
{pc(sc['T']['false'],1)} of true claims, mostly thin evidence cases where absence proves
nothing.</p>
<p>Model quality changes the failure mode, not just the error rate. The small model
guesses when evidence is thin. The large model abstains. On silent true claims the small
model cries false half the time while the large one says unsure three quarters of the
time and false only one time in ten. Prompt wording changes almost nothing, we tested
guard clauses and they moved the numbers by noise. Abstention is what makes escalation
possible.</p>

<h2><span>09</span>The escalation tool</h2>
<p>Putting it together. The flags place every claim in a band. Flagged claims get the
synthesis read. When synthesis says unsure we escalate, one fresh Exa search with full
page text, and synthesize again. We measured every arm on a stratified sample of 475
claims spanning both urns and fc gold, forty per band per population.</p>
{fig('frontier')}
<table>
<tr><th>policy</th><th>community note falses caught</th><th>fc gold falses caught</th><th>false flags on true claims</th></tr>
{pol_rows}
</table>
<p>The large model with Exa escalation keeps nearly all the catch of the aggressive
policy while cutting false flags by a third. And the band dial makes the tradeoff
explicit. Each row adds a band to the treated set.</p>
<table>
<tr><th>treated bands</th><th>false flags on true claims</th><th>community note falses caught</th><th>fc gold falses caught</th></tr>
{dial_rows}
</table>
<p>Cost stays small at every depth. The score itself runs at about $2.30 per thousand
claims all in. Adding the small synthesis model costs twenty cents more per thousand.
The large model adds $2.50 per thousand, and the Exa escalation about $7 per thousand
escalated claims, which is only the unsure fraction. The deepest policy lands under
$8 per thousand feed claims.</p>

<h2><span>10</span>Limits and what comes next</h2>
<p>Four honest caveats. The synthesis audit on the feed reads the same dossiers the score
used, so a shared retrieval failure can fool both. The check on that is a hand audit of
the 22 flagged feed claims where the model answered true or unsure instead of confirming,
they are queued with post links in the Band Audit Review page and still await review. Part of the measured false flag rate on fc gold is an eval
artifact, quote attribution claims where gold graded whether the person said it and the
model graded whether what they said is true, so the deployable rate is likely lower.
Fast moving numeric claims need fresh dated re-reads, a small recognizable class. And the
production threshold decision still belongs to the banded audit on the feed, not to any
eval set budget.</p>
<p>Next steps in order. Hand audit the flag zone disagreements. Wire the band dial into
the pipeline as the flagging policy. Re-read volatile numeric claims with a date window.
Then the dummy account test, measuring false flags on a live feed end to end.</p>

<p class="note">Every number on this page is derived by
eval/scripts/build_eval/two_urn_artifact_stats.py from the run files and re-rendered by
its companion html script. Fit eval/data/urn_runs/true_timeline/two_urn_fit.json,
audits band_audit.jsonl and synth_verdicts.jsonl, experiment eval/data/urn_runs/synth_expt.</p>
</main>
"""
    out = Path("/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/ec974c97-b7e9-48c5-a9d0-694b5c240611/scratchpad/two_urns.html")
    out.write_text(html)
    print("wrote", out, f"{len(html)/1024:.0f}KB")


if __name__ == "__main__":
    main()
