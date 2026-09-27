"""Render the pre-scorer coverage measurement as a standalone HTML page.

Primary pool is the TIMELINE (the deployment population); the community-noted acquitted pool
is carried as a comparison because it is the only place a human guarantees a claim exists.
Reads eval/data/coverage_audit_timeline/ and eval/data/coverage_audit/ and writes
coverage_funnel.html into the timeline dir. Every number comes from the audit modules — this
file lays them out and imports their `is_miss`/`decompose` rather than restating definitions.

  uv run python -m eval.scripts.build_eval.coverage_audit_html
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.scripts.build_eval.coverage_audit import wilson  # noqa: E402
from eval.scripts.build_eval import coverage_audit as CN  # noqa: E402
from eval.scripts.build_eval import coverage_audit_timeline as TL  # noqa: E402

TLD = SRC / "eval/data/coverage_audit_timeline"
CND = SRC / "eval/data/coverage_audit"

CSS = """
:root{
  --ground:#fbfaf8; --panel:#fff; --ink:#16161a; --mut:#6f6f79; --line:#e4e2dd;
  --keep:#23232a; --lost:#d6d3cc; --flag:#8c3a34; --flagbg:#8c3a341a;
}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
  --ground:#131315; --panel:#191920; --ink:#eceae6; --mut:#97959e; --line:#2c2c33;
  --keep:#eceae6; --lost:#3a3a42; --flag:#d98a80; --flagbg:#d98a8018;
}}
:root[data-theme=dark]{
  --ground:#131315; --panel:#191920; --ink:#eceae6; --mut:#97959e; --line:#2c2c33;
  --keep:#eceae6; --lost:#3a3a42; --flag:#d98a80; --flagbg:#d98a8018;
}
*{box-sizing:border-box}
body{
  margin:0; padding:4rem 1.5rem 7rem; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
  font-size:15.5px; line-height:1.65; -webkit-font-smoothing:antialiased;
}
main{max-width:64rem; margin:0 auto}
h1{font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; font-size:2.1rem;
  letter-spacing:-.015em; margin:0 0 .4rem; text-wrap:balance}
h2{font-family:"IBM Plex Serif",Georgia,serif; font-weight:600; font-size:1.15rem;
  margin:3.4rem 0 .2rem; text-wrap:balance}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.72rem;
  letter-spacing:.14em; text-transform:uppercase; color:var(--mut); margin:0 0 .5rem}
.deck{color:var(--mut); margin:0 0 3rem; max-width:47rem}
p{margin:.9rem 0}
.note{color:var(--mut); font-size:.93rem; max-width:53rem}
code{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.86em;
  background:var(--panel); border:1px solid var(--line); border-radius:3px; padding:.05em .35em}
.figure{border-top:2px solid var(--ink); border-bottom:1px solid var(--line);
  padding:1.4rem 0 1.6rem; display:grid; grid-template-columns:auto 1fr; gap:0 2.4rem;
  align-items:center}
.figure .big{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:4rem;
  font-weight:300; letter-spacing:-.04em; line-height:1}
.figure .cap b{display:block; font-weight:600; margin-bottom:.15rem}
.funnel{display:flex; flex-direction:column; gap:.15rem; margin:1.3rem 0 .4rem}
.row{display:grid; grid-template-columns:19rem 1fr 4.5rem 4rem 6.5rem; gap:.9rem;
  align-items:center; padding:.3rem 0}
.track{height:1.15rem; display:flex; border-radius:2px; overflow:hidden; background:var(--lost)}
.seg{height:100%; background:var(--keep)}
.n,.pct,.ci{font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums; text-align:right}
.n{font-size:.9rem}
.pct{font-weight:600; font-size:.95rem}
.ci{color:var(--mut); font-size:.78rem}
.wrap{overflow-x:auto; margin:1rem 0 0}
table{border-collapse:collapse; width:100%; font-size:.94rem}
th,td{text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--line)}
th{font-family:"IBM Plex Mono",ui-monospace,monospace; font-weight:400; font-size:.7rem;
  letter-spacing:.1em; text-transform:uppercase; color:var(--mut);
  border-bottom:1px solid var(--ink)}
td.n,th.n{text-align:right; font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums}
td.sub{color:var(--mut); font-size:.88rem}
tr.total td{border-top:1px solid var(--ink); border-bottom:0; font-weight:600}
tr.miss td:first-child{box-shadow:inset 3px 0 0 var(--flag)}
tr.miss{background:var(--flagbg)}
td.minibar{width:30%}
td.minibar span{display:block; height:.6rem; background:var(--keep); border-radius:1px}
.tag{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.68rem;
  letter-spacing:.08em; text-transform:uppercase; color:var(--mut)}
.tag.f{color:var(--flag)}
@media (max-width:760px){
  .row{grid-template-columns:1fr 4.5rem 4rem}
  .track,.ci{display:none}
  .figure{grid-template-columns:1fr; gap:.6rem}
  .figure .big{font-size:3.2rem}
}
"""


def main() -> None:
    tf = json.loads((TLD / "funnel.json").read_text())
    tb = [json.loads(l) for l in (TLD / "blind.jsonl").open()]
    N, sizes = tf["posts_served"], tf["strata"]
    cf = json.loads((CND / "funnel.json").read_text())
    cb = [json.loads(l) for l in (CND / "blind.jsonl").open()]

    def bars(f, total, rows):
        out = []
        for lab, key in rows:
            k = f[key]
            lo, hi = f["ci"][key]
            out.append(
                f'<div class="row"><div>{lab}</div>'
                f'<div class="track"><div class="seg" style="width:{k/total*100:.2f}%"></div></div>'
                f'<div class="n">{k:,}</div><div class="pct">{k/total*100:.1f}%</div>'
                f'<div class="ci">{lo*100:.1f}&ndash;{hi*100:.1f}</div></div>')
        return "\n".join(out)

    labels = {"prefilter": "dropped before screening",
              "screen_reject": "failed the STRICT qualify screen",
              "no_claims": "qualified, extraction emitted nothing",
              "not_eligible": "claims, none verify-eligible",
              "reached_verify": "reached verify &mdash; wrong-claim case"}
    aud, tot = [], 0.0
    for s in ("prefilter", "screen_reject", "no_claims", "not_eligible", "reached_verify"):
        sub = [b for b in tb if b["stratum"] == s]
        if not sub:
            continue
        hc = sum(1 for b in sub if b["blind_has_claim"])
        k = sum(1 for b in sub if TL.is_miss(b, s))
        lo, hi = wilson(k, len(sub))
        est = k / len(sub) * sizes[s]
        tot += est
        aud.append(
            f"<tr><td>{labels[s]}</td><td class=n>{sizes[s]:,}</td><td class=n>{len(sub)}</td>"
            f"<td class=n>{hc/len(sub)*100:.1f}%</td><td class=n><b>{k/len(sub)*100:.1f}%</b></td>"
            f"<td class=n>{lo*100:.1f}&ndash;{hi*100:.1f}</td><td class=n>{est:,.0f}</td></tr>")

    def weighted(field):
        w = collections.Counter()
        for s in sizes:
            sub = [b for b in tb if b["stratum"] == s]
            if not sub:
                continue
            per = sizes[s] / len(sub)
            for b in sub:
                if TL.is_miss(b, s):
                    w[b[field]] += per
        t = sum(w.values()) or 1
        return "\n".join(
            f'<tr><td>{k}</td><td class=n>{v:,.0f}</td><td class=n>{v/t*100:.1f}%</td>'
            f'<td class="minibar"><span style="width:{v/t*100:.1f}%"></span></td></tr>'
            for k, v in w.most_common())

    d = TL.decompose(tb, sizes)
    stages = [("language_policy", "language scope", "non-en/fr &mdash; 94% Spanish", False),
              ("scope_gate", "STRICT qualify screen was wrong",
               "in-scope claim, rejected anyway", True),
              ("topic_gate", "topic exclusion",
               "entertainment, sports, lifestyle", False),
              ("extracted_wrong_claim", "extracted the wrong claim",
               "claims emitted, not the one a reader sees", True),
              ("extraction_empty", "extraction emitted nothing", "no claim at all", True)]
    srows = "\n".join(
        f'<tr class="{"miss" if bad else ""}"><td>{lab}</td><td class=sub>{sub}</td>'
        f'<td class=n>{d.get(k,0):,.0f}</td><td class=n>{d.get(k,0)/N*100:.1f}%</td>'
        f'<td class="tag {"f" if bad else ""}">{"defect" if bad else "deliberate"}</td></tr>'
        for k, lab, sub, bad in stages)
    hard = d.get("extraction_empty", 0) + d.get("extracted_wrong_claim", 0)
    nonpol = tot - d.get("language_policy", 0) - d.get("topic_gate", 0)

    sr = [b for b in tb if b["stratum"] == "screen_reject"]
    cross = collections.Counter((b["blind_has_claim"], b["in_scope"]) for b in sr)
    crows = "\n".join(
        f'<tr class="{"miss" if (hc and sc) else ""}"><td>{"has a claim" if hc else "no claim"}</td>'
        f'<td>{"in scope" if sc else "out of scope"}</td><td class=n>{v}</td>'
        f'<td class=n>{v/len(sr)*100:.1f}%</td>'
        f'<td class=sub>{"the gate was wrong" if (hc and sc) else "correct rejection"}</td></tr>'
        for (hc, sc), v in sorted(cross.items(), key=lambda x: -x[1]))

    scr = "\n".join(f"<tr><td>{k}</td><td class=n>{v:,}</td>"
                    f"<td class=n>{v/sizes['screen_reject']*100:.1f}%</td></tr>"
                    for k, v in list(tf["screen_categories"].items())[:9])

    cn_tot = 0.0
    cn_sizes = {"no_claims": cf["loss"]["extraction_produced_nothing"],
                "not_eligible": cf["loss"]["claims_but_none_eligible"],
                "no_match": cf["loss"]["extracted_but_target_not_found"]}
    for s in cn_sizes:
        sub = [b for b in cb if b["stratum"] == s]
        if sub:
            cn_tot += sum(1 for b in sub if CN.is_miss(b, s)) / len(sub) * cn_sizes[s]
    cn_N = cf["posts_in"]

    e2e = tf["reaches_verify"] / N * 100
    html = f"""<title>Pre-Scorer Coverage</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Serif:wght@600&display=swap">
<style>{CSS}</style>
<main>
<p class="eyebrow">Measurement &middot; 28 August 2026 &middot; timeline pool</p>
<h1>What never reaches the scorer</h1>
<p class="deck">Every recall number we publish is computed over claims that survived to
scoring, so a claim we never extracted is a miss the evaluation cannot see. This measures
that hole against a post-level denominator on the population we deploy on: {N:,} posts an
account's For You feed actually served.</p>

<div class="figure">
  <div class="big">{e2e:.1f}%</div>
  <div class="cap"><b>of served posts reach the scorer at all</b>
  <span class="note">Everything we report about performance is conditioned on this slice.
  The blind audit below estimates that a further {tot/N*100:.1f}% carried a checkable,
  in-scope claim and were dropped anyway &mdash; a hole roughly three quarters the size of
  what we currently capture.</span></div>
</div>

<h2>Post-level funnel</h2>
<p class="note">Filled bar is what survives. Wilson intervals, 95%.</p>
<div class="funnel">
{bars(tf, N, [("posts served by the feed", "posts_served"),
              ("passed en/fr + length prefilter", "passed_prefilter"),
              ("passed STRICT qualify screen", "passed_screen"),
              ("yielded &ge;1 claim", "yielded_claim"),
              ("&ge;1 checkworthy claim", "has_checkworthy"),
              ("reaches verify", "reaches_verify")])}
</div>
<p class="note">The scope gate is the whole story. It removes 2,462 posts, 60.4% of
everything served; every stage after it is small by comparison &mdash; extraction emits
nothing on 48 posts and the eligibility gate takes 124. That is the opposite of the noted-post
pool, where extraction was the dominant loss, and the reason is simply that only screened-in
posts ever reach extraction here.</p>

<h2>Blind audit &mdash; how many drops are wrong?</h2>
<p class="note">The auditor sees the post text and nothing else: not our extraction, not
which stratum the post came from. It answers two <em>independent</em> questions &mdash; is
there a checkable factual claim, and does the post bear on public affairs or something with
real stakes. Both are needed, because the STRICT screen is a scope gate, not a claim-existence
gate: &ldquo;I paid $14 for a sandwich&rdquo; is perfectly checkable and correctly dropped. A
drop counts as wrong only when a post has a claim <em>and</em> is in scope. For posts that
reached verify, wrong means we extracted something other than what a blind reader saw.</p>
<div class="wrap"><table>
<tr><th>stratum</th><th class=n>posts</th><th class=n>audited</th><th class=n>has claim</th>
    <th class=n>+ in scope</th><th class=n>95% CI</th><th class=n>est. misses</th></tr>
{''.join(aud)}
<tr class="total"><td>total</td><td class=n>{N:,}</td><td class=n>{len(tb):,}</td>
    <td class=n></td><td class=n></td><td class=n></td><td class=n>{tot:,.0f}</td></tr>
</table></div>

<h2>Which stage dropped it</h2>
<div class="wrap"><table>
<tr><th>stage</th><th></th><th class=n>posts</th><th class=n>of served</th><th></th></tr>
{srows}
<tr class="total"><td>hole excluding deliberate scope policy</td>
    <td class=sub>what is arguably broken</td>
    <td class=n>{nonpol:,.0f}</td><td class=n>{nonpol/N*100:.1f}%</td><td></td></tr>
<tr><td>hole from extraction alone</td><td class=sub>nothing, or the wrong thing, emitted</td>
    <td class=n>{hard:,.0f}</td><td class=n>{hard/N*100:.1f}%</td><td></td></tr>
</table></div>
<p class="note">The prefilter's misses are almost entirely non-English &mdash; 94% Spanish in
the sample, zero among the en/fr posts it dropped &mdash; so that bucket is the language
scope working as designed, not a defect. Excluding both deliberate scopes, the arguable hole
is {nonpol/N*100:.1f}%, and it sits overwhelmingly in one place: the qualify screen.</p>

<h2>Is the qualify screen wrong, and how often?</h2>
<p class="note">The gate's own job, scored on the {len(sr)} rejected posts audited. Its error
rate is the one row where a claim exists and the post is in scope.</p>
<div class="wrap"><table>
<tr><th>blind verdict</th><th></th><th class=n>n</th><th class=n>share</th><th></th></tr>
{crows}
</table></div>
<p class="note">So the screen gets 89.4% of its rejections right. The 10.6% it gets wrong is
still the largest single bucket of loss in the pipeline, purely because it operates on so
many posts. What it rejects, by its own category:</p>
<div class="wrap"><table>
<tr><th>screen category</th><th class=n>posts</th><th class=n>share of rejects</th></tr>
{scr}
</table></div>

<h2>Misses by post type</h2>
<p class="note">Blind classification, weighted back to population &mdash; strata were sampled
at different rates.</p>
<div class="wrap"><table>
<tr><th>post kind</th><th class=n>posts</th><th class=n>share</th><th style="width:30%"></th></tr>
{weighted("post_kind")}
</table></div>

<h2>Where the claim lives</h2>
<div class="wrap"><table>
<tr><th>locus</th><th class=n>posts</th><th class=n>share</th><th style="width:30%"></th></tr>
{weighted("locus")}
</table></div>
<p class="note">Media-locus posts and quote attributions were the expected culprits and are
not: the claim sits in the post text in 99.6% of misses, and attributions are 8.2% of them.
Caveat &mdash; the auditor reads text only, so it could flag a media locus only when the text
pointed at one.</p>

<h2>Extraction itself is not the problem</h2>
<p class="note">Of the 350 audited posts that reached verify, 308 carried an in-scope
checkable claim by blind reading, and we recovered 290 of them &mdash; <b>94.2%</b>. Once a
post is in scope, the chain gets the right claim out of it nearly always. The loss is
concentrated in deciding which posts to look at, not in reading them.</p>

<h2>Comparison &mdash; the community-noted pool</h2>
<p class="note">The same instrument on {cn_N:,} noted posts from the L3 acquitted set, where
a human guarantees a contestable claim exists. That pool has no qualify screen, so it isolates
what extraction and the eligibility gate do on their own.</p>
<div class="wrap"><table>
<tr><th></th><th class=n>timeline</th><th class=n>community-noted</th></tr>
<tr><td>posts in</td><td class=n>{N:,}</td><td class=n>{cn_N:,}</td></tr>
<tr><td>yielded &ge;1 claim</td><td class=n>{tf['yielded_claim']/N*100:.1f}%</td>
    <td class=n>{cf['posts_with_claim']/cn_N*100:.1f}%</td></tr>
<tr><td>reaches verify</td><td class=n>{e2e:.1f}%</td>
    <td class=n>{cf['posts_reaching_verify']/cn_N*100:.1f}%</td></tr>
<tr><td>the specific noted claim recovered</td><td class=sub>no notes on a timeline</td>
    <td class=n>{cf['target_recovered_eligible']/cn_N*100:.1f}%</td></tr>
<tr><td>estimated silent misses</td><td class=n>{tot/N*100:.1f}%</td>
    <td class=n>{cn_tot/cn_N*100:.1f}%</td></tr>
</table></div>
<p class="note">The two pools fail differently. On the timeline the scope gate dominates and
extraction is nearly clean. On noted posts, which have already passed a human's judgement that
something contestable is there, extraction emits nothing for 38% of them. Neither number
transfers to the other population.</p>

<h2>What this is and is not</h2>
<p class="note">The timeline pool has no notes, so the blind auditor <em>is</em> the ground
truth here, in a way it is not on the noted pool where it only classifies posts a human already
flagged. Every miss figure above therefore has one model's judgement inside it, and the
scope question in particular is a judgement call that a human panel might draw differently.
The pool is one account's For You feed over a handful of capture sessions, so it carries that
feed's composition. Nothing in extraction or either gate was changed to produce these numbers
&mdash; this is a baseline on the shipped chain, measured before any tuning.</p>
</main>
"""
    p = TLD / "coverage_funnel.html"
    p.write_text(html)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
