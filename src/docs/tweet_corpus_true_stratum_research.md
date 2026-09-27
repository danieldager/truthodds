# Tweet corpus: sourcing the TRUE stratum (and a general / trivially-true stratum)

Literature scan, 2026-08-21. Question: is "posted by AP/Reuters/AFP" defensible as a
`true` label for a veracity scorer, and what else could serve? Companion question: how has
prior work sourced ordinary-timeline ("general / trivially true") tweets?

Every number below is cited. Where no measurement exists, that is stated rather than
estimated silently.

---

## Findings

### Q1 — Measured factual error rates of newspapers and wire services

**Accuracy-audit tradition (source-recontact studies).** The dominant method surveys the
people quoted in a story and asks them to flag errors.

- Maier (2005), 4,800 news sources across **14 newspapers**: sources found errors in **61%
  of local news and feature stories**, "an inaccuracy rate among the highest reported in
  nearly seventy years of accuracy research."
  https://journals.sagepub.com/doi/abs/10.1177/107769900508200304
- Maier's later 10-metro-daily study (**~3,600 stories**): about half contained ≥1 factual
  error; **2,615 errors across 1,220 stories**. Restricting to "hard", purely objective
  errors (misquotes, names, ages, titles, dates, addresses) gives **~48%** of stories.
  https://slate.com/news-and-politics/2007/08/newspapers-make-lots-of-mistakes-and-publish-damn-few-corrections.html
- Cross-national replication (Porlezza/Maier/Russ-Mohl): factual inaccuracy in **60%** of
  Swiss, **48%** of US, **52%** of Italian stories reviewed.
  https://www.poynter.org/reporting-editing/2012/new-study-shows-how-newspaper-inaccuracies-transcend-journalism-cultures-national-borders/

**These numbers do not transfer to wire copy directly.** They are (a) source-*perceived*
errors, including disputes about emphasis; (b) local news and features, the genre most
dependent on unverifiable local detail; (c) counted per *story*, where a single misspelled
title flips the story to "inaccurate."

**Corrections rates are a floor, not a rate.**

- Fewer than **2%** of factually flawed articles were corrected at the dailies studied; no
  newspaper corrected more than **4.2%** of its flawed articles. Of 130 sources who asked
  for a correction, **4** corrections were published. NYT correction volume rose from ~1/day
  (1982) to ~9/day (2004). (Slate, above.)
- **AP specifically**: 149 corrections issued between Jan 20 and Feb 19 (~5/day), against
  AP's ~**2,000 stories/day** — i.e. published corrections on roughly **0.25%** of stories,
  "less than 1% of output." The same analysis found uncorrected versions of **more than
  half** of those 149 stories still live on downstream sites (US News, CBS, Chicago Tribune).
  https://www.poynter.org/fact-checking/2018/when-wire-services-make-mistakes-misinformation-spreads-quickly/

**Gap, stated explicitly:** I found **no published source-recontact accuracy audit of AP,
Reuters, or AFP wire copy**, and no per-article measured error rate for any wire service.
The only wire-specific quantity in the literature is corrections *volume* (above), which the
audit tradition says captures under 2% of the errors sources can identify. Combining the two
is arithmetic, not measurement — see Recommendation for how I use it.

### Q2 — Tweet-level and headline-level distortion

- **Headlines unsupported by their own article**: Silverman (Tow Center, 2015) analysed
  **1,660 articles**; **~13%** had headlines not backed up by the article body. Of 14
  outlets that covered one false rumour, only **5 (~35%)** ever published a follow-up saying
  it was false.
  https://www.cjr.org/tow_center_reports/craig_silverman_lies_damn_lies_viral_content.php
- **Caveats lost in transmission** (Sumner et al., BMJ 2014; 462 press releases + matched
  papers and news): **40%** of releases exaggerated advice, **33%** exaggerated causal
  claims, **36%** exaggerated animal→human inference. When the release exaggerated, news
  did too in **58% / 81% / 86%** of cases; when it did not, only **17% / 18% / 10%**.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4262123/
- Patro et al. (arXiv:1811.07853) report some agencies exaggerate close to **60%** of the
  health articles they publish. https://arxiv.org/pdf/1811.07853
- **Breaking-news Twitter is heavily unverified at posting time.** PHEME's
  journalist-curated threads: Germanwings **50.7%** rumours, Ottawa shooting **52.8%**,
  Sydney siege **42.8%**, Charlie Hebdo **22.0%**, Ferguson **24.8%**.
  https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078
- **CREDBANK** (Mitra & Gilbert, ICWSM 2015): 60M tweets, 1,049 events, 30 annotators each —
  roughly **24%** of events in the global tweet stream were not perceived as credible.
  https://faculty.washington.edu/tmitra/public/papers/credbank-twitter.pdf

**Gap, stated explicitly:** no study measures the factual error rate of *news outlets' own
tweets*. Practice reporting is qualitative: outlets generally do not issue corrections on
social accounts, and erroneous tweets are deleted rather than corrected — only the Toronto
Star and NYT were found to run a dedicated social-corrections page.
https://newslab.org/making-corrections-on-social-media/

### Q3 — Precedent for source-level labels, and the critiques

**Precedent.** NELA-GT (2018–2022) is the canonical source-labelled corpus: ~1.8M articles
from 519 sources in the 2020 edition, labelled at *source* level from MBFC's factuality
score as reliable / mixed / unreliable. The authors themselves "encourage researchers to use
and compare veracity labels from multiple resources... particularly important when testing
machine learning models." https://arxiv.org/abs/2003.08444 · https://arxiv.org/pdf/2102.04567

**Critique 1 — source labels ≈ coin flip at article level.** Park et al., WebSci '23:
source-level credibility labels match article-level labels only **51%** of the time; source
labels are decent proxies for *political alignment* but "very poor proxies — almost the same
as flipping a coin — for credibility."
https://dl.acm.org/doi/10.1145/3578503.3583617 · https://www.eurekalert.org/news-releases/989502

**Critique 2 — models memorise the source, not the content.** "Hidden Biases in Unreliable
News Detection Datasets" (EACL 2021): with overlapping sources across train/test, models
"achieve good performance by directly memorizing the site-label mapping"; a clean
non-overlapping source split costs **>10% accuracy for every model tested**. In the
companion write-up, models trained on **random labels** came within **2%** of models trained
on true labels for the site-level NELA data.
https://arxiv.org/abs/2104.10130 · https://www.amazon.science/blog/amazon-paper-exposes-bias-in-unreliable-news-datasets

**Critique 3 — the exact failure mode we would be creating.** The ISOT fake-news dataset
drew its *real* class from Reuters and its fake class from PolitiFact-flagged sites. The
dateline token `(Reuters)` appears in **21,256 articles with 99.96% skew toward the real
class**; models hit F1 > 0.99 by reading the token. Subsequent work strips source strings
precisely because "body texts beginning with Reuters [are] always real news."
https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/02/ISOT_Fake_News_Dataset_ReadMe.pdf
· https://arxiv.org/pdf/2508.02074

**Counter-precedent (claim-level).** Twitter15/16 and FakeNewsNet label at *claim* level
from Snopes / Emergent / PolitiFact / GossipCop, not by outlet: Twitter15 has 1,490 claims
(non-rumour / false / true / unverified), Twitter16 has 818.
https://arxiv.org/pdf/2111.03299

### Q4 — Rating agencies

- **NewsGuard**: nine apolitical criteria, 0–100 score, ≥60 = green. Criterion I is "**Does
  not repeatedly publish false content**" — note "repeatedly", not "never"; criterion III is
  "Regularly corrects or clarifies errors"; criterion V is "Avoids deceptive headlines."
  Ratings are *site-level*; the Nutrition Label gives failure examples, **not** an error
  rate. https://www.newsguardtech.com/ratings/rating-process-criteria/
  · https://library.alaska.gov/documents/webinars/dev/newsguard/poster.pdf
- **Agencies agree with each other, which does not make them article-level ground truth.**
  Lin et al. aggregate six expert rating sets over **11,520 domains** (NewsGuard 8,178,
  Lasser 4,767, MBFC 3,216, Ad Fontes 283, professional fact-checkers 60): Pearson **r =
  0.32–0.86**, Spearman **0.32–0.90**, first PC explains **68.21%** of variance. On the
  aggregate 0–1 scale **Reuters = 1.00** and **AFP = 0.95** (NYT 0.86, Chicago Tribune 0.87).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10500312/
- No agency publishes a measured per-article or per-post falsity rate for any outlet. A
  secondary source reports AP News at 95/100 on NewsGuard; I could not verify this against
  NewsGuard directly and would not cite it in a paper.

### Q5 — Sourcing a general / trivially-true stratum

- **Random-sample corpus**: Archive Team's **Twitter Stream Grab** on the Internet Archive —
  a ~1% ("spritzer") sample archived continuously since 2013, **>12B tweets**, ~4M
  tweets/day per daily file. Collection ran through ~mid-2024.
  https://archive.org/details/twitterstream ·
  https://atcoordinates.info/2023/04/30/parsing-the-internet-archives-twitter-stream-grab-with-python/
- **Is the 1% sample representative?** Morstatter et al. (ICWSM 2013): the Streaming API
  sample performs *worse than a random Firehose sample* when coverage of the target set is
  low, but topical analyses converge when coverage is high. For an unfiltered sample of
  ordinary tweets (coverage = everything) this is the benign regime.
  https://arxiv.org/abs/1306.5204
- **Composition of an ordinary timeline** — only one published breakdown exists, and it is
  old and crudely coded: Pear Analytics (2009), 2,000 public-timeline English tweets —
  **40.55%** "pointless babble", **37.55%** conversational, **8.7%** pass-along, **5.85%**
  self-promotion, **3.75%** spam, **3.6%** news. Criticised at the time for its coding scheme.
  https://pearanalytics.com/wp-content/uploads/2012/12/Twitter-Study-August-2009.pdf ·
  https://www.zephoria.org/thoughts/archives/2009/08/16/twitter_pointle.html
- **How much of a feed is check-worthy at all**: ClaimHunter (Newtral, CEUR Vol-2877; 5,000
  tweets annotated by 3 fact-checkers) reports an empirical estimate that only **10–15% of a
  Twitter feed** is check-worthy. https://ceur-ws.org/Vol-2877/paper3.pdf
- **Topically filtered upper bounds**: CLEF CheckThat! 2022 Spanish politician tweets —
  **2,184 / 7,489 = 29.2%** check-worthy; CheckThat! 2020 COVID tweets — 231/672 ≈ 34%.
  https://ceur-ws.org/Vol-3180/paper-28.pdf
- **Our own prior**: the 862-post X capture gives **19% has-claim (cascade)** vs **43%
  (flat-pass)** — bracketing the ClaimHunter number depending on whether a scope gate runs
  first (`CLAUDE.md`, slides protocol).

---

## Recommendation

### Is outlet-identity-as-true defensible?

**As a weak/noisy training label, yes. As eval gold, no.** Three reasons, in order of force:

1. The only direct measurement of source→article label transfer puts agreement at **51%**
   for credibility (Park et al.). That figure is for a mixed reliable/unreliable pool, so it
   overstates the problem at the reliable end — but nothing in the literature rescues
   source labels as *gold*.
2. The ISOT `(Reuters)` result is a precise prediction of what our scorer will learn: wire
   tweets have a distinctive register (dateline conventions, `BREAKING:`, third person,
   always-a-link, no first-person opinion) that is perfectly anti-correlated with the
   Community-Notes register (informal, first-person, image-heavy, argumentative). A model
   can hit near-ceiling accuracy without representing veracity at all.
3. Clean-split evaluation costs **>10% accuracy** in the one study that measured it, meaning
   any headline number from an unsplit corpus is inflated by roughly that much.

### Estimated label-noise rate for a wire-tweet TRUE stratum

No one has measured this. Two defensible bounds, both flagged as extrapolation:

- **Floor**: AP publishes corrections on ~**0.25%** of stories (149/month vs 2,000/day).
- **Ceiling-ish**: audits find published corrections capture **<2%** of the factual errors
  sources can identify (Maier). Naively dividing gives an "any source-identifiable error"
  rate in the low double digits — but that population is local news/features and counts
  name/title/date slips, so it is a poor model for the single primary claim a scorer
  extracts from a wire tweet.

My working estimate, to be replaced by our own audit, not cited as literature:
**~1–3%** of wire tweets are *false in their primary claim*; **~5–15%** are "not cleanly true
as stated" once you include headline-body distortion (Silverman's **13%**), stale/superseded
claims, and hedged-claim-stated-flatly. The second number, not the first, is what will hurt
a scorer trained to output calibrated veracity.

### Audit size to bound it

Rule of three (Hanley & Lippman-Hand): with **zero** errors observed in n, the 95% upper
bound on the rate is **3/n**. https://en.wikipedia.org/wiki/Rule_of_three_(statistics)

| n | 95% upper bound if 0 errors found |
|---|---|
| 100 | 3.0% |
| 200 | 1.5% |
| 300 | 1.0% |

Since we expect nonzero errors, size for *estimation* instead: **n = 200** gives roughly
±3pp on a 5–10% rate (Wilson). **Recommendation: a 200-tweet human audit**, stratified by
outlet (AP/Reuters/AFP), by breaking vs non-breaking, and by topic, labelling each tweet on
two axes — (a) primary claim true/false, (b) cleanly-stated vs distorted/stale/hedged. Report
both rates; use (a) as the noise rate and (b) as an exclusion filter.

### Alternatives, ranked

1. **Claim-level TRUE verdicts from fact-checkers** — PolitiFact True/Mostly True, Snopes
   True, AFP Fact Check and Reuters Fact Check "true" rulings, ClaimReview corpus. Same
   provenance mechanism as the Community-Notes false stratum, symmetric, no outlet confound.
   Precedent: Twitter15/16, FakeNewsNet. **Best option.**
2. **Community Notes negatives** — posts where the proposed note was rated *not helpful*, or
   notes filed under "not misleading". Identical register and topic distribution to the false
   stratum, opposite label; kills the style shortcut outright. Caveat: **74%** of accurate
   notes about the 2024 US election were never shown, so "no visible note" is a weak signal —
   use the note-level ratings, not post-level visibility.
   https://cdt.org/insights/making-metas-community-notes-work-current-challenges-and-opportunities/
3. **Wire tweets, audited and de-marked** — keep the plan, but strip handles/URLs/dateline
   tokens, style-match against the false stratum, drop the audit-flagged distorted cases, and
   train with label smoothing set to the measured noise rate.
4. **Time-shifted objectively-verifiable tweets** — final scores, election results, earnings
   prints, official statistics. Near-zero label noise, cheap, but narrow register.
5. **Raw outlet identity, unaudited** — not defensible for any reported metric.

Practical combination: **1 + 2 for eval, 3 for training scale.**

### Building the general / trivially-true stratum

- **Source**: Internet Archive Twitter Stream Grab daily files (or our existing feed
  captures). This is the closest thing to a defensible random timeline sample and has
  Morstatter's representativeness result behind it in the high-coverage regime.
- **Expect**: only **10–15%** of sampled tweets to carry a check-worthy claim (ClaimHunter);
  our own cascade says 19%. Budget ~7–10× oversampling if the target is claim-bearing
  trivially-true content.
- **Stratify explicitly into three buckets**, since "trivially true" is doing two jobs:
  (a) **no claim** — conversational, subjective, promotional (~78% of an ordinary timeline
  by the Pear breakdown); (b) **checkable and trivially true** — verifiable from common
  knowledge or the immediate record; (c) **checkable, non-trivial**. Bucket (a) tests the
  scope gate; (b) tests calibration at the easy end; only (c) belongs beside the wire and
  Community-Notes strata.
- **Calibrate the router on 200–300 human-labelled tweets** before running the LLM over the
  sample, per the project's existing contextualise-before-eval rule.
- **Style-match check**: report scorer performance separately on a subset matched for
  length, hashtag/URL presence, and first-vs-third person across strata. If accuracy drops
  sharply on the matched subset, the headline number was register detection, exactly as in
  the ISOT case.
