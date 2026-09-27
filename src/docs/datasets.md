# The three claim piles behind Truth Odds

Written 2026-09-08 for issue #27. Every count below was re-derived from the files on disk,
not copied from an earlier deck or artifact. Companion: `dataset_improvements.md` (critique
and queued fixes). Scripts live in `src/eval/scripts/build_eval/` unless stated.

| pile | what it is | role | frozen file | claims |
|---|---|---|---|---|
| fc-gold | fact-checker verdicts, true and false | labelled fit (method 1), transfer test for the two-urn fit | `populations/fc_gold.parquet` | 3,280 |
| CN false urn | X posts with a Community Note that says the claim is false | the FALSE side of the label-free fit (method 2) | `populations/cn_false.parquet` | 1,739 |
| timeline urn | claims from our own X home timelines, mostly true, false share eps measured | the TRUE side of the label-free fit | `populations/x_feed.parquet` | 1,970 |

Since 2026-09-08 each pile has **one** frozen version, written to `eval/data/populations/`
by `freeze_populations.py` (section 7). The older cuts — 4,035 / 3,699 / 3,274 / 3,211 on
fc-gold, 1,969 / 1,719 / 1,658 on CN, 1,999 / 1,894 on the timeline — are **retired**; they
survive below as history, and any number quoted against them predates the freeze.

All three piles go through the same evidence instrument: one generated query (query-v3),
Serper top 10 under a date ceiling, every document read with read-v5, giving one of seven
flags per document (5 4 3 2 1 X I). The fit is a Laplace-smoothed log rate ratio per flag
between the two sides, over documents. Nothing is logistic regression.

## 1. fc-gold

**Source.** Google Fact Check Tools API, full history per publisher, English and French,
harvested 2026-07-23 (`harvest_full.py`, 83,038 rows, 38 publisher sites). Rating strings
are mapped to veracity 1 to 5 by a rule table (`eval/harmonize.py`), with subtypes
clear_true 5, mostly_true 4, mixed 3, unprovable 3, mostly_false 2, clear_false 1,
altered_media 1, satire 1. The LLM fallback for unmapped strings was never run, so 9,589
rows have no veracity. Duke Fact Check Insights (dump of 2025-09-08) adds columns only,
no rows.

**Publisher ruling** (Daniel, 2026-07-23, `eval/data/publisher_audits.md`). Eleven sites
admitted: AFP factcheck and factuel, Full Fact, AAP, Africa Check, FactCheck.org, Boom,
VERA Files, GhanaFact, PolitiFact, Snopes. Excluded: Lead Stories, Factly, Vishvas News,
Alt News, Rumor Scanner, Défacto, and every unaudited site under 100 rows. The audits found
selection-side bias only, no verdict-side bias. AFP, Snopes and PolitiFact are 68% of the
admitted corpus.

**Funnel.**

| stage | rule | script | out | nature |
|---|---|---|---|---|
| harvest | GFC full history, 38 sites, en+fr | `harvest_full.py` | 83,038 | API state |
| harmonize | rating string to veracity, subtype, judged_axis | `harmonize_full.py` | 83,038 (73,449 rated) | deterministic |
| admit | 11 sites and (review_date ≥ 2020 or claim_date ≥ 2020) | `build_fc_gold_selection.py` | 47,111 | deterministic |
| cluster | char-trigram Jaccard ≥ 0.60, union-find | `fc_claim_cluster.py` | 46,195 clusters | deterministic |
| enrich | FCI join on (review_url, claim_text) | `enrich_fc_gold_fci.py` | `fc_gold_v3.parquet` 47,111 × 30 | deterministic |
| claim screen | drop null veracity, altered_media, satire, "Says" prefix, scam; one row per cluster; LLM 7-class screen on a seed-303 subset | `claim_screen.py` | 20,630 screened | LLM Qwen3-235B, prompt screen-v1 |
| CAL / VAL | drop mundane and media_authenticity claim type; VAL 1,000 T / 1,500 F / 400 M year-stratified; CAL the rest of T plus 5,000 F plus the rest of M; cluster-disjoint | `build_calval_splits.py` seed 404 | CAL 9,406 (T 1,557 / F 5,000 / M 2,849), VAL 2,900 | seeded |
| E1 draw | all CAL trues plus 2,000 F plus 600 M, uniform within class | `contextualize_claims.py` seed 505 | 4,157 | seeded |
| contextualize | claim date, claimant, 40-word context pulled from the fact-check article by an LLM behind a leak firewall | `contextualize_claims.py`, `extract_claim_dates.py` | 4,118 ok | LLM plus scrape |
| leak audit | regex, then LLM flag, then LLM triage | clog 2026-08-03 | 449 contexts withheld | LLM |
| hard gate | 93 hand rulings on media-locus and demonstrative claims (58 keep, 35 drop) plus unresolvable and needs-article without context | `e1_gate_decisions.json` | 4,120 runnable | human |
| evidence run | query-v3, Serper top 10, ceiling = claim date, read-v5 on every document, x.com excluded | `evidence_urn_run.py` seed 707 | `urn_runs/e1_ctx/results-00.jsonl` | LLM plus live web |

VAL has never been used in any published number. Every fc-gold number reported since
August is the seed-505 E1 draw scored once, cut by inclusion rules applied in code.

**Populations, all cuts of the same 4,157 scored rows. RETIRED 2026-09-08**, kept as
history; the live population is `populations/fc_gold.parquet`, n=3,280.

| n | rule relative to parent | where the rule lives |
|---|---|---|
| 4,035 | at least one document with a directional flag (122 dropped, 37 gate plus 85 empty retrieval) | `fit_urn.load` |
| 3,699 | minus 336 whose fact-check judged media authenticity (`judged_axis_llm`, derived from the verdict text by an LLM) | `fit_urn.load_headline` |
| 3,274 | minus 425 with rating subtype mixed (the fact-checker graded the framing, not the proposition) | `model_ladder.SET_ASIDE` |
| 3,211 | minus 63 flagged by the three-judge media-provenance purge, on by default since 2026-08-28 | `e1_ctx/media_provenance_exclusions.json` |

Until 2026-09-08 none of these populations was saved as a file: each existed as a filter
inside a script plus two exclusion JSONs and two environment variables. They are now one
frozen parquet, and the zero-doc rule that shaped the 4,035 is gone (pad-to-ten, section 7).

**Labels.** At fit time only veracity 1 and 2 (false) against 4 and 5 (true) enter the
rates. At eval time y = 1 if veracity ≥ 4, so veracity 3 scores as false (Daniel,
2026-08-19). On the pinned cut the mixed subtype is gone but 118 unprovable rows stay and
score as false.

**Audits.**

| audit | n | who | result | applied |
|---|---|---|---|---|
| publisher reliability | 16 publishers | 16 web-research subagents | 8 admit, 8 admit with caveats | yes, 6 excluded |
| claim quality, four slices | 530 | LLM agents | deixis 21%, up to 49% on AFP, true-side mundane 23% | partly, remaps and the mundane screen |
| context leak | 4,116 | regex plus LLM | 449 withheld | yes |
| media and demonstrative gate | 93 | Daniel | 58 keep, 35 drop | yes |
| full claim screen | 4,120 | LLM, verdict-blind | 169 non-ok (4.1%) | no, tag only |
| judged_axis | 4,157 | LLM, title-aware | 374 media_authenticity | yes, the 336 exclusion |
| read quality | 239 reads | human and agent | 6.7% disagree, silence read as refutation in 11 | no |
| media-provenance purge | 1,768 false rows | 3 LLM judges, majority, Fleiss κ 0.62 | 63 purged (3.6%) | yes, default on |

The true half of the purge (1,502 rows) was never run.

## 2. Community Notes false urn

**Source.** X Birdwatch public data, snapshot 2026-07-23, notes and note status history
only (`harvest_community_notes.py`). A note is kept if it is currently rated helpful or
locked helpful. That gives 248,731 notes on 216,818 posts. All but 2 are classed
misleading, so the dump has no true stratum. Posts were hydrated through fxtwitter,
text only, free. 29.3% of noted posts are gone.

**Funnel.**

| stage | rule | script | out | nature |
|---|---|---|---|---|
| harvest | CRH current or locked, no classification filter | `harvest_community_notes.py` | 248,731 notes | deterministic on the dump |
| dedup | language heuristic on the note, scam-ad boilerplate, one note per post | `cn_dedup.py` | 140,791 | deterministic |
| cluster | char-trigram Jaccard ≥ 0.58 | `cn_cluster.py` | 127,093 clusters | deterministic |
| hydrate | cluster reps through fxtwitter | `cn_hydrate.py` seed 42 | 88,038 with text | API state |
| eligible | reps only, drop media-only notes (5,243), satire-only (1,532), posts under 30 days old (4,271), keep post lang en | `cn_false_stratum.select` | 60,196 | deterministic |
| draw | simple random, minus 200 dev posts, 2,000 then 4,500 | `cn_false_urn.draw` seeds 20260826, 20260827 | 6,500 posts, 4,257 authors | seeded |
| extract, normalize | same chain as the timeline, DeepSeek-V4-Flash, text only, user voice | `extract_tweet_claims.py`, `normalize_tweet_claims.py` | 12,660 claims | LLM |
| note-target match | LLM picks the one extracted claim the note targets, null if the note disputes media or framing or a claim we missed | `c2_audit.MATCH_SYS`, DeepSeek-V4-Flash temp 0 | 3,936 posts matched (60.6%) | LLM |
| falsity gate | LLM bands the matched claim against the note as false, partly, or untouched. Missing context is always partly | `c2_prompt_ab.GATE_SYS` | 2,857 false (413 partly, 666 untouched) | LLM |
| fit screens | checkworthy, not attribution form, not media locus, not a duplicate, not in fc-gold | `cn_false_stratum.screen` | 2,089 | deterministic |
| evidence run | identical `run_claim`, ceiling = post date from the snowflake id, x.com excluded | `cn_false_urn.reads` | 2,086 scored | LLM plus live web |
| residue | 34 hand-flagged (31 unresolved deixis, 3 media residue) | `fit_exclusions.json` | 2,052 | manual |
| zero-doc | at least one directional flag | `fit_two_urn.load_urn` | 1,969 | deterministic |
| media purge | three-judge majority, note disputes media not text | `c2_false/media_provenance_exclusions.json` | 1,719 | LLM, default on |

The note text reaches only the matcher and the gate. It never enters extraction, query
generation, or reading. Notes with a provisional (unlocked) status were not excluded and
are 3.5% of the urn. Posts are old, median 526 days at snapshot, minimum 29.

**Label.** A helpful note says the post misleads. The false label lands on our extracted
claim through two LLM steps, match then gate. Of the 1,969, 241 (12.2%) carry a note that
ticks missing context only, so the note corrects framing rather than the proposition.

**Audits.**

| audit | n | who | result | applied |
|---|---|---|---|---|
| smoke and prompt A/B | 50 then 200 posts | LLM matcher and reads | recovery 62% to 66% with user voice | yes, user voice adopted, 200 posts held out of the draw |
| residue sweep | 2,089 | hand read | 34 dropped | yes |
| note-target audit | 120 plus 69 booster | 3 blind LLM judges, seed 20260828 | note targets the claim 97.5% [92.9, 99.1], stratified mislabel rate 2.8% [0.5, 5.2], 10.1% inside the missing-context class | reported only |
| media-provenance purge | 1,969 | 3 LLM judges, Fleiss κ 0.68 | 250 majority (12.7%), 150 unanimous | yes, default on |
| mixed-framing screen | 241 | deterministic on note boxes | matches the missing-context class | not applied |

## 3. Timeline urn

**Source.** Home timelines of two X accounts Daniel controls, captured with a
Zeeschuimer-derived extension into NDJSON (`tweet_corpus/general_pool/`). 4CAT is not in
this path. Twelve sessions, four on 2026-06-02 and 06-05 and eight on 2026-08-26, 6,245
unique posts. Two frames. `feed` is HomeTimeline under a logged-in account (For You and
Following are not distinguished). `search_june` is every June capture, which is mostly
search results but also includes 204 home-timeline claims, 15 opened threads and 7 Explore
claims. June was kept for age spread, the feed half is hours old at read time. Ads, posts
under 16 characters and languages other than en and fr are dropped before the gate.

**Funnel.**

| stage | rule | script | out | nature |
|---|---|---|---|---|
| land | first id wins across the 12 files | `ingest_timeline_capture.land` | 6,245 posts | deterministic |
| prefilter | lang en or fr, over 15 chars, not promoted | same | feed pool 3,314 | deterministic |
| gate | public-affairs assertion with stakes, rejects personal, joke, promo, question, anecdote, rhetoric | `general_pool_screen` strict, DeepSeek-V4-Flash | 852 feed posts | LLM |
| extract, normalize | recall-first extraction then an audit pass assigning one category per claim | same chain as CN | 3,910 claims on 1,084 posts, 2,699 checkworthy | LLM |
| urn | concat the three voice files, stamp the frame | `timeline_urn.parquet` | 3,910 | deterministic |
| parity set | checkworthy, type assertion, lang en | `timeline_urn_run.parity` | 2,077 (feed 1,674, june 403, 779 posts, 466 handles) | deterministic |
| evidence run | same `run_claim`, x.com excluded, no ceiling needed | `timeline_urn_run.py --reads` | 1,999 with a directional flag | LLM plus live web |

No dedup and no residue screen is applied on this side. The residue screen was run (2,016
of 2,077 ok) and kept as a sensitivity cut only. The CN side drops media-locus and
duplicate claims, this side does not, so parity between the two urns is a matched set of
rules, not the same code.

**The false share eps.** 400 claims drawn uniformly from the 1,999, seed 20260828, in two
batches of 40 and 360. Packets carry claim and post text, handle, date, quote and media
flags only. A code assertion fails the build if any score field leaks in. Labellers were
Claude Sonnet agents with web search, not humans. Pass one labels all 400 true, false or
unverifiable. Pass two re-reads every false and 40 random trues. A deeper pass re-reads
every unverifiable. Raw 30 false of 349 decided gives 0.086. Folding the deep pass in,
38 of 372 gives 0.1022, 95% interval [0.0674, 0.1408] from a bootstrap over the 247 posts.
The earlier 9.6% figure counted synthesis falses in the lowest-scoring fifth only and is a
lower bound. eps falls monotonically with the score, Cochran-Armitage z = −5.38 on the
gold-fitted cut. The 333 claims no reader ever called false and at least one called true
form the hard-true corpus.

**Audits.**

| audit | n | who | result | applied |
|---|---|---|---|---|
| eps blind audit | 400 | Sonnet agents, web search | eps 0.1022 [0.0674, 0.1408] | yes, in the de-mix |
| band audit | 400 lowest by score | DeepSeek-V4-Flash synthesis over the same dossiers the score used | P(false given flag) 0.94 at the 2% threshold, floor 0.80, n=101 | reported only |
| coverage audit | 1,222 posts | DeepSeek-V4-Flash, blind | 12.4% of served posts are silent misses at the gate | no |
| residue screen | 2,077 | LLM | 97.1% ok | no, sensitivity only |

## 4. Independence of the three piles

Near-duplicate claims across piles at token Jaccard ≥ 0.6: fc-gold and CN 3 of 4,157,
fc-gold and timeline 1, CN and timeline 0. Shared evidence URLs: 1.2% of fc-gold documents
appear in the CN run and 2.3% of CN documents in fc-gold. All three retrieve through the
same Serper path and a shared query-keyed cache, so retrieval policy is common, claims are
not.

## 5. Which number sits on which population

**Current, on the frozen populations** (`populations/refit_results.json`, built
2026-09-08: folds cluster-disjoint on `cluster_id`, threshold nested inside the folds,
2,000-rep clustered bootstrap, two-urn at eps 0.10).

| number | fit | test | n |
|---|---|---|---|
| AUC 0.8480, recall 31.0% at FPR 1.88% | 3 voices, fc-gold | same | 3,280 |
| 0.8559, 40.0% at 2.14% | 7 flags, fc-gold | same | 3,280 |
| 0.8592, 41.8% at 2.01% | 28 cells, fc-gold | same | 3,280 |
| 0.8507, 31.0% at 1.88% | two urns, 3 voices, eps 0.10 | fc-gold | fit 1,739 F / 1,970 T, test 3,280 |
| 0.8453, 39.5% at 2.07% | two urns, 7 flags, eps 0.10 | fc-gold | fit 1,739 F / 1,970 T, test 3,280 |

The mixed subtype is now excluded in the frozen population itself, so the ladder sets
nothing aside and every cell above sits on the same 3,280 rows. AUC intervals are
cluster-bootstrapped (design effects 1.01 to 1.03 on fc-gold, up to 1.95 on a single urn
flag weight); the transfer AUC still has no interval, the urn bootstrap covers the weights
only.

**Pre-freeze ledger, RETIRED 2026-09-08.** Every row below sits on a retired cut with
non-clustered folds and a threshold picked on the rows it is reported on. Kept as history.

| number | fit | test | protocol |
|---|---|---|---|
| AUC 0.825, recall 29.9% at 2% FPR | 3 voices, fc-gold 4,035 | same | 5-fold out of fold |
| 0.8287, 31.3% (shipped constants) | 3 voices, 3,699 | same | out of fold |
| 0.8331, 38.7% | 7 flags, 3,699 | same | out of fold |
| 0.8519, 34.0% and 0.8620, 42.2% | 3 voices and 7 flags, pinned 3,274 | same | out of fold |
| 0.8625, 42.25% | 7 flags, 3,211 | same | out of fold, not yet pinned |
| 0.8129, 37.6% | two urns at eps 0.10, CN 1,969 | fc-gold 3,699 | frozen transfer |
| 0.8158, 35.8% | two urns, CN 1,719 | fc-gold 3,699 | frozen transfer, current file |
| 0.7904, 22.5% and 0.7926, 24.2% | fc-gold 7 flags | CN 1,969 or 1,719 plus timeline 1,999, eps corrected | frozen transfer |
| 0.868 | 7 flags, pinned 3,274 | AVeriTeC dev 500 | frozen transfer |
| P(false given flag) 0.94 | two urns at eps 0.05 | timeline bottom band, n=101 | synthesis on the same dossiers |

In that retired protocol, out-of-fold folds are assigned by a hash of the review URL, not by claim cluster. Every
bootstrap interval is a row bootstrap stratified by class, not clustered on post,
publisher or cluster. The threshold at each cell is the recall maximiser subject to FPR
≤ 2% on the pooled out-of-fold scores, chosen on the same rows it is reported on.

`urn_runs/e1_ctx/headline_metrics.json`, the file production code reads, holds the
SEVEN-FLAG fit on the frozen `fc_gold.parquet` since 2026-09-14 (n=3,280, AUC 0.8559,
nested recall@2%FPR 40.0%, flag boundary -4.0817), written by `graded_urn --ship` and
reproducible from the frozen population. Its `overall.weights` is keyed by FLAG, not by
voice. The three-voice n=3,699 fit it replaced is archived as
`headline_metrics_3voice_2026-09-14.json` and cannot be regenerated to its own numbers
(the gold media purge is now on by default and gives n=3,636); the three-voice fit on the
frozen population, which IS reproducible, is `headline_metrics_clustered.json`.

## 6. What can and cannot be rerun

Rerunnable from the repo given the files on disk: every deterministic stage, every seeded
draw (seeds 303, 404, 505, 707, 20260826, 20260827, 20260828, 42), and every population
cut. Not rerunnable: the GFC harvest (live index), the Birdwatch dump day (auto-selected,
not pinned, 2026-07-23 on disk only), the hydration (29% of posts gone), the two personal
timelines, every LLM stage (temperature 0, no seed, prompts are inline literals with no
version hash except query-v3, read-v5, screen-v1), every Serper retrieval, and the eps
labels (agent sessions, outputs kept, assignments not). All of `src/eval/data/` is
gitignored, so the corpora exist on one laptop.

## 7. Frozen populations (2026-09-08)

The cuts above are no longer in-script filters, and there is no longer a menu of them:
**one frozen version per pile**. `freeze_populations.py` writes the three to
`eval/data/populations/`, sorted by claim id, byte-identical on a re-run, with
`manifest.json` carrying the rule chain, row count, git rev and the sha256 of every
input and exclusion file. `build_exclusion_registry.py` writes
`eval/data/exclusions/registry.parquet`, 1,059 rulings from seven sources under one
schema (claim_id, corpus, rule, source_file, decided_on, decided_by, applied, note).
The four fit scripts take `--population` and read these files.

| file | n | rule chain |
|---|---|---|
| `fc_gold.parquet` | 3,280 | E1 draw 4,157, veracity 1 to 5, minus hand-gated 37, minus media-axis 346, minus mixed subtype 431, minus media purge 63 |
| `cn_false.parquet` | 1,739 | C2 + EXT 2,086 minus residue 34 minus media purge 250 minus provisional-note claims 63 |
| `x_feed.parquet` | 1,970 | parity set 2,077 minus residue screen 61 minus media locus 7 minus near-duplicate claim 39 |

**Pad to ten (Daniel, 2026-09-08).** The zero-doc rule is retired. A claim whose reads
returned fewer than ten documents — including none at all — is padded to ten slots with
silent documents (flag I, unrated, no domain, no evidence), which is what the retrieval
actually said. `fit_urn.PAD_TO = 10`, inherited by `graded_urn`, `model_ladder`,
`quality_urn` and `fit_two_urn`. That is why fc-gold keeps its 85 zero-doc claims, CN its
83 and the timeline its 78, and why the counts are not the old ones minus the same
exclusions.

Three chains go past what the default loaders did before, all on Daniel's 2026-09-08
ruling. fc-gold folds the media-axis, mixed-subtype and media-purge exclusions into the
population itself rather than into three separate loaders. The CN urn drops claims whose
note never locked (`tier != gold`), the 3.5% provisional share in block C. The timeline urn
applies the residue screen (a sensitivity cut until now) plus the two CN-side parity
screens the true side never had, media locus and near-duplicate claim, closing the block-D
item. Any number quoted against 4,035 / 3,699 / 3,274 / 3,211 / 1,969 / 1,719 / 1,894 /
1,999 predates the freeze.
