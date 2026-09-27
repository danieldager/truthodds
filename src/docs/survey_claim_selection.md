# Survey claim selection — process, criteria, open decisions

Status 2026-08-19 (evening): criteria agreed with Daniel, funnel not yet built. Two
forks decided today (see Open decisions): selection is pipeline-dependent, and
shortlist labels wait for the post-read pipeline revamp — so this thread shares a
session with the Thread-2 verifier revamp. Future sessions working on the survey
start here.

## Goal

Select ~100 claims from the X-harvested pool for the survey experiment (the empirical
arm of the nudging study). The final experiment set (~40, exact count open) is
double-checked by a professional fact checker, who provides the human label and a
written explanation per claim. Labels for the wider 100 come from running candidates
through the FULL fact-checking pipeline (not truth odds alone); truth odds is only a
prefilter/routing signal, never the label.

## Funnel

1. **Pool.** E2 tweet-claim pool: 3,953 claims / 1,991 complete posts / 50 outlets
   (`eval/data/urn_runs/e2_tweets/`), plus any newer harvests.
2. **Hard screens (automatic, cheap).** Drop before anything expensive:
   - Time-stability: no claims whose truth status can drift between selection and
     survey field date; no deictic time references ("yesterday", "this week").
     Contextualization screens already exist for deixis.
   - Self-containedness: judgeable by a survey respondent from the claim text alone.
     No image/video-locus claims, no thread context required.
   - Checkability: already screened at extraction (tweet_screen_claims).
   - Dedup: one claim per near-duplicate cluster (the pool has the same story worded
     differently across outlets).
3. **Stratified shortlist (~250-300).** Stratify on the criteria below, oversampling
   so post-pipeline attrition still leaves a balanced 100.
4. **Full pipeline run on the shortlist.** Produces the machine label + evidence
   dossier per claim. These labels are machine labels; their error rate is estimated
   from the professional fact-checker pass on the subset.
5. **Select the 100.** Balanced draw from the labeled shortlist against the strata.
6. **Professional fact-checker pass** on the experiment set (~40 or more): verify the
   label, write the human explanation. Disagreements with the pipeline label are
   findings, not noise; log them.

## Stratification criteria

- **Veracity mix**: true vs false-or-misleading, from the pipeline label (then human
  label for the final set). Target ratio open (50/50 default until the survey design
  says otherwise).
- **Political lean**: left/right balance. Outlet lean from `divine_outlet_lean`;
  spot-check claim-level lean, since an outlet's lean is not every claim's lean.
- **Outlet type / reliability tier**: spread across NewsGuard tiers and outlet kinds
  (wire, partisan aggregator, local, international).
- **Topic caps**: no topic dominates (e.g. max ~15% per topic), so 100 claims are not
  40 vaccine claims.
- **Plausibility balance**: false claims that are believable, true claims that are
  surprising. Otherwise the nudge has nothing to do and the survey measures ceiling
  effects. Cheap LLM prior-plausibility score for stratification; not a label.
- **Coverage**: mix of widely-covered and thinly-covered claims (the R3 finding:
  evidence availability tracks coverage, and coverage is orthogonal to truth).

## Open decisions

- **DECIDED 2026-08-19 — selection is PIPELINE-DEPENDENT.** Strata include pipeline
  behavior (flagged vs passed by truth odds): the survey tests the nudge as deployed.
  Accepted consequence: pipeline errors shape the stimulus set and the survey partly
  measures the pipeline; log the machine-vs-human disagreements as findings.
- **DECIDED 2026-08-19 — labels wait for the post-read revamp.** The shortlist's
  machine labels come from the revamped part of the pipeline AFTER the truth-odds
  read step (Thread 2's verifier revamp: v6 Arm C never ran live; v7.5 is the last
  live loop and is NOT to be used for these labels). Sequencing: revamp first, then
  label. Funnel steps 1-3 (hard screens, strata assets, stratified shortlist) do not
  depend on the revamp and can run first.
- **Is 100 the right N?** Should come from a power analysis of the survey design with
  claims as random effects, not from roundness.
- **Final experiment-set size** (~40) and whether the professional check covers all
  100 or only the experiment set.
- **Label scale**: binary true/false vs graded (mixed/misleading as its own cell).

## Motivation notes

The expensive resources are the pipeline run (per-claim dollars) and the professional
fact checker (per-claim time). The funnel is ordered so both only ever see claims that
already survived the free screens, and the shortlist is oversampled so balance survives
attrition. Selection bias to keep in mind: every screen (self-containedness especially)
moves the stimulus set away from the organic tweet distribution; that is acceptable for
the survey but means survey performance numbers do not transfer to the pipeline eval.
