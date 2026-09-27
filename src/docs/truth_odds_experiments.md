# Truth Odds experiment program (Daniel's roadmap, 2026-07-24)

Validates Truth Odds end-to-end on fc-gold v2: fit the constants on a calibration
split, then test whether thresholded posterior odds make the right nudge decision on
a held-out validation split. Framework: `truth_odds.md`. Datasets:
`fc_gold_v2_admitted.parquet` (47,111 rows; urn-fit pools TRUE 4,139 / FALSE 29,203).

## 0 · Standing rules for this program

- **Prompt versioning**: every query-gen and READ prompt carries a version tag; run
  records store the tag; changes = new version, never in-place edits (rollback must
  always be possible).
- **Resumable, extendable runs**: records store enough state (claim, query history,
  full per-result reads) that a ROUND 2 arm (adversarial refutation query, IDEA-016)
  can later be tacked onto saved round-1 runs without re-running round 1.
- Smoke → cost projection → Daniel's go before every full run. Serper only; Exa never.

## 1 · Datasets — CAL and VAL

Two disjoint splits, disjoint at the CLAIM-CLUSTER level (cross-publisher dedup
FIRST: the same viral claim is checked by AFP and Snopes and BOOM; row-level splits
would leak claims across CAL/VAL). Subagents used freely for build + audit.

**CAL (calibration)** — optimizes urn-parameter precision:
- All label classes; maximize effective n per (class × claim-date-year × claim-type)
  cell; every row through the quality + harmonization audit; deduped; date-corrected.
- Threshold selection for the nudge experiment happens HERE (or nested) — never on VAL.

**VAL (validation)** — optimizes decision realism:
- Held out untouched until the single validation pass; sized for confident precision/
  recall on the trust-critical error (nudging true posts): ~1k trues → ±2% on a 5% FPR.
- Contested/NEE rows included but reported separately from the binary metric.

**Both**: balanced on claim dates (the temporal studies need even coverage), then
RECENT claims oversampled — the purest live-ish test of how the system performs on
fresh claims.

**Rulings (Daniel 2026-07-24)**: MUNDANE screen approved — one LLM pass emitting
{mundane, claim_type, topic} (taxonomies discovered from a 180-claim audit first,
abstract definitions only in the prompt — no dataset examples); NO claim-age
exclusion (the ceiling makes old claims self-consistent; prior years extend the
temporal axis); splits approved: VAL = 1,000 T + 1,500 F + ~400 contested/NEE
(reported separately), CAL = remaining trues + year-stratified ~5k falses +
contested/NEE profiling pool. Allocation math in clog/240726.

**Audit (subagents, before splitting)**:
1. Claim quality — self-contained checkable proposition? (GFC claimReviewed
   pathologies: fragments, "this video…" deixis, non-EN strays, headline-shaped rows).
2. Harmonization correctness IN CONTEXT (claim + rating + verdict → is the veracity
   right for THIS claim; complements the rating-string table QC of 2026-07-23).
3. Cross-publisher claim clustering (port cn_cluster: rare-token + citation blocking →
   idf-trigram Jaccard → union-find) + split integrity check.
4. Balance report per split (class × year × publisher × claim-type).

## 2 · READ paradigm change (v-next, versioned)

- **NG/reliability scores are classification labels only** — signal weights, never
  gates. No triage. No unrated skip.
- **Read all 10**: scrape → Jina fallback → if both fail, extract stance from the
  SNIPPET; snippet-sourced signals are TAGGED (figures both ways: as their stance
  class, and as a separate category — do they earn a separate weight?).
- **Fate taxonomy** (per result): directional signal (support/refute) · on-claim
  no-direction · ∅_offclaim (real content, not about the claim) · ∅_junk (wholly
  irrelevant/garbage). **Junk ⊂ no-evidence** (Daniel): junk counts inside ∅, but as
  its own sub-type — hypothesis: a junk slot signals MORE toward false than off-claim
  real content does (a claim whose query pulls garbage lives further from the
  documented world). Fit p̂_∅junk and p̂_∅offclaim separately; test the hypothesis
  empirically. Junk definition to converge on ~20 hand-labeled examples during the
  audit phase.
- **READ output = pointer-based** (Daniel 2026-07-24, converges with pools.py
  Phase-6 pointer-READ): docs sentence-ID'd and SAVED with the run; per-doc output
  `{direction: 5-state, support: [ids], neutral: [ids], against: [ids]}` — lists of
  CLAIM-RELEVANT sentence numbers only. Buys: exact evidence reconstruction from
  pointers (cheaper AND more auditable than quotes); **evidence MASS per voice**
  (sentence counts) as a new exploratory Truth Odds axis; two free code-side QC
  checks (pointer range validity; direction-vs-counts consistency — divergence
  auto-flagged, itself a framing signal). `direction` is the model's holistic call,
  never derived from counts.
- **Mirror flagging**: wire/mirror/voice-key duplicates among the 10 reads are
  FLAGGED, not just collapsed — mirror count becomes an independent variable
  (hypothesis to test: fewer mirrors = more independent voices = more likely true).
  GATE: validate the mirror/echo detector itself before the full run (hand-audit a
  sample of flagged/unflagged pairs; extend _PUB_FAMILIES/wire detection as needed).

## 3 · Date regime — simulate live retrieval

- **Ceiling at claim date** for CAL and VAL (review_date − per-publisher median
  claim→review lag when claim date missing — lag estimable from the ~24k dual-dated
  rows; claim-date coverage: FALSE 82% / TRUE 14%, the Snopes gap).
- **Leakage guards**: flag and COUNT post-ceiling results that slip through; and
  adjudicate legitimacy — a page whose content predates the claim but whose visible
  date was updated later (site redesigns, "updated" stamps) is LEGITIMATE evidence
  that naive guards wrongly exclude, and vice versa. Report leakage rate + a
  legitimacy-audited subsample; decide whether residual leakage invalidates.
- **FC-block ON** (every fc-gold claim has a fact-check by construction) + per-claim
  exclusion of the reviewing publisher's domain. Consequence: the fact-checker-tier
  signal weight is NOT fitted by this program (exogenous; Daniel: fine for now, may
  not need weights at all — later question).

## 4 · Experiments

- **E1 — Calibration run**: CAL, one query, one round, read-all-10, ceiling ON, save
  everything. Figures: claim profiles per gold class — the 6 groups (veracity 1 · 2 ·
  3-contested · 3-NEE/unsupported · 4 · 5), then other axes (year, publisher, claim
  type, topic). Fit urn parameters per class; check monotonic ordering with veracity
  (construct validity); also the 3-way collapse {1–2} {3+NEE} {4–5}.
- **E2 — Nudge validation**: VAL, same single-round pipeline; per-claim posterior
  odds from E1's constants; threshold sweep (chosen on CAL) → nudge decision.
  Metrics: nudge-precision/recall on false vs true, ROC/AUC, calibration plot
  (odds 3:1 right 75% of the time?). NUDGE POLICIES to compare (none committed):
  (a) binary TRUE vs FALSE+UNSUPPORTED, nudge the latter; (b) 3-way
  nudge/soft/pass; (c) others as they suggest themselves.
- **E3 — Temporal drift**: rerun E1 (or a subset) with NO date ceiling → how do
  evidence profiles change as the world catches up? Direct study of the temporal
  dimension (evidence-accumulation speed per class).
- **E4 — Serper stochasticity**: repeat round 1 with IDENTICAL queries (cache
  bypassed) → variance of the urn draw itself.
- **E5 — Query-gen stochasticity**: same claim, multiple query generations → how much
  do the constants depend on query phrasing (relates to the query-conditioned urns).
- **E6 — Round-2 adversarial tack-on** (deferred, design-supported): refutation-
  seeking query appended to saved E1 runs → fits the refute urn (IDEA-016) without
  re-running round 1.
- **E7 — Mirror-count study**: empirical relation between independent-voice count
  (post mirror-collapse) and gold label.

## 4b · Pre-READ processing observatory (Daniel 2026-07-25)

Principle: context is MAXIMAL by default (prep-v4 grows the anchor radius ±1, ±2, …
until the token budget saturates) and the right amount is LEARNED from data, then
rolled back if the data says so. Four instruments, cheapest first:

1. **Processing metadata per doc** (live in prep-v4): raw_chars → cleaned/doc_chars →
   kept, n_sents, windowed?, anchor count, final radius, coverage fraction — recorded
   in every result. Zero marginal cost; feeds everything below.
2. **Domain processing profiles**: aggregate (1) by domain for the top ~200 domains —
   extraction yield, junk/empty rates, sentence-length shapes, READ outcomes. Finds
   the domains where processing systematically fails (bot walls, PDF routing, weird
   markup) → targeted per-domain fixes instead of global knob-turning.
3. **Pointer-utilization analysis** (the rollback metric): READ returns sentence ids,
   so we can measure WHERE evidence actually lives — distance of cited sentences from
   the nearest anchor, share of evidence found at window EDGES (radius too small if
   high), share found in budget-padding sentences far from any anchor (maximal
   context earning its keep — or not). This turns the ±radius choice from an argument
   into a measurement: if evidence never shows up beyond radius 2, roll back and
   pocket the tokens.
4. **Selection-recall A/B** (subsample, ~100 long docs): READ on the FULL document
   (budget waived) vs prep-selected — the fraction of full-doc evidence sentences
   that prep also surfaced = the prep stage's recall, the one number that certifies
   the whole pre-READ pipeline.

Iteration: findings → prep-v5+ (versioned, stamped per record, one live version per
run — the standing instrument rule).

## 4c · Metadata registry for future analyses (brainstorm 2026-07-25, Daniel+)

Analysis → required metadata (✓ = already recorded; + = add as meta-v2, code-only,
before full E1):

- SERP: rank-decay of evidence ✓rank; page-2 marginal value +result_page;
  snippet-sufficiency counterfactual & snippet-vs-body divergence +retain snippet
  alongside scrape; SERP-quality-as-signal (junk/unreadable mass share) ✓derivable.
- Document: yield by doc type +url/structure heuristic type; fetch-path quality
  +trafilatura/jina/snippet detail; +language; +scrape latency; EVIDENCE FRESHNESS
  (per-source age vs claim date, per truth class — refutation-lag hypothesis)
  ✓dates partially, +normalized parse.
- READ internals: selector ROC — BM25-score-vs-cited overlap (continuous
  do-we-need-embeddings answer, no reference reads) +per-sentence BM25 scores;
  +per-read in/out/cached tokens, retries.
- Claim covariates: complexity-conditioned urns +claim tokens/entities/has-number/
  quote-shaped; query distillation quality +query-claim term divergence; +claimant,
  language.
- Independence: echo topology beyond voice-key +SimHash per doc text → distinct-voice
  counts, echo-adjusted urn params (E7 substrate). ✓voice_key, mirror.
- Ops: estimator calibration +per-claim wall, per-stage timings.
- Landscape (SHIPPED in pilot): anchor_vocab/anchor_sents, read_regions spans/chars/
  mass, mass_read_share, skipped_top_mass/skipped_tiles, coverage.

Composite candidates: claim-difficulty score (predict starvation); SERP-quality
signal in the Truth Odds alphabet; snippet-only production pipeline (train from E1
labels).

## 5 · Open problems

- **TRUE scarcity** (the binding constraint): 4,139 total, 85% Snopes; CAL+VAL
  consume most of the pool. Candidate relief (unresolved, Daniel aware): PolitiFact
  site scraper beyond GFC's index (its true-history is mostly pre-2020/undated in
  GFC), AVeriTeC supported as a second fitting source, CN corrective trues as a
  biased-but-informative robustness stratum, FullFact free-text "correct" tail (needs
  the LLM pass we skipped). None decided.
- Snopes concentration on the TRUE side → tri-source robustness comparison is
  mandatory reporting, not optional.
- Junk definition convergence; mirror-detector validation (both gated pre-run).
