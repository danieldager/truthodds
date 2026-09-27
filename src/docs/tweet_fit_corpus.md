# Tweet fit corpus — design (C1, Log-Odds Sprint)

2026-08-21. Design doc for the corpus that answers "what dataset do we fit the
weights on". No runs; every count below was measured on local data today ($0),
anything unmeasured is labeled. Companion research: `docs/tweet_corpus_true_stratum_research.md`.
Review artifact (Daniel rules on the five forks there via inline comments): see
`clog/210826.md` for the URL.

**Working position (main session, 2026-08-21):** the fit of record stays fc-gold
(E1) until the tweet-population class-conditional frequencies are measured; the
threshold gets recalibrated on tweets regardless; promotion to pooled or
pure-tweet weights only on measured divergence. This doc's job is to make that
measurement — and the promotion decision — buildable and rulable.

## Measured today ($0, local CN dump of 2026-07-23 + repo assets)

- Raw dump: 2,906,696 notes on 1,954,305 tweets; statuses: 2,709,138 NMR /
  263,311 CRH / 130,207 CRNH (noteStatusHistory).
- Gold tier (CRH, misleading-class): 140,791 notes = 140,791 tweets = 127,093
  claim clusters (120,527 singletons; max 275). 87,132 cluster representatives
  already have hydrated text (fxtwitter, free).
- **Media-only notes: 6,315 = 4.8% of gold tier** flag ONLY
  `misleadingManipulatedMedia` — the CN analog of the media-authenticity axis;
  must be excluded for a text-only reader.
- Note recency (gold): 2023: 21,020 · 2024: 41,979 · 2025: 35,071 · 2026: 31,019.
- **Pool A — "affirmed true" (NOT_MISLEADING note rated helpful): n = 2.**
  Confirmed dead, as ROADMAP said.
- **Pool B — "accused and acquitted": 55,318 tweets** carry ≥1 misleading-class
  note that was REJECTED (CRNH) and zero CRH notes of any kind. None hydrated
  yet; hydration is free (fxtwitter, measured 0.75 req/s → a 5k sample ≈ 2h).
  Rejected-note years: 2023: 7,740 · 2024: 20,133 · 2025: 18,153 · 2026: 13,933.
- Hydration store: 128,506 attempts, 88,038 with text.
- Wire/outlet pool: `outlet_tweets.parquet`, 36,726 posts, 50 outlets, NG-binned.
- Already-scored organic tweets: E2 run, 3,953 claims (no labels).

---

## Fork 1 — false stratum: which CN slice

**Options.** (a) Gold tier, cluster representatives only; (b) gold tier, all
tweets; (c) gold + provisional (17,571 more notes, status not locked).

**Evidence.** Cluster reps kill near-duplicate claims (4,462 clusters of size 2,
one of 275 — fitting on all tweets would let one viral claim contribute up to
275 correlated rows to the frequencies). 87,132 reps already have text, so (a)
needs zero new hydration. Provisional-tier notes can still flip status.

**Recommendation.** (a) Gold tier, cluster representatives, English, text
hydrated, **minus media-only notes (4.8%)** and minus tweets whose note flags
ONLY `misleadingSatire`. One row per cluster. Additional guards: drop tweets
whose claim survives in E1 (cross-corpus leakage — claim-text near-dup check
against fc-gold), and drop notes younger than 30 days (status can still churn
before lock).

**What would change it.** If media-only exclusion by checkbox proves unreliable
in the smoke audit (checkboxes are crowd-set), escalate to the LLM judged-axis
screen used on E1.

**Daniel's ruling: OPEN.**

## Fork 2 — true stratum

**Options.** (a) Wire tweets (AP/Reuters/AFP) as weak label + 200-tweet audit;
(b) Pool B "accused and acquitted" CN tweets + audit; (c) fc-gold TRUE claims
that originated as X posts (already run in E1 — $0); (d) mix of a+b+c.

**Evidence.** (a) Scale is easy (36,726-post pool in repo) but carries the
register confound (ISOT precedent: classifiers learn the wire register;
source-level labels agree with article-level only ~51% in the literature) and
an unmeasured noise rate (~1–3% extrapolated, no wire audit exists). (b) Now
measured REAL at n=55,318: same register/population as the false stratum, so it
kills the style shortcut — but CRNH means "note rejected", not "tweet true"
(notes get rejected for snark, opinion, or missing sources too), so its noise
rate NEEDS the audit; it is "acquitted", not "verified". (c) is free, already
scored, and verified-true — but it is fc-selected (contested-then-vindicated
claims), not typical true tweets. Pool A (affirmed-true) is dead at n=2.

**Recommendation.** (d): Pool B sample (primary, audited) + wire sample
(secondary, audited, reported separately so any register effect is visible) +
the E1 x-post TRUE stratum as a $0 bridge column in every comparison table. The
200-item audit runs on BOTH audited strata before the full run; if Pool B's
audited false rate exceeds ~10%, it demotes to a robustness stratum and wire
becomes primary.

**What would change it.** The audit. Also: if Pool B hydration shows heavy
deletion (accused tweets get deleted), the surviving sample is biased toward
survivors — report the hydration failure rate and treat >40% loss as a red flag.

**Daniel's ruling: OPEN.**

## Fork 3 — general / trivially-true stratum

**Options.** (a) Archive Twitter Stream Grab (1% random sample, free bulk
download, coverage through ~mid-2024); (b) own-harvest via the extension /
dummy account; (c) skip it for the FIT and use it only for threshold + flag-rate
context.

**Evidence.** Random tweets are 85–90% non-checkworthy (research brief;
our own cascade: 19–43% has-claim depending on gate), so the stratum mostly
exercises extraction and the ∅ channel, and it carries no labels — it cannot
enter the class-conditional fit at all without labeling. Its real uses: the
realized score distribution of an ordinary feed (threshold context), the flag
rate on background content, and the trivially-true tail ("water is wet") that
neither CN nor fc-gold contains.

**Recommendation.** (c) with a small (a): ~500 Stream-Grab posts through the
full pipeline as a DESCRIPTIVE stratum (score distribution, flag rate, ∅ rate)
— excluded from the weight fit by design, since it has no truth labels. The
dummy-account feed (D4) later supersedes it as the live version of the same
measurement.

**What would change it.** If Daniel wants trivially-true claims IN the fit,
they must be labeled — cheap LLM labeling on obviousness is defensible for the
"water is wet" tail only; anything contestable goes to the audit.

**Daniel's ruling: OPEN.**

## Fork 4 — comparability protocol (the decision device)

Non-negotiables (inherited, restated as the checklist):
1. Instrument VERBATIM: same extraction chain, `run_claim` imported unmodified
   from `evidence_urn_run` (the E2 pattern), query-v3 / read-v5, same screens
   (no-context/teaser exclusion as in E2).
2. Date ceiling = post date (x_date chain), hard-clamped; note text NEVER
   enters the pipeline anywhere; exclude x.com + CN mirror domains from
   retrieval results.
3. Same eval conventions: media-axis excluded, cluster-level folds (fold by
   cluster_id, not tweet, so near-dup claims never straddle folds).
4. **The decision device is one table:** per-flag class-conditional frequencies
   (7 flags) — E1 TRUE vs tweet-TRUE strata, E1 FALSE vs CN-FALSE stratum —
   with cluster-bootstrap 95% CIs. Everything downstream (Fork 5) reads off
   this table.

**Daniel's ruling: OPEN** (mostly a confirm).

## Fork 5 — promotion rule, pre-committed

**Options.** (a) Binary trigger (e.g. "≥2 of 7 FALSE-class flag frequencies
have non-overlapping CIs → refit"); (b) continuous pooling with the promotion
decision reduced to one fitted number.

**Recommendation: (b), empirical-Bayes pooling with E1 as pseudo-counts.**
For each class c ∈ {TRUE, FALSE} and flag k:

    p̂_k^c = (n_k^c(tweet) + K · p_k^c(E1)) / (N^c(tweet) + K)

K = prior strength in pseudo-documents. K = 0 is the pure tweet fit; K → ∞ is
staying on E1; Laplace's +1/+7 is this formula with a flat prior. Choose K on
a grid {0, 100, 300, 1000, 3000, 10000, ∞} by held-out TWEET folds
(cluster-disjoint), primary metric recall @ FPR ≤ 2% on the tweet corpus,
AUC secondary — committed BEFORE the run. Report the whole K-curve. This
replaces a cliff-edge refit/don't-refit call with a measurement, degrades
gracefully when the tweet corpus is small, and makes the eventual answer
("K=300 fits best") itself the evidence of how far the populations diverge.
Binary readout for collaborators: K* ≤ 300 → "tweet data dominates, refit
was warranted"; K* ≥ 3000 → "E1 transfers, refit unnecessary".

**Daniel's ruling: OPEN.**

---

## Sizing + cost (v1, inside the $25 envelope)

Rates: verification $0.00137/claim (E1 measured), extraction $0.00005/post,
hydration free (fxtwitter). Claims/post ~1.2 eligible (E2 measured).

| Stratum | Posts | ~Claims | Verification | Notes |
|---|---|---|---|---|
| CN false (Fork 1) | 2,000 reps | ~2,400 | $3.29 | text already hydrated |
| Pool B acquitted | 1,500 | ~1,800 | $2.47 | hydrate ~2h free, audit 100 |
| Wire true | 1,000 | ~1,200 | $1.64 | from outlet_tweets, audit 100 |
| Stream Grab descriptive | 500 | ~150 | $0.21 | not in fit |
| E1 x-post TRUE bridge | — | — | $0 | already run |
| Extraction + screens | 5,000 | — | ~$0.50 | incl. no-context screen |
| Smoke (25) + audits LLM-assist | | | ~$0.60 | |
| **Total v1** | | **~5,550** | **≈ $8.7** | envelope $25 → redo headroom |

## What this corpus CANNOT support (named limits)

- Pool B is "acquitted", not "verified true" — bounded by a 100–200 audit, not
  eliminated. If audit false-rate > ~10%, Pool B demotes (Fork 2).
- CN selection = viral + contested; the FALSE frequencies are for claims
  someone bothered to accuse. Quiet falsehoods are unrepresented.
- No prevalence/π estimate — the strata mix is by design (that was always D4's
  job via flag-precision, not this corpus's).
- No per-publisher reliability, no non-English, no media-locus claims.
- Temporal skew 2023–2026; deletion bias in Pool B measured but not removable.
- Wire stratum retains the register confound; it is reported separately so the
  confound is visible instead of hidden.

## Timeline vs sprint

- W2 (–Sep 3): Daniel's five rulings → Pool-B hydration (free, overnight) +
  audits (200 items) + extraction + screens + smoke ($<1, checkpoint).
- W3 (–Sep 10): full run (≈$8, run_ledger + checkpoint discipline), frequency
  table, K-curve, threshold recalibration → C3.

---

# TRUE-stratum draw — design (C2, 2026-08-24)

Extends Fork 2/3 above with measured funnels and the lessons of the 2026-08-24
FALSE-side audits (recovery 66%, as-stated gate, yield 0.50 clean claims/post —
clog/240826). Applies two rules decided since C1: **fit only on competently-
extracted claims** (note-target recovered + gated; unrecovered posts deferred,
not forced) and the **standing mirror-audit rule** (every audit covers the TRUE
strata too). $0 design — nothing here has been run. Daniel rules on forks
F-T1…F-T5 below.

## Target sizing

FALSE stratum lands ~1,000 gated claims (measured yield × 2,000 posts). TRUE
side targets **~1,000–1,200 fit-eligible claims** — approximate balance, split:

| sub-stratum | posts drawn | ~fit claims | role |
|---|---|---|---|
| (a) Pool B accused-and-acquitted | ~1,600 (hydrate ~2,800) | ~650 | PRIMARY |
| (b) wire (AP+Reuters, already held) | ~450 | ~350 (cap ≤35%) | secondary, reported separately |
| (c) trivially-true tail | 0 in fit (descriptive 500) | 0 | threshold/flag-rate context |
| E1 x-post TRUE bridge | — | — | $0 comparison column |

Balance is not sacred — the class-conditionals are fit per class, and EB
pooling (Fork 5) keeps thin cells honest — but the FPR-budget threshold
calibration runs on the TRUE side, so it should not be the thin side.

## Sub-stratum (a): accused-and-acquitted — definition ladder (measured)

"Note rejected" ≠ "tweet true": CRNH can mean snark, opinion, or missing
sources in the NOTE. The defensible move is to stack independent signals of
acquittal. Funnel from the 2026-07-23 raw dump ($0, reproduced today):

| rung | definition | tweets |
|---|---|---|
| L0 | C1 loose: ≥1 misleading-note CRNH, zero CRH of any kind | 55,318 |
| L1 | + zero PENDING misleading notes (no unresolved accusations, NMR=0) | 40,917 |
| L2 | + screens: not all-rejected-notes media-only (870) or satire-only (1,185), all rejected notes status-LOCKED, latest note ≥30 d old | 36,699 |
| L3 | + a NOT_MISLEADING defender note was filed on the tweet (any status) | 21,511 |
| L4 | + ≥2 independent rejected accusations | 1,402 |

**Recommendation (F-T1): draw from L3.** Every L3 tweet was accused, the
accusation was rated down, no accusation is pending, and at least one rater
went further and filed "not misleading". L4 is too thin as a primary
(keep as a robustness slice inside the draw). **Hydration: 0/36,699 currently
hydrated** — a ~2,800-post fxtwitter pass (~1 h, free) covers the draw with
deletion headroom; the FAILURE RATE is itself a measurement (C1 red flag:
>40% loss = survivor bias, report it either way; accused tweets get deleted).

**Label unit + mirror audit** (the FALSE-side machinery, sign-flipped):
extract with `--voice user` → match the REJECTED note's target claim (note
text offline-only, same matcher) → only the on-target matched claim enters
the TRUE candidate set ("acquitted" is a property of the accused claim, not
of the tweet; co-extracted claims carry no label and are dropped). Then an
as-stated gate variant: "if the rejected note WERE accurate, would this claim
be false?" — bands: acquitted-clean (fit-eligible) / note-off-target (drop:
never really accused) / still-contested (drop, count). Audit: instrument
verdict distribution on the matched set + hand-read of every flagged claim +
150-claim LLM-assisted random audit with Daniel reading the disagreements.
**Demotion rule (C1, kept): audited false-rate >10% → Pool B demotes to
robustness stratum and wire becomes primary.** Expected yield ~0.4–0.5
fit claims/post (FALSE-side measured 0.50; assume similar until smoked).

## Sub-stratum (b): wire — no new harvest needed

We already hold **AP 560 + Reuters 560 posts** (outlet_tweets.parquet,
Sep 2025–Jul 2026, ~98% EN; 376+414 text-only). No AFP handle in the pool; a
new AFP harvest is NOT proposed (validate-handles rule + zero need at this
size). Draw ~450 AP+Reuters posts (prefer text-only), outlet voice prompt,
no-context/teaser screens as in E2.

**Register confound guards** (ISOT precedent). One structural advantage over
ISOT: the fit never sees prose — weights are fit on retrieval flag mixes of
extracted, normalized claims. The residual confound is **coverage, not
register**: wire claims are exactly the claims the news web corroborates, so
support-rates can be high for reasons of venue, not truth. Guards, all cheap:
1. **Cap** wire at ≤35% of TRUE fit rows (F-T3); never report a TRUE-side
   number pooled over sub-strata without the Pool-B-only column beside it
   (Fork 4 table gains one column per sub-stratum).
2. **Register-blind check:** a claim-text-only classifier run twice — wire-TRUE
   vs CN-FALSE, and PoolB-TRUE vs CN-FALSE. If the wire pair separates far more
   easily than the Pool-B pair, the wire stratum carries a text shortcut; report
   both AUCs next to the fit.
3. **K-sensitivity:** the Fork-5 K-grid is run with wire in and out; if K*
   moves materially, the wire stratum is doing something other than adding
   TRUE mass.

**Label noise budget** (research doc): ~1–3% primary-claim false; ~5–15% "not
cleanly true as stated". Audit: 200 posts, two axes — (i) primary claim
true/false, (ii) cleanly-stated vs distorted/stale/hedged — LLM-assisted,
Daniel spot-reads; axis-(ii) failures are EXCLUDED (they are the headline-
distortion class, not scorer food), axis-(i) rate becomes the label-smoothing
/ caveat number.

## Sub-stratum (c): general / trivially-true — descriptive, not fit

C1 Fork-3 recommendation stands (F-T4): ~500 Stream-Grab (or held-capture)
posts through the full pipeline as a DESCRIPTIVE stratum — score distribution,
flag rate, ∅-rate of an ordinary feed — excluded from the weight fit because
it has no labels. This also dissolves the circularity risk (labeling by the
instrument we then fit): no label, no fit, no circle. If Daniel wants a
trivially-true tail IN the fit: label via the FULL verify loop (a different
instrument than the urn score) + 50-claim hand-audit, cap at ≤10% of TRUE
rows. Default: out.

## Cross-cutting (both strata, restated as one checklist)

- Ceilings: snowflake post date, hard-clamped; note text offline-only
  (matcher + gates); x.com + CN mirrors excluded from retrieval.
- Dedup: claim-level near-dup WITHIN stratum, ACROSS strata (a viral claim can
  sit in both the CRH and CRNH pools — contradictory labels must dedup to
  zero), and vs E1/E2 (leakage tag).
- Competently-extracted-only applies to the TRUE side identically: unrecovered
  posts deferred, never forced in.
- Cluster-level folds (Fork 4) extended: fold key = claim cluster across BOTH
  classes.

**Expected label noise (one table):**

| sub-stratum | mechanism | expected | bound by |
|---|---|---|---|
| CN false (gated) | note wrong / gate error | ~3–8% | audit done (33-claim triage) + partly-band |
| Pool B L3 | rejection ≠ truth | unknown, guess 5–15% | 150-audit; demote >10% |
| Wire | wire error + tweet distortion | 1–3% (+5–15% axis-ii, excluded) | 200-audit |
| Trivially-true | — | — | not in fit |

## Cost (remaining C2+C3, inside envelope)

Hydration free (~1 h). Extraction ~$0.15 (2,050 TRUE-side posts + reruns).
Match+gates ~$0.35. TRUE-side verification ~1,150 × $0.0014 ≈ $1.6; FALSE full
build $0.55 + verification $1.2; descriptive 500 ≈ $0.25; audits ≈ $0.15.
**Total ≈ $4.3**; sprint spend to date ≈ $1.1 (ledger) + parallel query-gen
session ≤$6 cap — comfortably inside the $25 envelope.

## Forks for Daniel

- **F-T1 acquitted rung:** L2 (36,699) / **L3 (21,511) [REC]** / L4-only (1,402).
- **F-T2 TRUE-side size:** **~1,000 balanced [REC]** / 1.5× TRUE-heavy (helps
  threshold CI, +$1) / smaller (~600, thinner but cheaper).
- **F-T3 wire share:** 0% (Pool B only) / **≤35% cap, reported separately
  [REC]** / uncapped.
- **F-T4 trivially-true:** **descriptive-only [REC]** / labeled tail in fit
  (verify-loop labels + hand-audit, ≤10%).
- **F-T5 audit depth:** **wire 200 + Pool B 150 [REC]** / halve (100/75,
  looser bounds) / Daniel-reads-everything (slower, strongest).

Sequencing on sign-off: hydrate L3 sample (free, ~1 h) → TRUE-side smoke
50+50 posts (~$0.05, mirror-audit numbers incl. rejected-note recovery) →
CHECKPOINT → full TRUE build + full FALSE build together → C3.
