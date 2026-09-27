# Community Notes as a dated-TRUE gold subset — scoping brief

**Date:** 2026-08-03 · **Status:** scoping only, nothing built, nothing paid
**Question:** can CRH corrective notes become a validated, DATED, TRUE-claim gold subset
for the Truth Odds urn eval?
**Recommendation:** **conditional GO on a ~100-claim pilot** (<$0.15 LLM, ~5h human).
**NO** to CN as a primary TRUE source — it must be a separate stratum, never pooled.

---

## 0. Two premise corrections before anything else

**(a) Hydration is already finished.** `TASKS.md:20` and `ROADMAP.md:62` describe CN
hydration as GATED, ~128.5k requests, ~43h, unrun. It ran to completion on 2026-07-26 and
the completion was never logged (`src/clog/260726.md` does not exist; the trail stops at
`250726.md:20` "Hydration paused."). On disk:

| | count | of list |
|---|---|---|
| unique tweetIds attempted | 128,506 | 100.0% |
| HTTP 200 | 90,053 | 70.1% |
| with non-empty post text | **88,038** | **68.5%** |
| 404 (gone) | 37,604 | 29.3% |

The 70% deletion-probe estimate (`230726.md:225-229`, n=300) held exactly at scale (70.1%).
Consumers must dedup `hydrated.jsonl` by `tweetId` — 40,055 ids have two records from a
24-July double-run; 40,017/40,055 (99.9%) are code-concordant, 38 discordant. **There is no
hydration cost left in any estimate below.**

**(b) The date does not need hydration at all.** `tweetId` is a snowflake on 248,731/248,731
rows. `(id >> 22) + 1288834974657` decodes the post's creation instant. Validated against
20,000 hydrated posts carrying a real `created_at`: **20,000/20,000 within 60s, median
absolute error 0.5s, p99 1.0s.** Exact post timestamps are available for 100% of notes at
zero API cost — including the 29.9% whose posts are deleted.

---

## 1. YIELD — hand-read of 71 corrective notes

**Method.** Proportional stratified sample of 71 cluster representatives (`is_rep=True`,
n=127,093), seed 20260803, strata = the FactualError × MissingImportantContext flag cross
(A factual-only 18 / B both 25 / C context-only 18 / D other 10). Read by hand, no LLM. Codes
frozen in the session scratchpad before tabulation. One primary category per note.

| category | n | share | definition |
|---|---|---|---|
| **TEXT_TRUE** | **27** | **38.0%** | asserts a specific affirmative proposition verifiable from text sources alone |
| PROVENANCE | 20 | 28.2% | asserts origin/date/identity/authenticity of the *attached media* — needs the image or video |
| NEGATION | 10 | 14.1% | only denies, or asserts absence of evidence ("no official announcement") |
| META | 9 | 12.7% | platform behaviour, ToS, engagement farming, account identity, user allegations |
| VAGUE | 5 | 7.0% | too hedged or opinionated to verify |

**TEXT_TRUE = 27/71 = 38.0%, 95% Wilson CI [27.6%, 49.7%].**
Restricted to well-sourced (dropping 4 with thin/self-reported/time-decaying citations):
**23/71 = 32.4%.**

Flags barely discriminate — TEXT_TRUE rate is 44% (B), 44% (A), 39% (C), 10% (D). Only the
"neither factual-error nor missing-context" stratum is worth pre-filtering out. The
self-assigned flags co-occur heavily (`misleadingMissingImportantContext` is true on 59.8%
of all notes, `misleadingFactualError` on 58.4%) and carry little signal.

**Projection.** n=71 is small; the interval is wide and dominates everything below.

| stage | count | basis |
|---|---|---|
| cluster representatives | 127,093 | verified |
| × TEXT_TRUE 38.0% | **48,300** (CI 35,100–63,200) | n=71 |
| of which post text hydrated (68.5%) | **33,100** (CI 24,000–43,300) | verified |
| × EN only (76.8% of live posts) | **25,400** | X's own `lang` |

**Against a 500–1,000 target this is 25–50× headroom. Yield is not the binding constraint —
validation effort and distributional validity are.**

---

## 2. VALIDATION PROTOCOL — we are the validator

Three things need validating, and they are separable. Prior from the literature
(`truthodds_dataset_audit.md:87-90`): Wojcik 2022 found 98% of CRH COVID notes rated accurate
by medical experts; Borenstein 2025 finds professional-fact-checker-level quality. Those are
**priors on a different topic mix and era, not a substitute for our own measurement.**

### V1 — note veracity (is the note's affirmative assertion actually true?)

- **Frame:** TEXT_TRUE candidates surviving the §4 firewall.
- **Labeller:** Daniel adjudicates. Presented with note text + post text only — **blind** to
  the LLM screen score and to the extracted claim.
- **Critical design point:** the labeller must consult **at least one source independent of
  the note's own citations.** Reading only the cited page validates the citation, not the
  claim, and would inherit exactly the bias we are trying to measure.
- **Scale:** {supported / not supported / cannot determine}. "Cannot determine" counts
  against precision (it means the claim is not usable as gold).
- **n = 100, and this number is derived, not chosen.** Pre-register the bar at
  precision ≥ 0.85 (95% lower bound). At an expected p̂ = 0.92, Wilson gives:
  n=60 → [0.824, 0.966] (**fails the bar**); **n=100 → [0.850, 0.959] (clears it exactly)**.
  n=100 is the smallest sample that can pass a 0.85 bar at the expected precision.
- **Decision rule (committed before labelling):** BUILD if 95% LB ≥ 0.85. Report the point
  estimate with its denominator in every downstream use.
- **Futility stop:** halt and declare NO-GO if ≥8 failures occur in the first 40 items.
- **Reliability:** 25-item overlap with a second labeller (human, or a Sonnet labeller
  validated per ROADMAP Phase 1 — labellers are capped at Sonnet). Report agreement. If the
  second labeller agrees with itself but not with Daniel, that is the 25-July pattern: stop.

### V1b — a free precision instrument (no human cost)

2,202 cluster representatives (1.73%) cite a URL that **is** an `fc_gold_v3.review_url`
(exact match after normalisation), resolving to 1,534 distinct fc_gold rows. For those pairs
the note and a professional fact-checker addressed the same item, so agreement is computable
in code at zero cost. **Caveat that must travel with the number: these are notes that chose
to cite a fact-checker, so they are a quality-biased subsample — this measures an upper
bound on note precision, not the population value.** Use as a sanity floor, not as V1.

### V2 — extraction fidelity (does the extracted claim carry the assertion, not the verdict?)

Mechanical, free, code-only — run on 100% of extractions:

- (a) no evaluative/verdict tokens ("false", "misleading", "actually", "debunked", "in fact");
- (b) not negation-only — the output must assert, not deny;
- (c) no unresolved deixis ("this video", "the post", "OP", bare pronouns) — reuse the
  `q_flags` deixis regex from `fc_claim_cluster.py`;
- (d) output ≠ verbatim note text (must be a rewrite, not a copy);
- (e) length within the fc_gold TRUE band (median 125 chars).

Human: **50 extracted claims** read against their source notes, scored {faithful / distorted /
verdict-imported / unsupported-addition}. ~1.5 min each ≈ 1.25h. Bar: ≥90% faithful.

### V3 — source liveness and support

- **Liveness** is mechanical and free (98.8% of notes carry ≥1 URL, median 1, 44,123 distinct
  domains). Measure and report it.
- **Do NOT filter on liveness.** A dead citation does not make a claim false, and filtering on
  it biases the set toward recent claims and stable domains.
- **Support** (does the cited page actually back the assertion?) is measured on the *same*
  100-item V1 sample — it is the same reading act — and reported descriptively. It does not
  gate inclusion, because the gold label comes from the validated assertion, not the citation.

---

## 3. DATE SEMANTICS — the post timestamp is safe, but it is not "claim birth"

**What is exact.** Post creation instant, 100% coverage, zero cost, validated 20,000/20,000
(§0b). Note creation is also exact (`createdAtMillis`).

**Post → note lag** (n=140,791, computed here): median **5.9h**, p25 2.3h, p75 14.3h,
p90 24.6h, p99 232h. **89.5% of notes land within 24h of the post; 98.8% within 7 days.**

This is a genuine improvement over the current TRUE side. `evidence_urn_run.py:440-461`
(`ceiling_for`) falls back to `review_date − PUB_LAG`, and snopes is explicitly absent from
`PUB_LAG` so it inherits `GLOBAL_LAG = 6` days. Measured over the live splits: **1,212/1,557
CAL TRUE rows (78%) and 819/1,000 VAL TRUE rows (82%) are retrieved under that guessed
6-day ceiling.** CN would replace a guess with an exact instant.

**But the premise "TRUE claims born on social media at a known instant" holds for only about
half of the usable notes.** The claim we would put in the urn is the *note's corrective
assertion*, not the post's false claim — and that assertion is often about a pre-existing
fact. Of the 27 TEXT_TRUE notes:

| | n | of TEXT_TRUE | of all 71 |
|---|---|---|---|
| fact **contemporaneous** with the post (born ~now) | 13 | 48.1% | **18.3%** (CI 11.0–28.8%) |
| fact **pre-existing / timeless** | 14 | 51.9% | 19.7% |

Contemporaneous examples: the Aug-2022 Tesla 3:1 split, the Feb-2026 Bangladesh lynching, the
Feb-2024 European Commission arms proposal, the Apr-2026 erythritol study. Timeless examples:
protein per 100g of chicken, the offside rule, Orwell's birthplace, HR 8799.

**Consequences.**

1. `ceiling = post_date` is **safe but loose**. Evidence for a timeless fact long predates the
   ceiling, so retrieval is abundant. Roughly half the CN TRUE claims are *easier* than the
   production regime, not harder — a distribution shift in the flattering direction.
2. **Recirculation is real and concentrated.** 11/71 notes (15.5%) explicitly state the content
   is older than the post; among PROVENANCE notes it is 7/20 (35%). Since PROVENANCE notes are
   excluded anyway (§1), residual recirculation in the usable set is low — but it is exactly
   the timeless half, so it is not eliminated.
3. **Use `post_date`, not `note_date`, as the ceiling.** The note is downstream evidence; a
   ceiling at the note's timestamp would admit sources published in response to the post.

---

## 4. LEAKAGE — the firewall

**Measured exposure.** 5,151/127,093 cluster reps (**4.05%**) cite at least one of 46
fact-checking domains (snopes 1.14%, leadstories 0.53%, politifact 0.50%, factcheck.afp 0.50%,
fullfact 0.23%). Fact-check URLs are ~2.5% of all 438,914 cited URLs — CN citations are
dominated by X itself (17.6%) and Wikipedia (3.6%). Exposure is small and mechanically
detectable.

**Five layers, cheapest first:**

1. **Selection.** Drop any note citing a fact-check domain (−4.05%). Removes the direct import
   of a professional verdict. Cost: one regex.
2. **Extraction.** The extractor sees **note text with URLs stripped**, nothing else. It never
   sees the fact-check article. Output is the affirmative proposition only. Same posture as the
   tier-2 firewalled claim resolution already specified in ROADMAP Phase 0.
3. **Retrieval — the date ceiling does most of the work for free.** `ceiling = post_date`
   automatically excludes the note itself (median 5.9h later) and every downstream fact-check.
   This is a materially cleaner firewall than fc_gold has, where the review post-dates the
   claim by an estimated lag.
4. **Retrieval — blocklists.** `exclude_domains` must carry (a) `x.com`/`twitter.com`/`t.co` —
   the note lives there and CN notes are public and indexed; (b) CN mirrors
   (`communitynotes.x.com`, birdwatch aggregators); (c) a blanket fact-check domain blocklist,
   since a CN claim has no single origin publisher for the existing per-publisher rule at
   `evidence_urn_run.py:497`.
5. **The note's own cited domains — do NOT exclude them,** but measure the sensitivity.
   Rationale: they are ordinary evidence (Wikipedia, Reuters, primary sources), not verdicts,
   and excluding them would artificially depress support and distort the urn. Run a subsample
   both ways and report the delta.

---

## 5. OVERLAP with fc_gold — a free cross-validation subset, but it is FALSE-side

Exact `review_url` match (normalised: scheme/www/query stripped, lowercased):

- **2,202 cluster reps (1.73%) cite a URL that is an `fc_gold_v3` review_url.**
- These resolve to **1,534 distinct fc_gold rows.**
- By cited site: snopes 804, factcheck.afp 477, politifact 275, fullfact 220, factcheck.org 169,
  aap 99, factuel.afp 93, boomlive 59, verafiles 6.
- **Their fc_gold veracity: 1,190 veracity-1 · 188 veracity-3 · 12 veracity-2 · 28 veracity-5 ·
  5 veracity-4 · 111 null.**

**Read this carefully: the overlap is 1,190 FALSE vs 33 TRUE.** That is structural, not
sampling noise — notes correct false posts and fact-checkers debunk false claims, so the
intersection lives on the false side. It therefore does **not** deliver the "same claim, two
independent labels" TRUE subset. What it does deliver:

- a ~1,534-pair agreement instrument for **note quality** (V1b above), free, biased upward;
- a FALSE-side cross-validation subset, which is not the current scarcity.

**Beyond exact URL:** joint clustering is directly available — `fc_claim_cluster.py` is a port
of `cn_cluster.py` (`fc_claim_cluster.py:4`), same union-find, same idf-weighted trigram
Jaccard. Running fc_gold claim_text and CN extracted claims through one shared pass would find
paraphrase-level overlap. **Yield unknown — nobody has measured it, and CN cluster precision
itself has never been QC'd (`TASKS.md:23`, never run; the 0.58/0.25 thresholds are
unvalidated).** Estimate it in the pilot, do not assume it.

---

## 6. DISTRIBUTION vs fc_gold — the transfer question

**Language.** The fc_gold TRUE side is effectively monolingual: **en 4,131 / fr 8** of 4,139.
French exists only on the FALSE side. CN post language (X's own tag, n=87,188): **en 76.8%,
fr 9.9%, es 8.9%, pt 1.2%.** CN note language (langdetect, n=500 on the clustered file):
en 80.0%, fr 8.6%, es 7.2%. **CN is the only route on the table that would give a non-trivial
French TRUE set** — ~8,700 French live posts, projecting to roughly 3,300 French TEXT_TRUE
candidates. That is a real capability gain, not just a top-up.

**Topic.** fc_gold has no topic column; topics come from the LLM claim screen
(`claim_screen.py:36-43`), covering 3,623/4,139 TRUE rows. fc_gold TRUE is
**politics_government 43.8%**, celebrity 10.0%, health 6.6%, religion/culture 6.4%,
consumer/scams 6.2%, economy 6.1%, science 5.9%, animals 4.9%, crime 4.9%, sports 1.7%.

CN has **no topic column and has never been screened.** From eyeballing my 71 (not a validated
taxonomy — treat as a hypothesis, not a measurement): CN skews markedly toward
entertainment, sports, internet culture and consumer/scam material, and away from US politics.
**The honest answer to Q6 is that this is not yet measured; running `claim_screen.py` over a
CN sample would make the two directly comparable using the same taxonomy** — that is a pilot
task, ~$0.05, and it should be in the pilot.

**Claim style.** fc_gold TRUE claims are largely retrospective quote-attribution and historical
trivia (snopes "Correct Attribution" alone is 596 rows), median 125 chars. CN correctives are
present-tense factual corrections with citations, median note length 289 chars. These are
different objects, and that is the point of adding CN — but it also means the urn parameters
should not be assumed to transfer.

---

## 7. COST AND TIMELINE

**Cost anchor.** The only measured per-row LLM cost for a comparable short-text extraction on
this project is `extract_claim_dates.py`: **$0.0073 for 60 rows = $0.000122/row**
(`clog/030826.md:107-113`). N=60, reps=1 — that is a thin anchor and everything below inherits
its uncertainty. Hydration is $0 and already done.

### Minimum viable pilot (~100 validated claims) — RECOMMENDED

| step | cost | time |
|---|---|---|
| mechanical filters + snowflake dates | $0 | ~2h eng |
| LLM TEXT_TRUE screen, ~400 notes | ~$0.05 | minutes |
| firewalled claim extraction, ~150 | ~$0.04 | minutes |
| `claim_screen.py` topic pass for §6 comparability | ~$0.05 | minutes |
| mechanical fidelity checks (V2 a–e) | $0 | ~1h eng |
| free fc-overlap agreement instrument (V1b) | $0 | ~1h eng |
| **V1 hand validation, 100 items, blind** | $0 | **~5h Daniel** |
| V2 fidelity read, 50 items | $0 | ~1.25h |
| **total** | **<$0.15** | **~1 day eng + ~6h human** |

Output: a note-precision estimate with a committed decision rule, an extraction-fidelity rate,
a measured topic distribution, and ~90 usable dated-TRUE claims immediately available as a
robustness stratum.

### Full build (500–1,000 validated dated-TRUE claims)

| step | cost | time |
|---|---|---|
| LLM screen, ~6,000 notes | ~$0.72 | ~1h |
| firewalled extraction, ~2,300 | ~$0.57 | ~1h |
| topic screen on the survivors | ~$0.30 | ~1h |
| joint re-cluster with fc_gold (`fc_claim_cluster.py`) | $0 | ~2h eng |
| v3-shaped adapter + splitter integration | $0 | ~1–2 days eng |
| V1 hand validation, 150 items (±5pp) | $0 | ~7.5h human |
| **total** | **~$1.60–3** | **~3 days eng + ~9h human** |

**LLM spend is negligible at every scale. The real currency is human validation hours and
engineering time on the adapter.** Note the adapter is not trivial: `evidence_urn_run.py`
needs a 7-field CAL-shaped row (`review_url`, `veracity`, `claim_text`, `publisher_site`,
`rating_subtype`, `claim_type`, `topic`) plus splitter membership, and no existing loader
targets that contract — the two precedents (`run_averitec_benchmark.py:37-58`,
`evidence_profile_run.py:130-147`) feed the *profile* runner's 5-field pseudo-post path.
`publisher_site` is a genuine problem for CN: there is no origin publisher, so the field must
be synthesised and the origin-exclusion rule replaced by the §4 blocklist.

---

## 8. Recommendation

**GO on the pilot. NO on CN as a primary TRUE source.**

For:
- Yield has 25–50× headroom; it is not the constraint.
- Exact, evidence-backed post timestamps replace a guessed 6-day ceiling that currently governs
  **78–82% of the TRUE side** of CAL/VAL.
- The date ceiling at `post_date` is itself a stronger leakage firewall than fc_gold has.
- Leakage exposure is small (4.05%) and mechanically removable.
- CN is the only route offering a French TRUE set.
- The decisive experiment costs <$0.15 and ~6 human hours.

Against, and the reason for "separate stratum, never pooled":
- **The TRUE label is not independent of the evidence.** A note reaches CRH *because* raters
  agreed its cited sources check out. Conditioning on "a CRH note exists" selects claims for
  which retrievable supporting evidence exists — which is precisely the quantity the urn
  estimates, P(support | TRUE). This is not fixable by better validation; it can only be
  bounded by cross-source comparison. It is the same objection already recorded at
  `230726.md:193-198`; my sample sharpens it rather than resolving it.
- ~52% of usable CN TRUE claims are timeless facts with abundant pre-existing evidence — easier
  than the production regime, not harder.
- The corpus is survivor-biased: 29.9% of noted posts are deleted, deletion is early and
  note-triggered, so what survives skews to the contested end (`230726.md:226-229`).
- CN cluster precision has never been measured; CAL/VAL disjointness would be nominal until it
  is.

**Cheaper adjacent option that should be weighed first.** AVeriTeC is already on disk
(`eval/data/averitec/averitec_full.parquet`, 3,568 rows **with `claim_date`**), has a working
adapter precedent, is already a first-class Truth Odds fitting set (`docs/truth_odds.md:41-43`),
and is ranked above CN for TRUE-scarcity relief (`240726.md:160-164`). It is near-zero cost and
solves the dating problem for a smaller, cleaner set. CN has the higher ceiling — scale,
French, exact social-media timestamps — at meaningfully higher effort. These are not exclusive;
the tri-source comparison (fc-gold / AVeriTeC / CN) is what actually measures the selection
bias none of them escape alone.

**If GO:** run the pilot exactly as specified in §7, with the §2 decision rule committed in
writing before any labelling begins.
