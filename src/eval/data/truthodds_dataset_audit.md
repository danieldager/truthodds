# Truth Odds program — fc-gold v2 dataset audit results (2026-07-24)

Fleet: 3 claim-quality agents + 2 harmonization agents + local code passes
(fc_claim_cluster.py). Research agents (true-scarcity) reported separately.
Compiled as reports arrive; final synthesis at bottom once all in.

## Local code passes (fc_claim_cluster.py)

- 47,111 rows → 46,195 claim clusters; 826 multi-row; **374 cross-publisher clusters**
  (split-leak risk — CAL/VAL must be cluster-disjoint); largest cluster spans 6 pubs.
- Quality flags: **deixis 10,013 (21%)**, question 196, short 110, fragment 71,
  non-latin 9. Deixis >> altered_media (1,753): thousands of media-referring claims
  carry plain true/false ratings.
- Date dirt: 3 corrupt future dates (2027/2029/2323); zombie claims with claim_date
  back to 1994 inside the 2020+ window (entered via review_date) → ceiling rule needs
  `max(claim_date, review_date − lag)` or a pre-2018 claim-date exclusion. DECIDE.

## Claim quality — boomlive.in + verafiles.org (120 sampled, seed 103)

89% OK overall (VERA near-clean 98%; BOOM 84%, pathologies concentrated there:
deixis/fragment/meta ~16%). No Hindi/Tagalog leakage in slice. Rules: exclude
`^(The|This|These) (viral )?(video|photo|image|clip)s? (is|are|depict)`, "like this",
no-finite-verb noun phrases; ~75–80% pathology coverage, <1% FP. ~4–5% of rows would
generate a genuinely useless query.

## Claim quality — snopes.com + politifact.com (140 sampled, 70 true-side, seed 102)

- Structural pathology: TRUE-side 7%, FALSE-side 14%. All 8 CONTEXT-DEPENDENT cases
  are PolitiFact "Says…"/bare-quote style (no speaker in claim_text).
- **TRUE-side MUNDANE: 23%** — trivia/weird-news with no misinformation retrieval
  profile ("Two friends found pipes in the woods…"). Not regex-detectable; needs a
  cheap LLM screen (audit-scale, not labeling).
- Agent judgment: **~70–75% of the Snopes TRUE pool usable for urn calibration; ~60%
  if requiring evidence independent of the fact-check ecosystem.**
- Rules R1–R5: `^Says\b` / quote-dominant without PERSON-ORG outside quotes (exclude);
  trailing rating-token junk (repair); media-noun openers <90 chars w/o proper noun
  (exclude); deictic time w/o absolute date; fragment flag. ~75% coverage, <1% FP.

## Harmonization in context — polar classes (140 sampled, seed 105)

- **clear_true: 0/60 hard errors** (rule-of-three 95% UB ~5%; realistic ~0–1% → ≤40
  flips in the 4k TRUE pool). Its real defect is decontextualization (~5%: quote-only
  rows lacking speaker/comparator), not labels. All 9 Correct-Attribution rows
  correctly phrased as "X said Y".
- clear_false: 1 hard polarity trap ('Scam'-rated row whose claim_text IS the debunk)
  → code fix: drop/re-route `original_rating == 'Scam'` rows.
- altered_media/satire: 1 mis-tag leak-out (fullfact free-text "False. 26 June was
  the hottest…" routed to altered_media by keyword) → audit fullfact/aap long
  free-text ratings' subtype routing.
- **Structural finding: ~40% of clear_false are "Photo/Video shows X" media-shaped
  claims under generic 'False' ratings** — the media-exclusion boundary is
  rating-string-driven, not claim-driven; converges with the local deixis flag (21%).
  Fixing needs claim-level classification (the deixis flag IS that first pass).

## PENDING: wire-service claim quality; harmonization A (middle classes); 3 research
agents (true-claim sources, methods, politifact/DataCommons recon).

## Claim quality — wire-service pubs (140 sampled, seed 101) [4/8]

- OK only 59%; **DEIXIS 27%, FRAGMENT 9% — an AFP problem**: factcheck.afp.com 49%
  pathological, factuel 46% (claim_text is often the ARTICLE HEADLINE, e.g. "Photo of
  runners Abel Mutai and Ivan Fernandes"); fullfact/factcheck.org/aap near-clean.
  AFP = 73% of this pool → its rates dominate.
- Rules with FULL-SUBSET measured coverage (19,827 rows): R1 EN-deixis regex →
  **5,740 rows = 29.0%** excluded; R2 FR-deixis → 404 rows; R3 finite-verb check
  (spaCy one-liner safer than regex) ~9%; R4 strip `^Claims "` wrapper.
- **altered_media does NOT cover deixis**: only 10% of regex-deictic rows carry the
  flag (and 48% of altered_media rows aren't textually deictic) → the deixis regexes
  must run independently of the subtype flag. Confirms the local-pass finding.

## Research — methods for true-scarcity (lit review) [5/8]

1. **Fact-checker-sourced TRUEs are the field standard**; AVeriTeC is refuted-heavy
   for the same reason and fixes it with temporal cutoffs + FC-exclusion, not new
   label sources. Broaden via ClaimReview true-rated items across many orgs (= our
   harvest), keep the "doubted claims" distribution — it matches the nudge target.
2. **Glockner et al. EMNLP 2022**: every major dataset leaks fact-check articles into
   evidence; models learn the leak. Direct endorsement of our FC-block + ceiling.
3. **PU learning fits our shape**: labeled FALSE = positives, the tweet pool =
   unlabeled; mixture-proportion estimators (Elkan-Noto, TIcE, DEDPUL, BBE) recover
   the TRUE-side signal density as (f_unlabeled − α·f_FALSE)/(1−α) — cross-checkable
   against the 4k labeled trues; our ~77%-supported base rate anchors α.
4. **SCAR violated** (fact-checked falses are selected-not-at-random) → PULSNAR
   (PeerJ CS 2024); precedent Wright & Augenstein 2020.
5. **Publisher concentration → importance weighting** (KMM-style density-ratio
   reweighting of the 4k trues against the tweet pool), not resampling.
6. **Community Notes as weak supervision is VALIDATED in the literature**: 98% of
   COVID notes rated accurate by medical experts (Wojcik 2022); Borenstein 2025 finds
   professional-fact-checker-level quality but notes lean on FC sources (leakage
   caveat applies to CN correctives too).

## Research — PolitiFact + DataCommons recon [6/8]

- **PolitiFact scraping: easy + permitted.** Static WordPress HTML, 20/page, path
  pagination `/factchecks/list/page/N/?ruling=...`; robots permissive (Crawl-delay
  10s). Volume: **~750±100 true-side rulings 2020+** (True ~230, Mostly True ~530) in
  ~40 page fetches. Net add over our GFC politifact true-side (481): modest (~270)
  BUT includes proper stated-on dates (fixes politifact's missing claim dates).
- **DataCommons ClaimReview dump is LIVE and refreshed daily**: 199MB data.json,
  last-modified TODAY, schema.org DataFeed with FULL ClaimReview records —
  **appearance URLs included** (IDEA-011 gold: post-tied claims!) and complete rating
  text, richer than claims:search, offline-filterable without query restrictions.
  → Candidate single best true-side + coverage move: one 199MB download.

## Harmonization in context — middle classes (130 sampled, seed 104) [7/8]

- **Disagreement: mixed 22%, unprovable 23%; mostly_true/mostly_false 0/50.** The
  noise is concentrated exactly where it matters least for binary calibration —
  polar classes clean — but the contested urn + any {3+NEE} nudge policy need fixes
  first.
- Systematic pattern 1 (the big leak): **AFP/BOOM "Misleading"/"Trompeur" covers BOTH
  genuinely-contested claims AND false-by-context media** (old photo passed off,
  doctored image) → ~10/45 sampled mixed rows should be 1–2. Not remappable by rating
  string alone; title-keyword heuristic ("doctored", "old photo/video", "falsely",
  "Non,") catches most.
- Systematic pattern 2 (code-fixable): `Outdated`/`Outdated Figure` → remap
  unprovable→mixed (temporal invalidity ≠ absence of evidence); fullfact/snopes
  free-text verdicts opening with definitive negatives ("No, …", "This is a fake
  video", "…did not") must not land in unprovable ("Infondé" leans refuted too).

## Research — true-claim sources ranked [8/8]

- Recon reconciliation: the LIVE 199MB daily data.json = the Fact Check Markup Tool
  feed (fresh, but only tool-submitted markup; today's first record is BR24); the
  frozen thing = the 2019 research snapshot. The real bulk superset is
  **Fact-Check Insights (Duke Reporters' Lab)**: 240k+ claims from dozens of
  publishers, updated daily, free for researchers ON REGISTRATION — supersedes
  claims:search (which filters publishers "algorithmically").
- PolitiFact: no API; listing scrape robots-compliant (10s delay), ~800–1,200
  True+Mostly-True 2020+, `speaker_type=SOCIAL` filter for viral style; Kaggle 21k
  dump (2008–2022) as head start.
- True-verdict publishers at volume: Africa Check (Correct/Mostly Correct, EN+FR),
  Check Your Fact (best US-viral style fit), The Journal FactCheck, Demagog (political
  style, non-EN) — all inside FCI/ClaimReview anyway.
- Academic: **AVeriTeC ~970 Supported** (real fact-checker claims, best style fit,
  CC BY-NC), ClaimsKG v4 74k w/ normalized TRUE (heavy snopes/politifact overlap —
  dedupe), MuMiN ~650 factual, MultiFC stale, FEVER wrong distribution (skip).

## SYNTHESIS — proposed dataset-build rules (for Daniel's sign-off)

**Label trust**: polar classes clean (clear_true 0/60, mostly_* 0/50 hard errors);
middle classes ~22% mapping noise → binary calibration is safe NOW; contested/NEE
urns need the remaps below first.

**Exclusion/repair stack for CAL/VAL** (flag, don't delete; measured coverages):
1. Deixis R1/R2 regexes, independent of altered_media (~21% of all rows; 29% of wire
   pool) → excluded from retrieval experiments.
2. Finite-verb fragment check (spaCy) — ~9% of wire pool.
3. Quote-without-speaker / `^Says\b` exclusion (politifact style) — high precision.
4. Drops/repairs: 'Scam'-rating rows; trailing rating-token junk; `^Claims "` wrapper;
   corrupt dates (3 rows).
5. Remaps: Outdated→mixed; definitive-negative free-text out of unprovable; audit
   fullfact/aap long-rating subtype routing; AFP-Misleading title-keyword demotion
   (doctored|old photo|falsely|Non,).
6. MUNDANE screen on true side (~23% of snopes trues): cheap LLM audit-screen —
   DECISION (bounded cost, or accept + stratify).
7. Zombie-claim ceiling rule — DECISION (max(claim_date, review−lag) vs pre-2018 drop).
8. Splits cluster-disjoint (374 cross-pub clusters).

**Pool projection after stack**: TRUE ~2.4–2.8k calibration-clean; FALSE ~18–20k.
TRUE remains binding → ranked relief: (1) **Fact-Check Insights registration +
download** (thousands of trues, breaks the snopes skew, daily-updated); (2) AVeriTeC
Supported (~970) + ClaimsKG TRUE 2020+ slices, deduped in; (3) PolitiFact listing
scrape (~1k, restores stated-on dates); (4) CN correctives as robustness stratum
(lit-validated: 98% accuracy on COVID notes, FC-leakage caveat).

## E1 forensics — the "wordlist" catches (agent, 2026-07-24)

**Mechanism**: queries for fabricated claims are rare-term conjunctions ("KFC
Kristallnacht promotion apology Germany") that co-occur on NO real page; Google's
long-tail fallback then matches any document containing all terms ANYWHERE — and a
100k–333k-line dictionary/vocab file contains every English word, making wordlists
universal acceptors. Verified term-by-term on 10 cases (all query terms present as
isolated dictionary entries).

**Taxonomy of the 202 zero-prose results**: 130 wordlist/vocab/n-gram files (incl.
HuggingFace model vocabs), 23 bot-walled real articles (MDPI Akamai 403 etc. — often
topically RELEVANT; scraper gap), 18 real PDFs with failed extraction (FEC, court
briefs — real primary docs; pypdf routing gap to investigate), 18 HTML shells,
13 data files (xlsx/csv/sitemaps).

**Urn recode refinement**: junk-observed = wordlists+data+shells (161); UNREADABLE =
bot-walled + failed-PDF (41) — real content, observation failure; recode by the
agent's URL-pattern classes, not provenance alone.

**Diagnostic finding**: wordlist hits are a FINGERPRINT of claims with no genuine web
footprint — a code-detectable fabrication correlate; candidate explicit Truth Odds
signal (vocab-file-hit count per claim).

## Research — long-context READ degradation (agent, 2026-07-25)

- Positional loss (Lost in the Middle) visible at few-k tokens: mid-doc evidence is
  the at-risk region regardless of budget. RULER: claimed ≠ effective context (most
  models bend well before 32k). NoLiMa: SEMANTIC needles (no lexical match — our
  task's shape) degrade from single-digit-k. Context Rot: distractor density, not
  raw length, drives the bend. DeepSeek family: vendor NIAH flat to 128k but
  RULER-qa2 ~43% — don't budget off the NIAH heatmap. LongCite: fine-grained
  CITATION fails first (11–57% bad-reference rates in long contexts).
- **Defensible per-doc budget: ~4k tokens, ceiling 8k tokens** — our current 8k-char
  (~2k-token) cap is conservative-safe; room to raise if the cap-sweep supports it.
- Cap-sweep must instrument: (a) pointer validity/precision BY EVIDENCE POSITION
  (beginning/middle/end thirds) — expect citation drift + middle-third recall loss
  BEFORE direction accuracy moves; (b) distractor-dense pages at fixed length.
