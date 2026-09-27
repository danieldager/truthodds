> **⚠️ HISTORICAL SNAPSHOT (pipeline v0.1)** — superseded by `docs/pipeline.md` (v0.2, 2026-05-18). Numbers below are from the old retrieve→verify pipeline (`src/_archive/v0_1/pipeline/`) and do not reflect current architecture.

# Pipeline Baselines

---

## Baseline 1 — AVeriTeC Dev, n=18, seed=42 — 080526

### Setup

| Parameter | Value |
|---|---|
| Date | 08/05/26 |
| Dataset | AVeriTeC dev split (`pminervini/averitec`) |
| Sample | n=18 of 50 target (seed=42); run truncated at claim 19 by Groq daily TPD limit (llama-3.3-70b-versatile, 100k/day) |
| Extraction model | `qwen/qwen3-32b` via Groq |
| Verification model | `llama-3.3-70b-versatile` via Groq |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` + MMR (λ=0.75/0.25, k=10→6) |
| Search | Serper (Google) + FCTAPI (Google Fact Check) + Wikipedia |
| Retrieval rounds | 1 (MAX_RETRIEVAL_ROUNDS=0, iterative mode disabled) |
| Scrape timeout | 10s per URL |
| CRED-1 threshold | 0.3 |

**Accuracy: 11/18 = 61%**

Note: Target was 50 claims. Results treat 18 claims as the complete sample for baseline purposes. A full 50-claim run should be repeated after upgrading the verification model (see observations below).

---

### Per-Claim Results

| # | Gold | Predicted | Match | Claims | Source | Post (truncated) |
|---|---|---|---|---|---|---|
| 1 | Supported | Not Enough Evidence | ✗ | 2 | Debbie Mucarsel-Powell | Carlos Gimenez approved a 67% pay raise for himself... |
| 2 | Supported | Not Enough Evidence | ✗ | 1 | NY-27 debate | Nate McMurray voted to raise taxes on homeowners on Grand Island |
| 3 | Refuted | Not Enough Evidence | ✗ | 1 | Facebook | New Zealand's new Food Bill bans gardening |
| 4 | Refuted | Refuted | ✓ | 1 | Twitter | US Congress voted 49-46 to add repeal of Section 230 into defense bill |
| 5 | Supported | Supported | ✓ | 1 | Facebook | Charlotta Bass was the first Black woman to run for VP in 1952 |
| 6 | Supported | Not Enough Evidence | ✗ | 1 | Facebook | Nigeria GDP 2020 decreased from 2019 |
| 7 | Refuted | Supported | ✗ | 1 | Instagram | Donald Trump is facing a court case for raping a teen in 1994 |
| 8 | Refuted | Not Enough Evidence | ✗ | 1 | Twitter | India's imports from China increased by 27% April-August 2020 |
| 9 | Refuted | Refuted | ✓ | 1 | Facebook | Western Wildfires evidence of coordinated Antifa campaign |
| 10 | Refuted | Refuted | ✓ | 2 | Facebook | Marcos and Rizal founded the World Bank and IMF |
| 11 | Supported | Supported | ✓ | 3 | None | Georgia has 100k more COVID cases and twice the deaths as NC |
| 12 | Refuted | Refuted | ✓ | 1 | Twitter | Idol of goddess Kali burnt by Muslim community in Murshidabad |
| 13 | Refuted | Not Enough Evidence | ✗ | 2 | Facebook | Daniel Andrews borrowed from IMF and is in default |
| 14 | Refuted | Refuted | ✓ | 1 | Daily Mail | COVID pox parties could create herd immunity without a vaccine |
| 15 | Refuted | Refuted | ✓ | 1 | Facebook | Two Sigma Investments is owned by George Soros |
| 16 | Supported | Supported | ✓ | 1 | Independence Speech | 52% of Nigeria's population lives in urban areas |
| 17 | Refuted | Refuted | ✓ | 1 | The Federalist | Masks would not meaningfully help with aerosol transmission of COVID-19 |
| 18 | Not Enough Evidence | Not Enough Evidence | ✓ | 3 | Twitter | Border barriers decreased drug/border crossing/human smuggling activity |

---

### Label Breakdown

| Gold Label | N | Correct | Accuracy |
|---|---|---|---|
| Supported | 6 | 3 | 50% |
| Refuted | 11 | 7 | 64% |
| Not Enough Evidence | 1 | 1 | 100% |
| Conflicting Evidence | 0 | — | — |
| **Total** | **18** | **11** | **61%** |

All 7 errors involve the system predicting `Not Enough Evidence` (6 cases) or `Supported` when the gold is `Refuted` (1 case). The system never predicts `Conflicting Evidence`. There are zero false positive alarms (no claim is wrongly labelled `Supported` for a gold `Refuted`... except claim 7).

---

### Retrieval Statistics

| Metric | Across 18 claims |
|---|---|
| Avg atomic claims per post | 1.4 |
| Avg evidence items per claim (after MMR) | 5.1 |
| FCTAPI hits (at least one structured fact-check) | 8/18 (44%) |
| Wikipedia 429 rate limit errors | 5/18 (28%) — Wikipedia throttled aggressively |
| Scrape-blocked URLs (Facebook/Twitter/etc.) | ~40% of candidate URLs |
| Cred-blocked URLs | <5% — CRED-1 rarely fires on novel domains |
| Domains with cred score > 0.5 | Mostly only known fact-checkers (PolitiFact=0.90, AFP=0.90) |

---

### Failure Analysis

#### Pattern 1: NEE over-prediction on Supported claims (claims 1, 2, 6)

The verifier is calibrated conservatively: it returns `Not Enough Evidence` whenever it cannot find a source that *directly confirms* the claim, even when the indirect evidence is consistent with it.

- **Claim 1** (Gimenez pay raise): PolitiFact rated the claim "Half True." The verifier reasoned that "Half True ≠ Supported" and abstained. The correct call per AVeriTeC's schema is `Supported` (the core fact — a pay raise was approved — is true; the "67%" figure is where PolitiFact quibbled). The verifier is too literal about verdict labels from external fact-checkers.
- **Claim 6** (Nigeria GDP 2020 vs 2019): World Bank data was scraped from `data.worldbank.org` but the pages returned time-series tables without explicit year-to-year comparison text. The verifier could not synthesize a comparison from raw table data. This is a document summarization failure — the evidence existed, the verifier couldn't read it.

#### Pattern 2: NEE over-prediction on Refuted claims (claims 3, 8, 13)

These are retrieval failures, not verifier failures. The verifier correctly says "I can't tell" — because it wasn't given enough evidence.

- **Claim 3** (NZ Food Bill): No FCTAPI hit. Search returned NZ gardening/food pages but not the specific fact-check debunking the "bans gardening" claim. The fact-check exists (Snopes, AFP NZ bureau) but wasn't surfaced.
- **Claim 8** (India-China imports 27%): The key primary sources (NBS India reports, Indian trade data portals) are mostly on Facebook/Instagram/X (all blocklisted) or require authentication. Only 2 evidence items retrieved, neither addressing the specific figure.
- **Claim 13** (Andrews/IMF default): This is a fringe Australian conspiracy claim. No mainstream sources cover it because there is nothing to cover (subnational governments cannot borrow from the IMF). The correct reasoning is "the premise is institutionally impossible therefore false," but the verifier requires *positive evidence of refutation* rather than reasoning from absence.

#### Pattern 3: False Supported on Refuted (claim 7 — Trump rape 1994)

The claim was "Donald Trump is facing a court case for raping a teen in 1994." Gold label: `Refuted`. The verifier found that a lawsuit *was* refiled (by Jane Doe in 2016) and concluded `Supported`. 

Two issues: (a) the verifier's temporal anchoring is wrong — the claim was made in October 2020, and the lawsuit had been dropped in November 2016 before trial, so it was false at the time of the claim; (b) the verifier confused "there was a legal action" with "Trump is actively facing a court case." This is a nuanced legal/temporal reasoning failure, and exactly the kind of case IDEA-007 (document summarization with explicit temporal anchoring on evidence text) is designed to address.

#### Cross-cutting observations

1. **FCTAPI is the strongest single signal.** When there is a FCTAPI hit with a clear verdict (claims 5, 10, 11, 12, 16), the system performs very well. When there is no FCTAPI hit, retrieval quality drops sharply and so does accuracy.

2. **CRED-1 scoring is nearly inert.** Almost every scraped source gets `cred=0.50` (unknown domain). The high-credibility scores (0.90) only appear for fact-checkers like PolitiFact and AFP — exactly the sources already captured by FCTAPI. The CRED-1 database adds little marginal signal for general web sources.

3. **Wikipedia rate limiting is significant.** 5 of 18 claims (28%) hit Wikipedia 429 errors, removing what should be a reliable general-knowledge source. This needs a backoff/retry strategy.

4. **Facebook/Twitter/Instagram blocking removes ~40% of candidate URLs.** Many viral claims originate on these platforms, and debunking coverage also lives there. The blocklist is correct (these pages are unscrapable), but it means the query strategy needs to work around this — prioritizing news domains and fact-checking organizations in search queries.

5. **Multi-claim posts perform well.** Claims 10 (2 sub-claims) and 11 (3 sub-claims) and 18 (3 sub-claims) all returned correct verdicts. The decomposition step is working: breaking a compound claim into independent checkable assertions allowed the verifier to reason sub-claim by sub-claim.

6. **The verifier never predicts Conflicting Evidence.** In this sample there were zero gold `Conflicting Evidence` claims (one appears at claim 19 which was not completed). This label requires the verifier to actively surface contradictory sources, which the current pipeline prompt may not incentivize.

---

### Implications for Next Steps

| Observation | Relevant IDEA | Expected gain |
|---|---|---|
| Verifier can't synthesize from raw table data (claim 6) | IDEA-007: per-document summarization before verification | Fixes numeric/tabular claims |
| Temporal anchoring failure on evolving legal claims (claim 7) | IDEA-007: date-anchored evidence summarization | Fixes time-sensitive claims |
| Reasoning-from-absence fails on impossible premises (claim 13) | IDEA-009: misleadingness detection / implicit assumption checking | Catches "premise is false" cases |
| FCTAPI coverage is the strongest predictor of accuracy | Grow claims database (IDEA-003) | Long-term recall improvement |
| Retrieval fails for low-salience claims (claims 3, 8) | IDEA-005: multi-hop QA + IDEA-006: iterative retrieval | Better evidence surfacing |
| NEE bias: verifier won't commit without direct confirmation | Calibration layer / Likert threshold tuning | Better Supported/Refuted recall |

---

### Notes on Model Choice

The run exhausted 100k daily tokens on `llama-3.3-70b-versatile` across 18 claims (≈5,500 tokens/claim). At this rate a 50-claim run requires ≈275k daily tokens, which exceeds the free tier. For sustained testing, either:

- Upgrade to Groq Dev tier (higher TPD limits), or  
- Switch verification to a more token-efficient model: `llama-3.1-8b-instant` (much cheaper, will score lower) or `deepseek-r1-distill-llama-70b` (if budget allows), or  
- Move to the Anthropic API: `claude-haiku-4-5-20251001` is fast and cheap; `claude-sonnet-4-6` for higher accuracy.

The extraction model (`qwen3-32b`) used ~650 tokens/claim after disabling thinking mode (`/no_think`), well within limits.
