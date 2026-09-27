# Evaluation Dataset for Claim Verification — Findings & Decisions

**Session date:** 2026-05-18
**Goal:** Build a rolling, post-cutoff evaluation dataset for the claim-verification stage of the pipeline, to complement AVeriTeC (which has contamination risk and is frozen in 2022).

---

## TL;DR

- Built `eval/data/eval_v1.parquet`: **2,370 claims** from 8 publishers over Nov 2025 – Apr 2026 (~AVeriTeC 2.0 test scale).
- Harmonised to the project's 4-class verdict (`Supported / Refuted / Not Enough Evidence / Conflicting Evidence`) and a binary deployment label (`pass / flag`).
- Two data sources investigated (Google Fact Check Tools **API** and Data Commons **feed**); API ended up the better fit despite the feed being free.
- Label balance is heavily Refuted-skewed (~77%) — *consistent with AVeriTeC and standard for fact-check corpora*. Class imbalance is real but expected.
- Correction-mining experiment validated: can synthesise additional Supported claims from refutation articles. Parked for now; tagged outputs in `eval/data/mined_corrections_v1.parquet` if needed later.

---

## Data sources

### Google Fact Check Tools API

- Endpoint: `https://factchecktools.googleapis.com/v1alpha1/claims:search`
- Free; per-project Google Cloud daily request quota (default thousands/day).
- **Key constraints**:
  - Either `query` string OR `reviewPublisherSiteFilter` required — no "give me everything" mode.
  - No date-range query; only `maxAgeDays` ("newer than N days from today"). To target a past month, query wider and post-filter on `reviewDate`.
  - `pageSize` ≤ 100; implicit per-query cap around ~18–20 pages (~2,000 results), even for the most prolific publishers.
- **Fields per claim**: `text`, `claimant`, `claimDate`, `claimReview[]` → `{publisher{name,site}, url, title, reviewDate, textualRating, languageCode}`.
- **Notable absences vs ClaimReview JSON-LD**: no cited sources, no review body, no numeric rating.

### Data Commons live feed

- URL: `https://storage.googleapis.com/datacommons-feeds/claimreview/latest/data.json`
- 189 MB, ~92,742 entries, ClaimReview schema, **hourly refresh**, CC-BY, no API key.
- Strict superset of API fields — includes `itemReviewed.appearance[].url` (cited source URLs) for ~60% of entries.
- **Critical caveat**: the feed depends on publishers emitting ClaimReview JSON-LD markup, and many of the marquee EN publishers either don't or aren't ingested. Direct API queries are far more complete for those.

### Coverage asymmetry (key finding)

| Publisher | API 12-mo | Feed total (all-time) |
|---|---:|---:|
| Snopes | 1,781 | **0** |
| AFP Fact Check (EN) | 1,538 | **0** |
| Full Fact | 574 | **1** |
| AP News | 19 | **0** |
| The Quint (IN) | 910 | 2,080 |
| Factly (IN) | 828 | 12,153 |
| PolitiFact | 130 | 8,156 (historical) |
| FactCheck.org | 72 | 2,436 (historical) |
| Washington Post | 0 | 875 (historical) |

So:
- **API is essential for Snopes, AFP, Full Fact, AP, Reuters** — none of them publish workable ClaimReview markup.
- **Feed has PolitiFact / FactCheck.org / WaPo historical archives** but the API only returns ~last-year window (and seems to cap before exhausting these heavy hitters).
- The feed underestimates monthly EN volume by ~4× because it misses Snopes + AFP entirely.

---

## Publisher discovery

- Feed contains **1,307 unique publishers** worldwide; 89 have ≥100 historical entries.
- API has **23 publishers** with ≥1 EN claim in the last 365 days (probed all 89 candidates with one request each, with deep-paginate for the prolific ones). Total survey cost: **123 API requests / 4,001 EN claims**.
- Recent EN monthly volume (all 23): ~550-650/mo.

Full per-publisher data: `eval/data/survey/api_vs_feed.tsv`.

---

## Curated subset (current eval set v1)

Selection criteria:
1. **Exclude single-label / binary-only schemes** (would skew toward False): The Quint (all "False"), Factly ("FALSE"/"MISLEADING"), Vishvas News (2 labels).
2. **Exclude domain-specific publishers** (topic skew): THIP Media (medical), MedicalDialogues (medical).
3. **Exclude high free-text-ratio publishers** unless harmonisation is worth the LLM cost (`uniq_ratings / n_reviews ≥ 0.3`): DW (89% unique), AP News (68% unique), FactRakers (31%).
4. **Keep Full Fact** despite 97% unique rating strings — major Western source; uses LLM-based harmonisation.

Final 8: **snopes.com, factcheck.afp.com, newschecker.in, verafiles.org, rumorscanner.com, politifact.com, factcheck.org, fullfact.org**.

Window: **2025-11 to 2026-04** (6 months) → **2,370 claims**.

| Publisher | n | Region |
|---|---:|---|
| snopes.com | 857 | US |
| factcheck.afp.com | 713 | INT |
| newschecker.in | 300 | IN |
| fullfact.org | 237 | UK |
| verafiles.org | 132 | PH |
| rumorscanner.com | 60 | BD |
| politifact.com | 46 | US |
| factcheck.org | 25 | US |

---

## Scoring schemes — per-publisher characterisation

Top-10 ratings per publisher in our survey window:

- **PolitiFact**: Classic 6-point Truth-O-Meter (True / Mostly True / Half True / Mostly False / False / Pants on Fire).
- **Snopes**: 14 labels mixing verdict + media-type (True / Mostly True / Mixture / Mostly False / False / Unproven / Fake / Correct Attribution / Incorrect Attribution / Miscaptioned / Originated as Satire / Labeled Satire / Scam).
- **AFP**: 10 clean labels (True / False / Partly false / Misleading / Missing context / Unsubstantiated / Satire / AI-generated / Altered picture / Altered video).
- **Newschecker**: 8 labels (False / Altered Photo–Video / Misleading / Missing Context / Partly False / Satire / True / Altered Media).
- **Verafiles**: 4 labels (Fake / False / Misleading / Needs context / Satire).
- **Rumor Scanner**: 4 labels (False / Misleading / AI-GENERATED / Altered).
- **FactCheck.org**: 18 labels but loosely scoped — no clean True axis (Misleading / False / Unsupported / No Evidence / Exaggerated / Distorts the facts / Not the Whole Story / Disputed).
- **Africa Check**: 6 labels (Correct / Mostly correct / Incorrect / Misleading / Unproven / Understated–exaggerated).
- **Full Fact**: **free-text** — 557 unique strings in 603 ratings. Only 41% match a clean prefix pattern; the rest are full sentences with embedded verdict words.

---

## Harmonisation

Targets `config.VERDICT_OPTIONS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence"]`.

- **Rule-based** for 7 publishers — see `eval/harmonize.py`. Lookup table per publisher mapping lowercased rating → 4-class label.
- **LLM-based** for Full Fact + any rule-based fall-throughs (10.4% of total). Uses `qwen/qwen3-32b` on Groq via REST (skipped openai SDK due to environment-specific import issue).
- 240 of 246 LLM calls succeeded (97.6%). 6 failures were Full Fact AI-image rulings; Groq returned 400 (content-moderation filter trip on "image isn't real" phrasing). All 6 manually map to Refuted; left unlabeled for now since user wanted to defer the regex patch.

### Binary deployment label

Added `binary_label` column for the extension's actual decision (`pass` = no nudge, `flag` = nudge):
- `Supported` → `pass`
- `Refuted / Conflicting Evidence / Not Enough Evidence` → `flag` (any concern, including unverifiability, warrants a nudge)

---

## Label distribution

| 4-class label | n | % |
|---|---:|---:|
| Refuted | 1,836 | 77.5% |
| Supported | 307 | 13.0% |
| Conflicting Evidence | 196 | 8.3% |
| Not Enough Evidence | 25 | 1.1% |
| (unlabeled) | 6 | 0.3% |

| Binary | n | % |
|---|---:|---:|
| flag | 2,057 | 86.8% |
| pass | 307 | 13.0% |
| (unlabeled) | 6 | 0.3% |

- Snopes contributes **93%** of the Supported labels (285 of 307). Without Snopes, the Supported axis is critically thin.
- NEI (25) is too sparse to be meaningfully evaluated as a class on its own.
- This skew is **consistent with AVeriTeC** (and most fact-check corpora) — accepted as standard.

---

## Recommended metrics

Headline: **macro-F1** (per-class F1 averaged equally — not fooled by the 87/13 imbalance). Also report:
- Per-class precision/recall/F1 (to expose Supported/NEI weakness)
- Binary F1 on `binary_label` (matches deployment behaviour)
- Brier score if the pipeline outputs confidence (rewards calibration, not just hard labels)

Avoid headline accuracy — a "predict flag always" baseline scores 87% on this set.

---

## Correction-mining experiment (parked)

**Goal**: synthesise additional Supported claims from refutation articles' counter-facts.

**POC (10 claims)**: 8/10 successful. AFP unscrapable (anti-bot); Snopes / Full Fact / Newschecker / PolitiFact OK.

**50-claim run** (`eval/scripts/build_eval/mine_corrections.py`):
- Eligible pool (non-AFP Refuted, deduped by text): 1,138
- 50 sampled → 48 written (96% success)
- 27 positive assertions, 21 specific negations
- 0 null returns — LLM may be over-eager to extract; consider tightening prompt before scale
- Output: `eval/data/mined_corrections_v1.parquet` with `synthetic=True`, `is_positive_assertion`, `source_url` (for circular-evidence blocklist if used later)

**Extrapolated full-pool yield**: ~1,090 new Supported claims → binary balance would shift from 87/13 to ~57/43. Not pursued for v1 because:
- AVeriTeC has similar imbalance and is the field-standard benchmark
- Synthetic claims introduce risks (LLM hallucination, circular evidence) that need careful handling
- Better to measure pipeline performance on real data first, then decide if Supported recall is actually the bottleneck

**To revisit**: if pipeline eval shows pass-class recall is significantly worse than flag-class, scale to all 1,138 and merge synthetic rows into a v2 eval set.

---

## Caveats & known issues

- **AFP scraping blocked** — would lose 36% of any future article-level enrichment (e.g., evidence extraction from review bodies).
- **API page-cap** — Snopes and AFP hit ~2,000 results per query window; can't get deeper history without splitting by date or paginating multiple time windows.
- **Geographic skew** — Snopes + AFP dominate volume; the global publishers (Newschecker, Verafiles, Rumor Scanner) add topic diversity but skew Asia-Pacific.
- **PolitiFact / FactCheck.org volume is low via API** (~130/yr and ~100/yr) despite massive historical archives — they may not be markup-complete for the API, or the API may have an undocumented per-publisher window cap.
- **6 unlabeled rows** in eval_v1 (Full Fact AI-image refusals); easy regex patch later.

---

## Output artifacts

| Path | Contents |
|---|---|
| `eval/data/feed/data.json` | Raw Data Commons feed (189 MB; **gitignored** — re-download via cURL) |
| `eval/data/feed/publishers.txt` | All 1,307 publishers, sorted by feed volume |
| `eval/data/survey/12mo.jsonl` | Phase 1 API survey: 9 publishers, 365d |
| `eval/data/survey/api_publishers.jsonl` | Phase 2 API survey: 89 candidates probed |
| `eval/data/survey/api_vs_feed.tsv` | Per-publisher comparison table |
| `eval/data/eval_v1.parquet` | **The harmonised eval set** (2,370 rows) |
| `eval/data/eval_v1_summary.tsv` | Per-publisher × label breakdown |
| `eval/data/mined_corrections_v1.parquet` | Mining POC outputs (48 rows; parked) |
| `eval/harmonize.py` | Rule-based + LLM harmonisation |
| `eval/harvest.py` | API client |
| `eval/scripts/00..05_*.py` | Numbered pipeline steps (smoke → survey → probe → build → mine) |

---

## Next session

1. Run `pipeline/` against `eval_v1.parquet` and compute macro-F1 + per-class P/R + binary F1.
2. Decide whether Supported recall is actually the weakness driving overall macro-F1 down.
3. If yes → revisit correction mining (scale to ~1,138). If no → focus on whichever class drops macro-F1 (likely CE, given pipeline's misleadingness detection step).
