# Claim Extraction (Stage-2) — Results & Decisions

Stage-2 of the pipeline: from a social-media post (text + image(s) + quoted tweet), extract the
check-worthy claims for downstream verification, **and** flag whether the post also needs its
**attribution** or its media's **authenticity** verified. This doc is the running record of the
important eval results. Code: `eval/scripts/stage2_eval.py`. Prompt research: `docs/extraction_prompt_sota.md`.

## Dataset — core pool → Stage-2 eval subset

`dataset_12mo.parquet` is the **CORE** pool (every 12-month fact-check we could harvest — grab as much as
possible). Each stage derives its OWN eval subset with documented inclusion rules; you can only evaluate
**claim extraction** on rows that **have a post to extract from**. The Stage-2 subset is built by
`build_stage2_dataset.py` → `stage2_dataset.parquet` (frozen `md5(review_url)<0.4` dev/test, pool-independent):

| step | reason | drop | remaining |
|---|---|--:|--:|
| core fact-checks | — | — | **6,582** |
| − missing gold `claim_text` | need a judge target | 0 | 6,582 |
| − **empty-input** (no post text AND no image) | nothing to extract — deleted/login-walled/API-only-no-source posts where only the gold survives (**70% of the core**; clog 270626) | 4,629 | 1,953 |
| − video (`has_video`) | extractor can't parse video yet | 718 | **1,235  ← Stage-2 set** |

Stage-2 set = **1,235 rows** (dev 498 / test 737). Axes: content 679 / artifact 343 / attribution 213.
Image 927 / text-only ~308. **Heavily EN (FR 47, ES 9)** — a known coverage limit inherited from the harvest.
*Earlier results (2026-06-26) used the older image-native subset (`image_audit` ∩ harvests) — not directly
comparable. The 2×2 below ran on a non-representative first-100 slice of the contaminated core and needs a
clean re-run on `stage2_dataset`.*
- **Extractor:** Qwen3-VL (30B-A3B or 235B-A22B), temp 0, `json_object`, **no `max_tokens`** (Qwen
  truncates mid-string in JSON-mode), `finish_reason`/tokens recorded.
- **Judges (both independent of the Qwen extractor):**
  - **Claim coverage** — DeepSeek-V4-Flash, text-only, framing- & combination-aware → match/partial/miss
    vs the gold `claim_text`. Cross-validated vs Gemma-3-27B = **89% match-vs-not, κ 0.77**.
  - **Serialization fidelity** — **Llama-4-Maverick** (primary, *sees* the image) + **Gemma-3-27B**
    (cross-check) → faithful/partial/poor + missing + hallucinated. Matters because the downstream
    retrieval model is **text-only/blind** → the serialization *is* the image to it.
- **Metrics:** extraction match/partial/miss (sliced **image-locus vs text-locus** via `claim_location`
  from the Stage-1 audit); serialization fidelity; flags = **recall floor** vs `judged_axis` (the label is
  ONE axis the checker chose, NOT exhaustive — so precision is by adjudication, not this label).
- **Scope:** **empty-input rows excluded** — no recovered post text AND no image (deleted/failed/no-source
  posts where only the gold exists). Nothing to extract → guaranteed miss; not an extraction test (~22% of
  the raw dev set).

## Results timeline

### 2026-06-26 — Locked single-mode baseline (image-native subset)
- Single-primary prompt (extract THE one claim). Held-out test, content+attribution: **30B match 68% /
  235B 72%**. Production pick **30B-A3B** (235B +4pp not worth ~1.5× cost/latency). Authenticity excluded
  → Phase 1b. (Superseded by the redesign below; see [[project_stage2_extraction]].)

### 2026-06-27 — Redesign: combined axis-aware, multi-claim
- **Multi-claim, not single** (a 10-claim tweet must yield 10; single was an eval artifact of the
  one-gold label). Anti-fragmentation = a quality rule ("distinct AND complete"), no count cap.
- **Three axes (claims + attribution flag + authenticity flag) are orthogonal but correlated → ONE
  combined call**, not a separate authenticity classifier.
- `judged_axis` is **non-exhaustive** → flags scored as recall floor + adjudication.
- SOTA-hardened prompt: XML blocks, no persona, reason-before-answer field order, no `max_tokens`, 6 gold
  few-shot (balanced, contrastive negatives, "both fire" case), `image_serialization` detail **scales
  with the image** (full transcription for screenshots, a line for a photo).

### 2026-06-27 — 2×2 slice (n=100 dev, `dataset_12mo`)

Raw (incl. empty-input) vs **HAS-INPUT** (the real extraction test, n=78 after excluding 22 empty-input):

| config | extract (HAS-INPUT) match/part/miss | img-locus match | serialization fidelity | cross-judge agree |
|---|---|---|---|---|
| **30B combined** | **62 / 15 / 23** | 56 | 83 faithful / 17 part / 0 poor | 80% |
| 30B pure | 60 / 19 / 21 | 52 | 86 / 14 / 0 | 85% |
| 235B combined | 55 / 17 / 28 | 54 | **91 / 9 / 0** | 88% |
| 235B pure | 53 / 19 / 28 | 46 | 91 / 9 / 0 | 94% |

**Findings:**
1. **The flags do NOT degrade extraction** — combined ≈ pure (30B 62 vs 60; 235B 55 vs 53). Folding
   attribution + authenticity into the one call is free → **combined-call design validated.**
2. **30B ≥ 235B on extraction** (62 vs 55) — the smaller, cheaper model extracts claims at least as well.
   235B wins **only** on serialization fidelity (91% faithful vs 83–86%).
3. **Serialization fidelity is solid + trustworthy:** 83–91% faithful, **0% poor anywhere**, only ~2/65
   hallucination-flagged, **Maverick–Gemma agreement 80–94%** (not a single-judge artifact). The blind
   downstream gets a faithful rendering ~85–90% of the time; 235B buys ~6–8pp more.
4. **0 truncation** across all 4 configs — the `max_tokens` drop works at scale.
5. **Empty-input artifact:** 22% of the raw dev set has no recoverable input → guaranteed miss; now
   excluded. The scary "text-locus 40%" was this artifact — true text-locus (has-input) match is **70%**.
6. **Eyeball of genuine text misses (4):** ~1 judge error (Mariah Carey — extraction was correct), ~2
   axis-mismatch (gold is sourcing/image-context, extractor got content — reasonable), ~1 genuine miss
   (hantavirus multi-locus). → **true extraction is better than the 62% headline**; motivates the
   adjudication judge.
7. **Flags (recall floor):** authenticity recall 53–70%, attribution 60%; **30B over-fires authenticity**
   (56% of rows vs 235B's 40%). Precision pending the adjudication pass.

*Caveat: n=100, single dev slice — directional, not definitive.*

## Open decisions / next steps
- **Model choice:** 30B (better claims + cheaper) vs 235B (better serialization fidelity, 91 vs 83). The
  blind downstream values fidelity, but 30B's 83% with 0% poor may suffice. **TBD on a larger slice.**
- **Authenticity over-trigger** → adjudication pass + possible negative-rule/few-shot tightening.
- **Adjudication judge** (independent VLM rules whether partial/miss extractions are *reasonable*, catching
  judge-strictness + axis-mismatch) — deferred to the next run; will sharpen the true extraction number.
- **Empty-input rows** — excluded from extraction; could later re-resolve dead sources to recover some.
