# Adding image support to the pipeline — feasibility & cost brief

**Date:** 2026-06-23 · Investigation requested by Daniel. Context: text-only tiered pipeline
(Tier 0 encoders → claim extraction → Tier 1 cache → Tier 2 GFC API → Tier 3 ClaimCheck verify
loop on DeepSeek-V4-Flash/DeepInfra + Serper). **~42% of our eval corpus is media (artifact axis)**,
so image handling is central, not a side feature.

## Bottom line

- **Cheap + high-value → DO FIRST:** VLM OCR / claim-extraction of in-image text **+** reverse-image
  search for out-of-context/miscaptioning. Both **reuse the existing Tier-3 evidence loop** and add
  **~$0.0007/post amortized** (well under one cent).
- **Expensive/low-reliability → DEFER:** pixel-level AI-generated-image detection. Arms race,
  generalizes poorly to unseen generators; buy as a low-confidence advisory flag (Hive $0.006/img)
  only if needed — never gate verdicts on it.
- The dominant real-world image-misinfo case is **a real photo with a false caption**, not synthetic
  pixels (Google AMMeBa, 135,838 claims: context manipulations dominate; AI-gen a minority through
  late 2023). That is exactly the tractable route. [arXiv 2405.11697](https://arxiv.org/html/2405.11697v1)
- **One new dependency:** our **Serper** license has **no Lens/reverse-image endpoint** → the
  provenance route needs a new client (**Google Cloud Vision Web Detection**, $3.50/1k + 1k free,
  or SerpApi Lens). The VLM itself runs on **DeepInfra (our existing provider)** — no new billing.

## Per-image VLM cost (input image, ~1MP)

| Model | ≈ $/image | Note |
|---|---|---|
| DeepInfra Gemma-4-26B | ~$0.0002 | cheapest; our provider |
| DeepInfra Qwen3-VL-30B | ~$0.0004 | best open OCR/$ |
| Gemini 2.5 Flash-Lite | ~$0.0001 | cheapest frontier |
| GPT-4.1-mini | ~$0.00066 | |
| Claude Haiku 4.5 | ~$0.0013 | `⌈w/28⌉×⌈h/28⌉` patch tokens, cap 1568 tok/1568px |
| Claude Sonnet 4.6 | ~$0.0039 | |
| Claude Opus 4.8 | ~$0.0065 | cap 4784 tok/2576px (hi-res); overkill for triage |

The VLM pass is ~free per image; **the cost driver is the reverse-image-search API, not the model.**

## Three capabilities

1. **Image authenticity (AI-gen/deepfake/manipulated)** — Hive ($0.006/img, >98% in-distribution),
   Sightengine, Sensity/Reality Defender (enterprise); open HF detectors (free, low accuracy).
   **Maturity LOW / adversarial:** OOD collapse (one detector 97.6%→28.4% on an unseen generator;
   [2605.24906](https://arxiv.org/html/2605.24906v2)), "nearly worthless under slight degradation"
   ([2604.24163](https://arxiv.org/abs/2604.24163)), open detectors ~60% real-world acc / 17% FPR.
   C2PA/Content Credentials gaining adoption (Pixel 10, Samsung S25, camera makers) but **platforms
   strip metadata on upload → absence proves nothing**; use present credentials as confirmation only.
   → **HIGH difficulty, LOW reliability → DEFER** (advisory flag at most).
2. **Claims IN the image (OCR + extraction)** — one VLM call (Qwen3-VL best OCR / Gemma-4 cheapest)
   transcribes screenshots/charts/memes and emits claims in our **existing extraction schema**.
   Good on legible text; weaker on dense/stylized/multilingual layout (OCRBench v2:
   [2501.00321](https://arxiv.org/abs/2501.00321)). ~$0.0002–0.0005/img. → **LOW–MED difficulty,
   HIGH reuse → DO FIRST.**
3. **Caption ↔ image (out-of-context/miscaptioning)** — VLM caption-consistency + **reverse-image
   provenance** (the real workhorse for "real photo, false caption"). Zero-shot VLM OOC is weak
   (~51% F1, [MMFakeBench 2406.08772](https://arxiv.org/html/2406.08772v3)); evidence-augmented
   reaches ~90% on NewsCLIPpings — gains come from **retrieval, not pixels**, exactly what Tier-3
   already does. RIS pricing: **Google Vision Web Detection $3.50/1k (1k free)**, TinEye $40→$10/1k,
   SerpApi Lens $25→$9/1k; **Bing Visual Search retired Aug 2025**. → **MED difficulty, HIGHEST
   value → DO FIRST.**

## Cost-delta (text-only → +images)

Assumptions: 40% of posts carry an image; each image post gets one VLM pass; ~25% of image posts
trigger one reverse-image search; AI-gen detector deferred.

| Component | unit | applied to | $/post (amortized) |
|---|---|---|---|
| Multimodal VLM pass (DeepInfra) | ~$0.0005 | 40% | $0.0002 |
| Reverse-image search (Google Vision) | ~$0.004 | 40%×25% | $0.0005 |
| **Phase-1 delta** | | | **≈ $0.0007/post (~0.07¢)** |
| *(Phase 2) AI-gen detector (Hive)* | $0.006 | 40%×25% | +$0.0006 |

Per image-bearing, check-worthy post the full Phase-1 path is ~$0.005–0.006, dominated by the RIS
API call (VLM ~10%). **Delta is negligible vs. value; only material new spend = the RIS API.**

## Engineering: reuse vs net-new

**Reused:** the whole **Tier-3 loop** (image claims + reverse-image results feed the same
search→summarize→synthesize→verdict path); **same DeepInfra provider** (images bill as input
tokens, no per-image fee); claim schema + Tier-1/2 cache/lookup.
**Net-new:** (1) Tier-0 image triage (claim-bearing media vs decorative); (2) a multimodal
extraction stage (one VLM call → {in-image claims, caption-consistency flag, provenance query});
(3) a reverse-image-search client (new API; Serper can't do it); (4) *deferred* AI-gen detector.
**Hardest:** text+image claim fan-in/dedup; **verdict attribution** (true-text + false-image
miscaption → which proposition is judged — extends our `judged_axis`/attribution-eval work);
provenance is probabilistic → needs a graceful "insufficient evidence" path.

## Phased recommendation

- **Phase 1 (~0.07¢/post):** VLM OCR + in-image claims (Cap. 2) and VLM caption-consistency +
  reverse-image provenance (Cap. 3), on DeepInfra + Google Vision Web Detection. Handles the
  **majority** of media misinfo; reuses Tier-3.
- **Phase 1.5 (free):** parse C2PA when present; never treat absence as a fake signal.
- **Phase 2 (defer):** pixel-level AI-gen detection (Hive) as a low-confidence advisory only.
- **Out of scope (flag):** video — AMMeBa shows video grew to >60% of media misinfo; separate
  later workstream.

## Maps to our 42% media slice

- **Miscaptioned / out-of-context (largest, most persistent)** → Cap. 3 (reverse-image + VLM).
- **Claims depicted inside the image** (fake screenshots, charts, memes) → Cap. 2 (VLM OCR →
  existing text Tier-3 loop).
- **AI-generated/altered** (smallest, fastest-growing) → Cap. 1 → **deferred** (advisory flag).

Phase 1 covers the **majority** of our 42% media share at sub-cent cost; the least-tractable AI-gen
slice is the sensible punt. **Video remains the main uncovered residual.**

*Sourcing: Anthropic pricing/limits from the `claude-api` skill + Anthropic vision docs; non-Anthropic
pricing & reliability from a 5-angle research fan-out (citations inline). Uncertain: Gemini per-image
token counts vary by aspect ratio; some OpenAI flagship multipliers unpublished; Serper Lens treated
as absent; the "~80% of fact-checks involve media" figure is untraceable — AMMeBa is the anchor.*
