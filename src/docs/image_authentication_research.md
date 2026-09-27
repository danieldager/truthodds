# Image Authentication (Phase 1b) — Research Brief

*Research + design only. No pipeline edits, no paid-API calls were made (pricing/capability pages
were read in June 2026; URLs cited inline). Grounds: our `eval/data/image_audit.parquet` +
`*_harvest.parquet` (counts verified with polars, this brief), `docs/core_pipeline_spec.md`,
`clog/260626.md`. Author session: 2026-06-26.*

---

## 1. Problem framing + our data

**The gap.** Stage-2 (image-native extraction) was just evaluated. There is a class of image
fact-checks the VLM extractor **structurally cannot do**: the gold claim is not about the image's
*content* but about its **authenticity / provenance** — "this photo *authentically shows* X dining
with Y", "this 'leaked letter' is real", "this clip *actually depicts* the 2026 event". A VLM reads
pixels; it has no live web index, so it cannot know "this exact photo first ran on Reuters in 2019."
The Stage-2 eval confirmed this is **retrieval, not perception**: scaling the extractor 30B→235B
moved artifact-axis match only **+4pp** (48→52, still ~50% ceiling), and **74% of artifact
identity-miss rows had the name absent from the post text** (`clog/260626.md`, Gate-2 entry) — the
model can't *guess* who is in the photo, and shouldn't (a hallucinated name is invisible to the
text-only verifier). These are the dominant real-world image-misinfo patterns: **out-of-context
recaptioning**, **digitally altered** images, **AI-generated / deepfakes**, and **fabricated
screenshots / fake "leaked" documents**. We make this its own stage — **Phase 1b** — with its own
eval, separate from extraction.

**Our data (verified this session against the parquets).** `image_audit.parquet` = **1,620** audited
image posts (1,538 legible). `judged_axis`: content 701 / **artifact 658** / attribution 261. The
**artifact axis (658) = the authentication fact-checks** — our ready-made seed. Inside it:

| Cut of the 658 artifact rows | Count | Note |
|---|--:|---|
| Verdict `harmonised_label` = Refuted | 578 | manipulated/false |
| `harmonised_label` = Supported | 71 | **authentic** ("True") |
| Conflicting Evidence / Not Enough Evidence | 5 / 4 | abstain class |
| `binary_label` = flag / pass | 587 / 71 | product target |
| **`has_video` = true** | **333 (51%)** | poster-frame only → mostly **unservable** today |
| `has_video` = false **and** image-dependent (clean image seed) | **301** | the real Phase-1b image seed |
| Language EN / FR / ES | 645 / **12** / 1 | **FR coverage ≈ 0** |
| Has ≥1 fact-checker-cited `sources` URL (median **12**/row) | **657 / 658** | ready-made provenance gold |
| Publisher: leadstories / snopes / fullfact / politifact / 20min | 340 / 212 / 70 / 18 / 12 | |

The fact-checker `original_rating` vocabulary clusters cleanly into a label space (verified, full
distribution): **AI-generated** (~110: "AI Generated/AI-Generated/AI Video/AI Image/AI Made It/
Sora Product/AI Watermark"), **out-of-context/miscaptioned** (~55: "Miscaptioned/Old Photo/Old
Video/predates the attack"), **digitally-altered** (~20: "Edited/Altered/Doctored Image/Photo/
Video"), **fabricated** (~95: "Fake/Fake Report/Fake Image/Not Real"), **authentic** (71: "True/
genuine"), **satire/staged** (~25: "Originated as Satire/Labeled Satire/Staged Skit/Prank Video").

**Three structural facts that shape everything below:**
1. **Half the artifact axis is video** (333/658). We capture only the poster frame
   (`core_pipeline_spec` Decision 4) → Phase-1b serves **images**; the clean image seed is **~301**.
   Video authentication is out of scope for v1 (consistent with the existing video-drop policy).
2. **The gate's own `image_role=authenticity_subject` tag (266) under-triggers the artifact axis** —
   of the 658 artifact rows, the gate tagged only 216 `authenticity_subject` and **425
   `carries_claim`**. So `judged_axis=artifact` (a post-hoc harmonization tag, not available at
   inference) is the clean *eval* seed, but the *production* trigger (the gate's perceptual
   `image_role`) needs a sharpened "is the claim about whether this media is real / where it came
   from" signal to fire Phase-1b reliably. (Open question 5.)
3. **Contamination is already characterized** (`clog/260626.md` 10:40 + 13:30): the *input* images are
   clean — we use the post's own `x_media` image (0 contamination), **not** the fact-checker article
   `og:image` (which bakes the verdict onto the picture: Lead Stories' "Old Video"/"2025 Fire"
   annotations + branding = "verdict leakage", validation-failed and dropped). The *new* contamination
   surface in Phase-1b is at retrieval time (§4).

---

## 2. Approaches surveyed

**Comparison (coverage = what fraction of our patterns it addresses; maturity; cost at our scale; fit).**

| Approach | Addresses | Maturity | Cost (our scale) | Fit |
|---|---|---|---|---|
| **Reverse-image search / web-detection** (Google Vision Web Detection, SerpApi Lens, TinEye) | OOC recaptioning, identity gap, earliest-source provenance, reused-stock fabrications | High (productized APIs) | **$0.0035–0.025 / image** | **Primary tier** — directly closes our retrieval + identity gap |
| **C2PA / Content Credentials** | AI-gen + edit provenance *when present* | High standard, **low real-world coverage** (platforms strip) | ~free (local parse) | **Tier-0 positive-only** signal |
| **OOC consistency (caption↔image / caption↔evidence)** — CCN, SNIFFER, EXCLAIM | OOC recaptioning (the judgment layer over RIS) | Research-mature; **shortcut-prone** | ~$0.001–0.002 / image (reuses our VLM) | **Tier-2 reasoning** over RIS evidence |
| **AI-gen / deepfake detectors** (Hive, Sensity, Reality Defender, open models) | AI-generated / deepfake | **Brittle in the wild** (arms race) | $0.006 / image (Hive) | **Advisory flag only / defer** — validated below |
| **VLM-only "is this real?"** (our current extractor) | (none reliably) | — | (already paid) | **Baseline to beat** (+4pp 30B→235B proves it can't) |

### Reverse-image search / provenance APIs — the core of the solution
- **Google Cloud Vision — `WEB_DETECTION`.** Returns `webEntities` (named entities/topics **with
  scores** — e.g. an event or person), `fullMatchingImages` (exact copies), `partialMatchingImages`
  (crops/edits), `pagesWithMatchingImages` (**URLs + page titles**), `visuallySimilarImages`,
  `bestGuessLabels`. **$3.50 per 1,000 units; first 1,000 units/month free**
  ([cloud.google.com/vision/pricing](https://cloud.google.com/vision/pricing), read 2026-06-26).
  **No earliest-appearance date** — you infer recency from the matched pages yourself
  ([docs.cloud.google.com/vision/docs/detecting-web](https://docs.cloud.google.com/vision/docs/detecting-web)).
  `webEntities` + `bestGuessLabels` are the most direct fix for the **implicit-identity gap** the
  extraction eval exposed (the web index supplies the name the VLM had to guess) — but they are
  *scored guesses* (see §5 hallucination caveat). **Cheapest, one call, richest structured fields → start here.**
- **SerpApi — Google Lens API.** Returns visual matches, exact-match URLs with source pages/titles,
  and knowledge-graph-style entities (often stronger identity surfacing than Vision). Plans: **Starter
  $25/mo = 1,000 searches ($0.025/search) → Developer $75/5K ($0.015) → Production $150/15K ($0.010) →
  Big Data $275/30K (~$0.009)**, 100 free/mo, cached searches free
  ([serpapi.com/pricing](https://serpapi.com/pricing), 2026-06-26). 3–7× Vision's price; use as a
  **fallback / identity specialist**, not the default.
- **TinEye API.** The **only one of the three that returns an earliest-crawl date** ("sort: oldest")
  plus backlinks and match count — the single most direct **out-of-context signal** (was this exact
  image online *before* the claimed event?). Prepaid bundles **$200/5K ($0.04) · $300/10K ($0.03) ·
  $1,000/50K ($0.02) · $10,000/1M ($0.01)**, valid 2 years
  ([help.tineye.com/article/169](https://help.tineye.com/article/169-purchasing-search-bundles),
  2026-06-26). Smaller index than Google, weak on entity names → **OOC-date specialist**, fired
  selectively.
- **Bing Visual Search — RETIRED.** All Bing Search APIs (incl. Visual Search) were **decommissioned
  2025-08-11; endpoints now return HTTP 410**
  ([learn.microsoft.com/lifecycle/.../bing-search-api-retirement](https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement)).
  Replacement = "Grounding with Bing Search" inside Azure AI Agents (no raw SERP JSON, 40–483% pricier
  — [theregister.com 2025-05-15](https://www.theregister.com/2025/05/15/bing_search_apis_retired/)).
  **Do not design around Bing.**
- **Newer entrants (2026).** Bright Data, SearchApi.io, Zenserp, DataForSEO, Oxylabs all resell
  **Google Lens** scraping at SerpApi-class prices; none adds a provenance capability Vision/Lens/TinEye
  lack. No dedicated "image-provenance" SaaS beyond these surfaced. *Not load-bearing for v1.*

### C2PA / Content Credentials — positive-signal-only (confirmed)
Cryptographically signed provenance manifest in the file; steering committee now includes Adobe,
Google, **Meta, OpenAI**, Sony, Truepic; CAI > 6,000 members
([contentauthenticity.org/.../state-of-content-authenticity-in-2026](https://contentauthenticity.org/blog/the-state-of-content-authenticity-in-2026)).
2026 coverage is real on the *creation* side (OpenAI DALL-E/GPT-image + Sora, Adobe Firefly now
*mandatory*, Google Imagen/Gemini, Microsoft, FLUX.2; Leica/Sony/Nikon/Canon news cameras) — **but
social platforms strip it**: a screenshot or re-upload removes the manifest, and only TikTok is
documented to *preserve* its own credentials on download; X/Twitter is reported to strip on re-encode
([truescreen.io/.../c2pa](https://truescreen.io/articles/c2pa-standard-history-limitations);
[indicator.media audit, Oct 2025](https://indicator.media/p/tech-platforms-fail-to-label-ai-content-c2pa-metadata):
only ~30% of AI posts correctly labeled). And C2PA certifies *provenance, not truth* — a staged real
photo can carry a valid manifest. **Conclusion: Tier-0, positive-only.** Present manifest = exploit as
an origin/AI-use signal (then still verify the assertion); **absent = no information**, never treat
absence as authenticity.

### Out-of-context detection (literature) — the Tier-2 reasoning layer
The closest analogs to our design already exist and validate "RIS + consistency":
- **CCN** (Abdelnabi et al., CVPR 2022) — *exactly* our pattern: reverse-image-search for visual
  evidence + caption→text evidence, then cross-modal consistency. **84.7%** on NewsCLIPpings; released
  the retrieved web-evidence ([arxiv.org/abs/2112.00061](https://arxiv.org/abs/2112.00061)).
- **SNIFFER** (CVPR 2024) — adds an MLLM explanation layer (InstructBLIP) over RIS + Google entity
  detection: **88.4%** ([arxiv.org/abs/2403.03170](https://arxiv.org/abs/2403.03170)).
- **EXCLAIM** (2025) — multi-agent hierarchical retrieval, current SOTA **92.7%**
  ([arxiv.org/abs/2504.06269](https://arxiv.org/abs/2504.06269)).
- **COSMOS** (AAAI 2023) — self-supervised same-image / two-contradictory-captions test, ~85%, MIT
  ([arxiv.org/abs/2101.06278](https://arxiv.org/abs/2101.06278)).
- **Caveat — "Similarity over Factuality"** (WACV 2025): a deliberately simple baseline matches
  evidence-based SOTA, arguing these models exploit *similarity shortcuts, not factuality*
  ([arxiv.org/abs/2407.13488](https://arxiv.org/abs/2407.13488)). Treat the 85–93% headlines
  skeptically; they are on **synthetic** mismatches.
- Zero-shot GPT-4V/LLaVA are weak (~51% F1) — **a bigger VLM alone does not solve OOC**, corroborating
  our +4pp finding. We do **not** need to train a model: our Tier-2 is "feed RIS evidence (earliest
  page, entities, dates) + the post caption to the existing DeepSeek/Qwen reasoner and judge
  consistency" — the CCN recipe with our infra.

### AI-generation / deepfake detectors — defer (advisory only). Our stance is VALIDATED.
- **Independent in-the-wild benchmark Deepfake-Eval-2024**: tested Hive, Reality Defender, Sensity et
  al.; **no commercial model reached ≥90%** real-world accuracy (best image 0.82)
  ([arxiv.org/abs/2503.02857](https://arxiv.org/abs/2503.02857)).
- **Arms race**: on modern generators (Flux, Firefly v4, Midjourney v7, Imagen 4) mean zero-shot
  detection runs **18–31%, below random**; accuracy decays year-on-year as generators improve
  ([arxiv.org/abs/2602.07814](https://arxiv.org/html/2602.07814v1), moderate confidence). Trivial
  JPEG recompression / resizing (what every platform does on upload) costs 10–20 points; best open
  cross-generator generalizer (**SAFE**, Apache-2.0) averages only ~80% on unseen sets
  ([arxiv.org/abs/2505.12335](https://arxiv.org/abs/2505.12335)).
- APIs: **Hive** has the clearest pricing (**$6/1,000 images**, attributes the engine; 98% on one
  independent *image* study — [thehive.ai/pricing](https://thehive.ai/pricing),
  [arxiv.org/abs/2402.03214](https://arxiv.org/html/2402.03214v3)). **Sensity** (enterprise, no public
  price, unverified 98% claim) and **Reality Defender** (free 50 scans/mo, no published accuracy) are
  enterprise-gated.
- **SynthID** (Google) only flags *participating Google models* → absence proves nothing.
- **Decision: defer to an advisory flag.** Use Hive (or an open model) as a **low-confidence
  "possibly AI-generated" signal that softens a nudge, never a verdict.** The research says no
  detector is trustworthy enough to be load-bearing, and our own data shows AI-generation is only ~1/6
  of the artifact axis while OOC+identity (the RIS-solvable part) dominates.

---

## 3. Recommended architecture — tiered cascade

> Trigger Phase-1b **only** on images the gate flags as authenticity-relevant (today:
> `image_role=authenticity_subject`; see Open Q5 on widening). Everything below runs on that flagged
> minority, after an **image-hash (pHash) cache** lookup so a viral image is authenticated once and
> reused (misinfo images recur — dedup is real savings).

```
flagged authenticity image
  │
  ▼  pHash cache hit? ── yes → reuse prior verdict (≈ free)
  │
[Tier 0]  C2PA manifest parse  +  pHash lookup            (~free, local)
        └ valid manifest present → record as positive origin/AI signal (don't stop; still verify)
  │
[Tier 1]  REVERSE-IMAGE  — Google Vision WEB_DETECTION     ($0.0035 / image; first 1k/mo free)
        in:  the post image
        out: webEntities (names/topics), pagesWithMatchingImages (+titles), full/partialMatches, bestGuessLabels
        └ resolves identity gap + surfaces prior appearances (provenance)
  │     ┌ if OOC suspected (matches predate the claim, or caption is event-anchored):
  │     └→ TinEye "sort: oldest"  ($0.01–0.04) for the earliest-crawl DATE  ── selective
  │
[Tier 2]  OOC CONSISTENCY  — existing DeepSeek/Qwen reasoner over Tier-1 evidence   (~$0.002)
        judge: does the post caption match the earliest/authoritative appearance? assign a label (§4)
  │
[Tier 3]  AI-GEN FLAG (advisory)  — Hive or open SAFE, ONLY if Tier-1 found no provenance AND image looks synthetic
        $0.006 / image × ~25% fire-rate → soft "possibly AI-generated" flag, never a standalone verdict
  │
  ▼ label → nudge  (authentic→pass · OOC/altered/AI/fabricated→flag · no-provenance→soft/abstain)
```

**Which provider to start with: Google Cloud Vision Web Detection.** Cheapest ($0.0035, free first
1k/mo), one call, returns the two things we lack (identity via `webEntities`, prior appearances via
`pagesWithMatchingImages`). Add **TinEye selectively** for the earliest-date OOC signal Vision can't
give. Hold SerpApi Lens as an identity fallback. Skip Bing (retired).

**Cost model.** *Scale assumption:* a pilot extension serving a modest user base flags on the order
of **1,000–10,000 authenticity images/month**; viral-image pHash dedup removes ~half → **~500–5,000
unique images/month** actually hit the APIs. Blended per **unique** image: Tier-0 ~$0; Tier-1 Vision
$0.0035 (≈100%) + selective TinEye ~$0.02×50% = ~$0.013; Tier-2 ~$0.002; Tier-3 $0.006×25% = ~$0.0015
→ **≈ $0.018 / unique authenticity image.**

| Unique authenticity images / mo | Phase-1b marginal cost |
|---|--:|
| 500 | **~$9 / mo** |
| 2,500 | **~$45 / mo** |
| 5,000 | **~$90 / mo** |

**Headline: ~$10–90/month at pilot scale** (1k–10k flagged → 0.5k–5k unique), dominated by Tier-1
RIS; dropping TinEye to Vision-only roughly halves it. The **one-time eval** over the ~301-image seed
fits inside (or just past) Vision's 1,000-free-units/month tier → **a few dollars total** (≈$2 of Hive
on the ~300 suspected-synthetic), ideal for a cautious smoke→survey→full rollout.

---

## 4. Evaluation plan

**Gold construction from our data (no new labeling).** The artifact-axis fact-checks already carry the
authenticity ruling. Build the gold by mapping each row's `original_rating` → a **label space**, with
`harmonised_label` as the backstop:

| Phase-1b label | Maps from (rating keywords) | `harmonised` backstop | Nudge |
|---|---|---|---|
| **authentic** | "True / genuine / correct / real" | Supported | pass |
| **out-of-context / miscaptioned** | "Miscaptioned / Old Photo / Old Video / predates" | Refuted | flag |
| **digitally-altered** | "Edited / Altered / Doctored (Image/Photo/Video)" | Refuted | flag |
| **AI-generated** | "AI \* / Sora / AI Video / AI Watermark" | Refuted | flag |
| **fabricated screenshot/document** | "Fake / Fake Report / Fake Image / Not Real" | Refuted | flag |
| **satire / staged** | "Originated as Satire / Labeled Satire / Staged Skit / Prank" | Refuted | flag (satire nudge) |
| **unverifiable — no provenance** | "Unverified" | NEE / Conflicting Evidence | soft / abstain |

Implementation: a deterministic regex `rating → label` map over the verified 302-value vocab, spot-checked
by hand (cluster boundaries are clean). The **clean image seed = the 301 non-video, image-dependent
artifact rows**; report EN headline on it (FR is 12 rows → §5).

**Contamination control (verdict-leakage) — critical.** When we run RIS, it will surface the
fact-checker's *own* article (whose title/og-image states the verdict, often with the annotated
image). If the Tier-2 judge reads those pages, it reads the answer.
- **Reuse `pipeline/config.py :: FACT_CHECK_DOMAINS`** (already an eval-only retrieval exclusion of our
  9 publishers + major EN/FR checkers) and **broaden it to all IFCN signatories** — drop every matched
  page/`fullMatchingImage` whose host is a fact-checker before the judge sees evidence.
- Input image stays the clean post image (`x_media`), **never** the fc-article `og:image` (already the
  policy — 0 contamination measured).
- Audit: log the share of RIS hits that were FC-domain (expected high for viral debunked images) as a
  leakage tell.

**Metrics.**
- **Provenance recall** — fraction of OOC/altered cases where Tier-1 surfaced ≥1 page matching a
  fact-checker-cited `sources` URL/domain (median 12/row, **657/658 rows have them**, FC-domains
  excluded). This is *directly measurable from our data* — the single most important Phase-1b metric.
- **Earliest-date accuracy** (OOC) — did TinEye/RIS return a match dated *before* the claimed event?
- **Per-class F1** over the 7-label space; **binary manipulated-vs-authentic** accuracy.
- **Abstention / coverage** — share routed to "no-provenance/unverifiable" and precision on the rest
  (mirrors the verifier's abstain discipline).
- **Product metric — right-nudge rate**: label→nudge vs `binary_label` (flag/pass), the headline,
  analogous to the verifier's existing "binary nudge accuracy."
- **Identity-resolution lift** — with vs without `webEntities` names fed to the judge: does RIS close
  the implicit-identity gap (74% of artifact identity-misses had no in-text name)?

**External benchmarks to borrow (calibration only).** Real fact-check sets (preferred):
**VERITE** (1,000 pairs, Snopes/Reuters, Apache-2.0 — [arxiv.org/abs/2304.14133](https://arxiv.org/abs/2304.14133)),
**MOCHEG** (15,601 claims, PolitiFact/Snopes, CC-BY-4.0), **5Pils** (1,676 fact-checked images,
CC-BY-SA-4.0, has source/date/location meta — ideal for provenance-recall calibration). Synthetic
(upper-bound / training only, with the shortcut caveat): **NewsCLIPpings** (71k, license inherits
VisualNews non-commercial), **Fakeddit** (1M, distant-supervised 6-way incl. manipulated/false-connection).
Deepfake benches only to calibrate the advisory Tier-3 with the in-domain-vs-wild caveat: GenImage,
DFDC, FaceForensics++ (FF++ non-commercial + MIT code).

**Baselines + ablations.**
1. **VLM-only** ("is this image authentic / out-of-context?" to Qwen3-VL) — the do-nothing baseline;
   we expect it near the +4pp ceiling. *Quantifies the value Phase-1b adds.*
2. **Tier-1 only** (RIS webEntities + pages → judge).
3. **+ Tier-2** OOC consistency.
4. **+ Tier-3** advisory deepfake flag (measure precision/recall of the flag in isolation; expect
   poor — that's the point of "advisory").
5. **Identity ablation** (§ metric above) and **TinEye-date ablation** (does the earliest-date signal
   improve OOC F1 enough to justify $0.02/call?).

---

## 5. Risks & caveats

- **Deepfake arms race.** No detector is reliable in the wild (≤82% best image, <random on 2026
  generators); recompression/crops defeat them. → advisory-only, never load-bearing (validated above).
- **RIS misses on crops/heavy edits.** `partialMatchingImages` helps but aggressive edits/AI-regen
  evade exact match. → don't treat a Tier-1 *miss* as "fabricated."
- **Absence ≠ proof.** A genuinely *new authentic* photo also has no prior web matches. No-provenance
  must route to **soft/abstain**, not "flag," to avoid false nudges on real first-publication images.
- **Hallucinated names the text-only judge can't catch.** `webEntities`/`bestGuessLabels` are *scored
  guesses*; a wrong-but-confident name fed downstream is invisible to the text verifier (same risk the
  extraction eval flagged). → confidence-threshold `webEntities`, carry a `flags[]` for low-confidence
  identity, never assert an unconfirmed name into a verdict.
- **FR coverage.** Our artifact data is **12 FR rows**; Google's web index + entity names skew EN, and
  we can't *measure* FR Phase-1b on this data. → flag as a known gap; FR eval needs FR authenticity
  sources (France24 Observers, AFP Factuel, 20min) harvested first.
- **Contamination / verdict-leakage.** Handled by FACT_CHECK_DOMAINS exclusion + clean post images
  (§4); must be enforced or all metrics inflate.
- **Video.** Half the artifact axis is video; poster-frame-only. Phase-1b = images; video
  authentication deferred with the existing video-drop policy.
- **Trigger recall.** The gate's `authenticity_subject` tag (266) under-covers the true artifact
  population (658) → some authenticity cases won't reach Phase-1b until the Stage-1 prompt gains an
  explicit "is the claim about whether this media is real / where it came from" locus.

---

## 6. Open questions for Daniel + rollout

**Open questions**
1. **Provider to start with** — recommend **Google Vision Web Detection** ($0.0035, free 1k/mo,
   webEntities + pages) as primary + **TinEye** selectively for earliest-date OOC. OK to provision a
   GCP Vision key (and a small TinEye bundle), or Vision-only for v1 (halves cost, loses the explicit
   OOC date)?
2. **Deepfake tier** — confirm **advisory-flag-only / defer** (research supports it), or omit Tier-3
   entirely for v1?
3. **Label space** — 7 classes as proposed; keep **satire/staged** as its own class, and treat
   **video-authenticity** rows as out-of-scope (drop) vs poster-frame attempt?
4. **Eval scope** — build the **~301-image EN seed now**; accept **EN-only** Phase-1b for v1 (FR ≈ 0),
   or harvest FR authenticity cases first?
5. **Production trigger** — fire Phase-1b on the gate's high-precision `image_role=authenticity_subject`
   (266), or widen the Stage-1 "media-authenticity locus" signal toward the full artifact population
   (658, higher recall, more cost)?

**Phased rollout (smoke → survey → full; never burst a paid API).**
- **Smoke (1 call):** one Google Vision `WEB_DETECTION` request on one known-OOC seed image; verify
  `webEntities` names + `pagesWithMatchingImages` come back and that FACT_CHECK_DOMAIN exclusion fires.
- **Survey (~30–50 images):** a stratified slice across the 7 labels; hand-check **provenance recall**
  against the `sources` gold and confirm **no verdict-leakage** leaks through; decide if TinEye's
  earliest-date earns its $0.02.
- **Full (~301 seed):** run the cascade, compute the metrics + tier ablations (VLM-only → Tier-1 →
  +Tier-2 → +Tier-3), read the cost/quality knee, then wire Phase-1b in between Stage-2 and Stage-3.
