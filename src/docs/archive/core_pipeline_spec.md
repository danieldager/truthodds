# Core Pipeline Spec — VLM gate → normalize+serialize → verify  (FINAL v1)

*Final 2026-06-26 (supersedes the same-day draft). Reconciles the draft with the image-harvest +
gate-validation work — see `src/clog/260626.md`. Decisions below are LOCKED with Daniel.*

## What this is

A browser extension on X/Twitter intercepts batches of posts (read off the network response as the
user scrolls), fact-checks them, and nudges the user before they share likely-false/misleading
content. This is the **front of the pipeline**: a **two-stage VLM cascade** feeding the **existing
text-only verifier**. Design principle: lean on a capable VLM to collapse steps, but **split work by
stage/volume, not by modality** — a cheap gate over *all* posts, an expensive normalize over the
*flagged minority* only.

## Scope

- **BUILD (new):** Stage 1 (gate), Stage 2 (normalize + serialize), plus two data extensions
  (quote-tweet capture, video tagging — below).
- **CONSUME, do not rebuild:** Stage 3 verification — `src/pipeline/verify.py` (ClaimCheck-faithful
  Serper→Exa retrieve-then-reason, text-only, DeepSeek-V4-Flash). Your job is to feed it.
- **OUT OF SCOPE:** the cache / embeddings / previously-fact-checked layer (`docs/cache_design.md`),
  Tier-1/Tier-2 lookups, the retired encoder ensemble, and **video understanding** (we capture one
  frame only; video-borne claims are dropped — see Decision 4).

## Architecture (final)

```
intercepted posts (batched off the scroll buffer)
   │
   ▼
[1] GATE — Qwen3-VL-30B-A3B, runs on ALL posts
        in:  main post text + image(s)  +  quoted tweet text + image(s)  (+ reply/thread later)
        out: { reasoning, public_interest, misinfo_if_false, claim_locus, check_worthy }
        ── !check_worthy  → DROP
        ── claim_locus == "video" (claim lives in a video; we have 1 frame) → DROP (unservable)
   ▼ flagged minority
[2] NORMALIZE + SERIALIZE — Qwen3-VL-235B, flagged posts only (re-ingests the image)
        atomic decontextualized claims  +  text serialization of all image content  +  attribution split
   ▼ per claim
[3] VERIFY — src/pipeline/verify.py  (EXISTING, text-only, DeepSeek-V4-Flash over Serper→Exa)
        verify(AtomicClaim, post_context=...) → ClaimVerdict  (includes the post-level FLAG)
```

## Input object (per intercepted post)

```
{ post_text, images: [path|url|bytes], has_video: bool,
  quoted: { post_text, images, has_video } | null,   # quote-tweet (primary)
  reply_parent: {...} | null, thread_root: { post_text } | null }   # later
```
Image presence is free from the payload — but **do not route on it** (Decision 1).

---

## Stage 1 — GATE  (Qwen3-VL-30B-A3B, all posts)

- **Model:** `Qwen/Qwen3-VL-30B-A3B-Instruct` (DeepInfra, OpenAI-compatible, vision via `image_url`
  blocks). MoE → ~3B active, so it runs at ~8B-class cost/latency but stronger; **cheaper than the 8B
  dense on DeepInfra** ($0.15/$0.60 per M tok). Bake-off `Qwen3-VL-4B` ($0.10/$0.60) as a cheap-gate
  probe (the gate task was *validated at 30B*; 4B is unproven — see "what to measure").
- Runs on **100% of posts**, text-only included (a VLM bills image tokens only when an image is
  present, so text-only posts cost the same as a text LLM — Decision 1). Lean; prompt-cache the rubric.
- **Multimodal layout:** image(s) first, each labeled ("Image 1: main post", "Image 2: quoted-tweet
  image"); then labeled text blocks (main post, then quoted tweet); a short note on the **quote-tweet
  dynamic** (the user may be endorsing/amplifying/rebutting the quoted post); instruction last; be
  directive about reading in-image/OCR'd text and resolving "this/they/the parent" before deciding.
- **Structured output, reasoning-first:**

```jsonc
{
  "reasoning": "what the post (main+quoted, text+image) asserts; resolve references, the quoted post, and in-image text BEFORE deciding",
  "public_interest": true,        // politics/economics/society/current events/global affairs?
  "misinfo_if_false": true,       // CONSEQUENTIAL misinformation IF false? (judges consequence, NOT truth)
  "claim_locus": "image",         // enum: text | image | both | parent | video | none
  "check_worthy": true            // = public_interest AND misinfo_if_false
}
```
- **DROP** if `!check_worthy`. **DROP** if `claim_locus == "video"` (the checkable claim is in a
  video we can't process yet — gate is told `has_video`, and only the poster frame is available).
- **VALIDATED:** a 30B VLM does the perceptual `claim_locus` call reliably — on 1,620 image posts:
  **82% image-dependent, 95% legible, 0 contamination**; independent cross-check (Gemma-3-27B +
  Llama-4) agreed 83% on the *hardest* cases, **0 disagreement on artifact**, and showed the gate is
  *conservative* (under-calls image-dependence). Prototype: `eval/scripts/image_audit.py` (the
  scope booleans `public_interest`/`misinfo_if_false` are the part still to add + measure).

## Stage 2 — NORMALIZE + SERIALIZE  (Qwen3-VL-235B, flagged only)

- **Model:** `Qwen/Qwen3-VL-235B-A22B-Instruct` ($0.20/$0.88 per M tok). **Re-ingests the image(s)**
  (don't rely on the gate's reasoning text). Bake-off 30B-A3B for cost.
- **Claimify-style Selection → Disambiguation → Decomposition** over the *full* context (main +
  quoted + images): atomic, **decontextualized**, single-proposition claims (entities resolved, no
  "this/they"). Drop anything not disambiguable with high confidence.
- **Quote-tweet rule (TEST EMPIRICALLY — Daniel's hunch, 2026-06-26):** the unit fact-checked is
  **what the user puts forward by posting (main + quoted together)**. Extract a claim *from the
  quoted tweet* **only when it bears on / is endorsed by the main post** — amplification ("so true
  👇"), or a main-post claim *about* the quoted content. Otherwise the quoted tweet (+ its image) is
  **CONTEXT ONLY** for extraction and verification, not a claim source. The gate's `claim_locus`
  ("parent") flags the amplification case.
- **CRITICAL — serialize ALL verification-relevant image content to text**, for BOTH main and quoted
  images. Stage 3 is text-only and never sees pixels, so this is the only place the image becomes
  verifiable. Per image: **verbatim OCR** of in-image text (often *is* the claim); a **descriptive,
  non-interpretive** rendering of non-text visuals (what a photo depicts; chart axes + values);
  apparent **source/handle/watermark** if it's a screenshot.
- **Extractive, never judge truth** — zero-shot VLM description hallucinates, and that would
  propagate into a verdict the text verifier can't cross-check against pixels. Transcribe/describe;
  let Stage 3 judge. (This is the main new risk surface — measure serialization fidelity.)
- **Attribution split:** separate "X *said* Y" (attribution) from "Y" (content) — feeds nudge-safety
  (a true report of a false claim must not be flagged as the user's falsehood).

```jsonc
{
  "image_serialization": "verbatim OCR + descriptive transcript of the main + quoted image(s)",
  "claims": [
    { "normalized_claim": "standalone single verifiable proposition",
      "source_span": "original text / OCR'd image text it came from",
      "locus": "image",                                  // text | image | parent
      "attribution": { "speaker": "Jane Doe" | null, "content": "the claim itself" } }
  ]
}
```

## Stage 3 — VERIFY  (existing; integration only)

`src/pipeline/verify.py :: verify(claim, post_context=...) -> ClaimVerdict` (`AtomicClaim` /
`ClaimVerdict` in `src/pipeline/models.py`). Planning → Serper→Exa → snippet-always evidence →
confidence-gated synthesis → 4-class + Likert → post-level FLAG. **Text-only, DeepSeek-V4-Flash.**
- Feed each `normalized_claim` as `AtomicClaim.text`.
- `post_context = post_text + "\n[image content: " + image_serialization + "]"` (+ quoted context)
  so the post-FLAG step accounts for image- and quote-borne content.
- `AtomicClaim` may need `locus` / `attribution` / `source_span` (extend or sidecar — small).

---

## Key decisions — LOCKED (rationale so they're not undone)

1. **One multimodal gate on ALL posts; NO modality routing.** A VLM doesn't bill for vision it
   doesn't use, so routing text→text-LLM / image→VLM saves ~nothing while costing two models, a
   router, and two boundaries. Split by stage/volume.
2. **Two calls (gate, then normalize), not one.** Packing classify+generate degrades generation; the
   cascade runs the expensive normalize on the flagged minority; and it lets us measure gate
   precision/recall separately from extraction faithfulness. **Confirmed two-call.**
3. **Models: Qwen3-VL family** (not 2.5). Gate **30B-A3B**, normalize **235B**. Accuracy run first;
   **latency + $/post analysis AFTER**. 30B-A3B dominates the 8B (cheaper + stronger, MoE 3B-active);
   4B is the cheap-gate probe.
4. **Video: tag + drop, don't process.** Tag `media_type` from the syndication `mediaDetails.type`
   (photo/video/gif). Posts whose checkable claim is in a video are **DROPPED** by the gate
   (`claim_locus="video"`); **excluded from the eval run, skipped in production** until video support
   exists. (We only ever captured the poster frame.)
5. **Quote-tweet: harvest + feed both, extract conditionally.** Capture the quoted tweet (text +
   image) in the harvester. Feed main + quoted into BOTH stages with context on the dynamic. Extract
   quoted claims only if they bear on the main post (Stage 2 rule above); else context only. **Test
   empirically — be careful here.**
6. **Image→text serialization in normalize (flagged only), extractive not judgmental.** The verifier
   is text-only; this is the hallucination trust boundary.
7. **Verification stays Serper→Exa retrieve-then-reason** — native frontier web-search is ~16–40× the
   cost and post-rationalizes (cited source fails to support the claim ~half the time); the Serper
   loop gives a separate, inspectable evidence set.

## Built vs TODO

- **BUILT:** image-inclusive harvest (`eval/scripts/fetch_images.py`, `eval/media.py`); the image
  dataset (1,620 image posts, `image_*` cols on the per-source parquets); the `claim_locus` gate
  prototype + validation (`eval/scripts/image_audit.py` → `image_audit.parquet`,
  `image_gate_crosscheck.py`). Verified Qwen3-VL image-token sizing (`media.est_qwen_tokens`).
- **TODO:** (a) **quote-tweet harvest** — capture `quoted_tweet` text+media from the syndication
  payload (verify the field on real QT samples); (b) **video tagging** — `media_type` from
  `mediaDetails`; (c) **Stage-1 full gate** — add the scope booleans + video drop to the validated
  `claim_locus` call, then its eval; (d) **Stage-2 normalize+serialize** + its eval; (e) **bake-offs**
  (gate 4B vs 30B-A3B; normalize 30B-A3B vs 235B); (f) **latency/$ analysis** after the accuracy run.

## Eval plans

- **Stage 1:** positives = real harvested posts (should PASS → target ~100% recall), **excluding
  video-dependent posts which are correct DROPs**; negatives = **synthetic out-of-scope posts**
  (no-claim/opinion/ads/decorative-image/video-dependent), kept in a **separate file folded in at
  inference**, to measure specificity. Plus `claim_locus` accuracy (already validated).
- **Stage 2:** a **judge** compares Qwen's `normalized_claim` vs the fact-checker's gold normalized
  claim (the dataset's `claim_text`); plus **serialization fidelity / hallucination rate** on image
  posts (the main new risk).
- **Stage 3:** unchanged (existing AVeriTeC track).

## Prototyping notes

- `uv`; venv at `src/.venv`. Mirror the structured-output pattern in
  `src/pipeline/verify_prompts.py` (`build_*_messages` / `parse_*`; strict JSON; `_load_obj`).
- Vision via OpenAI-compatible `image_url` blocks (base64 or URL). Saved images are WebP, ~1280-token
  Qwen3-VL budget (text-legible). Add gate/normalize model + base-url to `src/config.py` alongside
  `VERIFICATION_*`.
- Test data: `src/eval/data/*_harvest.parquet` (image rows have `image_paths`/`has_image`/`image_*`;
  gate labels in `eval/data/image_audit.parquet`). Log in `src/clog/DDMMYY.md`.
