# Design document: post-level fact-checking pipeline

> **⚠️ HISTORICAL SNAPSHOT (2026-05-03) — superseded by `CLAUDE.md` § Fact-Checking Pipeline.**
> Kept as a point-in-time record of early pipeline design. Pre-dates the Tier 3 ClaimCheck/FIRE/HerO2 architecture and the calibrated verdict aggregation decision. Do not treat as current.

## Goal
Build a low-resource, open-source pipeline that takes a post (text now, image text later) and outputs calibrated post-level scores:

- `veracity_score` in `[0, 1]`
- `implication_truth_score` in `[0, 1]`
- `bias_score` in `[0, 1]`
- `confidence_score` in `[0, 1]`
- `context_completeness_score` in `[0, 1]`  
  (recommended extra score; captures omission / framing risk)

The system must:
1. decompose a post into verifiable claims and subjective statements,
2. extract implied claims / implications and verify them separately,
3. verify explicit and implied claims with web-grounded evidence,
4. aggregate claim-level results into post-level scores,
5. remain simple enough for a single engineer to ship and maintain.

## Recommended terminology
Use these labels internally:

- **Explicit claim**: directly stated factual assertion.
- **Implication**: a factual conclusion strongly suggested by the post but not directly stated.
- **Subjective statement**: opinion, stance, evaluation, rhetoric, sarcasm, or affective language.
- **Misleadingness**: whether the post is likely to induce a false interpretation through implication, omission, framing, or selective emphasis.
- **Bias**: subjective tilt in wording, framing, and argumentative posture.
- **Context completeness**: whether the post provides enough context to interpret the claims fairly.

If you need one umbrella term for “true but misleading,” use **framing risk** or **implication risk**. That is cleaner than overloading “misleadingness” alone.

## High-level architecture

```text
Post
  -> preprocess
  -> segment into spans
  -> classify spans: explicit claim / implication candidate / subjective / other
  -> normalize explicit claims
  -> extract implications from claims + context
  -> score subjectivity / bias
  -> retrieve evidence for each claim and implication
  -> verify each unit
  -> calibrate evidence quality and uncertainty
  -> aggregate to post-level scores
```

## Design principles

1. **Separate extraction from verification.**
   Do not try to verify raw post text directly.
2. **Make implications first-class objects.**
   Treat implied claims as evidence-checkable items, not as a side note.
3. **Use hybrid retrieval early.**
   BM25 + dense retrieval will outperform single-method retrieval for a low-resource build.
4. **Keep the verifier small.**
   A single open 8B-class model is enough for v1.
5. **Calibrate the final score.**
   The final score should be learned from dev data, not hard-coded.
6. **Add reasoning depth only when needed.**
   Escalate to multi-question / multi-agent logic only for hard claims.

---

# Intermediate representations

Use explicit JSON-like objects between modules. This makes the system easy to debug and easy for an AI agent to modify.

## 1) Post object
```json
{
  "post_id": "...",
  "source_type": "text|image|link_preview|mixed",
  "raw_text": "...",
  "language": "en",
  "metadata": {
    "author": "optional",
    "timestamp": "optional",
    "url": "optional"
  }
}
```

## 2) Span object
```json
{
  "span_id": "...",
  "text": "...",
  "start_char": 0,
  "end_char": 0,
  "span_type": "explicit_claim|implication_candidate|subjective|other",
  "confidence": 0.0
}
```

## 3) Claim object
```json
{
  "claim_id": "...",
  "source_span_id": "...",
  "claim_text": "normalized standalone claim",
  "claim_type": "explicit|implied",
  "topics": ["..."],
  "time_sensitivity": "low|medium|high",
  "checkable": true
}
```

## 4) Subjective statement object
```json
{
  "statement_id": "...",
  "source_span_id": "...",
  "text": "...",
  "bias_axes": {
    "loaded_language": 0.0,
    "speculation": 0.0,
    "sarcasm": 0.0,
    "aggressiveness": 0.0
  },
  "overall_bias": 0.0
}
```

## 5) Evidence bundle
```json
{
  "claim_id": "...",
  "queries": ["..."],
  "retrieved_items": [
    {
      "doc_id": "...",
      "title": "...",
      "url": "...",
      "snippet": "...",
      "retrieval_score": 0.0,
      "rerank_score": 0.0
    }
  ]
}
```

## 6) Verification result
```json
{
  "claim_id": "...",
  "label": "supported|refuted|nei|conflicting",
  "p_supported": 0.0,
  "p_refuted": 0.0,
  "p_nei": 0.0,
  "p_conflicting": 0.0,
  "evidence_quality": 0.0,
  "explanation": "...",
  "used_evidence_ids": ["..."],
  "calibration_features": {
    "retrieval_coverage": 0.0,
    "evidence_agreement": 0.0,
    "source_diversity": 0.0,
    "recency_match": 0.0
  }
}
```

---

# Module 1: preprocessing and segmentation

## Responsibilities
- Clean raw post text.
- Detect language.
- Split into clauses, sentences, and candidate spans.
- Detect whether the post contains text, a link preview, or image text.
- For images later, route OCR / VLM captions into the same span layer.

## Implementation notes
- Use a simple sentence splitter and clause splitter first.
- Add an LLM-based span refiner only for ambiguous or long posts.
- Preserve source offsets for every span.

## Output
A list of spans with provisional labels.

---

# Module 2: claim / implication / subjectivity decomposition

This is the most important module in the whole pipeline.

## 2.1 Explicit claim extraction
Extract atomic claims from factual spans.

### Desired behavior
- Split conjunctions into separate claims when possible.
- Normalize pronouns, ellipsis, and referents.
- Convert link previews into content claims if the preview is factual.

### Low-resource implementation
Use a two-stage approach:
1. deterministic rules for easy cases,
2. small open instruction model for difficult cases.

## 2.2 Implication extraction
For each claim or group of claims, generate implicit factual consequences that a reasonable reader would infer.

### What counts as an implication
- causal implications
- comparative implications
- temporal implications
- responsibility / blame implications
- policy implications
- “therefore” style conclusions

### Do not overgenerate
Only keep implications that are:
- strongly suggested by the text,
- checkable in principle,
- likely to affect interpretation.

### Output format
Each implication should be a standalone claim-like sentence.

## 2.3 Subjective statement extraction
Detect opinions, value judgments, insults, praise, suspicion, sarcasm, and loaded phrasing.

### Output
For each subjective statement:
- the span text,
- a bias profile,
- a one-sentence interpretation of how it frames the post.

### Bias dimensions
Recommended axes:
- loaded language
- certainty inflation
- emotional intensity
- partisanship / framing
- sarcasm / irony

---

# Module 3: retrieval

Use a hybrid retrieval stack.

## Why hybrid
- BM25 gives strong lexical recall.
- Dense retrieval gives semantic recall.
- Reranking improves precision.

This is the safest low-resource approximation of the better recent open systems.

## Retrieval stages
1. **Query generation**
   - from claim text,
   - from claim + surrounding context,
   - optionally from a HyDE-style hypothetical evidence snippet.
2. **BM25 retrieval**
   - broad recall,
   - top 50 to 200 docs.
3. **Dense retrieval**
   - sentence/document embeddings,
   - top 50 to 200 docs.
4. **Merge and deduplicate**
5. **Rerank**
   - small cross-encoder or lightweight LLM scoring.

## Evidence sources
Start with:
- a web search API,
- Wikipedia / reference corpora if available,
- fact-check archives,
- optionally a local snapshot of trusted sources.

## Retrieval output
Always return both:
- supporting evidence,
- refuting evidence,
- weak / contradictory evidence.

Do not optimize only for “topically relevant.”

---

# Module 4: verifier

## Responsibilities
Given a normalized claim or implication and its evidence bundle:
- determine support / refute / neither / conflict,
- produce a short rationale,
- estimate confidence from evidence quality.

## Recommended first implementation
Use one open-weight instruction model in the 7B–8B class.

### Verifier prompt contract
Input:
- claim text
- evidence snippets
- source URLs/titles
- short policy for verdict labels

Output:
- label
- rationale
- confidence
- evidence ids used

## Reasoning style
Ask the verifier to:
1. restate the claim,
2. identify which evidence supports/refutes it,
3. note missing context,
4. give a label.

Keep the output structured and short.

## Escalation rule
If the verifier confidence is low or evidence is conflicting:
- generate 1–3 targeted questions,
- answer them from evidence,
- re-run verification.

That gives you a cheap version of HerO / FACT5-style deeper reasoning.

---

# Module 5: special handling for implications

Implications should be verified exactly like claims, but with one extra step.

## Step
Before verification, attach the source claims that gave rise to the implication.

Example:
- explicit claim: “The policy cut inflation.”
- implication: “The policy is responsible for the observed price drop.”

The implication should be verified separately because the first claim can be true while the implication is false or unsupported.

## Suggested label set for implications
Use the same labels as claims:
- supported
- refuted
- nei
- conflicting

But track an extra field:
- `implication_strength`: low / medium / high

This helps later when aggregating misleadingness.

---

# Module 6: post-level aggregation

The final post score should not be a simple average.

## Recommended outputs
### 1. Veracity score
How true are the explicit claims?

### 2. Implication truth score
How well supported are the implied claims / inferred conclusions?

### 3. Bias score
How subjective, loaded, or rhetorically slanted is the post?

### 4. Confidence score
How reliable is the overall system judgment, given evidence quality and coverage?

### 5. Context completeness score
How much missing context or selective framing is present?

## Recommended aggregation logic
Compute separate scores first, then combine.

### Claim-level to post-level
For explicit claims:
- supported = 1.0
- conflicting = 0.4
- nei = 0.5
- refuted = 0.0

For implications:
- same mapping, but weight slightly higher than explicit claims if the implication is central to the post’s intended meaning.

### Evidence-weighted aggregation
Weight each item by:
- claim salience,
- verifier confidence,
- evidence quality,
- whether the claim is explicit or implied.

### Suggested final formulas
```text
veracity_score = calibrated_weighted_mean(explicit_claim_scores)
implication_truth_score = calibrated_weighted_mean(implication_scores)
bias_score = calibrated_mean(subjective_bias_scores)
context_completeness_score = 1 - omission_risk
confidence_score = calibration_model(evidence_quality, retrieval_coverage, agreement, source_diversity)
```

## Omission risk
This is the extra score worth keeping.
It captures cases where the post is technically true but misleading because it omits crucial qualifiers, timing, denominators, or exceptions.

Use:
- missing time qualifiers,
- missing denominators,
- selective comparison,
- cherry-picked time window,
- incomplete causal framing.

---

# Calibration layer

The final scalar scores should come from a learned calibration layer.

## Inputs to calibration
- mean claim score
- min claim score
- proportion refuted
- proportion NEI
- implication score mean
- evidence agreement
- retrieval coverage
- source diversity
- recency match
- subjectivity level
- omission risk

## Recommended first calibrator
- logistic regression or isotonic regression
- later, small MLP if needed

## Why calibration matters
Without calibration, a post with one unsupported claim can look too similar to a post with weak evidence everywhere.

---

# Roadmap: how to evolve from Fathom-like to HerO 2 / VILLAIN-like

## Phase 1: Fathom-like baseline
Minimal open-source system.

### Components
- sentence / clause splitter
- claim extractor
- HyDE-style query generation
- BM25 + dense retrieval
- lightweight verifier
- post-level calibration

### Strengths
- easy to build
- low compute
- good enough for a first shipping system

### Weaknesses
- limited hard-case reasoning
- no multimodal support
- weak on implication handling unless explicitly added

## Phase 2: HerO 2-style upgrade
Make retrieval and evidence preparation better.

### Additions
- document summarization before verification
- answer reformulation
- stronger reranker
- quantized open LLM verifier
- stricter runtime budget

### Effect
- better evidence quality
- better latency / cost
- more production-friendly

## Phase 3: FACT5-style nuance layer
Improve complex statement handling.

### Additions
- atomic decomposition into subclaims
- targeted questions per claim
- ordinal / nuanced verdicts for internal use
- stronger handling of partial truth

### Effect
- better on “true but misleading” claims
- better on compound statements

## Phase 4: VILLAIN-style multimodal escalation
Only for posts that include image claims or hard cross-modal evidence.

### Additions
- OCR and image captioning
- multimodal retrieval
- modality-specific analysis agents
- cross-modal consistency agent
- QA generation from evidence reports

### Effect
- supports image+text claims
- much stronger reasoning on multimodal misinformation

### Important
Do not start here. Add this only after the text pipeline is stable.

---

# Suggested model stack for a single engineer

## Cheap / practical default
- **Span classification / segmentation**: small encoder or lightweight LLM prompt
- **Claim decomposition**: open instruction model, small context window
- **Retrieval**: BM25 + dense embeddings
- **Reranking**: small cross-encoder or compact LLM
- **Verification**: one 7B–8B open instruction model
- **Calibration**: logistic regression / isotonic regression

## Better later
- stronger open verifier
- better reranker
- multilingual claim extraction
- OCR / VLM branch

## Avoid early
- training a large verifier from scratch
- fully multi-agent orchestration on every claim
- end-to-end black-box scoring without intermediate objects

---

# Evaluation plan

## Unit-level
- claim extraction F1
- implication extraction precision / recall
- subjective statement detection F1
- retrieval recall@k
- evidence attribution accuracy

## Claim-level
- veracity accuracy / macro-F1
- FEVER-style evidence score where applicable
- calibration error

## Post-level
- Spearman / Pearson correlation with human scores
- AUROC for false vs true post detection
- calibration curves for `veracity_score`
- human judgment of misleadingness / framing / bias

## Suggested human evaluation axes
- truthfulness of explicit claims
- truthfulness of implications
- degree of misleading framing
- bias / rhetorical slant
- adequacy of evidence

---

# Data strategy

## Training data sources
- FEVER / AVeriTeC-style fact verification data
- FACT5 for nuanced truthfulness
- CheckThat! claim normalization / subjectivity tasks
- AVerImaTeC / multimodal data later

## Labeling strategy
For your own data, annotate at three levels:
1. explicit claims,
2. implications,
3. subjective statements.

Also annotate:
- verdict
- evidence quality
- omission / framing risk
- bias direction

This will be much more useful than a binary true/false dataset.

---

# Implementation order

## Sprint 1
- post object schema
- segmentation
- claim extraction
- retrieval
- verifier
- final post score

## Sprint 2
- implication extraction
- subjective statement scoring
- calibration layer
- explanation output

## Sprint 3
- evidence summarization
- targeted question generation for hard cases
- multimodal OCR entry point

## Sprint 4
- VLM / image branch
- multi-agent escalation only for hard multimodal claims

---

# Reference systems to read next

## HerO
Open-source AVeriTeC runner-up with BM25 + dense retrieval, HyDE-style expansion, question generation, and fine-tuned veracity prediction.

## HerO 2
Improved HerO with document summarization, answer reformulation, quantization, and faster runtime.

## Fathom
Lightweight open-source RAG pipeline built on small models, HyDE-style question generation, BM25 + semantic retrieval, and lightweight verification.

## FACT5
Nuanced 5-way fact-checking benchmark plus pipeline for atomic claim decomposition and targeted questions.

## VILLAIN
Multimodal multi-agent verification system for image-text claims, with text/visual retrieval, modality-specific analysis, QA generation, and verdict prediction.

---

# Links

- HerO repository: `ssu-humane/HerO`
- HerO 2 paper: `Team HUMANE at AVeriTeC 2025: HerO 2 for Efficient Fact Verification`
- Fathom paper: `Fathom: A Fast and Modular RAG Pipeline for Fact-Checking`
- FACT5 paper/repo: `FACT5: A Novel Benchmark and Pipeline for Nuanced Fact-Checking of Complex Statements`
- VILLAIN paper/repo: `VILLAIN at AVerImaTeC: Verifying Image-Text Claims via Multi-Agent Collaboration`

## Default recommendation
Build the v1 as a **Fathom-like pipeline**, then add:
1. FACT5-style decomposition for nuance,
2. HerO 2-style retrieval and efficiency improvements,
3. VILLAIN-style multimodal and multi-agent escalation only for hard cases.

