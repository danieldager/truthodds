# Fact-Checking System Deep-Dive — Verdict Design, Calibration, Grounding, Images

*Compiled 2026-06-27 from a multi-agent literature sweep (verdict taxonomies, black-box
calibration, justification grounding, multimodal/OOC). Companion to
`factcheck_literature_briefing.md` (the raw scan). This doc applies the findings to our
pipeline (`pipeline/verify.py`, `VerdictScores`, the post-level FLAG). Citations carry arXiv
IDs; a handful flagged ⚠️ were not first-party-verified.*

---

## 0. The key realization — the 2-axis verdict is already latent in `VerdictScores`

Our verifier already emits four Likert dims:
`veracity / evidence_sufficiency (coverage) / evidence_agreement (consistency) / source_reliability`.
The fact-checking literature's cleanest verdict model is **two orthogonal axes** — evidence
**direction** (refute↔support) × evidence **sufficiency** (how decisive) — collapsed to a label
by thresholds. We already produce both:

| Literature axis | Our existing Likert dim |
|---|---|
| direction / stance | `veracity` |
| sufficiency (abstention gate) | `evidence_sufficiency` |
| conflict (CE vs NEI disambiguation) | `evidence_agreement` |
| source-credibility weighting | `source_reliability` |

This is the **subjective-logic opinion** `(belief b, disbelief d, uncertainty u)` of Evidential
Deep Learning (Sensoy et al., NeurIPS 2018, arXiv:1806.01768 ⚠️ listing-only): `b/d` = direction,
`u` = sufficiency, label = a **threshold over the (direction, u) plane**. No fact-checking paper
has adopted this as a verdict head — AVeriTeC factorizes sufficiency × direction only in its
*scoring metric* (arXiv:2410.23850), not the model.

**Design decisions (Daniel, 2026-06-27):** (a) **streamline** today's two evaluators (4-dim Likert
+ 4-class) into **one** verdict call; (b) **harm is NOT a verdict input** — it's assumed upstream
(the claim-selection gate already scored it; reaching verification implies harm); (c) split cleanly
into a **per-claim verdict (Layer A, context-free, cacheable)** and a **context-rich post-level
nudge gate (Layer B)** that re-reads the raw post + image. So the 2-axis collapses to a single
streamlined verdict `{verdict, confidence}` where `verdict` ∈ supported/refuted/no_support_found
encodes direction + abstention and `confidence` is the one calibratable scalar — see §1.5.

---

## 1. Verdict scheme

### 1.1 What the literature says (with the rationale for each granularity)
The binary→3→4→6-class progression is driven by one recurring failure: the **mixed/misleading
claim**. Each added class exists to stop forcing partially-true claims into true/false.
- **FEVER 3-class** S/R/**NEI** (arXiv:1803.05355) — but it's an *entailment-vs-corpus* scheme
  (≈ NLI entail/contradict/neutral), **not** real-world veracity; NEI conflates "unverifiable"
  with "we didn't retrieve it."
- **AVeriTeC 4-class** (arXiv:2305.13117) added **Conflicting Evidence/Cherrypicking** precisely
  for two real cases NEI can't hold: sources *legitimately disagree*, and *technically-true-but-
  misleading* (cherry-picked). Crucial distinction baked into the dataset: **NEE = "evidence could
  not be found" (absence)** vs **Conflicting = "evidence present on both sides."**
- **PolitiFact 6-way ordinal** (LIAR, arXiv:1705.00648) — graded truth, but mixes accuracy +
  egregiousness on one axis; low agreement; downstream work collapses it.
- **Stance schemes** (FNC-1, RumourEval SDQC, Emergent for/against/observing) — judging
  evidence→claim *relationship* is easier than truth; the ancestor of S/R/NEI.
- **Real-world open sets** (MultiFC: ~2–40 labels/site, ~165 distinct total — *not* the "2.5k"
  some sources repeat; Snopes adds "Mixture"; Full Fact uses free prose) — max nuance, min
  comparability → the standing argument *for* a small unified scheme.

### 1.2 NEI is an abstention state, not a midpoint
Strongly supported: Atanasova et al. (TACL 2022, arXiv:2204.02007) reframe NEI as **Evidence
Sufficiency Prediction** — a gate *before* the verdict. The epistemic/aleatoric split (DeepMind,
arXiv:2406.02543) is the theory: **NEI ≈ epistemic** (we lack evidence — *fixable by more
retrieval*) vs **Conflicting ≈ aleatoric** (the world genuinely disagrees — *not* fixable).
ProoFVer (arXiv:2108.11357) makes insufficiency a *structurally separate* outcome; Kochkina &
Liakata (arXiv:2005.07174) threshold an uncertainty estimate to **defer to humans**. Empirical
warning: in FEVER, **SUPPORTS↔NEI is the single most-confused annotator pair**, and ~70% of
"NEI" claims actually have retrievable evidence — i.e. NEI usually means *retrieval failed*, not
*unverifiable*. **So sufficiency must be earned (tie to retrieval signals) and calibrated, or
"unverified" becomes the model's lazy default.**

### 1.3 Merging NEI + Conflicting — precedented, and a *product* decision
The field both splits (AVeriTeC) and merges (QuanTemp uses True/False/**Conflicting** with *no*
NEI, arXiv:2403.17169; RumourEval's "unverified" absorbs both). **Decision rule: merge iff the
downstream action is identical.** For our nudge:
- The **flag trigger** doesn't need NEI vs CE → **merge them into one "could-not-support" state**
  for the trigger. ✅ matches our instinct ("we're really flagging absence of support").
- The **nudge *message*** differs ("we couldn't verify this" vs "sources disagree / cherry-
  picked") → **keep them distinguishable as metadata** (which `evidence_agreement` already gives
  us), without gating the flag on the distinction.
- **Cherry-picking** (technically-true-but-misleading): harm is **already assumed upstream** (the
  claim-selection gate scored it; a claim only reaches verification if it cleared that gate), so the
  verdict stage does **not** re-weigh harm. If cherry-picking is surfaced at all, it's the
  **post-level nudge gate's** job (Layer B sees the raw post), not the per-claim verdict's.

### 1.4 Operating point — precision-first (this is a nudge, not a benchmark)
For any user-facing flag, the intervention literature is unanimous: **optimize precision over
recall.**
- **Implied truth effect** (Pennycook et al., *Management Science* 2020, DOI 10.1287/mnsc.2019.3478):
  flagging *some* false claims makes *unflagged* ones seem more true ("unflagged ⇒ vetted").
  → "no flag" must not read as "true"; pair with a verified/uncertain state.
- **"Disputed" labels ≈ no effect; confident "False" works** — label-confidence matters as much
  as the threshold.
- **Community Notes** is the production blueprint: helpful at **intercept ≥ 0.40** + **bridging**
  (cross-viewpoint agreement). Cost is latency — notes land ~15 h late, after ~80% of reshares
  (Chuai et al., arXiv:2307.07960); but when they land, ~62% spread reduction (arXiv:2409.08781).
  → **tier by confidence × latency**: confident specific flag when sure; generic *accuracy nudge*
  (zero false-positive cost, Pennycook *Nature* 2021, DOI 10.1038/s41586-021-03344-2) when unsure.

### 1.5 Recommendation — streamlined per-claim verdict (Layer A) + post-level nudge gate (Layer B)

Two layers, matching the existing architecture.

**Layer A — per-claim verdict (context-free, cacheable, ONE LLM call).** Collapse today's two
evaluators into a single verdict emitted by the **synthesis call itself** — the moment synthesis
decides it's confident (`next_query=null`) IS the moment it has a verdict:

```json
{
  "analysis": "reasoning over the evidence pool, with [n] citations",
  "verdict": "supported | refuted | no_support_found",
  "confidence": 1-5,            // calibrated post-hoc on gold (§2) → P(verdict correct)
  "next_query": "<string|null>" // null = confident/done
}
```
- `verdict` carries both axes: **direction** (supported/refuted) + the **abstention state**
  (`no_support_found` = merged NEI∪CE — not distinguished).
- **Harm is NOT considered** — assumed upstream (claim-selection gate already scored it).
- **`no_support_found` is gated on retrieval exhaustion: never honored until Exa has been queried
  ≥2×** (§1.2's "NEI must be earned" made procedural). If synthesis returns
  `no_support_found`+`next_query=null` while `exa_queries < 2`, override → force-continue on Exa.
  `supported`/`refuted` may still resolve early (decisive evidence found).
- Caches as `{claim, verdict, confidence, justification=analysis, evidence_urls}`.

Removes 2 LLM calls/claim + one prompt family. **Tradeoff:** folding the verdict into synthesis is
maximally streamlined but slightly less auditable than a separate verdict pass (cf. IDEA-013's
"split is more auditable" note for a self-justifying tool) — if auditability wins, keep ONE
*separate* verdict call, same schema. Dropping `Conflicting` from the emit set costs a few % on
AVeriTeC (rare class) but is consistent with the existing NEE/Refuted realign.

**Layer B — post-level nudge gate (context-rich).** Qwen reads `{all per-claim verdicts +
justifications} + raw post + image` → `{flag, reason}`. This is where **context re-enters and can
*flip* the decision** — canonically **true attribution of a false claim** ("X said [false thing]"):
the claim verdict is `refuted`, but the post is accurate reporting, so the gate must *not* fire a
"you're spreading falsehood" flag (the existing attribution Open Question).

**Calibration + operating point** (§2): `confidence` → P(correct) on the gold set; the nudge fires
**precision-first** off `refuted` above a calibrated threshold τ_flag (set from the risk–coverage
curve, asymmetric per AVeriTeC-2024).

**The drop-Layer-B ablation (Daniel).** Compare:
- *Programmatic baseline:* `flag = any(v.verdict=="refuted" and calib(v.confidence) ≥ τ_flag)`
  (precision-first, "any claim trips it" — matches the streaming early-exit in design §3b).
- *Layer B:* the Qwen-with-raw-post gate.
Measure agreement (κ + confusion matrix) **stratified by post complexity** — crucially on the
**attribution/context-laden subset** (true-attribution, satire, image-dependent), not just overall.
Layer B earns its keep *only* if it correctly overrides the programmatic rule on exactly those
cases: high overall agreement with disagreements **concentrated on attribution cases** → keep B;
high agreement **including** those cases → drop B for the cheaper programmatic rule. (A high overall
κ alone is misleading — B's entire value is the rare context cases the programmatic rule can't see.)

> **Note — a CLAUDE.md inconsistency to reconcile:** the Key Architectural Decisions table still
> says *"Claim verdict scale → PolitiFact 6-point"*, but the pipeline body and `verify.py`
> implement **AVeriTeC 4-class + 4-dim Likert**. This streamlined `{verdict, confidence}` proposal
> supersedes both; the decision row should be updated once agreed.

---

## 2. Confidence calibration

### 2.1 ECE and friends
**ECE** = bin predictions by confidence, average the gap between confidence and accuracy:
`ECE = Σ_b (n_b/N)·|acc(b) − conf(b)|` (Guo et al. 2017, arXiv:1706.04599). With our **5 Likert
levels the levels *are* the bins** — the usual binning-hyperparameter problem vanishes; the only
residual issue is small-sample per-level accuracy (use Wilson/bootstrap CIs). Report alongside:
- **Brier score** — proper scoring rule, can't be gamed the way ECE can (ECE→0 is trivially
  reachable by predicting the base rate). Use as the headline number.
- **AUROC / risk–coverage (AURC)** — *discrimination*: does higher Likert rank more-correct
  verdicts above less-correct? This is what actually matters for an abstaining nudge, and it's
  invariant to monotone transforms (robust to coarse ordinals).
- **classwise ECE** — verdict classes are imbalanced (true is rare in a misinfo feed); top-label
  ECE hides minority miscalibration. Prefer **smoothECE** (arXiv:2309.12236, `pip install relplot`)
  or adaptive/equal-mass bins over equal-width.

"Calibrate the Likert" = build the per-level reliability table (for each level L, empirical
accuracy of verdicts emitted at L) and check it's monotone and matches; if not, fix with a
post-hoc map (§2.3).

### 2.2 DeepInfra DOES expose logprobs (premise correction)
Verified from `docs.deepinfra.com/chat/log-probs`: the OpenAI-compatible chat endpoint accepts
`logprobs: true` + `top_logprobs` (1–20). **So we are not strictly black-box.** *But* Tian et al.
(EMNLP 2023, arXiv:2305.14975) found that for RLHF/instruct models, **verbalized confidence is
often *better* calibrated than raw token logprobs** (~50% relative ECE reduction). And a 1-token
verdict logprob ignores the reasoning. So treat {verbalized Likert, vote-fraction, verdict
logprob} as candidate signals and let the gold set choose. Caveat: DeepInfra runs vLLM, which
often caps `top_logprobs` at 5 — test empirically.

### 2.3 Methods, cheap→expensive
1. **Verbalized Likert** (what we have) — 0 extra calls. Raw it's overconfident (Xiong et al.,
   ICLR 2024, arXiv:2306.13063 — scores pile at 80–100% "in multiples of 5"). Improve elicitation
   by having the model weigh the *alternative* verdict before committing (top-k / distractor
   effect; Chhikara TMLR 2025, arXiv:2502.11028 cut ECE up to 90%). CoT helps accuracy, not
   calibration.
2. **Self-consistency vote-fraction** — sample N≈5–10 verdicts at T>0, confidence = majority
   share over the closed label set (semantic entropy *reduces to vote entropy* on a fixed label
   set — the semantic machinery buys nothing here). Strongest cheap black-box signal.
3. **BSDetector** (Chen & Mueller, ACL 2024, arXiv:2308.16175) — `0.7·consistency + 0.3·self-
   reflection`, ~6 calls, AUROC up to 0.95; the best reported cheap black-box recipe, works on a
   closed label set.
4. **Post-hoc calibration map on the gold set (highest ROI, ~0 API cost)** — map raw signal →
   P(correct). With 5 levels start with **histogram-on-the-levels**; with hundreds of examples
   use **Platt** or **Beta calibration** (Beta handles the saturated/non-sigmoidal curves typical
   of LLM confidence and can't "uncalibrate" a good score). **Avoid isotonic/spline** at small N.
   *The recalibration matters far more than which raw signal you start from.*

### 2.4 Recommendation
Keep the verbalized Likert; add vote-fraction when budget allows (it doubles as the sufficiency
estimate for §1's thresholds); **fit a Platt/Beta/histogram map on the gold set**; pick the raw
signal empirically by **Brier + classwise ECE (calibration)** and **AUROC + AURC (abstention)**.
Set the nudge threshold τ_flag from the **risk–coverage curve** (precision-first), or use **split
conformal prediction** over the closed verdict set for a coverage guarantee (watch exchangeability
under topic/time shift).

---

## 3. Justification grounding (proving the verdict's justification is evidence-based)

Core caution: **a correct verdict with a plausible citation is *not* grounding** — up to **57% of
RAG citations are "post-rationalized"** (the model didn't actually use the cited source; Wallat et
al., arXiv:2412.18004). Verify each citation independently. Our setup (Serper snippets with IDs;
`EvidenceDoc` already carries `summary` + `quotes`) maps onto the **ALCE + Ev2R** recipe.

### 3.1 The per-verdict grounding score (measurement)
1. **Decompose** the justification into atomic claims (one cheap LLM call; FActScore/RAGAS style).
2. **Entail** each atomic claim against (i) its cited snippet IDs and (ii) the full pool, using a
   **small NLI model, not an LLM judge** — **MiniCheck-FT5 (770M, arXiv:2404.10774)** or
   **AlignScore (355M, arXiv:2305.16739)**: GPT-4-level grounding at ~400×/orders-of-magnitude
   lower cost, no API spend, ms/pair.
3. **ALCE citation precision/recall/F1** (Gao et al., EMNLP 2023, arXiv:2305.14627):
   *recall* = do cited snippets entail each sentence; *precision* = is each citation non-redundant.
   **Grounding score = citation F1.**
4. On gold-evidence slices add **Ev2R** (TACL 2025, arXiv:2411.05375) — atomic-fact precision/
   recall vs gold; best human correlation among evidence metrics, the AVeriTeC-2025 scorer.
5. Report the **JFR (Justification Flaw Rate** = correct verdict but poor justification; from
   FACT-AUDIT, arXiv:2502.17924) — isolates exactly the failure we care about. (IMR = *Insight
   Mastery Rate*; both are LLM-judge aggregates over a probe set, not per-claim.)
6. **Calibrate the scorer** against ~50–100 human binary-AIS judgments (AIS, Rashkin et al.,
   arXiv:2112.12870) — even GPT-4 attribution judges hit only ~80–83%. Judge with a *different*
   model than the generator (self-enhancement bias), reason-before-score.

### 3.2 Raising grounding (cheap→expensive)
1. **Inline `[id]` citations** + context-faithful "opinion-based" prompting (ALCE; Zhou et al.,
   arXiv:2303.11315) — free.
2. **Quote-then-verdict**: require a per-snippet verbatim supporting span *before* the verdict
   (Chain-of-Note arXiv:2311.09210 + GopherCite verbatim-quote). Grounds citations *by
   construction*, improves abstention, and hands §3.1 the spans to check. We already extract
   `quotes` per doc — wire them into the synthesis/justification as cited spans.
3. **RARR-lite verify-and-repair** (Gao et al., arXiv:2210.08726): run the scorer; for any
   sentence whose citation fails entailment, drop or minimally-revise it (preservation guard).
   One loop yields the score *and* the fix — directly attacks the 57% post-rationalization.
4. *(self-host only)* Context-aware Decoding (arXiv:2305.14739) needs logits — skip on the API.

---

## 4. Images / out-of-context (the ~2000 post images)

**OOC is the dominant social-media mode** (>40% of visual misinfo): a *real* image reused with a
false caption. Pixel/AI-forensics are useless here — it's an **evidence-retrieval** problem.
Recipe: reverse-image search → earliest/original appearance → does the original context (date,
place, event, people) match the caption? (Formalized as the **5 Pillars**, Tonglet et al., EMNLP
2024; follow-up **COVE**, NAACL 2025, arXiv:2502.01194.)

### 4.1 Templates & guardrails
- **SNIFFER** (CVPR 2024, arXiv:2403.03170) — InstructBLIP + Google entity detection + reverse-
  image-search external check; 88.4% on NewsCLIPpings.
- **DEFAME** (ICML 2025, arXiv:2412.10510) — strongest blueprint: MLLM orchestrating RIS +
  GeoCLIP + web search → structured report; **83.9% VERITE**, 70.5% AVeriTeC; **69.7% vs GPT-4o's
  35.2%** on post-cutoff claims (retrieval beats parametric, generalizes over time).
- **Guardrails (critical):** MLLM checkers have severe **label bias** (LLaVA-7b answered "True"
  337/400; Geng et al., arXiv:2403.03627), exploit **surface-similarity shortcuts** (MUSE,
  arXiv:2407.13488) and **collapse to ~13% on miscaptioned cases** (VERITE). They're **degraded by
  noisy retrieval** → need **evidence relevance-filtering** (RED-DOT arXiv:2311.09939, +33.7% on
  VERITE; CMIE arXiv:2505.23449) + calibrated abstention. Our **Qwen3-VL** is a reasonable but
  **under-benchmarked** OOC backbone (papers used LLaVA/InstructBLIP).

### 4.2 Tooling (at our scale)
- **Google Cloud Vision Web Detection** — official, ~**$3.50 total for 2,000 images** (1k free/mo
  then $3.50/1k); returns matching pages + web entities.
- **TinEye API** — the *only* engine with native **oldest-first** sorting = "find the original"
  (~$200/5k bundle).
- **Yandex via SerpAPI** — best for faces/landmarks. **Bing Visual Search API retired Aug 2025.**
- **EXIF/C2PA**: X strips both on upload — opportunistic only, ~0% hit rate.

### 4.3 Recommendation
Add a **reverse-image-search evidence branch** that feeds the *existing* Tier-3 loop (RIS pages →
the same synthesis/verdict path), with an OOC prompt ("does the image's earliest context match the
caption?"). This reuses everything and slots into the gate's existing **video/image locus routing**.
Evaluate on **VERITE + XFacta** (real, modality-balanced, leakage-resistant), **not** synthetic
NewsCLIPpings alone. (Naming: "Post-4V" → Geng et al. 2024; "MM-FEVER" → the FACTIFY family.)

---

## 5. Validation datasets & methods (consolidated)

| Goal | Method / metric | Dataset(s) |
|---|---|---|
| Verdict accuracy | 4-class acc + macro-F1; **label *and* evidence-adequate** bar | AVeriTeC (2305.13117) |
| Score the *retriever* | **Ev2R** (2411.05375) atomic-fact precision/recall | AVeriTeC + ClaimReview `appearance` (IDEA-011) |
| Extraction quality | **Focus & Coverage** (FEVERFact 2502.04955), not overlap | FEVERFact |
| Confidence | reliability table + **Brier / classwise ECE / AUROC / AURC**; isotonic→avoid, Platt/Beta | own gold set |
| Justification grounding | **ALCE citation-F1** via MiniCheck; **JFR**; calibrate vs **AIS** | own gold + Ev2R slice |
| Numeric/temporal claims | verdict acc + Ev2R justification | TSVer (2511.01101) |
| Abstention quality | risk–coverage; **Unknown-Rate** | RealFactBench (2506.12538) |
| Images / OOC | modality-balanced acc, miscaptioned subset | **VERITE** (2304.14133), **XFacta** (2508.09999), MMFakeBench (2406.08772), Twitter-COMMs |

Gold-set design notes: tag `judged_axis` (attribution alignment — existing Open Question); label
the **two axes** (direction, sufficiency) + a conflict flag rather than a flat 4-class, so any
label collapse can be evaluated post-hoc; bootstrap with LLM-as-judge + weak supervision, but
keep a human-AIS calibration slice.

---

## 6. Prioritized roadmap (effort × payoff)

**Tier 1 — cheap, high payoff, no new infra:**
1. **Calibrate the existing Likert** on the gold set (reliability table + Brier/ECE/AUROC + a
   Platt/Beta map). Pure analysis; unblocks the precision-gated nudge threshold.
2. **Streamline to the single per-claim verdict + Exa×2 abstention gate** (§1.5): fold the dual
   evaluators into one `{verdict, confidence, justification}`; gate `no_support_found` on Exa≥2;
   no harm input. Keep the post-level nudge gate (Layer B) and run the drop-Layer-B ablation.
3. **Grounding score**: decompose justification → MiniCheck entailment → ALCE citation-F1 + JFR,
   reusing the `quotes` we already extract. Off paid APIs.

**Tier 2 — medium effort:**
4. **Quote-then-verdict + inline `[id]`** in synthesis/justification, then **RARR-lite** repair.
5. **Reverse-image-search evidence branch** for OOC (Cloud Vision + TinEye), feeding the existing
   loop; eval on VERITE/XFacta.
6. **Source-credibility weighting** (CONFACT, arXiv:2505.17762) via the existing
   `source_reliability` dim into the sufficiency/verdict thresholds.

**Tier 3 — larger / research:**
7. **Self-consistency vote-fraction** confidence (N samples) as the sufficiency estimator.
8. **EDL/subjective-logic verdict head** unifying (direction, u) — the publishable angle.

---

## 7. References (key IDs; ⚠️ = not first-party-verified this sweep)

Verdict/labels: FEVER 1803.05355, AVeriTeC 2305.13117 + shared task 2410.23850, LIAR 1705.00648,
MultiFC 1909.03242, QuanTemp 2403.17169, Atanasova-sufficiency 2204.02007, ProoFVer 2108.11357,
AmbiFC 2104.00640, epistemic/aleatoric 2406.02543, EDL 1806.01768 ⚠️, Sahitaj 2502.08909.
Calibration: ECE 1706.04599, smoothECE 2309.12236, AURC/AUGRC 2407.01032, Tian 2305.14975, Xiong
2306.13063, BSDetector 2308.16175, self-consistency 2203.11171, semantic-entropy 2302.09664,
P(True) 2207.05221, verbalized-prob calibration 2410.06707, DeepInfra logprobs (docs, verified).
Grounding: AIS 2112.12870, ALCE 2305.14627, AttrScore 2305.06311, FActScore 2305.14251, SAFE
2403.18802, MiniCheck 2404.10774, AlignScore 2305.16739, SummaC 2111.09525, RARR 2210.08726,
Chain-of-Note 2311.09210, Ev2R 2411.05375, FACT-AUDIT 2502.17924, post-rationalization 2412.18004,
RAGAS 2309.15217, judge-bias 2306.05685.
Images: SNIFFER 2403.03170, DEFAME 2412.10510, CLIPScore 2104.08718, CCN 2112.00061, RED-DOT
2311.09939, MUSE 2407.13488, CMIE 2505.23449, MOCHEG 2205.12487, NewsCLIPpings 2104.05893,
MMFakeBench 2406.08772, VERITE 2304.14133, XFacta 2508.09999, 5Pils-EMNLP24 / COVE 2502.01194,
Geng 2403.03627.
Intervention: implied-truth (Mgmt Sci 2020, 10.1287/mnsc.2019.3478 ⚠️), accuracy-nudge (Nature
2021, 10.1038/s41586-021-03344-2 ⚠️), Birdwatch 2210.15723 ⚠️, CN rollout 2307.07960, CN
effect 2409.08781.
