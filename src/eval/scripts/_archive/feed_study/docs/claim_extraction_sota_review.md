# Claim Detection & Extraction — SOTA Literature Review (2025–2026 priority)

**Scope:** State of the art in *claim detection* and *claim extraction* for
fact-checking / misinformation, prioritizing 2025–2026 work. Research only — no
prompt, process, or code design here (that is a later, collaborative step).

**Method note:** Built from a fan-out web search → source fetch → 3-vote
adversarial verification → synthesis run (25 sources fetched, 116 candidate
claims extracted, top 25 adversarially verified, 24 confirmed / 1 refuted).
Findings tagged **[verified]** survived 3-vote adversarial checking; findings
tagged **[fetched, unverified]** come from the fetched source pool but were not
individually put through the adversarial vote (they were below the verification
cut or filtered for budget) — treat their specifics as indicative, not
confirmed. Each work is tagged with venue + date.

---

## Cross-cutting framing

The 2024–2026 literature increasingly treats fact-checking as a **staged
pipeline** — detection → extraction/normalization → (harm) prioritization →
verification — rather than one monolithic classifier. This is now
institutionalized by the field's main shared-task series (CLEF CheckThat! 2025),
which broke its tasks apart along exactly these lines. That validates the
*structure* of our cascade; the open questions are at the level of each stage's
method and how we judge its output. **[verified]**

---

## Theme 1 — Claim detection / check-worthiness

### 1.1 The shared-task series split detection from extraction — CLEF CheckThat! 2025 (6th ed.) **[verified]**
- **Idea:** CheckThat! 2025 structured its shared tasks as **Task 1 subjectivity
  identification** (follow-up from 2024), **Task 2 claim normalization**, **Task 3
  fact-checking numerical claims**, **Task 4 scientific web-discourse processing**.
  Treating *claim normalization* as a standalone numbered task (18 teams, 13
  languages) reflects the field's move to normalize/extract claims **separately**
  from detection and verification.
- **Why it matters:** The canonical benchmark series now mirrors a staged
  pipeline. "Is there a claim?" (detection/subjectivity) and "state the claim
  cleanly" (normalization) are no longer fused.
- **Venue/date:** Lab overview, ECIR/CLEF 2025 (Springer, DOI
  10.1007/978-3-031-88720-8_68), arXiv Mar 2025. *(Task-structure framing carried a
  2-1 vote; the "normalization is a distinct task" component was 3-0.)*
- **URL:** https://arxiv.org/abs/2503.14828 · lab site: https://checkthat.gitlab.io/clef2025/

### 1.2 Fine-tuned encoders/SLMs still beat large LLMs on detection — in data-rich settings **[verified]**
- **Idea:** Fine-tuned task-specific transformers (encoders and small LMs:
  XLM-RoBERTa, Llama2-7b, FLAN-T5) **outperform** zero/few-shot GPT-4 /
  GPT-3.5-Turbo / Mistral-7b on check-worthiness detection and veracity in
  real-world multilingual pipelines. Prompted LLMs remain a viable **label-free**
  option (no labeled data needed) and shine in **cross-lingual / low-resource**
  transfer, but do not claim superiority where labels exist.
- **Why it matters:** For a high-volume detection gate, the strongest accuracy /
  latency point is still a fine-tuned encoder, not a prompted generative LLM —
  directly relevant to whether detection should be an LLM call at all.
- **Results:** Setty: "superior performance over … GPT-4, GPT-3.5-Turbo, and
  Mistral-7b" across 90+ languages. CheckThat! 2025 winners used fine-tuned SLMs
  (podium in 15/20 languages).
- **Venue/date:** Setty — SIGIR 2024 industry track (Feb 2024); Majer & Šnajder —
  WASSA @ EMNLP 2024 (Apr 2024); winners survey — arXiv Sep 2025.
- **URLs:** https://arxiv.org/abs/2402.12147 · https://arxiv.org/pdf/2404.12174 · https://arxiv.org/abs/2509.11496

### 1.3 Prompt verbosity for detection is domain-dependent; LLM confidence ranks check-worthiness **[verified]**
- **Idea:** Across five CD/CW datasets (ClaimBuster, CheckThat! 2022,
  EnvironmentalClaims, NewsClaims, PoliClaim): **optimal prompt verbosity is
  domain-dependent** (more context helps ClaimBuster, hurts EnvironmentalClaims,
  no consistent trend elsewhere), **adding context does not uniformly help**, and
  **LLM confidence scores produce reliable check-worthiness rankings**.
- **Why it matters:** Cautions against a single "more detailed prompt = better"
  assumption; supports using model confidence as a *ranking* signal rather than a
  hard binary.
- **Venue/date:** Majer & Šnajder, WASSA @ EMNLP 2024 (Apr 2024).
- **URL:** https://arxiv.org/pdf/2404.12174

### 1.4 FactFinders — 1st on English check-worthiness via fine-tuned open LLMs + data pruning **[verified]**
- **Idea:** Ranked **1st** on English check-worthiness at CheckThat! 2024 Task 1
  (26 participants) with a fine-tuned **Llama2-7b**. Benchmarked 8 open LLMs
  (Llama 2, Mistral, Mixtral, Phi-2, Falcon, Gemma…) with fine-tuning + prompt
  engineering on political transcripts. Core contribution: a **two-step
  data-pruning** method that auto-selects high-quality training instances —
  competitive performance with ~44% of the training data.
- **Why it matters:** SOTA detection here is fine-tuned open-weights + careful
  data curation, not closed-LLM prompting. Data quality > data quantity.
- **Venue/date:** CLEF CheckThat! 2024 working notes, CEUR Vol-3740 (Jun 2024).
  *(1st-place result is self-reported but leaderboard-verifiable.)*
- **URL:** https://arxiv.org/pdf/2406.18297

### 1.5 One encoder for both check-worthiness AND harm — multi-label multilingual XLM-RoBERTa **[verified]**
- **Idea:** A single fine-tuned **XLM-RoBERTa-base** does **multi-label**
  classification: detect verifiable factual claims (check-worthiness) **and**
  harmful content simultaneously, across English + low-resource languages
  (Arabic, Bulgarian, Dutch, Polish, Czech, Slovak). Explicitly positioned as a
  low-inference-time encoder alternative to LLMs for fact-checker tooling.
- **Why it matters:** Directly relevant to fusing our scope/detection gate with a
  harm signal in one cheap model rather than separate stages.
- **Venue/date:** Kula & Gregor, arXiv Aug 2024.
- **URL:** https://arxiv.org/pdf/2408.06737

---

## Theme 2 — Claim extraction / decomposition / normalization

### 2.1 Fully atomic decomposition is *not* the right representation — "molecular facts" **[verified]**
- **Idea:** Fully atomic propositions can **lack the context** needed to interpret
  them. Proposed alternative — **molecular facts** — defined by two desiderata:
  **decontextuality** (how well a fact stands alone) and **minimality** (how
  little extra info is added). Molecular facts **outperform atomic facts in
  ambiguous settings** (74.7% vs 68.7% accuracy on ambiguous biographies).
- **Why it matters:** This is the conceptual pivot of the extraction literature.
  It directly informs how "decontextualized" should be scored — the goal is
  standalone-but-minimal, not maximally shredded.
- **Venue/date:** Gunjal & Durrett, Findings of EMNLP 2024 (Jun 2024). Reinforced
  by 2025 work (DnDScore, EMNLP 2025; Decontextualization & Decomposition for
  Factuality, EMNLP 2025; AFEV).
- **URL:** https://arxiv.org/pdf/2406.20079

### 2.2 Decomposition does NOT uniformly help verification — "Decomposition Dilemmas" **[verified]**
- **Idea:** Claim decomposition's effect on fact-checking is **inconsistent** —
  some studies report gains, others declines — revealing a **trade-off between
  accuracy gains and the noise introduced by decomposition**. This is the
  *emerging consensus* of 2025–2026 work.
- **Why it matters:** Over-decomposition is an empirically documented failure
  mode, not just a stylistic preference. Decomposing is a cost, not a free good.
- **Results / corroboration:** Hu et al. (NAACL 2025). Corroborated by
  arXiv:2503.15354, arXiv:2506.07446, and follow-ups finding decomposition helps
  "only when evidence is granular and strictly aligned … standard setups often
  degrade performance."
- **Venue/date:** Hu et al., **NAACL 2025** (arXiv Nov 2024, 2411.02400).
- **URLs:** https://aclanthology.org/2025.naacl-long.320/ · https://arxiv.org/html/2411.02400v1

### 2.3 Dynamic, iterative extraction guided by verified facts — AFEV ("Fact in Fragments") **[verified]**
- **Idea:** **AFEV** does **dynamic, iterative atomic-fact extraction**:
  decomposes a complex claim using **previously verified facts** to guide the next
  decomposition step. Contrasts with **static** decomposition that "prioritizes
  syntactic fragmentation over … claim intent" and, lacking supervision,
  **amplifies error propagation** in multi-hop reasoning.
- **Why it matters:** Points toward adaptive / feedback-driven granularity rather
  than one-shot decomposition — an answer to the over-decomposition problem.
- **Venue/date:** Zheng et al., arXiv Jun 2025 (also Expert Systems with
  Applications, S0957417425041879).
- **URL:** https://arxiv.org/html/2506.07446v1

### 2.4 Claim normalization ≠ summarization — the CheckThat! 2025 Task 2 standard **[verified]**
- **Idea:** Claim normalization = transform **noisy multilingual social-media
  posts** into **clear, self-contained, verifiable** claims. Explicitly distinct
  from generic abstractive summarization (which may omit facts or hallucinate);
  requires **entity resolution** so the claim is unambiguous in isolation (worked
  example: resolve "Bird" → the scooter company, not the animal). Standard traces
  to Sundriyal et al.'s ClaimNorm.
- **Why it matters:** Defines the target object of an extractor: standalone +
  verifiable + entity-resolved, **without** introducing facts not in the post —
  closely mirrors our own extraction constraints.
- **Venue/date:** CheckThat! 2025 Task 2; ClaimNorm "From Chaos to Clarity" —
  Findings of **EMNLP 2023** (arXiv:2310.14338).
- **URLs:** https://arxiv.org/abs/2503.14828 · https://arxiv.org/pdf/2511.05078 · https://arxiv.org/abs/2310.14338

### 2.5 A concrete 2025 normalization system — TIFIN (CheckThat! 2025 Task 2) **[verified]**
- **Idea:** Fine-tunes **Qwen3-14B** with **LoRA (4-bit)**; augments training with
  structured **5W1H** (Who/What/Where/When/Why/How) decomposition reasoning; uses
  **FAISS retrieval-augmented few-shot** prompting (top-5 cosine-similar
  examples).
- **Results:** METEOR 41.16 (English) → 15.21 (Marathi); 3rd (English), 4th
  (Dutch/Punjabi); +41.3% relative METEOR over baseline.
- **Why it matters:** Representative of the 2025 recipe — fine-tuned mid-size open
  model + structured reasoning scaffold + RAG few-shot.
- **Venue/date:** CheckThat! 2025 Task 2 working notes, arXiv Nov 2025.
- **URL:** https://arxiv.org/pdf/2511.05078

---

## Theme 3 — Harm / check-worthiness prioritization

### 3.1 FABLE remains the anchor — five magnitude dimensions **[verified]**
- **Idea:** FABLE operationalizes harm-based prioritization along five magnitude
  dimensions: **(social) Fragmentation, Actionability, Believability, Likelihood
  of spread, Exploitativeness** — a structured set of questions estimating a
  claim's potential harm/urgency. Concludes with a discussion of **computational
  approaches** to automate this prioritization.
- **Why it matters:** Still the reference framework for *why* to prioritize one
  claim over another. Our FABLE-based harm stage is aligned with current practice.
- **Venue/date:** Sehat et al., **CSCW 2024** (arXiv:2312.11678).
- **URL:** https://arxiv.org/abs/2312.11678

### 3.2 This is the thinnest theme — and a "first joint hate-speech + check-worthiness dataset" claim was REFUTED
- The search surfaced no strong **post-FABLE computational harm-ranking system**
  at high confidence. One candidate finding — that **WSF-ARG+** is the *first*
  dataset combining hate-speech and check-worthiness annotations
  (arXiv:2603.25269) — was **adversarially refuted (1-2 vote)** and dropped; do
  not cite it as a "first."
- A related harm-angle source (arXiv:2401.16558) was fetched but **[unverified]**;
  treat as a lead, not a result.
- **Takeaway:** Computational automation of harm/public-interest ranking beyond
  FABLE's manual framework is genuinely **under-explored** in 2025–2026 — a gap,
  and possibly an opportunity, for our work.
- **URLs:** https://arxiv.org/abs/2401.16558 (lead, unverified)

---

## Theme 4 — Evaluation of extraction quality

### 4.1 METEOR / single-gold overlap is a poor proxy for extraction quality **[verified]**
- **Idea:** On CheckThat! 2025 Task 2 (English) the UNH team found fine-tuning
  **FLAN-T5-Large (783M)** beat all prompting / in-context approaches **on
  METEOR** (0.5569 val / 0.37 test vs best prompting Claimify+Self-Refine on Grok3
  at 0.331 / 0.33). **But** the higher-METEOR fine-tuned outputs were often
  **subjectively worse** than lower-METEOR prompted claims, and the **gold claims
  themselves omitted verification-critical details** (e.g., a senator's actual
  salary, $174,000).
- **Why it matters:** This is the most pointed warning for *how we judge
  extraction*: optimizing overlap-with-one-gold-claim can actively reward worse,
  detail-stripped claims. It is direct evidence for an LLM-as-judge / rubric
  approach over n-gram overlap — and against trusting a single gold reference.
- **Venue/date:** UNH at CheckThat! 2025, arXiv Sep 2025 (team ranked 9th/17
  officially; the metric critique is the contribution, not the ranking).
- **URL:** https://arxiv.org/pdf/2509.06883

### 4.2 Additional evaluation sources surfaced **[fetched, unverified — leads]**
These were fetched and yielded candidate claims but were **not** individually
adversarially verified (below the verification cut / budget-filtered). Worth
reading directly when we design our extraction-quality judge; do not quote their
specifics as confirmed:
- **DnDScore — decontextualization for atomic-fact scoring**, EMNLP 2025 context
  — https://arxiv.org/html/2503.15354v2
- https://arxiv.org/abs/2502.10855 (extraction-quality evaluation angle)
- https://arxiv.org/html/2505.16973 (extraction-quality evaluation angle)
- CheckThat! 2025 working notes — https://ceur-ws.org/Vol-4038/paper_103.pdf
- https://www.mdpi.com/2227-7390/13/11/1778 (evaluation angle)

**Synthesis for this theme:** The field *recognizes* overlap metrics are
inadequate but has **not converged** on a replacement. LLM-as-judge rubrics
scoring faithfulness / atomicity / decontextualization / verifiability are
**emerging but not standardized** — which means our own fidelity / decontextualized
/ verifiability Likert judge is in step with the frontier rather than behind it.

---

## Theme 5 — End-to-end SOTA systems (detection → extraction → verification)

### 5.1 What the extraction stage looks like in 2025–2026 systems **[verified core + fetched leads]**
- **Verified backbone:** The most-cited extraction-stage moves in 2025–2026
  end-to-end work are (a) **normalization into self-contained verifiable claims**
  (Theme 2.4), (b) **molecular / minimal-but-decontextualized** facts over fully
  atomic ones (2.1), and (c) **dynamic/iterative** decomposition driven by
  verification feedback (AFEV, 2.3) — with growing caution that **over-decomposition
  hurts downstream verification** (2.2).
- **Fetched leads (unverified)** — recent end-to-end / survey sources to read for
  concrete pipeline architectures:
  - https://arxiv.org/abs/2601.02669 (end-to-end system, Jan 2026)
  - https://arxiv.org/abs/2506.17878 (end-to-end system, Jun 2025)
  - https://aclanthology.org/2025.knowledgenlp-1.26/ (KnowledgeNLP 2025)
  - **Survey / state-of-field** — https://arxiv.org/html/2502.04955v1 (Feb 2025)
- **Why it matters:** Confirms the staged architecture we already use; the
  contested choices are *granularity* and *how/whether to decompose before
  verifying*, not whether to extract at all.

---

## Consensus vs open debates

**Emerging consensus (2024–2026):**
- Fact-checking is a **staged pipeline**; detection, extraction/normalization, and
  verification are distinct, separately-benchmarked steps. **[verified]**
- For **detection** with labeled data, **fine-tuned encoders/SLMs** are the
  strongest accuracy/latency point; **prompted LLMs** win on label-free,
  cross-lingual, low-resource, and generative subtasks. **[verified]**
- **Fully atomic decomposition is not the target**: the goal is **decontextualized
  + minimal** ("molecular") claims; **over-decomposition injects noise** and does
  not reliably help verification. **[verified]**
- **Normalization ≠ summarization**: standalone, verifiable, entity-resolved,
  **no added facts**. **[verified]**
- **Overlap metrics (METEOR) are a poor proxy** for extraction quality; the field
  is moving toward rubric / LLM-as-judge evaluation. **[verified]**

**Open debates / gaps:**
- **Adaptive granularity:** No principled, automatic, per-claim way to choose
  decomposition granularity (molecular facts and AFEV both argue *against* fixed
  full atomicity, but the selection rule is unsettled).
- **No adopted replacement for METEOR/overlap** in extraction eval — no
  standardized LLM-as-judge rubric (faithfulness / atomicity / decontextualization
  / verifiability) the field agrees on.
- **Post-FABLE harm prioritization is under-developed** — little validated
  *computational* harm-ranking beyond FABLE's manual five-dimension framework;
  one "first joint dataset" claim was refuted.
- **Fine-tuned-beats-LLM is contested at the edges** — some 2025–2026 work shows
  large LLMs surpassing fine-tuned SLMs on certain veracity tasks; the result is
  setting-dependent, not absolute.

---

## Implications for our process (pointers only — design is a later step)

- **Staged cascade is well-aligned** with where the field landed (detection →
  extract/normalize → harm → verify). No architectural rethink implied by the SOTA.
- **Detection gate:** SOTA says a **fine-tuned encoder** (e.g. XLM-RoBERTa) is the
  strongest cheap detector, and a *single* multi-label encoder can do
  check-worthiness **and** harm together — worth weighing against our current
  LLM-in-one-pass detection (latency, multilingual EN+FR coverage, cost).
- **Our 4-prong "misinformation candidate" gate** is conceptually close to
  check-worthiness + harm fused; the multi-label-encoder line (1.5) and FABLE
  (3.1) are the closest external anchors to compare it against.
- **Extraction target:** "molecular" framing (decontextualized **+ minimal**,
  no added facts) matches our extraction constraints — and is an argument
  **against** aggressive atomic splitting. Watch for **over-decomposition** as a
  measurable failure mode.
- **Our per-claim judge** (fidelity / decontextualized / verifiability Likert) is
  *ahead of* overlap metrics like METEOR and **in step** with the emerging
  LLM-as-judge direction — but the field has **no standardized rubric**, so ours
  is a defensible bespoke choice, not a deviation. The UNH/METEOR finding (4.1) is
  strong external justification for not adopting overlap metrics.
- **"Decontextualized" scoring** should reward standalone-but-minimal, with entity
  resolution and **no introduced facts** — exactly the molecular-facts /
  ClaimNorm criteria.
- **Harm stage:** FABLE remains the reference; the **absence** of validated
  post-FABLE computational harm-ranking is a gap our work could speak to.
- **Quoted/implicit claims:** not directly resolved by the surfaced SOTA;
  normalization work assumes entity resolution but the quoted-amplification case
  (which our criterion handles explicitly) appears under-addressed in the
  literature — a possible differentiator.

---

## Source index

**Verified (3-vote adversarial, confirmed):**
- CheckThat! 2025 lab overview — https://arxiv.org/abs/2503.14828 (ECIR/CLEF 2025, Mar 2025)
- Setty, fine-tuned > LLM detection — https://arxiv.org/abs/2402.12147 (SIGIR 2024 industry)
- Majer & Šnajder, prompt verbosity / confidence ranking — https://arxiv.org/pdf/2404.12174 (WASSA@EMNLP 2024)
- FactFinders, CheckThat! 2024 Task 1 winner — https://arxiv.org/pdf/2406.18297 (CEUR Vol-3740, Jun 2024)
- Kula & Gregor, multi-label multilingual encoder — https://arxiv.org/pdf/2408.06737 (Aug 2024)
- Gunjal & Durrett, Molecular Facts — https://arxiv.org/pdf/2406.20079 (Findings of EMNLP 2024)
- Hu et al., Decomposition Dilemmas — https://aclanthology.org/2025.naacl-long.320/ (NAACL 2025)
- Zheng et al., AFEV — https://arxiv.org/html/2506.07446v1 (Jun 2025; ESWA)
- TIFIN, CheckThat! 2025 Task 2 — https://arxiv.org/pdf/2511.05078 (Nov 2025)
- UNH @ CheckThat! 2025, METEOR critique — https://arxiv.org/pdf/2509.06883 (Sep 2025)
- Sehat et al., FABLE — https://arxiv.org/abs/2312.11678 (CSCW 2024)
- Sundriyal et al., ClaimNorm — https://arxiv.org/abs/2310.14338 (Findings of EMNLP 2023)
- CheckThat! 2025 winners (SLMs) — https://arxiv.org/abs/2509.11496 (Sep 2025)

**Fetched, unverified (leads — read before relying):**
- https://arxiv.org/html/2503.15354v2 (DnDScore / decontextualization scoring)
- https://arxiv.org/abs/2502.10855 · https://arxiv.org/html/2505.16973 (extraction eval)
- https://ceur-ws.org/Vol-4038/paper_103.pdf (CheckThat! 2025 working notes)
- https://www.mdpi.com/2227-7390/13/11/1778 (extraction eval)
- https://arxiv.org/abs/2601.02669 · https://arxiv.org/abs/2506.17878 · https://aclanthology.org/2025.knowledgenlp-1.26/ (end-to-end)
- https://arxiv.org/html/2502.04955v1 (survey, Feb 2025)
- https://arxiv.org/abs/2401.16558 (harm-prioritization lead)

**Refuted (do NOT cite as stated):**
- WSF-ARG+ as "first dataset combining hate speech + check-worthiness" —
  https://arxiv.org/abs/2603.25269 (1-2 adversarial vote)
