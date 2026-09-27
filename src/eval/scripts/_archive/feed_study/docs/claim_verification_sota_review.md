# State of the Art in Automated Claim Verification — Literature Review

**Scope:** Evidence-based, retrieval-augmented verification of textual claims (given a
claim, retrieve evidence and decide veracity). Prioritises 2025–2026 primary sources;
the AVeriTeC task and FEVER shared tasks are the reference points. **Research only** —
this surveys the field; assessing our own `verify.py` loop against it is a later,
collaborative step. The "implications" at the end are pointers, not recommendations.

**Compiled:** 2026-06-02. This review supersedes the prior inline-research draft. Every
quantitative figure below has been verified against the primary source (3-vote
adversarial verification); venue + date noted per work, citation URLs inline.

---

## Orientation: the canonical pipeline everyone now shares

By 2025 the field has converged on a **retrieve → reason → verdict** decomposition that
mirrors the human fact-checker workflow, almost always built around the **AVeriTeC** task
([Schlichtkrull et al., NeurIPS 2023 D&B, arXiv:2305.13117](https://arxiv.org/abs/2305.13117)
— real-world claims from 50 fact-checking orgs, each with question–answer evidence pairs
and a 4-class verdict). The shared modules:

1. **Claim/question planning** — turn the claim into search queries or sub-questions.
2. **Evidence retrieval** — web search and/or a fixed knowledge store; often iterative.
3. **Evidence processing** — per-document summarisation, answer reformulation, selection.
4. **Synthesis + stopping** — integrate evidence; decide "enough" vs "search again".
5. **Verdict + justification** — emit a label (usually the AVeriTeC 4-class) with a rationale.

The frontier in 2025–2026 is **not** the template — it is *efficiency*, *iteration
control*, *evidence-quality evaluation*, *robustness to conflicting/leaky evidence*, and
above all **the evaluation metric**, which changed under the field's feet in 2025 (see
Theme 6).

---

## Theme 1 — End-to-end verification systems & pipelines (2025–2026)

### ClaimCheck — small-model SOTA, four-stage pipeline
- **Method:** A transparent, stepwise, *modular* LLM-guided pipeline with four named
  stages: **(1) Web search query planning → (2) Web-based evidence retrieval and
  summarization → (3) evidence synthesis and re-retrieval → (4) claim verdict
  evaluation.** Each component performs a distinct function; the design mirrors human
  fact-checking. The paper sometimes expands stage 2 into separate retrieval/summarization
  steps ("five core stages"), but the four-stage framing is the abstract's exact wording.
  Uses **live web search** for evidence.
- **Why it matters:** Demonstrates SOTA-level verdict accuracy is reachable with a *small*
  model. Authors' ablations show **modular design and prompting — not model scale — drive
  performance**: "stepwise reasoning within synthesis and evaluation modules is crucial,"
  and "careful module design and prompting strategies can overcome the limitations
  associated with smaller LLMs." Hybrid-thinking ablation: all-think 75.0% vs all-no-think
  ~54%, so reasoning is load-bearing. Qwen3-32B scored 76.0% — scaling the model up did
  *not* help.
- **Results:** **76.4% verdict-prediction accuracy on AVeriTeC** with **Qwen3-4B (thinking
  enabled)**, beating self-run baselines HerO (75.2%), GPT-4o+search (74.6%), InFact
  (72.4%), PASS-FC (72.0%), DEFAME (70.5%) — systems relying on much larger models
  (LLaMA3.1-70B, GPT-4o) and pre-fetched knowledge stores.
- **⚠ Three load-bearing caveats on the 76.4% headline** (literally true but easy to misread):
  1. **It is plain 4-class verdict-prediction accuracy, NOT the official AVeriTeC-score /
     Ev2R recall** used on the FEVER leaderboard. It is therefore **not comparable** to the
     ~0.63 (FEVER-7 2024) or ~0.33 (FEVER-8 2025) leaderboard numbers; never juxtapose them.
  2. **Dev subset, not test set:** "a random subset of 100 claims from AVeriTeC's
     development dataset due to monetary and time constraints" (n=100).
  3. **Self-run baselines, narrow margin:** competitors re-run by the authors on the same
     100-claim subset; lead over HerO is ~1.2 pts. Appendix B notes label noise likely caps
     achievable accuracy near 80%. The GPT-4o competitor is also noted to benefit from
     temporal leakage that ClaimCheck removes via strict publication-date cutoffs.
- **Venue/date:** Putta, Devasier & Li (UT Arlington). arXiv:2510.01226, v1 22 Sep 2025.
- [arXiv:2510.01226](https://arxiv.org/abs/2510.01226)

### FIRE — confidence-gated iterative retrieve+verify
- **Method:** An **agent-based framework integrating evidence retrieval and verification
  into a single iterative loop**. A **unified mechanism decides, at each step, whether to
  output a final verdict or generate another search query — based on its confidence in the
  current judgment** (confidence-gated stopping). Contrasts with the traditional pipeline
  that fixes the number of evidence pieces *before* verifying ("retrieve-then-verify").
- **Why it matters:** The canonical reference for confidence-gated stopping; directly
  targets the cost/latency of fixed-round RAG.
- **Results:** **Slightly better performance while reducing LLM costs by an average of
  7.6× and search costs by 16.5×** vs the fixed-retrieve-then-verify baseline. ("Slightly
  better"/"average" are the authors' own framing; author-reported, not independently
  replicated.)
- **Venue/date:** Xie et al. (MBZUAI / Univ. Melbourne; incl. Preslav Nakov, Iryna
  Gurevych). **✅ Findings of NAACL 2025** (April 2025, pp. 2901–2914); arXiv:2411.00784.
- [arXiv:2411.00784](https://arxiv.org/abs/2411.00784) ·
  [ACL 2025.findings-naacl.158](https://aclanthology.org/2025.findings-naacl.158/)

### AVeriTeC / FEVER shared-task winners — the competitive frontier

**FEVER-7 (2024) — the 1st AVeriTeC shared task.**
- **Winner: team TUDA_MAI, AVeriTeC score 63%**, vs an **11% baseline**. **21 submissions,
  18 surpassed the baseline.** Scored under the *legacy* hu-METEOR-based AVeriTeC score.
- **Venue/date:** Schlichtkrull et al. (organizers), 7th FEVER Workshop, Oct 2024.
- [arXiv:2410.23850](https://arxiv.org/pdf/2410.23850) ·
  [ACL 2024.fever-1.1](https://aclanthology.org/2024.fever-1.1)

**FEVER-8 (2025) — the 2nd AVeriTeC shared task ("AVeriTeC 2.0").**
- **Winner: CTU AIC, AVeriTeC score 33.17% (0.332)** under the **new Ev2R-recall metric**,
  beating the 0.20 baseline and all others (6 of 7 submissions beat baseline). Runner-up
  **HerO 2 (Team HUMANE) 0.271**, then **yellow_flash 0.253**. The much lower absolute
  scores vs 2024 are **entirely a metric-change artifact** (see Theme 6), not regression.
- **Venue/date:** Akhtar, Aly, Chen, Deng, Schlichtkrull, Whitehouse, Vlachos (organizers),
  FEVER-8 workshop 2025.
- [ACL 2025.fever-1.15](https://aclanthology.org/2025.fever-1.15/)

### CTU AIC (FEVER-8 winner) — long-context RAG, single LLM call
- **Method:** A **simple two-step long-context RAG pipeline** (based on their 2024 entry),
  three modules: **(a) precomputation** — chunk knowledge sources (2048-char chunks) and
  embed (**mxbai-embed-large-v1**, stored in a **FAISS** index with exact search);
  **(b) retrieval** — embed the claim, retrieve k-NN (k=40), then **rerank with Maximal
  Marginal Relevance (MMR, λ=0.75)** to select 10 diverse sources; **(c) generation** — a
  **single LLM call** (**Qwen3-14B via Ollama/llama.cpp**) produces **Question-Answer-Source
  evidence triples plus Likert-scale scores for each of the four verdicts**, via
  chain-of-thought, ingesting up to **~60K characters** of evidence. The 4-class label is
  argmax over the Likert scores: Supported / Refuted / Not-enough-evidence /
  Conflicting-Cherrypicking.
- **Why it matters:** Shows a *minimal* one-shot RAG design wins under the new efficiency
  constraints — long context replaces elaborate iteration.
- **Results:** **1st on the FEVER-8 test leaderboard, new AVeriTeC score 0.33** (Q-only
  Ev2R 0.20; Q+A Ev2R 0.48) vs baseline 0.20 (Q+A 0.34); ran **~54s/claim** (≤60s budget,
  ~10% reserve). On the *old* hu-METEOR metric it scored **0.41 — BELOW the 0.50
  baseline.** It placed **3rd in 2024** and **4th on the 2025 dev leaderboard.** Authors:
  "The rise … from 3rd place … to 1st place in FEVER 8 without any major system change can
  therefore also be attributed to the used scoring method." (They did scale down to
  Qwen3-14B and drop knowledge-store pruning — "no *major* change," not zero change.)
- **Venue/date:** Ullrich & Drchal (AI Center, CTU FEE Prague), FEVER-8 workshop, 5 Aug 2025.
- [arXiv:2508.04390](https://arxiv.org/html/2508.04390v1) ·
  [ACL 2025.fever-1.22](https://aclanthology.org/2025.fever-1.22)

### HerO 2 (FEVER-8 runner-up) — heterogeneous, efficiency-tuned stack
- **Method:** A **heterogeneous LM stack**, each module a different model:
  - **Query expansion: Llama3.1 8B** for **HyDE-FC** (hypothetical-document expansion).
  - **Document summarization + question generation (+ answer reformulation): Qwen3 8B.**
  - **Veracity prediction: Qwen3 32B, 4-bit AWQ-quantized** (fits an A10G 23GB GPU).
- **Why it matters:** Canonical reference for (a) HyDE-style query expansion in
  fact-checking and (b) **quantization to meet the single-GPU efficiency constraint**.
- **Results:** **2nd place, AVeriTeC score 0.271±0.004** (behind CTU AIC 0.332, ahead of
  yellow_flash 0.253), and **fastest of the top three at 29.19s/claim** (vs CTU AIC 53.67s,
  yellow_flash 31.71s, baseline 33.88s) — well inside the 60s budget.
- **Venue/date:** Yoon et al. (Team HUMANE), 15 Jul 2025, FEVER-8 workshop.
- [arXiv:2507.11004](https://arxiv.org/abs/2507.11004) ·
  [ACL 2025.fever-1.16](https://aclanthology.org/2025.fever-1.16)

---

## Theme 2 — Evidence retrieval & query planning

- **HyDE-style query expansion:** HerO 2 uses **HyDE-FC** with **Llama3.1 8B** to generate
  a hypothetical document whose embedding drives retrieval — the clearest 2025 instance of
  HyDE in a competitive fact-checker ([arXiv:2507.11004](https://arxiv.org/abs/2507.11004)).
- **Query planning as an explicit stage:** ClaimCheck makes **"Web search query planning"**
  its first stage and **"evidence synthesis and re-retrieval"** its third — a planned first
  query plus a synthesis-driven second retrieval round
  ([arXiv:2510.01226](https://arxiv.org/abs/2510.01226)).
- **Iterative / confidence-gated retrieval:** FIRE integrates retrieval and verification in
  one loop, issuing a new query only when not yet confident — replacing fixed-round
  retrieval with confidence-gated multi-hop search
  ([NAACL Findings 2025 / arXiv:2411.00784](https://aclanthology.org/2025.findings-naacl.158/)).
- **Hybrid retrieval + reranking over a fixed store:** CTU AIC uses **kNN over a FAISS store
  + MMR reranking** to maximise source *diversity* (maximise pairwise embedding distance
  among results while minimising distance to the claim), then leans on **long context
  (~60K chars)** instead of many retrieval rounds
  ([arXiv:2508.04390](https://arxiv.org/html/2508.04390v1)).
- **Web vs fixed knowledge store** is a live design axis: FEVER-8 (2025) *mandated* a
  **precompiled knowledge store** (no live web) for reproducibility/efficiency; ClaimCheck
  and FIRE use **live web search**. Live web buys recency but invites temporal leakage
  (Theme 6).

---

## Theme 3 — Evidence processing

- **Per-document summarization** is standard: ClaimCheck's stage 2 is "Web-based evidence
  retrieval *and summarization*"; HerO 2 summarizes "each document into paragraph-level
  evidence candidates" with Qwen3 8B
  ([arXiv:2510.01226](https://arxiv.org/abs/2510.01226);
  [arXiv:2507.11004](https://arxiv.org/abs/2507.11004)).
- **Answer reformulation / QA-triple structuring:** HerO 2 uses Qwen3 8B for question
  generation *and* answer reformulation; CTU AIC structures all evidence as
  **Question-Answer-Source triples** in the single verdict call
  ([arXiv:2508.04390](https://arxiv.org/html/2508.04390v1)).
- **Selection / relevance filtering for diversity:** CTU AIC's **MMR reranking** is an
  explicit relevance-vs-diversity filter over retrieved chunks
  ([arXiv:2508.04390](https://arxiv.org/html/2508.04390v1)).
- **Contradictory / low-credibility sources (CONFACT):** CONFACT presents the **first
  systematic evaluation of RAG fact-checking under conflicting evidence** and introduces a
  dataset of **questions paired with conflicting information from sources of varying
  credibility** — each instance is "a claim paired with documents exhibiting conflicting
  stances, annotated with source-credibility ratings." Makes source-credibility modeling a
  first-class evaluation target. ("First" is the authors' narrowly-scoped framing; the
  related RAGuard benchmark, arXiv:2502.16101, targets *misleading*-evidence robustness, a
  distinct scope.)
- **Venue/date:** "Resolving Conflicting Evidence in Automated Fact-Checking: A Study on
  Retrieval-Augmented LLMs," **IJCAI 2025 (AI-and-Social-Good track)**, submitted 23 May
  2025. [arXiv:2505.17762](https://arxiv.org/abs/2505.17762).

---

## Theme 4 — Verdict prediction schemes

- **The AVeriTeC 4-class label set:** **supported / refuted / not enough evidence /
  conflicting evidence-cherrypicking.** For the *shared task* (unlike the original AVeriTeC
  dataset) participants were **not required to submit a justification**
  ([arXiv:2410.23850](https://arxiv.org/pdf/2410.23850)). This is the de-facto standard
  label set across every system surveyed.
- **In-context LLM prediction dominates over NLI heads:** all four leading 2025 systems
  (ClaimCheck, FIRE, CTU AIC, HerO 2) predict the verdict by **prompting an LLM over
  retrieved evidence**, not a dedicated NLI classifier.
- **Multi-dimensional / Likert scoring:** CTU AIC emits **Likert-scale scores for *each* of
  the four verdicts and takes the argmax**, rather than a single hard label — a
  graded-confidence scheme over the label set, and the closest published analogue to a
  per-class Likert verdict ([arXiv:2508.04390](https://arxiv.org/html/2508.04390v1)).
- **Behavior under insufficient/conflicting evidence:** the 4-class set explicitly separates
  **Not-Enough-Evidence** from **Refuted** and adds **Conflicting/Cherrypicking** — so
  absence-of-evidence and contradiction are distinct labels, not collapsed. CONFACT
  ([arXiv:2505.17762](https://arxiv.org/abs/2505.17762)) is the dedicated study of behavior
  under *conflicting* evidence of varying credibility.
- **Calibration / confidence as a control signal:** FIRE uses the model's **confidence in
  its current judgment** as the gating signal to stop or keep searching
  ([arXiv:2411.00784](https://arxiv.org/abs/2411.00784)) — confidence used operationally,
  not merely reported. (The verified claim set did not include independent primary evidence
  on verbal-confidence miscalibration or probabilistic-certainty methods — see Open Questions.)

---

## Theme 5 — Agentic / iterative loops, tool use, cost & latency

- **Confidence-gating vs fixed/budgeted rounds vs one-shot RAG** is the central 2025 axis:
  - **FIRE** = confidence-gated stopping (search again only if not confident), yielding
    **7.6× LLM and 16.5× search cost reduction** at slightly better accuracy vs fixed-round
    retrieve-then-verify ([arXiv:2411.00784](https://arxiv.org/abs/2411.00784)).
  - **ClaimCheck** = a bounded loop with **synthesis-driven re-retrieval** — agentic but
    capped ([arXiv:2510.01226](https://arxiv.org/abs/2510.01226)).
  - **CTU AIC** = effectively **one-shot** generation (single LLM call) over long context,
    trading iteration for context length ([arXiv:2508.04390](https://arxiv.org/html/2508.04390v1)).
- **The 2025 efficiency constraint shaped architecture.** FEVER-8 (AVeriTeC 2.0) mandated:
  **open-weights models only**, a **single GPU with 23GB RAM (Nvidia A10, g5.2xlarge)**,
  **≤1 minute per claim on average**, and **evidence from a precompiled knowledge store**
  ([ACL 2025.fever-1.15](https://aclanthology.org/2025.fever-1.15/)). This is why the
  winners look as they do:
  - **Quantization to meet the budget:** HerO 2's veracity model is **Qwen3 32B in 4-bit
    AWQ** to fit 23GB ([arXiv:2507.11004](https://arxiv.org/abs/2507.11004)).
  - **Latency in practice:** HerO 2 **29.19s/claim** (fastest of top 3); CTU AIC
    **~54s/claim** (10% reserve under the 60s cap) — same constraint met with opposite
    strategies (heterogeneous small models vs one long-context call).

---

## Theme 6 — Evaluation (the most consequential theme for reading the numbers)

- **The legacy AVeriTeC score (2024):** a claim counts as verified **only if BOTH (a) the
  verdict is correct AND (b) retrieved evidence clears a quality threshold** —
  operationalized as a **Q+A Hungarian-METEOR cutoff of 0.25**. Claims below 0.25 evidence
  score get an AVeriTeC score of **0**; above it, the score is verdict accuracy
  ([arXiv:2410.23850](https://arxiv.org/pdf/2410.23850)). Because evidence quality is
  coupled to the verdict, label-only accuracy (e.g. ClaimCheck's 76.4%) is a *different*
  number from the AVeriTeC score.
- **The metric changed in 2025 — Ev2R recall.** FEVER-8 replaced symbolic hu-METEOR
  comparison with **Ev2R, an LLM-as-a-judge recall metric** (Llama 3.3 70B grader),
  designed to fix noise and exploits (e.g. evidence duplication) in the legacy metric. Ev2R
  is the documented reason **2025 absolute scores (~0.33) sit far below 2024's (~0.63): a
  metric change, not regression.**
  - **Proof by the same system:** CTU AIC went **3rd (2024) → 1st (2025) with no major
    system change**; on the *old* metric it scored 0.41 (below the 0.50 baseline), and
    **neither the old hu-METEOR dev scores nor the 2025 dev leaderboard (where it placed
    4th) predicted its 1st-place finish** — the rank flip is attributable to the scoring
    method ([arXiv:2508.04390](https://arxiv.org/html/2508.04390v1)). Ev2R reference:
    [arXiv:2411.05375](https://arxiv.org/abs/2411.05375).
- **Temporal leakage in live-web retrieval:** ClaimCheck notes its GPT-4o+search competitor
  benefits from temporal data leakage that ClaimCheck removes via **strict
  publication-date cutoffs** — confirming date filters matter and that live-web numbers
  without date control are inflated ([arXiv:2510.01226](https://arxiv.org/abs/2510.01226)).
- **Datasets:** **AVeriTeC** (real-world claims, 4-class, QA evidence) remains the core
  benchmark; the FEVER-8 test set is **1,000 claims**. (The verified claim set did not
  contain primary detail on AVerImaTeC multimodal, OpenFactCheck, or Factcheck-Bench
  specifics — see Open Questions.)

---

## Consensus vs open debates

**Emerging consensus (multiple primary sources):**
- The **retrieve → reason → verdict** modular pipeline is universal; the **AVeriTeC 4-class
  label set** is the de-facto standard, and **in-context LLM verdict prediction** has
  displaced dedicated NLI classifiers.
- **Small / quantized open models + good module design beat frontier-scale models** on
  verdict accuracy (ClaimCheck Qwen3-4B; HerO 2 AWQ Qwen3-32B). Scale is not the lever;
  **prompting and stepwise reasoning are** (ClaimCheck ablations).
- **Evidence quality must be scored alongside the verdict** — both the legacy AVeriTeC score
  and Ev2R bind them together; label-only accuracy is not the AVeriTeC score.
- **Efficiency is now a first-class constraint** (FEVER-8: ≤1 min, 23GB, open weights),
  reshaping winners toward quantization plus either confidence-gated iteration or one
  long-context call.

**Open debates / unsettled:**
- **Confidence-gated iteration (FIRE) vs one-shot long-context RAG (CTU AIC) vs bounded
  multi-round (ClaimCheck)** — no consensus on the best iteration-control regime; each wins
  in its own setting/metric.
- **Live web vs fixed knowledge store** — recency/realism vs reproducibility/leakage
  control. FEVER-8 chose fixed store; ClaimCheck/FIRE chose web.
- **Handling conflicting / low-credibility evidence** is newly opened, not settled — CONFACT
  is framed as the *first* systematic study, implying the problem is unsolved.
- **Metric instability:** the 2024→2025 hu-METEOR→Ev2R switch shows leaderboard rankings are
  metric-sensitive; the "right" evidence-evaluation metric is still contested.

---

## Implications for our process (POINTERS ONLY — assessment is a later collaborative step)

- Our **plan-query → search → per-doc-summarize → synthesize-or-query (hard cap) → 4-class +
  Likert** design maps almost 1:1 onto **ClaimCheck's 4 stages** and **CTU AIC's
  Likert-per-class verdict** — pointer: we are squarely on the consensus template.
- **Our hard-cap loop sits between FIRE (confidence-gated) and CTU AIC (one-shot)** —
  pointer: the iteration-control regime is an open debate; a fixed cap is defensible but
  worth contrasting against confidence-gating.
- **Beware the metric:** "76.4% AVeriTeC" (ClaimCheck) is **label accuracy on a 100-claim
  dev subset**, NOT the AVeriTeC/Ev2R leaderboard score (~0.33 in 2025) — pointer: pick ONE
  metric and never juxtapose label-accuracy with leaderboard scores.
- **Temporal leakage is real in live-web retrieval** — pointer: strict publication-date
  cutoffs (as ClaimCheck uses, as our Tier 3 date-ceiling intends) are necessary, and date
  filters are known-leaky.
- **Conflicting / low-credibility evidence** (CONFACT) is an unsolved axis — pointer: our
  "synthesize-or-query" step is where source-credibility / conflict handling would live.
- **Efficiency reference:** FEVER-8 winners run **29–54s/claim on a single 23GB GPU** with
  quantized open models — pointer for any self-hosted budget we set.

---

## Source ledger (all primary, all verified against source text)

| Work | Venue / date | URL |
|---|---|---|
| ClaimCheck | arXiv, 22 Sep 2025 | https://arxiv.org/abs/2510.01226 |
| FIRE | Findings of NAACL 2025 (Apr 2025) | https://aclanthology.org/2025.findings-naacl.158/ · https://arxiv.org/abs/2411.00784 |
| AVeriTeC Shared Task (FEVER-7, 2024) | 7th FEVER Workshop, Oct 2024 | https://arxiv.org/pdf/2410.23850 · https://aclanthology.org/2024.fever-1.1 |
| 2nd AVeriTeC Shared Task (FEVER-8, 2025) | FEVER-8 Workshop 2025 | https://aclanthology.org/2025.fever-1.15/ |
| CTU AIC (FEVER-8 winner) | FEVER-8 Workshop, 5 Aug 2025 | https://arxiv.org/html/2508.04390v1 · https://aclanthology.org/2025.fever-1.22 |
| HerO 2 (Team HUMANE) | FEVER-8 Workshop, 15 Jul 2025 | https://arxiv.org/abs/2507.11004 · https://aclanthology.org/2025.fever-1.16 |
| CONFACT | IJCAI 2025 (AI & Social Good), 23 May 2025 | https://arxiv.org/abs/2505.17762 |
| Ev2R (metric) | arXiv (referenced) | https://arxiv.org/abs/2411.05375 |
| AVeriTeC dataset (anchor) | NeurIPS 2023 D&B | https://arxiv.org/abs/2305.13117 |
