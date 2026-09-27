# Tier-1 verdict cache — research brief

*Compiled 2026-06-05 from a `/deep-research` survey (5 angles, 23 sources fetched,
105 claims extracted, 25 adversarially verified → 21 confirmed / 4 refuted).
Confidence tags below are the survey's own (high / medium / low). This brief
feeds `tier1_cache_design.md` and `eval/scripts/cache_eval/`.*

## Question

Best practices for a **semantic verdict cache**: confidently-resolved claims
(text + 384-d MiniLM embedding + AVeriTeC verdict + sources/metadata) are stored
and reused for semantically-equivalent future claims. New claim → embed → cosine
nearest neighbors → confirm logical equivalence → reuse verdict → fold into the
neighborhood (recompute representative). Four areas: (1) claim equivalence /
matching, (2) online clustering + representative maintenance, (3) threshold
calibration + cache-quality eval, (4) storage/serving.

## Headline

**Cosine similarity is a retrieval filter, never the reuse decision.** Every
strand of the fact-check claim-matching literature treats matching as a
retrieval/ranking problem *distinct from* semantic textual similarity, and the
adversarial flips we care about (negation, number/entity/scope swaps) are
**near in embedding space but flip the verdict**. The reuse decision must be made
by a separate **logical-equivalence gate** (bidirectional NLI/entailment or an
LLM judge), which is exactly the layer that catches those flips. This is a
**two-stage** design: ANN (high-recall candidate retrieval) → equivalence gate
(precision arbiter). Our existing stub already anticipates this shape.

---

## Area 1 — Claim equivalence / claim matching

**1.1 Claim matching ≠ semantic similarity. Cosine alone is insufficient.** *(high)*
- CLEF CheckThat! Task 2 (Shaar et al. 2020, "That is a Known Lie") frames the
  task as *ranking already-verified claims* so the ones that verify the input
  rank on top — a retrieval/ranking task, not an STS task.
- Kazemi et al. 2021 define claim matching as "pairs of textual messages that
  can be served with one fact-check," which "does not always translate to message
  pairs having the same meanings" — explicitly distinct from paraphrase/STS.
- In CLEF-2021, the only team ranking by S-BERT cosine directly (DIPS) came
  **last** of the three English-tweet teams (MAP@5 0.787 vs 0.883 winner); on
  the harder political-debates subtask **no system beat the BM25 baseline**.
- Sources: `github.com/sshaar/clef2020-factchecking-task2`,
  `aclanthology.org/2020.acl-main.332`, `arxiv.org/pdf/2106.00853`,
  CLEF-2021 Task 2 overview (academia.edu/76177267).

**1.2 Best architecture is two-stage: cheap retrieval → fine-tuned re-ranker.** *(high)*
- CLEF-2021 best Arabic system: BM25 retrieve top-20 → AraBERT re-rank →
  MAP@5 0.908 vs 0.794 for the BM25 baseline. Shaar et al. 2020 report
  learning-to-rank gives "sizable improvements over state-of-the-art retrieval
  and textual similarity approaches."
- **Mapping to our cache:** pgvector ANN (cosine) = the retrieval stage; the
  equivalence gate = the decision stage.

**1.3 The equivalence gate: bidirectional NLI / entailment, run both directions.** *(high)*
- FACT-GPT (`arxiv.org/pdf/2310.09223`) operationalizes claim matching as a
  3-way NLI task (entailment / neutral / contradiction) between a post and a
  previously-debunked claim, using BM25+S-BERT *only* to surface candidates.
- Direction matters: "A→B" ≠ "B→A". FACT-GPT runs both orders and aggregates.
- **Recommendation:** require **mutual entailment** (A⊨B *and* B⊨A) to reuse a
  verdict. One-directional entailment is not equivalence.

**1.4 The adversarial flips need the logical gate — embeddings and vanilla NLI miss them.** *(high)*
- SOTA universal embeddings **lack negation awareness** and treat negated pairs
  as roughly similar (`arxiv.org/html/2504.00584v1`).
- Neural NLI models trained on general NLI data **fail systematically on
  negation** unless fine-tuned on negation-specific data (MoNLI,
  `arxiv.org/pdf/2004.14623`).
- Adversarially-written claims (FoolMeTwice, `arxiv.org/pdf/2506.04583`) are
  built to defeat one-shot semantic matching, requiring normalization/decomposition.
- **Implication:** the gate must be negation/quantity/entity/scope-robust — either
  an LLM judge *explicitly instructed* to check those four, or an NLI model
  fine-tuned on such variants. **Do not trust the contradiction case to
  embeddings or an unaugmented NLI model.**
- ⚠️ Caveat: the stronger framing "a sentence is more similar to its own
  negation than to a different sentence in 99.27% of cases" was **refuted (1-2)**
  in verification — cite the *lack-of-negation-awareness* conclusion, not that
  specific statistic.

**1.5 Normalize before embedding and before the gate.** *(high)*
- ClaimNorm (`arxiv.org/pdf/2310.14338`) decomposes noisy social posts into
  simpler normalized claims as a distinct upstream step.
- **Recommendation:** key the cache on **canonical/normalized** forms, not raw
  post text. We already do attribution-stripped canonicalization in
  `synth_representatives.py` — the same idea.

---

## Area 2 — Online clustering + representative maintenance

**2.1 Cluster growth = DP-means / leader threshold clustering.** *(high)*
- DP-means (`icml.cc/2012/papers/291.pdf`): behaves like k-means except a **new
  cluster forms whenever a point is farther than λ from every existing centroid**.
  Assignment is threshold-based: min distance > λ → spawn; else assign to nearest.
- This *is* the streaming pattern a verdict cache needs; **λ corresponds to the
  cosine-distance cut**. DP-means uses squared Euclidean — on L2-normalized
  embeddings (which we have), cosine is monotone with squared Euclidean, so the
  threshold logic transfers directly.

**2.2 Representative: prefer an LLM-synthesized canonical claim over raw centroid/medoid.** *(high)*
- k-LLMmeans (`arxiv.org/abs/2502.09667`) replaces numeric centroids with
  **LLM-generated textual summaries** as cluster representatives while keeping
  k-means assignment in embedding space; it has a **mini-batch variant for
  streaming**.
- **Recommendation for the cache:**
  - keep the **centroid embedding** for fast ANN assignment,
  - store an **LLM-synthesized canonical claim** as the human-readable, auditable
    neighborhood key (re-synthesized periodically as members are folded in),
  - use **mini-batch** updates for online folding.
  - This avoids online **medoid** recomputation cost *and* the staleness of
    keying on any single member claim.

---

## Area 3 — Threshold calibration + cache-quality eval

**3.1 Calibrate for precision, not recall; cosine is the recall filter, the gate is the precision arbiter.** *(medium — synthesized)*
- Because cosine alone is insufficient and flips on adversarial variants, the
  safe design uses a **generous (high-recall) cosine cut** for candidate
  retrieval and delegates the precision-critical *never-reuse-a-wrong-verdict*
  decision to the entailment/LLM-judge gate.
- **Evaluate** with controlled variants — negation, quantity (52%→62%), entity
  swap, scope (some→all) — and measure **gate precision per category**.

**3.2 Do NOT copy threshold numbers from the literature.** *(caveat, high)*
- Two quantitative threshold claims were **refuted** in verification: a fixed
  "F1 ≈ 0.37–0.53 for cosine matching" claim (**0-3**) and an ordinal-scale
  entity-swap example (**1-2**). Absolute cosine/F1 cut values from papers are
  unreliable here.
- **Any threshold must be re-derived empirically on our own variant set.** This
  is exactly what `eval/scripts/cache_eval/` is for.

---

## Area 4 — Storage / serving

**4.1 Neon+pgvector is appropriate; the false-hit risk is the embedding near-but-not-equivalent problem.** *(low — engineering best-practice, not corpus-backed)*
- The GPTCache-style semantic-cache failure mode (embedding-similarity hits that
  aren't logical equivalents) is exactly the problem the entailment gate
  neutralizes.
- Concrete serving guidance (validate against pgvector/Neon + GPTCache docs
  directly — *no primary source survived 3-vote verification here*):
  - normalize embeddings, use cosine ops;
  - **hnsw** = higher recall / lower query latency at higher build cost;
    **ivfflat** = cheaper build/memory. Choose per index-size + latency budget.
  - gate *every* ANN hit through the equivalence check before reuse;
  - attach a **TTL/recheck timestamp** per verdict; re-verify stale/volatile claims.

---

## Caveats (carry into design)

- **Area 4 (storage) and Area 3 (absolute thresholds) are the weakest-supported.**
  No GPTCache / pgvector-index / TTL primary source survived verification; treat
  those as standard engineering practice to validate against docs, not findings.
- Benchmark numbers (CLEF MAP@5) are historical shared-task results on
  tweets/debates — **relative orderings transfer; absolute scores do not** to our
  X-feed/AVeriTeC distribution.
- Most embedding evidence used `all-MiniLM-L6-v2`; our
  `paraphrase-multilingual-MiniLM-L12-v2` is the same S-BERT family with the same
  negation weakness — findings transfer, but the **multilingual (EN+FR) setting
  adds cross-lingual equivalence risk** not directly measured here.
- FACT-GPT's *contradiction* class models a post rebutting a debunked claim,
  which maps onto but isn't identical to the cache's verdict-flipping
  non-equivalence — hence the **mutual-entailment** safety requirement.

## Open questions (for the eval + design to resolve)

1. What cosine threshold (multilingual-MiniLM, 384-d) maximizes recall while the
   entailment gate holds precision ≈ 1.0 on a purpose-built variant set?
   **Must be measured.**
2. Which gate is best cost/accuracy here: a negation-robust fine-tuned NLI model
   run bidirectionally, vs an LLM-as-judge prompted to check
   negation/quantity/entity/scope? (No verified head-to-head exists.)
3. How do AVeriTeC 4-class verdicts interact with reuse — can a *Refuted* verdict
   be reused for a negated variant *by flipping*, or only reuse verbatim for
   entailment-equivalent claims? (Design decision — see design doc; default:
   verbatim reuse only.)
4. TTL/recheck cadence and staleness signals for misinformation verdicts — no
   surviving source quantifies verdict drift over time.

## Source list (by angle)

- **Claim-matching academic:** sshaar/clef2020-factchecking-task2;
  2020.acl-main.332; arxiv 2310.14338 (ClaimNorm); arxiv 2310.09223 (FACT-GPT);
  arxiv 2506.04583 (adversarial/FoolMeTwice).
- **Adversarial equivalence (NLI/LLM-judge):** 2020.acl-main.332; arxiv
  2106.00853; arxiv 2402.05904; arxiv 2004.14623 (MoNLI); arxiv 2504.00584
  (embedding negation awareness).
- **Online streaming clustering:** icml 2012/291 (DP-means);
  Wikipedia: Data_stream_clustering; arxiv 2502.09667 (k-LLMmeans).
- **Threshold calibration + cache eval:** 2020.findings-emnlp.117; arxiv
  2503.06648; mdpi 2073-431X/14/9/385.
- **LLM semantic cache serving:** 2023.nlposs-1.24 (GPTCache); AWS pgvector
  ivfflat-vs-hnsw deep-dive; plus blogs (portkey, tianpan, databricks,
  dataquest) — lower reliability.
