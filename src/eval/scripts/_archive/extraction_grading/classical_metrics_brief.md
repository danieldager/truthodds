# Classical metrics for claim-extraction quality — brief

**Goal.** Cheap, reference-free, remote-only metrics on `(post, claim)` pairs that
proxy the LLM-judge rubric (fidelity, decontextualized, conciseness,
check-worthiness) so we can extend evaluation beyond a few thousand posts.

**Constraint.** HF Inference API only (no local heavy compute). Metric must be
joinable with `judgments_per_claim_n{N}.parquet` by `(post_id, claim_index)`.

## Literature snapshot (2023–2026)

The dominant family for reference-free factuality is **NLI-as-alignment**: cast
`source → target` as `premise → hypothesis` and use entailment probability as a
fidelity score. Three recent results matter:

- **FENICE** (Scirè et al., ACL 2024) — atomic-claim + NLI alignment; SOTA on
  AGGREFACT at 74.0 % BAcc, beating AlignScore (70.8) and QAFactEval (67.0).
- **MiniCheck-FT5** (Tang et al., EMNLP 2024) — 770 M Flan-T5 fine-tuned on
  synthetic factuality data; +4.3 pt BAcc over AlignScore on LLM-AggreFact;
  GPT-4-level accuracy at ~1/400 the cost. Not hosted on HF Inference.
- **Tang et al. "Do Automatic Factuality Metrics Measure Factuality?"** (2024)
  — warns that NLI-based metrics are *over-sensitive to benign edits*. Useful
  as a directional signal, not as a verdict.

For **embedding cosine** the consensus is unchanged: weak but cheap signal
for semantic drift; high-precision negative filter ("anything < τ is junk")
rather than a quality ranker.

For **NER / entity overlap** the recent angle is *check-worthiness*: a claim
that drops all named entities from its source is by construction not
check-worthy in isolation. Useful as a per-claim sanity check, orthogonal to
entailment.

## Why classical here, given an LLM-judge already exists

LLM-judge cost is ~$0.002–0.005 per claim. At N=2 000 that's $5–10 — fine. At
N=50 000+ (the trajectory for the Tier-0 study and the Bluesky firehose
analysis) it becomes the bottleneck. Classical metrics on the HF API are 10–
100× cheaper per call and run with no rate-limit drama. The point of this
session is to find the 2–3 that *track* the judge rubric well enough to scale.

## Recommended metrics

### 1. NLI entailment — `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- HF Inference task: `text-classification` on the sentence-pair `(post, claim)`.
- Score: `P(entailment) − P(contradiction)` ∈ [-1, 1].
- Targets: **fidelity** primarily; secondary signal on extractor false
  positives (low entailment + has_claim=true).
- Why this checkpoint: AGGREFACT-relevant pretraining (MNLI+FEVER+ANLI), well
  hosted on HF Inference (Moritz Laurer's checkpoints are reliably served via
  `hf-inference`), large enough to be useful, small enough to be cheap.
- Hypothesis: Spearman ρ ≈ 0.55–0.70 with `fidelity`; lower with
  `decontextualized`, near-zero with `conciseness`.

### 2. Semantic similarity cosine — `sentence-transformers/all-MiniLM-L6-v2`
- HF Inference task: `feature-extraction` on `post` and `claim` separately;
  cosine in client.
- Score: cos ∈ [-1, 1].
- Targets: catastrophic drift / hallucinated claims (extractor produced
  something semantically unrelated to the post).
- Why this checkpoint: tiny (22 M), free tier covers it, near-instant.
- Hypothesis: weakly correlated with fidelity (ρ ≈ 0.3–0.5); main value is a
  **low-cosine floor** that identifies clear-failure rows for triage.

### 3. NER overlap — `dslim/bert-base-NER`
- HF Inference task: `token-classification` on both `post` and `claim`.
- Score: `|entities(post) ∩ entities(claim)| / max(|entities(claim)|, 1)`
  (i.e. precision of the claim's entities against the post). Lowercased,
  type-agnostic match.
- Targets: **fidelity** from an entity-preservation angle, plus a weak
  **decontextualized** proxy (claim with zero entities tends to be vague).
- Hypothesis: ρ ≈ 0.30–0.50 with fidelity, distinct from NLI signal — entity
  *substitution* errors (Trump → Biden) tank NLI but not cosine; entity
  *dropping* tanks NER overlap but may leave NLI high.

## Out of scope (deferred)

- **MiniCheck-FT5 / Bespoke-MiniCheck-7B** — best-in-class on AGGREFACT but
  not hosted via HF Inference Providers. Reasonable Modal one-shot candidate
  later; not for this prototype.
- **FENICE pipeline** — strong but its own claim-extraction step double-counts
  ours; can re-evaluate after baseline results.
- **AlignScore** — superseded by MiniCheck; same hosting issue.

## Output

`remote_metrics.py --smoke` produces a parquet with columns:

```
post_id, claim_index,
nli_entailment, nli_contradiction, nli_score,
cosine_minilm,
ner_overlap, ner_post_count, ner_claim_count,
error
```

joinable with `judgments_per_claim_n{N}.parquet`. Validation: Spearman ρ
between each metric and the four judge dimensions, on the first n=50 run.
