# cache_eval — Tier-1 verdict cache quality evaluation

*Spec for TASKS.md "Evaluate Tier-1 cache quality (recall vs precision over
claim-distance)". Design context: `src/docs/tier1_cache_design.md`; evidence:
`src/docs/tier1_cache_research.md`.*

**Status: implemented + Run 1 complete (2026-06-05).** `generator.py` + `scorer.py`
work end-to-end; results + analysis in **`RESULTS.md`**. Headline: cosine-only is
unsafe (precision ~0.53); the bidirectional LLM gate restores precision to ~0.98
and catches negation/entity flips perfectly → `SIMILARITY_THRESHOLD ≈ 0.60` (the
cut is a recall lever; the gate owns precision). Note: `scorer.py` caches gate
results to `*_gated.parquet`, so re-sweeping thresholds costs no API calls.

## What this measures

The cache reuses a stored verdict for a new claim only when a **logical-equivalence
gate** confirms the new claim is equivalent to a cached one. Two failure modes:

- **False MISS** (recall loss): an equivalent variant fails to hit → wasted
  re-verification, lost cost savings. Tolerable.
- **False HIT** (precision loss): a *non-equivalent* variant reuses a cached
  verdict → **a wrong verdict served to a user**. This is the dangerous failure
  and the primary metric. The adversarial cases — negation, number/quantity swap,
  entity swap, scope change — are **near in embedding space but flip the verdict**
  (see research brief Area 1), so they are exactly what must MISS.

**Precision is the safety metric; recall is the cost metric.** We optimize the
cosine cut + gate so precision ≈ 1.0 (esp. on negation/number-swap), then take
whatever recall that allows.

## Protocol

1. **Seed** a set of "cached" claims with known verdicts. Source: correctly-verified
   AVeriTeC claims — join `verification_grading/data/claims_n100.parquet`
   (`claim_id, claim_text, gold_label`) with a verdicts parquet
   (`verdicts_n100_tavily.parquet`, `verdict_4class`) and keep rows where
   `verdict_4class == gold_label`. These become the cache contents.

2. **Generate variants at controlled distance** (LLM + rules), each gold-labeled:
   - **`equivalent` (should HIT):** paraphrase, synonym swap, word reorder,
     active↔passive, added filler. Low semantic distance, same verdict.
   - **`not_equivalent` (must MISS):** the four adversarial axes —
     | axis | transform | example |
     |---|---|---|
     | negation | insert/remove negation | "X happened" → "X did NOT happen" |
     | number | swap a quantity | "52%" → "62%" |
     | entity | swap a named entity | "PM Modi" → "Virat Kohli" |
     | scope | change quantifier | "some X" → "all X" |
   - Rule-based transforms are preferred for the adversarial axes (deterministic,
     unambiguous gold label); LLM for natural paraphrases.

3. **Measure distance** per variant, several ways (so we can plot HIT/MISS *over
   distance*): cosine (multilingual-MiniLM, via `pipeline.embedding.embed`),
   lexical (token Jaccard / edit distance), and the gold verdict-equivalence label.

4. **Run each variant through the lookup path:** ANN/cosine gate (varying the
   threshold) → equivalence gate → record HIT/MISS and which prong matched.

5. **Report:**
   - **recall** = equivalent variants that HIT,
   - **precision** = HITs that are truly equivalent (the safety metric),
   - both **as a function of cosine distance**, and **per adversarial category**
     (the negation / number-swap false-HIT rate is the headline),
   - → a **`SIMILARITY_THRESHOLD` recommendation** and confirmation the
     equivalence gate catches what cosine alone cannot.

## Deliverable

- `generator.py` — seed loader + variant generator → `variants.parquet`
  (`seed_claim_id, variant_text, axis, gold_equivalent, cosine, jaccard`).
- `scorer.py` — runs variants through the cache lookup path, sweeps the cosine
  threshold, prints recall/precision overall + per-axis + over-distance, and the
  threshold recommendation.

## Notes from the research (carry in)

- Do **not** copy a threshold number from the literature — two such claims were
  refuted in verification. Re-derive on this variant set.
- The gate must be **bidirectional** (mutual entailment) and **negation-robust**;
  if using an LLM judge, name the four axes explicitly in the prompt. Vanilla NLI
  fails on negation without negation-specific fine-tuning.
- Multilingual (EN+FR) adds cross-lingual equivalence risk not measured in the
  surveyed work — consider FR variants in a later pass.
