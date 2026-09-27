# Analysis brief — distribution analysis for `ours` extractor on Bluesky

Scope: characterise `openai/gpt-oss-120b` claim-extraction behaviour on a representative
sample of modern social-media posts (Bluesky, N≈2,000). Outputs feed `analyze.py`. The
methodology below is grounded in a short survey of 2023-2026 claim-extraction and
summarization-fidelity literature (Min et al. 2023 FActScore; Metropolitansky et al. 2025
Claimify; Deutsch & Stickel 2025 *Claim Extraction for Fact-Checking*; CheckThat! 2024 Task 1
overview; LongEval 2023; FineSurE 2024; DecMetrics 2025; Counting on Consensus 2026).

## What the literature says

1. **Detection is reported as a confusion matrix + F1**. CheckThat! Task 1 uses F1 on the
   positive class; comprehensive write-ups add precision, recall, and the full TP/FP/FN/TN
   table. We have judge-as-oracle for detection (`judge_has_claim`) so this maps directly.
2. **Per-dimension Likert distributions are shown as stacked bars / counts per level**, not
   histograms with imputed means. Likert is ordinal: median and mode are the correct centre
   statistics (Statistics By Jim; rcompanion). Means are reported only as a secondary
   summary and never as the sole figure.
3. **Multi-claim handling**: dominant practice is to score each claim independently and
   report two views: (a) the **per-claim distribution** (every claim is one observation),
   (b) a **per-post aggregate** computed as the mean across that post's claims (Claimify,
   Deutsch & Stickel). Worst-of-K (min) is also reported when the downstream system fails
   on a single bad claim — relevant here because the nudge UI surfaces every extracted
   claim, so the user sees the worst one. We will report mean and min per post.
4. **Failure-mode taxonomy** in Claimify (2025): Coverage, Decontextualization,
   Informativeness, Redundancy, Invalidity. Our rubric maps to four of these (we dropped
   atomicity by design): fidelity↔Coverage-of-source, decontextualized↔Decontextualization,
   conciseness↔Informativeness/Redundancy, check-worthiness↔Invalidity. We will bucket
   low-scoring (≤2) claims into these named failure modes for qualitative review.
5. **Correlation with post features**: Spearman ρ is the default for ordinal scores against
   continuous features (FineSurE; LongEval). When features are mutually correlated (e.g.
   `n_chars` ↔ `n_tokens`), partial correlations or a single multivariate linear/ordinal
   regression are reported alongside marginal Spearman. Multi-claim adds non-independence,
   which the literature handles either by per-post aggregation (treats post as the unit) or
   by mixed-effects with a random intercept per post. We will do **per-post aggregation as
   primary** (simpler, matches the unit of decision) and flag mixed-effects as a follow-up
   if claim counts are highly skewed.
6. **Extractor↔evaluator disagreement is reported as a labelled disagreement set with
   sampled transcripts** (Claimify §6; CLAIMSCAN-2023 overview). Both quantitative (the
   confusion table) and qualitative (5-10 worked examples per cell). This is what we want.

## Analyses to run

`analyze.py` produces the following on `graded_posts_n{N}.parquet` and
`graded_claims_n{N}.parquet`:

A. **Detection block**
   - 2×2 confusion matrix on (`extractor_has_claim`, `judge_has_claim`).
   - Precision, recall, F1, accuracy, Cohen's κ.
   - Bar chart of `n_claims` (0,1,2,3,4+).

B. **Quality block (per-claim view, primary)**
   - 4 stacked-bar plots (one per Likert dim): count of 1–5 ratings.
   - Table: median, mode, mean, % ≥4 ("acceptable"), % ≤2 ("failure") per dimension.

C. **Quality block (per-post view, aggregated)**
   - For each Likert dim, mean and min across post's claims. Same summary table as (B).

D. **Feature correlations**
   - Spearman ρ between each Likert dim (per-post mean) and each numeric post feature.
   - Same matrix on per-post min, to expose worst-claim drivers.
   - Multivariate OLS per dim with the top-5 marginally correlated features, reporting
     coefficients and partial-R² to identify confounded vs independent drivers.
   - Heatmap; annotate cells with |ρ| ≥ 0.15 (Bonferroni-aware threshold for ~25 features).

E. **Failure-mode buckets**
   - Define a failure as any per-claim Likert ≤ 2 on any dimension. Bucket the
     low-scoring claim by which dimension fired (Coverage / Decontext / Concise /
     Check-worthiness). Report counts, % of all claims, and the top-3 distinguishing
     post features per bucket (mean feature value vs corpus mean, ranked by Cohen's d).

F. **Disagreement set**
   - All posts where `extractor_has_claim ≠ judge_has_claim` OR `detection_appropriate = False`.
   - Stratified sample of 5 transcripts per cell of the confusion matrix; dump
     `data/disagreement_examples.md` with post text, extracted claims, judge verdict.

G. **(If classical metrics present)** Spearman ρ between each classical metric and the
   per-claim LLM-judge dim it should track (e.g. NLI-entail ↔ fidelity; cosine ↔
   decontextualized).

## Outputs

All written under `src/eval/scripts/extraction_grading/results/`:
- `figures/*.png` — plots from (A)–(D).
- `tables/*.csv` — summary tables.
- `disagreement_examples.md`, `failure_buckets.md` — qualitative.
- `summary.md` — single-page rollup with the headline numbers.
