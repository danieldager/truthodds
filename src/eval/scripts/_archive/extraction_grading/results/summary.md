# Extraction-grading summary v4 (n=1000)

- posts analysed: 1000
- claims analysed: 177

## Detection (FP-precision; FN unmeasurable)
- positive extractions: 106  (of 1000 posts)
- TP (extractor=yes & any claim is misinfo_candidate): 92
- FP (extractor=yes & NO claim is misinfo_candidate): 14
- precision = 0.868
- TN (extractor=no): 894 (assumed correct, FN unmeasurable)

## Quality (per-claim Likert)
- **fidelity**: median=5.0, mean=4.85, %≥4=96%, %≤2=2%  (n=177)
- **decontextualized**: median=5.0, mean=4.47, %≥4=82%, %≤2=12%  (n=177)
- **verifiability**: median=5.0, mean=4.43, %≥4=81%, %≤2=6%  (n=177)

## Misinformation-candidate (per-claim bool)
- true: 153 (86%)
- false: 24
- n=177

## Failure rates (per-claim)
- fidelity: 4 claims (2.3%)
- decontextualized: 22 claims (12.4%)
- verifiability: 10 claims (5.6%)
- misinfo_candidate_false: 24 claims (13.6%)

## Disagreement (post-level)
- 14 FP posts (extractor=yes, all claims judged misinfo_candidate=false)
- 8 posts with partial agreement (some claims flagged, some not)

## Artefacts
- `figures/` — confusion, n_claims histogram, Likert distributions, misinfo_candidate bar, Spearman heatmaps
- `tables/` — detection summary, Likert summary, Spearman correlations, OLS, failure buckets
- `failure_buckets.md`, `disagreement_examples.md` — qualitative