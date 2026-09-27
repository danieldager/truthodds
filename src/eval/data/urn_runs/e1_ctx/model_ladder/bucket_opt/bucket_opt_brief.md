# Bucket optimisation brief — choosing the partition, not just the depth

## Objective

The truth-odds urn scores a claim as `sum over its documents of w[bucket(doc)]`, where
`w` is the Laplace log-LR of the bucket's document rate under gold-TRUE vs gold-FALSE.
Today the bucket is one of a *fixed* nesting: 3-voice ⊂ 7-flag ⊂ 28-cell (flag × tier).
Daniel's question is whether the partition itself can be **chosen from the data** —
merging some of the 28 cells, splitting none — to maximise honest out-of-fold AUC
subject to every surviving weight being decently estimated.

This is a bias–variance problem over *bucketings*. More cells = less bias (a PRIMARY
support really is different from an UNRATED support) and more variance (28 weights on
31,888 documents, with the thin cells estimated on tens of docs). The 28-cell already
buys +0.0171 AUC over 3-voice on gold and *loses* 3.7 points out of corpus, which is the
classic signature of over-fitting the partition to one corpus's retrieval composition.

## Prior art in this repo (checked before running)

- `quality_urn.py` already has **hard** and **soft shrinkage** of the 28 cells toward the
  7-flag parent (`MIN_CELL = 200`, `lam = m/(m+MIN_CELL)`, `m = min(class counts)`), and
  its own docstring records the objection I inherit: min-per-class support is the WRONG
  constraint here, because a strongly directional cell is *by construction* rare in one
  class (`5|rated` = 2,184 T / 89 F) and that imbalance IS the signal. The hard rule guts
  22/28 cells. I therefore do not reuse `MIN_CELL`; see the support constraint below.
- `quality_urn.py` also already **considered and dropped NewsGuard numeric bins**: `ng`
  covers ~37% of documents and UNRELIABLE holds ~249 of them, so bins would be mostly
  imputation. I do not re-litigate that; I test the two axes it did *not* try (retrieval
  rank, snippet-vs-full-read provenance), both already in the saved reads.
- `clog/270826.md` 19:40 records the only **merge probes** ever run: hand-picked, not
  searched — flag 3 → X (−0.0007), 3 → I (−0.0003), 3 dropped (−0.0006), all straddling
  zero; and the warning that folding X into I costs 3.9pp recall@2%. Daniel kept flag 3
  as semantically distinct. So: no systematic search over partitions has been done.
- `clog/270826.md` also pins the transfer inversion (0.852 / 0.844 / 0.816 urn-fitted).

## Candidate methods, one line each on why

- **Baselines** 3-voice / 7-flag / 12-cell / 28-cell — must reproduce the pinned ladder to
  ±0.001 or the whole exercise is untrustworthy.
- **Additive log-LR (flag + tier, ~10 params)** — the natural intermediate between 7 and
  28: it says tier shifts every flag by the same amount, buying the tier signal for 3
  extra parameters instead of 21.
- **L2 logistic on the count vectors (7 and 28)** — a discriminative reference point. It
  is a *different estimator* from the naive-Bayes log-LR, so it bounds what the 28 counts
  can do at all when regularised properly; `C` by inner CV.
- **Partial pooling toward the flag parent** — the statistically correct answer to "28
  cells but thin ones": keep all 28 buckets, shrink each toward its 7-flag marginal by
  `lam = tau^2/(tau^2 + se^2)`, `tau^2` estimated by moments within each flag (no tuning
  knob), plus a `soft(M)` variant with `M` chosen by inner CV for comparability with
  `quality_urn`.
- **Agglomerative merging (the direct answer to the question)** — greedily merge the pair
  of buckets whose separation is least supported by the data (smallest likelihood-ratio
  G² on the 2×2 of TRUE/FALSE doc counts), run in two regimes: *constrained* (only cells
  sharing a flag may merge, so we learn a per-flag tier grouping and never destroy the
  flag semantics Daniel defended) and *unconstrained* (any two of the 28). Two stopping
  rules: **support** (stop when every bucket meets the CI constraint) and **inner-CV**
  (stop at the merge count that maximises inner-fold AUC).
- **Extra raw axes** — rank bins and snippet/full-read provenance, both present in the
  saved reads, tested as a third axis feeding the merger. NG bins skipped, cited above.

## Support constraint

Not min-per-class counts. The quantity we need well-estimated is the *weight*, so the
constraint is on its interval directly. Analytic delta-method SE for the Laplace log-LR:

    se(w_b) ~= sqrt( 1/(cT_b + 1) + 1/(cF_b + 1) )     half-width h_b = 1.96 * se

`5|RELIABLE`-style imbalance is not penalised (2,184/89 → h ≈ 0.21) while a genuinely
thin cell is (5/3 → h ≈ 1.26). Documents within a claim are correlated, so the Poisson SE
understates; I calibrate an inflation factor once against the 2,000-rep bootstrap weight
CIs already in `ladder.json` and apply it inside folds (cheap). **Constraint: h_b ≤ 0.35**
for every bucket, reported alongside a sweep of h so the cost of the constraint is visible.

## Honest evaluation protocol

Nested, and the selection happens **inside** the training data at every level:

- Outer: the pinned 5 folds (blake2b of `review_url`, seed-free, same as the ladder).
- Inside each outer training set: the merge sequence is computed on the *training* counts
  only, and where a stopping rule needs AUC, an **inner 5-fold CV within that training
  set** picks the merge count / `C` / `M`. The outer test fold never touches selection.
- Weights are then refit on the full outer training set under the selected partition and
  applied to the held-out fold. Concatenating gives the oof score vector.
- 2,000-rep paired bootstrap, seed 707, resampled within gold class — **the entire
  procedure, selection included, is redone inside every replicate**, so the interval
  carries partition-selection variance and not just weight variance. All structures share
  replicates, so dAUC intervals are paired.
- Reported per structure: oof AUC + 95% CI, paired dAUC vs 7-flag + CI, recall@2% FPR,
  effective number of parameters (buckets used, or the trace-based count for the ridge
  fits), min and median weight CI half-width.

## Transfer check

Same machinery as `transfer_ladder.py` and reported at its headline `eps = 0.10`: fit the
structure's weights on the two urns (CN-false urn as FALSE, timeline urn as a mixed draw
de-mixed at eps), score the pinned fc-gold with **no refit**. For the data-driven
structures the partition is the one selected on the full gold data, held fixed — the
analogue of shipping a chosen bucketing. A partition that wins on gold and collapses here
is corpus-specific and should not be promoted, which is exactly what happened to the
28-cell.
