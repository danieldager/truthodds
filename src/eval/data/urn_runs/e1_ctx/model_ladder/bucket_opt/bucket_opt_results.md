# Choosing the bucketing from the data

**Question (Daniel).** "Here we've gone up to 28 classes and AUC was still improving,
albeit marginally. Is there another way to split up the classes, maybe merging a few,
basically an optimization such that we can have decent CIs on all the weights, and such
that we have the highest AUC possible, given the data that we have on hand?"

**What was run.** Seventeen bucket structures on one population and one protocol: the
pinned fc-gold cut (n=3,274, T 1,502 / F 1,772, 31,888 documents, media axis out,
`rating_subtype=mixed` set aside, `unprovable` scored FALSE at eval and out of every fit),
5 folds by blake2b of `review_url`, unshrunk Laplace log-LR unless the structure says
otherwise, 2,000-rep paired bootstrap resampled within gold class, seed 707. $0 — a local
refit of the saved E1 reads, no API calls.

**Honesty of the numbers.** Every selection the data makes — which cells to merge, how
deep to merge, the ridge `C`, the shrink `M` — happens on an inner 5-fold CV of the outer
training rows only. The outer test fold never informs the partition. The bootstrap redoes
the merge search inside every replicate and every fold, so the intervals carry
partition-selection variance, not just weight variance. The one deliberate exception: the
ridge `C` and the shrink `M` are frozen at their full-data inner-CV values inside the
bootstrap (2,000 × 5 × 5 × 7 logistic refits was not affordable); those two rows' intervals
are therefore slightly optimistic and are marked.

**Reproduction gate, run before anything else.** 3-voice 0.851887, 7-flag 0.861966,
28-cell 0.868983 all match `ladder.json` to |d| = 0.00e+00, the 12-cell matches issue-24's
0.8619 to 2.9e-06, and the three transfer figures match `transfer.json` to 0.00e+00.

**Prior art it builds on.** `quality_urn.py`'s hard/soft shrinkage (and its warning that
min-per-class support is the wrong constraint, since directional cells are *supposed* to be
class-imbalanced — `5|RELIABLE` is 1,953 T / 54 F and that imbalance is the signal);
`quality_urn.py`'s recorded decision to drop NewsGuard numeric bins; `clog/270826.md`'s
hand-picked merge probes (flag 3 → X, 3 → I, 3 dropped, all within noise; folding X into I
costs 3.9pp recall) and the transfer inversion. No systematic search over partitions had
been run before this.

**The support constraint.** Not min-per-class counts. The thing that must be
well-estimated is the *weight*, so the constraint is on its interval:
`se(w) ≈ sqrt(1/(cT+1) + 1/(cF+1))`, half-width `1.96 × 1.264 × se`, where **1.264** is the
median inflation of the analytic SE against the pinned 2,000-rep bootstrap (documents
inside one claim are correlated). Headline constraint `hw ≤ 0.35`; swept in Table 3.

### Table 1 — the 28 cells, diagnosed

Laplace log-LR per document, 95% interval from the pinned 2,000-rep bootstrap. `hw` is the interval half-width. Sorted by document share.

| cell | T docs | F docs | % docs | log-LR | 95% CI | hw |
|---|---:|---:|---:|---:|:---:|---:|
| `I \| UNRATED` | 2,928 | 4,554 | 24.35 | -0.346 | [-0.41, -0.28] | 0.067 |
| `I \| RELIABLE` | 2,440 | 2,921 | 17.45 | -0.085 | [-0.17, +0.00] | 0.085 |
| `I \| PRIMARY` | 2,302 | 3,046 | 17.41 | -0.185 | [-0.26, -0.11] | 0.076 |
| `5 \| RELIABLE` | 1,953 | 54 | 6.53 | +3.665 | [+3.31, +4.12] | 0.402 |
| `1 \| RELIABLE` | 120 | 1,432 | 5.05 | -2.377 | [-2.66, -2.14] | 0.261 |
| `X \| RELIABLE` | 746 | 531 | 4.16 | +0.434 | [+0.28, +0.60] | 0.160 |
| `X \| UNRATED` | 672 | 592 | 4.11 | +0.222 | [+0.07, +0.37] | 0.151 |
| `4 \| RELIABLE` | 811 | 129 | 3.06 | +1.927 | [+1.69, +2.21] | 0.258 |
| `5 \| UNRATED` | 849 | 86 | 3.04 | +2.374 | [+2.09, +2.69] | 0.301 |
| `1 \| UNRATED` | 61 | 852 | 2.97 | -2.527 | [-2.87, -2.24] | 0.320 |
| `X \| PRIMARY` | 426 | 379 | 2.62 | +0.212 | [+0.03, +0.39] | 0.178 |
| `4 \| UNRATED` | 473 | 177 | 2.12 | +1.074 | [+0.85, +1.32] | 0.237 |
| `1 \| PRIMARY` | 36 | 473 | 1.66 | -2.455 | [-3.09, -1.95] | 0.568 |
| `2 \| RELIABLE` | 163 | 289 | 1.47 | -0.475 | [-0.76, -0.20] | 0.279 |
| `2 \| UNRATED` | 88 | 210 | 0.97 | -0.768 | [-1.08, -0.49] | 0.293 |
| `5 \| PRIMARY` | 231 | 35 | 0.87 | +1.958 | [+1.47, +2.60] | 0.565 |
| `4 \| PRIMARY` | 170 | 71 | 0.78 | +0.960 | [+0.64, +1.30] | 0.330 |
| `2 \| PRIMARY` | 25 | 110 | 0.44 | -1.356 | [-1.81, -0.94] | 0.434 |
| `I \| UNRELIABLE` | 49 | 56 | 0.34 | -0.036 | [-0.48, +0.39] | 0.437 |
| `3 \| RELIABLE` | 23 | 25 | 0.16 | +0.015 | [-0.70, +0.67] | 0.684 |
| `3 \| UNRATED` | 11 | 28 | 0.13 | -0.787 | [-1.62, -0.08] | 0.770 |
| `X \| UNRELIABLE` | 15 | 11 | 0.08 | +0.383 | [-0.45, +1.31] | 0.882 |
| `5 \| UNRELIABLE` | 22 | 1 | 0.07 | +2.537 | [+1.64, +3.49] | 0.924 |
| `4 \| UNRELIABLE` | 14 | 5 | 0.06 | +1.011 | [+0.10, +2.10] | 1.003 |
| `3 \| PRIMARY` | 1 | 11 | 0.04 | -1.697 | [-2.90, -0.41] | 1.243 |
| `2 \| UNRELIABLE` | 2 | 6 | 0.03 | -0.752 | [-2.21, +0.60] | 1.407 |
| `1 \| UNRELIABLE` | 0 | 8 | 0.03 | -2.102 | [-2.62, -1.29] | 0.665 |
| `3 \| UNRELIABLE` | 0 | 1 | 0.00 | -0.598 | [-1.30, +0.11] | 0.705 |

Analytic delta-method SE understates the bootstrap by a median factor of **1.264** (documents inside one claim are correlated); that factor is applied wherever the constraint is evaluated inside a fold. **13 of 28** cells miss `hw <= 0.35`: `5 | PRIMARY`, `5 | UNRELIABLE`, `4 | UNRELIABLE`, `3 | PRIMARY`, `3 | RELIABLE`, `3 | UNRELIABLE`, `3 | UNRATED`, `X | UNRELIABLE`, `I | UNRELIABLE`, `2 | PRIMARY`, `2 | UNRELIABLE`, `1 | PRIMARY`, `1 | UNRELIABLE`.

**119 of 378** cell pairs have overlapping 95% intervals, so most of the 28-cell grid is not separating anything.


### Table 2 — every structure, same population, folds, seed and bootstrap

Out-of-fold AUC on the pinned fc-gold n=3,274; dAUC paired against 7-flag; `hw` = min / median / max 95% half-width across the structure's own weights; transfer = urn-fitted, gold-evaluated at eps = 0.10, no refit.

| structure | params | oof AUC | 95% CI | dAUC vs 7-flag | rec@2% FPR | hw min/med/max | transfer AUC |
|---|---:|---:|:---:|:---:|---:|:---:|---:|
| 3-voice | 3 | **0.8519** | [0.8388, 0.8643] | -0.0101 [-0.0152, -0.0036] | 34.0% | 0.03 / 0.15 / 0.15 | 0.8531 |
| 7-flag | 7 | **0.8620** | [0.8483, 0.8731] | +0.0000 [+0.0000, +0.0000] | 42.2% | 0.04 / 0.20 / 0.52 | 0.8466 |
| 12-cell (voice x tier) | 12 | **0.8619** | [0.8478, 0.8736] | -0.0001 [-0.0071, +0.0070] | 35.4% | 0.06 / 0.23 / 1.16 | 0.8209 |
| 28-cell (flag x tier) | 28 | **0.8690** | [0.8553, 0.8799] | +0.0070 [+0.0020, +0.0118] | 42.9% | 0.07 / 0.37 / 1.41 | 0.8229 |
| additive log-LR (flag+tier) | 10 | **0.8675** | [0.8541, 0.8784] | +0.0056 [+0.0013, +0.0101] | 41.4% | 0.07 / 0.21 / 0.48 | 0.8250 |
| logit-7 (L2) | 7 | **0.8637** | [0.8492, 0.8753] | +0.0018 [-0.0045, +0.0079] | 37.8% | 0.05 / 0.12 / 0.16 | n/a |
| logit-additive-10 (L2) | 10 | **0.8702** | [0.8535, 0.8807] | +0.0082 [-0.0008, +0.0136] | 39.4% | 0.04 / 0.14 / 0.30 | n/a |
| logit-28 (L2) | 28 | **0.8711** | [0.8537, 0.8799] | +0.0092 [-0.0015, +0.0134] | 40.1% | 0.02 / 0.21 / 0.36 | n/a |
| shrink-EB toward flag | 14.3 | **0.8640** | [0.8524, 0.8775] | +0.0021 [-0.0000, +0.0102] | 42.4% | 0.07 / 0.31 / 0.75 | 0.8255 |
| shrink-soft M by inner CV | 26.6 | **0.8688** | [0.8551, 0.8797] | +0.0068 [+0.0019, +0.0115] | 42.6% | 0.07 / 0.29 / 0.61 | 0.8235 |
| shrink-hard M=200 (quality_urn) | 13 | **0.8667** | [0.8529, 0.8776] | +0.0048 [+0.0001, +0.0092] | 40.3% | 0.04 / 0.21 / 0.52 | 0.8204 |
| merge within-flag, support | 16 | **0.8679** | [0.8544, 0.8793] | +0.0059 [+0.0016, +0.0111] | 41.8% | 0.07 / 0.25 / 0.52 | 0.8228 |
| merge within-flag, support+LRT | 13 | **0.8676** | [0.8544, 0.8794] | +0.0056 [+0.0013, +0.0111] | 42.0% | 0.07 / 0.22 / 0.52 | 0.8237 |
| merge any, support | 15 | **0.8683** | [0.8544, 0.8793] | +0.0063 [+0.0013, +0.0112] | 43.0% | 0.07 / 0.24 / 0.40 | 0.8224 |
| merge any, support+LRT | 11 | **0.8679** | [0.8544, 0.8792] | +0.0059 [+0.0010, +0.0112] | 43.3% | 0.06 / 0.22 / 0.40 | 0.8230 |
| merge within-flag, inner-CV | 10 | **0.8691** | [0.8549, 0.8799] | +0.0072 [+0.0005, +0.0119] | 42.2% | 0.05 / 0.22 / 0.52 | 0.8376 |
| merge any, inner-CV | 8 | **0.8692** | [0.8545, 0.8797] | +0.0072 [+0.0006, +0.0119] | 42.4% | 0.05 / 0.21 / 0.40 | 0.8363 |


### Table 3 — what the support constraint costs

Merge until every bucket's 95% half-width is at or below `h`, nested inside every fold.

| h | scope | buckets | oof AUC | rec@2% FPR | max hw reached |
|---:|---|---:|---:|---:|---:|
| 0.20 | within-flag | 11 | 0.8668 | 41.1% | 0.513 |
| 0.20 | any | 9 | 0.8623 | 38.1% | 0.183 |
| 0.25 | within-flag | 12 | 0.8668 | 41.1% | 0.513 |
| 0.25 | any | 10 | 0.8657 | 41.0% | 0.183 |
| 0.35 | within-flag | 16 | 0.8679 | 41.8% | 0.513 |
| 0.35 | any | 15 | 0.8683 | 43.0% | 0.348 |
| 0.50 | within-flag | 19 | 0.8688 | 42.8% | 0.513 |
| 0.50 | any | 18 | 0.8690 | 42.9% | 0.444 |
| 0.75 | within-flag | 20 | 0.8689 | 42.8% | 0.540 |
| 0.75 | any | 20 | 0.8691 | 42.8% | 0.701 |
| 1.00 | within-flag | 22 | 0.8691 | 42.8% | 0.946 |
| 1.00 | any | 22 | 0.8690 | 42.7% | 0.946 |


### Table 4 — the two untested raw axes

`rank` (retrieval position 1-10) and `provenance` (snippet vs scraped full read) are on every saved document. NewsGuard numeric bins are not retested: `quality_urn.py` already recorded that verdict (ng covers ~37% of documents).

| structure | base grid | oof AUC | 95% CI | dAUC vs 7-flag | rec@2% FPR |
|---|---:|---:|:---:|:---:|---:|
| 7-flag (reference) | 53 | 0.8620 | [0.8483, 0.8731] | +0.0000 [+0.0000, +0.0000] | 42.2% |
| 28-cell (reference) | 53 | 0.8690 | [0.8553, 0.8799] | +0.0070 [+0.0020, +0.0118] | 42.9% |
| 7-flag x provenance (14) | 53 | 0.8632 | [0.8498, 0.8742] | +0.0012 [-0.0011, +0.0043] | 42.1% |
| 7-flag x tier x prov (56) | 53 | 0.8686 | [0.8549, 0.8796] | +0.0067 [+0.0012, +0.0116] | 43.1% |
| 7-flag x rankbin (21) | 81 | 0.8616 | [0.8482, 0.8725] | -0.0003 [-0.0015, +0.0008] | 42.4% |
| merge-any inner-CV over 56 | 53 | 0.8684 | [0.8541, 0.8794] | +0.0064 [+0.0002, +0.0115] | 43.3% |
| merge-any inner-CV over 84 | 81 | 0.8692 | [0.8540, 0.8792] | +0.0072 [-0.0002, +0.0114] | 42.3% |


### The 10-bucket structure the constrained search chose

Weights and their 2,000-rep bootstrap intervals with that partition frozen. The tier axis survives in exactly three places; four of the seven flags keep no source split at all.

| bucket | T docs | F docs | log-LR | 95% CI | hw |
|---|---:|---:|---:|:---:|---:|
| `5 \| PRIMARY+UNRELIABLE+UNRATED` | 1,102 | 122 | +2.289 | [+2.01, +2.59] | 0.291 |
| `5 \| RELIABLE` | 1,953 | 54 | +3.665 | [+3.31, +4.12] | 0.401 |
| `4 \| PRIMARY+UNRELIABLE+UNRATED` | 657 | 253 | +1.047 | [+0.84, +1.27] | 0.216 |
| `4 \| RELIABLE` | 811 | 129 | +1.927 | [+1.69, +2.21] | 0.258 |
| `3 \| all tiers` | 35 | 65 | -0.511 | [-1.07, -0.02] | 0.523 |
| `X \| all tiers` | 1,859 | 1,513 | +0.301 | [+0.19, +0.41] | 0.111 |
| `I \| PRIMARY+RELIABLE+UNRELIABLE` | 4,791 | 6,023 | -0.134 | [-0.19, -0.08] | 0.055 |
| `I \| UNRATED` | 2,928 | 4,554 | -0.346 | [-0.41, -0.28] | 0.067 |
| `2 \| all tiers` | 278 | 615 | -0.697 | [-0.90, -0.50] | 0.201 |
| `1 \| all tiers` | 217 | 2,765 | -2.445 | [-2.68, -2.24] | 0.219 |


### The partitions the search actually chose


**merge within-flag, inner-CV** — 10 buckets on the full data, per-fold bucket counts [20, 15, 15, 10, 19], NOT identical across the five outer folds.

```
  5 | PRIMARY+5 | UNRELIABLE+5 | UNRATED
  5 | RELIABLE
  4 | PRIMARY+4 | UNRELIABLE+4 | UNRATED
  4 | RELIABLE
  3 | PRIMARY+3 | RELIABLE+3 | UNRELIABLE+3 | UNRATED
  X | PRIMARY+X | RELIABLE+X | UNRELIABLE+X | UNRATED
  I | PRIMARY+I | RELIABLE+I | UNRELIABLE
  I | UNRATED
  2 | PRIMARY+2 | RELIABLE+2 | UNRELIABLE+2 | UNRATED
  1 | PRIMARY+1 | RELIABLE+1 | UNRELIABLE+1 | UNRATED
```

**merge any, inner-CV** — 8 buckets on the full data, per-fold bucket counts [8, 14, 14, 8, 8], NOT identical across the five outer folds.

```
  5 | PRIMARY+5 | UNRELIABLE+5 | UNRATED+4 | RELIABLE
  5 | RELIABLE
  4 | PRIMARY+4 | UNRELIABLE+4 | UNRATED
  3 | PRIMARY+1 | PRIMARY+1 | RELIABLE+1 | UNRELIABLE+1 | UNRATED
  3 | RELIABLE+I | PRIMARY+I | RELIABLE+I | UNRELIABLE
  3 | UNRELIABLE+3 | UNRATED+2 | PRIMARY+2 | UNRELIABLE+2 | UNRATED
  X | PRIMARY+X | RELIABLE+X | UNRELIABLE+X | UNRATED
  I | UNRATED+2 | RELIABLE
```

**merge within-flag, support** — 16 buckets on the full data, per-fold bucket counts [13, 14, 12, 13, 13], NOT identical across the five outer folds.

```
  5 | PRIMARY+5 | UNRELIABLE+5 | UNRATED
  5 | RELIABLE
  4 | PRIMARY
  4 | RELIABLE
  4 | UNRELIABLE+4 | UNRATED
  3 | PRIMARY+3 | RELIABLE+3 | UNRELIABLE+3 | UNRATED
  X | PRIMARY
  X | RELIABLE+X | UNRELIABLE
  X | UNRATED
  I | PRIMARY
  I | RELIABLE+I | UNRELIABLE
  I | UNRATED
  2 | PRIMARY+2 | UNRELIABLE+2 | UNRATED
  2 | RELIABLE
  1 | PRIMARY+1 | UNRELIABLE+1 | UNRATED
  1 | RELIABLE
```

**merge any, support** — 15 buckets on the full data, per-fold bucket counts [12, 13, 11, 11, 13], NOT identical across the five outer folds.

```
  5 | PRIMARY+4 | RELIABLE
  5 | RELIABLE
  5 | UNRELIABLE+5 | UNRATED
  4 | PRIMARY
  4 | UNRELIABLE+4 | UNRATED
  3 | PRIMARY+1 | RELIABLE
  3 | RELIABLE+I | RELIABLE+I | UNRELIABLE
  3 | UNRELIABLE+3 | UNRATED+2 | PRIMARY+2 | UNRELIABLE+2 | UNRATED
  X | PRIMARY
  X | RELIABLE+X | UNRELIABLE
  X | UNRATED
  I | PRIMARY
  I | UNRATED
  2 | RELIABLE
  1 | PRIMARY+1 | UNRELIABLE+1 | UNRATED
```

**merge any, support+LRT** — 11 buckets on the full data, per-fold bucket counts [8, 10, 10, 8, 10], NOT identical across the five outer folds.

```
  5 | PRIMARY+4 | RELIABLE
  5 | RELIABLE
  5 | UNRELIABLE+5 | UNRATED
  4 | PRIMARY+4 | UNRELIABLE+4 | UNRATED
  3 | PRIMARY+1 | PRIMARY+1 | RELIABLE+1 | UNRELIABLE+1 | UNRATED
  3 | RELIABLE+I | RELIABLE+I | UNRELIABLE
  3 | UNRELIABLE+3 | UNRATED+2 | PRIMARY+2 | UNRELIABLE+2 | UNRATED
  X | PRIMARY+X | UNRATED
  X | RELIABLE+X | UNRELIABLE
  I | PRIMARY
  I | UNRATED+2 | RELIABLE
```

---

## The answer, plainly

Yes. Merging finds a much cheaper structure than the 28-cell, but the gain over 7-flag is
smaller than it looks and it is corpus-specific.

**Best honest oof AUC among log-LR structures: within-flag agglomerative merging, depth by
inner CV — 0.8691 on 10 weights.** It matches the 28-cell's 0.8690 to 0.0001 while cutting
the median weight half-width from 0.37 to 0.22 and the worst from 1.41 to 0.52. It keeps
the source axis in three places only (flag 5 RELIABLE vs rest, 4 RELIABLE vs rest, I
UNRATED vs rest) and drops it for flags 3, X, 2, 1. The L2 logistic fits score a shade
higher (0.8711 / 0.8702) but straddle zero against 7-flag and lose 2-3pp of recall.

**On transfer nothing fine wins.** The ranking inverts: 3-voice 0.8531 > 7-flag 0.8466 >
merged-10 0.8376 > 28-cell 0.8229.

**The constraint is nearly free** above `h = 0.35` (0.8679 at 16 buckets vs 0.8690 at 28)
and costs ~0.003 AUC / ~4pp recall below `h = 0.20`.

**Is the gain beyond 7-flag real?** On gold, barely: +0.0072 [+0.0005, +0.0119], only just
clear of zero. Out of corpus it is negative. **This data supports about 10 well-estimated
weights, not 28.**

### Notes behind that answer

- One weight cannot be fixed by merging inside its flag: flag 3 has 35 TRUE and 65 FALSE
  documents in total, `hw = 0.52`. Only merging it into a *different* flag tightens it —
  the move Daniel already rejected on semantic grounds in `clog/270826.md`. Unconstrained
  merging duly does exactly that, pooling `3 | PRIMARY` with the whole of flag 1 and
  `I | UNRATED` with `2 | RELIABLE`; those partitions score the same as the constrained
  ones and are not defensible.
- The chosen depth is not stable across outer folds (10 on the full data; 20, 15, 15, 10,
  19 per fold), which is another way of saying the AUC curve is flat from ~10 to ~20
  buckets. The *shape* is stable: every fold keeps the RELIABLE split on the support flags
  and collapses the tier axis on the refute flags.
- Neither untested raw axis helps. Retrieval rank is inert (dAUC -0.0003 [-0.0015,
  +0.0008]); snippet-vs-full-read is worth at most +0.0012 [-0.0011, +0.0043]. Merging over
  the 56- and 84-cell grids lands back on the 28-cell's number.
- Partial pooling is not the answer here either: empirical-Bayes shrinkage toward the flag
  parent lands at 0.8640 with an effective 14.3 parameters, below the merged-10 on both AUC
  and interval width, because it must keep all 28 buckets and pays for each.
- Recommendation, for Daniel to weigh: 7-flag stays the production choice on the transfer
  evidence; merged-10 is the only fine structure worth testing next, since it is the only
  one that both matches the 28-cell on gold and keeps most of its transfer. Promoting
  anything at 28 buckets is not supported by this table.

### Files

`bucket_opt_brief.md` (protocol, written before running), `bucket_opt_diag.json` (Table 1 +
the SE calibration), `bucket_opt_results.json` (every structure, CIs, chosen partitions,
constraint sweep), `bucket_opt_extra_axes.json` (Table 4), `bucket_opt_oof.npy` (per-claim
oof scores, structures in Table 2 order). Generators: `bucketopt_core.py`,
`bucketopt_structs.py`, `bucketopt_run.py`, `extra_axes.py`, `diag28.py`, `make_tables.py`.
Nothing was written into the repo.
