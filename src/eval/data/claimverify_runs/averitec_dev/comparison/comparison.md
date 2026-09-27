# AVeriTeC dev 500: paper, claim verifier, log-odds urn

All three systems are scored on the same 500 gold rows (491 unique claims, the 9 duplicate rows inherit their claim's prediction). Intervals are 95% percentile bootstraps over the gold rows, 1000 resamples, seed 0. The ClaimCheck number is taken from the paper, so it has no interval and cannot enter a paired test.

## Four-class accuracy, ClaimCheck convention

The convention maps our unsupported verdict to Refuted, which is what the paper does when it finds no support.

| system | 4-class accuracy | 95% CI |
| --- | --- | --- |
| ClaimCheck (paper) | 0.764 | not reported |
| top3 | 0.716 | 0.676 to 0.752 |
| all10 | 0.720 | 0.682 to 0.758 |
| noceil | 0.744 | 0.706 to 0.782 |

## Binary flag decision, map A (Supported passes, everything else flags)

| system | n | accuracy | flag recall | FPR | precision |
| --- | --- | --- | --- | --- | --- |
| top3 | 500 | 0.854 | 0.960 | 0.475 | 0.862 |
| all10 | 500 | 0.872 | 0.955 | 0.385 | 0.885 |
| noceil | 500 | 0.866 | 0.937 | 0.352 | 0.892 |
| urn 3-voice at the fitted 2% threshold | 500 | 0.562 | 0.434 | 0.041 | 0.970 |
| urn 3-voice matched to top3's FPR | 500 | 0.822 | 0.913 | 0.459 | 0.860 |
| urn 3-voice matched to all10's FPR | 500 | 0.824 | 0.889 | 0.377 | 0.880 |
| urn 3-voice matched to noceil's FPR | 500 | 0.822 | 0.878 | 0.352 | 0.885 |
| urn 7-flag at the fitted 2% threshold | 500 | 0.590 | 0.471 | 0.041 | 0.973 |
| urn 7-flag matched to top3's FPR | 500 | 0.830 | 0.929 | 0.475 | 0.858 |
| urn 7-flag matched to all10's FPR | 500 | 0.838 | 0.910 | 0.385 | 0.880 |
| urn 7-flag matched to noceil's FPR | 500 | 0.832 | 0.889 | 0.344 | 0.889 |

Urn coverage is 500 of 500 rows (491 claims); full coverage. The urn flags a claim when its log-odds score sits at or below the threshold, so a matched comparison holds the false positive rate at or below the verifier's and asks what recall is left.

## Urn ranking quality

ROC AUC for 3-voice on map A is 0.847 (0.808 to 0.883, n=500).
ROC AUC for 7-flag on map A is 0.868 (0.832 to 0.900, n=500).

| model | operating point | threshold | FPR | flag recall |
| --- | --- | --- | --- | --- |
| 3-voice | FPR budget 0.02 | -9.678 | 0.016 | 0.180 |
| 3-voice | FPR budget 0.05 | -4.206 | 0.041 | 0.442 |
| 3-voice | FPR budget 0.10 | -3.342 | 0.057 | 0.466 |
| 3-voice | FPR budget 0.20 | -1.797 | 0.123 | 0.630 |
| 3-voice | matched to top3 | 1.350 | 0.459 | 0.913 |
| 3-voice | matched to all10 | 0.529 | 0.377 | 0.889 |
| 3-voice | matched to noceil | -0.112 | 0.352 | 0.878 |
| 7-flag | FPR budget 0.02 | -8.170 | 0.016 | 0.257 |
| 7-flag | FPR budget 0.05 | -3.984 | 0.041 | 0.471 |
| 7-flag | FPR budget 0.10 | -2.202 | 0.082 | 0.558 |
| 7-flag | FPR budget 0.20 | -0.592 | 0.197 | 0.799 |
| 7-flag | matched to top3 | 3.190 | 0.475 | 0.929 |
| 7-flag | matched to all10 | 2.066 | 0.385 | 0.910 |
| 7-flag | matched to noceil | 1.448 | 0.344 | 0.889 |

## Paired tests

McNemar with an exact binomial on the discordant pairs.

| comparison | discordant | split | p |
| --- | --- | --- | --- |
| top3 vs all10 on 4-class convention | 54 | 26 / 28 | 0.8919 |
| top3 vs all10 on binary A | 35 | 13 / 22 | 0.1755 |
| top3 vs noceil on 4-class convention | 48 | 17 / 31 | 0.0595 |
| top3 vs noceil on binary A | 46 | 20 / 26 | 0.4614 |
| all10 vs noceil on 4-class convention | 64 | 26 / 38 | 0.1686 |
| all10 vs noceil on binary A | 45 | 24 / 21 | 0.7660 |
| top3 vs urn 3-voice at matched FPR on binary A | 68 | 42 / 26 | 0.0681 |
| top3 vs urn 7-flag at matched FPR on binary A | 58 | 35 / 23 | 0.1480 |
| all10 vs urn 3-voice at matched FPR on binary A | 74 | 49 / 25 | 0.0071 |
| all10 vs urn 7-flag at matched FPR on binary A | 67 | 42 / 25 | 0.0498 |
| noceil vs urn 3-voice at matched FPR on binary A | 82 | 52 / 30 | 0.0198 |
| noceil vs urn 7-flag at matched FPR on binary A | 79 | 48 / 31 | 0.0712 |

## Speed and cost per claim

| system | claims per minute | LLM calls | search calls | scrapes | tokens in | cost | uncached LLM p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top3 | 49.9 | 11.3 | 1.72 Serper plus 0.65 Exa | 5.9 | 19,931 | $0.00148 | 8.4s |
| all10 | 20.1 | 17.2 | 1.59 Serper plus 0.52 Exa | 13.7 | 34,066 | $0.00210 | 17.1s |
| noceil | 38.1 | 9.9 | 1.56 Serper plus 0.47 Exa | 5.3 | 17,805 | $0.00120 | 14.2s |
| urn | 16.0 | 10.3 | 1.00 Serper | 8.3 | 17,437 | $0.00130 | 9.6s |

The verifier's per-claim elapsed time is not a latency number. At K=100 claims in flight the median claim takes 113s of wall, which is queueing as much as work; the per-call number in the table is the median latency of uncached LLM calls, which is what a single claim would feel.

Verifier throughput comes from the harness wall clock (top3.log finish line); the urn's is estimated as sum(per-claim wall) / workers, at 32 workers, so treat it as an estimate.

## Date ceiling caveat

For top3, 15.4% of 843 Serper searches returned at least one hit published after the claim's date ceiling, 6.1% of all hits were post-ceiling, and 14.6% of hits carried no date at all, so the ceiling is a filter with leaks and not a guarantee.
The 318 Exa searches in top3 are not date instrumented, so their leakage is unmeasured rather than zero.
For all10, 16.0% of 781 Serper searches returned at least one hit published after the claim's date ceiling, 6.3% of all hits were post-ceiling, and 15.1% of hits carried no date at all, so the ceiling is a filter with leaks and not a guarantee.
The 257 Exa searches in all10 are not date instrumented, so their leakage is unmeasured rather than zero.
For noceil, 0.0% of 766 Serper searches returned at least one hit published after the claim's date ceiling, 0.0% of all hits were post-ceiling, and 39.9% of hits carried no date at all, so the ceiling is a filter with leaks and not a guarantee.
The 229 Exa searches in noceil are not date instrumented, so their leakage is unmeasured rather than zero.

## Slices

Binary A flag recall by gold class, on the rows each system covers. The other two pre-committed slices, whether the claim carries an original claim URL and the claim's month, are in comparison.json under slices, together with accuracy.

| gold class | rows | top3 | all10 | noceil | urn 3-voice @fitted | urn 3-voice @matched:top3 | urn 3-voice @matched:all10 | urn 3-voice @matched:noceil | urn 7-flag @fitted | urn 7-flag @matched:top3 | urn 7-flag @matched:all10 | urn 7-flag @matched:noceil |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Conflicting Evidence/Cherrypicking | 38 | 0.895 (n=38) | 0.868 (n=38) | 0.763 (n=38) | 0.289 (n=38) | 0.737 (n=38) | 0.711 (n=38) | 0.658 (n=38) | 0.289 (n=38) | 0.737 (n=38) | 0.737 (n=38) | 0.711 (n=38) |
| Not Enough Evidence | 35 | 0.886 (n=35) | 0.857 (n=35) | 0.829 (n=35) | 0.114 (n=35) | 0.743 (n=35) | 0.686 (n=35) | 0.686 (n=35) | 0.114 (n=35) | 0.829 (n=35) | 0.771 (n=35) | 0.686 (n=35) |
| Refuted | 305 | 0.977 (n=305) | 0.977 (n=305) | 0.970 (n=305) | 0.489 (n=305) | 0.954 (n=305) | 0.934 (n=305) | 0.928 (n=305) | 0.534 (n=305) | 0.964 (n=305) | 0.948 (n=305) | 0.934 (n=305) |
| Supported | 122 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
