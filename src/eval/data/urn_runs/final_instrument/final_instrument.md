# Final Instrument for Survey Experiment

## Two ways to fit the weights

| Fit | What the search sees | AUC | 95% CI width | AVeriTeC AUC | 95% CI width |
|---|---|--:|--:|--:|--:|
| **Training mode** | Documents dated before the claim. Fact-checks removed. | 0.857 | 0.024 | 0.869 | 0.074 |
| Production mode | Any date. Fact-checks removed. | 0.924 | 0.020 | 0.858 | 0.074 |

## What we found in production mode

Production mode scores seven points higher. We hand read 60 refuting documents; none leaked a fact-check. In training mode ten irrelevant documents happen to 333 true claims and 431 false ones; in production silence drops to 46 true and stays at 305 false, so the irrelevant weight moves to −0.59 from −0.19. Ten irrelevant documents score −5.9 (boundary −6.7); one contradiction among nine scores −7.6 and is flagged.

*Figure (weights, training vs production) — not regenerable from committed files (needs untracked production weights).*

## Each weight set on the other evidence

| Weights | Evidence | AUC | 95% CI width | Recall at 2% FPR |
|---|---|--:|--:|--:|
| Training mode | Training mode | 0.857 | 0.024 | 41.7% |
| Production mode | Training mode | 0.840 | 0.026 | 40.5% |
| Training mode | Production mode | 0.927 | 0.020 | 37.2% |
| Production mode | Production mode | 0.924 | 0.020 | 38.9% |

Production weights on day-zero evidence lose 0.016; training weights on production evidence come out 0.003 above the production fit.

## The evidence production will actually see

Median age 29 days, 82% older than two weeks, 1,660 survey claims.

| Weights applied to the mixed evidence | AUC | 95% CI width | Recall at 2% FPR | Boundary |
|---|--:|--:|--:|--:|
| Fitted on the mix | 0.894 | 0.023 | 35.3% | −4.61 |
| Training weights | 0.897 | 0.022 | 32.5% | −3.99 |
| Production weights | 0.893 | 0.022 | 34.9% | −6.23 |

Training weights come out 0.004 ahead of the mix fit. Margin above the silent pile: 2.1 (training), 1.3 (mix), 0.3 (production).

## Which weights we freeze

Training weights, boundary −4.0 (−4.0 on the mix, −4.1 on production).

> **Frozen instrument.** Seven flags, read-v5 on DeepSeek-V4-Flash, Serper top ten, training-mode weights, boundary −4.0. AUC 0.857 on fc-gold in training mode, 0.869 on AVeriTeC. Expected on production-age evidence, AUC 0.897.

## The reader

120 documents hand read, 30 per cell, two judges.

| Document | Reader errors, judge one | Reader errors, blind judge |
|---|--:|--:|
| Refuting flag on a false claim | 0 of 30 | 3 of 30 |
| Supporting flag on a true claim | 0 of 30 | 0 of 30 |
| Refuting flag on a true claim | 23 of 30 | 22 of 30 |
| Supporting flag on a false claim | 14 of 30 | 11 of 30 |

Disagreeing cells are 5% of documents on true claims and 7% on false claims, ~500 wrong refutations of true claims and ~350 wrong supports of false claims; no leaks in the 60 refuting documents. 22 of 37 errors were off-scope grading, 8 were quoted-claim reads.

### What fixing the reader would be worth

| Training mode evidence | AUC | Recall at 2% FPR | False flagged | True flagged |
|---|--:|--:|--:|--:|
| Frozen instrument | 0.860 | 38.2% | 573 | 31 |
| Reader errors corrected | 0.892 | 50.1% | 751 | 13 |

| Production mode evidence | AUC | Recall at 2% FPR | False flagged | True flagged |
|---|--:|--:|--:|--:|
| Frozen instrument | 0.927 | 37.2% | 558 | 30 |
| Reader errors corrected | 0.962 | 53.8% | 807 | 12 |

Training: 178 more false (573 to 751), 18 fewer true (31 to 13). Production: 249 more false (558 to 807).

### A stronger reader on the same claims

| Reader | Prompt | Claims | AUC frozen | AUC own | Own-weights diff vs current |
|---|---|--:|--:|--:|--:|
| **Flash, current** | v5 | 500 | 0.873 | 0.866 | |
| V4-Pro | v5 | 500 | 0.863 | 0.851 | −0.014 [−0.037, +0.010] |
| V4-Pro | v5b | 500 | 0.853 | 0.857 | −0.008 [−0.031, +0.015] |
| **Flash, current** | v5 | 1,000 | 0.862 | 0.858 | |
| V4-Pro | v5b | 1,000 | 0.851 | 0.849 | −0.009 [−0.026, +0.008] |
| Kimi K2.6 | v5 | 500 | 0.819 | 0.848 | −0.019 [−0.054, +0.015] |

V4-Pro withdrew 270 votes (214 right), added 112 (77 right). v5b added 454 against 416 withdrawn; 153 of 518 new strong refutations land on true claims. Kimi withdrew 419 (336 right), added 95.

## Retrieval

| Frozen weights, at the 2% boundary | Training, recall | Production, recall |
|---|--:|--:|
| Frozen instrument | 38.2% | 37.2% |
| Reader errors corrected | 50.1% | 53.8% |
| One contradicting document to every false claim that had none | 50.0% | 40.9% |
| Both | 87.5% | 84.1% |

### What a second search engine actually bought

Exa on 1,500 claims: a quarter more refuting documents on false claims. Serper points the right way seven times for every wrong one, Exa six. Same ratio, same AUC; Exa alone a point below Serper, both together tie Serper. 60 refuting documents hand checked per engine, no leaks.

*Figure (collab_exa_simple: right-way vs wrong-way documents per claim) — not regenerable from committed files (needs untracked prodregime reads).*

## What we freeze

- Seven flags, read-v5 on DeepSeek-V4-Flash, Serper top ten.
- Training-mode weights, fitted on 3,000 balanced fc-gold claims, boundary −4.0.
- Evaluation number, AUC 0.857 on fc-gold in training mode and 0.869 on AVeriTeC.
- No silence rule. Ten irrelevant documents never flag, one contradiction among nine does.
- No stronger reader model, no second search engine.
