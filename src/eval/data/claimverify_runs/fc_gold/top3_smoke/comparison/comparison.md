# fc-gold: the verify loop against the log-odds urn

The loop run is `eval/data/claimverify_runs/fc_gold/top3_smoke`; the urn scores are `src/eval/data/urn_runs/e1_ctx/model_ladder/oof_scores.parquet`. The population is the urn's 3274 out-of-fold rows. The loop has 50 usable records (0 failed), 3224 population rows are missing from it, and everything below is computed on the 50 claims both systems cover: 23 flag-worthy (gold_true False, mixed counted as false, including 2 veracity-3 unprovable rows) and 27 pass-worthy. Intervals are 95% percentile bootstraps over claims, 1000 resamples, seed 0, one shared index matrix so every difference is paired.

Missing from the loop run (first 10):

- https://www.snopes.com/fact-check/patrick-mahomes-did-not-test-positive-for-drugs-after-super-bowl-lvii/
- https://www.boomlive.in/world/president-donald-trump-falsely-claims-us-election-has-been-rigged-10529
- https://www.snopes.com/fact-check/bell-hooks-solitary-art-of-loving/
- https://www.boomlive.in/fact-check/who-herd-immunity-covid-19-definition-change-not-to-boost-covid-19-vaccination-russell-okung-11883
- https://www.politifact.com/factchecks/2020/may/22/donald-trump/farm-income-federal-aid-accounts-much-increase/
- https://verafiles.org/articles/fact-check-marcos-did-not-order-halt-to-trash-collection-in-davao-city
- https://www.snopes.com/fact-check/trump-shooter-republican/
- https://fullfact.org/environment/no-the-government-hasnt-ordered-mandatory-livestock-reductions-this-month-to-cut-emissions/
- https://factcheck.afp.com/doc.afp.com.33DH3YG
- https://politifact.com/factchecks/2020/apr/23/facebook-posts/no-democrats-arent-pushing-microchips-fight-corona/

## Loop verdicts against gold

| status | gold false | gold true |
| --- | --- | --- |
| supported | 0 | 18 |
| refuted | 11 | 0 |
| unsupported | 12 | 8 |
| conflicting | 0 | 1 |

## Operating points, loop against the urn at the same false positive rate

The urn flags when its score sits at or below the threshold, so matching means taking the largest threshold whose FPR stays at or below the loop's, then asking what recall is left. A positive difference means the loop finds more of the false claims than the urn does at the same cost in false alarms.

| point | loop recall | loop FPR | urn 7-flag recall | diff | p | urn 3-voice recall | diff | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A (pass = supported) | 1.000 [1.000, 1.000] | 0.333 [0.167, 0.520] | 0.913 [0.778, 1.000] | 0.087 [0.000, 0.222] | 0.500 | 0.957 [0.852, 1.000] | 0.043 [0.000, 0.148] | 1.000 |
| B (pass = supported, conflicting) | 1.000 [1.000, 1.000] | 0.296 [0.133, 0.467] | 0.913 [0.778, 1.000] | 0.087 [0.000, 0.222] | 0.500 | 0.957 [0.852, 1.000] | 0.043 [0.000, 0.148] | 1.000 |
| R (pass = supported, conflicting, unsupported) | 0.478 [0.280, 0.680] | 0.000 [0.000, 0.000] | 0.304 [0.120, 0.500] | 0.174 [-0.115, 0.455] | 0.388 | 0.304 [0.120, 0.500] | 0.174 [-0.115, 0.455] | 0.388 |

| point | loop accuracy | loop precision | urn 7-flag threshold | urn 7-flag FPR | urn 3-voice threshold | urn 3-voice FPR |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.820 [0.700, 0.920] | 0.719 [0.559, 0.868] | 1.321 | 0.333 | 0.779 | 0.296 |
| B | 0.840 [0.740, 0.920] | 0.742 [0.581, 0.885] | 0.787 | 0.296 | 0.779 | 0.296 |
| R | 0.760 [0.640, 0.860] | 1.000 [1.000, 1.000] | -6.544 | 0.000 | -4.760 | 0.000 |

## The urn on its own terms

ROC AUC for 7-flag is 0.867 [0.752, 0.955] on n=50. At its published fitted threshold -3.9536 it flags with recall 0.391 at FPR 0.037 (precision 0.900, accuracy 0.700).
ROC AUC for 3-voice is 0.891 [0.789, 0.968] on n=50. At its published fitted threshold -4.6256 it flags with recall 0.304 at FPR 0.000 (precision 1.000, accuracy 0.680).

Fitted thresholds come from src/eval/data/urn_runs/e1_ctx/model_ladder/ladder.json. The full ROC curves, one point per distinct score, are in comparison.json under urn.models.<model>.roc.

## Loop anatomy

Of the 20 unsupported verdicts, 15 carry a close-below-bar guard event (the bar refused a close the model wanted), 1 reached the end with no evidence at all (a retrieval miss), and 4 are neither.

The fc-undated drop fired 0 times across 0 claims.

Per claim the loop makes 10.460 LLM calls, 1.660 Serper searches and 0.500 Exa searches, and costs $0.00137. 0.500 of claims stop at exa-final.
Of 83 instrumented serper searches, 0.193 returned at least one hit published after the claim's date ceiling (0.107 of all hits).
The 25 exa searches are not date instrumented.

## Sign flips

0 true claims were refuted and 0 false claims were supported.
