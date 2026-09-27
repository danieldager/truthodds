# fc-gold: the verify loop against the log-odds urn

The loop run is `eval/data/claimverify_runs/fc_gold/top3`; the urn scores are `eval/data/urn_runs/e1_ctx/model_ladder/oof_scores.parquet`. The population is the urn's 3274 out-of-fold rows. The loop has 3274 usable records (0 failed), 0 population rows are missing from it, and everything below is computed on the 3274 claims both systems cover: 1772 flag-worthy (gold_true False, mixed counted as false, including 118 veracity-3 unprovable rows) and 1502 pass-worthy. Intervals are 95% percentile bootstraps over claims, 1000 resamples, seed 0, one shared index matrix so every difference is paired.

## Loop verdicts against gold

| status | gold false | gold true |
| --- | --- | --- |
| supported | 50 | 936 |
| refuted | 700 | 20 |
| unsupported | 973 | 520 |
| conflicting | 49 | 26 |

## Operating points, loop against the urn at the same false positive rate

The urn flags when its score sits at or below the threshold, so matching means taking the largest threshold whose FPR stays at or below the loop's, then asking what recall is left. A positive difference means the loop finds more of the false claims than the urn does at the same cost in false alarms.

| point | loop recall | loop FPR | urn 7-flag recall | diff | p | urn 3-voice recall | diff | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A (pass = supported) | 0.972 [0.964, 0.979] | 0.377 [0.351, 0.401] | 0.911 [0.897, 0.924] | 0.060 [0.048, 0.075] | 0.000 | 0.907 [0.893, 0.920] | 0.064 [0.051, 0.079] | 0.000 |
| B (pass = supported, conflicting) | 0.944 [0.934, 0.954] | 0.360 [0.336, 0.383] | 0.894 [0.879, 0.909] | 0.050 [0.035, 0.066] | 0.000 | 0.887 [0.872, 0.900] | 0.057 [0.042, 0.073] | 0.000 |
| R (pass = supported, conflicting, unsupported) | 0.395 [0.373, 0.419] | 0.013 [0.008, 0.020] | 0.353 [0.330, 0.377] | 0.042 [0.016, 0.068] | 0.002 | 0.286 [0.264, 0.307] | 0.109 [0.085, 0.136] | 0.000 |

| point | loop accuracy | loop precision | urn 7-flag threshold | urn 7-flag FPR | urn 3-voice threshold | urn 3-voice FPR |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.812 [0.798, 0.825] | 0.753 [0.734, 0.771] | 0.484 | 0.377 | -0.143 | 0.376 |
| B | 0.805 [0.791, 0.818] | 0.756 [0.738, 0.774] | -0.025 | 0.359 | -0.590 | 0.359 |
| R | 0.666 [0.651, 0.682] | 0.972 [0.959, 0.984] | -4.507 | 0.013 | -5.122 | 0.013 |

## The urn on its own terms

ROC AUC for 7-flag is 0.862 [0.850, 0.875] on n=3274. At its published fitted threshold -3.9536 it flags with recall 0.422 at FPR 0.020 (precision 0.961, accuracy 0.678).
ROC AUC for 3-voice is 0.852 [0.839, 0.865] on n=3274. At its published fitted threshold -4.6256 it flags with recall 0.340 at FPR 0.017 (precision 0.960, accuracy 0.635).

Fitted thresholds come from src/eval/data/urn_runs/e1_ctx/model_ladder/ladder.json. The full ROC curves, one point per distinct score, are in comparison.json under urn.models.<model>.roc.

## Loop anatomy

Of the 1493 unsupported verdicts, 680 carry a close-below-bar guard event (the bar refused a close the model wanted), 271 reached the end with no evidence at all (a retrieval miss), and 542 are neither.

The fc-undated drop fired 58 times across 55 claims.

Per claim the loop makes 10.994 LLM calls, 1.693 Serper searches and 0.606 Exa searches, and costs $0.00136. 0.606 of claims stop at exa-final.
Of 5542 instrumented serper searches, 0.201 returned at least one hit published after the claim's date ceiling (0.096 of all hits).
The 1985 exa searches are not date instrumented.

## Sign flips

20 true claims were refuted and 50 false claims were supported.

gold true, loop refuted:

- `fb51fafe5925f622` https://www.politifact.com/factchecks/2020/apr/22/joe-biden/biden-says-trump-agency-isnt-doing-enough-protect-/
- `847766b93d9f0a6b` https://www.snopes.com/fact-check/trump-golden-statue-miami/
- `d85860c2d726d76a` https://www.snopes.com/fact-check/trump-quiet-acts-kindness/
- `8539180b2f98e257` https://www.politifact.com/factchecks/2022/nov/05/kathy-hochul/state-pays-newest-overtime-costs-now/
- `395884653fe7c568` https://africacheck.org/fact-checks/reports/verifying-claims-bola-tinubus-media-team-amid-endbadgoverance-protests-nigeria
- `e714be2de347f31d` https://www.snopes.com/fact-check/matt-gaetz-anti-human-trafficking/
- `94350b7e5f6e6912` https://africacheck.org/fact-checks/reports/no-game-fact-checking-songezo-zibis-rise-mzansi-election-manifesto-launch
- `ff17cff489bdc10b` https://www.snopes.com/fact-check/tim-walz-younger-than-brad-pitt/
- `cfe7d7bc03217244` https://www.politifact.com/factchecks/2020/sep/08/ron-johnson/johnson-mostly-track-claim-flu-harder-kids-covid-1/
- `45ecea0ae013cbbe` https://www.snopes.com/fact-check/bannon-charged-contempt-congress/
- `19f78b5c5950f697` https://politifact.com/factchecks/2020/feb/03/val-demings/ukraine-really-still-waiting-us-aid-small-amount-s/
- `74690d3c38db365f` https://www.aap.com.au/factcheck/labor-claim-lays-bare-government-failure-on-tree-target/
- `fa4d084a9c2c38aa` https://www.politifact.com/factchecks/2023/jun/12/chris-christie/trump-did-not-sign-any-major-immigration-laws-but/
- `1b73dcb86f8e440d` https://www.snopes.com/fact-check/elon-musk-c-sections/
- `60108b5185723581` https://www.politifact.com/factchecks/2021/aug/13/elise-stefanik/refereeing-andrew-cuomo-elise-stefanik-firearm-ind/
- `ad8493cbec6da344` https://africacheck.org/fact-checks/reports/kenyas-finance-minister-defends-infrastructure-fund-includes-misleading-claims
- `effd15d2201a2ac5` https://www.politifact.com/factchecks/2023/dec/15/asa-hutchinson/asa-hutchinsons-mostly-true-claim-that-china-is-al/
- `c22cae03af2efda8` https://www.politifact.com/factchecks/2021/jan/08/mitch-mcconnell/mitch-mcconnell-says-accurately-joe-bidens-win-was/
- `ae3763cd62abdf2c` https://www.snopes.com/fact-check/trump-revokes-1965-dei-executive-order/
- `e8deb5c73ae71e49` https://www.snopes.com/fact-check/biden-fall-poland/

gold false, loop supported:

- `99c502daf5fe6a09` https://www.boomlive.in/world/dr-roger-hodkinson-makes-false-claims-to-state-covid-19-is-a-hoax-10962
- `b83fb53a4ff15699` https://verafiles.org/articles/vera-files-fact-check-roque-contradicts-ano-covid-house-hous
- `bb79a401e75c09ef` https://www.snopes.com/fact-check/harris-mcdonalds-job-college/
- `462636b2761b6995` https://www.snopes.com/fact-check/george-santos-white-power-sign-mccarthy/
- `26592757a89b8586` https://www.factcheck.org/2022/04/scicheck-covid-19-data-comparing-vaccinated-vs-unvaccinated-continues-to-be-available-contrary-to-viral-posts/
- `a99f250bcc2f74da` https://www.boomlive.in/fake-news/fake-message-claims-thieves-are-posing-as-mha-officials-on-census-duty-9613
- `f05c20ee25351429` https://www.politifact.com/factchecks/2021/feb/19/viral-image/texas-energy-company-accidentally-billed-customers/
- `b967f24c25d11c9c` https://www.politifact.com/factchecks/2022/may/26/ted-cruz/research-armed-campus-police-do-not-prevent-school/
- `50208408c08b28ea` https://www.politifact.com/factchecks/2020/oct/13/mike-pence/pence-said-biden-copied-trumps-pandemic-response-p/
- `c7c634867cbf273a` https://www.snopes.com/fact-check/drone-cheating-wife-cvs/
- `a6b4fe8068c26492` https://www.politifact.com/factchecks/2022/feb/23/facebook-posts/survey-results-about-trudeau-trucker-protest-misch/
- `b32b345fec129139` https://www.boomlive.in/world/no-pfizer-biontech-covid-19-vaccine-does-not-contain-nanotechnology-11131
- `c07cbbb97b196f52` https://www.politifact.com/factchecks/2021/nov/16/rick-scott/critical-race-theory-isnt-virginias-curriculum/
- `f4ffc7a4c3da991e` https://www.snopes.com/fact-check/trump-hitler-good-things/
- `c71865a1c4d6da62` https://factuel.afp.com/doc.afp.com.9MH3V7
- `e30c7f89f645934e` https://factcheck.afp.com/doc.afp.com.9QK8F2
- `8b07e4efa3970f61` https://www.snopes.com/fact-check/cuomo-vaccine-bad-news-trump/
- `239445da448c412a` https://factcheck.afp.com/doc.afp.com.33E82Q2
- `e2ee799147e66dd1` https://www.politifact.com/factchecks/2021/may/07/kamala-harris/kamala-harris-said-corruption-costs-much-5-worlds-/
- `a4296ff80f17df04` https://factcheck.afp.com/doc.afp.com.342C6HC
- `752b6e83b3d3b687` https://www.snopes.com/fact-check/bernie-sanders-honeymoon-russia/
- `48a8ecfe1f0774fd` https://verafiles.org/articles/vera-files-fact-check-false-list-of-lto-motorcycle-fines-circulates
- `127515b240777d79` https://factcheck.afp.com/doc.afp.com.9X38YL
- `7cacbe252a854793` https://www.snopes.com/fact-check/melania-trump-obama-white-house-toilet/
- `bd6eb67d2ef2b02c` https://www.snopes.com/fact-check/biden-fart-duchess-cornwall/
- `0af49128873e1b32` https://politifact.com/factchecks/2026/may/07/tweets/california-gasoline-supply-six-weeks/
- `f7a01ccc210a7a5c` https://www.politifact.com/factchecks/2025/jun/10/donald-trump/big-beautiful-bill-tax-increase-68-percent/
- `881b67d5ba084181` https://www.politifact.com/factchecks/2023/jul/26/tim-scott/tim-scott-is-wrong-about-more-illegal-immigration/
- `e95e147272ba8d1d` https://politifact.com/factchecks/2022/apr/08/jd-vance/jd-vances-ad-about-open-border-and-immigrant-voter/
- `fb1b220f51b974ad` https://www.aap.com.au/factcheck/tree-coverage-claim-leaves-out-the-facts/
- `d38434dac69503e4` https://www.politifact.com/factchecks/2023/aug/09/donald-trump/trump-says-doj-is-trying-to-criminalize-asking-que/
- `71adef7fdf4f087d` https://www.factcheck.org/2020/07/trumps-false-military-equipment-claim/
- `c829495ec7e814eb` https://www.factcheck.org/2021/03/factchecking-trumps-cpac-speech-2/
- `4d69877553f36f7f` https://factcheck.afp.com/beef-products-targeted-canada-food-safety-hoax
- `5467c822aebc6d9b` https://www.politifact.com/factchecks/2020/nov/20/jon-ossoff/david-perdue-opposes-biden-didnt-say-hed-do-everyt/
- `3e9afeb4a9f82c74` https://www.factcheck.org/2021/08/factchecking-bidens-statements-about-afghanistan/
- `d124bd4d50145ce6` https://www.snopes.com/fact-check/ted-nugent-underage-girl/
- `eb0e604cbf5f01ff` https://factcheck.afp.com/doc.afp.com.78K94KE
- `607f8bbbbe151a25` https://www.politifact.com/factchecks/2022/mar/29/facebook-posts/space-foundation-didnt-revoke-honors-russian-cosmo/
- `2abb39b05b7a61e2` https://politifact.com/factchecks/2023/jun/22/donald-trump/fact-check-trumps-bogus-claim-on-fox-news-that-bal/
- `77a133663504eeb3` https://factcheck.afp.com/doc.afp.com.39FB462
- `e2281ccc35930243` https://factcheck.afp.com/doc.afp.com.34T243N
- `8efd4c92fae6f69e` https://www.factcheck.org/2023/06/factchecking-bidens-campaign-style-speeches/
- `56402e62bc502d14` https://www.factcheck.org/2020/05/trump-misleads-on-hydroxychloroquine-again/
- `b0a58d571e139133` https://www.factcheck.org/2026/01/trumps-claims-about-greenland/
- `d6c75419e6bcef97` https://www.politifact.com/factchecks/2020/sep/22/william-barr/covid-19-rules-second-only-slavery-civil-liberties/
- `f81647500a85c81b` https://factcheck.afp.com/doc.afp.com.46AY39D
- `9f1733d41114617d` https://factcheck.afp.com/pelosi-pence-did-not-fake-covid-19-vaccinations-using-capped-needles
- `606eadb9a2d395e3` https://factuel.afp.com/doc.afp.com.32ET3UL
- `29bd8a65510844ba` https://www.snopes.com/fact-check/twitter-star-of-david/
