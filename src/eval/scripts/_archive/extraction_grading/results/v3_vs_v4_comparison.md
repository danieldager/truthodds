# v3 vs v4 comparison — misinformation-anchored prompts on Bluesky N=100

## What changed

v4 replaced both the extraction prompt and the judge prompt with a misinformation-anchored design grounded in the literature review (Stage 1):

- **Criterion**: "possible-misinformation candidate" = factual assertion + verifiable in principle + public-consequence domain + news-substitutable. Synthesized from Guriev/Henry/Marquis/Zhuravskaya 2023 (primary), Vraga & Bode 2020, Wardle & Derakhshan 2017, Lazer et al. 2018.
- **Judge schema**: dropped `conciseness` (always 5.0), replaced `check_worthiness` with `verifiability` (1-5, content-agnostic), added `misinfo_candidate` (bool) gate.
- **Architecture**: removed separate detection judge. The per-claim judge applies the *same* criterion as the extractor, so disagreement is meaningful.

**Caveat**: criterion AND judge schema both changed, so v3 vs v4 numbers are not directly comparable on detection metrics. The case-by-case analysis is the load-bearing comparison.

## Headline numbers

| Metric | v3 | v4 |
|---|---|---|
| `has_claim=true` posts (of 100) | 27 | **13** |
| Total claims extracted | 43 | 17 |
| FPs (judge says no claim) | 10 | **1** |
| FNs | 0 (measured)* | unmeasurable** |
| Precision | 0.63 | **0.923** |
| Per-claim fidelity mean | 4.79 | 4.94 |
| Per-claim decontextualized mean | 4.42 | 4.59 |
| Per-claim check_worthiness mean | 4.16 | — |
| Per-claim verifiability mean | — | 4.18 |
| Per-claim misinfo_candidate=true | — | 15/17 (88%) |

*v3's 0 FNs measured against a separate detection judge that had architectural issues.
**v4 has no separate signal for FN (we don't judge claims when has_claim=false).

## Case-by-case resolution of the 10 v3 FPs

| # | Post (truncated) | v3 outcome | v4 outcome | Resolution |
|---|---|---|---|---|
| 1 | Podcast plug: "manifestation of Noel Edmonds this week" | FP (extracted) | SKIPPED | ✅ resolved — promotional content not extracted |
| 2 | Rhetorical premise: "horrors of turkey factories... ideas?" | FP (extracted) | SKIPPED | ✅ resolved — rhetorical premise not load-bearing |
| 3 | Event promo: "preview show this Sunday 7pm Eastern" | FP (extracted) | SKIPPED | ✅ resolved — event logistics not in public-consequence domain |
| 4 | Personal hope: "Today my granddaughter... her rights are being taken away" | FP (extracted) | SKIPPED | ✅ resolved — first-person emotional framing too vague (matches Q1 decision) |
| 5 | Cato op-ed: "Agricultural trade benefits US farmers and consumers" | FP (extracted) | EXTRACTED + misinfo_candidate=true | ✅ correctly handled — quoted-opinion-as-content amplification (matches Q2 decision) |
| 6 | Cato op-ed: "A new Congress and administration set to take power in 2025" | FP (extracted) | SKIPPED | ✅ resolved — trivially true scheduled event |
| 7 | "My luck it'll only be Prime in the US lol" | FP (extracted) | SKIPPED | ✅ resolved — personal speculation framing |
| 8 | "positive externalities of going to a workplace vanishing" | FP (extracted) | SKIPPED | ✅ resolved — abstract sociological musing |
| 9 | "The mosquitoes are WILD this summer!" | FP (extracted) | SKIPPED | ✅ resolved — hyperbolic personal observation |
| 10 | "Workplace and leadership are changing" | FP (extracted) | SKIPPED | ✅ resolved — vague generalization |

**10/10 v3 FPs correctly resolved by v4**, including the two design-decision cases (Q1 = skip "rights being taken away"; Q2 = extract Cato op-ed claim).

## The 1 new v4 FP

`61904d24` — local birdwatching observation:

> "Very early morning on a sunny day gives you a good chance at seeing one in the open too. They certainly aren't doing great in the county though. Slowly going the same way as Marsh and Willow Tit it seems."

The v4 extractor pulled two claims:
- "They aren't doing great in the county." (verifiability=3, decontextualized=2)
- "It is slowly going the same way as Marsh and Willow Tit." (verifiability=2)

Both correctly judged `misinfo_candidate=false`. Failure mode: extractor produced borderline claims that the judge then rejected. The verifiability scores are low (2-3) which signals the issue — these aren't really verifiable assertions, just informal conversational claims. A future v5 could tighten the extractor's "verifiable in principle" criterion (e.g. require named entities or specific predicates).

## Distribution of v4 verifiability and misinfo_candidate

`verifiability` (1-5, per-claim, n=17):
- 5: 11 claims (65%)
- 4: 1 claim (6%)
- 3: 3 claims (18%)
- 2: 2 claims (12%)
- 1: 0 claims

`misinfo_candidate` (per-claim bool, n=17):
- true: 15 (88%)
- false: 2 (12%)

The 2 false-misinfo claims are both from the birdwatching post above. Otherwise, every extracted claim is a misinformation candidate per the same criterion the extractor used — perfect criterion alignment.

## Methodological notes

- The v4 extractor is highly conservative. 13/100 posts (13%) flagged as containing misinformation candidates. v3 was 27/100 (27%). v1 was 54/100 (54%).
- The "lean false if unsure" instruction is doing real work. The cost asymmetry of the nudging use case justifies this.
- Verifiability and misinfo_candidate are now meaningfully distinct — the birdwatching claims have verifiability=2-3 and misinfo_candidate=false, while "EC ensures dictatorship" has verifiability=2 and misinfo_candidate=true (subjective surface but political-institution content). The 1-5 verifiability gradient captures "how falsifiable is this text" independent of "is this in our misinfo scope".

## Risks / open items for follow-up

- **FN unmeasurable**: we don't judge claims when has_claim=false. If we want to measure recall, we need a post-level judge pass on the 87 has_claim=false posts (cheap: ~$0.30). The fact that v4 extracted 13 vs v3's 27 *could* mean we missed real misinformation claims in those 14 posts. Worth a sample check.
- **The 1 FP** suggests the extractor occasionally produces low-verifiability borderline claims. A post-extraction filter ("drop claims with verifiability < 4") could be considered but would be applied *after* judging, defeating the purpose of the extractor's own filter. Better to tighten the extractor's "verifiable in principle" anchor in the next iteration.
- **Sample size**: N=100 is small. The 1 FP could be noise. A confirmation run at N=500-1000 would tighten the precision CI.
