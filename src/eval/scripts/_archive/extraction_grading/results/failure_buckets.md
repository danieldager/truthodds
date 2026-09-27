# Failure-mode buckets

A claim is bucketed under a dimension when its Likert on that dimension is ≤ 2. A claim may appear in multiple buckets. An additional misinfo_candidate=false bucket is included.

## fidelity ≤2 (n=4, 2.3% of claims)
Top distinguishing post features (|Cohen's d|):

| feature | bucket mean | corpus mean | d |
|---|---|---|---|
| n_emojis | 0.500 | 0.151 | +0.66 |
| punctuation_density | 0.028 | 0.044 | -0.60 |
| caps_ratio | 0.021 | 0.058 | -0.51 |

Example claims:
- `97800024475f57bb` [#0] (fidelity=2, misinfo=False): 'Masks work.'
- `5b645479cccd6bcb` [#0] (fidelity=1, misinfo=False): 'Western media produced many headlines about a ceasefire in the war but largely ignored and under‑reported the ceasefire.'
- `98b1922faacc09c9` [#0] (fidelity=1, misinfo=True): 'He puts every scrap of greenery into a heavy plastic clamshell.'

## decontextualized ≤2 (n=22, 12.4% of claims)
Top distinguishing post features (|Cohen's d|):

| feature | bucket mean | corpus mean | d |
|---|---|---|---|
| lexical_diversity | 0.950 | 0.908 | +0.49 |
| n_named_entities | 1.579 | 2.755 | -0.47 |
| flesch_kincaid_grade | 7.986 | 9.794 | -0.39 |

Example claims:
- `ff5f2bb5bb188f2e` [#0] (decontextualized=2, misinfo=True): 'The EC ensures a dictatorship.'
- `5b645479cccd6bcb` [#0] (decontextualized=2, misinfo=False): 'Western media produced many headlines about a ceasefire in the war but largely ignored and under‑reported the ceasefire.'
- `98b1922faacc09c9` [#0] (decontextualized=2, misinfo=True): 'He puts every scrap of greenery into a heavy plastic clamshell.'

## verifiability ≤2 (n=10, 5.6% of claims)
Top distinguishing post features (|Cohen's d|):

| feature | bucket mean | corpus mean | d |
|---|---|---|---|
| lexical_diversity | 0.953 | 0.908 | +0.51 |
| n_chars | 144.100 | 185.538 | -0.48 |
| n_named_entities | 1.600 | 2.755 | -0.45 |

Example claims:
- `5b645479cccd6bcb` [#0] (verifiability=2, misinfo=False): 'Western media produced many headlines about a ceasefire in the war but largely ignored and under‑reported the ceasefire.'
- `0b719c1965dc072c` [#1] (verifiability=2, misinfo=False): 'An editor was blocked from receiving the flowers she deserved.'
- `2d0fe880b840890b` [#0] (verifiability=2, misinfo=False): 'Steam has to offer refunds in my country.'

## misinfo_candidate=false (n=24, 13.6% of claims)
Top distinguishing post features (|Cohen's d|):

| feature | bucket mean | corpus mean | d |
|---|---|---|---|
| n_chars | 155.000 | 185.538 | -0.35 |
| exclamation_count | 0.409 | 0.189 | +0.34 |
| flesch_kincaid_grade | 8.233 | 9.794 | -0.33 |

Example claims:
- `97800024475f57bb` [#0] (verifiability=4, misinfo=False): 'Masks work.'
- `a8cc3f32a01674f3` [#0] (verifiability=3, misinfo=False): 'Tomb Raider established many tropes in adventure games that we can take for granted today.'
- `5b645479cccd6bcb` [#0] (verifiability=2, misinfo=False): 'Western media produced many headlines about a ceasefire in the war but largely ignored and under‑reported the ceasefire.'
