# Filter sweep — post-mortem (agreement only, no gold)
posts: 1000 | claims: 2166 (by extractor: {'gpt-oss-120b': 1182, 'qwen3-32b': 984})
## 1. Cross-model agreement per filter × mode
### RAW (post-level, both models score the post)
- `verif_raw_g`(56%) vs `verif_raw_q`(51%): n=999, agree=84.1%, kappa=0.681 (substantial)
- `relev_raw_g`(34%) vs `relev_raw_q`(32%): n=1000, agree=91.5%, kappa=0.808 (almost-perfect)
- `harm_raw_g`(28%) vs `harm_raw_q`(23%): n=1000, agree=86.1%, kappa=0.636 (substantial)

### CLAIM / BOTH (claim-level, both scorers score the same claim)
- `relev_claim_g`(53%) vs `relev_claim_q`(51%): n=2166, agree=90.3%, kappa=0.806 (almost-perfect)
- `relev_both_g`(59%) vs `relev_both_q`(58%): n=2165, agree=91.6%, kappa=0.828 (almost-perfect)
- `harm_claim_g`(17%) vs `harm_claim_q`(14%): n=2166, agree=91.8%, kappa=0.693 (substantial)
- `harm_both_g`(26%) vs `harm_both_q`(23%): n=2166, agree=86.6%, kappa=0.640 (substantial)

## 2. Extraction agreement between models
- n_claims: exact match=61.9%, |diff|<=1=85.5%
- both-extracted=376, gpt-only=102, qwen-only=78, neither=444
- llama claim-set comparison (both-extracted posts): agreement {'high': 231, 'medium': 128, 'low': 17}
  more-complete: {'equal': 172, 'A': 134, 'B': 70} (A=gpt, B=qwen)

## 3. Normalization/decomposition quality (llama judge)
- **qwen3-32b** per-claim means: {'faithful': 4.84, 'decontextualized': 4.66, 'atomicity': 4.96}
- **gpt-oss-120b** per-claim means: {'faithful': 4.88, 'decontextualized': 4.79, 'atomicity': 4.99}
  qwen3-32b: coverage mean=4.47, flag dist={'bad': 11, 'borderline': 138, 'good': 305}
  gpt-oss-120b: coverage mean=4.37, flag dist={'borderline': 148, 'good': 308, 'bad': 22}
- flagged (borderline/bad) claims dumped: 713 -> results/quality_flagged.md

## 4. Raw-vs-claim stripping (same model scores post and its claims)
- **gpt-oss** harm: post-harmful claims=1013, claim-harmful=378, post→claim drop (harmful post, claim not)=651 of 2166
- **qwen** harm: post-harmful claims=794, claim-harmful=305, post→claim drop (harmful post, claim not)=532 of 2166

## 5. Filter-order simulator (RAW post-level, AND of V/R/H)
_Final survivor set is order-independent (AND); the funnel shows where volume dies._

**gpt-oss** (start 1000 posts):
- V→R→H: V=556 R=282 H=188
- V→H→R: V=556 H=235 R=188
- R→V→H: R=338 V=282 H=188
- R→H→V: R=338 H=215 V=188
- H→V→R: H=280 V=235 R=188
- H→R→V: H=280 R=215 V=188

**qwen** (start 1000 posts):
- V→R→H: V=510 R=247 H=133
- V→H→R: V=510 H=162 R=133
- R→V→H: R=321 V=247 H=133
- R→H→V: R=321 H=172 V=133
- H→V→R: H=231 V=162 R=133
- H→R→V: H=231 R=172 V=133

**Cross-model agreement on final survivor set (V∧R∧H, raw):**
- `surv_g`(19%) vs `surv_q`(13%): n=1000, agree=91.5%, kappa=0.686 (substantial)

## 6. Inter-filter overlap (gpt-oss, raw)
- `verif_raw_g`(56%) vs `relev_raw_g`(34%): n=1000, agree=67.0%, kappa=0.363 (fair)
- `verif_raw_g`(56%) vs `harm_raw_g`(28%): n=1000, agree=63.4%, kappa=0.302 (fair)
- `relev_raw_g`(34%) vs `harm_raw_g`(28%): n=1000, agree=81.2%, kappa=0.561 (moderate)
