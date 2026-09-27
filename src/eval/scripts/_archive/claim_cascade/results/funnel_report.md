# Claim cascade — funnel & rubric comparison

## 1. Funnel

| Stage | Unit | Count | % of unit | % of prev |
|---|---|---:|---:|---:|
| Captured posts | posts | 862 | 100.0% | 100.0% |
| In scope (LLM, Stage 2) | posts | 208 | 24.1% | 24.1% |
| Has ≥1 claim (Stage 3) | posts | 163 | 18.9% | 78.4% |
| Extracted claims | claims | 410 | 100.0% | 251.5% |
| Check-worthy (4-prong OR FABLE) | claims | 391 | 95.4% | 95.4% |

*Parallel measurement (not a gate):* in_scope_embed = **378** (43.9% of posts) — Stage 2 runs the LLM on all 862 posts, so embedding scope is logged, not gated on.

**Tier-2 short-circuit (FCT):** of the 391 check-worthy claims, **14** (3.6%) already have a published fact-check; the other **377** need novel verification. FCT runs on the whole check-worthy union — this is a short-circuit *rate*, not a narrowing gate.

**Monotonic gate chain:** PASS (posts [862, 208, 163], claims [410, 391])

## 2. Embedding ↔ LLM scope agreement (n = posts)

| | LLM in-scope | LLM out |
|---|---:|---:|
| **Embed in-scope** | 172 | 206 |
| **Embed out** | 36 | 448 |

Agreement: **71.9%** (620/862). Embed-only (embed yes / LLM no): 206; LLM-only (LLM yes / embed no): 36.

## 3. Check-worthiness rubrics — 4-prong vs FABLE (n = claims)

| | FABLE checkworthy | FABLE not |
|---|---:|---:|
| **4-prong candidate** | 56 | 332 |
| **4-prong not** | 3 | 19 |

Agreement: **18.3%** (75/410). Both: 56; 4-prong only: 332; FABLE only: 3; neither: 19.

**4-prong says check-worthy but FABLE-low (verifiable-but-low-harm):**

- The left under Mitterrand was proud of having liberalized the radio waves.
- The city installed concrete walls without lighting, creating a trap.
- Elon Musk once offered to eat a Happy Meal on live TV if McDonald’s accepted Dogecoin.
- A political party emailed the author offering 1,000 euros per month to promote liberal ideas on social media.
- Today the left boasts of defending the nationalization of television.

**FABLE says check-worthy but 4-prong says no (high-harm, not 4-prong):**

- The Macrons do not care about women who have been assaulted.
- Invaders smashed the place up in Toulouse.
- Ils tirent sur les pompiers !

## 4. Over-decomposition

Posts with > 8 claims: **5**.
Worst case: `f7c976179b8b7fd8` with **30** claims. This is simply a fact [QUOTED] Foreign born population of each country :   Jan 2001                     Jan 2025  🇦🇹 8.7%                         22.5% 🇧🇪 8.4% 
