# Verdict Likert dimensions — current set, proposed extensions, and what last run showed

Companion to `verdict_confidence_design.md` (the binary-verdict + confidence redesign) and
`confidence_metrics.md` (the calibration metric suite). This doc is the single reference for
**the score dimensions themselves**: what we score today (verbatim from the prompt), how to
grow to **3 truth + 3 confidence** dimensions, and what the last run's distributions imply.

**TL;DR**
- Today the verifier scores **1 truth dimension** (`veracity`) + **3 confidence dimensions**
  (`evidence_sufficiency`, `evidence_agreement`, `source_reliability`), each 1–5.
- To reach **3 truth + 3 confidence**, add **2 truth dimensions**: **Contextual Integrity
  (misleadingness)** and **Attribution Fidelity** — both orthogonal to `veracity`, both
  measurable text-only, both mapping to problems we already care about (the §3b post-flag
  step and the attribution-safety rule).
- The 3 confidence dims are not equal: `evidence_sufficiency` is the best evidence-side signal,
  `source_reliability` carries **almost no signal** last run (AUROC 0.56; the literature agrees
  source-credibility is a weak heuristic) → **redefine toward evidence independence/provenance, or
  drop it.** And the model's own verdict-decisiveness is **anti-predictive** — the usable
  confidence signal is **cross-model agreement** (see §5).

---

## 1. The current dimensions (verbatim from `verify_prompts.py::LIKERT_SYSTEM`)

One LLM call scores the synthesized analysis. Prompt header: *"Output one veracity score (the
verdict — which way the evidence points) and three confidence scores (how much to trust that
read). Score each 1-5 independently against its own definition. Use the full scale, including 2
and 4 deliberately."* Parsed by `parse_likert`, clamped 1–5, stored in `models.py::VerdictScores`.

### `veracity` — truth (the verdict direction) — *which way, and how strongly, does the evidence point on the claim's MAIN assertion?*
| | |
|---|---|
| 5 | Clearly true — evidence confirms the main assertion and its load-bearing specifics (numbers, dates, named entities). |
| 4 | Mostly true — the main assertion holds, but a load-bearing specific may be approximate or slightly off. |
| 3 | Mixed or indeterminate — neither clearly confirms nor clearly contradicts the main assertion. |
| 2 | Mostly false — the main assertion is contradicted, though a detail may be unclear. |
| 1 | Clearly false — the evidence contradicts the main assertion. |

### `evidence_sufficiency` — confidence — *how directly and fully does the evidence address the claim?*
| | |
|---|---|
| 5 | Speaks directly to the main assertion AND its load-bearing specifics. |
| 4 | Directly addresses the main assertion, but leaves a specific (a number, date, qualifier) untouched. |
| 3 | Partial — touches the topic but addresses the specific assertion only loosely, or through a single piece of evidence. |
| 2 | Only topically adjacent — does not speak to the specific assertion. |
| 1 | Nothing in the analysis bears on the claim's substance. |

### `evidence_agreement` — confidence — *do the pieces of evidence point the same way?*
| | |
|---|---|
| 5 | All pieces point the same way; no contradictions. |
| 4 | Broad convergence; only minor differences of scope or wording. |
| 3 | Some tension between pieces — or only one piece exists, so agreement cannot be judged. |
| 2 | Notable conflict — at least one piece points each way. |
| 1 | The evidence flatly contradicts itself on a load-bearing point. |

### `source_reliability` — confidence — *how trustworthy are the cited sources, regardless of what they say?*
| | |
|---|---|
| 5 | Primary or authoritative — official records, peer-reviewed work, named domain experts, court/legislative documents. |
| 4 | Established mainstream outlets or institutional reporting. |
| 3 | Mixed — some reputable, but leaning on secondary or lower-tier reporting. |
| 2 | Mostly blogs, opinion, or low-traffic sites. |
| 1 | Only fringe, anonymous, or unidentifiable sources — or none usable. |

*(History: the earlier set was `veracity / coverage / consistency / source_quality`; coverage→
evidence_sufficiency, consistency→evidence_agreement, source_quality→source_reliability. The
binary verdict (`pass` iff supported, else `flag`) is derived from `veracity` by a threshold —
see `verdict_confidence_design.md`.)*

Two related prompts score the same analysis: a 4-class evaluator (ClaimCheck verbatim:
Supported / Refuted / Conflicting Evidence-Cherrypicking / Not Enough Evidence) and the §3b
post-level FLAG step (judges the raw post for misleadingness given the atomic verdict).

---

## 2. Why decompose "truth" into more than one axis

`veracity` is a single true↔false spine. Every mature fact-checking scheme refuses to live on
that spine alone — they either fold context into the mid-scale or add **non-degree tags**:

| Scheme | Truth-degree | Context/framing | Attribution | Temporal |
|---|---|---|---|---|
| PolitiFact (6-pt) | ✓ | **✓ baked in** (Half True = "out of context") | — | implicit |
| WaPo Pinocchios | partial | **✓** (2 = "misleading impression… playing with words") | — | repetition axis |
| **Snopes** | ✓ | tag: *Miscaptioned* | **tags: *Misattributed* / *Correct Attribution*** | tag: *Outdated* |
| schema.org ClaimReview | numeric + textual | free-text | — | — |
| FEVER | 3-way (+NEI) | — | — | — |
| **AVeriTeC** | 4-way | **✓ Cherrypicking is a verdict class** | — | — |
| Wardle 7 types | — | Misleading / False-context / False-connection | Imposter / misattribution | False context |

Journalistic scales (PolitiFact/WaPo/Snopes) separate literal truth from context, attribution,
and time; Snopes makes attribution and time **first-class tags orthogonal to its truth scale** —
the cleanest existence proof that these are distinct axes, not sub-cases of degree-of-truth. A
data-driven mapping study finds these scales only align once collapsed to binary real/fake, and
that "Misleading/Unproven/Miscaptioned" categories don't map onto a linear truth line. That is
the empirical case for **adding axes rather than stretching `veracity`.**

---

## 3. Proposed 2 new truth dimensions

### Truth dim #2 — **Contextual Integrity** (misleadingness / framing fidelity)
*Beyond the literal assertion, does the impression the post creates survive the full evidence —
or do omitted context, cherry-picking, or implied causation make a literally-true claim mislead?*

| | |
|---|---|
| 5 | Literal content **and** its evident implication are supported; no material context omitted that would change a reader's takeaway. |
| 3 | Literally accurate but missing context/qualification a reasonable reader needs; a *partially* unsupported impression (≈ PolitiFact **Half True**). |
| 1 | True-ish details arranged to imply a conclusion the evidence contradicts — cherry-picking, implied causation, **stale event shown as current** (≈ paltering / **Cherrypicking**). |

- **Why:** this is the single largest gap between `veracity` and ground truth, and it is exactly
  what the §3b post-flag step needs to judge. Fact-checkers treat it as core (PolitiFact's busiest
  mid-ratings are *defined* by context omission; AVeriTeC promotes Cherrypicking to a verdict
  class). The psychology is named: **paltering** = "deceiving through truthful statements."
- **Distinct from** `veracity` (scores the literal assertion; CI scores whether the *gist* holds
  even when the literal claim is true) and from `evidence_*` (those are about the evidence we
  gathered; CI is about claim-framing-vs-evidence).
- **Measurable text-only:** yes — we already retrieve the surrounding evidence. The added step:
  (i) state the literal proposition, (ii) name the *implied* proposition, (iii) test whether full
  evidence supports the implication / whether a material fact was omitted. **Anchor it to hold
  literal truth fixed** ("literally accurate **but**…") so it can't silently re-score `veracity`.
- **Folds in temporal:** the "stale-as-current" case is encoded as the level-1 anchor for now (see
  §5 on Temporal Validity as a later standalone axis).

### Truth dim #3 — **Attribution Fidelity**
*For "X said/did Y" claims, does the evidence confirm X is correctly the source/agent of Y —
scored independently of whether Y itself is true.*

| | |
|---|---|
| 5 | Evidence confirms the named source/actor said/did exactly what's claimed, in context. |
| 3 | Roughly the right source but the quote is altered, partial, or stripped of qualifying context; or right words, wrong stance. |
| 1 | Misattributed/fabricated quote, imposter source, or words so out of context the meaning inverts (≈ Snopes **Misattributed**; Wardle **imposter**). |
| N/A | Claim makes no attribution → **masked, not scored low** (see risks). |

- **Why:** this directly implements our hard nudge-safety constraint — *do not flag accurate
  reporting of a false claim as if the poster lied.* "X said Y" is two checks (did X say it; is Y
  true). Snopes carries distinct *Misattributed* / *Correct Attribution* tags orthogonal to its
  truth scale; claim-normalization research explicitly keeps original quotes and decontextualizes
  the reference separately from verifying the content.
- **Distinct from** `veracity`: a post can be attribution 5 / veracity 1 (faithfully quotes a
  false statement → should **pass** as accurate reporting). Not a statement about evidence quality.
- **Measurable text-only:** yes for quoted/attributed claims — retrieval can locate the quote and
  check wording + context. **Needs an explicit "applicable?" gate**; most claims are non-attributed
  and must be *masked* (N/A), not penalized.

---

## 4. Re-assessing the 3 confidence dimensions

- **`evidence_sufficiency` — keep.** The most-attested confidence facet in claim verification
  (FEVER's *NEI*, AVeriTeC's *Not Enough Evidence* are dedicated classes), and it helped in our
  data. Our cleanest confidence signal.
- **`evidence_agreement` — keep, watch overlap.** Conflicting evidence is a first-class state.
  Risk: AVeriTeC couples agreement with *cherry-picking*, which is Contextual Integrity above —
  keep them disjoint (**agreement = evidence-vs-evidence; CI = claim-framing-vs-evidence**).
- **`source_reliability` — redefine or replace.** It carried **almost no signal** last run
  (AUROC 0.56), consistent with the literature: source-credibility is a weak heuristic and
  predicting correctness from source features is empirically poor (esp. on the unreliable end). It
  is plausibly **confounded** — hard/contested claims attract lower-grade sources, so low
  reliability co-occurs with hard-but-correct reads. **Options:** (a) redefine toward **evidence
  provenance/independence** (count of *independent* corroborating sources, primary-vs-secondary)
  rather than reputational trust; or (b) drop it. The slot is better spent on **cross-model
  agreement** — the single best confidence signal we measured (§5), i.e. a self-consistency / panel
  facet, the best-attested confidence family.

---

## 5. Distribution & calibration analysis (last run)

**The collaborators' question — did lower confidence track lower accuracy? Short answer: no, not
reliably, and the model's own decisiveness is the *worst* confidence signal.** Verified by
re-running `eval/scripts/verdict_confidence/dimension_distributions.py` (full tables in
`results/dimension_analysis.md`). Data: **`agreement.parquet`** (n=60, all 4 dims across a 4-model
panel; `correct = deployment_binary==gold`, acc 0.80) for the multi-dim read; **`flag_eval.parquet`**
(n=1024, `veracity` + `bare_correct`, acc 0.85) for power. *n=60 → panel numbers are directional;
the n=1024 results carry the powered claims.*

**(a) The scale is barely used — opposite failures on the two halves.** The prompt says "use 2 and 4
deliberately"; it largely doesn't. `veracity` collapses to the **poles** (level-3 = 2.7%, **81.8% of
mass at {1,5}**); the 3 confidence dims pin to the **ceiling** (means 4.09–4.37, ~85% at ≥4). Little
dynamic range — this is the **bigger calibration limiter** than the choice of dims.

**(b) `veracity` is a direction, not a confidence — and "confident" is the high-risk bucket.** Errors
don't sit at the uncertain middle; they cluster at **confident-TRUE**:

| | confident-FALSE (`veracity=1`) | confident-TRUE (`veracity=5`) |
|---|---|---|
| flag_eval (n=1024) | err **2.2%** (n=544) | err **30.6%** (n=294) — *all 90 errors are misinformation that PASSED* |
| panel (n=60) | err 7.1% | err 54.5% — *all errors gold=flag* |

Accuracy by `veracity` level (flag_eval): v1 **0.978**, v2 0.719, v3 0.821, v4 0.652, v5 **0.694**;
raw `veracity` correlates **−0.36** with correctness — a high "true" call is *less* often right. The
deployment headline: **never trust a confident PASS at face value.**

**(c) Which signals actually predict correctness** (panel n=60, AUROC / AURC; base-err 0.200):

| Signal | AUROC | AURC | |
|---|---|---|---|
| cross-model agreement (−veracity std) | **0.658** | **0.113** | best — monotone (acc 0.65→0.85→0.90 by tertile) |
| `evidence_sufficiency` mean | **0.650** | 0.117 | best evidence-side dim |
| panel veracity magnitude | 0.622 | 0.139 | |
| weakest-link `min` of conf dims | 0.616 | 0.128 | |
| `evidence_agreement` mean | 0.598 | 0.152 | |
| `source_reliability` mean | 0.560 | 0.153 | **near-useless** |
| ensemble verdict decisiveness `\|veracity−3\|` | **0.486** | 0.218 | **anti-predictive** (acc *falls* 0.85→0.80→0.75) |

The only useful confidence signals are **cross-model agreement** (ask several models; trust what they
agree on) and **`evidence_sufficiency`**. The verdict's own decisiveness is worse than random
ordering. **This does NOT replicate the earlier "veracity-magnitude AUROC ≈0.76" note** — that was a
single-model/dev-slice read; on the reproducible run data magnitude is a poor signal. Flag for
re-confirmation on a larger panel.

**(d) The dims are NOT redundant.** Confidence-dim Spearman ρ = 0.33–0.59 (all well below 0.9) → the
decomposition is real; keep all three. Each confidence dim is *negatively* correlated with `veracity`
(−0.10 to −0.40) — the model rates evidence "sufficient/agreeing" more when it leans false.

**Implications:** (i) fix the scale collapse first (few-shot anchors / force 2–4) — it caps everything
downstream; (ii) gate deployment confidence on **agreement + evidence_sufficiency**, not the verdict's
decisiveness; (iii) treat any confident PASS as the high-risk bucket — exactly what the new
**Contextual Integrity** + **Attribution Fidelity** truth dims are meant to catch (a confident-true
literal claim that is misleading or misattributed).

---

## 6. Proposed final dimension set (3 truth + 3 confidence)

| Class | Dimension | Status |
|---|---|---|
| Truth | `veracity` | keep (drives the binary verdict) |
| Truth | **Contextual Integrity** | **new** — misleadingness/framing; powers the §3b flag |
| Truth | **Attribution Fidelity** | **new** — masked when no attribution; powers nudge-safety |
| Confidence | `evidence_sufficiency` | keep (strongest confidence facet) |
| Confidence | `evidence_agreement` | keep (keep disjoint from Contextual Integrity) |
| Confidence | `source_reliability` → **provenance/independence** *or* **self-consistency** | redefine or replace (anti-predictive as-is) |

**Gate every new axis on a validation check** before shipping: does it add *marginal* AUROC over
the existing set, and does its learned weight have the expected sign? `source_reliability` is the
cautionary tale — a plausible axis that encoded claim-hardness, not trustworthiness.

---

## 7. Open decisions (for the meeting)
- The two new truth dims — agree on **Contextual Integrity + Attribution Fidelity**, or swap in
  **Temporal Validity** (real but harder to measure text-only; recommended as a phase-2 promotion
  from CI's stale-as-current anchor).
- Does `veracity` stay one 1–5 verdict score, or do the 3 truth dims **compose** into the verdict?
- `source_reliability`: redefine (provenance/independence) vs replace (self-consistency) vs drop.
- Few-shot vs the current zero-shot Likert prompt (the "use 2 and 4" instruction is currently
  zero-shot — the scale-usage numbers in §5 will say whether it's obeyed).

---

## References
- PolitiFact methodology (context baked into mid-ratings) · WaPo Pinocchios · Snopes ratings
  (Misattributed/Miscaptioned/Outdated tags).
- Wardle, *Fake news. It's complicated.* (First Draft 7 types).
- Schlichtkrull et al., **AVeriTeC** (FEVER 2024) — Cherrypicking & NEI as first-class verdicts.
- Rogers et al., **paltering** (deception via truthful statements), PMC7259623.
- **ChronoFact** / Evidence-Based Temporal Fact Verification (arXiv:2407.15291) — temporal axis.
- Source-reliability prediction is weak (arXiv:2410.18803); source-credibility-as-heuristic
  (Liu et al. 2025, *Communication Research*).
- Claim Normalization (EMNLP'23 Findings); Document-level Claim Extraction & Decontextualisation
  (arXiv:2406.03239).

*Full annotated survey (landscape table, rankings, all citations): the research brief feeding
this doc; calibration metric definitions in `confidence_metrics.md`.*
