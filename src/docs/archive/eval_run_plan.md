# End-to-end verification eval — run plan (dev200)

Plan for running the first **end-to-end DeepSeek V4 Flash** verification eval on the `dev200`
content slice, once the dataset is finalized (other session). Companion docs:
`verdict_confidence_design.md` (binary + confidence reframe), `confidence_metrics.md` (calibration
metric suite — referenced, not duplicated here), `likert_dimensions.md` (the 5 dims).

Status when this was written: dim-grouping + reasoning-at-*scoring* already settled on a proxy
substrate → **scorer = `all5`, scoring reasoning OFF** (clog/230626 15:00). This run tests what
that proxy could not: real end-to-end accuracy, reasoning at *synthesis*, and the raw-claim context.

---

## 0. The decision this run informs

> Is the Tier-3 verifier (DeepSeek V4 Flash) accurate and well-calibrated enough, on real recent
> fact-checks, to gate a nudge — and does reasoning-at-synthesis or raw-claim context move that?

Concretely we want, at the end: a **binary pass/flag** quality number with CIs, a **risk–coverage**
curve (the deployment knob), a **reasoning on/off** verdict (worth the 6× cost?), and a **raw-context**
verdict (does it help, and does it finally make `contextual_integrity` a distinct signal?).

---

## 1. Dataset preconditions ("ready" checklist)

Run only once `dev200.parquet` has:
- [ ] `judged_axis == content` for all rows (no nulls); satire kept + flagged (`is_satire`).
- [ ] Harmonization audit fixes applied to content rows (the ~20 vetted disagreements); binary +
      4-class labels final.
- [ ] `raw_claim` + `claim_context` columns populated where recoverable, with a `has_raw` flag and a
      `raw_source` (publisher/url). **`claim_context` must be the raw circulated framing, NOT the
      fact-checker's analysis/verdict** (leakage — see §7).
- [ ] Leakage fields present: `review_url` (exclude) and the best available date (`claim_date` →
      else `review_date`) for the ceiling.

---

## 2. System under test (fixed config)

`pipeline/verify.py`, DeepSeek V4 Flash via DeepInfra. Cascade Serper→Exa, credibility-rerank,
snippet-always evidence. Scorer = **5-dim `all5` Likert** (veracity, evidence_sufficiency,
evidence_agreement, source_reliability, contextual_integrity) + 4-class evaluator, **scoring
reasoning OFF**, temperature 0. `§3b` post-flag step ON (gets the raw post/context).

---

## 3. Experiments (what we test, why, how)

Run **2 full pipeline passes + 1 cheap re-score** (not a 2×2 — avoids a wasted cell):

| id | arm | synth reasoning | raw context (→ CI + flag) | how |
|----|-----|----|----|-----|
| **A** | baseline | OFF | ON (where `has_raw`) | full pipeline |
| **B** | reasoning | **MEDIUM** | ON | full pipeline |
| **C** | no-context ablation | OFF | **OFF** | re-score CI+flag off A's analyses (no retrieval) |

- **E1 — End-to-end baseline (A).** *Why:* the headline we never had — every prior number was
  gpt-oss or a proxy substrate. Establishes real binary + 4-class quality and the confidence
  calibration on recent content claims.
- **E2 — Reasoning at synthesis, B vs A.** *Why:* the A/B you committed to; scoring-level reasoning
  was already shown inert, synthesis is where thinking could actually change the verdict. *Caveat:*
  synthesis reasoning also drives `next_query`, so B may **retrieve different evidence** than A — this
  is an *end-to-end* reasoning comparison, not "same evidence, better synthesis." Named, not hidden.
- **E3 — Raw context, A vs C.** *Why:* tests whether feeding the raw claim + claimant to CI/flag (a)
  changes flag decisions/calibration and (b) finally makes `contextual_integrity` a **distinct**
  signal. The prior CI↔veracity redundancy (ρ=0.82) was measured with no claim context; C vs A is
  the clean within-claim on/off test of whether context decorrelates them. Core veracity/4-class are
  identical in A and C (context only touches CI/flag), so C is cheap.
- **Observational — has_raw vs no_raw** (slice on A): does the system do better where raw exists?
  **Confounded with publisher** (has_raw≈snopes/fullfact, no_raw≈AFP) — report it, but treat C-vs-A
  as the *causal* context test; the slice is descriptive only.

---

## 4. Metrics (what each is, why it matters)

Primary task = **binary pass/flag** (the nudge decision). Decision mapping: report metrics
**threshold-swept** on the continuous veracity score rather than committing to one cutoff.

### 4.1 Verdict quality
| metric | definition | why |
|---|---|---|
| **macro-F1, flag-F1, pass-F1** | per-class + macro harmonic mean of P/R | imbalance-robust headline; flag is the action class |
| **PR-AUC (flag)** | area under precision–recall for flag | the right curve under 88% prevalence; insensitive to TN flood |
| **ROC-AUC** | rank-discrimination of veracity vs gold | prevalence-invariant; comparable across slices/arms |
| confusion matrix + per-class P/R | — | where it fails (esp. pass↔CE boundary) |
| **4-class** acc + macro-F1 + per-class recall | vs `harmonised_label` | secondary; continuity with ClaimCheck/AVeriTeC; watch NEI/CE recall |

All prevalence-sensitive metrics (precision, PR-AUC, F1) reported **importance-weighted back to
population prevalence** (dev200 oversampled pass 12%→35%); ROC-AUC needs no weighting.

### 4.2 Confidence & calibration (the deployment layer — see `confidence_metrics.md`)
| metric | why |
|---|---|
| **Risk–coverage curve** | THE deployment metric: precision/accuracy achievable vs fraction of claims we act on as the confidence threshold rises. Sets the nudge threshold. |
| **ECE + reliability diagram** | does stated confidence match empirical accuracy? the threshold is only meaningful if calibrated |
| **AUROC of each confidence signal → correctness** | which of {|veracity−3|, evidence_sufficiency, contextual_integrity} predicts being right (prior: abs-veracity 0.76, source_reliability anti-predictive) |
| **confident-pass vs confident-flag reliability** | the asymmetry (prior: confident-FALSE ~100% vs confident-TRUE ~60%); confident-TRUE is the high-risk bucket for a nudge tool |

### 4.3 Dimension analysis (the 5-dim Likert in production)
- Per-dim **scale usage** (%@2|4, ceiling-pinning) — does it collapse like the proxy run?
- **Inter-dim Spearman**, headline = **CI↔veracity with context (A) vs without (C)** — the decisive CI test.
- Each dim's marginal **AUROC for correctness** — which dims earn their place (drop the deadweight).

### 4.4 Cost / latency / retrieval diagnostics
- LLM calls/claim, completion tokens (incl `reasoning_content`), **$/claim**, latency p50/p95 — per arm (the reasoning tax: ~6× expected).
- **0-evidence rate**, rounds_used + `stopped_reason` distribution, cap-hit rate — pipeline health on recent claims.
- **Leakage rate** (§7): fraction of claims where retrieved evidence includes a fact-check domain.

---

## 5. Slices
reasoning {A,B} · context {A,C} · `has_raw` {y,n} · `is_satire` {incl,excl} · publisher · binary label ·
4-class (esp. **CE/Half-True** boundary, reported separately — known gold-noise zone).

---

## 6. Statistics
- **Bootstrap 95% CIs** on every headline metric (Wilson for proportions) — NOT Wald/normal-approx
  (`classifier_dataset_methods.md`).
- **Paired tests across arms** (same claims): **McNemar** on binary verdict flips A↔B; bootstrap CI on
  metric deltas.
- **Power caveat (set expectations up front):** n=200, 70 pass, K=1. CIs will be wide; a reasoning
  effect smaller than ~5–8 F1 points likely won't clear significance. The scoring-level reasoning A/B
  at n=60 detected nothing — treat a null E2 as "no large effect," not "no effect."

---

## 7. Leakage controls (highest-validity risk — see §9)
These are RECENT (2026) fact-checks; the top web results for a claim are often the fact-checks
themselves (verdict included) → catastrophic contamination. Controls:
1. `exclude_urls = [review_url]` per claim (already supported).
2. **date_ceiling** = `claim_date` (only 82/200) else `review_date` (weaker — review postdates the
   claim, so it still admits the fact-check; treat as partial).
3. **Fact-check-domain blocklist** at retrieval (snopes/afp/politifact/fullfact/… ) — recommended,
   since (2) is weak. Tradeoff: also removes legit fact-checker-cited primary sources; measure both.
4. **Measure leakage explicitly**: flag any claim whose evidence domains intersect a fact-check set;
   report metrics including AND excluding leaked claims. A big leak-vs-clean gap = the number is inflated.

---

## 8. Run order & cost
1. Smoke (n=5, arm A) — validate leakage controls actually fire (inspect evidence domains) + parse.
2. **Arm A** full 200 (reasoning off) — cheapest; the baseline.
3. **Arm C** = re-score CI+flag off A (no retrieval; cheap).
4. **Arm B** full 200 (reasoning medium) — the expensive arm (~6× tokens + retrieval may diverge).
Budget gate: project $ after the A smoke; if B is too costly, run B on a **stratified n=100 subset**
and report E2 at reduced power. Retrieval credits (Serper/Exa) are the other cost axis — track separately.
Snapshot every per-claim trace (evidence + analysis) so all §4 analysis re-runs offline at zero cost
(and to mitigate live-retrieval non-reproducibility).

---

## 9. Gaps & risks in our current thinking (ordered by severity)

1. **Retrieval leakage (top threat).** Recent claims → the fact-check is often the top hit. If
   uncontrolled, we measure "can it read the answer," not "can it verify." §7 mitigates but the date
   ceiling is weak (118/200 lack `claim_date`). *Action:* blocklist fact-check domains + measure
   leak rate + report clean-subset metrics. If leak rate is high even after controls, the headline is
   suspect.
2. **has_raw confounded with publisher.** "Raw helps" is inseparable from "snopes is easier than AFP"
   in the observational slice. *Action:* rely on the within-claim A-vs-C ablation for causal claims;
   present the slice as descriptive only.
3. **contextual_integrity may still be ill-posed.** Even with the raw claim, CI is judged from an
   analysis built to verify the *proposition*, not assess *framing* — it may stay collapsed onto
   veracity. *Action:* C-vs-A correlation is the test; if CI doesn't decorrelate, either give it its
   own framing-focused pass or cut it (don't ship a redundant dim).
4. **Reasoning A/B is end-to-end, not isolated.** Synthesis reasoning changes `next_query` → different
   evidence in B. We can't attribute a B−A gap purely to "better reasoning." *Action:* name it; if
   E2 is interesting, a follow-up freezes A's evidence and toggles only synthesis reasoning.
5. **Statistical power.** n=200/K=1 → wide CIs; small effects invisible. *Action:* CIs on everything,
   no over-claiming; pre-state the minimum detectable effect.
6. **Binary gold noise at the CE/Half-True boundary.** 4-class→binary collapse forces CE→flag, but CE
   is the genuinely ambiguous zone (prior: pass-side "errors" mostly weren't). *Action:* report a
   clean subset excluding CE; analyze CE separately.
7. **Recency → sparse independent evidence.** Very recent claims may have little non-fact-check
   evidence → inflated 0-evidence/NEI. *Action:* report 0-evidence rate; correlate errors with
   claim recency.
8. **No human audit of system analyses.** Aggregates can hide right-for-wrong-reasons and surviving
   gold errors. *Action:* read ~15 analyses/arm spanning correct/incorrect + the CE boundary.
9. **Single-model, no panel.** This run measures verbalized/absolute-confidence calibration only; the
   prior study found **inter-model agreement** was the usable confidence signal. *Action:* scope a
   panel run as the natural follow-up, not this run.
10. **Decision rule + success criterion undefined.** "Good enough to nudge" needs a number (e.g.
    flag-precision ≥ X at coverage ≥ Y). *Action:* agree the target with the team before reading
    results, so the run is a test, not a fishing trip.
11. **Live-retrieval non-reproducibility.** Re-running later hits a changed web. *Action:* snapshot
    traces (§8); all analysis reads from the snapshot.

---

## 10. Deliverables
- `dim_grouping`-style harness extended for end-to-end runs (or reuse `verify_run.py` with the
  dev200 input, leakage flags, and the §3b context wiring) → per-claim parquet + saved traces.
- A scorer producing §4 tables (binary, 4-class, calibration, dimensions, cost) × §5 slices, with §6
  CIs, for arms A/B/C.
- A short results write-up: the four verdicts (baseline quality, reasoning worth-it?, context
  worth-it?, CI distinct?) + the leakage-rate caveat + the qualitative read.
