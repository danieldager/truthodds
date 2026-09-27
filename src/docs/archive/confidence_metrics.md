# Confidence signals & calibration metrics — reference

Companion to `verdict_confidence_design.md`. This doc says, for each number we compute,
**what it is, why we chose it, and how we calculate it.** Two parts:
- **Part A — confidence signals**: computed *per claim* from the K resampled verdicts.
- **Part B — evaluation metrics**: judge *which* confidence signal best predicts whether the
  binary verdict was correct, and set the deployment threshold.

Setup recap: off the fixed synthesis we sample the verdict step **K = 10 times** (temp > 0).
Each sample yields a graded `veracity` (1–5, → binary by `veracity ≥ 4`), three confidence dims
(`evidence_sufficiency`, `evidence_agreement`, `source_reliability`, each 1–5), and a free-text
`justification` (→ semantic entropy across the 10 samples). Gold is binary (pass/flag), ~13–16% pass.

---

## Part A — Confidence signals (per claim, from K samples)

### A1. Predictive entropy — per criterion and joint
**What:** how much the K samples *disagree* on a score. Low entropy = samples agree =
confident; high = spread = uncertain. More informative than a raw vote fraction because it
uses the whole distribution, and it's the right tool for the graded (1–5) scores.

**Why:** confirmed by the deep-research as the better-calibrated self-consistency aggregation
(vs. vote fraction). For a *binary* it ≈ vote fraction, but for the 1–5 `veracity` and the 1–5
confidence dims it distinguishes "tightly clustered at 4" from "smeared 2–5."

**How (per criterion c):** across the K samples, form the empirical distribution over the 5
levels, `p_c(v) = count(score=v)/K`. Then
`H_c = -Σ_v p_c(v)·ln p_c(v)`, normalized to [0,1] by `ln 5`. Report **confidence_c = 1 −
H_c/ln5** (1 = unanimous). Compute for `veracity` (decision-side) and each confidence dim.

**How (joint, "all criteria together"):** use the **mean of the per-criterion entropies** —
stable at small K. (The alternative — entropy over the full distinct-vector distribution —
saturates when K is small and most vectors are unique, so avoid it unless K is large.)

### A2. Verdict agreement (decision-side)
**What/why:** does the model land on the same pass/flag each time? The simplest decision-side
confidence. **How:** `p = fraction of K samples voting the majority verdict`; report `p` (or
its binary entropy `1 − H([p,1−p])/ln2`). For the graded `veracity`, A1's veracity-entropy is
the finer version.

### A3. Likert mean & spread (evidence-side)
**What:** the **mean** confidence dims = the "decomposed" confidence (how strong the model
thinks the evidence is). The **spread** (std across samples) = stability of self-assessment.

**Why:** these are the evidence-side confidence; complementary to A1/A2 (a claim can have high
verdict agreement but low/​unstable evidence ratings).

**How:** per dim, mean and std over the K samples; aggregate the three. **Aggregation is a
design choice the deep-research said should NOT be a plain mean** — compute and compare:
`mean`, **weakest-link `min`** (a verdict is only as confident as its shakiest facet), and a
later **learned weighting** (logistic regression of correctness on the three dims). The chosen
aggregation is itself one of the candidate confidence signals in Part B.

### A4. Semantic entropy on the justification (free-text)
**What:** how much the K free-text justifications *disagree in meaning* (not wording). Low =
the model reasons consistently; high = it rationalizes differently each run.

**Why:** a reasoning-stability signal we get **for free** (the `justification` field already
exists). The **discrete** variant needs no logprobs — fits our backend.

**How:** embed the K justifications (`paraphrase-multilingual-MiniLM`, already in repo),
cluster by cosine ≥ τ (or bidirectional-NLI equivalence), then entropy over the cluster-size
distribution `H = -Σ_j (n_j/K)·ln(n_j/K)`. No token probabilities involved.

### A5. Inter-dimension correlation (redundancy check — NOT a per-claim signal)
**What/why:** the deep-research *killed* the "rubric dims collapse to one signal" claim, so
redundancy is an open risk. If our three confidence dims move together, three is pointless.
**How:** per claim take each dim's mean over K → an `[n_claims × 3]` matrix → **Spearman**
correlation between the columns (ordinal scores). If any pair |ρ| > ~0.9, collapse/drop a dim.
Run once over the dataset, before committing to the dim set.

---

## Part B — Evaluation metrics (which signal to trust + where to threshold)

Two distinct targets — keep them separate:
- **Verdict quality** — does the binary verdict match gold? (rare PASS class)
- **Confidence calibration** — does the confidence *number* predict correctness, and where do
  we cut for deployment?

### B1. AUC-PR — verdict discrimination on the rare class
**What/why:** area under the precision–recall curve for detecting the rare **PASS** class
(ranking claims by `veracity`/P(pass)). **Preferred over AUROC under imbalance** — AUROC is
optimistic because the large easy-FLAG majority inflates it; AUC-PR stays honest on the 13–16%
positives. (AUROC is also *discrimination, not calibration*.) **How:** sweep the veracity
threshold, plot precision vs recall for PASS, integrate.

### B2. Class-wise ECE (adaptive binning) — calibration, per class
**What:** Expected Calibration Error = bin predictions by confidence, and in each bin compare
**average confidence to actual accuracy**; the size-weighted average gap is the ECE. **Why
class-wise + adaptive:** pooled ECE is dominated by the FLAG majority and hides PASS
miscalibration → compute **separately for pass-predictions and flag-predictions**. Plain ECE is
*gameable* and *binning-sensitive* → use **equal-mass (quantile) bins** (equal sample count per
bin), not equal-width. **How:** per predicted class, adaptive-bin by confidence,
`ECE = Σ_b (n_b/N)·|conf_b − acc_b|`. Report `ECE_pass`, `ECE_flag` (+ macro-avg).

### B3. Balanced Brier — calibration, proper scoring rule
**What/why:** `Brier = mean((confidence − correct)²)` — a proper scoring rule (rewards honest
probabilities), reported **per-class then averaged** so the rare class isn't drowned out.
**How:** compute Brier over pass-predictions and over flag-predictions; average the two.

### B4. Risk–coverage — the deployment threshold tool
**What:** as we **abstain** on low-confidence claims, error among the rest drops. **Coverage** =
fraction acted on; **risk** = error rate among those. **Why:** this is literally how confidence
→ thresholds; the bottom (abstained) slice gets the IDEA-002 "still checking" soft-nudge.
**How:** sort claims by confidence (desc); for each coverage c (top-c% by confidence), compute
risk = error rate; plot risk vs c. Summarize with **AURC** (area under risk–coverage). Pick the
operating threshold for a target risk via the principled rule (Geifman & El-Yaniv 2017,
arXiv:1705.08500). **This is also how we rank the Part-A signals** — the signal whose
risk–coverage curve drops fastest (most error removed per unit of abstention) wins.

### Caveat (deep-research)
Bin-based calibration (B2/B3) needs **large samples**; our PASS class is small (~300, fewer
after artifact exclusion) → per-class numbers will be **noisy**. Concrete argument for the
two-run protocol below and for generating more PASS via corrections-mining. Even the best
black-box confidence has limited calibration → temper expectations.

---

## Eval protocol — two runs

Run the **whole eval twice**, identical except for the eval set composition:
1. **Without** the synthetic true claims — the *honest, in-the-wild* number on real publisher
   gold (heavily flag-skewed; PASS calibration will be noisy — that's the real-world story).
2. **With** the synthetic true claims (`mined_corrections_v1.parquet`) — a *balanced* set that
   gives enough PASS examples to actually measure PASS-side discrimination (B1) and calibration
   (B2/B3) with less noise.

Report both; the gap shows how much the synthetic balancing buys us on the rare class. Each run
is **also** reported including vs. excluding the artifact/satire `modality` bucket (headline =
text-only) — so 2×2 = four metric tables in total.
