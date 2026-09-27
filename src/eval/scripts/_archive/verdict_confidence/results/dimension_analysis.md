# Likert dimension distributions & calibration — does lower confidence predict lower accuracy?

**Data:** last run's verdict data — `agreement.parquet` (n=60, 4-model panel, the only file
with all four dims) and `flag_eval.parquet` (n=1024, single-model, powered single signal).
**Reproduce:** `uv run python -m eval.scripts.verdict_confidence.dimension_distributions`
(formulas — AURC, equal-mass binning — mirror `score.py`).

**Orientation note.** `veracity` is a *direction* (1 clearly-false … 5 clearly-true), **not** a
confidence magnitude. So "confidence" is derived as `|veracity−3|`, cross-model `veracity` std,
or the three `evidence_*` dims. Correctness = `deployment_binary==gold` (panel) / `bare_correct`
(flag_eval). n=60 is small — treat panel numbers as directional; flag_eval (n=1024) is the
powered read.

---

## Headline answer

**No — lower confidence does *not* reliably predict lower accuracy, and the model's own
decisiveness (veracity magnitude) is the *worst* of the candidate signals.** On the n=60 panel
the deployment-verdict magnitude `|deployment_veracity−3|` has **AUROC 0.486** and
**AURC 0.218 — worse than random-order (0.200)**: the model's most decisive verdicts are not its
most accurate. The powered flag_eval read shows why — the magnitude signal looks weakly positive
(AUROC 0.595) **only** because it is dominated by a huge, easy confident-FALSE mass (veracity=1,
97.8% accurate); on the confident-TRUE side accuracy *collapses to 69.4%*.

**The two signals that *do* weakly track correctness are cross-model agreement and
evidence_sufficiency** (panel AUROC ≈ 0.65, AURC ≈ 0.11 — they roughly halve error at top
coverage). `source_reliability` carries essentially no signal (AUROC 0.56). The honest framing:
**errors concentrate at the confident-*true* pole, not at the uncertain middle** — so
"low confidence ⇒ low accuracy" is the wrong mental model for this flag task.

---

## 1. Scale usage — both halves of the rubric collapse, in opposite directions

The prompt says "use 2 and 4 deliberately." It is **not** honored.

**`veracity` collapses to the poles (drops the neutral 3):**

| file | 1 | 2 | 3 | 4 | 5 | mid {2,4} | poles {1,5} |
|---|--:|--:|--:|--:|--:|--:|--:|
| flag_eval (n=1024) | 53.1% | 8.7% | **2.7%** | 6.7% | 28.7% | 15.4% | **81.8%** |
| panel pooled (n=240) | 36.2% | 30.8% | **2.9%** | 16.7% | 13.3% | 47.5% | 49.5% |

Level-3 (neutral) is effectively abandoned (≈3% in both). flag_eval is near-binary at {1,5}; the
panel uses level 2 heavily but still avoids the dead center — the model commits to a direction.

**The three confidence dims collapse to the *ceiling* (drop 1–2):** pooled means 4.09–4.37,
almost everything ≥4.

| dim (panel pooled, n=240) | 1 | 2 | 3 | 4 | 5 | mean | share ≥4 |
|---|--:|--:|--:|--:|--:|--:|--:|
| evidence_sufficiency | 0.8% | 2.5% | 12.1% | 27.9% | 56.7% | 4.37 | 84.6% |
| evidence_agreement | 0.8% | 8.8% | 10.4% | 16.2% | 63.7% | 4.33 | 79.9% |
| source_reliability | 0.4% | 1.7% | 12.9% | 58.3% | 26.7% | 4.09 | 85.0% |

**Implication:** the confidence dims have almost no dynamic range — the model nearly always says
"evidence sufficient, agreeing, sources reliable." A signal that is 4-or-5 ~85% of the time can't
discriminate much. This ceiling, more than the veracity bimodality, is what limits calibration.

---

## 2. Redundancy — the 3-dim decomposition is *not* cosmetic

Spearman among the confidence dims (per-model means, n=60): **all |ρ| well below the 0.9
collapse threshold**, so keep all three.

| pair | ρ |
|---|--:|
| evidence_sufficiency ~ evidence_agreement | +0.586 |
| evidence_sufficiency ~ source_reliability | +0.489 |
| evidence_agreement ~ source_reliability | +0.331 |

They share a common "evidence-quality" component (all positive) but are far from interchangeable.
Note each confidence dim is **negatively** correlated with `veracity` (−0.10 to −0.40): stronger
evidence tends to accompany *lower* veracity, because in this flag-heavy sample solid evidence
usually points to "false." (gpt-oss raw values give even weaker inter-dim ρ, ≤0.17 — the modest
correlation above is partly a panel-averaging effect.)

---

## 3. Calibration / discrimination — the core question

### Panel (n=60), every signal vs `deployment_binary==gold`; base-rate correct = 0.800

Ranked by AURC (risk-coverage; **lower = better**, random-order ≈ 0.200):

| signal (higher = more confident) | point-biserial | AUROC | AUC-PR | AURC |
|---|--:|--:|--:|--:|
| cross-model agreement (−veracity std) | +0.159 | **0.658** | 0.872 | **0.113** |
| evidence_sufficiency mean | +0.169 | 0.650 | 0.872 | 0.117 |
| weakest-link min(3 dims) | +0.094 | 0.616 | 0.875 | 0.128 |
| \|panel mean veracity − 3\| | +0.152 | 0.622 | 0.861 | 0.139 |
| evidence_agreement mean | +0.058 | 0.598 | 0.856 | 0.152 |
| source_reliability mean | +0.080 | 0.560 | 0.843 | 0.153 |
| **\|deployment_veracity − 3\|** | **−0.032** | **0.486** | 0.796 | **0.218** |

Accuracy by ascending-confidence tertile makes the contrast concrete:

- **cross-model agreement** 0.65 → 0.85 → **0.90** (clean monotone: panel disagreement flags error)
- **evidence_sufficiency** 0.65 → 0.90 → 0.85 (rises then plateaus)
- **source_reliability** 0.85 → 0.70 → 0.85 (flat / non-monotone — no usable signal)
- **\|deployment_veracity−3\|** 0.85 → 0.80 → **0.75** (*decreasing* — more decisive ⇒ less accurate)

### flag_eval (n=1024, powered); base-rate correct = 0.848

Accuracy by veracity **level** exposes the asymmetry the magnitude signal hides:

| veracity | n | accuracy | bare pred |
|--:|--:|--:|:--|
| 1 | 544 | **0.978** | flag |
| 2 | 89 | 0.719 | flag |
| 3 | 28 | 0.821 | flag |
| 4 | 69 | 0.652 | pass |
| 5 | 294 | **0.694** | pass |

By **confidence magnitude** `|veracity−3|` it is **non-monotonic**: |v−3|=0 → 0.821,
|v−3|=1 → 0.690, |v−3|=2 → 0.878. The most-confident bucket is most accurate *only* because it
pools the 97.8%-accurate v=1 with the 69.4%-accurate v=5. The aggregate `|veracity−3|` signal is
therefore weak (AUROC 0.595, AURC 0.124 vs base-err 0.152) and its risk-coverage curve is
non-monotone (risk at 50% coverage 0.137 ≈ full-coverage 0.152). Raw veracity even correlates
**−0.358** with correctness — higher "true" calls are *less* often right.

---

## 4. Confident-error asymmetry — errors cluster at confident-TRUE

The single most important caveat for any "trust high confidence" rule. Same pattern in both files:

| slice | n | error rate | what the errors are |
|---|--:|--:|---|
| flag_eval, confident-TRUE (v=5) | 294 | **0.306** | **all 90 errors gold=flag** (misinformation passed) |
| flag_eval, confident-FALSE (v=1) | 544 | 0.022 | all 12 errors gold=pass (over-flag) |
| panel, confident-TRUE (v=5) | 11 | **0.545** | all 6 errors gold=flag |
| panel, confident-FALSE (v=1) | 28 | 0.071 | all 2 errors gold=pass |

In flag_eval, **58% of all errors land at veracity=5** vs only **8% at veracity=1**. A confident
"true" verdict is ~14× more error-prone than a confident "false" one, and *every* confident-true
error is a piece of flagged-gold content that slipped through. This is structural to a
flag-skewed task: confidence is anti-protective exactly on the dangerous side.

---

## 5. Per-dimension predictiveness verdict

Ranked best → worst at predicting verdict correctness:

1. **cross-model `veracity` agreement (low spread)** — best (AUROC 0.658, AURC 0.113, monotone).
   *Disagreement among the 4 models is the most reliable error flag.* Needs the panel.
2. **evidence_sufficiency** — best single dim (AUROC 0.650, AURC 0.117). "Does the evidence
   actually address the claim" tracks correctness.
3. **weakest-link min / panel veracity magnitude** — middling (AUROC ≈ 0.62).
4. **evidence_agreement** — weak (AUROC 0.598).
5. **source_reliability** — **near-useless** (AUROC 0.560; sliced AUROC 0.51 on flag-preds, 0.59
   on pass-preds; flat tertiles). *Refines the prior "anti-predictive" finding:* on this run it
   is not reliably negative, but it carries essentially no correctness signal — don't weight it.
6. **`|deployment_veracity−3|` (decisiveness)** — **anti-predictive** (AUROC 0.486, AURC 0.218 >
   random). The deployment verdict's own confidence-as-magnitude should *not* gate abstention.

**Take-away for the confidence wrapper:** drop magnitude-of-veracity and source_reliability as
confidence signals; gate on **cross-model agreement** + **evidence_sufficiency**, and treat any
**confident-TRUE (veracity≥4)** verdict as the high-risk bucket regardless of stated confidence.
All panel numbers rest on n=60 — directional, to be re-confirmed on a larger panel run.
