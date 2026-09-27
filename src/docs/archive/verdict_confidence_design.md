# Verdict eval redesign: binary decision + self-consistency confidence

**Status:** design brief (2026-06-13, w/ Daniel). Supersedes the 4-class AVeriTeC
framing for *our own* verdict eval. Prior-art `/deep-research` running in parallel —
open questions tagged **[DR]** will be filled from it.

## Motivation

The 4-class AVeriTeC scheme (Supported/Refuted/NEI/CE) is a poor fit for *our* eval:
- On the publisher gold (`eval_v1`, N=2370) it is **77.5% Refuted, 1.1% NEI** — accuracy
  ≈ the majority baseline, NEI statistically unmeasurable, and the CE bucket is a
  grab-bag (51% "misleading" + 21% "mixture" + context/half-true).
- It throws away the two things publisher schemes actually encode — an **ordinal
  veracity** spine and an orthogonal **mode** tag (manipulation / context / attribution /
  satire) — and it gives us no **confidence** signal.

**Resolution:** make the *verdict* simple (binary, deployable, gold-mappable) and move all
the multidimensionality into a **confidence layer**, which is where it earns its keep —
because confidence is what will **set the deployment thresholds** (when to nudge, when to
abstain with a "still checking" soft-nudge — IDEA-002).

## 1. The binary verdict (gold + model)

**Decision rule (gold + deployment):**
- **PASS** — sufficient evidence to *support* the claim.
- **FLAG** — otherwise: not enough evidence to support, OR evidence refutes, OR conflicting.

This is exactly `add_binary_label.py`'s existing `binary_label` (`pass` iff Supported, else
`flag`) → flag 86.8% / pass 13.0%.

**Harmonization collapses to ONE cut-point.** Binarizing moves all the per-publisher
harmonization difficulty to a single decision: *where on each veracity spine does PASS
become FLAG?* Decided with Daniel 2026-06-13: **`Mostly True` → PASS**, `Half True` /
`Mixture` and below → FLAG. That single boundary is now the whole harmonization judgment.

**Mode tag rides along.** The artifact/satire bucket (image/video/AI-generated/altered/
miscaptioned/satire) is **17.6% of `eval_v1`, ~all Refuted** — and a *text-only* verifier
structurally cannot reproduce it. Tag a `modality ∈ {text, artifact}` per claim and
**report every metric twice — including and excluding the artifact bucket — with the
text-only number as the headline.**

**Model side:** off the synthesis, emit a **graded veracity (1–5)** → threshold to the
binary. Keep it graded (not a bare binary) so the threshold is a *tunable knob* and
veracity's sample-spread becomes a confidence signal (below).

## 2. The confidence layer (self-consistency engine)

Logits are unavailable (Groq does not expose `logprobs`, confirmed 2026-06-13), so
confidence is **sampling-based**. Resample the **verdict + confidence-Likert step K times
off the FIXED synthesis** (the current `_evaluate_*` calls on `analysis`) — this isolates
*judgment* confidence given fixed evidence, needs no re-retrieval, and is a tiny code
change. (Resampling the whole synthesis = a later, pipeline-level variant.)

**One K-sample run yields four confidence signals, in two complementary families:**

| Family | Signal | Reads |
|---|---|---|
| Decision-side | verdict agreement (majority-vote fraction) | does the model waffle on flag↔pass? |
| Decision-side | veracity spread (variance across samples) | how stable is the verdict direction? |
| Evidence-side | confidence-Likert **mean** (= the "decomposed" confidence) | does the model think the evidence is strong? |
| Evidence-side | confidence-Likert **spread** | how stable is its self-assessment? |

They catch different failures (high agreement + low Likert = "consistently flags on thin
evidence"), so the eval's job is to find **which signal best predicts binary correctness.**

**Confidence-Likert dims** (the old Likert, `veracity` removed → it now drives the binary;
the rest reframed as pure confidence facets): **evidence sufficiency · source reliability ·
evidence agreement**. Keep to 3–4. **[DR✓]** decomposition *does* help (confirmed); the
"rubric dims collapse to one unidimensional signal" caution (arXiv:2509.20293) was the only
claim the adversarial pass **killed** (1-2) — so redundancy is a *risk, not a finding*. →
**add a cheap empirical check: inter-dimension correlation on our own Likert outputs**; if
the 3 dims correlate >~0.9, collapse them. **Aggregation: NOT a simple mean** (confirmed) —
use weighted/learned aggregation.

**Aggregation of the K samples — prefer entropy over a raw vote fraction.** **[DR✓]**
predictive / **semantic entropy** over the sampled verdicts is the better-calibrated
self-consistency signal, and Farquhar et al.'s **discrete semantic entropy needs no
logits/NLI** (Nature 2024) — directly usable for us. Keep vote-fraction as the simple
baseline.

**Params:** **K = 10** (pinned 2026-06-14) and temperature (>0, tuned on our data). 10 samples
also gives semantic entropy on the justification enough resolution.

## 3. Evaluation = a calibration study

Compare the confidence signals by how well they predict binary correctness. **[DR✓]** the
metric suite is shaped by our class imbalance (~13–16% PASS):
- **Do NOT report a lone ECE.** ECE is *gameable* and binning-sensitive; if used, use
  **adaptive / equal-mass (quantile) binning**, and report **class-wise ECE** (calibration
  conditioned on the predicted class) — standard ECE hides minority-class miscalibration.
- **Discrimination under imbalance: AUC-PR**, not AUROC alone (AUROC is optimistic under
  imbalance — and is *discrimination, not calibration*).
- **Balanced / class-conditional Brier score** as the proper scoring rule.
- **Selective-prediction / risk–coverage** (accuracy@coverage) — *this is the
  threshold-setting tool*, and there's a **principled threshold-setting algorithm** off the
  risk–coverage curve (Geifman & El-Yaniv, arXiv:1705.08500).

Report on `eval_v1` binary, incl/excl the artifact bucket. **[DR✓] Caveats:** bin-based
calibration metrics need *large* samples, and our PASS class is small (~300, fewer after
artifact exclusion) → per-class calibration will be noisy (a real argument for *more* PASS
examples — ties to the synthetic-corrections mining). Even the best black-box confidence
methods have limited calibration → temper expectations. FC-specific failure mode = systematic
**overconfidence**, worst on the minority class.

## 3b. Post-level FLAG step (decided 2026-06-15, w/ Daniel)

The 2026-06-14/15 audit showed the binary-vs-publisher gap is dominated NOT by verifier errors but
by a **category mismatch**: we score an atomic CLAIM's truth against a POST-level flag, while the
misleading element (framing / implied causation / date / partisan spin) lives in the raw post and
was distilled out of `claim_text`. Of the 6 confident-pass "errors" inspected: **3 weren't errors**
(true claim, context absent from input — ev2215/ev1689/ev1630), **2 were prompt-fixable mis-scores**
(the debunk was in the retrieved evidence but the verdict scored the literal shell — ev1229/ev569),
**1 was a genuine retrieval miss** (ev716, echo-chamber sources). Resolution — **split the judgment**:

- **Atomic claim verdict** (`verify()` as today: retrieve → synthesise → veracity + evidence).
  Context-free, **cacheable** (Tier-1), the expensive part; this is what recurs across posts.
- **Post FLAG step (NEW):** one cheap LLM call, **no retrieval, not cached**. Inputs = the **raw
  post** + every atomic claim's {verdict, evidence/analysis, confidence} (fresh or cache-pulled).
  Prompt: *given the evidence gathered (or its absence) for each atomic claim AND that the claim was
  made in the context of THIS post, should the post be flagged as potential misinformation?* The
  flag is on the POST; the per-claim fact-check is still cached & reused. **Same cached claim →
  different post-flags in different posts** (misleadingness is contextual). This step IS the parked
  misleadingness pass (IDEA-009) + the open post-level aggregation question, unified. It needs a
  misleadingness-aware prompt (fixes Category B), not only the raw post (which fixes Category A).

**Production streaming / early-exit (decided 2026-06-15):** the flag must fire as EARLY as possible.
- Decompose the post → verify atomic claims (Tier-1 hits return instantly; novel claims run Tier-3,
  possibly in parallel).
- **As each claim's verdict returns, run the FLAG step over ALL claims verified so far + the raw
  post.** If flag → **flag the post immediately**, don't wait for the rest. If not → wait for the
  next claim's verdict and re-check.
- A post **PASSES only once ALL claims are back** AND the flag step clears the whole post.
- Asymmetry by design: **flag = early-exit on the first triggering claim; pass = all-clear of every
  claim.** Minimises latency on the harmful case, conservative on the safe case. Ties to IDEA-002
  (soft "still checking" nudge while later claims are pending).

**Eval caveat:** `eval_v1` will UNDERSTATE this step's value — its `claim_text` is already distilled
(the source ClaimReview/FCT API never carried the raw post; we didn't strip it, it was never there).
So the post-flag step's real payoff needs **raw-post gold** (feed_study annotation). On `eval_v1` we
can only proxy the post with `claim_text` + `claimant` and report `soft_flag` separately.

## 4. Deployment payoff

The risk–coverage curve sets the operating threshold(s): high-confidence FLAG → hard nudge;
high-confidence PASS → silent; low-confidence → the IDEA-002 "we're still checking this"
soft-nudge / abstain. Confidence is not just an eval metric — it *is* the nudge/abstain knob.

## Backend note

Self-consistency is the primary method *because* Groq has no logprobs. If a logit arm is
ever wanted, stand up a **local vLLM** model (native logprobs + matches the self-hosted
deployment target). Verbalized confidence is dropped as redundant/weak.

## Implementation sketch (incremental)

1. Add graded `veracity` + the 3 confidence dims to the verdict output; derive binary by
   threshold; tag `modality`. Review/tighten the Likert prompt (rubric anchors + decide on
   few-shot examples — currently zero-shot).
2. Resample the verdict+Likert step K times off the fixed synthesis (reuse `_evaluate_*`).
3. A scorer computing the confidence signals + evaluation metrics — full spec (what / why /
   how to calculate each) in **`confidence_metrics.md`**.
4. **Two runs** (× incl/excl artifact = four metric tables): **without** the synthetic true
   claims (honest in-the-wild number) and **with** them (`mined_corrections_v1.parquet`, gives
   enough PASS to measure the rare class). Pick the best signal off the risk–coverage curve;
   read τ for the deployment threshold.

## Deep-research findings (2026-06-13, 22 sources, 24/25 claims confirmed)

The design held up — the research **validated** the core choices and sharpened the metrics +
aggregation. Net changes folded in above:

**Confirmed:**
- Self-consistency is the right logprob-free confidence, and **outperforms verbalized**
  (which is *systematically overconfident* — dropping it was correct).
- **Decomposition into a multidimensional rubric helps**; the lone counter-claim (dims are
  redundant) was the one claim *killed* in verification → treat redundancy as a risk to
  check, not a finding.
- Risk–coverage is the right threshold-setting tool, with a principled algorithm.

**Sharpened (changed the brief):**
- Aggregate the K samples via **predictive / semantic entropy** (discrete variant = logprob-
  free), not just a vote fraction. Rubric aggregation ≠ simple mean.
- Metric suite under imbalance = **AUC-PR + class-wise ECE (adaptive binning) + balanced
  Brier + risk–coverage**, *not* a lone ECE; AUROC is discrimination, not calibration.
- Small PASS-class n makes per-class calibration noisy → argues for more PASS examples
  (synthetic-corrections mining) and tempered expectations.

**Key sources:** Wang et al. self-consistency (2203.11171); Kuhn semantic entropy (2302.09664)
+ Farquhar discrete semantic entropy (Nature s41586-024-07421-0); SelfCheckGPT (2303.08896);
Xiong black-box confidence (2306.13063); Nixon ECE pitfalls (1904.01685); Geifman & El-Yaniv
selective prediction (1705.08500). Full set + the killed claim: `clog/130626.md` /
deep-research run `wf_9070d389-3ca`.
