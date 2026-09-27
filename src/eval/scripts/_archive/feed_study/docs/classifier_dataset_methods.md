# Classifier dataset & training — methods notes

Reference notes for building the selection-gold dataset and training the
post-discrimination classifier (renamed 2026-06-08 from "Tier-0 selection
classifier"). Written 2026-06-08 from two deep-research passes (see
`src/clog/080626.md`). The forward-looking task lives in `src/clog/TASKS.md` #1/#2;
this doc is the *prose* explanation of the three subtle methodology points, kept
here so they don't get lost in the compressed task bullets.

Context: we are fine-tuning a small multilingual encoder (**mmBERT**) as a cheap
production gate over short social posts (fr/en primary, some es/ar), concept-bottleneck
architecture (3 binary heads V/R/H + a learned holistic gate head), trained on
~5–10k LLM-teacher *silver* labels + human correction, evaluated on a ~1000-post
human *gold* test set. Positive class is rare (~10–19% prevalence).

---

## 1. Generate our own learning curve

**What it is.** A learning curve plots performance (y = F1) against training-set
size (x = number of labeled examples). You fine-tune the *same* model with the
*same* hyperparameters on increasingly large subsets of the training pool, and
evaluate every version on the *same fixed* held-out gold set. The shape tells you
the relationship between "more labels" and "more accuracy."

**Why the literature's "~5–10k is enough" doesn't transfer.** Data efficiency is
task-specific — it depends on task difficulty, number of classes, class imbalance,
and how many languages the model must learn the task in. The 5–10k figure comes
from English, balanced, often single-domain studies. Our task is harder on three
axes at once, and they compound:

- **Multilingual** — effectively learning the task separately in fr/en/es/ar, so
  the data is split four ways.
- **Rare positive (~10–19%)** — of 5k examples, only ~500–950 are positives.
- **Multi-head** — V, R, H, and the gate each need enough positives.

Stacked: 5k × 15% ÷ 4 languages ≈ **~190 positives per language**, split again per
head. *That* — positives per (language × head) cell — is the scarce resource, not
the 5k total. The real question is not "is 5k enough?" but "how does F1 climb as we
add data, and where does each cell flatten?"

**How to do it:**
1. Freeze everything except training size: same mmBERT base, same hyperparameters,
   same fixed gold eval set, same preprocessing.
2. Subset sizes, denser at the low end where the curve moves fast:
   e.g. 250, 500, 1k, 2k, 4k, 6k, 8k, 10k.
3. For each size *n*: draw *n* examples (stratified to preserve positive rate +
   language mix), train, evaluate on the fixed gold set, record **F1 overall +
   per-head + per-language**.
4. Repeat each *n* with 3–5 seeds; plot **mean ± spread** — small subsets are
   high-variance and you can't read a plateau without error bars.
5. Plot F1 vs *n*. Plateau = where marginal gain per +1k drops below what we care about.

**Decisions it drives:**
- Curve still climbing at 10k → need more data; keep sourcing/labeling.
- Plateaus at 4k → labeling past that is wasted effort.
- **Per-language curves diverge** — e.g. English flattens at 3k but Arabic still
  rising at 10k → concentrate *new* collection on Arabic specifically.
- Fit a power law (F1 ≈ a − b·n^−c) and extrapolate "how many more labels to hit
  target F1" — rough, but turns "need more data" into a number.

Cheap for us: train on growing *silver* subsets evaluated against gold → the whole
curve nearly for free. Converts the annotation budget from a guess into a measurement.

---

## 2. Precision is the predicted weak spot

**The teacher's two error rates** (LLM scored as a labeler vs truth):
- **Recall** = of all true positives, how many it caught. High recall = rarely misses.
- **Precision** = of everything it *labeled* positive, how many are actually positive.
  Low precision = many "yes" labels are false alarms.

LLM teachers skew **high-recall / low-precision** — trigger-happy, saying
"yes, check-worthy / harmful" too readily: catches the real ones but sweeps in
many negatives. (Keeping-Humans-in-the-Loop 2024: across 27 tasks median recall
0.83 ≫ median precision 0.65; recall > precision in 20/27.)

**Why rarity makes this bite — the math is the point.** Precision = TP/(TP+FP).
When positives are rare, negatives dominate, so even a small false-positive *rate*
on the huge negative pool yields a large *absolute* number of false positives that
swamp the few true positives. Worked example — 1000 posts at 15% prevalence (150
TP, 850 neg):
- 90% recall → 135 of 150 real positives caught ✓
- modest 20% FP rate on negatives → 170 of 850 wrongly flagged
- "positive" pile = 135 + 170 = 305, only 135 correct → **precision = 44%.**

Nearly half the silver *positive* labels are wrong despite excellent recall — and
the positive class is exactly the one the classifier most needs clean. The student
learns a bloated, fuzzy "positive."

**What to do:**
- **Report precision AND recall per head vs gold — not accuracy.** Accuracy is
  dominated by easy negatives and hides this; the honest number is per-head
  positive-class precision.
- **Route the teacher's *positive* predictions (+ low-confidence ones) preferentially
  to humans** — errors concentrate there. Active learning should be precision-aware,
  not just uncertainty-based.
- Expect **harm = worst-precision head** (LLMs over-flag anything edgy); verifiable
  likely cleanest.
- In the student: class weighting / focal loss for imbalance; don't inherit the
  teacher's over-eager positive bias.

---

## 3. Treat LLM noise as structured, not random

**Two kinds of label noise:**
- **Random/uniform** — errors independent of content, scattered. Models are fairly
  robust: averages out, and early stopping learns the signal before memorizing it.
- **Structured/systematic** — errors *correlated with content*: the teacher makes the
  *same* mistake on the *same kind* of example. For us, concretely: LLM reads
  **satire/jokes as genuine harmful claims** (already seen in extraction eval —
  "Squeezie uses the blood of Cyprien's father"), or calls **well-sourced news "not
  check-worthy"** (the CheckThat construct bias), or misjudges a French idiom.

**Why structured is worse.** The errors form a consistent pattern, so the student
learns the pattern as if it were signal — it can't average out because it isn't
random, it's a fake regularity. The student reproduces the teacher's blind spots.
(Our protection: eval is *human* gold, so it exposes the blind spot — as long as the
gold annotators don't share it; a silver eval would hide it entirely.) And DNNs will
fit it — high-capacity nets memorize even random labels given enough epochs (Zhang
et al. 2017); coherent structured noise they fit faster and more confidently
(SiDyP, arXiv:2505.19675, shows LLM noise is structured and DNNs inadvertently fit it).

**Mitigations:**
- **Early stopping on the GOLD dev set.** Noise-fitting signature: gold-dev F1 rises,
  peaks, then *falls* while silver-train loss keeps dropping — the falling part is the
  model learning the teacher's mistakes. Stop at the gold-dev peak; **monitor gold dev,
  never silver loss.**
- **Robust loss functions** — Generalized Cross Entropy, symmetric cross entropy,
  label smoothing, bootstrapping loss — down-weight probably-wrong labels. Cheap swap.
- **Filter / down-weight** silver examples where the teacher (or an ensemble) is unsure
  or disagrees; train on the cleaner subset, route the rest to humans.
- **Human-corrected hard cases anchor the model** against the systematic errors,
  especially if over-sampled. A handful of correctly-labeled satire examples breaks
  the pattern — the real antidote to the satire / sourced-news blind spots.

**Confidence-routing — tune the threshold on our own data:**
- **Don't hard-code a human-correction %.** The literature claim giving a specific
  routing fraction (e.g. "route 4–15%") was refuted in verification — committing to a
  number is unfounded.
- Routing rule: "send everything where teacher confidence < τ to a human," and **pick
  τ by measuring on our own gold/validation data** to hit a precision target or budget.
- **Why you can't trust the confidence number directly:** LLMs are miscalibrated and
  *overconfident* — "95% sure" is wrong more than 5% of the time; stated probabilities
  run systematically high. So don't read 0.9 as a true 90%. **But** confidence is useful
  *relatively*: low-confidence predictions are genuinely more error-prone (study example:
  11% error overall but ~50% among <70%-confidence cases). Use confidence to **rank**
  what to route, set the cutoff **empirically**, never lift a threshold from a paper.

---

## How the three connect

- **Learning curve** → *how much* data, and *which* languages/heads are starved.
- **Precision-awareness** → *where* the teacher errs (its positive labels, esp. harm)
  and what to route.
- **Structured-noise handling** → *how to train* so the student doesn't inherit the
  teacher's systematic blind spots.
- **Confidence-routing tuned on gold** → the mechanism feeding the human-correction
  loop that anchors all of it.

---

## Key citations (for the eventual methods section)

- **Silver ≈ gold for distillation:** Pangakis & Wolken 2024, "Knowledge Distillation
  in Automated Annotation" (arXiv:2406.17633) — student on GPT-4 silver ≈ on human gold
  across 14 CSS tasks.
- **LLM labels high-recall/low-precision; human validation essential:** "Keeping Humans
  in the Loop" (arXiv:2409.09467); Gilardi et al. 2023 (arXiv:2303.15056).
- **Structured LLM noise, DNNs fit it:** SiDyP (arXiv:2505.19675); Zhang et al. 2017.
- **Confidence-routed active learning:** Rouzegar & Makrehchi 2024 (arXiv:2406.12114);
  M-RARU (arXiv:2511.11574) — both preprints, best-case/balanced — tune on own data.
- **Test-set power + F1 CIs:** Card et al. 2020, "With Little Power…" (EMNLP) — size by
  power; F1 CIs via **Wilson or bootstrap/Bayesian**, NOT Wald/delta/t-test
  (arXiv:2309.14621; Wang & Li 2019, ACL P19-1405).
- **Data efficiency:** Sun et al. 2019 (arXiv:1905.05583); SetFit (Tunstall et al. 2022,
  arXiv:2209.11055). All English/balanced → external validity to our setting is the
  standing gap our own learning curve closes.
- **IAA for subjective check-worthiness:** Ocampo et al. 2026 (arXiv:2603.25269) —
  3 experts, κ 0.485–0.627, α≈0.54–0.57 binary.
