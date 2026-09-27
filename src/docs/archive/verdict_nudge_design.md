# Verdict → nudge redesign — design brief

**Status:** design brief (decisions, not a spec). Opened 2026-07-01.
**Supersedes:** the deleted `verdict_confidence_design.md` (the 3 Likert confidence
dims are being dropped — see §1). Related: `cache_design.md` (the claim/post cache).

This brief records *why* Stage-3 output is changing and *what* the target shape is,
so the caching vision and the nudge decision stay coherent while we implement.

---

## TL;DR — the decisions

1. **Drop the three confidence dims** (`evidence_sufficiency`, `evidence_agreement`,
   `source_reliability`). They don't predict our errors (§Evidence).
2. **`veracity` (1-5) judges the normalized claim's *literal* truth only** — it stops
   absorbing framing/misleadingness. That makes it a context-free property, cacheable
   against the claim.
3. **Replace the free-text `justification` with a structured `flags[]` enum.** The model
   already names the failure mode in prose on 82% of errors; we tag it instead of
   writing (and discarding) a sentence. Same move = flag taxonomy + ~70% Likert
   output-token cut + machine-readable.
4. **One evaluator call, two storage destinations.** The call sees **post + evidence**
   and emits `{veracity, flags[]}`; `veracity` (+ epistemic flags) → **claim DB**,
   framing flags (+ nudge) → **post DB**. No separate nudge *stage*.
5. **Nudge rule:** `nudge = veracity ≤ 3 OR any flag`. Flags map to nudge **type**
   ("false" vs softer "missing context"), not just yes/no.

---

## 1. Why change anything

Three findings from the clean 400-claim dev run (`verdicts_dev_clean.parquet`):

- **The confidence dims are dead weight for the decision.** As a per-claim confidence
  gate they have AUC ≈ 0.5 (`source_reliability` 0.365 — anti-predictive); abstaining
  on low `conf_mean` *lowers* flag-accuracy (0.915 → 0.885). The only calibrated free
  signal is `|veracity − 3|` (AUC 0.693). Nothing in the pipeline consumes the three
  dims today (caching TTL keys on `stopped_reason`, not these).
- **Errors are a 3↔4 calibration problem, not a knowledge problem.** On 28/34 flag
  errors (82%) the justification *already names the decision-relevant caveat*
  ("core confirmed **but** overstates / omits context / implies a link") — then maps it
  to the wrong number. The reasoning is right; the scalar collapse loses it.
- **The gold itself bundles framing, inconsistently.** All 14 over-flags are the model
  penalizing framing that the fact-checker rated *True*. A context-free `veracity`
  structurally cannot match a gold label that is a (claim, context) judgment — which is
  the real reason match caps below 100%, and the reason framing must live elsewhere.

## 2. The two-object model

The fact-checker's 1-5 answers "is this claim, *as published in this post*, misleading?"
— a **(claim, context)** judgment. We split that into two objects with different
lifetimes and cache keys:

| object | question | context? | cache key |
|---|---|---|---|
| **claim verdict** | is the normalized proposition true, per evidence? | no | **normalized claim** (broadly reusable) |
| **post nudge** | should we nudge *this post*, and how? | yes | **post** (near-duplicate reuse only) |

This is what the original design implied (context-free cache + context-aware nudge). We
keep it, but realize it with **one call** that has the post in context, splitting the
*output* by destination rather than adding a second stage.

## 3. Verify output schema

Before: `{veracity, evidence_sufficiency, evidence_agreement, source_reliability, justification}`
After:  `{veracity, flags[]}` (+ optional `justification` in eval/debug mode only).

`veracity` rubric loses the "score misleading down to 3" overloading:
5 true · 4 minor detail off/unconfirmed · 3 genuinely mixed/indeterminate on the
*literal* proposition · 2 mostly false · 1 false.

## 4. Flag taxonomy (and the cache split falls out of it)

| flag | fires when | judges | → cache |
|---|---|---|---|
| `NOT_ENOUGH_EVIDENCE` | load-bearing proposition can't be confirmed/refuted | claim vs world | **claim** |
| `CONFLICTING_EVIDENCE` | credible sources point both ways / cherry-picking | claim vs world | **claim** |
| `AUTHENTICATION_REQUIRED` | truth hinges on media genuineness (route out of text path) | claim type | **claim** |
| `MISLEADING_CONTEXT` | literal core true, but this post's framing creates a false impression | claim vs *this post* | **post** |
| `PROP_MISMATCH` | evidence confirms an adjacent fact, not the claim as stated — causation, scope word, exact figure, or media authenticity (absorbs the old `UNSUPPORTED_IMPLICATION`) | claim vs *this post* | **post** |

The **epistemic flags** are context-free (travel with the claim); the **framing flags**
are context-dependent (travel with the post). This is why storage splits by flag, and
why baking a framing flag into the claim-keyed verdict would be a bug (a
neutrally-framed future post would inherit a prior post's `MISLEADING`).

Caveat carried from the data: in *this* eval, `NOT_ENOUGH_EVIDENCE` skews
"false-by-absence" (a fabrication nothing on the web confirms), so `NEE → nudge` is
right here; in production, genuine unverifiability is different and may warrant abstain.
Don't hard-wire `NEE → false` without re-checking on live traffic.

## 5. Nudge rule

```
if AUTHENTICATION_REQUIRED:    route to media check (or abstain in text-only path)
elif veracity <= 3:            nudge, type = "false / mostly false"
elif MISLEADING_CONTEXT or PROP_MISMATCH:  nudge, type = "missing context / misleading"
elif NOT_ENOUGH_EVIDENCE:      nudge (this eval) — revisit for prod
else:                          pass
```

`OR any-flag` maximizes misinfo recall, but the model is *more* framing-sensitive than
gold, so the framing flags need a firing-rate calibration or they'll over-nag. Mapping
them to a softer nudge *type* (not a hard "false") is the lever, and it's the reason to
keep type, not just a boolean.

## 6. Caching & the one open empirical question

- Fresh (claim, post): the single call does everything. No second call.
- **Claim-cache hit on a newly-framed post:** we save retrieval+synthesis (the expensive
  part), but the cached verdict has no framing judgment for *this* post's wording. Two
  options: (a) a cheap framing-only pass over `(post + cached verdict)`, or (b) accept
  the claim-level flag as an approximation.
- **Which we need is empirical: how often does the same claim recirculate with
  *different* framing?** Rare → (b) is fine and there is genuinely never a second call;
  common → (a), a cheap no-retrieval pass. We can't answer this until the claim/post DBs
  exist and we observe traffic. Until then, build both DBs, log claim-cache hit rate and
  reframe rate, decide later.

## Evidence appendix (all from `verdicts_dev_clean.parquet`, N=400, flag-acc 0.915)

- Confidence dims AUC (separate flag-correct from flag-wrong): suff 0.501, agreement
  0.485, source_reliability 0.365, conf_mean 0.404, `|veracity−3|` **0.693**.
- Selective prediction by `conf_mean`: 0.915 → **0.885** at 50% coverage (worse);
  by `|veracity−3|`: 0.915 → **0.960** (works).
- 31/34 flag errors are high-confidence (conf_mean ≥ 4) — the mistakes are *confident*.
- Worst 4-class bucket: **Supported, flag-acc 0.726** — "literal core supported" ≠
  "don't nudge". Exactly the hole the framing flags fill.
- Justifications: 82% of errors already name the caveat in text; post-hoc (emitted after
  the scores; real CoT lives in the discarded `<think>` block) → safe to drop / replace.
  ~276 chars ≈ 69 tok ≈ **70.6% of Likert output tokens**.
- The 6 "confident-no-caveat" misses, on inspection, are **NOT gold noise** (2026-07-02
  audit). They split into (a) genuine causal/framing misses the flags target — Southport
  ("jailed *over* social-media-posts": fact-checker disputes the **causal** link, we
  confirmed only adjacent facts → PROP_MISMATCH); hantavirus 323-vials ("deadly viruses
  *went missing*" overstates the risk → MISLEADING_CONTEXT); living-standards (Full Fact
  *mixture* = contested by measure → an undetected CONFLICTING_EVIDENCE) — and (b)
  **over-normalization / eval-misalignment**: NHS "interactive report", David Geier "no
  medical license" — our atomic claim is a true shell but gold judges the broader original
  (the `judged_axis` proposition-scope issue, not a wrong label).

## Open / to validate before shipping

- [ ] A/B the `{veracity, flags}` evaluator + `nudge = veracity≤3 OR flags` on the
      **held-out test split** (not dev) — headline = nudge-worthiness, not gold 1-5.
- [ ] Calibrate `MISLEADING_CONTEXT` / `PROP_MISMATCH` firing rate (over-nag risk).
- [ ] Decide fold-vs-keep the AVeriTeC 4-class (benchmark comparability vs one clean
      evaluator).
- [x] Audit the suspected gold-noise misses — done 2026-07-02: none are gold noise;
      reclassified as causal/framing misses (flag targets) + over-normalization
      eval-misalignment. Real remaining item ↓.
- [ ] Fix over-normalization: ensure the extracted proposition matches what gold judges
      (NHS-report / Geier cases) — ties into the `judged_axis` eval-alignment work.
- [ ] Build claim + post DBs; log claim-cache hit rate and per-claim reframe rate to
      resolve §6.
