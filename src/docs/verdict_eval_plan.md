# Verdict Eval & Re-Harmonization Plan

*Owner: Daniel. Created 2026-06-27. The plan for validating + iterating the Tier-3 verdict
prompt on the re-harmonized 24-month fact-check gold, without circularity. Execute top-to-bottom;
workstreams 0→3 build the gold, 4→5 use it.*

---

## Decision locked

**Verdict = `veracity 1–5` + three confidence dims + a justification, from ONE evaluator call.**
We revert to the existing 4-dim Likert evaluator (`verify_prompts.build_likert_messages` /
`_evaluate_likert`) and **drop the second `_evaluate_4class` (ClaimCheck) evaluator** — that's the
streamlining (two evaluator calls → one). The synthesis call is unchanged (it still drives the
loop and emits `{analysis, next_query}`).

```
veracity            1–5   1 clearly-false … 3 not-clearly-either … 5 clearly-true   (the VERDICT, direction)
evidence_sufficiency 1–5  how much of the claim the evidence actually addresses     (confidence dim)
evidence_agreement   1–5  how concordant the evidence is (low = conflicting)        (confidence dim)
source_reliability   1–5  trustworthiness of the sources the verdict rests on       (confidence dim)
justification        text 1–3 sentences, cites evidence ids [n]
```

`veracity 3` is **"not clearly true or false"** and covers two distinct mechanisms, separated by
the dims, not by the label:
- **contested / half-true / mixture** → veracity 3 + **high** sufficiency + **low** agreement
- **unproven / no evidence** → veracity 3 + **low** sufficiency

Abstention lives in `evidence_sufficiency`, not in a separate label. The Tier-3 loop keeps an
**Exa×2 floor**: don't conclude with low sufficiency until Exa has been queried ≥2× (system
config; orthogonal to this eval).

---

## Non-circularity & leakage guardrails (everything obeys these)

1. **Gold = the publisher's own published rating, translated** by a frozen rule table — never by
   running our pipeline. Re-harmonize from **`original_rating`** (raw string), NOT the existing
   `harmonised_label` (it already collapsed True+Mostly-True→Supported and lost the 5-vs-4
   gradation we now need).
2. **Shared definitions ≠ shared assignment.** The verifier and the harmonizer share the codebook;
   the harmonizer only *translates a rating string* (sees publisher + rating, never the evidence,
   the fc-article body, or our verdict). LLM fallback (`harmonize.llm_map`) is a *different model*
   from the verifier and is used only for ratings the table misses.
3. **Only `veracity` needs gold.** Confidence dims are validated by **calibration + construct
   validity** (below); the justification by a **grounding score** — none of those need a gold
   label, so re-harmonization is purely about the 1–5 verdict.
4. **Leakage control at eval time** (separate from gold construction). **Revised 2026-06-28 (Daniel):**
   `date_ceiling` = **`claim_date`** (when the claim circulated — the *live-fact-checking* cutoff:
   evaluate how the system would perform at deployment, before the fact-check existed), NOT
   `review_date`. Where `claim_date` is missing (~40% of rows) fall back to `review_date`, tagged
   `ceiling_source` so the lenient-fallback rows are reported separately. Exclude the row's own
   `review_url` to be safe; **no fact-check-domain block** — the `claim_date` ceiling already
   withholds the fact-check (and same-claim sibling checks, which postdate the claim). (`verify.py`
   supports `date_ceiling` / `exclude_urls` / `exclude_domains`.)

---

## Workstreams

### WS0 — Re-harmonize raw ratings → `veracity 1–5` + `rating_subtype`
Extend `eval/harmonize.py` with a veracity map (keyed on `publisher_site` + lowercased
`original_rating`), plus a `rating_subtype` tag (drives the construct-validity test in WS4).

| veracity | rating examples | `rating_subtype` |
|--:|---|---|
| 5 | True, Correct, Vrai, Legit | `clear_true` |
| 4 | Mostly True | `mostly_true` |
| 3 | Half-True, Mixture, Partly-false, Misleading, Missing/Needs context, Exaggerated, Distorts | `mixed` |
| 3 | Unproven, Unsubstantiated, No evidence, Infondé, Unverified, Disputed, Outdated | `unprovable` |
| 2 | Mostly False | `mostly_false` |
| 1 | False, Pants on Fire, Fake, Scam | `clear_false` |
| 1 | Altered/AI-generated/Miscaptioned/Misattributed media | `altered_media` |
| — | Satire (`is_satire`) | excluded from veracity gold by default (revisit in audit) |

- Numeric publishers (LeadStories, 20 Minutes) via existing `numeric_map`/`refine_numeric`,
  re-binned to 1–5 (`frac<0.2→1, <0.4→2, 0.4–0.6→3, <0.8→4, else 5`); the `_NEI_PAT`/`_CE_PAT`
  alternateName disambiguation routes to `unprovable`/`mixed` at veracity 3 (LeadStories is
  ~all-false → mostly 1, with "No Evidence"→3-unprovable, "Nuanced"→3-mixed).
- `judged_axis` (artifact vs content) and `is_attribution` are carried through — the attribution
  Open Question still applies (a true attribution of a false claim is not "veracity 1" for the
  post).
- Output: a 24-month `verdict_dataset.parquet` with `gold_veracity`, `rating_subtype`,
  `original_rating`, `publisher_site`, `judged_axis`, leakage fields (`review_url`, `review_date`),
  `has_image`, `image_paths`, `language_code`.

### WS1 — Image-serialize the dataset (DeepSeek is text-only)
- For each `has_image` claim — prioritizing `claim_location ∈ {image, both}` from
  `image_audit.py` (those *need* the picture) — run the **Stage-2 VLM serializer**
  (`stage2_eval.py`, Qwen3-VL → `image_serialization`) over `image_paths`, store as a column.
- The verifier receives `claim_text + image_serialization` as context. Guardrails: serialize the
  **post image only** (not any fc-article image → no verdict leakage); description is **content-
  only** (no verdict language). Cache per (image, model) so it's computed once.

### WS2 — Audit, extensively
- Mirror the gate audit (`audit_dataset.py` pattern): an **independent judge** (different model
  from the verifier) per item — `rating→veracity` correct? in-scope/checkable? satire?
  attribution-axis? image needed & serialized? — with **human review of every borderline and every
  rating→veracity disagreement**, logged to a provenance ledger (cf. `gate_dataset_provenance`).
  Drop satire / uncheckable / leakage rows.
- Special check: a sample of `rating_subtype=unprovable` rows — confirm they're genuinely
  "no evidence either way" and not mislabeled resolved claims (protects the gold-3 split).

### WS3 — smoke / dev / test split
- Hash-stable: fold = `md5(review_url) % k`, **frozen forever** (the bug already fixed on the gate).
- **Stratify** by `gold_veracity` (1–5), `publisher_site`, `language_code` (EN/FR), `has_image`.
- **Balance the skew:** the corpus is false-heavy (LeadStories ≈ all-1). Down-sample the false pole
  in dev/test so all five veracity levels are represented — otherwise metrics are dominated by easy
  confident-FALSE (the prior analysis's "flag-skewed" caveat). Keep the natural distribution in a
  separate `natural` slice for a realism read.
- Sizes: **smoke ~20** (instant), **dev ~150–200** (iteration loop), **test ~500+** (held out).

### WS4 — Eval harness: how we judge each signal
**Veracity (vs `gold_veracity`):** ordinal — **MAE on 1–5**, off-by-one accuracy, Spearman — plus
the **nudge collapse** reported separately: `≤2 → hard-flag candidate · 3 → soft "not established"
nudge · ≥4 → pass`. Every run under `date_ceiling = claim_date` (review_date fallback, tagged) +
`review_url` exclude (see §4).

**Confidence (item-1, re-run + extended):** port the archived
`_archive/verdict_confidence/dimension_distributions.py`. Two tests:
- **Predictiveness** (does the dim predict verdict-correctness): per-dim **AUROC / AURC / Brier**,
  redundancy (Spearman), **dynamic-range** check, **confident-TRUE error rate**. *(Prior finding to
  re-confirm: `source_reliability` near-useless (AUROC 0.56), `evidence_agreement` weak (0.60),
  `evidence_sufficiency` best single (0.65), cross-model agreement best (0.66, needs a panel),
  veracity-magnitude **anti-predictive** (0.49); dims collapse to the 4–5 ceiling; errors cluster
  at confident-TRUE (v=5 only 69% accurate). n=60 → re-test on the bigger balanced set.)*
- **Construct validity** (do the dims encode what they claim — the stronger, non-circular test):
  group by `rating_subtype`, test
  - **H1** `evidence_agreement` is lower on `mixed` than on `clear_true`/`clear_false`,
  - **H2** `evidence_sufficiency` is lower on `unprovable` than on resolved,
  - **H3** `source_reliability` shows no separation (re-confirm it's a dud from a 2nd angle).
  This also disambiguates the two gold-3 sub-types and **diagnoses the ceiling problem** (flat dims
  across sub-types ⇒ the prompt isn't giving the dims range ⇒ WS5 target).
- **Calibrated confidence** = learned `f(dims + verdict-token logprob)` fit on dev (DeepInfra
  exposes logprobs); pick/weight signals by Brier/AUROC. Run a small **2–3 model panel on dev** so
  the cross-model-agreement signal (strongest last time) is available.

**Justification (grounding — orthogonal axis, never feeds the verdict gold):** decompose →
**independent-judge** entailment of each cited `[n]` → **ALCE citation precision/recall/F1** +
**JFR** (correct verdict but ungrounded justification). Scorer validated on ~50 hand-labeled
(sentence, cited-evidence) AIS pairs. (Cheap-NLI MiniCheck/AlignScore is the scale-up if it becomes
a production gate.)

### WS5 — Iterate the prompt on dev
Tight loop (dev cached): run → veracity MAE + confidence predictiveness/construct-validity +
grounding → edit prompt → re-run. **Two concrete targets from the prior finding:** (a) break the
**ceiling collapse** so the dims use 1–3 (validated by the construct test actually separating
sub-types), and (b) attack the **confident-TRUE error** (veracity-5 passing misinfo). Promote to
test only when dev stabilizes.

---

## Open decisions (defaults marked ✓)
1. **Balance the false skew** in dev/test ✓ (vs keep natural — kept as a separate slice).
2. **2–3 model panel on dev** for the cross-model-agreement signal ✓ (extra eval cost, dev-only).
3. **`source_reliability`**: keep emitting (redundancy says keep) but let WS4 decide its *weight* in
   the calibrated confidence ✓ (don't assume).
4. **Satire**: excluded from veracity gold by default ✓ (revisit specific cases in the audit).

## File map
- `eval/harmonize.py` — add veracity-1–5 map + `rating_subtype` (WS0).
- harvest parquets (`{politifact,snopes,leadstories,twentyminutes,afp,aap,fullfact,...}_harvest.parquet`)
  → rebuilt `verdict_dataset.parquet` (WS0–1).
- `eval/scripts/stage2_eval.py` (image serializer), `eval/scripts/image_audit.py` (`claim_location`) — WS1.
- `eval/scripts/audit_dataset.py` pattern — WS2.
- `_archive/verdict_confidence/dimension_distributions.py` — port for WS4.
- `pipeline/verify.py` (`_evaluate_likert`, drop `_evaluate_4class`), `pipeline/verify_prompts.py`,
  `pipeline/models.py` (`VerdictScores`) — the system under test (WS5).
