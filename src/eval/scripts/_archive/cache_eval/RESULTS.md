# cache_eval — Run 1 results (2026-06-05)

Gate model: `llama-3.3-70b-versatile` (bidirectional, temp 0). Seed: 66
correctly-verified AVeriTeC claims (`verdict_4class==gold_label`; 52 Refuted, 14
Supported). 500 variants generated (264 equivalent / 236 adversarial), matched
each against its source seed. Artifacts: `data/variants.parquet`,
`data/variants_gated.parquet` (gate columns cached → re-score for free).

## The adversarial cases really are cosine-near

Mean cosine to seed: **equivalent 0.926 vs adversarial 0.868** — only ~0.06
apart. Negations land at cos **0.92–0.93**. So no cosine cut separates them: a
cosine-only cache false-HITs nearly every adversarial variant.

## Threshold sweep (recall / precision / adversarial false-HIT)

| thr | cosine-only R | P | advFH | cosine+GATE R | P | advFH |
|---|---|---|---|---|---|---|
| 0.55 | 0.996 | **0.530** | **0.987** | 0.875 | **0.983** | **0.017** |
| 0.70 | 0.977 | 0.545 | 0.911 | 0.860 | 0.983 | 0.017 |
| 0.85 | 0.879 | 0.604 | 0.644 | 0.769 | 0.981 | 0.017 |
| 0.90 | 0.746 | 0.642 | 0.466 | 0.640 | 0.983 | 0.013 |
| 0.95 | 0.489 | 0.662 | 0.280 | 0.405 | 0.991 | 0.004 |

**cosine-only precision tops out at ~0.66** (and only by discarding half the true
equivalents); **cosine+gate holds precision ~0.98 at every threshold.**

## Headline findings

1. **Cosine alone is unsafe.** At any usable threshold it false-HITs 28–99% of
   verdict-flipping variants (precision 0.53–0.66). Confirms the research thesis
   directly on our data + embedding model.
2. **The gate is the precision arbiter, and it works.** Threshold-independent gate
   stats: recall **0.879**, adversarial false-HIT **0.017**. Per axis, the
   verdict-flips that matter most are caught **perfectly**:
   - negation **0/66** ✅  · entity **0/61** ✅ · number **1/43** · scope **3/66**
3. **Threshold is a RECALL/cost lever, not a precision lever.** Gated precision is
   ~flat (0.98–0.99) across the whole sweep — precision is bounded by the *gate*,
   not the cut. → set the cut **low** to keep recall; sending more candidates to
   the gate costs gate-calls, not safety. **Recommended `SIMILARITY_THRESHOLD ≈
   0.60`** (gated recall ~0.87, precision ~0.98, F1 ~0.93). To push precision past
   0.98 you must improve the *gate* (scope handling), not raise the threshold.

## Qualitative: the residual errors are mostly label noise, not gate failures

- **4 false-HITs** (gate said equivalent on an adversarial variant): 3 are
  fuzzy-**scope** edits that are arguably truth-equivalent ("fine people"→"a fine
  person"; added "Most"/"some"); 1 **number** case where the generator *appended*
  a date/attribution rather than changing the claim's number — so the claim is in
  fact unchanged (generator label noise). **No negation or entity slipped.**
- **32 false-MISSes** (gate rejected a "should-HIT" variant): **23/32 are `filler`
  variants with epistemic hedges** ("apparently X", "it appears that X"). The gate
  is *logically right* — "apparently X" does not mutually entail "X" — so these
  aren't gate errors either; the **`filler` axis using epistemic hedges isn't
  truly equivalence-preserving.** The genuine recall cost is the ~9
  paraphrase/synonym misses.

So the gate's *effective* safety on clean verdict-flips (negation, entity, number)
is ~100%, and its *effective* recall is higher than 0.879 once hedge-filler label
noise is discounted.

## Follow-ups (generator + gate)

- **Generator:** drop epistemic-hedge filler from the `equivalent` set (or relabel
  it); make the `number` axis *change* a number rather than append one.
- **Gate:** scope is the one soft axis — worth a targeted prompt tweak or the
  fine-tuned-NLI comparison (research open-Q 2). Negation/entity need no work.
- **Multilingual:** all seeds are EN; add FR variants to test cross-lingual
  equivalence (a risk the research flagged, untested here).
- Optional: match variants against the *whole* seed cache (ANN retrieval) rather
  than the known source pair, to also measure cross-claim false-HITs.
