# Stage-3 (VERIFY) baseline — dev results

**Run date:** 2026-06-29. **Model under test:** Tier-3 verification loop (`pipeline/verify.py`),
serper→exa cascade, current working-tree prompts (incl. the corroboration gate). **No prompt
tuning** — this is an untuned baseline.

## Configuration (exactly what ran)
- **Retrieval:** `SEARCH_CASCADE = [serper, exa]` — Serper first (2 rounds), escalate to Exa on thin
  evidence. Confirmed live: serper resolved 15/20 smoke, exa 5/20; exa-escalation on ~25% of claims.
  *(An earlier run accidentally used tavily-only via a stale `.env` `SEARCH_PROVIDER` override +
  missing `--cascade` flag; discarded. This run is the correct cascade.)*
- **Leakage controls (locked 2026-06-28):** `date_ceiling = claim_date` (live-fact-checking cutoff;
  review_date fallback, tagged), exclude the row's own `review_url`. No fact-check-domain block.
- **Image input:** post-image serialization (Qwen3-VL) fed as labeled CLAIM CONTEXT (WS1).
- **Population:** balanced dev, `judged_axis=content`, **n=366** (319 in stage-3 scope after the
  deterministic media-dependency carve). 0 errors. avg 2.07 rounds, 13.7 LLM calls, 87s/claim.

## Headline — veracity 1-5 vs publisher gold
**Content, stage-3 in-scope (n=319):** MAE **0.925** · exact **0.46** · off-by-one **0.71** ·
Spearman **+0.60**.

> **Read this as a pessimistic bound.** We read all 16 confident-TRUE errors (below): the majority
> are gold/claim-alignment mismatches (miscaption, framing, decontextualization), not the verifier
> getting facts wrong. The WS2 gold audit (dev not yet audited) is expected to lift the headline.

## Nudge collapse (≤2 flag · 3 soft · ≥4 pass), n=319
Accuracy **0.60**. **Misinfo-passed (gold≤2 → pred≥4): 16/127 = 12.6%** — the load-bearing
"confident-TRUE" failure for a nudge tool.

```
gold\pred   flag  soft  pass
flag         103     8    16
soft          70     9    24     <- v3/mixed flattened to a pole, mostly to flag
pass           9     1    79
```
**Dominant error = MIXED (v3) claims flattened to the poles.** Gold has 115 v3; the verifier
predicts v3 only 20×, pushing 70 v3→flag and 24 v3→pass. The verifier avoids the middle.

## What the 16 confident-TRUE errors actually are (we read every one)
- **~6 miscaption/provenance** ("Video shows [real event]") — verifier correctly confirms the event,
  misses that the *footage* is recycled. **Phase-1b authentication, not text-verifiable.**
- **~5 framing/granularity** — fact-checker rates the post's misleading framing "mostly false"
  (welfare £70bn, Nightingale £532m, Trump "bleach"); verifier correctly assesses the literal claim.
- **~5 decontextualization / candidate gold errors** — e.g. "82nd Airborne *not* deployed": analysis
  correctly refutes the claim but the score inverted to 5 (negation slip); facts-check-out-but-gold=1.

Dropping all 53 media-phrased rows barely moves the headline (MAE 0.925→0.921), so media is *not*
the dominant driver — the gold/framing mismatch is broader and spread across sources.

## Confidence dimensions (n=366)
- **Scale usage:** veracity uses the full range (mean 2.66). `evidence_sufficiency` mean 4.02,
  `source_reliability` 4.26. **`evidence_agreement` collapses to 5 (78.7%)** — ceiling-bound, weak.
- **Predictiveness:** all dims AUROC ≈ 0.50 — they do **not** yet predict verdict-correctness, so they
  are not usable as an abstention/gating signal yet (WS5 target). `|veracity-3|` is anti-predictive
  (AUROC 0.44 — confident-when-wrong).
- **Construct validity (by `rating_subtype`):**
  - H2 `sufficiency(unprovable) < sufficiency(resolved)`: **✓ p=0.005** (the strongest signal —
    sufficiency genuinely drops on unprovable, 3.52 vs ~4.0).
  - H1 `agreement(mixed) < agreement(clear)`: ✓ but weak (p=0.04).
  - H3 `source_reliability` expected dud: weakly separates (p=0.03), tracks sufficiency.

## Breakdowns (content in-scope)
- **By source:** best = aap 0.50 / politifact 0.74 / snopes 0.77; worst = **afp 1.63 (Spearman −0.15)**
  — AFP is miscaption-heavy, concentrating the Phase-1b provenance problem.
- **By ceiling:** `claim_date` MAE 1.04 (n=167) vs `review_date` fallback 0.80 (n=152) — the strict
  live cutoff is harder, as expected.
- **By language:** en 0.90 (n=295), fr 1.17 (n=23).
- **By image:** has_image 0.84 (n=77) vs no-image 1.00 — images don't hurt (serialization helps).

## Caveats for the meeting
1. **Gold not yet dev-audited.** WS2 cross-model audit covered test only (hash collision); dev is 0%
   audited. Gold = publisher rating (defensible by construction), but the confident-TRUE read shows
   real gold/axis mismatch — the headline is preliminary pending the audit.
2. **Deterministic floor under-carves video** — untagged (page_meta) video posts stay in the headline.
3. **Untuned prompts** — WS5 (mixed/v3 handling, the dim ceiling, negation) not yet done.
