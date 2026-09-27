# Adversarial two-phase verification — design brief (IDEA-016)

Daniel's proposal (2026-07-23), refined against the Truth Odds fit and the deep-research
harness internals (Explore agent report, clog/230726). Status: DESIGN — nothing built;
the profiling run below is the next gate.

## The decision procedure

**Phase 0 — structural noise block (search-time).** UGC/socials done (33 domains, UGC-only since 2026-07-27).
Further blocking ONLY by platform type (stock/video licensing, ticketing, listings) —
NOT by outcome stats until the read-all-10 run de-biases them (triage's unrated
aversion made JSTOR/SAGE/NCSL/ap-newsroom zero-yield; blocking on that would launder
the bias into infrastructure).

**Phase 1 — open-armed collection.** One support-seeking query; ALL signals count,
unrated included. Routing:
- Support anchored by a rated/institutional voice (proportionality rule below) →
  **supported**, pass. (Variant B: even these get the adversarial check — the two-loop
  profiling run measures what the shortcut would miss; decide after.)
- Support exists but is unrated-only / single-voice / stances conflict / origin outlet
  is low-NG → **doubt** → Phase 2.
- Refuting signal in round 1 → Phase 2 directly (confirm it's refutation, not contest).

**Phase 2 — adversarial panel (doubt branch only).** Refutation-seeking queries
("<claim core> debunked / fact check / false" — or LLM-pointed at the load-bearing
proposition). 2–3 independent skeptic votes, each prompted to REFUTE (deep-research
pattern), each seeing the support dossier. Outcomes:
- Quorum of clean counter-searches finds nothing → **supported** (silence under a
  refutation-hunting query is evidence FOR truth — see urn model).
- Majority finds refuting EVIDENCE → **refuted** (asymmetric weight already justified:
  fitted LRs −2.0/−2.4 per refuting voice vs +1.5/+1.8 supporting).
- Refute-signals exist but are opinion/partisan-attack corroboration → **contested** →
  AVeriTeC Conflicting class; soft nudge, not the misinfo flag. (Future work: a system
  that actually parses these sources to an intelligent verdict; for now contested ≠
  isolated misinformation is the honest reading.)
- Counter-search starved / infra failure → **open**. NEVER refuted, NEVER supported
  (deep-research's `unverified` quorum rule; our NEE≠Refuted realignment; we have the
  retrieval-health signal `resampled` to detect starvation).

**Deliberate deviation from deep-research:** their voters default to refuted-if-
uncertain — right for research reports, wrong polarity for a nudge tool (flagging
accurate posts is our most expensive error; precision 0.96 is the asset). Our default
on uncertainty is OPEN.

## Signal rules

- **Tier gates SUPPORT only, never refutation** (deep-research rule, = Daniel's
  asymmetry): any credible non-opinion source can refute; support strength must be
  proportional to claim extraordinariness — mundane claims may pass on rated-secondary
  or strong corroborated-unrated support; extraordinary claims need a strong anchor
  (NG≥90 / institutional / fact-checker). Replaces the uniform bar.
- **Opinion rule stays** for evidence; opinion-corroborated refutation feeds
  "contested", not "refuted".
- **Independence / anti-echo** (Daniel: amplification ≠ corroboration). Existing:
  voice-key registrable-base collapse, _PUB_FAMILIES, wire/republication detection,
  origin aliases. GAP: same-wording clusters across unrelated-looking sites
  (amplification networks). Add evidence-level near-dup clustering — same quote/phrasing
  across domains = ONE voice regardless of domain count. Becomes MORE load-bearing once
  unrated signals count.

## Truth Odds integration — query-conditioned urns

Signals are conditioned on (θ, query intent q). The existing p̂'s are the q=support urn.
The q=refute urn is unmeasured: P(∅|T, refute-q) high/stable; P(∅|F, refute-q) depends
on coverage latency + virality (fresh fabrications have no debunk yet). Fitting it is
the profiling run's core deliverable; the empty-slot sign flips between urns, and both
effects are small per-slot (Daniel: "slight in both directions") — decision thresholds
come from the fitted LRs, not intuition.

## The profiling run (next concrete step — GATED, not launched)

Extend `evidence_profile_run.py`: **two loops per post** — loop 1 support-seeking query,
loop 2 adversarial query per targeted claim — and **read ALL 10 results** in both
(no triage picking; kills the selection bias in every fate stat and gives the first
unbiased zero-yield numbers for future blocking decisions).

- Datasets: AVeriTeC 119T/200F + fc-gold 32T/98F (ceiling ON) — refute-urn parameters
  per θ; dev-1000 tweet sample (ceiling OFF) — production routing rates.
- Measure: refute-urn p̂'s; Phase-1 routing shares (clean-pass / doubt / flip);
  shortcut-vs-always-adversarial delta (variant A vs B); opinion/partisan share of
  refute signals; echo-cluster rate among corroborating signals; unbiased per-domain
  yield.
- Cost: read-all-10 ≈ 10 scrapes+summaries × 2 loops × ~900 claims — an order more LLM
  reads than the round-1 profiles. Smoke 25 posts first, cost-estimate, THEN Daniel's
  go for scale (checkpoint rule). Serper only; Exa excluded as always.
- Offline first: Phase-1 routing rates are simulatable from the SAVED round-1 profiles
  (zero API cost) — do this before the run to size the doubt branch.

## Open questions (Daniel)

1. Variant A (rated-anchor shortcut) vs B (adversarial for everything) — decide after
   the run measures what A misses. A is cheaper; B is one uniform code path.
2. Panel size on the doubt branch: 2 votes (cheap) vs 3 (majority semantics).
3. Does "contested" get its own nudge copy, or fold into the existing soft-nudge design
   (verdict_nudge_design.md)?
4. Where does the doubt-trigger list live — code constants (auditable, replayable) is
   the default; anything LLM-judged here reopens the gameability the v4 audit closed.
