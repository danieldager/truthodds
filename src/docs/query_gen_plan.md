# Query-gen improvement plan (opened 2026-08-24)

Goal: maximize directional-evidence retrieval per claim (recall on FALSE claims at
fixed FPR) by improving the query step. Product frame: lowest FPR + highest recall
on FALSE; headline convention = full-population, gold-cleaned, 7-flag graded.

## Established facts (2026-08-22 sessions, see clog/220826.md + run_ledger)
- query-v3 defect census (753 silent hard-FALSE): 70% good / 22% too_specific /
  8% drifted. Dominant root cause: CLAIM DATE year copied into the query (66% of
  defects). Artifacts: e1_ctx/query_rootcause.parquet, query_judge_silent_false.parquet.
- v4 prompt (+year-strip guard): judge-good 69.6%→89.0% offline; LIVE paired A/B on
  159 fixed-stratum silent falses: any-directional 3.1%→20.1%, +8 flag crossings,
  v3-requery control proves churn negligible. e1_ctx/query_live_ab/.
- Gold-TRUE live check (150 paired): 8/100 support-bearing trues lost ALL support —
  audited: 6/8 were GENUINE claim-specific evidence; mechanism = date-restricted
  Google is brittle without a year token for real events (two v4 queries returned
  0 docs for heavily-covered events). Trues newly flagged: 1 (artifact-anchored
  gold-axis case, refuted under both arms). e1_ctx/query_live_true/.
- ~80% of silent falses stay silent under a judged-good query = claim poverty;
  query work has a bounded surface.

## Failure strata (each needs a different fix)
(a) zero/few docs returned — over-constrained query / date-restriction brittleness
(b) docs returned, none on-claim — drift, entity swap, wrong locus
(c) docs on-claim, all X/I — claim poverty; NOT addressable by queries

## Phases (each gated on the previous; nothing ships without Phase 3)
- [ ] **Phase 0 — $0 diagnosis.** Stratify every silent/starved claim in
      results-00.jsonl into (a)/(b)/(c) from saved doc counts + reads. Output:
      bucket sizes + worst-150 dev set + permanent regression set (the 8
      lost-support trues). No API calls.
- [ ] **Phase 1 — live bake-off on worst-150** (paired, downstream unchanged):
      - Arm R — retry ladder (policy): v4 first; <2 docs → retry with claim-date
        year appended; docs-but-none-on-claim → retry broadened.
      - Arm T — type-conditioned prompt keyed on existing claim_mode/claim_type
        metadata (quote → distinctive phrase; statistic → number+entity;
        event → entity+event+place).
      - Arm F — fan-out: 2-3 diverse queries (entity-led/event-led/quote-led),
        union top-10s, dedup. Highest ceiling, 2-3x Serper on this subset only.
      Est: ~600 Serper credits + ~$1 Flash.
- [ ] **Phase 2 — read the evidence.** Hand-inspect samples of newly-retrieved
      docs per arm (Baldwin/Gates precedent: "support" can be mode-level).
      Metrics ladder: docs → on-claim docs → directional reads → flag crossings.
- [ ] **Phase 3 — policy + held-out validation.** Pick arm/combination (working
      hypothesis: escalation ladder chassis, type-conditioned first rung, fan-out
      reserved for hard cases); sweep requery trigger; then a ~200-claim
      STRATIFIED HELD-OUT live run (gold T/F × directional/silent, ~50/cell,
      claims unseen in Phase 1). Gates: healthy claims byte-identical behavior;
      trues keep supports, no new true flags beyond noise; false conversion
      transfers. Only then edit evidence_urn_run.py (+ pipeline tracker HTML).

## Standing constraints
- Run ledger + preflight before every live phase; checkpoint at smoke.
- Parallelize optimally (12+ workers, arms interleaved; Serper 1 req/s bucket is
  the only serializer).
- No dataset-derived examples in prompts. Exa excluded. Never label with truth odds.
