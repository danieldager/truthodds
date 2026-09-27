# claimverify build status (2026-09-07)

Tests exist and are green: `uv run pytest claimverify/tests -q` -> 63 passed
(test_pools, test_loop_offline, test_ceiling, test_origin, test_score). `score_averitec.py`
and `README.md` are written. A 10-claim smoke ran and its traces were inspected; three
changes came out of that inspection:

- READ segment ids like "S3" were dropped by an int-only filter, taking the whole evidence
  entry with them. They are coerced now, with a `seg-coerced` guard event; an entry with no
  usable segment still drops but records `evidence-dropped-no-segs`. v7.5 has the same bug.
- `final` was true only on the Exa round, so with Exa off no round was final and the RESOLVE
  final note never went out. A round is final now on the Exa round or, with Exa disabled, on
  the last Serper try. v7.5 has the same bug; Exa-on behaviour is unchanged.
- Each Serper search record carries `post_ceiling_hits` (hits dated after the claim-date
  ceiling) and `undated_hits` (hits whose date does not parse). Counting only, nothing is
  filtered, so the ceiling leak is measurable per run.

## Files done (all import cleanly under `uv run`; `run_averitec --dry-run` works for both arms)
       1 __init__.py
     124 config.py
      69 credibility.py
      56 disk_cache.py
     215 harness.py
     865 loop.py
      22 origin.py
     739 pools.py
      77 prompts.py
     130 read_select.py
     389 retrieval.py
     130 run_averitec.py
     172 score_averitec.py
      83 trace.py
    3072 total

- `config.py` — ClaimVerifyConfig (v7.5 constants + arm switches) and OrchestrationConfig
  (llm 150 / serper 16 / exa 3 / scrape 64x2 / K 100); PRICE_PER_M fallback prices.
- `disk_cache.py` — configurable root (default src/pipeline/.cache), new namespaces llm, serper_raw, exa_raw.
- `origin.py` — origin_domain(url) with web.archive.org unwrap (same logic as averitec_urn_run.origin_site;
  NOT yet imported back from that script).
- `credibility.py`, `read_select.py` — trimmed copies (is_off_topic stub removed).
- `retrieval.py` — tbs (M/D/YYYY unpadded), cache_key (identical to pipeline/search.py), serper
  payload/parse/finalize/merge, scrape_detail() reporting source/reason/status.
- `pools.py` — trimmed pools + CallRecord tracing (messages, raw output, usage, latency, attempts,
  cache_hit), exact-match LLM cache, raw Serper/Exa JSON cached + traced, scrape records, serper credits counter.
- `trace.py` — Trace (llm_calls/searches/scrapes/reads/resolves/ledger/notes) + content-addressed PageStore.
- `prompts.py` — six prompts rewritten for one claim (claim_id kept = 1); pair/post/outlet/republication
  language removed, all evidence/stance/date/bar rules verbatim.
- `loop.py` — verify_claim(): full port of verify_post minus CONTEXT, pairs, origin-token republication,
  nudge verdict. Returns status, verdict_raw, verdict_cc, rounds (full results[]), evidence, guards,
  counts, tokens, cost_usd, elapsed_s. `snippet_evidence` = "fallback" (v7.5 triage-off behaviour) | "all".
- `harness.py` — async K-in-flight runner, shard-XX.jsonl resume, trace dump per claim, progress every
  10 claims + every 60 s, stall alarm, budget abort, manifest.json + config.json.
- `run_averitec.py` — CLI (--arm top3|all10, --smoke, --k, --exa, --budget, --out, --dry-run).
- `score_averitec.py` — 4-class raw + ClaimCheck convention over the 500 gold rows, macro-F1,
  confusion, binary variants A/B/C, per-claim parquet.
- `tests/` — test_pools, test_loop_offline, test_ceiling, test_origin, test_score.

## Not started (none; REPO_MAP line and the averitec_urn_run import were done by the scorer agent)
- REPO_MAP.md line; making `eval/scripts/build_eval/averitec_urn_run.py` import `claimverify.origin`.

## Known gaps vs the spec
- `is_republication` (wire byline / syndication-host / reprint phrase) was DROPPED: every branch depends
  on origin tokens or the origin being a wire agency, neither of which exists for an AVeriTeC claim.
  READ's `republication` flag removed from the prompt; `_qualifying` still honours the field (always False).
- Cost: DeepInfra `estimated_cost` is read from the SDK usage object's model_extra when present; otherwise
  PRICE_PER_M (0.20 / 0.02 / 0.80 USD per M, ledger-derived list-price ceiling, unverified) — the record's
  usage carries `cost_source` so the two are distinguishable.
- Serper "top-3" is the first 3 of our credibility-ranked list of the 10 Google hits, not Google's top 3.
- `llm_cache` is on by default: a re-run replays cached responses (reproducible, $0) — pass a cold
  `disk_cache.set_dir` or CLAUDE_PIPELINE_CACHE_BYPASS=1 for a fresh measurement of latency.

## Next steps, in order
1. Refactor averitec_urn_run.origin_site -> claimverify.origin.origin_domain; REPO_MAP line.
2. Full arms with --budget (the 10-claim smoke is done).
