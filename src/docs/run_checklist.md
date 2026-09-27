# Pre-run checklist (mandatory before ANY paid or long run)

Born 2026-07-25 from the E1 incident cluster. Every item traces to a real failure.
Walk it top to bottom; log "checklist ✓" in the clog with the run's ledger row.

## Projection (Daniel's rule: no run without both numbers, written down first)
- [ ] **Cost estimate + ETA written to `eval/data/run_ledger.md` BEFORE launch**,
      derived from measured unit costs (ledger history / run_estimator.py), not vibes.
- [ ] Input DISTRIBUTION measured, not assumed (doc lengths, results/claim, tail
      shape) — the ×4 doc-blowup and Tavily-shaped-smoke errors both came from here.
- [ ] Budget meter armed (hard cap + projection-abort) where the runner supports it.

## Provider hygiene
- [ ] Every external provider PINNED explicitly in code (no env-default inheritance —
      the SEARCH_PROVIDER=tavily fossil).
- [ ] Rate gates for every provider with a budget/window (serper token-bucket;
      politeness pacing for free services), and third-party quotas checked (credits
      remaining, window size) BEFORE launch.
- [ ] No two search-phase jobs concurrently (shared window = 429 cascade).
- [ ] **Serper: `num` above 10 is SILENTLY IGNORED on our plan** (measured 2026-08-28,
      oracle probe): a `num=100` request returns 10 organic results and bills as one
      credit. So a retrieve-k above 10 buys nothing, and any "depth" condition built on
      `num` is a top-10 repeat — that mistake wasted ~240 credits. Depth needs the
      `page` parameter, at ONE BILLED REQUEST PER PAGE. And `search()`'s cache key does
      NOT include `page`, so a paginated call OVERWRITES the cached production page-1
      entry — bypass `search()` or key it before paginating.
- [ ] No silent provider/engine fallbacks (cascades change the sampled distribution).

## Instrument integrity
- [ ] **Prompt CURRENCY verified before launch, not after** (Daniel 2026-08-04, E2):
      when a run reuses a chain whose yields/costs were measured on an earlier build,
      byte-compare its prompts against the rev that produced those numbers
      (`git show <rev>:<file>` + hash each triple-quoted block) AND check no newer
      version exists unwired. "Stamped per record" is provenance after the fact; it
      does not tell you the prompt was the right one. The E2 extraction happened to
      be identical to the dev500 rev, but that was luck — nobody had checked.
- [ ] All versions stamped per record: prompts (query/read), cleaner, prep, doc cap,
      model, provider.
- [ ] No silent fallbacks that change distributions (raw-claim query substitution
      class): fail → retry → SKIP-and-rerun, never substitute.
- [ ] Failed items are NEVER written as done (skip-not-poison); purge policy known.
- [ ] Instrument change mid-run = STOP, archive, restart uniform (never mix versions
      within one measurement).

## Mechanics
- [ ] Parallel from v1 (worker pool; locks only around shared state) — see memory.
- [ ] Resume semantics verified: seen-filter + deterministic ordering
      (maintain_order; seeded shuffles) — the 6.3k-row scope-creep bug.
- [ ] Launch WITHOUT piping through tail/head (progress must stream to the task log).
- [ ] **NEVER launch a paid run via `nohup` or any detached process** (2026-08-28
      double-runner incident, ~12-16 Exa requests double-spent): a backgrounded tool
      call reports COMPLETE while the python process keeps running, so reading the
      meter and relaunching double-spends. Use the tool's own backgrounding, so the
      process stays tracked and killable.
- [ ] After any restart: `ps` check for zombie processes (double-runner incident).
- [ ] Smoke (fresh smoke file — rm the old one) before scale; smoke analyzed, not
      just completed.

## After
- [ ] Ledger row updated with actual cost (usage-sum) + wall; error factor noted.
- [ ] Anomalies clogged; new failure class → new checklist line.
