# Repo layout — what is live, by feature (2026-07-28)

The repo accumulated four generations of tooling. This file states, per **core feature**,
the ONE chain that is current. If you are about to write a script, check here first: if a
feature already has a source of truth, extend it rather than starting a parallel line.

Everything not listed here is under an `_archive/` directory. Archived ≠ deleted: it is
still in git, `git log --follow` works, and several archived lines are paused rather than
dead (noted below).

---

## The four core features

### 1. Dataset building — three live sources, deliberately not merged

**Fact-check gold (`fc_gold`)** — the labelled claim set behind CAL/VAL:

    build_eval/harvest_full.py         one exhaustive Google Fact Check pull, 38 sites
      -> harmonize_full.py             publisher rating strings -> veracity 1-5
                                       (rules + LLM fallback, eval/harmonize.py)
      -> fc_claim_cluster.py           near-duplicate claims -> clusters
      -> enrich_fc_gold_fci.py         Fact Check Insights join (claim dates, rationales)
         + build_fci_join_audit.py     -> fc_gold_v3
      -> build_fc_gold_selection.py    the selection (+ build_fc_gold_audit_html.py to review it)
      -> claim_screen.py               claim_type / topic / mundane (Qwen3-235B)
      -> build_calval_splits.py        CAL / VAL, cluster-disjoint

**Per-publisher scrapers** — NOT superseded by `harvest_full`; they cover what GFC misses.

    eval/scripts/harvesters/harvest_leadstories.py   sitemap discovery — no GFC at all
    eval/scripts/harvesters/harvest_politifact.py    RSS + GFC
    eval/scripts/harvesters/harvest_snopes.py        GFC discovery + on-page ClaimReview
    eval/scripts/harvesters/harvest_twentyminutes.py, harvest_{aap,afp,factcheckorg,fullfact}.py
    eval/scripts/harvesters/fetch_sources.py         fills the verbatim raw claim
    eval/<publisher>.py                              the per-site parsers + eval/claimreview.py

**Measured 2026-07-28** (`factcheck_textonly.parquet` vs `fc_harvest_full.parquet`,
83,038 rows): **538 review URLs — 12.6% of the old line — are NOT in the exhaustive GFC
harvest**, and 1,234 (29%) are not in `fc_gold_v3`. The gap sits exactly where discovery is
independent of Google: **leadstories.com 851, politifact.com 204, 20minutes.fr 161**, aap 16,
snopes 2. Google Fact Check coverage is genuinely thin for some publishers, and these
scrapers are the only way we reach those claims. They also carry data the GFC API does not
return at all — source URLs, article body, context.

I archived this line on 2026-07-28 as "superseded" and Daniel caught it; restored the same
day. **Do not archive it again without re-running that coverage comparison.** Open loose
end, unchanged: `fc_gold_lineage.md` still carries reconciling `factcheck_textonly.parquet`
into the fc-gold line as an open task — that reconciliation is how these 538 get in.

**Community Notes** — the third, independent claim source:

    build_eval/harvest_community_notes.py -> cn_hydrate.py -> cn_dedup.py -> cn_cluster.py

Shared library: `eval/harvest.py` (API pulls), `eval/harmonize.py` (rating harmonisation).

### 2. Claim extraction — v4.7 chain

    claim_sourcing/harvest_outlet_tweets.py   (validate_handles.py first — handles go stale)
      -> extract_tweet_claims.py              two-pass, text-only, DeepSeek-V4-Flash
      -> normalize_tweet_claims.py
      -> build_verify_input.py

Review/annotation tooling for this chain: `build_full_review.py` + `apply_review_edits.py`,
`build_script_doc.py` + `apply_script_edits.py`, `build_review_doc.py`.

### 3. Verification — v7.5 loop

    pipeline/verify_tweet_claims.py    THE loop (post-level, per-claim budget, code verdict)
      pipeline/pools.py                concurrency: LLM / Serper / Exa / scrape pools
      pipeline/search.py               THE Serper definition + scrape + no_content() junk gate
      pipeline/read_select.py          region selection, junk/off-topic filters
      pipeline/credibility.py          source tiers
      pipeline/harness.py              K-posts-in-flight runner
      pipeline/disk_cache.py           namespaced on-disk cache

Drivers: `claim_sourcing/run_tweet_verify.py` (production runs),
`scripts/run_dev50_verify_review.py` (dev review), `verification_grading/run_averitec_benchmark.py`
(AVeriTeC adapter), `verification_grading/score.py` (`make score`).

**Single source of truth note:** Serper lives in `pipeline/search.py` and ONLY there.
`pools.py::SerperPool` owns concurrency and delegates request shape, parsing, blocklist
and cache key to it. They were two implementations until 2026-07-28 and silently diverged
— do not re-split them.

### 4. Truth Odds — the current program

    build_eval/evidence_urn_run.py       the E1 runner: query -> Serper -> regions -> READ -> urn
    build_eval/cap_sweep.py              does a wider read window buy real evidence
    build_eval/run_estimator.py          cost / ETA projection -> eval/data/run_ledger.md
    claim_sourcing/evidence_profile_run.py   round-1 profiler through the verify loop
    claim_sourcing/build_truthodds_figures.py
    claim_sourcing/compute_scorecard.py, unrated_pool_census.py, build_source_dashboard.py

`evidence_urn_run` and `evidence_profile_run` are NOT duplicates: the first is a standalone
query->read->urn runner over CAL claims, the second profiles what the full verify loop
retrieves. Keep the distinction explicit if either grows.

---

## What was archived, and why

| location | what | status |
|---|---|---|
| `_archive/claim_selection_gate/` | Stage-1 check-worthiness gate, filter/stage builders, review tooling | **PAUSED, not dead** — resume via `docs/claim_selection_status.md`; the live v8 prompt is inside `gate_eval.py` |
| `_archive/feed_image_study/` | image harvest/serialisation/audit, post-media research | **deferred** — IDEA-015 sequences the media tier last |
| `_archive/verdict_dataset/` | verdict dataset + enrichment audit builders | superseded by the fc-gold line |
| `_archive/tier1_tier2_unbuilt/` | `cache.py`, `embedding.py`, `models.py`, `fact_api.py` | **spec, never wired** — the live loop imports none of them. Design still stands: `docs/tier1_cache_design.md` |
| `_archive/verify_v0_3_satellites/` | scripts feeding/scoring the claim-level loop | that loop is in `_archive/verify_v0_3_claimcheck/` |
| `_archive/loop_audits_v3_v4/` | dev50 audits of loop v3/v4 | loop is v7.5 |
| `eval/scripts/_archive/{cache_eval,searxng_probe}/` | completed one-off studies | each has its own RESULTS/FINDINGS |
| `eval/scripts/build_eval/read_ab.py` | READ prompt A/B | arms abandoned — header says so; kept deliberately |

## Rule going forward

One feature, one chain. Before adding a script, find its feature above. If the new script
would do the same job as an existing one for a different dataset or a different era, that
is the signal to extend the existing chain — or to archive the old one in the same commit.
