# fc-gold — version lineage (how each version was made, and how to remake it)

**Current version: `fc_gold_v3.parquet` (47,111 rows, 30 cols, built 2026-07-27).**

This is the reproduction ledger for the fc-gold line: every version, the script that
produced it, its input, what changed, and what was deliberately NOT done. Machine-readable
provenance for the current version sits beside it in `fc_gold_v3_provenance.json`
(input sha256s, FCI dump sha256, per-status counts, git commit).

Sibling ledgers, same purpose: `gate_dataset_provenance.md` (Stage-1 gate),
`truthodds_calval_provenance.json` (CAL/VAL splits).

---

## The chain

```
GFC Fact Check Tools API
  └─ harvest_full.py ─────────────► fc_harvest_full.parquet     83,038 × 10   2026-07-23
       └─ harmonize_full.py ──────► fc_gold_v2.parquet          83,038 × 16   2026-07-23
            └─ build_fc_gold_selection.py ► fc_gold_v2_admitted 47,111 × 16   2026-07-23
                 └─ fc_claim_cluster.py ──► fc_gold_v2_clustered 47,111 × 20  2026-07-24
                      └─ enrich_fc_gold_fci.py ► fc_gold_v3     47,111 × 30   2026-07-27
                                                 ▲
                    Fact Check Insights dump ────┘  (Duke Reporters' Lab, snapshot 2025-09-08)
```

Every step is deterministic and re-runnable in order, from `src/`, as
`uv run python -m eval.scripts.build_eval.<script>`. Row counts above are the acceptance
test: a re-run that produces different counts means an input moved.

| version | rows | sha256 (first 16) | built |
|---|---|---|---|
| `fc_harvest_full.parquet` | 83,038 | `721ca3adc2e83db1` | 2026-07-23 |
| `fc_gold_v2.parquet` | 83,038 | `d9baf9062841e6cd` | 2026-07-23 |
| `fc_gold_v2_admitted.parquet` | 47,111 | `e41b39f4b220e50c` | 2026-07-23 |
| `fc_gold_v2_clustered.parquet` | 47,111 | `abdbdc8a699c32db` | 2026-07-24 |
| `fc_gold_v3.parquet` | 47,111 | see `fc_gold_v3_provenance.json` | 2026-07-27 |

Predecessors **not** in this line: `eval_v1.parquet` and `factcheck_textonly.parquet` are
the older, separately-built eval sets (rolling GFC window + per-publisher scrapers). They
are NOT ancestors of fc-gold; reconciling them is still an open TASKS item.

---

## v2 — exhaustive harvest + 1–5 harmonization (2026-07-23)

**`harvest_full.py` → `fc_harvest_full.parquet` (83,038).** Full per-publisher history from
the GFC Fact Check Tools API (no `maxAgeDays`, paginate to exhaustion), EN + FR, over the
curated 8 + the internationals the verify loop already trusts. 38 publisher sites. Review and
claim dates kept intact — harvest depth is the temporal axis for Truth Odds fitting.
Resumable via `fc_harvest_full_state.json`.

**`harmonize_full.py` → `fc_gold_v2.parquet` (83,038 × 16).** Rule-based mapping of
`original_rating` → `veracity` 1–5 + `rating_subtype`, with `nee` (unprovable = evidence
state, not a veracity point) and `contested` (veracity 3 + mixed) tagged separately.
Resolved 73,449; **veracity null on 9,589**.
- **The LLM tail was SKIPPED (Daniel's call): zero LLM spend.** Unresolved rows keep
  `veracity = null` and are excluded from any gold slice rather than guessed at.
- QC artifacts: `fc_gold_v2_mapping_audit.csv` (every distinct (publisher, rating) pair and
  what the rules did with it — exhaustive, not a sample) and `fc_gold_v2_spotcheck.csv`
  (12 random rows per subtype, checked in context of the claim).

**`build_fc_gold_selection.py` → `fc_gold_v2_admitted.parquet` (47,111 × 16).** Daniel's
publisher ruling: the 8 clean-admit publishers from the reliability audits
(`publisher_audits.md`, page `fc_gold_v2_audit.html`) + politifact + snopes admitted with
caveats. Excluded by ruling: leadstories, factly, vishvasnews, altnews, rumorscanner,
defacto-observatoire (aggregator). Publishers under 100 in-window rows were never audited
and stay out until they are. Window: `claim_date >= 2020 OR review_date >= 2020`.
Flags are kept, not applied — media-authenticity and satire exclusions are per-analysis.

**`fc_claim_cluster.py` → `fc_gold_v2_clustered.parquet` (47,111 × 20).** Adds `cluster_id`,
`cluster_size`, `cluster_pubs`, `q_flags` — cross-publisher claim clustering so CAL/VAL
splits can be made cluster-disjoint. Consumed by `build_calval_splits.py`.

---

## v3 — Fact Check Insights enrichment (2026-07-27)

**`enrich_fc_gold_fci.py` → `fc_gold_v3.parquet` (47,111 × 30).** Adds fields the GFC API
never returns, from the Duke Reporters' Lab ClaimReview dump. **Row count is unchanged and
no existing value was overwritten** — v3 is v2-clustered plus columns, plus null-fills.

### The enrichment source
`fact_check_insights.json`, 313 MB, `meta.retrievedAt = 2025-09-08`, sha256 recorded in
`fci/fci_source.json` and in the v3 provenance. 257,877 claimReviews + 2,986 mediaReviews.
Flattened once to `fci/fci_claimreviews.parquet` + `fci/fci_mediareviews.parquet`
(regenerable: `--flatten --fci-json <path>`).

Note the dump is a **historical breadth** source, not a recency one: it holds zero 2026 rows
and its volume collapses after 2023 (2023: 56,095 → 2024: 14,603 → 2025: 4,364), while our
own harvest runs to 2026. It can only ever be merged with fc-gold, never substituted for it.

### The match key — and why it is not the URL
**Key = (normalised review_url, normalised claim_text).** A shared review URL does NOT mean
a shared claim: both corpora carry multi-claim articles (debate wrap-ups, speech checks), and
at those URLs the two sides legitimately hold *different* claims from the same page. Rows
whose claim text disagrees at a shared URL are **rejected, not guessed at**.

URL normalisation (`NORM_VERSION = url-v2/text-v1`) keeps the query string minus tracking
params. Stripping the whole query collapses publishers that carry the article id in the
query — it merged 330 distinct globes.co.il reviews onto one key.

| status | rows | in gold? |
|---|---|---|
| `exact` — claim identical, single-claim URL | 21,732 | accepted |
| `exact_multiclaim` — claim identical, multi-claim URL | 2,524 | accepted |
| `near` — jaccard ≥ .9 | 12 | accepted |
| `claim_mismatch` — claim at that URL disagrees | 286 | rejected, no payload written |
| `ambiguous` — multi-claim URL, no confident claim | 2 | rejected, no payload written |
| `no_url_match` — not in the FCI dump | 22,555 | no payload |

**24,268 accepted = 98.8% of URL-matched rows.**

### Validation (empirical, four independent checks)
1. **Claim-text agreement** — 98.8% exact on gold (99.1% on the full 83k harvest).
2. **Negative control** — shuffling the FCI side within publisher collapses exact agreement
   to **0.00–0.02%** (politifact 98.34→0.02, snopes 99.00→0.01, factcheck.org 94.28→0.00,
   factly 98.53→0.02). The agreement is not boilerplate similarity.
3. **The residual disagreements are explained, not noise** — 384/384 of the disagreeing URLs
   carry ≥2 distinct claims across the two corpora, versus **0/20,000** of exact-matching
   URLs. Perfect separation, so rejecting them is correct behaviour.
4. **Fields outside the key agree anyway** — rating string identical on 98.4%, claim_date
   identical on 99.7% (99.9% within one day). Not achievable by pairing wrong rows.

Hand-audit page: `fci_enrichment_audit.html`, regenerate with
`build_fci_join_audit.py`. Narrative: `clog/270726.md`.

### What v3 adds
| column | rows populated | note |
|---|---|---|
| `post_urls` | 6,521 | the ORIGINAL post(s) that made the claim |
| `post_urls_social` | 4,416 | FB 2,770 · YouTube 827 · Twitter/X 660 · IG 501 · TikTok 123 · Threads 55 (7,257 URLs total, 688 of them archive proxies) |
| `n_post_urls` / `n_post_urls_social` | all | 0 where none |
| `fc_rationale` | 4,920 | **EVAL-ONLY — never feed to the verifier** (states verdict + reasoning) |
| `fci_match_status` / `fci_match_jaccard` | all | provenance of the match |
| `fci_id` | 24,268 | the FCI record uuid — traces any addition to one source record |
| `fci_rating` | 24,268 | their raw label, for audit |
| `claim_date_source` | all | `gfc` \| `fci` \| `""` |

Null-fills (never overwrites, asserted at build time):
- **`claim_date` +1,143** (1,141 of them Full Fact, which GFC leaves blank). Those rows
  previously fell back to `review_date` as the date ceiling — a ceiling that is median 6 days
  / p90 24 days too generous, and >30 days on 78 rows.
- **`veracity` +204**, harmonised from the FCI rating through the *same*
  `harmonise_veracity` rules v2 used, tagged `harmonisation_source = "fci_rule"`.
  Null veracity 3,729 → 3,525.

### Deliberately not taken (and why)
- **`reviewRating.ratingValue`.** Direction is per-publisher and unreliable: PolitiFact
  declares worst=0/best=9 but emits 0="True", 4="False", 5="Pants on Fire" (a list index);
  dpa declares worst=5/best=1; 15,654 rows declare worst=2/best=2. Every row carrying one
  already has a harmonised veracity, so it adds risk and no information.
- **Same-site claim-text-only fallback** for the 22,555 unmatched rows. Measured: recovers
  1,362 rows (6.0%) and only 79 post URLs, against a real collision risk (snopes re-checks
  the same claim under two slugs; AFP has percent-encoded duplicate doc.afp.com URLs).
- **The 202,476 FCI rows absent from our harvest.** A v3 *expansion* is a separate decision
  with real harmonisation cost — 113k of those rows sit at free-text-verdict publishers
  (Full Fact puts whole sentences in `alternateName`), and the mass is non-EN/FR.
- **`fci_mediareviews.parquet` (2,986 rows)** is flattened and parked, not joined. It is a
  standalone image/video authenticity eval (Phase 1b), not gold enrichment.

### Leakage guards (both are load-bearing)
- **Self-referential post URLs dropped: 28.** An appearance URL on a fact-checker's own
  domain is not "the original post". Matched by domain *suffix*, since the fact-checkers host
  archived copies on `cdn.factcheck.org` / `static.politifact.com`. The curated blocklist is
  `TRUSTED_FACTCHECKERS` + our gold publisher sites — **never derived from the dump's own
  `author.url`**, which lists `facebook.com` as a publisher and would blocklist every genuine
  Facebook post URL (2,353 of them, most of the payload).
- **Post URLs are stripped of embed/tracking params.** Twitter's oEmbed `ref_url` carries the
  fact-check's own URL, so an uncleaned post URL leaks the verdict source. Post-identifying
  params (`story_fbid`, `fbid`, `v`, …) are kept.
- Known residual: 1 appearance URL is itself a Reuters fact-check article. Left in rather than
  build a path heuristic on n=1.

### Caveat not yet closed
Post-URL **liveness is unmeasured**. A naive HEAD sweep would mislead — Facebook and X return
login walls whether or not the post exists. Measuring it honestly needs the archive-proxy
subset (688 URLs, directly checkable) or a hand-check batch. Until then, treat `post_urls` as
a pointer, not as hydrated content.

---

## Reproducing from scratch

```bash
cd src
uv run python -m eval.scripts.build_eval.harvest_full             # GATED: paid GFC pull
uv run python -m eval.scripts.build_eval.harmonize_full           # rules only; --llm is gated
uv run python -m eval.scripts.build_eval.build_fc_gold_selection
uv run python -m eval.scripts.build_eval.fc_claim_cluster
uv run python -m eval.scripts.build_eval.enrich_fc_gold_fci --flatten \
    --fci-json ~/Downloads/fact_check_insights.json               # once; then omit --flatten
uv run python -m eval.scripts.build_eval.build_fci_join_audit     # the audit page
```

Steps 1–4 reproduce v2 exactly (deterministic given the same GFC responses; step 1 is the
only network-dependent one). Step 5 is fully deterministic given the dump — verify you have
the same dump via the sha256 in `fci/fci_source.json`. The 313 MB raw JSON is not tracked;
`fci/*.parquet` are regenerable from it.
