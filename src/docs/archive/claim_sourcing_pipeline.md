# Claim-sourcing pipeline (survey-experiment claim pool)

Turns real US news outlets into a pool of checkable claims that become **synthetic tweets** for the
misinformation-nudging survey. The 4 experiment groups = outlet political **side** (L/R, from AllSides
editorial bias) × claim **veracity** (true/false, assigned later by Stage-3). This pipeline produces the
claim pool spanning the reliability × bias grid; it is **separate** from the eval-dataset build.

Scripts: `eval/scripts/claim_sourcing/`. Data (all gitignored — X ToS + NewsGuard): under
`eval/data/survey_claims/` and `eval/data/ace_attempts.parquet`. Chronology: `clog/030726.md`.

## Key decision — source articles via RSS, not tweets

We first captured outlet tweets (the `zeerover` browser extension) and ran the Article Claim Extractor
(ACE) on them. **ACE-on-tweets yielded 7%** — tweets are one-line headline teasers with no body for ACE
to assess risk or extract a decontextualized claim. Two more findings killed the tweet→article path as
the spine: **`trib.al` (the SocialFlow link-wrapper NYT/Breitbart route through) is dead** (DNS no longer
resolves), and a large share of outlet tweets link to their own X video, not an article.

**Decision:** harvest articles directly from each outlet's **RSS feed / news sitemap**, run ACE on the
article body (0.29 yield vs 0.07). The tweet capture is retained for the ~5–10% self-contained-claim
tweets + image/framing analysis, but is not the claim spine.

## Steps (run order)

**0. (context) Tweet capture** — `zeerover` extension → NDJSON. Characterized in
`survey_claims/tweet_characterization.md` (+ `originals.csv`). Not the claim spine (see above).

**1. Source accessibility audit** — which outlets can we free-read?
- `audit_source_accessibility.py` — per outlet (from `source_ratings_unified.xlsx` Tab 3 = 64 outlets),
  discover an RSS feed (guessed paths + homepage `<link>`), pull 5 articles, `trafilatura.extract`,
  classify FREE / PAYWALLED by extracted char count. Homepage-scrape fallback for no-feed outlets.
- `finalize_source_accessibility.py` — honest relabel: **trust only RSS-sourced verdicts** (homepage
  fallback grabbed asset/hub URLs → unreliable); builds the free-outlets-per-cell grid.
- A subagent then resolved the 27 "undetermined" outlets by finding their real feeds/sitemaps.
- Output: **`source_accessibility.xlsx`** (64 rows: **47 FREE, 13 PAYWALLED→Jina, 3 js-render, 1 dead**).
  `probe_tweet_link_accessibility.py` is the earlier tweet-link probe that surfaced the dead-`trib.al`
  finding (kept for provenance; superseded by the RSS audit).

**2. Claim harvest** — RSS → article → ACE.
- `harvest_free_batch.py` — the 32 original free outlets × 5 recent articles → ACE →
  `claims_smoke.csv` (97 claims) + appends every attempt to `ace_attempts.parquet`.
- `harvest_extend_newfree.py` — the 13 newly-free outlets × 5, with a per-outlet source spec handling
  **RSS / sitemap (incl. gzipped + index-drill) / filtered scrape** → `claims_newfree.csv` (30 claims).
  Dropped NY Daily News + Post Millennial (no reachable feed; their cells already well-covered).

**2b. Deepen thin cells** — `harvest_deepen_scarce.py` (iterative). The thin cells (M-L/U-L/M-R) have
few free outlets, so the lever is depth: pull more recent articles from each thin-cell outlet's feed,
**skipping URLs already in `ace_attempts.parquet`** → `claims_deep.csv` (+67 raw claims). M-L (Mother
Jones) is feed-capped (~10 entries), so it stays thinnest.

**2c. Bulk deepen + headline pass** — to scale volume: `harvest_bulk_all_free.py` re-walks ALL free
outlets at depth (N=40, skips URLs already in `ace_attempts`) and extracts from BOTH the article body
AND the headline (`extraction_source` tags which); `harvest_headlines.py` extracts headline claims from
every already-fetched article (no re-fetch). Headlines hit ~28%, near the body's ~29% (measured,
`headline_vs_article_test.py`), but skew toward lower-risk routine claims so the filter prunes more of
them. Batches: `claims_bulk.csv`, `claims_headline.csv`.

**3. Filter + dedup** — `filter_and_dedup_claims.py` over all harvest batches.
- **Dedup:** MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`) cosine ≥ 0.80 clusters near-dup / same-event
  claims; keep the longest (most complete) representative.
- **Filter:** LLM checkworthiness judge (DeepSeek, temp 0) — KEEP specific, checkable, self-contained
  factual/**attributed** claims; DROP opinion / vague / superlative-or-prediction / not-self-contained.
  Every claim's decision is persisted (auditable).
- Output: **`claim_pool.csv`** (714 kept, all batches) + **`claim_pool_audit.csv`** (every claim with
  decision + reason).

**4. Shape for Stage-3** — `build_verify_input.py` maps `claim_pool.csv` → the schema `pipeline/verify_text.py`
consumes: `claim_id`, **`raw_context`** (= the claim = the synthetic-tweet text, framing intact — NOT
de-framed), `claim_date` (per-article ceiling: URL-derived or harvest-date fallback), `side`/`reliability`/
`cell` (grouping), + `outlet`/`domain`/`source_url`/`headline`/`risk_reason`. Output:
**`claim_pool_verify.parquet`** (714; `.csv` twin). Handoff to the verify session: `docs/conversation_log.md`
(030726). Run `verify_text(raw_context, block_factcheck=True, date_ceiling=claim_date)` → `veracity 1–5`
= the true/false axis → the 4 groups (`side` × veracity).

## Current numbers

| Stage | Count |
|---|---|
| Accessibility | 47 FREE · 13 PAYWALLED · 3 js-render · 1 dead (of 64) |
| Claims harvested | 955 raw (5 batches: free-batch + newly-free + deepen + headline-pass + bulk) |
| After dedup | 830 (125 clusters merged) |
| **Clean pool** | **714** — R-L 233 · U-R 179 · M-R 106 · R-R 95 · R-C 78 · U-L 15 · M-L 8 |
| Extraction source | body 538 · headline 176 |
| Side balance | L 256 · R 380 · C 78 |
| Filter drops | 116 (vague 53 · opinion 40 · superlative/prediction 18 · not-self-contained 5) |

## Gaps / next

- **M-L still thinnest (8)** and **U-L (15)** — few free mixed/unreliable-left outlets exist and Mother
  Jones' feed is capped (~10). M-C/U-C empty (no such outlets). These left cells are the structural limit.
- **Scale further** — feeds cap at ~25–50 recent entries; for more, go to per-outlet archives/sitemaps.
- **13 PAYWALLED → Jina** (NYT, WaPo, WSJ, Bloomberg, Axios, USA Today, The Nation, Commentary, IBD, New
  Yorker, The Hill, RealClearPolitics, National Review); **3 js-render → headless renderer** (Reuters,
  Newsmax, InfoWars); BuzzFeed News dead.
- **Stage-3 verification** — `claim_pool_verify.parquet` handed off to the verify session (see above);
  `verify_text` FC-blocked assigns veracity 1–5 → the true/false axis. 714 claims (body 538 / headline
  176); veracities provisional until the verify scale is re-validated (the 81-with-post run).
- **Side imbalance** — grid is R-heavy (R 380 · L 256 · C 78; ratio improved from 0.45 to 0.67 after
  the bulk); balance L/R downstream once veracity is assigned (subsample R, or add left-leaning sources).
