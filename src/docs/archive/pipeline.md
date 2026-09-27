# Pipeline v0.2 — Spec

**Version:** 0.2 (2026-05-18, replaces v0.1)
**Status:** Design landed; implementation in `src/pipeline/`.
**Source of truth:** this file + `CLAUDE.md` (high-level overview). When they disagree, this file wins for *implementation detail*; CLAUDE.md wins for *architectural intent*.

---

## Goals

1. **Tiered, escalating cost.** Each tier resolves to a verdict if it can; only cache- and API-misses pay the LLM/search cost of Tier 3.
2. **Cache hot.** Every Tier 2 and Tier 3 verdict goes into a Neon (Postgres + pgvector) claims DB so future near-matches resolve at Tier 1.
3. **Tier 3 = ClaimCheck loop.** One LLM, one query per round, synthesise-or-query decision, hard-capped at 4 rounds.
4. **Clean and compact.** Whole pipeline package should fit in ~8 readable Python files.

Non-goals for v0.2 (deferred): misleadingness detection, calibrated post-level aggregation, French-language coverage, deployment plumbing.

---

## Tier flow

```
post
 │
 ▼
[Tier 0] encoder ensemble  ──▶  risk score (advisory; nudge shortcut, no verdict)
 │
 ▼
[extract]  ──▶  list[AtomicClaim]   (text + embedding)
 │
 ▼  for each claim, in order:
 │
 ├─ [Tier 1] vector-DB lookup     ──hit──▶  cached verdict (subject to recheck policy)
 │
 ├─ [Tier 2] Google FCT API       ──hit──▶  harmonised verdict  ──▶ write to DB
 │
 └─ [Tier 3] ClaimCheck loop      ─────────▶  verdict + Likert    ──▶ write to DB
```

Per-post output: `list[ClaimVerdict]` (one per atomic claim). Post-level aggregation is downstream and out of scope for v0.2.

---

## Module map (`src/pipeline/`)

| File | Responsibility | Calls out to |
|---|---|---|
| `__init__.py` | Public exports. | — |
| `config.py` | All pipeline tunables (model names, thresholds, caps). Re-exports shared constants from `src/config.py`. | `src/config.py` |
| `models.py` | Pydantic data models: `AtomicClaim`, `EvidenceDoc`, `Synthesis`, `ClaimVerdict`, `PipelineResult`. | — |
| `extract.py` | Post → list[AtomicClaim] (LLM call). Includes embedding. | OpenAI-compat client, sentence-transformers |
| `cache.py` | Tier 1: cosine-similarity lookup + verdict write to Neon pgvector. Implements recheck policy. | psycopg + pgvector |
| `fact_api.py` | Tier 2: FCT API query + harmonisation map (lifted from `eval/harmonize.py`). | requests, `eval/harmonize.py` |
| `search.py` | Web search (self-hosted SearXNG) + per-URL scrape (Trafilatura). Shared util for Tier 3. | requests, trafilatura |
| `verify.py` | Tier 3 ClaimCheck loop: plan → search → summarise → synthesise (decide verdict or next query). | search.py, OpenAI-compat client |
| `pipeline.py` | Orchestrator: glue Tier 0 + extract + Tier 1 + Tier 2 + Tier 3 in order. | all of the above |

A thin runner (e.g. `src/cli.py`) drives the orchestrator from the command line for ad-hoc claims and eval batches.

---

## Tier 0 — Encoder ensemble pre-filter

**Status in v0.2:** advisory only; does not gate the rest of the pipeline.
**Input:** raw post text.
**Output:** `float ∈ [0,1]` risk score + per-classifier breakdown.
**Models:** as listed in CLAUDE.md (fallacy, bias, clickbait, propaganda, rule-based).
**Use:** drives the "early nudge" UX in the extension; treated as a separate concern from verdict resolution. Implementation lives outside `src/pipeline/` (probably in the extension/backend layer once that exists).

---

## Claim extraction

**Input:** raw post text.
**Output:** `list[AtomicClaim]` where each claim has `text: str` and `embedding: list[float]`.
**Model:** Qwen3-32B via Groq (Extraction model in `config.py`).
**Embedding:** `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers).
**Behaviour:**
- Single LLM call decomposes the post into atomic, independently-checkable claims (existing prompt is fine; lift from v0.1).
- For each extracted claim, compute embedding once. The (text, embedding) pair is the cache key for the rest of the pipeline.

Always runs (no `is_checkable` early-exit at this stage — that role is folded into the LLM returning `[]` when the post has no factual content).

---

## Tier 1 — Vector-DB lookup (Neon + pgvector)

**Input:** `AtomicClaim`.
**Output:** `ClaimVerdict | None` (None = miss → fall through to Tier 2).

**Lookup:**
- ANN search via pgvector on the `claim_embedding` column.
- Match if top-1 cosine similarity ≥ `SIMILARITY_THRESHOLD` (config; start at 0.92, tune later).
- Apply recheck policy: if `match.created_at < now - RECHECK_AFTER_DAYS` → treat as miss, force re-verification through Tiers 2/3. Default `RECHECK_AFTER_DAYS = 30` (configurable).

**Schema (one table, `claim_verdicts`):**

| column | type | notes |
|---|---|---|
| `id` | uuid | pk |
| `claim_text` | text | normalised atomic claim |
| `claim_embedding` | vector(384) | MiniLM dimension |
| `verdict` | text | one of `Supported / Refuted / Not Enough Evidence / Conflicting Evidence` |
| `likert_supported` | smallint | 1–5 (NULL for Tier 2 verdicts) |
| `likert_refuted` | smallint | 1–5 |
| `likert_nei` | smallint | 1–5 |
| `likert_ce` | smallint | 1–5 |
| `tier_resolved` | smallint | 2 or 3 |
| `cap_hit` | bool | true if Tier 3 force-committed at round 4 |
| `evidence_urls` | jsonb | list of URLs that drove the verdict (Tier 3 only) |
| `source_publisher` | text | Tier 2 only: publisher whose rating was used |
| `created_at` | timestamptz | server default `now()` |

Indexes: `USING ivfflat (claim_embedding vector_cosine_ops)` on the embedding column.

**Writes:** Tier 2 and Tier 3 successes both write a row. Tier 3 force-verdicts write with `cap_hit=true` and downstream consumers can decide whether to trust them based on the Likert margin.

---

## Tier 2 — Google Fact Check Tools API

**Input:** `AtomicClaim`.
**Output:** `ClaimVerdict | None`.

**Behaviour:**
1. Call FCT API with `claim.text`.
2. Filter results to trusted publishers (same set used to build `eval_v1`: Snopes, AFP, Newschecker, Verafiles, RumorScanner, PolitiFact, FactCheck.org, Full Fact).
3. If ≥1 trusted result: take the most recent, harmonise its `textualRating` → 4-class verdict via `eval/harmonize.py` rule table (LLM fallback if publisher not in rules).
4. Return `ClaimVerdict(verdict=..., tier_resolved=2, source_publisher=...)` and write to DB.
5. No trusted result → return None.

**Reused code:** lift `RULES`, `rule_map`, and `llm_map` directly from `eval/harmonize.py`. Do not duplicate them; import.

---

## Tier 3 — Full verification (ClaimCheck loop)

**Input:** `AtomicClaim`.
**Output:** `ClaimVerdict` (always — force-commits at round cap).

**One LLM throughout.** Same model handles planning, summarisation, and synthesis-or-query (different prompts).

### Loop

```
state = {
  claim: str,
  evidence_pool: list[EvidenceDoc],   # accumulates across rounds
  past_queries: list[str],
  round: int = 0,
}

loop while round < MAX_ROUNDS (=4):
  round += 1

  # PLAN
  query = plan_next_query(claim, evidence_pool, past_queries)
  past_queries.append(query)

  # EXECUTE
  results = search(query, top_k=3)   # SearXNG JSON API

  # SUMMARISE (per doc)
  for url in urls:
    text = scrape(url)
    if text is None: continue
    summary = summarise_for_claim(claim, text, url, date)   # LLM
    if summary.relevant:
      evidence_pool.append(EvidenceDoc(url, date, summary, ...))

  # SYNTHESISE + DECIDE
  result = synthesise_or_query(claim, evidence_pool)
  if result.verdict is not None:
    return ClaimVerdict(verdict=result.verdict, likert=result.likert,
                        cap_hit=False, evidence_urls=[d.url for d in evidence_pool])

# Cap hit — force a verdict from current pool
result = synthesise_or_query(claim, evidence_pool, force_verdict=True)
return ClaimVerdict(verdict=result.verdict, likert=result.likert,
                    cap_hit=True, evidence_urls=[d.url for d in evidence_pool])
```

### Per-step LLM calls

| Step | Prompt purpose | Returns |
|---|---|---|
| `plan_next_query` | Given the claim, current evidence, and past queries, produce ONE search query that targets what's still uncertain. On round 1, evidence is empty and the model produces a broad opening query. | `str` (single query) |
| `summarise_for_claim` | Given a scraped article, extract claim-relevant facts with explicit date anchoring. Discard articles that don't address the claim. | `EvidenceDoc` (url, date, summary, relevant: bool) |
| `synthesise_or_query` | Given claim + all evidence summaries, produce either (a) verdict + Likert per-label, or (b) one more search query. `force_verdict=True` overrides (b) and demands (a). | discriminated union: `Verdict \| NextQuery` |

### Termination criteria (all explicit)

| Condition | Outcome |
|---|---|
| `synthesise_or_query` returns a verdict | Exit loop, return verdict (`cap_hit=False`). |
| Round counter reaches `MAX_ROUNDS` (=4) without verdict | One final `synthesise_or_query(force_verdict=True)` call. Return verdict with `cap_hit=True`. |
| `synthesise_or_query` returns a query with cosine sim ≥ `REDUNDANCY_THRESHOLD` (=0.9) against any past query | Force verdict on the spot. |

**No NEI default.** Even at cap, the synthesiser is forced to commit to one of the four labels (Likert per-label still emitted; verdict = argmax). Background: NEI is already over-predicted at ~28% of pipeline outputs vs 1.1% of gold labels in `eval_v1`. Defaulting to NEI on cap-hit makes that worse.

### Config (defaults; live in `pipeline/config.py`)

| Name | Default | Notes |
|---|---|---|
| `MAX_ROUNDS` | 4 | Hard cap on Tier 3 loop iterations. |
| `SEARCH_TOP_K` | 3 | URLs per query. Matches ClaimCheck. |
| `REDUNDANCY_THRESHOLD` | 0.9 | Cosine sim above which a follow-up is considered redundant. |
| `SIMILARITY_THRESHOLD` | 0.92 | Tier 1 cache-hit cosine threshold. |
| `RECHECK_AFTER_DAYS` | 30 | Tier 1 cache entries older than this trigger re-verification. |
| `SCRAPE_TIMEOUT` | 10 | seconds per URL |
| `DATE_CEILING_MODE` | `today` | Verifier post-filters docs whose extracted publication date exceeds this; eval runs override to claim-date. |

---

## Data models (`pipeline/models.py`)

Pydantic. Kept deliberately flat — no nested helper types unless something concrete needs them.

```python
class AtomicClaim(BaseModel):
    text: str
    embedding: list[float]

class EvidenceDoc(BaseModel):
    url: str
    publication_date: str | None      # ISO date if extractable
    summary: str                       # claim-relevant excerpt with date anchoring
    relevant: bool                     # False → discarded by summarise step

class Likert(BaseModel):
    supported: int                     # 1–5
    refuted: int
    nei: int
    ce: int

class ClaimVerdict(BaseModel):
    claim: AtomicClaim
    verdict: Literal["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence"]
    likert: Likert
    tier_resolved: Literal[1, 2, 3]
    cap_hit: bool = False
    evidence_urls: list[str] = []
    source_publisher: str | None = None
    justification: str = ""

class PipelineResult(BaseModel):
    post: str
    risk_score: float | None           # Tier 0 output (advisory)
    verdicts: list[ClaimVerdict]       # one per atomic claim
```

---

## Evaluation hooks

- Eval runs disable Tier 1 (force fresh) and may disable Tier 2 (to measure pure Tier 3) via flags on the orchestrator.
- Tier 3 must be runnable on a single claim string for per-claim eval (AVeriTeC dev, `eval_v1.parquet`).
- The orchestrator returns enough metadata (`tier_resolved`, `cap_hit`, `likert`, `evidence_urls`) for the eval scripts to compute macro-F1 and ECE without re-running the pipeline.

---

## What changed from v0.1

Brief, for orientation; full audit in `clog/180526.md`.

- **Architecture flattened.** Old Tier 3 had 6 steps (question gen → hybrid retrieval → evidence prep → FIRE iterative → misleadingness → calibrated aggregation), most of them partially or not implemented. New Tier 3 has one loop, four prompts, hard cap.
- **Tiers re-numbered.** Encoder ensemble → Tier 0 (was Tier 1). Vector-DB lookup → its own tier (Tier 1, was buried inside Tier 2). FCT API → Tier 2 (was buried inside Tier 3's `retrieve_evidence`).
- **Dropped:** multi-hop verifying questions, intra-doc BM25 chunking, cross-encoder rerank, MMR diversity, FIRE per-verifier follow-ups, hard-priority aggregation, calibration-via-self-consistency.
- **Parked:** misleadingness, calibrated post-level aggregation.
- **Added:** per-doc LLM summarisation with date anchoring (was specced as IDEA-007 but never built), Neon claims DB (was specced as IDEA-003 but never built), recheck policy.
- **Reused unchanged:** claim-extraction prompt, `eval/harmonize.py` (now used in Tier 2), Trafilatura scrape utility. Search backend swapped Serper → SearXNG (2026-05-19, under test — see CLAUDE.md "Novel claim search" decision row and IDEA-012).
