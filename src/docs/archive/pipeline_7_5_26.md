# Live Fact-Checking Pipeline — PoC Design
*Created: 2026-05-07*

> **⚠️ HISTORICAL SNAPSHOT (2026-05-07) — superseded by `CLAUDE.md` § Fact-Checking Pipeline.**
> Kept as a point-in-time record of mid-May pipeline design. Pre-dates the Tier 3 ClaimCheck/FIRE/HerO2 evolution and the calibrated weighted verdict aggregation decision. Do not treat as current.

## Overview

A proof-of-concept fact-checking pipeline for social media posts (Twitter/X, Facebook) that works in the wild using live web retrieval. No pre-built knowledge base assumption (as in FEVER benchmark systems). Targets real-time veracity scoring for the empirical arm of the Guriev et al. nudging study.

**Inspired by:** ClaimCheck (live web search architecture), FIRE (adaptive retrieval stopping criterion), CTU AIC (structured LLM evidence output + Likert confidence), HerO2 (model sizing + pipeline staging).

**Languages:** English + French (embedding model supports both from the start).

**Evaluation:** LiveFact claim corpus — run pipeline on their claims, compare verdicts to ground truth labels. Monthly updates allow longitudinal testing.

---

## Pipeline

```
Social media post
  │
  ▼
[1] Claim detection + decomposition + query generation
    Single LLM call → structured JSON
  │
  ├─ No checkable claims → exit (no nudge)
  │
  ▼
[2] Claims DB lookup (vector similarity)
    paraphrase-multilingual-MiniLM-L12-v2 → ChromaDB
  │
  ├─ Cache hit (similarity > threshold) → return stored verdict (fast path)
  │
  ▼
[3] Live web retrieval  ← novel claims only
    Serper API → domain filter → Trafilatura extraction
    Adaptive iteration: up to 2 extra rounds if evidence insufficient
  │
  ▼
[4] Veracity prediction
    Large LLM → structured JSON verdict
  │
  ▼
[5] DB update
    Embed + store claim, verdict, sources in ChromaDB
```

---

## Stage 1 — Claim Detection + Decomposition + Query Generation

One structured LLM call handles all three tasks. Since we must call the LLM to detect and decompose claims anyway, generating search queries in the same call costs nothing extra.

**Decomposition methodology:** Claimify (ACL 2025) — breaks posts into atomic, independently verifiable claims.

**Query strategy:** 2–3 queries per claim, each targeting a different angle:
- Direct factual query (e.g. `"WHO monkeypox pandemic declaration March 2025"`)
- Context/background query (e.g. `"WHO monkeypox announcement March 2025 details"`)
- Debunking-oriented query (e.g. `"WHO monkeypox pandemic 2025 false misleading"`)

Multiple angles improve retrieval recall, especially for claims using unusual phrasing.

**Model:** Qwen2.5-7B-Instruct via vLLM (~15GB VRAM on A40). Fast, free (self-hosted).

**Output schema:**
```json
{
  "is_checkable": true,
  "claims": [
    {
      "text": "The WHO declared monkeypox a pandemic in March 2025",
      "queries": [
        "WHO monkeypox pandemic declaration March 2025",
        "WHO monkeypox announcement March 2025 details",
        "WHO monkeypox pandemic 2025 false misleading"
      ]
    }
  ]
}
```

---

## Stage 2 — Claims DB Lookup

- Embed each atomic claim with `paraphrase-multilingual-MiniLM-L12-v2` (EN+FR, 420MB, fast)
- Query ChromaDB by cosine similarity
- **Similarity threshold:** tunable parameter (default: 0.85) — key experimental variable
- Cache hit → return stored verdict immediately, skip retrieval entirely
- DB seeded at startup from **Google Fact Check Tools API** (free, broad coverage)
- Grows with every novel claim verified in Stage 4

This is the primary cost-reduction mechanism. The DB fill-up phase is the only expensive period; once coverage is established, the vast majority of checks hit the cache.

---

## Stage 3 — Live Web Retrieval

### Search
**API:** Serper (`google.serper.dev/search`)
- 3 queries per claim, top 3 URLs per query = up to 9 candidate URLs
- Date-bounded via Serper's `tbs=cdr:1,cd_max:DD/MM/YYYY` — prevents surfacing articles published after the claim was made

### Domain Filtering (before extraction)
Two local lookups — no API calls, negligible latency:

- **Iffy+** — 1,300+ unreliable domains, free JSON. Blocklist: drop URLs on this list entirely.
- **CRED-1** (2025) — 2,672 domains, composite credibility score 0–1, free JSON (145KB). Domains scoring < 0.3 are dropped. Surviving domains carry their score as metadata passed to the veracity LLM as source weighting context.

Both files loaded into memory at startup.

### Content Extraction
**Library:** Trafilatura (local Python, free, no API)
- Fetches URL, returns clean article text (strips nav, ads, boilerplate)
- Returns `None` on 403/404 or failed fetch

**Bot blocking mitigation:**
- Per-domain request delay: 2.0–3.5 seconds (randomized jitter) between requests to the same domain, tracked via a `last_seen` dict keyed by domain
- Rotating User-Agent from a pool of real browser strings
- One retry max per URL — don't hammer sites

**Concurrency:**
- Serper queries fired in parallel (`ThreadPoolExecutor`)
- Scraping parallelized across domains — per-domain delays are domain-scoped, so scraping `reuters.com` doesn't block `apnews.com`

### Adaptive Iteration (FIRE-style)
After the first retrieval pass, the LLM assesses whether evidence is sufficient to reach a confident verdict.

- Yes → proceed to Stage 4
- No + < 2 extra rounds used → generate follow-up queries, search again
- Stop on: sufficient evidence | 2 extra rounds exhausted | duplicate queries

---

## Stage 4 — Veracity Prediction

**Model:** Qwen3-32B-AWQ via vLLM (~23GB VRAM on A40). HerO2-proven at AVeriTeC. ~29s/claim throughput → ~3,000 claims/day ceiling on one A40.

Claim + all retrieved evidence (with CRED-1 credibility weights per source) passed in a single call. Chain-of-thought reasoning before structured output.

**Output schema:**
```json
{
  "reasoning": "Reuters (credibility: 0.94) reports that... A second source (credibility: 0.41) contradicts this but is a known low-reliability outlet...",
  "verdict": "Refuted",
  "confidence": 0.81,
  "justification": "Three independent high-credibility sources directly contradict the claim.",
  "key_sources": ["https://reuters.com/...", "https://apnews.com/..."]
}
```

**Verdict scale** (4-class, AVeriTeC / ClaimCheck convention):
- `Supported`
- `Refuted`
- `Not Enough Evidence`
- `Conflicting Evidence`

---

## Stage 5 — DB Update

- Embed verified claim with `paraphrase-multilingual-MiniLM-L12-v2`
- Store in ChromaDB: embedding + verdict + confidence + timestamp + key sources
- Available for future lookups in Stage 2

---

## Tech Stack

| Component | Tool | Notes |
|---|---|---|
| LLM serving | vLLM | Both models on A40 |
| Claim/query gen model | Qwen2.5-7B-Instruct | ~15GB VRAM |
| Veracity model | Qwen3-32B-AWQ | ~23GB VRAM; HerO2-proven |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | EN+FR, 420MB |
| Vector DB | ChromaDB | PoC; swap to Qdrant for production |
| Search API | Serper | $0.30–1/1k; 2,500 free/month |
| Content extraction | Trafilatura | Local, free |
| Domain blocklist | Iffy+ | Free JSON, 1,300+ domains |
| Domain credibility | CRED-1 | Free JSON, 2,672 domains, 0–1 score |
| DB seed | Google Fact Check Tools API | Free |

---

## Cost Model

| Phase | Novel claims/day | Search (Serper) | LLM | Total/month |
|---|---|---|---|---|
| PoC / testing | < 83 | Free (within 2.5k free tier) | $0 (self-hosted) | **~$0** |
| DB fill-up | ~3,000 | ~$81 | $0 | **~$81** |
| Production (mature DB) | ~50–200 | ~$2–6 | $0 | **~$5** |

---

## Project Structure

```
project/
├── docs/
│   └── live_factchecking_pipeline_poc_2026-05-07.md
├── pipeline/
│   ├── __init__.py
│   ├── models.py           # Pydantic schemas for all stage I/O
│   ├── claim_extraction.py # Stage 1: LLM → structured JSON
│   ├── db_lookup.py        # Stage 2: ChromaDB query
│   ├── retrieval.py        # Stage 3: Serper + domain filter + Trafilatura
│   ├── verification.py     # Stage 4: LLM veracity call
│   └── db_update.py        # Stage 5: embed + store
├── domain_data/
│   ├── cred1.json          # CRED-1 credibility scores
│   └── iffy.json           # Iffy+ blocklist
├── config.py               # Thresholds, model names, API keys
├── evaluate.py             # LiveFact benchmark runner
└── main.py                 # Single-claim entrypoint
```

---

## Tunable Parameters

| Parameter | Default | Role |
|---|---|---|
| `similarity_threshold` | 0.85 | DB cache hit/miss boundary |
| `serper_top_k` | 3 | URLs per search query |
| `max_retrieval_rounds` | 2 | Extra adaptive iterations |
| `cred1_min_score` | 0.3 | Minimum credibility to include source |
| `domain_delay_seconds` | 2.0–3.5 | Per-domain scraping delay (randomized) |
| `verdict_confidence_threshold` | 0.65 | Minimum confidence to issue nudge |

---

## Deferred to Later Iterations

- **Encoder ensemble pre-filter** (IDEA-001) — language-based fast pre-filter before retrieval
- **Bias / misleadingness rubrics** — additional scoring dimensions beyond veracity
- **Async processing for first-user novel claims** (IDEA-002)
- **French language fine-tuning** — multilingual embeddings already chosen with this in mind

---

## Evaluation Plan

1. Unit test each stage with synthetic inputs (known-false claim, known-true claim, non-checkable post)
2. End-to-end test on 10 LiveFact claims — manually verify reasoning quality and source selection
3. Full benchmark on LiveFact corpus — 4-class accuracy vs. ground truth
4. Cost audit — Serper credit usage per run; Trafilatura success rate across URL types

---

---

## Local Development Adjustments (M1 Mac, 8GB unified memory)

These changes apply only during local testing. The pipeline structure and all interfaces remain identical — the goal is to validate logic before moving to the A40.

**Stages 2 and 5 (Claims DB) are skipped.** Every claim goes through live retrieval. The DB will be implemented once on the server.

**Single model for both Stage 1 and Stage 4:** Use **Qwen3-4B via Ollama** for both claim extraction and veracity prediction.

| | Production | Local dev |
|---|---|---|
| Stage 1 model | Qwen2.5-7B (vLLM) | Qwen3-4B (Ollama) |
| Stage 4 model | Qwen3-32B-AWQ (vLLM) | Qwen3-4B (Ollama) — same instance |
| Stage 2/5 | ChromaDB lookup/write | Skipped |
| VRAM used | ~38GB (A40) | ~2.5GB (M1 unified) |
| Throughput | ~3,000 claims/day | ~100–200 claims/day |

**Ollama setup:**
```bash
ollama pull qwen3:4b
# Accessible at http://localhost:11434 (OpenAI-compatible API)
```

Qwen3-4B was chosen over Qwen2.5-1.5B for better reasoning quality during development, and over Qwen2.5-7B because it fits comfortably on 8GB with macOS overhead (~5.5GB headroom). The thinking mode (`/think`) can be enabled for veracity prediction to improve chain-of-thought quality at the cost of latency.
