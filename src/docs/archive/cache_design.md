# The cache — design

*Canonical, consolidated 2026-06-26. Merges the built text-claim verdict cache (formerly
`tier1_cache_design.md`) with the multimodal extension into one doc. We call it **the cache**
(the old "Tier-1" label is dropped). One shared Neon/pgvector DB. Parameters marked
**[TBD by eval]** are set by `eval/scripts/cache_eval/`.*

## What the cache is (plain terms)

The cache is the system's **"have we already fact-checked this?" memory.** When a new post
arrives, before paying for slow verification we ask the cache whether we already know a verdict
for this claim (or have already seen this image). A hit returns the stored verdict instantly; a
miss falls through to the live fact-check, whose result is then written back so the next person
sharing the same claim is served from memory.

The whole design hangs on one hard problem: **two claims can look almost identical to a
similarity model yet have opposite verdicts** ("unemployment is 4%" vs "unemployment is 14%",
"X happened" vs "X did *not* happen"). So the cache is built to retrieve loosely but **reuse a
verdict only when it provably still applies.**

## The one principle

**Cosine retrieves candidates; a logical-equivalence gate decides reuse.** Embedding
similarity flips on exactly the cases that flip a verdict — negation, number/entity/scope swaps.
So lookup is **two-stage**: a cheap high-recall nearest-neighbor search proposes candidates, then
a precise gate decides whether the cached verdict carries over.

```
new claim
  → normalize  (canonical, attribution-stripped form)
  → embed      (claim-level text embedder → L2-normalized vector)
  → ANN nearest neighbors in pgvector   [STAGE 1: recall filter, generous cut]
  → equivalence gate (bidirectional) + slot-veto   [STAGE 2: precision arbiter]
       confirmed → reuse cached verdict + fold claim into the neighborhood
       rejected  → fall through to the live fact-check (FCT API / full verification)
```

This is the named, well-studied task **PFCR — Previously Fact-Checked Claim Retrieval** (prior
art below). *In plain terms:* find claims that look similar (recall), then strictly check that the
old verdict really fits the new claim before reusing it (precision).

---

## 1. Embedding spaces — text, image, and a hash index

Three indexes over **one shared vector DB**, each doing a different job. *In plain terms:* one
fingerprint for a claim's **text** (the verdict key), one for a post's **image** (to spot a known
image coming around again), and a cheap exact **image hash** for verbatim reposts.

| Space | Job (plain) | Self-host (Apache, ship-safe) | Managed option |
|---|---|---|---|
| **Claim-level (text)** — *the verdict key* | match a new claim to an already-checked one | **Qwen3-Embedding-0.6B/4B** or **BGE-M3** (EN+FR) | Gemini Embedding / Voyage-3-large / Cohere v4 |
| **Post-level (multimodal)** | spot a re-cropped / re-captioned known image; link quote-tweets to the parent | **SigLIP 2 So400m** (1152-d, 109 langs) | **Cohere Embed v4** (text+image; 128k ctx) |
| **Perceptual hash (PDQ / pHash)** | catch a **verbatim** image repost instantly, no AI | local, cheap | — |

### Claim-embedder upgrade (the verdict key)

- **Current (built):** `paraphrase-multilingual-MiniLM-L12-v2`, 384-d, via
  `pipeline/embedding.py::embed` (the single entry point; already L2-normalized so cosine = dot
  product). Storage is `halfvec(384)` + HNSW (below).
- **Upgrade path:** move the verdict key to **Qwen3-Embedding-0.6B/4B** or **BGE-M3**. *In plain
  terms:* a newer, sharper text model — better at the fine distinctions (a number, a "not") that
  flip a verdict, which is exactly what the cache must not get wrong. Cost: a re-embed + HNSW
  rebuild and a dimension change; keep it behind the same `embed` entry point so the swap is
  local.

### Post-level multimodal space + perceptual hash

- **pHash/PDQ** is the cheap, strong signal for **exact** image recirculation. A quote-tweet
  usually carries the parent's image **verbatim** → a pHash exact-match is the best parent-link
  signal. Prior art: **PixelMod** (USENIX Security 2024), visual-misinfo soft-moderation on X via
  perceptual hashing.
- **SigLIP / Cohere embedding NN** layers on top for **re-captioned / re-screenshotted** variants
  the hash misses. *In plain terms:* the hash catches identical reposts; the AI embedder catches
  edited versions of the same image.
- **Quote-tweet → parent linking:** the verbatim-image hash links a QT back to the post it
  quotes, so a known-false parent taints its quote-tweets.

**Why two embedding spaces, not one:** the multimodal tower optimizes image–text *alignment* and
blurs fine textual detail (numbers, negation) — the very thing that flips a verdict. The verdict
key must ride a **text-specialized** retriever. Collapse to one (Cohere v4) only if minimizing
vendors outweighs the precision/cost hit.

**Licensing (decide early — it constrains productization):** Apache/clean → **SigLIP 2,
Qwen3-Embedding** (ship-safe). **Non-commercial / research-only** → Jina CLIP v2/v4, Nomic,
NV-Embed (do **not** build the shippable extension on these).

---

## 2. Verdict reuse — lookup, the gate, and the slot-veto

### Lookup path (`cache.lookup`)

1. **Normalize** the incoming claim to a canonical, attribution-stripped sentence — same transform
   as `synth_representatives.py:SYNTH_SYSTEM` (single claim, no "X said that…" unless the speaker
   is the point). Key the cache on canonical forms, not raw post text.
2. **Embed** via `pipeline.embedding.embed` (L2-normalized; cosine = dot product).
3. **ANN retrieve** the top-`K_ANN` neighbors by cosine (`<=>`), keeping any with
   `cosine ≥ SIMILARITY_THRESHOLD`. This cut is a **high-recall filter, not the decision** — set
   it generously **[TBD by eval]** (well *below* the old 0.92 placeholder; the gate provides
   precision).
4. **Two-pronged equivalence gate** — for each candidate neighborhood, run the gate against
   **both** (a) the neighborhood's **canonical claim** (the representative) and (b) the **nearest
   individual member** claim, so a match is still caught when the new claim's wording differs from
   the canonical phrasing.
5. **Decision = mutual entailment + slot-veto** (below): reuse the verdict only if the gate
   confirms equivalence against (a) **or** (b) **and** no load-bearing slot mismatches. On the
   first confirmed match, return that cached verdict; else `None` (fall through).
6. **Staleness:** a confirmed hit is still a miss if `created_at` is older than the recheck TTL —
   re-verify rather than serve a stale verdict.

### The equivalence gate

Recommend **starting with an LLM-as-judge**, prompted to *explicitly* check four adversarial
axes, run **bidirectionally**, requiring mutual entailment. *In plain terms:* an LLM asks "would a
single fact-check verdict apply, unchanged, to both claims?" and must check the specific ways two
claims can secretly differ.

- The research shows vanilla NLI fails on negation without negation-specific fine-tuning (MoNLI);
  an instructed LLM judge sidesteps that and is fastest to stand up. We already run LLM judges
  elsewhere (the 4-class evaluator, the synth prompt), so the infra exists.
- **Open alternative:** a negation-robust fine-tuned NLI model (cheaper per call, no API). The
  gate is a swappable function behind one signature, so this choice isn't load-bearing.

**Gate criterion — verdict-applicability, not strict logical equivalence.** The reuse question is
*"would a single fact-check verdict apply identically to both?"* — looser than mutual entailment.
Consequence, **hedge scoping** (resolved with Daniel 2026-06-05), three buckets:

- **Evidential hedges** ("reportedly / apparently / allegedly X") → **equivalent** (HIT): they
  assert X and get X's verdict.
- **Modal hedges** ("X might / could / possibly", "some say X") → **separate claim** (own cluster):
  they assert only possibility; collapsing could misapply a Refuted verdict to a defensibly-true
  hedge. Minor recall cost, accepted.
- **Attribution / speech-act** ("X said Y", X a public figure) → **NEVER collapse with "Y"** —
  hard guardrail. "Did X say it" and "is Y true" are two different checks with different verdicts
  and different nudge semantics; reusing a "Y is false" verdict to flag an accurate "X said Y"
  post would flag accurate reporting as misinformation (see the nudge-safety note in
  `src/CLAUDE.md` Open Questions). Keep them in separate clusters.

Adversarial axes the gate prompt must name explicitly:

| axis | example (must be judged NOT equivalent) |
|---|---|
| negation | "X happened" vs "X did NOT happen" |
| number/quantity | "52%" vs "62%" |
| named-entity | "PM Modi" vs "Virat Kohli" |
| scope | "some X" vs "all X" |

**The gate returns a *relationship*, not just HIT/MISS** — a near-miss is signal, not noise:

- `equivalent` → reuse the verdict; **join** the candidate's cluster.
- `negation` → do NOT reuse; record a `negation_of` edge between the two clusters.
- `related_independent` → cosine-near but neither equivalent nor a clean negation; record a
  `related` edge.
- `unrelated` → ignore (a coincidental neighbor).

Only `equivalent` triggers reuse; the other two are stored as cross-cluster edges for the
observatory (§4).

### The deterministic slot-veto (belt-and-suspenders)

On top of the LLM gate, a **rule-based** veto. *In plain terms:* a dumb, always-the-same check
that pulls the **numbers, dates, named entities, and yes/no polarity** out of both claims; if
**any** load-bearing slot mismatches, **reuse is vetoed regardless of cosine or the LLM's call.**
Catches "$5B vs $50B", "2019 vs 2024", "did vs did *not*", "Biden vs Trump." It's redundant by
design — a hard safety net under the model judgment, since a wrong reuse is the cache's most
dangerous failure.

**Never infer a verdict from embedding/image similarity** (the "Similarity over Factuality"
failure): embeddings encode *topic*, not truth-conditions. Similarity (text or image) is only
allowed to **cluster and link** posts; the verdict is always a property of the **claim**.

---

## 3. Images — recirculation and out-of-context reuse

- Use pHash + the multimodal embedder to **cluster and link** recirculating posts (§1), never to
  assign a verdict.
- **Out-of-context reuse** — a real image with a false new caption — looks "consistent" to an
  embedder. So image signals link the recirculation; the **claim** (text + caption, normalized via
  §2) carries the verdict. *In plain terms:* the same authentic photo can be true in one post and
  the centerpiece of a lie in another — the image tells us "this is the same picture," the claim
  tells us whether it's being used to mislead.

---

## 4. Neighborhoods / representatives (the observatory's overlay)

Adopt the **k-LLMmeans** pattern over online medoid recomputation:

- **Cluster growth = DP-means / leader rule:** if the nearest representative is within the cut, the
  confirmed-equivalent claim **joins** that neighborhood (lookup already established this). A
  brand-new claim that falls through *seeds* a new neighborhood on `write`.
- **Per neighborhood, store:**
  - `centroid_embedding` — running mean of member embeddings for fast ANN assignment (cheap
    incremental update `c ← c + (x − c)/n`, renormalize).
  - `canonical_claim` — **LLM-synthesized** representative (reuse `synth_representatives.py:synth`),
    the human-readable, auditable key.
  - `members` — `(claim_text, embedding)` list, so the gate can compare against the nearest member
    (lookup prong b) and canonical re-synthesis has inputs.
- **Re-synthesize the canonical claim periodically** (every N folds or on a schedule), not on every
  fold — keeps cost bounded (the mini-batch idea from k-LLMmeans).

---

## 5. Data model — store every claim (the observatory)

The cache is not just a verdict store; it's a **longitudinal record of what misinformation
circulates and how often.** We **never collapse claims away** — every claim is its own row;
clusters are an overlay.

- **`claims` (one row per claim ever seen):** `claim_id`, raw `text`, normalized `canonical_text`,
  `embedding`, `created_at`, `source` (post/platform id), `language`, `cluster_id`. The verdict
  lives on the cluster, not the claim.
- **`clusters` (one row per equivalence class):** `cluster_id` (UUID — the stable analytics
  handle), `canonical_claim`, `centroid_embedding`, the cached verdict payload,
  `verdict_confidence` (= the source `stopped_reason`, see §6), `created_at`, `last_seen_at`,
  `member_count`.
- **`cluster_edges` (relationships between clusters):** `(cluster_a, cluster_b, relation)` with
  `relation ∈ {negation_of, related_independent}` — emitted by the gate's relationship
  classification. Lets us study claim/counter-claim dynamics.

**What this unlocks:** cluster `member_count` over time = recurrence / virality;
`created_at`→`last_seen_at` = a claim's lifespan; `language` spread = cross-lingual diffusion
(EN↔FR); `negation_of` edges = where a claim and its rebuttal both circulate. This is the
claim-side empirical dataset that complements the extension's behavioral-data arm.

---

## 6. Write path & policy (`cache.write`)

- **What to persist** (from `models.ClaimVerdict`): `claim_text` (canonical), `claim_embedding`,
  `verdict_4class`, the four `VerdictScores` Likerts, `tier_resolved`, `cap_hit`, `evidence_urls`,
  source publisher(s), `justification`, `created_at`.
- **Cache *everything*, with a confidence-tiered TTL.** We store every verdict (complete
  observatory + never re-verify from scratch), but how long we *reuse* it before re-checking
  depends on the verification loop's `stopped_reason` (`verify.py`). *In plain terms:* if the model
  stopped because it was **satisfied**, trust the verdict longer; if **we** cut it off, re-check
  it sooner.

  | `stopped_reason` | how it arises | meaning | reuse TTL |
  |---|---|---|---|
  | `confident` | synthesis returned `next_query: null` — the model volunteered it had enough | high trust | full `RECHECK_AFTER_DAYS` |
  | `redundant` | next query was cosine ≥0.9 vs a past one — *we* cut it off | lower trust | **short** TTL, then re-verify |
  | `cap` | hit `MAX_ROUNDS` — *we* cut it off mid-search | lower trust | **short** TTL, then re-verify |

  Persist `stopped_reason` on the cluster as `verdict_confidence`. FCT-API (trusted-publisher)
  verdicts are high-trust by construction.

- **Verdict reuse is verbatim only — for now; flip-reuse is a planned extension.** The first cut
  reuses a cached verdict only for *equivalent* claims. **Planned next capability — flip-reuse:**
  when a cluster holds a *well-sourced Supported* verdict on a specific value (e.g. "the figure is
  90%"), any same-proposition claim asserting a *different* value is provably **Refuted** by that
  cluster — one verified-true value resolves the whole numeric family (and a negation of a
  Supported claim → Refuted, etc.). Needs the gate to (a) classify "same proposition, contradictory
  value/polarity" and (b) know which side is verified-true, then emit the *flipped* verdict. Kept a
  distinct follow-up.

---

## 7. Storage / serving

- **Neon + pgvector**, three tables (§5): `claims`, `clusters`, `cluster_edges`.
- **Store everything; index every claim (A1)** — validated against pgvector/Neon docs; the choice
  is *what goes in the ANN index*, and it's reversible:
  - **A1 (chosen):** index all claim embeddings with **HNSW**. Highest recall (many landing pads
    per cluster → ANN reliably lands in the right equivalence class; resolve via `cluster_id`),
    members available for the two-pronged gate, re-validatable clusters. HNSW absorbs inserts
    without rebuilds and is the recommended index for **<10M vectors with active writes**.
  - **`halfvec` (float16)** — embeddings are L2-normalized, so halfvec keeps recall **>99%** while
    halving storage + index size (384-d → **768 B/vector**). Use from day one.
  - **A2 migration trigger:** if HNSW index RAM approaches the Neon compute tier (~5–10M vectors),
    switch the *index* to centroids-only (keep `centroid_embedding` as a running-mean field so this
    is a config change). Then ANN over `N_clusters`, fetch members by `cluster_id`.
  - **Not** centroids-only *storage*: the one irreversible choice; it breaks the two-pronged gate,
    cluster re-validation, and recurrence measurement.
  - Cost at scale (halfvec): 1M claims ≈ 0.75 GB vectors + ~1.5–2 GB HNSW RAM; 10M ≈ 7.5 GB +
    ~15–20 GB. Disk is cheap; HNSW RAM is the binding constraint.
- **Index op:** `vector_cosine_ops` (embeddings already normalized).
- **TTL:** the confidence-tiered policy in §6 (`RECHECK_AFTER_DAYS` for `confident`,
  `RECHECK_AFTER_DAYS_LOWCONF` for `redundant`/`cap`). **[TBD]** — start conservative; no source
  quantifies misinformation-verdict drift.

---

## 8. Parameters to pin (all [TBD by eval] unless noted)

| param | role | current | recommendation |
|---|---|---|---|
| `SIMILARITY_THRESHOLD` | ANN recall cut (stage 1) | 0.92 | **≈0.60** — pinned by `cache_eval` Run 1: precision is gate-bound + flat across the sweep, so the cut is a recall/cost lever |
| `K_ANN` | candidates retrieved | — | small (e.g. 3–5) |
| equivalence-gate mode | reuse decision (stage 2) | — | bidirectional LLM judge, mutual entailment — **validated**: negation 0/66, entity 0/61 false-HITs; precision ~0.98 (`cache_eval/RESULTS.md`) |
| slot-veto | deterministic reuse veto | — | new; number/date/entity/polarity mismatch → veto |
| `RECHECK_AFTER_DAYS` | TTL for `confident` verdicts | 30 | conservative; revisit |
| `RECHECK_AFTER_DAYS_LOWCONF` | short TTL for `redundant`/`cap` | — | well under the confident TTL |
| canonical re-synth cadence | representative freshness | — | every N folds |

---

## 9. Modules to reuse (don't rebuild)

- `pipeline/embedding.py::embed` — embedding (the only entry point).
- `pipeline/cache.py` — fill the `lookup`/`write`/`init_schema` contract.
- `eval/scripts/_archive/feed_study/synth_representatives.py::synth` + `SYNTH_SYSTEM` —
  canonical-claim synthesis. *(feed_study archived 2026-06-26; lift this module out when the cache
  build consumes it.)*
- `eval/scripts/_archive/feed_study/dedup_claims.py` — centroid/medoid logic (offline analog;
  reuse the math, not the batch flow).
- `pipeline/verify.py` confidence gate (`stopped_reason`) — cache-eligibility.
- `pipeline/models.py::ClaimVerdict` — the cached payload.

---

## 10. Build order / status

1. ✅ **Run `eval/scripts/cache_eval/`** → pinned `SIMILARITY_THRESHOLD≈0.60`, validated the gate
   (2026-06-05).
2. ✅ **Implement `cache.py`** (`init_schema`, `write`, `lookup`, `_gate`, `_fold`) — done
   2026-06-05; smoke-tested end-to-end against local pgvector (paraphrase→HIT, negation/unrelated
   →MISS, fold-on-hit verified). Uses `halfvec(384)` + HNSW. **DB:**
   `docker compose up -d claims-db` (Postgres 16 + pgvector 0.8.2); DSN in `.env`
   (`NEON_DATABASE_URL`).
3. ✅ **Wire `lookup`/`write` into the pipeline** — done 2026-06-05. `pipeline/pipeline.py` calls
   `cache.lookup`/`write` around the FCT-API and full-verification stages, with `skip_cache` a
   **full** bypass (no read *and* no write → no DB needed) + a `source` field threaded through for
   the observatory. The verification eval is untouched (`verify_run.py` calls `verify.verify()`
   directly).
4. ⏳ **`cluster_edges` population** (negation/related observatory edges) — table exists, not yet
   written; the natural home is `write()` gating the new cluster against nearby ones.
5. ⏳ **Claim-embedder upgrade** (MiniLM → Qwen3-Embedding / BGE-M3) — re-embed + HNSW rebuild +
   dimension change behind the `embed` entry point.
6. ⏳ **Slot-veto** — add the deterministic number/date/entity/polarity check on top of the gate.
7. ⏳ **Multimodal post-space + pHash index** — SigLIP 2 / Cohere v4 + PDQ/pHash for image
   recirculation and quote-tweet→parent linking (the image workstream).
8. ⏳ Provision a hosted Neon DB for non-local use (currently local-only).
9. ⏳ Measure production hit-rate from the next X scrape.

---

## Prior art

Shaar et al. 2020 *"That Is a Known Lie"* (BM25→BERT) · CLEF CheckThat! Task 2 · Kazemi et al. 2021
*Claim Matching Beyond English* (matching = semantic equivalence, not topical) · MultiClaim
(EMNLP 2023) · **SemEval-2025 Task 7** (bi-encoder retrieval → LLM reranker) · **FACT-GPT** (NLI as
the matching decision) · CheckThat! 2025 claim normalization · **PixelMod** (USENIX Security 2024,
perceptual-hash visual-misinfo moderation) · *Similarity over Factuality* (WACV 2025). Background
research: `tier1_cache_research.md`.
