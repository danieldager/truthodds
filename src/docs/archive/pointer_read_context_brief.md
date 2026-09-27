# READ-step context sizing: literature brief

**Question this answers.** In the verify loop's READ step, an LLM (DeepSeek-V4-Flash,
non-thinking, JSON) sees a POST, its 2–11 extracted CLAIMS, and 1–3 scraped articles
whose sentences are pre-numbered `[S1] [S2] …`. It emits *segment pointers*
`{claim_id, segs:[14,15], stance}` instead of quotes. The open design call: **how much
of each article to show the model before comprehension / segment-ID matching degrades**,
and whether keyword paragraph-selection is needed at all.

This brief surveys the attributed-generation, long-context, and multi-doc literature and
ends with concrete recommendations plus what we still have to measure ourselves.

---

## 1. Sentence-level citation / pointing accuracy (attributed generation)

**Our scheme is essentially LongCite's, and LongCite validates it.**
[LongCite (Zhang et al., 2024, arXiv:2409.02897)](https://arxiv.org/abs/2409.02897)
does exactly what we do: segment the context into sentences (NLTK), number them, and have
the model emit spans `[k]` (one sentence) or `[a-b]` (sentences a..b) rather than quotes.
Findings that transfer directly:
- **Sentence-level > chunk-level** for precision and "semantic integrity"; better than
  128-token fixed chunks.
- **Multi-sentence spans beat isolated single sentences.** Their best data comes from
  extracting sentence *spans* out of a chunk, not forcing one-sentence citations — spans
  are more stable and semantically coherent. → keep our `segs:[14,15]` span capability.
- Citation F1 (precision+recall of cited sentences): **LongCite-8B 72.0, LongCite-9B 69.2,
  GPT-4o 65.6** (avg across datasets, contexts up to 128K). So even a purpose-built 8B
  model tops out around **70–72 F1**; an untrained strong model (GPT-4o) sits at **~66**.
- Correctness and citation quality are **mutually reinforcing** — getting the answer right
  and citing correctly rise together, so bad pointing is also a signal of bad reading.

**ALCE (Gao et al., 2023, EMNLP; arXiv:2305.14627)** — the canonical attribution
benchmark ([GitHub](https://github.com/princeton-nlp/ALCE),
[paper](https://arxiv.org/abs/2305.14627)). Key numbers:
- On ASQA with GPT-4, **citation recall 68.5% at 5 passages → 73.0% at 20 passages**;
  correctness also rose (41.3→44.4). So *more evidence helped, not hurt*, for a strong
  model on a clean task.
- But on the harder ELI5, "even the best models lack complete citation support **~50% of
  the time**." Attribution quality is task- and model-dependent and far from solved.
- General finding echoed across the attribution survey
  ([arXiv:2311.03731](https://arxiv.org/pdf/2311.03731)): **50–90% of citations in
  long-form answers are not fully supported** by the cited source in older systems.

**What degrades pointing accuracy** (synthesis across the above +
[ALiiCE, arXiv:2406.13375](https://arxiv.org/html/2406.13375)):
1. **Number of candidate segments / distractor density** — more numbered sentences =
   harder matching (this is the same mechanism as long-context retrieval, below).
2. **Claim–evidence lexical gap** — the single biggest risk for *us* (see §2, NoLiMa).
3. **Formatting**: numbered inline markers that the model can copy verbatim (`[S14]`) work
   better than asking for char offsets or free-form quotes. Per-sentence numbering with
   span output is the validated sweet spot.

## 2. Long-context comprehension degradation curves

**Lost-in-the-middle** ([Liu et al., 2023, arXiv:2307.03172]; summarized via
[Found-in-the-Middle, arXiv:2406.16008](https://arxiv.org/html/2406.16008v1)): U-shaped
performance — models attend most to the **start (primacy) and end (recency)** of the
prompt and neglect the middle. Evidence buried mid-context is the most likely to be missed.
Actionable: put the material you most need attended-to at the **edges**.

**RULER** ([Hsieh et al., COLM 2024, arXiv:2404.06654](https://arxiv.org/html/2404.06654v1)):
near-perfect vanilla needle-in-haystack scores collapse on *realistic* multi-needle /
multi-hop / aggregation tasks as length grows. **Effective context ≈ 50–65% of the
advertised window** for most models; many "128K" models only truly handle ~32K, and some
fail before that. Takeaway: advertised window ≫ usable window for reasoning tasks.

**NoLiMa** ([Modarressi et al., ICML 2025, arXiv:2502.05167](https://arxiv.org/html/2502.05167v1))
— **the most relevant paper to our lead's concern.** It removes lexical overlap between
query and evidence (exactly our "evidence may not share words with the claim" worry) and
measures where accuracy falls below 85% of each model's short-context base:

| Model | base | 1K | 2K | 4K | 8K | 16K | 32K | effective len (≥85% base) |
|---|---|---|---|---|---|---|---|---|
| GPT-4o | 99.3 | 98.1 | 98.0 | 95.7 | 89.2 | 81.6 | 69.7 | **8K** |
| Llama-3.3-70B | 97.3 | 94.2 | 87.4 | 81.5 | 72.1 | 59.5 | 42.7 | **2K** |
| Gemini-1.5-Pro | 92.6 | 86.4 | 82.7 | 75.4 | 63.9 | 55.5 | 48.2 | **2K** |

- **When the match is non-lexical, degradation starts almost immediately (by 1–4K) and is
  steep.** The best frontier model holds to ~8K; strong open models fall below 85% of their
  own baseline by **2K tokens**. At 32K, 10–11 of 13 models are below half their baseline.
- This is the empirical basis for: *the lead's instinct to avoid brittle keyword selection
  is right, but the opposite failure (dump a huge non-lexical context and hope) is exactly
  what NoLiMa shows breaks down.* The lever that helps is **keeping the candidate context
  small**, not making the matching lexical.

**Where this lands us.** Our total prompt is **5–12K tokens** with **~40–100 sentences per
article** (an 800–2,000-word article; ~100–300 candidate segments only if all 3 are pooled).
We are on the *left, safer end* of these curves — but our task is the *hard* (non-lexical,
many-candidate, 11-claim) variant, so we should treat NoLiMa's early-degradation regime as
the governing one, not the friendly NIAH curve DeepSeek reports.

## 3. Multi-document interference

Pooling several docs in one prompt introduces **"mistaken synthesis"** — models mix up which
entity/event/fact came from which document, and attribution chains break, more so than in
single-doc prompts ([attribution survey arXiv:2311.03731](https://arxiv.org/pdf/2311.03731);
[PaperAsk arXiv:2510.22242](https://arxiv.org/pdf/2510.22242) shows multi-doc queries fail
where the *same* docs succeed queried individually — e.g. GPT-5 refused/incompleted 66% of
multi-paper cases that worked one-at-a-time). Practical mitigations reported: **process
documents individually** when attribution must be exact, and if pooling, **namespace IDs per
document** (D1:S1–S40, D2:S1–S35) so the model can't silently cross a boundary. The dominant
driver of error is again just **more candidate segments in one decision**, which per-doc
processing directly reduces.

## 4. DeepSeek-specific

**Thin — flag as an evidence gap.** The
[DeepSeek-V3 technical report (arXiv:2412.19437)](https://arxiv.org/html/2412.19437v1) only
reports **NIAH** ("robust up to 128K") — the easy test RULER/NoLiMa show is *not predictive*
of realistic retrieval. No published RULER or NoLiMa numbers for V3/V3.1/V3.2/V4 surfaced,
and none for **non-thinking mode** specifically. General leaderboard guidance still applies
(effective ≈ 60–70% of advertised;
[awesomeagents long-context leaderboard](https://awesomeagents.ai/leaderboards/long-context-benchmarks-leaderboard/)),
but **DeepSeek-V4-Flash non-thinking pointer accuracy at our lengths is unmeasured in the
literature — we must measure it ourselves.**

## 5. Practical RAG / reader-side chunking

Practitioner convergence for **citation-grounded reading** (not embedding retrieval):
- Factoid extraction favors **256–512 token** units; analytical/comparative reasoning wants
  **1,024+**; **~400–512 tokens is the common balanced default**
  ([firecrawl](https://www.firecrawl.dev/blog/best-chunking-strategies-rag),
  [langcopilot](https://langcopilot.com/posts/2025-10-11-document-chunking-for-rag-practical-guide)).
- For citation, number **every sentence** and let the model emit spans; expand chunks with
  neighbors so the true sentence isn't clipped at a boundary (LongCite; also
  [Tensorlake citation-aware RAG](https://www.tensorlake.ai/blog/rag-citations)). No source
  advocates numbering every 2–3 sentences — per-sentence + span output is the norm.

---

## Recommendations for our design

**(a) How much article per page.** Include the **full article up to a cap of ~2,000 words
/ ~2,500 tokens per doc** (covers the ~800–2,000-word typical article). Prefill is nearly
free so cost isn't the constraint — *accuracy* is, and at these sizes we're inside the safe
zone of the curves. For the rare long article (>~2,500 tokens), don't keyword-slice; instead
**truncate/segment the article and process it in windows** (or split into 2 per-doc calls),
preserving contiguous prose so non-lexical evidence isn't dropped.

**(b) Is keyword paragraph-selection needed?** **No, not as a default** — the lead is right;
NoLiMa shows lexical pre-filtering is exactly what drops non-lexical evidence. Only introduce
a length-reduction step **above a threshold (~2,500 tokens / one long article)**, and even
then prefer *contiguous windowing* over keyword extraction. Below that, feed whole articles.

**(c) Segment granularity + ID format.** **Number every sentence**, format `[S14]`
(bracketed, copyable inline). **Allow span pointers** `segs:[14,15]` — LongCite shows
multi-sentence spans are more accurate/stable than forcing single sentences. This is what
we already do; keep it.

**(d) Prompt layout (exploit position bias).** Put **articles first (numbered)**, then the
**POST + CLAIMS + task instruction + output schema last**, immediately before generation —
this puts the query in the high-attention *recency* zone. Optionally **also state the claims
briefly before the articles** (primacy zone) so the model reads with them in mind; this is a
cheap A/B to test given free prefill. Keep the numbered articles as the middle block, since
they're the searchable haystack, not the thing that must be attended-to as a whole.

**(e) Batched (3 docs/call) vs per-doc.** **Process per-document (1 doc/call).** The
multi-doc literature (§3) and the candidate-count mechanism (§2) both favor it: per-doc keeps
the candidate set to ~40–100 sentences and eliminates cross-doc "mistaken synthesis," at the
cost of 2–3× calls — which is nearly free here (tiny output, cheap prefill, TTFT ~1.8s). If
we ever must batch, **namespace IDs per doc** (`D1:S1…`, `D2:S1…`) and expect a measurable
accuracy hit. The 11-claim fan-out stays within each per-doc call (11 claims × ~60 segments,
not × 300).

**(f) Expected pointer accuracy + what to measure.** Ballpark from LongCite/ALCE for a
*non-fine-tuned* model on grounded sentence pointing: **~65–75% citation F1** in general;
we may do a bit better because pointing to *given* IDs is easier than open retrieval, but
worse because we're non-thinking + non-lexical. **Plan for ~70% and verify.** Our re-test
should hand-label a small gold set (claim → supporting sentence IDs + stance) and measure, as
functions of the levers above:
1. **Segment recall / precision / F1** (did we point at the right sentences?).
2. **Stance accuracy** (supports/refutes/neutral) conditioned on correct segment.
3. **Sweep article-length cap** (500 / 1,000 / 2,000 / full+truncate) to find where *our*
   model's accuracy knees — the NoLiMa-style curve for DeepSeek-V4-Flash is unpublished, so
   this is the one number the literature can't give us.
4. **Per-doc vs batched-3** head-to-head to quantify the interference cost for our data.
5. **Claims-before-and-after vs claims-after-only** layout A/B.

**Honest gaps.** (i) No published DeepSeek RULER/NoLiMa numbers, none for non-thinking mode
— §4 is our measurement, not the literature's. (ii) NoLiMa is a QA-retrieval proxy, not a
verify/stance task; the *shape* of degradation transfers, the exact thresholds may not.
(iii) The "70% F1" ballpark is cross-model — our own gold-set number supersedes it.

### Sources
- LongCite — https://arxiv.org/abs/2409.02897
- ALCE — https://arxiv.org/abs/2305.14627 · https://github.com/princeton-nlp/ALCE
- ALiiCE (positional fine-grained citation) — https://arxiv.org/html/2406.13375
- LLM attribution survey — https://arxiv.org/pdf/2311.03731
- Lost-in-the-Middle / Found-in-the-Middle — https://arxiv.org/html/2406.16008v1
- RULER — https://arxiv.org/html/2404.06654v1
- NoLiMa — https://arxiv.org/html/2502.05167v1
- PaperAsk (multi-doc failure) — https://arxiv.org/pdf/2510.22242
- DeepSeek-V3 technical report — https://arxiv.org/html/2412.19437v1
- RAG chunking practice — https://www.firecrawl.dev/blog/best-chunking-strategies-rag · https://langcopilot.com/posts/2025-10-11-document-chunking-for-rag-practical-guide
- Citation-aware RAG — https://www.tensorlake.ai/blog/rag-citations
- Long-context leaderboard — https://awesomeagents.ai/leaderboards/long-context-benchmarks-leaderboard/
