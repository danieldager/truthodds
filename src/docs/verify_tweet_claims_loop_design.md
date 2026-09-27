# Post-level verify loop — design & open questions

Status: **in design** (2026-07-09). Builds on `pipeline/verify_text.py` (raw-text controller loop,
veracity 1-5 + nudge flag). This is the survey verifier for objective-2 (per-outlet misinformation
vs NewsGuard). Keep this doc current as we resolve questions.

**Scale (REVISED 2026-07-10, w/ Daniel):** verify **5,000 tweets — 100 per source × 50 sources**
(supersedes 500×20=10k of 2026-07-09), sampled time-stratified from ~1k harvested per source so no
single news cycle dominates; plus a **dev subset of 10/source = 500** for prompt iteration. More
sources > more posts/source for the NewsGuard gradient (correlation across outlets is the metric;
per-source ±6-9pp CI acceptable; top-bin rates read pooled per bin — the verifier FP floor, not
sampling, dominates there). Roadmap: `docs/survey_5k_roadmap.md`.

## Agreed design so far

**Shape:** post-level accumulating verification over the post's **extracted checkworthy claims**
(double-DeepSeek), evolving `verify_text.py` from IMPLICIT claim decomposition → an EXPLICIT claim
ledger.

- **Claim ledger:** every checkworthy claim starts "open"; a shared evidence pool accumulates across
  rounds; the synthesis step checks off claims as they resolve. Because a post's claims are related,
  one round often closes several at once (the efficiency win).
- **Query:** prioritize the **most critical open claim(s)** (most-likely-false / most-harmful /
  most-checkable). Opening query = **non-thinking**, Serper, keyword-bag style (see
  `query_best_practices.md`). Up to **N Serper rounds (2 or 3 — OQ1)** then **one Exa escalation**
  (thinking, declarative neural query).
- **Summarise (reader) — redesigned (research-backed, OQ12):** a claim-aware, **page-local-stance**
  evidence extractor, non-thinking. Sees ALL the post's checkworthy claims + a post-context slice +
  the page (NOT prior verdicts). Per source, emits per-claim `{relevant, quotes, summary, stance ∈
  supports|refutes|neutral, date}` — **quotes before stance** so the label is quote-grounded; explicit
  "not relevant" exit. Folds **page-local stance** into the read (the synthesis offload the user
  wanted) but NOT the 4-way {Supported/Refuted/Conflicting/Unsupported} — those are cross-source
  (Conflicting/Unsupported need ≥2 sources), so they stay in synthesis, which becomes **aggregation
  over an accumulated stance-tagged evidence table**. Batch CLAIMS always (all claims × one page);
  batch DOCS only in small ID-tagged groups (2-4 short pages) to avoid attribution bleed.
  Sources: ClaimCheck, FIRE, Self-RAG (ISREL/ISSUP), AVeriTeC, RARR; see the research brief.
  **Audited 2026-07-10** (`summarize_audit.md`): batched = 0 hallucinated quotes (39 checked),
  0 attribution bleed (14/14), no coverage loss; ONE regression = laxer stance on a compound
  claim (Nicks+McGraw). Adopted with guardrails: (1) compound-claim rule — page confirms only
  part of a conjoined claim → cap at neutral; (2) ≤3 pages/call; (3) **output cap** — emit ONLY
  claims the page speaks to (no relevant=false padding: 27%+ of run-1 output tokens; killed the
  11-claim NEWSMAX post outright), ≤2 quotes ≤40 words, one-line summary. Latency is
  OUTPUT-bound (~13-15 tok/s; prefill ~free at 1.8s TTFT/3k tok), so batched ≠ faster — it
  serializes the same tokens per-doc parallelism would overlap; its win is fewer requests /
  one timeout-exposure point / one coherent table.
- **READ — LOCKED 2026-07-10 (v4 schema, prompt-iterated against the 3-rater gold):** pointer
  output `{claim_id, segs, stance}`, **5-way stance** (Daniel's scale: supports /
  partially-supports / neutral=contextualizes / partially-refutes / refutes, exact-proposition
  definitions) + **minimal-citation rule** (cite only the sentences that carry the stance;
  context is added mechanically later). Per-doc, claims-both layout, **2500-tok cap** (knee
  confirmed: v2 sweep degraded monotonically beyond 2500), resolver adds **±3-sentence merged
  context windows, cited core marked, no per-entry cap**. Measured @gold: **P 0.831 / R 0.726 /
  F1 0.775, segs/cell 2.44, stance 0.794 (rater ceiling 0.823), danger-class 4 (was 10)**.
  Iteration history (all 15-call runs via pools): v2 stance-FREE collapsed precision 0.77→0.56 —
  stance is citation-discipline scaffolding, NOT a redundant field (reversed the drop-stance
  lean; FEVER-style separation holds for JUDGING, but emitting a judgment sharpens selection);
  v3 5-way with loose defs over-cited (+33% segs/cell — "adjacent fact" wording legitimized
  adjacent citations); v4 fixed via minimal-cite. STEP treats stance as ADVISORY — claims close
  only on STEP's own reading of the resolved text. Watch item: 1 supports→refutes flip (n=1).
- **Non-LLM preprocessing (before READ):** (1) keyword-overlap paragraph selection — score
  paragraphs by content-word overlap with the claims, keep top-scoring + lede, instead of a
  blind `text[:5000]` head-slice (biggest quality win); (2) strip residual boilerplate lines
  (newsletter/share/Read-more); (3) unicode-normalize at scrape time (makes verbatim-quote
  checking exact); (4) publication date from trafilatura metadata / Serper, NOT from the model.
- **Synthesis / controller (NON-thinking for now):** reads post + claims + evidence, updates the
  ledger, and either issues a query targeting the still-open claims or concludes. Flash-thinking is
  OFF for now (the cost is latency, 2×/call). If one non-thinking call can't resolve the ledger,
  **split into two non-thinking calls** (ledger-update + control) before considering thinking/a
  smarter model.
- **STEP — BUILT + evaluated 2026-07-10 (`pipeline/verify_tweet_claims.py`, prompt v4):** single call =
  ledger-update + control + verdict-on-conclude (verdict-fold adopted; LLM-judged ledger adopted).
  Evaluated on 25 fresh dd500 posts / 83 claims vs 3-blind-Opus-rater gold (rater agreement:
  status 0.948, nudge 0.947). **v4: nudge acc 0.840 / precision 0.500 / recall 0.750, veracity
  MAE 0.36, misinfo_type 21/25, ledger 4-state 0.771-0.819.** Iteration: v1 recall 1.0/precision
  0.33 (over-flags — treats every claim as load-bearing); v2 added attribution rule ("X said Y" =
  the SAYING), refutation bar (near-miss numbers ≠ refuted), core-weighting → precision 0.50 but
  recall 0.50 (missed misleading-headline posts); v3 aggressive framing rule → see-saw (precision
  0.30); **v4 = v2 + framing-check-with-restraint (clear cases only) = keeper.** Known residuals:
  borderline 3-vs-4 posts flip run-to-run (raters themselves disagree there); ~14/83 claims
  rater-flagged insufficient_search (loop gives up early on side claims); 1 supported→refuted.
  STOP tuning at n=25 (4 gold positives — overfit risk); next calibration on the 500-post dev
  subset per `survey_5k_roadmap.md`.
- **Circularity guard (Daniel, 2026-07-10):** republished origin content is not independent
  corroboration. THREE layers, deliberately conservative: (1) non-LLM reprint-phrase check
  ("originally published by ⟨outlet⟩" — precise); (2) non-LLM cross-doc shingle near-dup (mirrors
  are textual copies; a debunk piece is not, so it can't false-positive); (3) **READ emits a
  per-doc `republication` flag** (a page that merely relays the outlet's reporting vs one that
  disputes it with own sourcing) → rendered in the evidence table → STEP's independence rule
  weighs it as the outlet's own voice. Bare "outlet named in page head" trigger REJECTED (would
  drop debunk articles that lead with the outlet's name). Discovered via the Grayzone/Pilgrims
  eval post: all "supporting" sources were mirrors of the origin article.
- **Verdict:** ONE **post-level** `{veracity 1-5, nudge (=veracity ≤3), misinfo_type
  FALSE|MISLEADING|UNSUPPORTED|NONE, justification}`. (Rename `misinfo`→`nudge`; apply when we build
  the reworked verdict step — `score_vtext.py` already uses `pred_nudge`.) `nudge` = the flag;
  `misinfo_type` = the kind of problem (FALSE=contradicted, MISLEADING=framing deceives,
  UNSUPPORTED=no evidence either way, NONE=fine).
- **Per-claim resolution (EVAL):** resolve EVERY checkworthy claim to one of **Supported / Refuted /
  Conflicting / Unsupported** — no early-exit. This 4-state is the ledger's closed status and the
  basis for the NewsGuard ratio. **Early-exit nudge (bail on first refutation) is PRODUCTION-only.**
- **Post verdict / nudge:** single post-level `veracity 1-5 + nudge (=veracity ≤3)`, derived from the
  resolved claims.
- **Source exclusion:** never use the originating outlet's own report as evidence — drop it before
  synthesis via `exclude_domains` (see [[project_verify_exclude_origin_source]]).
- **Cache (per post):** claims `[{text, status: open/closed, evidence refs}]`, the evidence pool, and
  the post verdict.
- **NewsGuard validation:** per outlet, the **ratio of resolved : unresolved claims** (definition —
  OQ6).

## Open questions

| # | Question | Lean / next step |
|---|---|---|
| OQ1 | Serper rounds before the Exa escalation — 2 or 3? | Make it a config knob; A/B on the eval slice. |
| OQ2 | Synthesis model. | **DECIDED: non-thinking Flash for now** (thinking off — latency; smarter model off — $200-320 even at 10k). Revisit only if quality forces it. |
| OQ3 | Split synthesis into two simpler NON-thinking calls (ledger-update + control)? | The quality fallback if one non-thinking call can't resolve the ledger. Cost = added latency, not $. |
| OQ4 | Summarise: batch multiple docs per call for speed? | **DECIDED 2026-07-10: adopt batched (with guardrails)** — audit found 0 attribution bleed / 0 hallucinated quotes / no coverage loss. NOT for speed (latency is output-bound; same tokens either way) but for fewer requests + one coherent evidence table. Guardrails in the agreed-design section. |
| OQ5 | Summarise thinking vs no-thinking. | **Next concrete test.** |
| OQ6 | "resolved : unresolved" ratio — exact definition over the 4-state {Supported/Refuted/Conflicting/Unsupported}. Is "unresolved" = Unsupported only? Is the misinfo signal the Refuted+Conflicting share? | Pin down before the NewsGuard correlation. |
| OQ7 | Early-exit vs full ratio. | **DECIDED: eval resolves ALL checkworthy claims to Supported/Refuted/Conflicting/Unsupported (no early-exit); early-exit nudge is production-only.** |
| OQ12 | Summarize prompt framing: what it sees (claim only / +post / +all claims) and whether it folds partial synthesis (per-claim stance) into the read step. | Research subagent running; then the summarize test. |
| OQ8 | Source-exclusion granularity: whole outlet domain vs just the specific article URL. | Lean whole domain (self-citation is still circular). |
| OQ9 | Attribution claims: true-attribution-of-false-content → contextual nudge, not a false-flag. | Deferred; first cut treats a validated attribution as pass. |
| OQ10 | DeepSeek/DeepInfra latency (30-60s/call, worse under concurrency) at corpus scale (~20,800 posts × several calls). Is the loop fast enough? | **RESOLVED 2026-07-10** (probe agent, `scratchpad/probe.log`): latency = OUTPUT tokens ÷ ~10-16 tok/s (DeepSeek decode rate; prefill free, json-mode free, no proxy issue). Qwen3-235B-Instruct decodes **28 tok/s**, more concise (1,640 vs 2,662+ tok on the identical payload), cheaper output ($0.10 vs $0.18/M): 59s vs 176-288s/call. → **DeepSeek for extraction (short outputs), Qwen for long-output steps (READ/STEP)**. Timeout policy: stream + TTFT ~30s + inter-chunk idle ~30s; retry only connect/5xx/stall, NEVER plain slow (re-bills a doomed generation). max_tokens cap hits = truncated JSON — treat finish_reason=length as data error. Concurrency: DeepSeek capped ~2-4 workers (weakest lever); Qwen ceiling untested (small ramp before relying on it). 10k posts: DeepSeek loop 35-80d infeasible; Qwen batched ~1.5-4d feasible. |
| OQ11 | Cache schema details (fields, keying, recheck policy). | Spec when we build the cache. |
| OQ13 | Pointer-READ accuracy: does the model cite the RIGHT segment IDs (vs vaguely-related ones)? | **MEASURED 2026-07-10 vs 3-Opus-rater consensus gold** (5 posts, 81 cells, World Cup post dropped as non-checkworthy; rater reliability: evidence-presence 0.951 / seg Jaccard 0.814 / stance 0.823). Best condition **per-doc × claims-both × 2500: seg F1 0.775, buffered(±1) recall 0.821, coverage 35/39, false-evidence 2/42** — at the rater-agreement ceiling for pointing; beats LongCite's ~70 (softer metric). Error anatomy: 10/11 stance errors are neutral→supports (over-crediting partial/adjacent evidence); all 4 missed cells gold-NEUTRAL background; every supports/refutes gold cell found. Residual fix: generalize the partial-evidence prompt rule + STEP adjudicates from resolved text. Batched collapses at length (precision 0.29-0.46 at 2000+, over-citation) → **OQ4 CLOSED: per-doc**. No length knee through 2500 (visible-window recall flat ~0.8 at every cap; recall gains = more article visible) → OQ14 validated. Scores: `scratchpad/pointer_scores.json`, scorer `score_pointer.py`. |
| OQ14 | How much article context before comprehension / segment-ID pointing degrades? Keyword selection judged too restrictive; prefill is ~free, so the binding constraint is pointing accuracy. | **RESEARCHED 2026-07-10** → `pointer_read_context_brief.md`. Pointer scheme validated by LongCite (sentence numbering + span output; sentence-level > chunks, spans > single sentences; ceiling ~70 F1 even for GPT-4o). NoLiMa: NON-lexical evidence matching degrades fast (<85% of baseline by ~2K tok for open models) → keyword-slicing wrong (drops non-lexical evidence) AND context-dumping wrong; use CONTIGUOUS full article capped ~2,500 tok/doc, windowing above. Layout: articles first, POST+claims+instruction last. Per-doc > batched (multi-doc interference + candidate count). DeepSeek's length-knee unpublished (NIAH-only, non-predictive) → our gold-set sweep must measure it. |

## Implementation checklist (2026-07-10)

All agreed changes, grouped by component. Check off as implemented.

**Preprocess (non-LLM, scrape → READ prep — one function):**
- [ ] Unicode normalize at scrape time (curly quotes → straight, collapse whitespace)
- [ ] Boilerplate line strip (newsletter / share / "Read more" regexes — cheap ones only)
- [ ] Article inclusion: **contiguous full article capped ~2,500 tok/doc** (windowing above, never
      keyword extraction) — per OQ14 research (`pointer_read_context_brief.md`). Cap value to be
      validated in the re-test sweep (DeepSeek's length-knee is unpublished).
- [ ] Publication date from trafilatura metadata / Serper, not the model
- [ ] Sentence-level segment numbering `[S14]` for pointer-READ

**Search / selection (non-LLM):**
- [ ] Programmatic read selection: NewsGuard rank + freshness + domain dedupe; walk all 10 hits until 3 good scrapes (replaces the controller READ round-trip)

**Loop architecture (LLM calls, `1 + 2R` shape):**
- [ ] OPEN = prioritized opening query (subsumes PLAN; prompt tested 2026-07-09, non-thinking)
- [ ] READ = claim-aware call with **pointer output**: `{claim_id, segs, stance}` — relevant-only
      (no relevant=false padding), no summary field, compound-claim guardrail (partial → neutral).
      Script resolves seg IDs → verbatim text for the evidence table, **with a ±1-sentence buffer**
      (Daniel, 2026-07-10): converts the dominant citation-error class (boundary/clipped-antecedent,
      off-by-one IDs) at zero output cost; cannot fix wrong-region pointers. Re-test scores BOTH
      raw segment F1 (diagnostic, LongCite-comparable, expect ~70) AND buffered window recall
      (operational — what STEP actually receives; estimated 80s, unpublished, the sweep measures it).
      **Batched vs per-doc RE-OPENED** (Daniel, 2026-07-10): pointer output makes output tokens tiny
      either way, so the trade is request-count/concurrency headroom (batched) vs stance discipline +
      failure isolation (per-doc; the McGraw over-claim happened with a weak source next to strong
      ones in-call). A/B in the pointer re-test; OQ4's "decided" is downgraded to pending.
- [ ] STEP = ledger-update + control over the compact evidence table (verdict-fold: PENDING Daniel;
      ledger LLM-judged vs rules+fallback: PENDING Daniel)
- [ ] Exa escalation with thinking query-gen, final round only
- [ ] `misinfo` → `nudge` rename (score_vtext.py already uses pred_nudge)

**Models & transport:**
- [x] Model: **DeepSeek-V4-Flash everywhere — Qwen ruled out (Daniel, 2026-07-10).** Implication:
      at 10-16 tok/s decode, pointer-READ's output reduction is load-bearing, not an optimization
      (~200 tok ≈ 15-20s/call).
- [x] Streaming + TTFT ~30s / inter-chunk idle ~30s timeouts; retry ONLY connect/5xx/stall, never
      "slow" — `pipeline/pools.py` (LLMPool; fake-clock tested in `tests/test_pools.py`)
- [x] `finish_reason=length` treated as data error — `pools.py` (LENGTH_CAP, never retried)
- [x] Concurrency — SUPERSEDED by the 2026-07-10 ramp (flat latency to 192 workers, 0 429s):
      pooled clients own all concurrency (`pipeline/pools.py`, LLM pool 150 / serper 8 / exa 3
      docs-only / scrape 32 + 2-per-domain); harness `pipeline/harness.py` runs K posts with
      jsonl-shard checkpointing + progress/stall alarm. Pipeline code stays a plain sequential
      async fn — never re-tune concurrency on pipeline changes.

**Eval / validation (unchanged decisions, listed for completeness):**
- [ ] 4-state per-claim resolution, no early-exit (eval); early-exit nudge production-only
- [ ] OQ6 resolved:unresolved ratio definition before the NewsGuard correlation
- [ ] Cache schema (OQ11)
