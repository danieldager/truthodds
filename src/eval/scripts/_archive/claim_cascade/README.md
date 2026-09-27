# Claim-filtering cascade on the X collection

A multi-stage filter cascade over the 862 captured X posts
(`fourcat/exports/x_collection_010626.ndjson`). Each stage writes its own
joinable, resumable parquet (keyed on `post_id`, claims also on `claim_index`)
so analysis never re-runs the LLM.

```
Stage 0  build posts        posts_x862.parquet        (862, no API)
Stage 1  topic embed        topics_x862.parquet       (862, local MiniLM)
Stage 2  LLM scope          scope_x862.parquet        (862, gpt-oss-120b)
         └─ gate            posts_inscope.parquet     (in_scope_llm rows)
Stage 3  extract claims     stage3_extractions.parquet (gated, gpt-oss-120b)
Stage 4  per-claim judge    stage4_judgments.parquet  (claims, qwen3-32b)   ← 4-prong
Stage 5  FABLE harm score   fable_x862.parquet        (claims, qwen3-32b)   ← parallel rubric
Stage 7  FCT cross-check    fct_x862.parquet          (check-worthy union)
Stage 6  join               graded_posts_x862 + graded_claims_x862
         analyze            results/funnel_report.md
```

Stages 4 and 7 use **qwen3-32b** and **Google Fact Check Tools**; Stages 4 and
5 run the *same model* on the *same claims* so the 4-prong vs. FABLE comparison
isolates the **rubric**, not the model.

## Run order

All commands run from `src/` (so `eval...` resolves on the path) with `uv`.

```bash
# Stage 0 — build posts (no API)
uv run python -m eval.scripts.claim_cascade.stage0_build_posts

# Stage 1 — topic embedding (local, no API)
uv run python -m eval.scripts.claim_cascade.stage1_topic_embed

# Stage 2 — LLM scope on all 862; writes scope_x862 + posts_inscope gate
uv run python -m eval.scripts.claim_cascade.stage2_scope_llm -m openai/gpt-oss-120b

# Stage 3 — extract claims on the in-scope gate (NO new code; reuse extract.py)
uv run python -m eval.scripts.extraction_grading.extract \
    -i eval/scripts/claim_cascade/data/posts_inscope.parquet \
    -o eval/scripts/claim_cascade/data/stage3_extractions.parquet \
    -m openai/gpt-oss-120b
```

### Stage 4 — per-claim judge (NO new code; reuse `extraction_grading/judge.py`)

The 4-prong `misinfo_candidate` rubric is exactly the per-claim judge already
validated at precision 0.868. Stage 4 is just that script pointed at the
cascade data dir, with the judge model and the per-claim output path made
explicit:

```bash
uv run python -m eval.scripts.extraction_grading.judge \
    -p eval/scripts/claim_cascade/data/posts_inscope.parquet \
    -e eval/scripts/claim_cascade/data/stage3_extractions.parquet \
    --per-claim-output eval/scripts/claim_cascade/data/stage4_judgments.parquet \
    -m qwen/qwen3-32b -w 12
```

Output (`stage4_judgments.parquet`, one row per `(post_id, claim_index)`):
`fidelity`, `decontextualized`, `verifiability` (1–5) + `misinfo_candidate`
(bool). Resumable — already-judged `(post_id, claim_index)` pairs are skipped.
`-p` is the gate so the join restricts to in-scope posts; `-e` supplies the
claim list per post.

```bash
# Stage 5 — FABLE harm rubric on the same claims (qwen3-32b)
uv run python -m eval.scripts.claim_cascade.stage5_fable -m qwen/qwen3-32b

# Stage 7 — Google FCT cross-check on the UNION of claims flagged check-worthy
#            by EITHER rubric (misinfo_candidate OR fable_checkworthy)
uv run python -m eval.scripts.claim_cascade.stage7_fct -w 6

# Stage 6 — join every stage into graded_posts_x862 + graded_claims_x862
uv run python -m eval.scripts.claim_cascade.stage6_grade

# Analysis — funnel, embedding↔LLM agreement, 4-prong vs FABLE
uv run python -m eval.scripts.claim_cascade.analyze_funnel
```

Each script takes `--data-dir` (default `./data`). Stage 7 also takes
`-w/--workers` (default 6; back off on FCT 429s) and `-n/--limit` for smoke
tests. All API/LLM stages are resumable.

## Stage 7 — FCT cross-check (detail)

Runs **only** on the union of check-worthy claims (Tier-2 short-circuit
measurement). `fct_x862.parquet`:

| column | meaning |
|---|---|
| `post_id`, `claim_index`, `claim_text` | claim key + text |
| `lang` | the post's Twitter `lang` (audit) |
| `fct_match` | true iff FCT returned ≥1 indexed ClaimReview |
| `fct_publisher` | top result's publisher site (or name) |
| `fct_rating` | top result's `textualRating` |
| `fct_url` | top result's review URL |
| `n_results` | number of claims returned by the query |
| `latency_s`, `error` | audit |

`languageCode` is derived from the post `lang` (`in`→`id`, `iw`→`he`;
`und`/`zxx`/`qXX` → omitted, search all languages).

## Stage 5 — FABLE rubric (detail)

The five dimensions are the **FABLE** misinformation-harm framework (Sehat et al.,
*"Misinformation as a Harm: Structured Approaches for Fact-Checking
Prioritization"*, CSCW 2024 / arXiv:2312.11678). `prompts_scope.FABLE_SYSTEM`
scores one claim (the post is shown for context, mirroring the Stage-4 judge so
the 4-prong-vs-FABLE comparison isolates the rubric, not the input) on:

| dim | F-A-B-L-E | high (5) means |
|---|---|---|
| `fragmentation` | Fragmentation | erodes trust in shared institutions / a whole community (NOT context-collapse) |
| `actionability` | Actionability | explicit call to act + logistics / identifying info enabling real harm |
| `believability` | Believability | the target audience would readily accept it as true |
| `spread_likelihood` | Likelihood of spread | reach / virality of the framing (kept distinct from harm) |
| `exploitativeness` | Exploitativeness | preys on fear / identity / a vulnerable audience |

`fable_x862.parquet` (one row per `(post_id, claim_index)`): the five 1–5 dims,
`fable_total` (their sum, 5–25), `fable_checkworthy` (`fable_total >=
--fable-threshold`, default 15), `raw_response`, `latency_s`, `error`.

The **Likert + sum + threshold is a bespoke scoring layer** — the paper uses
binary diagnostics and no composite score. `fable_checkworthy` is **derived in
code and recomputed across the whole table on every run**, so recalibrating is a
re-run with a new `--fable-threshold` (no LLM calls). Calibrate on the 50-claim
set. *Known limitation:* a pure sum misses concentrated high-harm claims (e.g.
`actionability=5` doxxing, `total<15`); at calibration consider an OR-gate
(`total>=T OR any dim==5 OR count(dim>=4)>=2`). All five raw dims are persisted,
so any such rule is computable post-hoc.

```bash
# recalibrate the bool only (no LLM calls — re-derives from stored totals):
uv run python -m eval.scripts.claim_cascade.stage5_fable --fable-threshold 17
```

First full run (2026-06-01): 862 scored → 208 in-scope → 163 has-claim / 410
claims → 59 FABLE-check-worthy @15 (130 @13, 25 @17). 0 errors across all stages.

## Development against stubs

Before Stages 0/1/2/5 land, `make_stubs.py` writes 5-row stub parquets for
every upstream schema into `data/_stub/`, so Stages 6/7 + analysis can be
exercised end-to-end:

```bash
uv run python -m eval.scripts.claim_cascade.make_stubs
uv run python -m eval.scripts.claim_cascade.stage7_fct       --data-dir eval/scripts/claim_cascade/data/_stub -w 4
uv run python -m eval.scripts.claim_cascade.stage6_grade     --data-dir eval/scripts/claim_cascade/data/_stub
uv run python -m eval.scripts.claim_cascade.analyze_funnel   --data-dir eval/scripts/claim_cascade/data/_stub \
    --results-dir eval/scripts/claim_cascade/data/_stub/results
```
