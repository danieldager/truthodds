# feed_study — claim detection + extraction, rebuilt

A from-scratch replacement for the 4-prong "misinformation-candidate" cascade
(now in `eval/scripts/_archive/`). Grounded in the SOTA review in
[`docs/claim_extraction_sota_review.md`](docs/claim_extraction_sota_review.md).

**This phase measures AGREEMENT, not accuracy.** There is no gold set yet — see
the TODO at the bottom. We compare what different methods *decide*, not whether
they are *right*.

## What changed vs the archived cascade

- **No more 4-prong criterion.** The old detector fused verifiability with a
  public-consequence / news-substitutable scope gate. Detection here is **pure
  verifiability** (Claimify Selection). Prioritization by harm is a **separate**
  FABLE pass, not baked into detection. (Consequence: the has-claim rate is much
  higher — it behaves like flat-pass detection, ~40%+, not the cascade's 19%.)
- **Claimify-style 3-stage extraction** replaces the single-pass extractor.
- **Two model families run the same pipeline** (gpt-oss-120b + qwen3-32b) so we
  can measure inter-model agreement on detection and normalization.
- **FABLE is kept**, lifted self-contained into `fable.py`, and now runs in two
  modes (on raw posts as a front-filter, and on normalized claims).

## Components

| File | Role |
|---|---|
| `prompts_claimify.py` | The 3 stage prompts + parsers: **Selection** (verifiable?), **Disambiguation** (decontextualize; hybrid/confidence-flagged abstain), **Decomposition** (molecular, minimal-but-standalone claims). |
| `run_claimify.py` | Runs the pipeline over the feed, one row/post, model-parameterized + resumable. |
| `fable.py` | FABLE harm rubric (5 dims, 1-5), self-contained. `build_fable_post_messages` = raw-post mode; `build_fable_messages` = per-claim mode. Identical prompt both ways. |
| `run_fable.py` | `--mode post` (front-filter) or `--mode claim` (on a claimify run's claims). |
| `agreement.py` | Pairwise detection agreement (% + Cohen's κ + confusion) across runs. Extension point for the non-LLM detectors. |
| `detect_nonllm.py` | **NOT BUILT YET** — ClaimBuster + CLEF XLM-R. Blocked, see below. |
| `data/posts_x862.parquet` | The canonical 2026-06-01 X capture (copied from the cascade; gitignored). |
| `docs/` | SOTA reviews + system map. |
| `results/` | Agreement reports. |

## Run order

```bash
# from src/
# 1. Claimify pipeline, both model families (the mirror)
uv run python -m eval.scripts.feed_study.run_claimify -m openai/gpt-oss-120b
uv run python -m eval.scripts.feed_study.run_claimify -m qwen/qwen3-32b

# 2. Detection agreement between the families
uv run python -m eval.scripts.feed_study.agreement

# 3. FABLE — front-filter on raw posts, and on the extracted claims
uv run python -m eval.scripts.feed_study.run_fable --mode post  -m qwen/qwen3-32b
uv run python -m eval.scripts.feed_study.run_fable --mode claim -m qwen/qwen3-32b \
    --claims eval/scripts/feed_study/data/claimify_gpt-oss-120b.parquet
```

## Design decisions (agreed with the user, 2026-06-02)

- **Detector = non-LLM, benchmark BOTH** ClaimBuster + CLEF XLM-R. Idea: if a
  cheap non-LLM detector agrees closely with Claimify's Selection, the LLM only
  needs to run for *extraction* on posts the detector flags.
- **LLM pipeline = Claimify clone**, open-source models, qwen mirror for
  agreement. Prompts reconstructed (not copied) and adapted for social posts.
- **Ambiguity = hybrid / confidence-flagged**: resolve when confident, else
  abstain with `can_disambiguate=false` + a `confidence` level.
- **Normalization validation = Claimify's 3 metrics**: Entailment (≈ fidelity),
  Decontextualization, Coverage (new; needs a reference set).

## Open blockers (for the non-LLM half)

1. **`torch` / `transformers` not installed.** Needed for the XLM-R detector and
   a local ClaimBuster model (and for any embedding-based normalization
   similarity). Install is heavyweight — left as a deliberate decision.
2. **ClaimBuster access undecided** — public API (needs a free key; none in
   `src/.env`) vs a local HuggingFace reimplementation.

## TODO — gold set (deferred, Daniel)

Manually annotate ~50–100 feed posts for check-worthiness + reference claims.
Until then everything here is **agreement-only**; a gold set unlocks real
detection precision/recall and the Coverage metric. Reuses the 50-claim
calibration originally planned for the cascade.
