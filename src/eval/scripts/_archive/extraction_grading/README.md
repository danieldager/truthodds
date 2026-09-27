# extraction_grading

Distribution analysis of the `ours` claim-extraction prompt on representative modern social-media posts (Bluesky). Plan: `~/.claude/plans/this-is-good-news-snappy-perlis.md`.

## Planned layout

| File | Purpose | Workstream |
|---|---|---|
| `load_posts.py` | Bluesky data loader — HF `alpindale/two-million-bluesky-posts` → EN filter → sample → parquet with stable `post_id` | T2 |
| `prompts.py` | Extraction prompt (production-style, multi-claim JSON output) | T4 |
| `extract.py` | Extraction runner — one LLM call per post, resumable | T4 |
| `judge.py` | Per-claim Likert + per-post detection-correctness judge | T5 |
| `post_features.py` | Lexical / structural / sentiment / semantic feature extractor | T3 |
| `remote_metrics.py` | Classical NLP metrics via HF Inference API (NLI, embeddings, etc.) | S-classical |
| `grade.py` | Orchestration — joins all parquets into wide `graded_n{N}` outputs | S-analysis |
| `analyze.py` | Distribution analyses, correlations, failure modes | S-analysis |
| `data/` | Parquet outputs (gitignored) | — |
| `classical_metrics_brief.md` | Literature survey + recommendation for classical NLP metrics | S-classical |
| `analysis_brief.md` | Literature survey + recommendation for analysis methodology | S-analysis |

## Models

- Extraction: `openai/gpt-oss-120b` (Groq)
- Judge: `qwen/qwen3-32b` (Groq, different family from extractor)
- Classical metrics: TBD per `classical_metrics_brief.md`

## Schema

Canonical `post_id` is `sha1("bsky:" + source_id)[:16]`, stable across runs. All parquets are joinable on this key. See plan Part 2 for full schemas.
