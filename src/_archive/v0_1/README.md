# `_archive/v0_1/` — Pipeline v0.1 (reference only)

Superseded by pipeline v0.2 on 2026-05-18. See `docs/pipeline.md` for the current spec and `src/pipeline/` for the current implementation.

**This code does not run as-is.** It references constants in `src/config.py` (e.g. `MAX_RETRIEVAL_ROUNDS`, `FIRE_*`, `RERANK_*`, `MMR_*`, `CRED1_*`) that were stripped when v0.2 trimmed `config.py`. Kept here only as reference for design decisions and prompts that may be useful when filling out v0.2 module bodies.

## Contents

| Path | Role in v0.1 |
|---|---|
| `pipeline/claim_extraction.py` | Claim decomposition + verifying-question generation prompts. The decomposition prompt is being lifted into `src/pipeline/extract.py`. |
| `pipeline/retrieval.py` | Serper + FCTAPI + Wikipedia parallel retrieval, cross-encoder rerank, MMR diversity, intra-doc BM25 chunking. Useful reference for the Serper/Trafilatura plumbing being lifted into `src/pipeline/search.py`. |
| `pipeline/verification.py` | Likert-per-label verifier prompt. Useful reference for the synthesis prompt in `src/pipeline/verify.py`. |
| `pipeline/iterative.py` | FIRE-style follow-up loop (post-verifier). Superseded by the synthesis-driven loop in v0.2. |
| `pipeline/misleadingness.py` | 4-step claim-stripping algorithm. Parked in v0.2; may return as a post-Tier-3 pass. |
| `pipeline/aggregation.py` | Hard-priority post-level aggregation. Out of scope for v0.2. |
| `pipeline/models.py` | Pydantic models. Partly inspires `src/pipeline/models.py` but does not match it. |
| `main.py` | Single-post CLI driver. |
| `compare.py` | Head-to-head verifier-model comparison harness. |
| `evaluate.py` | AVeriTeC eval runner with `--approach {baseline,fire,sc}` and `--verifier` flags. The CSV-logging pattern (`eval_results.csv` / `eval_claims.csv`) is worth keeping in mind when wiring v0.2's eval hooks. |

## Why kept

- Prompt strings are non-trivial; reusing them saves iteration time.
- The retrieval module's per-domain rate limiting, user-agent rotation, and Trafilatura usage are battle-tested.
- The `verify_claim_sc` self-consistency wrapper and the calibration discoveries (ECE 0.35, CE recall 0%) are referenced in `clog/110526.md` and the corresponding eval artifacts in `docs/archive/v0_1/`.
