"""Classical reference-free metrics for (post, claim) pairs via HF Inference API.

See ``classical_metrics_brief.md`` for the recommendation rationale. Three
metrics, all served by `hf-inference` (no local heavy compute):

  * ``nli_score``     — P(entail) - P(contradict) from
                        ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli``
                        with the post as premise and the claim as hypothesis.
  * ``cosine_minilm`` — cosine similarity between post and claim embeddings from
                        ``sentence-transformers/all-MiniLM-L6-v2``.
  * ``ner_overlap``   — |entities(post) ∩ entities(claim)| / max(|entities(claim)|, 1)
                        using ``dslim/bert-base-NER``. Lowercased, type-agnostic.

Output schema (one row per (post_id, claim_index)), joinable with
``judgments_per_claim_n{N}.parquet``:

    post_id, claim_index, claim_text,
    nli_entailment, nli_neutral, nli_contradiction, nli_score,
    cosine_minilm,
    ner_post_count, ner_claim_count, ner_overlap,
    latency_s, error

Usage::

    uv run python -m eval.scripts.extraction_grading.remote_metrics --smoke
    uv run python -m eval.scripts.extraction_grading.remote_metrics \\
        --extractions data/extractions_n50.parquet \\
        --posts       data/posts_n50.parquet
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import polars as pl
import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NER_MODEL = "dslim/bert-base-NER"

HF_BASE = "https://router.huggingface.co/hf-inference/models"

# The hf-inference router serves task-specific endpoints under
# /models/{model}/pipeline/{task}. The plain /models/{model} default-pipeline
# is sometimes wrong (e.g. all-MiniLM-L6-v2 defaults to sentence-similarity
# instead of feature-extraction).
NLI_URL = f"{HF_BASE}/{NLI_MODEL}/pipeline/text-classification"
EMBED_URL = f"{HF_BASE}/{EMBED_MODEL}/pipeline/feature-extraction"
NER_URL = f"{HF_BASE}/{NER_MODEL}/pipeline/token-classification"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"

OUTPUT_SCHEMA = {
    "post_id": pl.String,
    "claim_index": pl.Int64,
    "claim_text": pl.String,
    "nli_entailment": pl.Float64,
    "nli_neutral": pl.Float64,
    "nli_contradiction": pl.Float64,
    "nli_score": pl.Float64,
    "cosine_minilm": pl.Float64,
    "ner_post_count": pl.Int64,
    "ner_claim_count": pl.Int64,
    "ner_overlap": pl.Float64,
    "latency_s": pl.Float64,
    "error": pl.String,
}

SMOKE_PAIRS: list[tuple[str, str]] = [
    # Faithful extraction.
    (
        "Breaking: NASA confirms the Perseverance rover collected its 20th rock "
        "sample from Jezero crater today.",
        "NASA's Perseverance rover collected a 20th rock sample from Jezero crater.",
    ),
    # Entity substitution (should tank NLI + NER).
    (
        "President Macron announced new climate funding of €5 billion at COP29 "
        "in Baku yesterday.",
        "President Biden announced new climate funding of $5 billion at COP29.",
    ),
    # No factual claim in the post — extractor hallucinated one.
    (
        "ugh monday mornings :( need coffee asap before this meeting",
        "Mondays cause a 30% drop in workplace productivity.",
    ),
    # Faithful but heavily compressed.
    (
        "Just read that the FDA approved Eli Lilly's new Alzheimer's drug "
        "donanemab after a 10-month review. Stock is up 8% premarket.",
        "The FDA approved Eli Lilly's Alzheimer's drug donanemab.",
    ),
    # Opinion / non-checkworthy.
    (
        "this new Marvel movie was honestly the worst superhero film of the decade",
        "The new Marvel movie is the worst superhero film of the decade.",
    ),
]


# ---------------------------------------------------------------------------
# HF Inference API helpers


def _hf_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN missing in environment / .env")
    return {"Authorization": f"Bearer {token}"}


def _hf_post(url: str, payload: dict, *, max_retries: int = 5,
             timeout: int = 60) -> object:
    """POST with retry/backoff on 429/503/5xx and cold-start ``estimated_time``."""
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=_hf_headers(), json=payload,
                              timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503) or r.status_code >= 500:
                # 503 with estimated_time means model is loading.
                try:
                    body = r.json()
                except Exception:  # noqa: BLE001
                    body = {}
                wait = float(body.get("estimated_time", delay)) if isinstance(body, dict) else delay
                time.sleep(min(wait, 30))
                delay = min(delay * 2, 30)
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError(f"HF API failed after {max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# Individual metric calls


def nli_pair(post: str, claim: str) -> dict[str, float]:
    """Returns {entailment, neutral, contradiction, score} probabilities.

    The HF text-classification endpoint accepts ``{"text": premise,
    "text_pair": hypothesis}`` for sentence-pair NLI cross-encoders.
    """
    payload = {
        "inputs": [{"text": post, "text_pair": claim}],
        "parameters": {"top_k": None},
        "options": {"wait_for_model": True},
    }
    out = _hf_post(NLI_URL, payload)
    # Response shape: [[{label, score}, ...]] (batched-of-one).
    items = out[0] if (isinstance(out, list) and out and isinstance(out[0], list)) else out
    scores = {item["label"].lower(): float(item["score"]) for item in items}
    ent = scores.get("entailment", 0.0)
    neu = scores.get("neutral", 0.0)
    con = scores.get("contradiction", 0.0)
    return {
        "entailment": ent,
        "neutral": neu,
        "contradiction": con,
        "score": ent - con,
    }


def embed(text: str) -> list[float]:
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    out = _hf_post(EMBED_URL, payload)
    # all-MiniLM-L6-v2 via feature-extraction returns flat list[float].
    if isinstance(out, list) and out and isinstance(out[0], list):
        return out[0]  # batched-of-one
    return out  # type: ignore[return-value]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def ner_entities(text: str) -> set[str]:
    payload = {
        "inputs": text,
        "parameters": {"aggregation_strategy": "simple"},
        "options": {"wait_for_model": True},
    }
    out = _hf_post(NER_URL, payload)
    # Response: [{entity_group, score, word, start, end}, ...]
    return {item["word"].strip().lower() for item in out if item.get("word")}


# ---------------------------------------------------------------------------
# Per-pair scoring


def score_pair(post_id: str, claim_index: int, post: str, claim: str) -> dict:
    t0 = time.time()
    row = {
        "post_id": post_id,
        "claim_index": claim_index,
        "claim_text": claim,
        "nli_entailment": 0.0,
        "nli_neutral": 0.0,
        "nli_contradiction": 0.0,
        "nli_score": 0.0,
        "cosine_minilm": 0.0,
        "ner_post_count": 0,
        "ner_claim_count": 0,
        "ner_overlap": 0.0,
        "latency_s": 0.0,
        "error": None,
    }
    try:
        nli = nli_pair(post, claim)
        row["nli_entailment"] = nli["entailment"]
        row["nli_neutral"] = nli["neutral"]
        row["nli_contradiction"] = nli["contradiction"]
        row["nli_score"] = nli["score"]

        e_post = embed(post)
        e_claim = embed(claim)
        row["cosine_minilm"] = cosine(e_post, e_claim)

        ents_post = ner_entities(post)
        ents_claim = ner_entities(claim)
        row["ner_post_count"] = len(ents_post)
        row["ner_claim_count"] = len(ents_claim)
        if ents_claim:
            row["ner_overlap"] = len(ents_post & ents_claim) / len(ents_claim)
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
    row["latency_s"] = round(time.time() - t0, 3)
    return row


# ---------------------------------------------------------------------------
# IO


def _load_existing(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=OUTPUT_SCHEMA)
    return pl.read_parquet(path)


def _infer_output(extractions_path: Path) -> Path:
    import re
    m = re.search(r"extractions_n(\d+)", extractions_path.stem)
    suffix = f"n{m.group(1)}" if m else extractions_path.stem
    return extractions_path.parent / f"classical_metrics_{suffix}.parquet"


def _explode_pairs(extractions: pl.DataFrame, posts: pl.DataFrame) -> list[dict]:
    """Yields {post_id, claim_index, post, claim} rows for every extracted claim."""
    joined = extractions.join(
        posts.select(["post_id", "text"]), on="post_id", how="inner"
    )
    pairs: list[dict] = []
    for r in joined.iter_rows(named=True):
        for i, claim in enumerate(r["claims"] or []):
            pairs.append({
                "post_id": r["post_id"],
                "claim_index": i,
                "post": r["text"],
                "claim": claim,
            })
    return pairs


# ---------------------------------------------------------------------------
# Entry points


def run_smoke(out_path: Path) -> pl.DataFrame:
    print(f"[smoke] scoring {len(SMOKE_PAIRS)} hard-coded (post, claim) pairs")
    rows = []
    for i, (post, claim) in enumerate(SMOKE_PAIRS):
        print(f"  [{i+1}/{len(SMOKE_PAIRS)}] post={post[:60]!r}...")
        rows.append(score_pair(post_id=f"smoke_{i:02d}", claim_index=0,
                               post=post, claim=claim))
    df = pl.DataFrame(rows, schema=OUTPUT_SCHEMA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"[smoke] wrote {out_path}")
    print(df.select([
        "claim_index", "nli_score", "cosine_minilm",
        "ner_overlap", "error",
    ]))
    return df


def run_full(extractions_path: Path, posts_path: Path, out_path: Path) -> None:
    extractions = pl.read_parquet(extractions_path)
    posts = pl.read_parquet(posts_path)
    pairs = _explode_pairs(extractions, posts)
    existing = _load_existing(out_path)
    done = set(
        zip(existing["post_id"].to_list(), existing["claim_index"].to_list())
    )
    todo = [p for p in pairs if (p["post_id"], p["claim_index"]) not in done]
    print(f"[full] {len(pairs)} pairs total, {len(done)} done, {len(todo)} todo")
    new_rows: list[dict] = []
    for i, p in enumerate(todo):
        if i and i % 25 == 0:
            print(f"  [{i}/{len(todo)}]")
        new_rows.append(score_pair(p["post_id"], p["claim_index"],
                                   p["post"], p["claim"]))
        # Incremental flush every 50 rows so we don't lose a long run.
        if len(new_rows) >= 50:
            existing = pl.concat([existing, pl.DataFrame(new_rows,
                                                          schema=OUTPUT_SCHEMA)])
            existing.write_parquet(out_path)
            new_rows = []
    if new_rows:
        existing = pl.concat([existing, pl.DataFrame(new_rows,
                                                      schema=OUTPUT_SCHEMA)])
        existing.write_parquet(out_path)
    print(f"[full] wrote {out_path} (rows={len(existing)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="Run on 5 hard-coded pairs and exit.")
    ap.add_argument("--extractions", type=Path,
                    default=DEFAULT_DATA_DIR / "extractions_n50.parquet")
    ap.add_argument("--posts", type=Path,
                    default=DEFAULT_DATA_DIR / "posts_n50.parquet")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output parquet (default inferred from --extractions).")
    args = ap.parse_args()

    if args.smoke:
        out = args.out or (DEFAULT_DATA_DIR / "classical_metrics_smoke.parquet")
        run_smoke(out)
        return

    out = args.out or _infer_output(args.extractions)
    run_full(args.extractions, args.posts, out)


if __name__ == "__main__":
    main()
