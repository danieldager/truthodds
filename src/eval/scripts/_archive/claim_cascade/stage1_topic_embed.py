"""Stage 1 — local multilingual topic classifier over the posts parquet.

Input : data/posts_x862.parquet         (Stage 0 output)
Output: data/topics_x862.parquet        (one row per post_id)

Each post's reader-visible ``text`` is embedded with the shared multilingual
MiniLM model (``pipeline.embedding.embed`` — the SAME model used elsewhere in
the pipeline, NOT the English-only model in post_features.py). We score it
against every topic anchor (``anchors.build_anchor_matrix``) by cosine
similarity (= dot product, both are L2-normalized), take the argmax as
``top_topic``, and flag ``in_scope_embed`` when the top topic is an in-scope
label and its cosine clears ``anchors.TAU`` (0.0 to start — the LLM scope
stage is the real gate; this layer favors recall).

Posts with ``char_len < anchors.MIN_CHARS`` (essentially empty / media-only)
are not embedded: ``top_topic="empty"``, ``in_scope_embed=False``, scores and
embedding are zero vectors (so ``topic_scores`` length still equals the label
count for every row).

This step is a logged classifier, not a filter — the full score vector and the
384-d post embedding are persisted for reuse in later cascade stages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipeline.embedding import embed  # noqa: E402  (shared multilingual model)
from config import EMBEDDING_DIM  # noqa: E402

from eval.scripts.claim_cascade import anchors  # noqa: E402

DEFAULT_INPUT = Path(__file__).parent / "data" / "posts_x862.parquet"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "topics_x862.parquet"

EMPTY_TOPIC = "empty"


def _schema() -> dict:
    return {
        "post_id": pl.String,
        "top_topic": pl.String,
        "top_cosine": pl.Float64,
        "in_scope_embed": pl.Boolean,
        "margin": pl.Float64,
        "topic_scores": pl.List(pl.Float64),
        "topic_labels": pl.List(pl.String),
        "post_embedding": pl.List(pl.Float32),
    }


def classify(
    text: str,
    char_len: int,
    labels: list[str],
    matrix: np.ndarray,
    in_scope: set[str],
) -> dict:
    n = len(labels)
    if char_len < anchors.MIN_CHARS:
        zeros = [0.0] * n
        return {
            "top_topic": EMPTY_TOPIC,
            "top_cosine": 0.0,
            "in_scope_embed": False,
            "margin": 0.0,
            "topic_scores": zeros,
            "topic_labels": labels,
            "post_embedding": [0.0] * EMBEDDING_DIM,
        }

    vec = embed(text)  # (384,) L2-normalized
    scores = matrix @ vec  # (n_labels,) cosine similarities
    order = np.argsort(scores)[::-1]
    top_i = int(order[0])
    top_cos = float(scores[top_i])
    second = float(scores[order[1]]) if n > 1 else 0.0
    top_topic = labels[top_i]
    return {
        "top_topic": top_topic,
        "top_cosine": top_cos,
        "in_scope_embed": bool(top_topic in in_scope and top_cos >= anchors.TAU),
        "margin": round(top_cos - second, 6),
        "topic_scores": [round(float(s), 6) for s in scores],
        "topic_labels": labels,
        "post_embedding": vec.astype(np.float32).tolist(),
    }


def run(input_path: Path) -> tuple[list[str], pl.DataFrame]:
    posts = pl.read_parquet(input_path)
    labels, matrix = anchors.build_anchor_matrix()
    in_scope = set(anchors.IN_SCOPE)

    rows: list[dict] = []
    texts = posts["text"].to_list()
    char_lens = posts["char_len"].to_list()
    pids = posts["post_id"].to_list()
    for i, (pid, text, clen) in enumerate(zip(pids, texts, char_lens), 1):
        r = classify(text, clen, labels, matrix, in_scope)
        r["post_id"] = pid
        rows.append(r)
        if i % 100 == 0 or i == len(pids):
            print(f"  embedded {i}/{len(pids)}")

    return labels, pl.DataFrame(rows, schema=_schema())


def _eyeball(posts: pl.DataFrame, topics: pl.DataFrame, langs: list[str]) -> None:
    """Print the most/least in-scope posts per major language to vet anchors."""
    joined = posts.join(topics, on="post_id")
    for lang in langs:
        sub = joined.filter(pl.col("lang") == lang)
        if sub.is_empty():
            continue
        print(f"\n{'=' * 70}\nLANG = {lang}  (n={len(sub)})")
        for tag, frame in (
            ("TOP-5 IN-SCOPE", sub.filter("in_scope_embed").sort("top_cosine", descending=True)),
            ("TOP-5 OUT-SCOPE", sub.filter(~pl.col("in_scope_embed")).sort("top_cosine", descending=True)),
        ):
            print(f"  --- {tag} ---")
            for row in frame.head(5).iter_rows(named=True):
                snippet = row["own_text"].replace("\n", " ")[:90]
                print(f"   [{row['top_topic']:<22} {row['top_cosine']:.3f}] {snippet}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--langs", nargs="+", default=["en", "fr", "es", "ar"])
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input parquet not found (run Stage 0 first): {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    labels, topics = run(args.input)
    topics.write_parquet(args.output)
    print(f"\nwrote {len(topics)} rows to {args.output}")

    # --- verification summary ---
    n_in = topics.filter("in_scope_embed").height
    emb_lens = topics["post_embedding"].list.len()
    score_lens = topics["topic_scores"].list.len()
    print(f"rows: {len(topics)} | in_scope_embed: {n_in} ({100*n_in/len(topics):.0f}%)")
    print(f"post_embedding len: min {emb_lens.min()} max {emb_lens.max()} (expect 384)")
    print(f"topic_scores len == labels ({len(labels)}): "
          f"{'PASS' if score_lens.min()==len(labels) and score_lens.max()==len(labels) else 'FAIL'}")
    print("\ntop_topic distribution:")
    print(topics["top_topic"].value_counts(sort=True))

    posts = pl.read_parquet(args.input)
    _eyeball(posts, topics, args.langs)


if __name__ == "__main__":
    main()
