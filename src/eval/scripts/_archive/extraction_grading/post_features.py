"""Post-feature extraction for the extraction_grading harness.

Reads a posts parquet (Part 2 schema), computes ~25 features per post, writes
a features parquet keyed by ``post_id``.

Feature groups: lexical, structural, sentiment, semantic. One function per
group, glued via :func:`extract_all_features`.

CPU-light. Models (spaCy `en_core_web_sm`, sentence-transformers
`all-MiniLM-L6-v2`) are loaded once via lazy singletons.

CLI:
    uv run python -m eval.scripts.extraction_grading.post_features [INPUT_PARQUET]
        [-o OUTPUT_PARQUET] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import polars as pl

# Heavy imports done lazily inside singleton getters to keep --smoke startup
# moderate and to avoid import-time side effects on the test runner.

# ---------------------------------------------------------------------------
# Singletons for heavy resources
# ---------------------------------------------------------------------------

_NLP = None
_EMBEDDER = None
_VADER = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy

        # Disable parser+lemmatizer; we only need NER + tokenizer.
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _NLP


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _get_vader():
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _VADER = SentimentIntensityAnalyzer()
    return _VADER


# ---------------------------------------------------------------------------
# Regexes (compiled once)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+")
_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_PUNCT_RE = re.compile(r"[^\w\s]")
# Broad emoji unicode-range fallback; the `emoji` package is preferred.
_EMOJI_FALLBACK_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002700-\U000027bf"
    "\U00002600-\U000026ff"
    "]",
    flags=re.UNICODE,
)


# ---------------------------------------------------------------------------
# Lexical
# ---------------------------------------------------------------------------


def lexical_features(text: str) -> dict[str, Any]:
    import textstat

    tokens = text.split()
    n_tokens = len(tokens)
    n_chars = len(text)

    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    n_sentences = len(sentences)

    avg_word_length = (
        sum(len(t) for t in tokens) / n_tokens if n_tokens else 0.0
    )
    lexical_diversity = (
        len(set(t.lower() for t in tokens)) / n_tokens if n_tokens else 0.0
    )

    # textstat is robust to empty strings (returns 0 / known sentinels).
    flesch_reading_ease = float(textstat.flesch_reading_ease(text)) if text else 0.0
    flesch_kincaid_grade = float(textstat.flesch_kincaid_grade(text)) if text else 0.0

    n_uppercase_words = sum(1 for t in tokens if t.isupper() and len(t) > 1)
    alpha_chars = [c for c in text if c.isalpha()]
    caps_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars
        else 0.0
    )

    punct_count = len(_PUNCT_RE.findall(text))
    punctuation_density = punct_count / n_chars if n_chars else 0.0
    exclamation_count = text.count("!")
    question_count = text.count("?")

    return {
        "n_chars": n_chars,
        "n_tokens": n_tokens,
        "n_sentences": n_sentences,
        "avg_word_length": float(avg_word_length),
        "lexical_diversity": float(lexical_diversity),
        "flesch_reading_ease": flesch_reading_ease,
        "flesch_kincaid_grade": flesch_kincaid_grade,
        "n_uppercase_words": n_uppercase_words,
        "caps_ratio": float(caps_ratio),
        "punctuation_density": float(punctuation_density),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
    }


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def _count_emojis(text: str) -> int:
    try:
        import emoji

        # emoji>=2 returns a list of dicts via emoji_list.
        return len(emoji.emoji_list(text))
    except Exception:
        return len(_EMOJI_FALLBACK_RE.findall(text))


def structural_features(text: str) -> dict[str, Any]:
    urls = _URL_RE.findall(text)
    hashtags = _HASHTAG_RE.findall(text)
    mentions = _MENTION_RE.findall(text)
    n_emojis = _count_emojis(text)

    stripped = text.lstrip()
    # is_repost: Bluesky doesn't use "RT @". We check the Twitter-style prefix
    # for cross-platform safety; for Bluesky-native reposts the loader should
    # mark this via a metadata flag in a future iteration.
    is_repost = stripped.startswith("RT @")
    # is_reply: starts with '@' (a mention as the first token). Metadata-based
    # reply_to is not yet plumbed through; fall back to the textual heuristic.
    is_reply = stripped.startswith("@")

    return {
        "has_url": bool(urls),
        "n_urls": len(urls),
        "has_hashtag": bool(hashtags),
        "n_hashtags": len(hashtags),
        "has_mention": bool(mentions),
        "n_mentions": len(mentions),
        "has_emoji": n_emojis > 0,
        "n_emojis": n_emojis,
        "is_repost": is_repost,
        "is_reply": is_reply,
    }


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------


def sentiment_features(text: str) -> dict[str, Any]:
    from textblob import TextBlob

    vader = _get_vader().polarity_scores(text)
    blob = TextBlob(text)
    return {
        "vader_compound": float(vader["compound"]),
        "vader_pos": float(vader["pos"]),
        "vader_neg": float(vader["neg"]),
        "vader_neu": float(vader["neu"]),
        "textblob_polarity": float(blob.sentiment.polarity),
        "textblob_subjectivity": float(blob.sentiment.subjectivity),
    }


# ---------------------------------------------------------------------------
# Semantic
# ---------------------------------------------------------------------------


def semantic_features(text: str) -> dict[str, Any]:
    nlp = _get_nlp()
    doc = nlp(text)
    ents = list(doc.ents)
    ent_types = sorted({e.label_ for e in ents})

    embedder = _get_embedder()
    # convert_to_numpy=True -> ndarray; tolist() for parquet-friendly list[float].
    emb = embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return {
        "n_named_entities": len(ents),
        # parquet handles list<string>, but the spec asks for json-encoded for
        # stability/portability with the downstream pandas analyses.
        "dominant_entity_types": json.dumps(ent_types),
        "post_embedding": emb.tolist(),
    }


# ---------------------------------------------------------------------------
# Glue
# ---------------------------------------------------------------------------


def extract_all_features(text: str) -> dict[str, Any]:
    """Compute all feature groups for one post.

    Returns a flat dict ready to be fed into a polars row.
    """
    out: dict[str, Any] = {}
    out.update(lexical_features(text))
    out.update(structural_features(text))
    out.update(sentiment_features(text))
    out.update(semantic_features(text))
    return out


# ---------------------------------------------------------------------------
# Parquet runner
# ---------------------------------------------------------------------------


def run(input_path: Path, output_path: Path) -> None:
    df = pl.read_parquet(input_path)
    if "post_id" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Input parquet {input_path} missing required columns post_id/text; "
            f"got {df.columns}"
        )

    # Warm singletons before the loop.
    _get_nlp()
    _get_embedder()
    _get_vader()

    rows: list[dict[str, Any]] = []
    for post_id, text in zip(df["post_id"].to_list(), df["text"].to_list()):
        feats = extract_all_features(text or "")
        feats["post_id"] = post_id
        rows.append(feats)

    # Put post_id first.
    out_df = pl.DataFrame(rows)
    cols = ["post_id"] + [c for c in out_df.columns if c != "post_id"]
    out_df = out_df.select(cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_path)
    print(f"wrote {len(out_df)} rows -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_SMOKE_POSTS = [
    "BREAKING: scientists confirm coffee adds 10 years to your life!!! 😱 https://example.com/news",
    "@alice you up? thinking about that thing we discussed earlier 🤔",
    "RT @newsbot: The unemployment rate dropped to 3.5% in October, per BLS data.",
    "honestly idk why anyone still uses twitter when bluesky exists #migration #bsky",
    "What if the moon is actually a hologram? Just asking questions.",
]


def _smoke() -> None:
    for i, post in enumerate(_SMOKE_POSTS):
        print(f"\n--- post {i} ---")
        print(post)
        feats = extract_all_features(post)
        # Truncate embedding for readability.
        emb = feats.pop("post_embedding")
        feats["post_embedding"] = f"<list[float] len={len(emb)}>"
        for k, v in feats.items():
            print(f"  {k}: {v}")


def _default_output_for(input_path: Path) -> Path:
    """Derive default output: data/posts_n{N}.parquet -> data/features_n{N}.parquet."""
    name = input_path.name
    if name.startswith("posts_"):
        out_name = "features_" + name[len("posts_") :]
    else:
        out_name = "features_" + name
    return input_path.parent / out_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("data/posts_n2000.parquet"),
        help="Input posts parquet (default: data/posts_n2000.parquet)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Override output path (default: derived from input)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run on 5 hard-coded examples, print to stdout, write nothing",
    )
    args = parser.parse_args()

    if args.smoke:
        _smoke()
        return

    output = args.output or _default_output_for(args.input)
    run(args.input, output)


if __name__ == "__main__":
    main()
