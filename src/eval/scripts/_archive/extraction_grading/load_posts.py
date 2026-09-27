"""Load a sample of English posts from the Bluesky HF dataset.

Dataset: `alpindale/two-million-bluesky-posts` (2.1M firehose posts, late 2024).

Schema written to `data/posts_n{N}.parquet`:
    post_id     : str        # stable sha1("bsky:" + source_id)[:16]
    text        : str        # raw post body
    source      : str        # "bluesky"
    source_id   : str        # the AT-URI (preferred — stable across re-downloads
                             # of the dataset, unlike HF row index which can shift
                             # if the dataset is re-shuffled or re-uploaded)
    lang        : str        # "en" after filter (column kept for future-proofing)
    char_len    : int
    created_at  : timestamp  # parsed from dataset's `created_at` ISO string

The dataset does NOT expose a per-row language column, so we use `langdetect`
on the raw text. `langdetect.detect_langs(...)` is deterministic given a seeded
RNG (DetectorFactory.seed); we set it from the CLI --seed.

Posts with empty/whitespace-only text or duplicated text are dropped before
language detection.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import polars as pl
from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect
from tqdm import tqdm

DATASET_NAME = "alpindale/two-million-bluesky-posts"
SOURCE = "bluesky"

DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def _post_id(source_id: str) -> str:
    """Stable 16-hex-char id from the source URI."""
    return hashlib.sha1(f"bsky:{source_id}".encode()).hexdigest()[:16]


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def load_posts(n: int, seed: int) -> pl.DataFrame:
    """Stream the HF dataset, filter to EN, return N rows as a polars DataFrame."""
    DetectorFactory.seed = seed  # deterministic langdetect

    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    # Shuffle the stream so we sample across the dataset rather than its head.
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    seen_texts: set[str] = set()
    rows: list[dict] = []
    pbar = tqdm(total=n, desc="collecting EN posts")
    for row in ds:
        text = row.get("text") or ""
        text = text.strip()
        if not text:
            continue
        if text in seen_texts:
            continue
        if not _is_english(text):
            continue
        seen_texts.add(text)

        source_id = row.get("uri") or ""
        if not source_id:
            continue

        rows.append(
            {
                "post_id": _post_id(source_id),
                "text": text,
                "source": SOURCE,
                "source_id": source_id,
                "lang": "en",
                "char_len": len(text),
                "created_at": row.get("created_at"),
            }
        )
        pbar.update(1)
        if len(rows) >= n:
            break
    pbar.close()

    df = pl.DataFrame(rows)
    # Parse created_at (ISO 8601 with Z) -> Datetime; null on failure.
    df = df.with_columns(
        pl.col("created_at")
        .str.to_datetime(strict=False, time_zone="UTC")
        .alias("created_at")
    )
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--n", type=int, default=2000, help="posts to keep after filtering")
    p.add_argument("--seed", type=int, default=42, help="reproducibility seed")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="override default output path (data/posts_n{N}.parquet)",
    )
    args = p.parse_args()

    out: Path = args.output or (DEFAULT_DATA_DIR / f"posts_n{args.n}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = load_posts(args.n, args.seed)
    df.write_parquet(out)
    print(f"wrote {len(df)} rows to {out}")
    print(df.head(3))


if __name__ == "__main__":
    main()
