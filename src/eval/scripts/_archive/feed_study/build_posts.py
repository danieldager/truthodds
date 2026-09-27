"""Build the merged post corpus (old 862 + new flat captures) and a seeded sample.

The old corpus (`data/posts_x862.parquet`) was parsed from full Twitter GraphQL
objects by the archived `stage0_build_posts.py`. The NEW captures in
`../zeerover/x_capture_*.ndjson` are FLAT (`id, full_text, screen_name,
created_at, lang, conversation_id, operation, ...`) with no engagement / quoted /
card / author meta — so we parse them with a flat branch and concat under a
superset schema, defaulting the missing GraphQL-only fields to null.

Provenance columns added to BOTH corpora so analysis can condition on them:
  source_corpus  : "graphql_x862" | "flat_capture"
  has_quoted     : old post had non-empty quoted_text ([QUOTED] appended to text)
  has_card       : old post had a link card ([LINK] appended)
These matter because RAW-mode prompts see `text`, and old `text` can carry
[QUOTED]/[LINK] cruft that the flat captures never have.

Outputs:
  data/posts_merged.parquet          (all, deduped on post_id)
  data/posts_sample<N>.parquet       (seeded random sample, with --sample N)

No LLM. Run BEFORE any sweep so we validate the corpus before spending tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]  # -> src/
sys.path.insert(0, str(ROOT))

DEFAULT_OLD = Path(__file__).parent / "data" / "posts_x862.parquet"
DEFAULT_NEW_DIR = ROOT.parent.parent / "zeerover"  # misinform/zeerover (sibling of repo)
DATA_DIR = Path(__file__).parent / "data"

# Twitter's classic created_at format, e.g. "Mon Jun 01 04:58:38 +0000 2026".
CREATED_AT_FMT = "%a %b %d %H:%M:%S %z %Y"

# Columns shared with stage0's OUTPUT_SCHEMA (created_at kept as String during
# construction, parsed to Datetime after) + provenance/flat extras.
BASE_COLS = [
    "post_id", "rest_id", "text", "own_text", "quoted_text", "card_snippet",
    "lang", "created_at",
    "favorite_count", "retweet_count", "reply_count", "quote_count",
    "bookmark_count", "views_count",
    "author_handle", "author_followers", "author_verified",
    "has_community_note", "is_quote_status", "char_len",
]
EXTRA_COLS = ["source_corpus", "has_quoted", "has_card",
              "operation", "conversation_id", "is_reply", "topic"]

NEW_SCHEMA = {
    "post_id": pl.String, "rest_id": pl.String, "text": pl.String,
    "own_text": pl.String, "quoted_text": pl.String, "card_snippet": pl.String,
    "lang": pl.String, "created_at": pl.String,
    "favorite_count": pl.Int64, "retweet_count": pl.Int64, "reply_count": pl.Int64,
    "quote_count": pl.Int64, "bookmark_count": pl.Int64, "views_count": pl.Int64,
    "author_handle": pl.String, "author_followers": pl.Int64, "author_verified": pl.Boolean,
    "has_community_note": pl.Boolean, "is_quote_status": pl.Boolean, "char_len": pl.Int64,
    "source_corpus": pl.String, "has_quoted": pl.Boolean, "has_card": pl.Boolean,
    "operation": pl.String, "conversation_id": pl.String, "is_reply": pl.Boolean,
    "topic": pl.String,
}


def _post_id(rest_id: str) -> str:
    """Stable 16-hex-char id from the X rest_id (mirrors stage0_build_posts)."""
    return hashlib.sha1(f"x:{rest_id}".encode()).hexdigest()[:16]


def parse_record_flat(obj: dict) -> dict:
    """Flat Zeeschuimer record -> post row. Missing GraphQL fields default null."""
    rest_id = str(obj.get("id") or "")
    own = obj.get("full_text") or ""
    return {
        "post_id": _post_id(rest_id),
        "rest_id": rest_id,
        "text": own,            # no quoted/card in flat capture
        "own_text": own,
        "quoted_text": "",
        "card_snippet": "",
        "lang": obj.get("lang"),
        "created_at": obj.get("created_at"),
        "favorite_count": None, "retweet_count": None, "reply_count": None,
        "quote_count": None, "bookmark_count": None, "views_count": None,
        "author_handle": obj.get("screen_name") or "",
        "author_followers": None,
        "author_verified": None,
        "has_community_note": None,
        "is_quote_status": None,
        "char_len": len(own),
        "source_corpus": "flat_capture",
        "has_quoted": False,
        "has_card": False,
        "operation": obj.get("operation"),
        "conversation_id": obj.get("conversation_id"),
        "is_reply": obj.get("is_reply"),
        "topic": obj.get("topic"),
    }


def load_new(new_dir: Path) -> pl.DataFrame:
    files = sorted(new_dir.glob("x_capture_*.ndjson"))
    if not files:
        raise SystemExit(f"no x_capture_*.ndjson in {new_dir}")
    rows: list[dict] = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(parse_record_flat(json.loads(line)))
    df = pl.DataFrame(rows, schema=NEW_SCHEMA)
    return df.with_columns(
        pl.col("created_at").str.to_datetime(format=CREATED_AT_FMT, strict=False, time_zone="UTC")
    )


def load_old(old_path: Path) -> pl.DataFrame:
    """Old GraphQL-derived parquet + provenance columns."""
    df = pl.read_parquet(old_path)
    return df.with_columns(
        pl.lit("graphql_x862").alias("source_corpus"),
        ((pl.col("quoted_text").fill_null("") != "")).alias("has_quoted"),
        ((pl.col("card_snippet").fill_null("") != "")).alias("has_card"),
        pl.lit(None, dtype=pl.String).alias("operation"),
        pl.lit(None, dtype=pl.String).alias("conversation_id"),
        pl.lit(None, dtype=pl.Boolean).alias("is_reply"),
        pl.lit(None, dtype=pl.String).alias("topic"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", type=Path, default=DEFAULT_OLD)
    ap.add_argument("--new-dir", type=Path, default=DEFAULT_NEW_DIR)
    ap.add_argument("--sample", type=int, default=None, help="write a seeded random N-row sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cols = BASE_COLS + EXTRA_COLS
    old = load_old(args.old).select(cols)
    new = load_new(args.new_dir).select(cols)

    # old first so the richer GraphQL row wins any cross-corpus post_id collision.
    merged = pl.concat([old, new], how="vertical_relaxed")
    before = merged.height
    merged = merged.unique(subset=["post_id"], keep="first", maintain_order=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(DATA_DIR / "posts_merged.parquet")

    print(f"old: {old.height} | new: {new.height} | concat: {before} -> "
          f"deduped merged: {merged.height} (dropped {before - merged.height} dup post_id)")
    print(merged.group_by("source_corpus").len().sort("source_corpus"))
    print(f"empty text: {merged.filter(pl.col('text').str.strip_chars() == '').height} | "
          f"char_len<5: {merged.filter(pl.col('char_len') < 5).height}")
    print("lang dist:")
    print(merged["lang"].value_counts(sort=True).head(8))

    if args.sample is not None:
        n = min(args.sample, merged.height)
        sample = merged.sample(n=n, seed=args.seed, shuffle=True)
        out = DATA_DIR / f"posts_sample{args.sample}.parquet"
        sample.write_parquet(out)
        print(f"\nwrote sample: {out} ({sample.height} rows, seed={args.seed})")
        print(sample.group_by("source_corpus").len().sort("source_corpus"))


if __name__ == "__main__":
    main()
