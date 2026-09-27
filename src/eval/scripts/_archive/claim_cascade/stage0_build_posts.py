"""Stage 0 — parse the Zeeschuimer/4CAT X capture into a flat posts parquet.

Input : fourcat/exports/x_collection_010626.ndjson  (862 raw X tweet objects)
Output: data/posts_x862.parquet                      (one row per post)

Each NDJSON line is a full X GraphQL tweet object. We extract the
reader-visible text and the engagement/author/provenance signals the
downstream cascade needs, with the author's *own* text, any *quoted* text,
and any link-*card* snippet kept in separate columns so later stages can
measure each component's contribution.

Key text rules
--------------
* ``own_text`` is the author's complete text. When a post carries a
  ``note_tweet`` (>280 chars), ``legacy.full_text`` is a truncated stub
  ending in "… https://t.co/…"; the *full* text lives in
  ``note_tweet.note_tweet_results.result.text``. ``own_text`` therefore
  prefers the note-tweet text and falls back to ``legacy.full_text``.
* ``quoted_text`` applies the same note-tweet-expansion rule to the quoted
  post (``quoted_status_result.result``), so quoted long-posts are not
  truncated either.
* ``card_snippet`` is the title + description of a link preview card, read
  from ``card.legacy.binding_values`` (most captures stripped the card to a
  bare ``rest_id``; only a handful retain the snippet).
* The reader-visible ``text`` composes them:
      own_text [+ "\\n[QUOTED] " + quoted_text] [+ "\\n[LINK] " + card_snippet]
  The note-tweet is already folded into ``own_text`` (it is *the* own text),
  so it is not appended a second time. ``char_len = len(text)``.

post_id = sha1("x:" + rest_id)[:16], mirroring extraction_grading/load_posts.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT.parent / "fourcat" / "exports" / "x_collection_010626.ndjson"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "posts_x862.parquet"

# Twitter's classic created_at format, e.g. "Mon Jun 01 04:58:38 +0000 2026".
CREATED_AT_FMT = "%a %b %d %H:%M:%S %z %Y"

OUTPUT_SCHEMA = {
    "post_id": pl.String,
    "rest_id": pl.String,
    "text": pl.String,
    "own_text": pl.String,
    "quoted_text": pl.String,
    "card_snippet": pl.String,
    "lang": pl.String,
    "created_at": pl.String,  # parsed to Datetime after the frame is built
    "favorite_count": pl.Int64,
    "retweet_count": pl.Int64,
    "reply_count": pl.Int64,
    "quote_count": pl.Int64,
    "bookmark_count": pl.Int64,
    "views_count": pl.Int64,
    "author_handle": pl.String,
    "author_followers": pl.Int64,
    "author_verified": pl.Boolean,
    "has_community_note": pl.Boolean,
    "is_quote_status": pl.Boolean,
    "char_len": pl.Int64,
}


def _post_id(rest_id: str) -> str:
    """Stable 16-hex-char id from the X rest_id."""
    return hashlib.sha1(f"x:{rest_id}".encode()).hexdigest()[:16]


def _note_text(obj: dict) -> str:
    """The expanded note-tweet text of a tweet-result dict, or ""."""
    res = (
        (obj.get("note_tweet") or {}).get("note_tweet_results") or {}
    ).get("result") or {}
    return res.get("text") or ""


def _own_text(obj: dict) -> str:
    """Author's complete text: note-tweet if present, else legacy.full_text."""
    note = _note_text(obj)
    if note:
        return note
    return (obj.get("legacy") or {}).get("full_text") or ""


def _card_snippet(obj: dict) -> str:
    """title — description of a link card, from card.legacy.binding_values."""
    binding = ((obj.get("card") or {}).get("legacy") or {}).get("binding_values") or []
    vals: dict[str, str] = {}
    for b in binding:
        key = b.get("key")
        sv = (b.get("value") or {}).get("string_value")
        if key in ("title", "description") and sv:
            vals[key] = sv
    parts = [vals.get("title"), vals.get("description")]
    return " — ".join(p for p in parts if p)


def parse_record(obj: dict) -> dict:
    legacy = obj.get("legacy") or {}
    rest_id = str(obj.get("rest_id") or legacy.get("id_str") or "")

    own = _own_text(obj)
    quoted_result = (obj.get("quoted_status_result") or {}).get("result") or {}
    quoted = _own_text(quoted_result) if quoted_result else ""
    card = _card_snippet(obj)

    text = own
    if quoted:
        text += f"\n[QUOTED] {quoted}"
    if card:
        text += f"\n[LINK] {card}"

    author = ((obj.get("core") or {}).get("user_results") or {}).get("result") or {}
    a_legacy = author.get("legacy") or {}
    a_core = author.get("core") or {}

    views = (obj.get("views") or {}).get("count")
    views_count = int(views) if views and str(views).isdigit() else 0

    return {
        "post_id": _post_id(rest_id),
        "rest_id": rest_id,
        "text": text,
        "own_text": own,
        "quoted_text": quoted,
        "card_snippet": card,
        "lang": legacy.get("lang"),
        "created_at": legacy.get("created_at"),
        "favorite_count": legacy.get("favorite_count", 0),
        "retweet_count": legacy.get("retweet_count", 0),
        "reply_count": legacy.get("reply_count", 0),
        "quote_count": legacy.get("quote_count", 0),
        "bookmark_count": legacy.get("bookmark_count", 0),
        "views_count": views_count,
        "author_handle": a_core.get("screen_name") or a_legacy.get("screen_name") or "",
        "author_followers": a_legacy.get("followers_count"),
        "author_verified": bool(author.get("is_blue_verified")),
        "has_community_note": "birdwatch_pivot" in obj,
        "is_quote_status": bool(legacy.get("is_quote_status")),
        "char_len": len(text),
    }


def build(input_path: Path, limit: int | None = None) -> pl.DataFrame:
    rows: list[dict] = []
    with open(input_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(parse_record(json.loads(line)))
            if limit is not None and len(rows) >= limit:
                break

    df = pl.DataFrame(rows, schema=OUTPUT_SCHEMA)
    df = df.with_columns(
        pl.col("created_at").str.to_datetime(
            format=CREATED_AT_FMT, strict=False, time_zone="UTC"
        )
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("-n", "--limit", type=int, default=None, help="cap rows for smoke")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input NDJSON not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df = build(args.input, args.limit)
    if args.limit is None:
        df.write_parquet(args.output)
        print(f"wrote {len(df)} rows to {args.output}\n")

    # --- report / verification ---
    lang_dist = df["lang"].value_counts(sort=True)
    print(f"rows: {len(df)} | unique post_id: {df['post_id'].n_unique()}")
    print(f"quoted (non-empty quoted_text): {df.filter(pl.col('quoted_text') != '').height}")
    print(f"has_community_note: {df.filter('has_community_note').height}")
    print(f"is_quote_status: {df.filter('is_quote_status').height}")
    print(f"char_len<5 (would skip embed): {df.filter(pl.col('char_len') < 5).height}")
    print("\nlang distribution:")
    print(lang_dist.head(8))
    print("\nsample rows:")
    print(df.select(["post_id", "lang", "char_len", "author_handle", "views_count"]).head(5))


if __name__ == "__main__":
    main()
