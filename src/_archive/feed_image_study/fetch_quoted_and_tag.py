"""One re-resolution pass per X row: quote-tweet capture + media-type tagging.

  uv run python -m eval.scripts.fetch_quoted_and_tag --parquet eval/data/snopes_harvest.parquet
  uv run python -m eval.scripts.fetch_quoted_and_tag --parquet eval/data/snopes_harvest.parquet --max 15

Runs AFTER fetch_images (which saves main-post images). For every row whose claim_source_url is an
X/Twitter status URL, it re-resolves the syndication payload ONCE and extracts:

  A) Video tagging (all X rows):
       media_types       list[str]   one entry per mediaDetails item: 'photo'|'video'|'animated_gif'
       has_video         bool        any entry is 'video' or 'animated_gif'

  B) Quote-tweet capture (rows that ARE quote-tweets):
       quoted_tweet_id   str|None
       quoted_text       str|None
       quoted_image_urls list[str]   QT photo / poster-frame URLs (same extraction as main post)
       quoted_image_paths list[str]  paths to downscaled WebP files saved under images/<source>/
       quoted_has_image  bool
       quoted_media_types list[str]  one entry per QT mediaDetails item
       quoted_has_video  bool        any QT entry is 'video' or 'animated_gif'

Files → eval/data/images/<source>/quoted_<quoted_tweet_id>_<n>.webp

ADDITIVE only: existing rows/columns are untouched. Resumable: rows with media_types already set
are skipped (whether the prior run succeeded or failed — failed rows get media_types=[]).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

from eval.media import download, media_urls_from_x, save_image
from eval.scripts._pool import pooled_checkpointed
from eval.source_fetch import _STATUS, _x_result

MAX_IMAGES = 4           # cap quoted images at the same limit as main-post images
CHECKPOINT_EVERY = 50
_VIDEO_TYPES = frozenset(("video", "animated_gif"))


def _media_types_from_payload(payload: dict) -> list[str]:
    """Extract mediaDetails[].type list from a tweet JSON payload."""
    return [
        md.get("type", "")
        for md in (payload.get("mediaDetails") or [])
        if isinstance(md, dict)
    ]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max", type=int, default=None, help="cap rows processed (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=1280, help="Qwen3-VL token budget per image")
    args = ap.parse_args()

    p = Path(args.parquet)
    source = p.stem.split("_harvest")[0]
    img_dir = p.parent / "images" / source
    df = pl.read_parquet(p)
    rows = df.to_dicts()
    n = df.height

    def _col(name, dtype_default=None):
        """Load column as a list or initialize with None/defaults."""
        return df[name].to_list() if name in df.columns else [dtype_default] * n

    # New column arrays — initialized from existing data if present
    media_types      = _col("media_types",      None)   # list[str]|None
    has_video        = _col("has_video",         None)   # bool|None
    qt_id            = _col("quoted_tweet_id",   None)   # str|None
    qt_text          = _col("quoted_text",       None)   # str|None
    qt_image_urls    = _col("quoted_image_urls", None)   # list[str]|None
    qt_image_paths   = _col("quoted_image_paths",None)   # list[str]|None
    qt_has_image     = _col("quoted_has_image",  None)   # bool|None
    qt_media_types   = _col("quoted_media_types",None)   # list[str]|None
    qt_has_video     = _col("quoted_has_video",  None)   # bool|None

    # todo: X-URL rows not yet processed (media_types is None → not yet touched)
    todo = [
        i for i, r in enumerate(rows)
        if r.get("claim_source_url") and _STATUS.search(r["claim_source_url"])
        and media_types[i] is None
    ]
    if args.max:
        todo = todo[: args.max]

    print(
        f"{len(todo)} X-URL rows to process (of {n} total rows in {p.name})",
        flush=True,
    )

    def build() -> pl.DataFrame:
        return df.with_columns(
            media_types=pl.Series(media_types,    dtype=pl.List(pl.Utf8)),
            has_video=pl.Series(has_video,        dtype=pl.Boolean),
            quoted_tweet_id=pl.Series(qt_id,      dtype=pl.Utf8),
            quoted_text=pl.Series(qt_text,        dtype=pl.Utf8),
            quoted_image_urls=pl.Series(qt_image_urls,   dtype=pl.List(pl.Utf8)),
            quoted_image_paths=pl.Series(qt_image_paths, dtype=pl.List(pl.Utf8)),
            quoted_has_image=pl.Series(qt_has_image,     dtype=pl.Boolean),
            quoted_media_types=pl.Series(qt_media_types, dtype=pl.List(pl.Utf8)),
            quoted_has_video=pl.Series(qt_has_video,     dtype=pl.Boolean),
        )

    abandoned = False
    if todo:
        def work(i):
            url = rows[i]["claim_source_url"]
            m = _STATUS.search(url)
            tweet_id = m.group(1)
            j = _x_result(tweet_id)

            if not j:
                # Deleted / protected — mark processed with empty/null so re-runs skip it
                return i, ([], False, None, None, None, None, None, None, None)

            # A) media-type tagging (main post)
            types = _media_types_from_payload(j)
            vid = any(t in _VIDEO_TYPES for t in types)

            # B) quote-tweet capture
            qt = j.get("quoted_tweet")
            if not qt or not isinstance(qt, dict):
                return i, (types, vid, None, None, [], [], False, [], False)

            q_id   = qt.get("id_str") or None
            q_text = (qt.get("text") or "").strip() or None
            q_types = _media_types_from_payload(qt)
            q_vid  = any(t in _VIDEO_TYPES for t in q_types)

            # Download + save quoted images (poster frames for video entries too)
            q_urls = media_urls_from_x(qt)
            saved_paths = []
            for k, u in enumerate(q_urls[:MAX_IMAGES]):
                b = download(u)
                if not b:
                    continue
                stem = q_id or tweet_id
                info = save_image(
                    b,
                    img_dir / f"quoted_{stem}_{k}.webp",
                    max_tokens=args.max_tokens,
                )
                if info:
                    saved_paths.append(info["path"])

            return i, (
                types, vid,
                q_id, q_text,
                q_urls, saved_paths,
                bool(saved_paths),
                q_types, q_vid,
            )

        def apply(i, payload):
            (types, vid, q_id, q_text, q_urls, q_paths, q_has_img, q_types, q_vid) = payload
            media_types[i]    = types
            has_video[i]      = vid
            qt_id[i]          = q_id
            qt_text[i]        = q_text
            qt_image_urls[i]  = q_urls if q_urls is not None else []
            qt_image_paths[i] = q_paths if q_paths is not None else []
            qt_has_image[i]   = q_has_img
            qt_media_types[i] = q_types if q_types is not None else []
            qt_has_video[i]   = q_vid

        abandoned = pooled_checkpointed(
            todo, work, apply,
            lambda: build().write_parquet(p),
            args.workers, "qt+tag",
            checkpoint_every=CHECKPOINT_EVERY,
        )

    df = build()
    df.write_parquet(p)

    # ── summary ──────────────────────────────────────────────────────────────
    x_rows = df.filter(pl.col("media_types").is_not_null())
    vid_rows  = x_rows.filter(pl.col("has_video") == True)   # noqa: E712
    qt_rows   = x_rows.filter(pl.col("quoted_tweet_id").is_not_null())
    qt_img_rows = x_rows.filter(pl.col("quoted_has_image") == True)   # noqa: E712

    # count saved quoted images
    q_paths_all = [p for row in qt_image_paths if row for p in row]

    print(f"\nWrote {p}")
    print(f"X rows processed: {x_rows.height}  |  has_video: {vid_rows.height}  |  "
          f"quote-tweets: {qt_rows.height}  |  QT w/ image: {qt_img_rows.height}  |  "
          f"QT image files saved: {len(q_paths_all)}")
    if vid_rows.height:
        print(f"media_types breakdown: {x_rows.explode('media_types').group_by('media_types').agg(pl.len().alias('n')).sort('n', descending=True).to_dicts()}")

    if abandoned:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
