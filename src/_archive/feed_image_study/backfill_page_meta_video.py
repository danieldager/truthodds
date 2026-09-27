"""Backfill `has_video` for the 207 non-X `page_meta` posts — the X-syndication quote/video tagger
(`fetch_quoted_and_tag`) only covers x_syndication rows, so page_meta (Bluesky/YouTube/Telegram/FB/web)
posts were left `has_video=None`. Video-dependent posts must be SKIPPED by the Stage-2 extractor (we
can't parse video yet), so they need a correct tag to be dropped by `subset()`'s `has_video != True`.

Signal: a YouTube URL → video; else fetch the page and look for `og:type=video.*`, `og:video`, or
`twitter:card=player` / `twitter:player`. Unfetchable (login-walled FB) → left None (unknown).
Updates each `*_harvest.parquet` in place — `has_video` on page_meta rows only, additive (every other
row and column untouched). Then re-run `build_dataset_splits` to refresh `dataset_12mo.parquet`.
"""
from __future__ import annotations

import glob
from concurrent.futures import ThreadPoolExecutor

import re

import polars as pl

from eval.source_fetch import _page_fetch

_OG_TYPE_VIDEO = re.compile(r'["\']og:type["\'][^>]+content=["\']video', re.I)
_OG_VIDEO = re.compile(r'(?:property|name)=["\']og:video', re.I)
_TW_PLAYER = re.compile(r'["\']twitter:card["\'][^>]+content=["\']player["\']|["\']twitter:player["\']', re.I)


def is_video(url: str | None) -> bool | None:
    """True=video, False=not video, None=can't tell (page unfetchable)."""
    if not url:
        return None
    if "youtube.com" in url or "youtu.be" in url:
        return True
    html = _page_fetch(url, timeout=15, wayback=False)
    if not html:
        return None
    return bool(_OG_TYPE_VIDEO.search(html) or _OG_VIDEO.search(html) or _TW_PLAYER.search(html))


def main() -> None:
    files = sorted(glob.glob("eval/data/*_harvest.parquet"))
    targets: dict[str, str] = {}
    for f in files:
        d = pl.read_parquet(f)
        if "source_method" not in d.columns:
            continue
        for r in d.filter(pl.col("source_method") == "page_meta").select("review_url", "claim_source_url").iter_rows(named=True):
            targets[r["review_url"]] = r["claim_source_url"]
    print(f"page_meta posts: {len(targets)}", flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        verdicts = dict(zip(targets.keys(), ex.map(is_video, targets.values())))
    upd = {u: v for u, v in verdicts.items() if v is not None}
    nv = sum(1 for v in upd.values() if v)
    print(f"determined {len(upd)}/{len(targets)}: video={nv}, not-video={len(upd)-nv}, unknown(left None)={len(targets)-len(upd)}")

    if upd:
        patch = pl.DataFrame({"review_url": list(upd.keys()), "_nv": list(upd.values())})
        for f in files:
            d = pl.read_parquet(f)
            if "has_video" not in d.columns or "review_url" not in d.columns:
                continue
            if not set(d["review_url"].to_list()) & set(upd):
                continue
            d2 = d.join(patch, on="review_url", how="left").with_columns(
                pl.coalesce("_nv", "has_video").alias("has_video")).drop("_nv")
            d2.write_parquet(f)
            print(f"  updated {f}")


if __name__ == "__main__":
    main()
