"""Fetch + save the IMAGE(S) attached to each resolved original post — the image-native pass.

  uv run python -m eval.scripts.fetch_images --parquet eval/data/snopes_harvest.parquet
  uv run python -m eval.scripts.fetch_images --parquet eval/data/snopes_harvest.parquet --max 10  # smoke

Runs AFTER fetch_sources (which resolves the post text + source_method). For every row whose post we
resolved (source_method in {x_syndication, page_meta}), it re-resolves the post's media URLs
(X syndication mediaDetails / og:image — see source_fetch.resolve_media_urls), downloads each, and
downscales to a ~1280-token Qwen3-VL budget as high-quality WebP (eval.media). Files land at
eval/data/images/<source>/<id>_<n>.webp and the per-image provenance is written back as new columns.

A SEPARATE pass (not folded into fetch_sources) on purpose: the heavy image I/O is re-runnable
without re-touching the LLM relevance gate, and the text-resolution semantics stay untouched.

Best-effort, idempotent, CRASH-SAFE: checkpointed every CHECKPOINT_EVERY rows, applied out-of-order,
stragglers abandoned after the idle timeout — a re-run skips rows already attempted (image_source
set) and finishes the rest (same driver as fetch_sources — clog 250626/260626).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import polars as pl

from eval.media import download, save_image
from eval.scripts._pool import pooled_checkpointed
from eval.source_fetch import _STATUS, resolve_media_urls

MAX_IMAGES = 4          # X caps a post at 4 photos; bound og fallback the same
CHECKPOINT_EVERY = 50   # images are heavier than the text pass -> checkpoint more often
RESOLVED = ("x_syndication", "page_meta")  # rows whose original post we actually resolved


def _ident(url: str) -> str:
    """Stable per-post file stem: the X status id, else a sha1 prefix of the URL (NOT Python hash(),
    which is per-process salted -> unstable filenames across runs)."""
    m = _STATUS.search(url or "")
    return m.group(1) if m else hashlib.sha1((url or "").encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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

    def col(name):
        return df[name].to_list() if name in df.columns else [None] * n

    has_image, image_count, image_source = col("has_image"), col("image_count"), col("image_source")
    image_urls, image_paths = col("image_urls"), col("image_paths")
    orig_dims, saved_dims, est_tokens = col("image_orig_dims"), col("image_saved_dims"), col("image_est_tokens")

    todo = [i for i, r in enumerate(rows)
            if r.get("claim_source_url") and r.get("source_method") in RESOLVED and image_source[i] is None]
    if args.max:
        todo = todo[: args.max]
    print(f"{len(todo)} rows to fetch images for (of {n}; resolved-post rows only)", flush=True)

    def build() -> pl.DataFrame:
        return df.with_columns(
            has_image=pl.Series(has_image, dtype=pl.Boolean),
            image_count=pl.Series(image_count, dtype=pl.Int64),
            image_source=pl.Series(image_source, dtype=pl.Utf8),
            image_urls=pl.Series(image_urls, dtype=pl.List(pl.Utf8)),
            image_paths=pl.Series(image_paths, dtype=pl.List(pl.Utf8)),
            image_orig_dims=pl.Series(orig_dims, dtype=pl.List(pl.Utf8)),
            image_saved_dims=pl.Series(saved_dims, dtype=pl.List(pl.Utf8)),
            image_est_tokens=pl.Series(est_tokens, dtype=pl.List(pl.Int64)),
        )

    abandoned = False
    if todo:
        def work(i):
            url = rows[i]["claim_source_url"]
            urls, src = resolve_media_urls(url)
            saved = []
            for k, u in enumerate(urls[:MAX_IMAGES]):
                b = download(u)
                if not b:
                    continue
                info = save_image(b, img_dir / f"{_ident(url)}_{k}.webp", max_tokens=args.max_tokens)
                if info:
                    saved.append(info)
            return i, (urls, src, saved)

        def apply(i, payload):
            urls, src, saved = payload
            image_source[i] = src
            image_urls[i] = list(urls)
            image_paths[i] = [s["path"] for s in saved]
            orig_dims[i] = [s["orig_dims"] for s in saved]
            saved_dims[i] = [s["saved_dims"] for s in saved]
            est_tokens[i] = [s["est_tokens"] for s in saved]
            image_count[i] = len(saved)
            has_image[i] = bool(saved)

        abandoned = pooled_checkpointed(todo, work, apply, lambda: build().write_parquet(p),
                                        args.workers, "images", checkpoint_every=CHECKPOINT_EVERY)
    df = build()
    df.write_parquet(p)

    with_img = df.filter(pl.col("has_image") == True)  # noqa: E712 (polars expr)
    n_resolved = df.filter(pl.col("source_method").is_in(RESOLVED)).height
    toks = [t for row in (est_tokens or []) if row for t in row]
    print(f"\nWrote {p}")
    print(f"has_image: {with_img.height}/{n_resolved} resolved-post rows "
          f"({with_img.height / n_resolved * 100:.0f}%)  |  images saved under {img_dir}/")
    if "image_source" in df.columns:
        print("by image_source:", df.filter(pl.col("image_source").is_not_null())
              .group_by("image_source").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())
    if toks:
        toks.sort()
        print(f"est_tokens/image over {len(toks)} images: "
              f"min={toks[0]} median={toks[len(toks)//2]} max={toks[-1]} "
              f"mean={sum(toks)//len(toks)}")

    if abandoned:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
