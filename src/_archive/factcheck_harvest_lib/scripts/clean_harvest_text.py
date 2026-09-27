"""One-time pass: decode HTML entities / normalise Unicode in the text columns of every harvest in place.

  uv run python -m eval.scripts.clean_harvest_text            # report only, writes nothing
  uv run python -m eval.scripts.clean_harvest_text --apply    # rewrite the files that changed

Fixes raw entities baked into harvested text. Two families:
  - fact-check harvests (clog 280626): `&#039;` etc. in resolved claim/context text.
  - tweet corpora (clog 200826): X HTML-escapes & < > in the API text field, so `&amp;`
    reached the extractor, the read prompts and every display. The ingest points
    (harvest_outlet_tweets, sample_handle_tweets, cn_hydrate) now call clean_text, so
    this pass is only for corpora harvested before that.

Idempotent. Backs up each file it rewrites once, as <name>.preclean.parquet, so an
in-place rewrite of an untracked data file is recoverable.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import polars as pl

from eval.textnorm import clean_text

# claim / context columns (fact-check harvests) + post text columns (tweet corpora)
COLS = ["claim_text", "raw_claim", "raw_context", "context", "claimant", "quoted_text",
        "original_rating", "text", "post_text", "text_nourl", "claim", "claim_resolved"]

GLOBS = ["eval/data/*_harvest.parquet", "eval/data/**/*.parquet"]


def targets() -> list[str]:
    seen: dict[str, None] = {}
    for g in GLOBS:
        for f in sorted(glob.glob(g, recursive=True)):
            # never touch snapshots kept on purpose (.archived., .pre-*, our own backups)
            if not any(m in Path(f).name for m in (".preclean.", ".archived.", ".pre-")):
                seen.setdefault(f, None)
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite files (default: report only)")
    args = ap.parse_args()

    total = 0
    frozen: list[str] = []
    for f in targets():
        try:
            d = pl.read_parquet(f)
        except Exception as e:
            print(f"{f:70} SKIP ({type(e).__name__})")
            continue
        cols = [c for c in COLS if c in d.columns and d[c].dtype == pl.Utf8]
        if not cols:
            continue
        cleaned = d.with_columns([pl.col(c).map_elements(clean_text, return_dtype=pl.Utf8).alias(c)
                                  for c in cols])
        changed = {c: int((d[c].fill_null("") != cleaned[c].fill_null("")).sum()) for c in cols}
        n = sum(changed.values())
        if not n:
            continue
        total += n
        detail = ", ".join(f"{c}:{v}" for c, v in changed.items() if v)
        print(f"{f:70} {n:5} cells  ({detail})")
        if args.apply:
            try:
                bak = Path(f).with_suffix(".preclean.parquet")
                if not bak.exists():
                    d.write_parquet(bak)
                cleaned.write_parquet(f)
            except PermissionError:
                # read-only on purpose (frozen draw sets). Report, never chmod.
                frozen.append(f)
                print(f"{'':70} READ-ONLY, left untouched")

    print(f"\n{total} cells {'cleaned' if args.apply else 'would be cleaned'}"
          f"{'' if args.apply else '  — rerun with --apply to write'}")
    if frozen:
        print(f"\n{len(frozen)} file(s) are read-only and were NOT rewritten:")
        for f in frozen:
            print(f"  {f}")


if __name__ == "__main__":
    main()
