"""AFP source-side pass — recover the checked-post URL + cited sources that the GFC API strips.

  uv run python -m eval.scripts.fetch_afp_sources --parquet eval/data/afp_harvest.parquet

AFP's site is Akamai edge-blocked (403 to plain requests AND headless Chrome), so we fetch each
article body through the Jina Reader proxy (r.jina.ai, off-IP). AFP tags its body links by `rel`,
so `eval.afp.afp_sources` splits them cleanly: rel="appearance" → the checked post (claim_source_url),
rel="evidence" → the cited sources (the evidence column). Idempotent/resumable — only fetches rows
whose `sources` isn't set yet, so a connection drop just resumes. Needs JINA_API_KEY (500 RPM).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from config import JINA_API_KEY
from eval.afp import AFP_SELECTOR, afp_sources
from eval.claimreview import fetch_jina


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--workers", type=int, default=8)  # 500 RPM with a key → 8 concurrent is safe
    ap.add_argument("--max", type=int, default=None, help="cap rows (validation)")
    args = ap.parse_args()
    if not JINA_API_KEY:
        print("WARN: no JINA_API_KEY — keyless is 20 RPM; this will be slow.")

    p = Path(args.parquet)
    df = pl.read_parquet(p)
    rows = df.to_dicts()
    cp = df["claim_source_url"].to_list()
    srcs = df["sources"].to_list()
    todo = [i for i, r in enumerate(rows) if not r.get("sources")]  # resumable
    if args.max:
        todo = todo[: args.max]
    print(f"{len(todo)} AFP rows to fetch via jina (of {df.height})")
    if not todo:
        print("nothing to do.")
        return

    def work(i):
        # selector returns only the rel-tagged anchors → ~1-2k tok/article (vs ~50k full-page)
        html = fetch_jina(rows[i]["review_url"], target_selector=AFP_SELECTOR)
        return (i, *afp_sources(html)) if html else (i, None, None)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, checked, sources in tqdm(ex.map(work, todo), total=len(todo), desc="jina"):
            if checked:
                cp[i] = checked
            if sources:
                srcs[i] = sources

    df = df.with_columns(
        claim_source_url=pl.Series(cp, dtype=pl.Utf8),
        sources=pl.Series(srcs, dtype=pl.List(pl.Utf8)),
    )
    df.write_parquet(p)

    got_src = sum(1 for s in srcs if s)
    got_cp = sum(1 for c in cp if c)
    print(f"\nWrote {p}")
    print(f"cited sources filled: {got_src}/{df.height} ({got_src/df.height*100:.0f}%)  |  "
          f"checked post: {got_cp}/{df.height} ({got_cp/df.height*100:.0f}%)")
    avg = df.filter(pl.col("sources").is_not_null())["sources"].list.len().mean()
    print(f"avg cited sources/row (where present): {avg:.1f}" if avg else "")


if __name__ == "__main__":
    main()
