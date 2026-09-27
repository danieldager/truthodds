"""Characterize search providers on the SAME queries: who returns sources for which claims.

Samples claims (mixing ones the prior run retrieved vs. its zero-retrieval misses), runs each
provider on that claim's plan query (no date ceiling — we want raw coverage), and prints the
source URLs each provider returns. This is the "where is the consequential evidence found"
measurement, and it shows every provider returns linkable sources (not synthesized text).

    SEARCH_PROVIDER=tavily uv run python -m eval.scripts.verification_grading.compare_providers \
        -c .../claims_n100.parquet -v .../verdicts_n100_tavily.parquet -n 10
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import polars as pl

from pipeline import search

PROVIDERS = ["searxng", "tavily", "exa"]  # serper excluded: no credits (add when topped up)
TOP_K = 5


def dom(u: str) -> str:
    return urlparse(u).netloc.replace("www.", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--claims", type=Path, required=True)
    ap.add_argument("-v", "--verdicts", type=Path, required=True)
    ap.add_argument("-n", "--n", type=int, default=10)
    args = ap.parse_args()

    claims = pl.read_parquet(args.claims).select(["claim_id", "claim_text", "gold_label"])
    v = pl.read_parquet(args.verdicts).select(["claim_id", "past_queries", "n_urls_seen"])
    df = claims.join(v, on="claim_id", how="inner")

    # Mix: half from claims the prior run RETRIEVED, half from its ZERO-retrieval misses.
    half = args.n // 2
    got = df.filter(pl.col("n_urls_seen") > 0).head(args.n - half).to_dicts()
    zero = df.filter(pl.col("n_urls_seen") == 0).head(half).to_dicts()
    sample = got + zero
    print(f"sample: {len(got)} prior-retrieved + {len(zero)} prior-ZERO = {len(sample)} claims\n")

    # provider -> set of claim indices where it returned >=1 source
    covered = {p: set() for p in PROVIDERS}
    counts = {p: 0 for p in PROVIDERS}

    for i, r in enumerate(sample):
        q = r["past_queries"][0] if r["past_queries"] else r["claim_text"]
        tag = "ZERO" if r["n_urls_seen"] == 0 else "got "
        print("=" * 96)
        print(f"[{i}] ({tag}) gold={r['gold_label']}  query: {q[:84]}")
        for p in PROVIDERS:
            try:
                res = search.search(q, TOP_K, date_ceiling=None, provider=p)
            except search.SearchError as e:
                print(f"  {p:<8}: ERROR {str(e)[:80]}")
                continue
            if res:
                covered[p].add(i)
                counts[p] += len(res)
            doms = [f"{dom(x['url'])}{'*' if x.get('content') else ''}" for x in res]
            print(f"  {p:<8}: {len(res)} -> {doms}")

    print("\n" + "=" * 96)
    print("COVERAGE (claims with >=1 source):")
    n = len(sample)
    for p in PROVIDERS:
        print(f"  {p:<8}: {len(covered[p])}/{n}  ({len(covered[p])/n:.0%})   total sources={counts[p]}")
    # who covered the prior-zero misses?
    zero_idx = set(range(len(got), len(sample)))
    print("\nPrior-ZERO claims now covered:")
    for p in PROVIDERS:
        print(f"  {p:<8}: {len(covered[p] & zero_idx)}/{len(zero_idx)}")
    # union vs any single
    union = set().union(*covered.values())
    print(f"\nUNION coverage (any provider): {len(union)}/{n}  ({len(union)/n:.0%})")
    print("('*' after a domain = full page content available, not just a snippet)")


if __name__ == "__main__":
    main()
