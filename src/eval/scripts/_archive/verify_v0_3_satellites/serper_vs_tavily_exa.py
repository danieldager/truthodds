"""Content-REACHABILITY test: Serper vs Tavily vs Exa on the same claim queries.

compare_providers.py answers "who returns a URL". This answers the sharper question:
who yields *usable page content*? Serper returns URLs only -> we must scrape (which can
fail on paywall/anti-bot); Tavily/Exa bundle the page body. So we run all three on the
same query, scrape Serper's URLs, and report where Tavily/Exa reach content Serper can't
(URL not surfaced, OR surfaced but scrape failed) and vice-versa.

One query per claim per provider => N calls to each (3N total). Exa free tier = 1000/mo,
so N=12 costs ~1.2% — leaves the n=500 AVeriTeC run intact.

    uv run python -m eval.scripts.verification_grading.serper_vs_tavily_exa -n 12
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import polars as pl

from pipeline import search

PROVIDERS = ["serper", "tavily", "exa"]
TOP_K = 5
MIN_CHARS = 500  # a real page body, not an error stub / cookie wall
DATA = Path(__file__).parent / "data" / "claims_n100.parquet"


def dom(u: str) -> str:
    return urlparse(u).netloc.replace("www.", "")


def usable_content(provider: str, r: dict) -> tuple[bool, int]:
    """Chars of usable content. Serper has no body -> scrape now; others bundle it."""
    body = r.get("content")
    if provider == "serper" and not body:
        body = search.scrape(r["url"]) or ""
    n = len((body or "").strip())
    return n >= MIN_CHARS, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=12)
    args = ap.parse_args()

    claims = pl.read_parquet(DATA).head(args.n).to_dicts()
    print(f"{len(claims)} claims | top_k={TOP_K} | usable >= {MIN_CHARS} chars\n")

    cov = {p: 0 for p in PROVIDERS}          # claims with >=1 result
    content_cov = {p: 0 for p in PROVIDERS}  # claims with >=1 usable-content doc
    docs = {p: 0 for p in PROVIDERS}         # total results
    usable = {p: 0 for p in PROVIDERS}       # total usable-content docs
    serper_unreach = 0                       # serper URLs surfaced but scrape failed
    tavily_exa_rescued = 0                   # of those domains, content reached by tavily/exa

    for i, c in enumerate(claims):
        q = c["claim_text"]
        print("=" * 100)
        print(f"[{i}] gold={c['gold_label']}  {q[:88]}")
        per = {}  # provider -> {domain: usable_bool}
        for p in PROVIDERS:
            try:
                res = search.search(q, TOP_K, date_ceiling=None, provider=p)
            except search.SearchError as e:
                print(f"  {p:<7}: ERROR {str(e)[:70]}")
                per[p] = {}
                continue
            if res:
                cov[p] += 1
            docs[p] += len(res)
            marks, dmap = [], {}
            any_content = False
            for r in res:
                ok, n = usable_content(p, r)
                dmap[dom(r["url"])] = ok
                any_content = any_content or ok
                if ok:
                    usable[p] += 1
                marks.append(f"{dom(r['url'])}({'OK' if ok else f'{n}'})")
            if any_content:
                content_cov[p] += 1
            per[p] = dmap
            print(f"  {p:<7}: {len(res)} -> {marks}")

        # serper URLs whose scrape failed, rescued by tavily/exa reaching content?
        serper_failed = {d for d, ok in per.get("serper", {}).items() if not ok}
        other_ok = {d for prov in ("tavily", "exa") for d, ok in per.get(prov, {}).items() if ok}
        for d in serper_failed:
            serper_unreach += 1
            if d in other_ok:
                tavily_exa_rescued += 1

    n = len(claims)
    print("\n" + "=" * 100)
    print("COVERAGE (>=1 result) | CONTENT-COVERAGE (>=1 usable doc):")
    for p in PROVIDERS:
        print(f"  {p:<7}: url {cov[p]}/{n} ({cov[p]/n:.0%})  |  content {content_cov[p]}/{n} "
              f"({content_cov[p]/n:.0%})  |  usable docs {usable[p]}/{docs[p]}")
    print(f"\nSerper URLs surfaced-but-scrape-failed: {serper_unreach}")
    print(f"  ...of those, same domain reached by Tavily/Exa: {tavily_exa_rescued}")
    print("(OK = >=500 chars usable content; number = chars when below threshold / 0 = unreachable)")


if __name__ == "__main__":
    main()
