"""Per-URL diagnostic for the Tier 3 round-1 flow.

Calls Planning + web search + per-URL scrape + per-URL summarisation and prints
the intermediate state for every URL so we can see exactly which step
discards each document.

Usage:
    uv run python -m scripts.tier3_diagnose --claim "<claim text>"
    uv run python -m scripts.tier3_diagnose --claim "..." --date-ceiling "10/27/2020"
"""
from __future__ import annotations

import argparse
from datetime import datetime

from pipeline.config import SEARCH_TOP_K
from pipeline.search import scrape, search
from pipeline.verify import _plan_initial_query, _summarise_for_claim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", required=True)
    ap.add_argument(
        "--date-ceiling",
        default=datetime.today().strftime("%m/%d/%Y"),
    )
    args = ap.parse_args()

    print(f"claim: {args.claim}")
    print(f"date_ceiling: {args.date_ceiling}")
    print()

    # 1. Planning
    print("--- Planning ---")
    q = _plan_initial_query(args.claim)
    print(f"opening query: {q!r}")
    print()

    # 2. Execution
    print("--- Execution (SearXNG) ---")
    results = search(q, SEARCH_TOP_K)
    print(f"{len(results)} url(s):")
    for r in results:
        print(f"  {r['url']}")
    print()

    # 3. Per-URL scrape + summarise
    print("--- Per-URL scrape + summarise ---")
    for i, r in enumerate(results, 1):
        u = r["url"]
        print(f"\n[{i}] {u}")
        body = scrape(u)
        if body is None:
            print(f"    scrape: FAILED (None — blocklist, 403/404, empty, or <100 chars)")
            continue
        print(f"    scrape: ok, {len(body)} chars")
        print(f"    preview: {body[:200].replace(chr(10), ' ')!r}")
        doc = _summarise_for_claim(args.claim, u, body, "scrape")
        print(f"    summary.relevant : {doc.relevant}")
        print(f"    summary.pub_date : {doc.publication_date}")
        print(f"    summary.text     : {doc.summary!r}")


if __name__ == "__main__":
    main()
