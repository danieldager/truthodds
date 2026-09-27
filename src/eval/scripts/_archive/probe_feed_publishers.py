"""Discover EN publishers reachable via the API.

The API has no list-publishers endpoint, so probe each candidate from the
feed's top publishers (>=100 historical entries). For each, do one probe
request (page 1). If full and there's a nextPageToken, paginate further
up to MAX_DEEP_PAGES.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from eval.harvest import fetch_claims_raw

CANDIDATES_PATH = Path("eval/data/feed/publishers.txt")
OUT_PATH = Path("eval/data/survey/api_publishers.jsonl")
SUMMARY_PATH = Path("eval/data/survey/api_publishers_summary.tsv")

MIN_FEED_ENTRIES = 100
PAGE_SIZE = 100
MAX_DEEP_PAGES = 9             # so a "full" publisher caps at 1000 claims
MAX_TOTAL_REQUESTS = 250
MAX_AGE_DAYS = 365
LANG = "en"


def host(url: str) -> str:
    """Strip scheme + trailing slash to match the API's site filter format."""
    u = url.removeprefix("https://").removeprefix("http://")
    u = u.removeprefix("www.")
    return u.rstrip("/")


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates = []
    feed_count: dict[str, int] = {}
    for line in CANDIDATES_PATH.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        n, url = line.split("\t")
        n = int(n)
        if n < MIN_FEED_ENTRIES:
            break
        site = host(url)
        candidates.append(site)
        feed_count[site] = n
    print(f"Probing {len(candidates)} candidates (feed >= {MIN_FEED_ENTRIES})", flush=True)

    total_requests = 0
    results: dict[str, dict] = {}
    fout = OUT_PATH.open("w")
    try:
        for site in candidates:
            if total_requests >= MAX_TOTAL_REQUESTS:
                print(f"[abort] reached MAX_TOTAL_REQUESTS={MAX_TOTAL_REQUESTS}", file=sys.stderr)
                break
            try:
                resp = fetch_claims_raw(
                    review_publisher_site_filter=site,
                    max_age_days=MAX_AGE_DAYS,
                    page_size=PAGE_SIZE,
                    language_code=LANG,
                )
            except Exception as e:
                print(f"  [{site}] ERROR {e}", flush=True)
                results[site] = {"pages": 1, "claims": 0, "has_more": False, "error": str(e)}
                total_requests += 1
                continue
            total_requests += 1
            claims = resp.get("claims", []) or []
            token = resp.get("nextPageToken")
            pages = 1
            for c in claims:
                c["_harvest_publisher_site"] = site
                fout.write(json.dumps(c, ensure_ascii=False) + "\n")

            # Deep-probe prolific publishers
            while token and len(claims) == PAGE_SIZE and pages < MAX_DEEP_PAGES + 1:
                if total_requests >= MAX_TOTAL_REQUESTS:
                    break
                resp = fetch_claims_raw(
                    review_publisher_site_filter=site,
                    max_age_days=MAX_AGE_DAYS,
                    page_size=PAGE_SIZE,
                    language_code=LANG,
                    page_token=token,
                )
                total_requests += 1
                pages += 1
                claims = resp.get("claims", []) or []
                token = resp.get("nextPageToken")
                for c in claims:
                    c["_harvest_publisher_site"] = site
                    fout.write(json.dumps(c, ensure_ascii=False) + "\n")

            # Recount from file? simpler: tally as we go
            total_claims = sum(1 for _ in [None] * 0)  # placeholder
            results[site] = {
                "pages": pages,
                "has_more": bool(token),
            }
            print(f"  [{site}] pages={pages} has_more={bool(token)} (running req={total_requests})", flush=True)
    finally:
        fout.close()

    # Tally claims per publisher from file
    counts: Counter = Counter()
    for line in OUT_PATH.read_text().splitlines():
        c = json.loads(line)
        counts[c["_harvest_publisher_site"]] += 1

    with SUMMARY_PATH.open("w") as f:
        f.write("api_claims\tfeed_entries\tsite\thas_more\n")
        for site in sorted(candidates, key=lambda s: -counts.get(s, 0)):
            n_api = counts.get(site, 0)
            n_feed = feed_count.get(site, 0)
            r = results.get(site, {})
            f.write(f"{n_api}\t{n_feed}\t{site}\t{r.get('has_more', False)}\n")

    print()
    print("=" * 70)
    print(f"Total requests used: {total_requests}")
    print(f"Total EN claims fetched (12mo): {sum(counts.values())}")
    print(f"Publishers with >0 EN in API: {sum(1 for v in counts.values() if v>0)}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
