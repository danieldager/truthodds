"""Phase 1 survey: 12-month volume per marquee EN publisher via the API.

Estimates monthly request cost for a future harvester. Hard-caps total
requests as a safety net (default 200). Writes raw claims to JSONL so we
never re-fetch.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

from eval.harvest import fetch_claims_raw

OUT_PATH = Path("eval/data/survey/12mo.jsonl")
PUBLISHERS = [
    "politifact.com",
    "factcheck.org",
    "snopes.com",
    "fullfact.org",
    "factcheck.afp.com",
    "africacheck.org",
    "apnews.com",
    "reuters.com",
    "washingtonpost.com",
]
PAGE_SIZE = 100
MAX_PAGES_PER_PUB = 20         # 20 * 100 = 2000 results/publisher ceiling
MAX_TOTAL_REQUESTS = 200       # hard kill-switch
MAX_AGE_DAYS = 365
LANG = "en"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_requests = 0
    by_pub_requests: Counter = Counter()
    by_pub_claims: Counter = Counter()
    all_claims = []

    for site in PUBLISHERS:
        page_token = None
        pages = 0
        while pages < MAX_PAGES_PER_PUB:
            if total_requests >= MAX_TOTAL_REQUESTS:
                print(f"[abort] reached MAX_TOTAL_REQUESTS={MAX_TOTAL_REQUESTS}", file=sys.stderr)
                _finalise(all_claims, total_requests, by_pub_requests, by_pub_claims)
                return
            resp = fetch_claims_raw(
                review_publisher_site_filter=site,
                max_age_days=MAX_AGE_DAYS,
                page_size=PAGE_SIZE,
                language_code=LANG,
                page_token=page_token,
            )
            total_requests += 1
            by_pub_requests[site] += 1
            pages += 1
            claims = resp.get("claims", [])
            by_pub_claims[site] += len(claims)
            for c in claims:
                c["_harvest_publisher_site"] = site
                all_claims.append(c)
            page_token = resp.get("nextPageToken")
            print(
                f"  [{site}] page {pages}: {len(claims)} claims "
                f"(running total req={total_requests})",
                flush=True,
            )
            if not page_token:
                break
            time.sleep(0.2)  # polite pacing

    _finalise(all_claims, total_requests, by_pub_requests, by_pub_claims)


def _finalise(claims, total_requests, by_pub_requests, by_pub_claims):
    with OUT_PATH.open("w") as f:
        for c in claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print()
    print("=" * 60)
    print(f"Total requests used: {total_requests}")
    print(f"Total claims fetched: {len(claims)}")
    print(f"Output: {OUT_PATH}")
    print()
    print("Per-publisher breakdown:")
    print(f"  {'publisher':<28} {'requests':>10}  {'claims':>10}")
    for site in PUBLISHERS:
        print(f"  {site:<28} {by_pub_requests[site]:>10}  {by_pub_claims[site]:>10}")


if __name__ == "__main__":
    main()
