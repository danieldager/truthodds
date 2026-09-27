"""Phase 0 smoke test: one request to the Google Fact Check Tools API.

Hits PolitiFact, last 30 days, 10 results. Prints the raw JSON so we can
eyeball field shapes before building anything else.
"""
import json

from eval.harvest import fetch_claims_raw


def main() -> None:
    resp = fetch_claims_raw(
        review_publisher_site_filter="politifact.com",
        max_age_days=30,
        page_size=10,
        language_code="en",
    )
    print(json.dumps(resp, indent=2, ensure_ascii=False))
    n = len(resp.get("claims", []))
    print(f"\n--- {n} claims returned ---", flush=True)


if __name__ == "__main__":
    main()
