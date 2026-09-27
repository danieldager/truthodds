import time
from typing import Iterator

import requests

from config import FCTAPI_ENDPOINT, FCTAPI_KEY


def fetch_claims_raw(
    query: str | None = None,
    review_publisher_site_filter: str | None = None,
    max_age_days: int | None = None,
    language_code: str = "en",
    page_size: int = 10,
    page_token: str | None = None,
) -> dict:
    """One raw call to the Google Fact Check Tools claims:search endpoint.

    Returns the parsed JSON response. Either `query` or
    `review_publisher_site_filter` must be set.
    """
    params: dict = {"key": FCTAPI_KEY, "languageCode": language_code, "pageSize": page_size}
    if query is not None:
        params["query"] = query
    if review_publisher_site_filter is not None:
        params["reviewPublisherSiteFilter"] = review_publisher_site_filter
    if max_age_days is not None:
        params["maxAgeDays"] = max_age_days
    if page_token is not None:
        params["pageToken"] = page_token

    # Retry transient network/5xx errors — a long paginated pull shouldn't die on one timeout.
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = requests.get(FCTAPI_ENDPOINT, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def paginate_claims(
    max_pages: int = 20,
    page_size: int = 100,
    **kwargs,
) -> Iterator[dict]:
    """Yield individual claim dicts across paginated responses.

    Stops at `max_pages` as a hard cap to bound API usage.
    """
    page_token: str | None = None
    for _ in range(max_pages):
        resp = fetch_claims_raw(page_size=page_size, page_token=page_token, **kwargs)
        for claim in resp.get("claims", []):
            yield claim
        page_token = resp.get("nextPageToken")
        if not page_token:
            return
