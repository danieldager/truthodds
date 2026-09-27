"""HTTP transport. One tiny surface (`post`) so tests can inject a fake."""
from __future__ import annotations

import os

DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"


def api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        raise SystemExit("no API key: set DEEPINFRA_API_KEY (or LLM_API_KEY), or pass api_key=")
    return key


def base_url(explicit: str | None = None) -> str:
    return (explicit or os.environ.get("DEEPINFRA_BASE_URL")
            or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


class RequestsTransport:
    """requests.Session with a pool wide enough for 200 in-flight requests."""

    def __init__(self, pool=256):
        import requests
        self.session = requests.Session()
        ad = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=pool, max_retries=0)
        self.session.mount("https://", ad)
        self.session.mount("http://", ad)

    def post(self, url, headers, body, timeout):
        return self.session.post(url, headers=headers, json=body, timeout=timeout)
