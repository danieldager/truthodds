"""Client for X's Community Notes AI Note Writer API (the note-eligible post feed).

Ported from llm_bench/cn_api/client.py, which polled the same endpoint for latency
measurement only. Two changes here: the request asks for everything the survey
selection needs (author expansion, public_metrics, conversation_id, entities), and
OAuth 1.0a is signed with the stdlib so the uv project needs no new dependency.

Auth is OAuth 1.0a user context - every Community Notes endpoint is user-scoped.
Credentials come from src/.env (or the environment), names fixed by the note-writer
guide: X_API_KEY, X_API_KEY_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET.

Rate budget: 90 requests / 15 min shared across all Community Notes endpoints.
The feed is free of charge inside that budget.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]            # .../src
ENV = SRC / ".env"
BASE = "https://api.x.com"
ELIGIBLE_POSTS = "/2/notes/search/posts_eligible_for_notes"

ENV_KEYS = ("X_API_KEY", "X_API_KEY_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET")

# The endpoint speaks the legacy tweet.* aliases (verified live 2026-09-10 and again on
# today's smoke); the OpenAPI spec's post.* spelling 400s. A 400 naming one of these
# fields means the alias was finally retired.
TWEET_FIELDS = (
    "id,text,author_id,created_at,lang,conversation_id,public_metrics,entities,"
    "possibly_sensitive,in_reply_to_user_id,note_tweet,referenced_tweets,media_metadata,"
    "suggested_source_links_with_counts,note_request_suggestions"
)
EXPANSIONS = (
    "author_id,attachments.media_keys,referenced_tweets.id,"
    "referenced_tweets.id.author_id,referenced_tweets.id.attachments.media_keys"
)
USER_FIELDS = (
    "id,name,username,description,created_at,location,url,verified,verified_type,"
    "protected,profile_image_url,public_metrics,entities"
)
MEDIA_FIELDS = (
    "alt_text,duration_ms,height,media_key,preview_image_url,public_metrics,type,url,"
    "variants,width"
)

RATE_HEADERS = ("x-rate-limit-limit", "x-rate-limit-remaining", "x-rate-limit-reset",
                "x-user-limit-24hour-limit", "x-user-limit-24hour-remaining",
                "x-user-limit-24hour-reset")


def load_env():
    """Environment wins; src/.env fills the gaps. Never print the values."""
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    creds = [os.environ.get(k) for k in ENV_KEYS]
    if not all(creds):
        missing = [k for k, v in zip(ENV_KEYS, creds) if not v]
        sys.exit(f"missing X credentials in src/.env: {', '.join(missing)}")
    return creds


def _quote(s):
    return urllib.parse.quote(str(s), safe="~-._")


def _oauth_header(method, url, params, creds):
    """RFC 5849 HMAC-SHA1 signature over the query params (no body params: GET only)."""
    ck, cs, tok, ts = creds
    oauth = {
        "oauth_consumer_key": ck,
        "oauth_nonce": secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": tok,
        "oauth_version": "1.0",
    }
    allp = {**{k: str(v) for k, v in params.items()}, **oauth}
    norm = "&".join(f"{_quote(k)}={_quote(allp[k])}" for k in sorted(allp))
    base = "&".join([method.upper(), _quote(url), _quote(norm)])
    key = f"{_quote(cs)}&{_quote(ts)}".encode()
    sig = base64.b64encode(hmac.new(key, base.encode(), hashlib.sha1).digest()).decode()
    oauth["oauth_signature"] = sig
    return "OAuth " + ", ".join(f'{_quote(k)}="{_quote(v)}"' for k, v in sorted(oauth.items()))


class CNClient:
    """Community Notes note-eligible feed. One method, one endpoint, raw JSON out."""

    def __init__(self, creds=None):
        self.creds = creds or load_env()
        self.last_headers = {}

    def get(self, path, params, timeout=45):
        url = f"{BASE}{path}"
        auth = _oauth_header("GET", url, params, self.creds)
        req = urllib.request.Request(
            f"{url}?{urllib.parse.urlencode(params)}",
            headers={"Authorization": auth, "Accept": "application/json",
                     "User-Agent": "factchecking_with_LLMs/cn-harvest"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                self.last_headers = {k.lower(): v for k, v in r.headers.items()}
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            self.last_headers = {k.lower(): v for k, v in e.headers.items()}
            body = e.read().decode()[:800]
            raise RuntimeError(f"HTTP {e.code} on {path}: {body}") from None

    def eligible_posts(self, test_mode=True, feed_lang="all", max_results=100,
                       pagination_token=None, tweet_fields=TWEET_FIELDS,
                       expansions=EXPANSIONS, user_fields=USER_FIELDS):
        """One page of the note-eligible feed. Returns the raw response dict."""
        params = {
            "test_mode": "true" if test_mode else "false",
            "max_results": max_results,
            "post_selection": f"feed_lang:{feed_lang}",
            "tweet.fields": tweet_fields,
            "expansions": expansions,
            "media.fields": MEDIA_FIELDS,
        }
        if user_fields:
            params["user.fields"] = user_fields
        if pagination_token:
            params["pagination_token"] = pagination_token
        return self.get(ELIGIBLE_POSTS, params)

    def rate_state(self):
        return {k: self.last_headers.get(k) for k in RATE_HEADERS
                if self.last_headers.get(k) is not None}
