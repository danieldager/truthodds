"""Bonus source-fetch — resolve a claim's source URL to the raw claim text + context.

Builds the parallel `(raw → normalized)` extraction dataset, best-effort. X/Twitter posts resolve
via the syndication endpoint (powers embedded tweets; needs a token derived from the tweet id),
with a Wayback fallback for deleted/protected ones; other URLs fall back to the page's
og:description. Misses (deleted tweets with no snapshot) are expected — the verbatim raw claim is a
bonus, not required. The stubborn tail can be swept later with a logged-in headless browser.
"""
from __future__ import annotations

import json
import math
import re

import requests

from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL, EXTRACTION_MODEL
from eval import media
from eval.claimreview import UA, fetch as _page_fetch
from eval.textnorm import clean_text

_STATUS = re.compile(r"(?:twitter|x)\.com/[^/]+/status/(\d+)")
_OG = re.compile(r'<meta[^>]+(?:property="og:description"|name="description")[^>]+content="([^"]+)"')


def _x_token(tweet_id: str) -> str:
    """Reproduce the syndication token: ((id/1e15)*pi) in base-36, zeros and dot stripped."""
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    n = (int(tweet_id) / 1e15) * math.pi
    i, f = int(n), n - int(n)
    s = "" if i else "0"
    while i:
        s, i = digits[i % 36] + s, i // 36
    frac, c = "", 0
    while f > 0 and c < 24:
        f *= 36
        k = int(f)
        frac += digits[k]
        f -= k
        c += 1
    return re.sub(r"(0+|\.)", "", s + ("." + frac if frac else ""))


def _x_result(tweet_id: str, timeout: int = 15) -> dict | None:
    try:
        r = requests.get(
            "https://cdn.syndication.twimg.com/tweet-result",
            params={"id": tweet_id, "token": _x_token(tweet_id), "lang": "en"},
            headers=UA, timeout=timeout,
        )
        if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
            return r.json()
    except (requests.RequestException, ValueError):
        pass
    return None


_STOP = set("the a an and or of to in on at for with from by is are was were be been being this "
            "that these those it its as has have had not no than then so if but will would can "
            "could into about over they their them you your we our his her she he who".split())


def _content_toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) > 3 and w not in _STOP}


def _related(raw: str, claim: str, thresh: float = 0.30) -> bool:
    """Token-overlap backstop for the LLM gate: does the resolved post share enough of the claim's
    salient words to plausibly be its source? Used only when post_states_claim errors out."""
    c = _content_toks(claim)
    return True if not c else len(c & _content_toks(raw)) / len(c) >= thresh


_GATE_SYS = (
    "You decide whether a social-media POST is the SOURCE of a fact-checked CLAIM. Answer true only "
    "if the POST asserts, shows, or repeats the same factual assertion as the CLAIM (wording may "
    "differ). Answer false if the POST is about a different topic, is the fact-checker's own "
    "evidence, an opposing or critical reply, or carries no factual assertion (e.g. a bare link or "
    'a one-word reply). Respond with JSON: {"states_claim": true|false}.'
)


def post_states_claim(raw_claim: str | None, claim_text: str | None,
                      model: str | None = None, timeout: int = 30) -> bool | None:
    """LLM gate: does the resolved post actually make the checked claim? Guards the wrong-embed case
    — a fact-check page cites the claim's post AND its evidence/amplifiers. None on error (caller
    falls back to the token-overlap backstop)."""
    if not raw_claim or not claim_text:
        return None
    try:
        r = requests.post(
            f"{EXTRACTION_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {EXTRACTION_API_KEY}"},
            json={
                "model": model or EXTRACTION_MODEL,
                "messages": [
                    {"role": "system", "content": _GATE_SYS},
                    {"role": "user", "content": f"CLAIM: {claim_text}\n\nPOST: {raw_claim}"},
                ],
                "temperature": 0,
                "reasoning_effort": "none",  # thinking off → JSON lands in content (clog 230626)
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return bool(json.loads(r.json()["choices"][0]["message"]["content"] or "{}").get("states_claim"))
    except (requests.RequestException, ValueError, KeyError):
        return None


def assemble_raw(row: dict) -> tuple[str | None, str | None]:
    """The raw claim paired with the normalized `claim_text`, richest first:
      - 'post'  : the resolved original post text (row['raw_claim']) — the messiest real-world form,
                  kept only when it passed the relevance gate (row['raw_related']; guards wrong embeds).
      - 'quote' : a fuller verbatim quote the fact-check reproduces (row['body_quote']) — the
                  speaker's own words, less trimmed than the normalized claim.
      - None    : no source post and no fuller quote → no usable (raw -> normalized) pair.
    The old 'attributed' tier (claimant + normalized verbatim) was dropped — prefixing a claimant
    to the normalized claim is not a genuine raw, just the claim restated (w/ Daniel, clog 240626).
    `raw_related` is computed by the fetch_sources gate pass (post_states_claim).
    """
    if row.get("raw_claim") and row.get("raw_related"):
        return row["raw_claim"], "post"
    bq = (row.get("body_quote") or "").strip()
    if bq:
        return bq, "quote"
    return None, None


def fetch_raw_claim(source_url: str | None, timeout: int = 20) -> dict:
    """-> {raw_claim, source_method, claim_date}. Best-effort; raw_claim is None on a miss.
    For X posts, claim_date is the tweet's `created_at` (the original-claim date)."""
    if not source_url:
        return {"raw_claim": None, "source_method": None, "claim_date": None}
    m = _STATUS.search(source_url)
    if m:
        j = _x_result(m.group(1), timeout=timeout)
        if j:
            txt = clean_text((j.get("text") or "").strip())
            if txt:
                return {"raw_claim": txt, "source_method": "x_syndication",
                        "claim_date": (j.get("created_at") or "")[:10] or None}
        # X post failed syndication (deleted/protected) → record a miss; do NOT fall through to the
        # page+Wayback path: deleted tweets aren't Wayback-recoverable and hundreds of those lookups
        # get throttled by archive.org, which dominated the Lead Stories run (clog 250626).
        return {"raw_claim": None, "source_method": "failed", "claim_date": None}
    # non-X URL (FB/IG/TikTok/web): direct page fetch → og:description. wayback=False: ~46% of Lead
    # Stories sources are login-walled socials that Wayback can't recover, and the hundreds of
    # throttled archive.org lookups were the real bottleneck (clog 250626).
    html = _page_fetch(source_url, timeout=timeout, wayback=False)
    if html:
        og = _OG.search(html)
        if og and len(og.group(1)) > 20:
            return {"raw_claim": clean_text(og.group(1).strip()), "source_method": "page_meta", "claim_date": None}
    return {"raw_claim": None, "source_method": "failed", "claim_date": None}


def resolve_media_urls(source_url: str | None, timeout: int = 20) -> tuple[list[str], str]:
    """The image(s) attached to the original post -> (media_urls, image_source). Mirrors the
    raw-text resolution in fetch_raw_claim, but for media and keyed off the URL (not the stored
    source_method) so it also captures IMAGE-ONLY posts that text-resolution marks 'failed':
      - X status URL -> the syndication payload's photo/poster media ('x_media')
      - any other URL -> the page's og:image ('og_image')
    Empty + 'none' on a miss. Re-fetches by design (the image pass runs after the text pass)."""
    if not source_url:
        return [], "none"
    m = _STATUS.search(source_url)
    if m:
        j = _x_result(m.group(1), timeout=timeout)
        urls = media.media_urls_from_x(j) if j else []
        return (urls, "x_media") if urls else ([], "none")
    html = _page_fetch(source_url, timeout=timeout, wayback=False)
    urls = media.og_image_urls(html) if html else []
    return (urls, "og_image") if urls else ([], "none")
