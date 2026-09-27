"""Direct PolitiFact harvester (RSS -> article parse).

The Google FCT API under-ingests PolitiFact badly (~83% miss in a settled window,
~13d lag; measured 2026-06-23 — see clog/230626.md). PolitiFact has no on-page
ClaimReview JSON-LD, but its article markup is stable, so we parse the visible
structure with lxml/XPath:

  m-statement__quote  -> the checked claim (PolitiFact's normalized statement)
  m-statement__name   -> claimant / speaker
  m-statement__desc   -> "stated on <date> in <venue>:"
  m-statement__meter img@alt -> Truth-O-Meter rating slug ("barely-true", ...)
  "Our Sources"       -> cited-source links
  og:description      -> short ruling summary (fallback for justification)

Rating slugs are mapped to the textual strings the existing harmonize.RULES
["politifact.com"] table already understands, so harvested rows harmonise with
no rule changes. ("barely-true" is PolitiFact's legacy slug for "Mostly False".)
"""
from __future__ import annotations

import datetime
import re
import urllib.parse
from email.utils import parsedate_to_datetime

import lxml.html
import requests

PUBLISHER_SITE = "politifact.com"
PUBLISHER_NAME = "PolitiFact"
RSS_URL = "https://www.politifact.com/rss/factchecks/"
UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# Truth-O-Meter meter-image slug -> the textual rating in harmonize.RULES["politifact.com"].
RATING_SLUG = {
    "true": "True",
    "mostly-true": "Mostly True",
    "half-true": "Half True",
    "barely-true": "Mostly False",   # legacy slug for "Mostly False"
    "mostly-false": "Mostly False",
    "false": "False",
    "pants-fire": "Pants on Fire",
    "pants-on-fire": "Pants on Fire",
}

# Flip-O-Meter slugs — a SEPARATE meter (position changes, not veracity). Skip these.
FLIP_SLUGS = {"no-flip", "half-flip", "full-flop"}


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

# Full month names, EN + ES — PolitiFact has a Spanish vertical ("dicho el Mayo 27, 2026 …").
_MONTH_NAMES = {n: i for i, n in enumerate([
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december"], 1)}
_MONTH_NAMES.update({n: i for i, n in enumerate([
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "septiembre", "octubre", "noviembre", "diciembre"], 1)})


def _get(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text


def _url_date(url: str) -> datetime.date | None:
    """PolitiFact review date is encoded in the article URL: /factchecks/2026/jun/18/..."""
    m = re.search(r"/factchecks/(\d{4})/([a-z]{3})/(\d{1,2})/", url)
    if not m or m.group(2) not in _MONTHS:
        return None
    return datetime.date(int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))


def _rss_date(pubdate: str | None) -> str | None:
    try:
        return parsedate_to_datetime(pubdate).date().isoformat()
    except Exception:  # noqa: BLE001
        return None


def rss_items() -> list[dict]:
    """Recent fact-checks from the PolitiFact RSS feed: title, url, review_date."""
    xml = _get(RSS_URL)
    out = []
    for block in re.findall(r"<item>(.*?)</item>", xml, re.S):
        def tag(t):
            m = re.search(fr"<{t}>(.*?)</{t}>", block, re.S)
            return re.sub(r"^<!\[CDATA\[|\]\]>$", "", m.group(1).strip()) if m else None
        url = tag("link")
        if not url:
            continue
        out.append({
            "title": tag("title"),
            "url": url.replace("http://", "https://"),
            "review_date": _rss_date(tag("pubDate")),
        })
    return out


def archive_urls(since: datetime.date, max_pages: int = 40):
    """Yield {url, review_date} from the paginated archive back to `since`.

    The archive (`/factchecks/list/?page=N`) is reverse-chronological, 30/page;
    we stop on the first page that dips below the cutoff. Used for backfill — the
    RSS feed only carries the ~20 most recent items.
    """
    base = "https://www.politifact.com"
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        html = _get(f"{base}/factchecks/list/?page={page}")
        urls = list(dict.fromkeys(re.findall(
            r'href="(/factchecks/\d{4}/[a-z]{3}/\d{1,2}/[^"]+?/)"', html)))
        if not urls:
            return
        dated = [(base + u, _url_date(u)) for u in urls]
        for full, d in dated:
            if d and d >= since and full not in seen:
                seen.add(full)
                yield {"url": full, "review_date": d.isoformat()}
        if any(d and d < since for _, d in dated):  # crossed the cutoff
            return


def _first_text(tree, xpath: str) -> str | None:
    nodes = tree.xpath(xpath)
    if not nodes:
        return None
    text = nodes[0] if isinstance(nodes[0], str) else nodes[0].text_content()
    return re.sub(r"\s+", " ", text).strip() or None


def _claim_date(desc: str | None) -> tuple[str | None, str | None]:
    """('June 17, 2026', '2026-06-17') from 'stated on June 17, 2026 in an ad:' —
    handles EN ('stated on June 17, 2026') and ES ('dicho el Mayo 27, 2026')."""
    if not desc:
        return None, None
    m = re.search(r"\b([A-Za-zéíáúñ]+)\s+(\d{1,2}),?\s+(\d{4})\b", desc)
    if not m:
        return None, None
    mon = _MONTH_NAMES.get(m.group(1).lower())
    iso = None
    if mon:
        try:
            iso = datetime.date(int(m.group(3)), mon, int(m.group(2))).isoformat()
        except ValueError:
            pass
    return m.group(0), iso


def _unwrap(href: str) -> str:
    """PolitiFact source links are often google.com/url?q=<real> redirects."""
    if "google.com/url" in href:
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("q")
        if q:
            return q[0]
    return href


def _sources(tree) -> list[str]:
    """External links from the '<section id="sources">' (Our Sources) block."""
    out = []
    for h in tree.xpath('//section[@id="sources"]//a/@href'):
        h = _unwrap(h)
        if h.startswith("http") and "politifact.com" not in h and h not in out:
            out.append(h)
    return out


# --- raw-claim body mining ------------------------------------------------
# The header (m-statement__quote) is PolitiFact's *normalized* claim. The genuine raw lives in
# the article body, two ways: (a) an embedded ORIGINAL post (social claims), (b) the fuller
# verbatim quote a named speaker actually said (claim_text is a trimmed span of it).

_POST_URL = re.compile(
    r'^https?://(?:'
    r'(?:www\.)?(?:twitter|x)\.com/[^/?\s]+/status/\d+'
    r'|(?:www\.)?facebook\.com/[^/?\s]+/(?:posts|videos)/[\w./-]+'
    r'|(?:www\.)?instagram\.com/(?:p|reel|tv)/[\w-]+'
    r'|(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+'
    r')', re.I)
_POST_SKIP = re.compile(r'/(share|sharer|intent)\b|twitter\.com/share|/politifact/?$', re.I)


def _body_source_urls(tree) -> list[str]:
    """Original-post URLs the article cites as the CLAIM (not its evidence). Embedded posts
    (the thing being debunked, shown as a platform blockquote) rank first; X/Twitter status URLs
    (resolvable via syndication) before FB/IG/TikTok. Excludes PolitiFact's own share/profile links.
    """
    embeds = tree.xpath(
        '//blockquote[contains(@class,"twitter-tweet") or contains(@class,"instagram-media")]//a/@href'
        ' | //*[contains(@class,"tiktok-embed")]//a/@href')
    body = tree.xpath('//article//a/@href') or tree.xpath('//*[contains(@class,"m-textblock")]//a/@href')
    out: list[str] = []
    for href in list(embeds) + list(body):
        h = href.strip()
        if _POST_URL.match(h) and not _POST_SKIP.search(h) and h not in out:
            out.append(h)
    out.sort(key=lambda u: 0 if re.search(r'(twitter|x)\.com/[^/]+/status/', u) else 1)
    return out


def _qnorm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("“", '"').replace("”", '"')).strip()


def _body_quote(tree, claim_text: str | None) -> str | None:
    """The fuller verbatim quote in the body that contains the checked claim's quoted span.
    e.g. claim '“No Republican wins in Nassau.”' -> body '“You know, when he ran years ago … No
    Republican wins in Nassau. And he ran and won.”' Returns None if no meaningfully-fuller quote.
    """
    if not claim_text:
        return None
    spans = re.findall(r'"([^"]{12,})"', _qnorm(claim_text))
    if not spans:
        return None
    core = max(spans, key=len).strip().lower()
    paras = tree.xpath('//article//p') or tree.xpath('//*[contains(@class,"m-textblock")]//p')
    body = _qnorm(" ".join(p.text_content() for p in paras))
    best = None
    for q in re.findall(r'"([^"]{0,700})"', body):
        if core in q.lower() and len(q) > len(core) + 5 and (best is None or len(q) > len(best)):
            best = q.strip()
    return best


def _rating_slug(tree) -> str | None:
    """The main statement's Truth-O-Meter slug.

    Inside the first `m-statement__meter` the `<picture>` original image's `alt`
    carries the real slug ("pants-fire", "barely-true", …); the sibling thumbnail
    `alt` is a lazy-load placeholder ("true") and related fact-checks add decoy
    meters elsewhere on the page — so scope to the first meter, original image only.
    """
    meter = tree.xpath('(//*[contains(@class,"m-statement__meter")])[1]')
    if not meter:
        return None
    for a in meter[0].xpath('.//img[contains(@class,"c-image__original")]/@alt'):
        if a.strip():
            return a.strip().lower()
    for src in meter[0].xpath('.//img/@src'):  # fallback: slug in the file path
        m = re.search(r"meter-([a-z-]+?)(?:-th)?(?:/|\.(?:jpg|png))", src)
        if m:
            return m.group(1)
    return None


def parse_article(url: str, html: str | None = None) -> dict:
    """Parse one PolitiFact fact-check page into eval-schema row fields + extras."""
    tree = lxml.html.fromstring(html or _get(url))

    desc = _first_text(tree, '//*[contains(@class,"m-statement__desc")]')
    claim_date_raw, claim_date = _claim_date(desc)
    slug = _rating_slug(tree) or ""
    og_desc = _first_text(tree, '//meta[@property="og:description"]/@content')
    claim_text = _first_text(tree, '//*[contains(@class,"m-statement__quote")]')
    src_urls = _body_source_urls(tree)

    return {
        "claim_text": claim_text,
        "claim_source_url": src_urls[0] if src_urls else None,  # original post (resolve -> raw)
        "body_quote": _body_quote(tree, claim_text),            # fuller verbatim (raw fallback)
        "claimant": _first_text(tree, '//*[contains(@class,"m-statement__name")]'),
        "claim_date": claim_date,
        "claim_date_raw": claim_date_raw,
        "claim_venue": desc,
        "publisher_site": PUBLISHER_SITE,
        "publisher_name": PUBLISHER_NAME,
        "review_url": url,
        "review_title": _first_text(tree, '//meta[@property="og:title"]/@content'),
        "review_date": (_url_date(url).isoformat() if _url_date(url) else None),
        "language_code": ("es" if desc and "dicho" in desc.lower() else "en"),
        "original_rating": RATING_SLUG.get(slug, slug.replace("-", " ").title() or None),
        "rating_slug": slug or None,
        "justification": og_desc,
        "sources": _sources(tree),
    }
