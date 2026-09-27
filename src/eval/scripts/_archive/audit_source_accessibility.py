"""Per-source paywall audit via RSS: can we free-read each outlet's articles?

For every outlet (all 64, from Tab 3), discover an RSS feed, pull recent article URLs, fetch each via
the free path (requests + trafilatura.extract), and measure extracted body length. Classify per source
FREE / PARTIAL / PAYWALLED so we know which need a paywall tool (Jina) and which we can harvest now.
Independent of tweet-link shorteners (trib.al is dead). Sequential + polite; every network call has a
timeout so a paywalled/slow host can't hang the run.
"""
import os, re, sys, time
import pandas as pd
import requests
import feedparser
import trafilatura

REPO = "src"
T3 = REPO + "/eval/data/source_ratings_unified.xlsx"
OUTDIR = REPO + "/eval/data/survey_claims"
BR = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")}
FEED_PATHS = ["/feed/", "/rss/", "/rss", "/feed", "/rss.xml", "/index.xml",
              "/?feed=rss2", "/arc/outboundfeeds/rss/?outputType=xml"]
TIMEOUT = 12
N_ART = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ONLY = set(sys.argv[2].split("|")) if len(sys.argv) > 2 else None
READABLE = 1000  # >= this many extracted chars = a full, free-readable article


def side(b):
    b = str(b)
    return "L" if b in ("left", "lean-left", "left-center") else "R" if b in ("right", "lean-right", "right-center") else "C"


def tier(r):
    t = r.get("newsguard_tier")
    if isinstance(t, str) and t:
        return t[0]
    try:
        s = float(r.get("newsguard_score"))
        return "R" if s >= 75 else "M" if s >= 60 else "U"
    except Exception:
        return "?"


def get_entries(url, sess):
    """Fetch a candidate feed with a timeout, parse it, return entries ([] on any failure)."""
    try:
        r = sess.get(url, timeout=TIMEOUT)
        if r.status_code != 200 or not r.content:
            return []
        return feedparser.parse(r.content).entries
    except Exception:
        return []


def discover_feed(domain, sess):
    """Homepage <link rel=alternate rss> first, then common feed paths."""
    try:
        html = sess.get(f"https://{domain}/", timeout=TIMEOUT).text
        m = re.search(r'<link[^>]+type=["\']application/(?:rss|atom)\+xml["\'][^>]+href=["\']([^"\']+)', html, re.I) \
            or re.search(r'<link[^>]+href=["\']([^"\']+)["\'][^>]+type=["\']application/(?:rss|atom)\+xml', html, re.I)
        if m:
            href = m.group(1)
            if href.startswith("/"):
                href = f"https://{domain}{href}"
            if get_entries(href, sess):
                return href
    except Exception:
        pass
    for p in FEED_PATHS:
        url = f"https://{domain}{p}"
        if get_entries(url, sess):
            return url
    return None


def dom(u):
    m = re.match(r"https?://([^/]+)", u or "")
    return m.group(1).lower().replace("www.", "") if m else ""


def homepage_articles(domain, sess, n):
    """Fallback when no RSS feed: scrape same-domain article-looking links off the homepage."""
    try:
        html = sess.get(f"https://{domain}/", timeout=TIMEOUT).text
    except Exception:
        return []
    base = domain.replace("www.", "")
    urls = []
    for m in re.finditer(r'href=["\'](https?://[^"\']+|/[^"\']+)["\']', html):
        u = m.group(1)
        if u.startswith("/"):
            u = f"https://{domain}{u}"
        if base not in dom(u):
            continue
        path = re.sub(r"https?://[^/]+", "", u).split("?")[0].split("#")[0]
        segs = [s for s in path.split("/") if s]
        # article-looking: >=2 path segments and a slug with a hyphen or a long number
        if len(segs) >= 2 and re.search(r"-|\d{5,}", segs[-1]) and not u.lower().endswith((".jpg", ".png", ".css", ".js", ".xml")):
            if u not in urls:
                urls.append(u)
        if len(urls) >= n * 3:
            break
    return urls[:n]


def fetch_body(url, sess):
    try:
        html = sess.get(url, timeout=TIMEOUT, allow_redirects=True).text
        return trafilatura.extract(html, include_comments=False, include_tables=True) if html else None
    except Exception:
        return None


def main():
    t3 = pd.read_excel(T3, sheet_name=2)
    sess = requests.Session(); sess.headers.update(BR)
    rows = []
    recs = [r for r in t3.to_dict("records") if (ONLY is None or r["outlet"] in ONLY)]
    for oi, r in enumerate(recs, 1):
        outlet, domain = r["outlet"], str(r["domain"])
        cell = f"{tier(r)}-{side(r['bias_editorial'])}"
        feed = discover_feed(domain, sess)
        if feed:
            urls = [getattr(e, "link", "") for e in get_entries(feed, sess)[:N_ART]]
            src = "feed"
        else:
            urls = homepage_articles(domain, sess, N_ART)
            src = "homepage" if urls else "none"
        urls = [u for u in urls if u]
        if not urls:
            rows.append(dict(outlet=outlet, cell=cell, domain=domain, feed="", art_url="",
                             body_chars=0, status="no_source"))
            print(f"[{oi}/{len(recs)}] {outlet:24} ({cell}) NO SOURCE", flush=True)
            continue
        marks = []
        for url in urls:
            n = len(fetch_body(url, sess) or "")
            st = "readable" if n >= READABLE else ("short" if n > 0 else "failed")
            rows.append(dict(outlet=outlet, cell=cell, domain=domain, feed=(feed or src), art_url=url,
                             body_chars=n, status=st))
            marks.append(f"{n}" if n else "x")
            time.sleep(0.4)
        print(f"[{oi}/{len(recs)}] {outlet:24} ({cell}) [{src}] {' '.join(marks)}", flush=True)

    art = pd.DataFrame(rows)
    summ = []
    for outlet, g in art.groupby("outlet", sort=False):
        tested = g[g["status"].isin(["readable", "short", "failed"])]
        rate = (g["status"] == "readable").mean() if len(tested) else 0.0
        med = int(tested["body_chars"].median()) if len(tested) else 0
        if not len(tested):
            v, need = "NO_SOURCE", "unknown"
        else:
            v = "FREE" if rate >= 0.6 and med >= READABLE else ("PAYWALLED" if med < 500 else "PARTIAL")
            need = "no" if v == "FREE" else ("yes" if v == "PAYWALLED" else "maybe")
        summ.append(dict(outlet=outlet, cell=g["cell"].iloc[0], domain=g["domain"].iloc[0],
                         n_tested=len(tested), readable_rate=round(rate, 2), median_chars=med,
                         verdict=v, needs_paywall_tool=need))
    summ = pd.DataFrame(summ).sort_values(["verdict", "readable_rate"], ascending=[True, False])

    os.makedirs(OUTDIR, exist_ok=True)
    tag = "smoke" if ONLY else "full"
    xlsx = f"{OUTDIR}/source_accessibility_{tag}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        summ.to_excel(w, sheet_name="by_source", index=False)
        art.to_excel(w, sheet_name="by_article", index=False)
    print("\n=== per-source accessibility ===", flush=True)
    print(summ.to_string(index=False), flush=True)
    print(f"\nVERDICT COUNTS: {dict(summ['verdict'].value_counts())}", flush=True)
    print(f"wrote {xlsx}", flush=True)


if __name__ == "__main__":
    main()
