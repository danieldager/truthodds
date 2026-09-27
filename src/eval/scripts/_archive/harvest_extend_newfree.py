"""Extend the claim harvest to the 15 newly-free outlets (from the 27-investigation).

Pulls REAL article URLs per source type — RSS items, sitemap <loc>s (incl. gzipped + index drill),
or filtered scrape URLs from the investigation's evidence — then trafilatura -> ACE. --dry prints the
URLs it would use (to confirm they're articles, not section pages) without calling ACE.
"""
import os, sys, re, time, gzip, json, datetime
import pandas as pd
import requests
import feedparser
import trafilatura

sys.path.insert(0, "src")
from eval import ace

REPO = "src"
XLSX = REPO + "/eval/data/survey_claims/source_accessibility.xlsx"
OUTDIR = REPO + "/eval/data/survey_claims"
ATTEMPTS = REPO + "/eval/data/ace_attempts.parquet"
BR = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")}
TIMEOUT = 12
MIN_CHARS = 400
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
DRY = "--dry" in sys.argv
ONLY = None
for a in sys.argv[2:]:
    if a != "--dry":
        ONLY = set(a.split("|"))

# outlet -> (type, url). cells come from the workbook.
SPEC = {
    "CNN": ("rss", "http://rss.cnn.com/rss/cnn_topstories.rss"),
    "ABC News": ("rss", "https://feeds.abcnews.com/abcnews/topstories"),
    "Politico": ("rss", "https://rss.politico.com/politics-news.xml"),
    "Democracy Now": ("rss", "https://www.democracynow.org/democracynow.rss"),
    "BBC": ("rss", "https://feeds.bbci.co.uk/news/rss.xml"),
    "The Washington Times": ("rss", "https://www.washingtontimes.com/rss/headlines/news/"),
    "The Daily Wire": ("rss", "https://www.dailywire.com/feeds/rss.xml"),
    "Forbes": ("sitemap", "https://www.forbes.com/news_sitemap.xml"),
    "The Epoch Times": ("sitemap", "https://www.theepochtimes.com/sitemap/sitemap-news.xml.gz"),
    "Christian Science Monitor": ("rss", "https://rss.csmonitor.com/feeds/all"),
    "HuffPost": ("scrape", None),
    "Associated Press": ("scrape", None),
    "Newsweek": ("scrape", None),
}
# Dropped: NY Daily News (R-L) + The Post Millennial (R-R) — no reachable feed (403/empty);
# both cells already well-covered (R-L 18 free, R-R 8 free). Revisit with a renderer if needed.
SECTION = re.compile(r"/(account|login|log-in|section|sitemap|tag|topic|topics|author|authors|video|videos|live|newsletter|newsletters|subscribe|hub|category|photos|gallery)(/|$)", re.I)


def dom(u):
    m = re.match(r"https?://([^/]+)", u or "")
    return m.group(1).lower().replace("www.", "") if m else ""


def is_article(u, domain):
    if domain.replace("www.", "") not in dom(u):
        return False
    if SECTION.search(u) or u.rstrip("/").lower().endswith((".xml", ".gz", ".jpg", ".png", ".css", ".js")):
        return False
    path = re.sub(r"https?://[^/]+", "", u).split("?")[0]
    segs = [s for s in path.split("/") if s]
    # article = a date in the path, OR a long multi-word slug, OR a long numeric id
    return bool(re.search(r"/\d{4}/\d{2}/", path)) or bool(segs and (segs[-1].count("-") >= 3 or re.search(r"\d{6,}", segs[-1])))


def get_entries(url, sess):
    try:
        r = sess.get(url, timeout=TIMEOUT)
        return feedparser.parse(r.content).entries if r.status_code == 200 and r.content else []
    except Exception:
        return []


def sitemap_urls(url, sess, n, domain, depth=0):
    try:
        r = sess.get(url, timeout=TIMEOUT)
        content = gzip.decompress(r.content) if url.endswith(".gz") else r.content
        text = content.decode("utf-8", "ignore")
    except Exception:
        return []
    locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text)
    if depth < 2 and locs and all(l.endswith((".xml", ".xml.gz")) for l in locs[:5]):
        pick = next((l for l in locs if "news" in l.lower()), locs[-1])
        return sitemap_urls(pick, sess, n, domain, depth + 1)
    arts = [l for l in locs if is_article(l, domain)]
    return [(u, None) for u in arts[:n]]


def _clean(pairs):
    """Strip #fragments, drop section/login URLs, dedup by base URL — keep order."""
    seen, out = set(), []
    for u, t in pairs:
        if not u:
            continue
        base = u.split("#")[0]
        if SECTION.search(base) or base in seen:
            continue
        seen.add(base)
        out.append((base, t))
    return out


def article_urls(outlet, domain, sess, n, ba):
    typ, url = SPEC[outlet]
    if typ == "rss":
        pairs = [(getattr(e, "link", ""), getattr(e, "title", "")) for e in get_entries(url, sess)]
        return _clean(pairs)[:n]
    if typ == "sitemap":
        return _clean(sitemap_urls(url, sess, n * 3, domain))[:n]
    # scrape: reuse the investigation's readable URLs, filtered to article-looking
    g = ba[(ba.outlet == outlet) & (ba.body_chars >= MIN_CHARS)]
    return _clean([(u, None) for u in g["art_url"].tolist() if is_article(u, domain)])[:n]


def fetch_article(url, sess):
    try:
        html = sess.get(url, timeout=TIMEOUT, allow_redirects=True).text
    except Exception:
        return None, None
    if not html:
        return None, None
    body = trafilatura.extract(html, include_comments=False, include_tables=True)
    title = None
    try:
        md = trafilatura.extract_metadata(html)
        title = getattr(md, "title", None) if md else None
    except Exception:
        pass
    return body, title


def main():
    s = pd.read_excel(XLSX, sheet_name="by_source")
    ba = pd.read_excel(XLSX, sheet_name="by_article_raw")
    cellmap = dict(zip(s["outlet"], s["cell"]))
    dommap = dict(zip(s["outlet"], s["domain"]))
    sess = requests.Session(); sess.headers.update(BR)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    targets = [o for o in SPEC if ONLY is None or o in ONLY]

    attempts, claims = [], []
    for oi, outlet in enumerate(targets, 1):
        cell, domain = cellmap.get(outlet, "?"), str(dommap.get(outlet, ""))
        urls = article_urls(outlet, domain, sess, N, ba)
        if DRY:
            print(f"[{oi}/{len(targets)}] {outlet} ({cell}) [{SPEC[outlet][0]}] {len(urls)} urls:", flush=True)
            for u, t in urls:
                print(f"      {u}", flush=True)
            continue
        got = 0
        for url, rss_title in urls:
            body, meta_title = fetch_article(url, sess)
            headline = rss_title or meta_title or ""
            rec = {"source": outlet, "cell": cell, "url": url, "headline": headline,
                   "attempted_at": now, "body_chars": len(body) if body else 0}
            if not body or len(body) < MIN_CHARS:
                rec.update({"status": "fetch_failed", "n_claims": 0, "claims_json": "[]"})
            else:
                try:
                    r = ace.extract(body, headline=headline)
                    nn, cl = r.get("n", len(r.get("claims", []))), r.get("claims", [])
                except Exception as ex:
                    nn, cl = 0, []
                    rec["ace_error"] = str(ex)[:120]
                rec.update({"status": "none" if nn == 0 else "claims", "n_claims": nn,
                            "claims_json": json.dumps(cl, ensure_ascii=False)})
                got += 1
                for c in cl:
                    claims.append({"outlet": outlet, "cell": cell, "url": url, "headline": headline,
                                   "claim": c.get("claim"), "risk_reason": c.get("risk_reason"), "quote": c.get("quote")})
            attempts.append(rec)
            time.sleep(0.4)
        nc = sum(a["n_claims"] for a in attempts if a["source"] == outlet)
        print(f"[{oi}/{len(targets)}] {outlet:24} ({cell}) fetched {got}/{len(urls)} -> {nc} claims", flush=True)

    if DRY:
        return
    at = pd.DataFrame(attempts)
    keep = [c for c in at.columns if c != "cell"]
    save = at[keep].copy()
    if os.path.exists(ATTEMPTS):
        old = pd.read_parquet(ATTEMPTS)
        save = pd.concat([old, save]).drop_duplicates(subset="url", keep="last").reset_index(drop=True)
    save.to_parquet(ATTEMPTS, index=False)
    pd.DataFrame(claims).to_csv(OUTDIR + "/claims_newfree.csv", index=False)
    at["has_claim"] = at["status"] == "claims"
    tot_f = int((at["status"] != "fetch_failed").sum())
    print(f"\nTOTAL: {len(at)} articles, {tot_f} fetched, {len(claims)} claims from "
          f"{int(at['has_claim'].sum())} articles", flush=True)
    print(at.groupby("cell")["has_claim"].agg(["count", "sum"]).to_string())
    print(f"wrote {OUTDIR}/claims_newfree.csv", flush=True)


if __name__ == "__main__":
    main()
