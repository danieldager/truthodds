"""Big bulk ingest over ALL free outlets: go deep per feed, skip already-processed URLs, and extract
claims from BOTH the article BODY and the HEADLINE of each new article (headlines and bodies yield
partly different claims, per the headline test). Grows the claim pool toward the 300 target.
Writes claims_bulk.csv (with extraction_source = body|headline) + appends body attempts to ace_attempts.
"""
import os, sys, re, time, gzip, json, datetime
import pandas as pd
import requests, feedparser, trafilatura

sys.path.insert(0, "src")
from eval import ace

REPO = "src"
OUTDIR = REPO + "/eval/data/survey_claims"
XLSX = OUTDIR + "/source_accessibility.xlsx"
FULLXL = OUTDIR + "/source_accessibility_full.xlsx"
ATTEMPTS = REPO + "/eval/data/ace_attempts.parquet"
BR = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")}
TIMEOUT = 12
MIN_CHARS = 400
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
ONLY = set(sys.argv[2].split("|")) if len(sys.argv) > 2 else None

NEWFREE = {
    "CNN": ("rss", "http://rss.cnn.com/rss/cnn_topstories.rss"),
    "ABC News": ("rss", "https://feeds.abcnews.com/abcnews/topstories"),
    "Politico": ("rss", "https://rss.politico.com/politics-news.xml"),
    "Democracy Now": ("rss", "https://www.democracynow.org/democracynow.rss"),
    "BBC": ("rss", "https://feeds.bbci.co.uk/news/rss.xml"),
    "The Washington Times": ("rss", "https://www.washingtontimes.com/rss/headlines/news/"),
    "The Daily Wire": ("rss", "https://www.dailywire.com/feeds/rss.xml"),
    "Christian Science Monitor": ("rss", "https://rss.csmonitor.com/feeds/all"),
    "Forbes": ("sitemap", "https://www.forbes.com/news_sitemap.xml"),
    "The Epoch Times": ("sitemap", "https://www.theepochtimes.com/sitemap/sitemap-news.xml.gz"),
    "HuffPost": ("scrape", None), "Associated Press": ("scrape", None), "Newsweek": ("scrape", None),
}
SECTION = re.compile(r"/(account|login|log-in|section|sitemap|tag|topic|topics|author|authors|video|videos|live|newsletter|newsletters|subscribe|hub|category|photos|gallery)(/|$)", re.I)


def dom(u):
    m = re.match(r"https?://([^/]+)", u or ""); return m.group(1).lower().replace("www.", "") if m else ""


def is_article(u, domain):
    if domain.replace("www.", "") not in dom(u): return False
    if SECTION.search(u) or u.rstrip("/").lower().endswith((".xml", ".gz", ".jpg", ".png", ".css", ".js")): return False
    path = re.sub(r"https?://[^/]+", "", u).split("?")[0]
    segs = [s for s in path.split("/") if s]
    return bool(re.search(r"/\d{4}/\d{2}/", path)) or bool(segs and (segs[-1].count("-") >= 3 or re.search(r"\d{6,}", segs[-1])))


def _clean(pairs):
    seen, out = set(), []
    for u, t in pairs:
        if not u: continue
        b = u.split("#")[0]
        if SECTION.search(b) or b in seen: continue
        seen.add(b); out.append((b, t))
    return out


def get_entries(url, sess):
    try:
        r = sess.get(url, timeout=TIMEOUT); return feedparser.parse(r.content).entries if r.status_code == 200 and r.content else []
    except Exception:
        return []


def sitemap_pairs(url, sess, n, domain, depth=0):
    try:
        r = sess.get(url, timeout=TIMEOUT); content = gzip.decompress(r.content) if url.endswith(".gz") else r.content
        text = content.decode("utf-8", "ignore")
    except Exception:
        return []
    locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text)
    if depth < 2 and locs and all(l.endswith((".xml", ".xml.gz")) for l in locs[:5]):
        pick = next((l for l in locs if "news" in l.lower()), locs[-1]); return sitemap_pairs(pick, sess, n, domain, depth + 1)
    return [(l, None) for l in locs if is_article(l, domain)][:n]


def article_pairs(outlet, typ, url, domain, sess, n, ba):
    if typ == "rss":
        return _clean([(getattr(e, "link", ""), getattr(e, "title", "")) for e in get_entries(url, sess)])[:n]
    if typ == "sitemap":
        return _clean(sitemap_pairs(url, sess, n * 3, domain))[:n]
    g = ba[(ba.outlet == outlet) & (ba.body_chars >= MIN_CHARS)]
    return _clean([(u, None) for u in g["art_url"].tolist() if is_article(u, domain)])[:n]


def fetch(url, sess):
    try:
        html = sess.get(url, timeout=TIMEOUT, allow_redirects=True).text
    except Exception:
        return None, None
    if not html: return None, None
    body = trafilatura.extract(html, include_comments=False, include_tables=True)
    title = None
    try:
        md = trafilatura.extract_metadata(html); title = getattr(md, "title", None) if md else None
    except Exception:
        pass
    return body, title


def main():
    bs = pd.read_excel(XLSX, sheet_name="by_source")
    ba_raw = pd.read_excel(XLSX, sheet_name="by_article_raw")
    ba_full = pd.read_excel(FULLXL, sheet_name="by_article")
    cellmap = dict(zip(bs["outlet"], bs["cell"])); dommap = dict(zip(bs["outlet"], bs["domain"]))
    free = set(bs[bs.verdict == "FREE"]["outlet"])
    FEEDS = {}
    for o, g in ba_full[ba_full.outlet.isin(free)].groupby("outlet"):
        f = [x for x in g.feed.tolist() if str(x).startswith("http")]
        if f: FEEDS[o] = ("rss", f[0])
    FEEDS.update(NEWFREE)
    targets = [o for o in FEEDS if o in free and (ONLY is None or o in ONLY)]

    seen = set(pd.read_parquet(ATTEMPTS)["url"]) if os.path.exists(ATTEMPTS) else set()
    sess = requests.Session(); sess.headers.update(BR)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    attempts, claims = [], []
    for oi, outlet in enumerate(sorted(targets), 1):
        typ, url = FEEDS[outlet]; cell, domain = cellmap.get(outlet, "?"), str(dommap.get(outlet, ""))
        pairs = article_pairs(outlet, typ, url, domain, sess, N, ba_raw)
        fresh = [(u, t) for u, t in pairs if u not in seen][:N]
        nb = nh = 0
        for aurl, rss_title in fresh:
            seen.add(aurl)
            body, meta_title = fetch(aurl, sess)
            headline = rss_title or meta_title or ""
            rec = {"source": outlet, "url": aurl, "headline": headline, "attempted_at": now,
                   "body_chars": len(body) if body else 0}
            if not body or len(body) < MIN_CHARS:
                rec.update({"status": "fetch_failed", "n_claims": 0, "claims_json": "[]"})
                attempts.append(rec); time.sleep(0.3); continue
            # BODY
            try:
                rb = ace.extract(body, headline=headline); bcl = rb.get("claims", [])
            except Exception:
                bcl = []
            rec.update({"status": "none" if not bcl else "claims", "n_claims": len(bcl),
                        "claims_json": json.dumps(bcl, ensure_ascii=False)})
            attempts.append(rec)
            for c in bcl:
                claims.append({"outlet": outlet, "cell": cell, "url": aurl, "headline": headline,
                               "claim": c.get("claim"), "risk_reason": c.get("risk_reason"),
                               "quote": c.get("quote"), "extraction_source": "body"}); nb += 1
            # HEADLINE
            if headline:
                try:
                    rh = ace.extract(headline, headline=None); hcl = rh.get("claims", [])
                except Exception:
                    hcl = []
                for c in hcl:
                    claims.append({"outlet": outlet, "cell": cell, "url": aurl, "headline": headline,
                                   "claim": c.get("claim"), "risk_reason": c.get("risk_reason"),
                                   "quote": c.get("quote"), "extraction_source": "headline"}); nh += 1
            time.sleep(0.3)
        print(f"[{oi}/{len(targets)}] {outlet:24} ({cell}) [{typ}] {len(fresh)} new -> body {nb}, headline {nh}", flush=True)

    # persist
    at = pd.DataFrame(attempts)
    if len(at):
        if os.path.exists(ATTEMPTS):
            old = pd.read_parquet(ATTEMPTS); at = pd.concat([old, at]).drop_duplicates("url", keep="last").reset_index(drop=True)
        at.to_parquet(ATTEMPTS, index=False)
    cl = pd.DataFrame(claims); dp = OUTDIR + "/claims_bulk.csv"
    if os.path.exists(dp) and len(cl):
        cl = pd.concat([pd.read_csv(dp), cl]).drop_duplicates(subset=["url", "claim", "extraction_source"]).reset_index(drop=True)
    cl.to_csv(dp, index=False)
    if len(cl):
        print("\nnew bulk claims by cell:", flush=True); print(cl.groupby("cell").size().to_string(), flush=True)
        print("\nby extraction_source:", flush=True); print(cl.groupby("extraction_source").size().to_string(), flush=True)
    print(f"\nwrote {len(claims)} claims this run -> {dp}", flush=True)


if __name__ == "__main__":
    main()
