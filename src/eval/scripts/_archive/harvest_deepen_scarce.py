"""Deepen the claim harvest for the THIN reliability x bias cells (M-L, U-L, M-R).

Those cells have few free outlets, so the only lever is depth: pull more recent articles from each
thin-cell outlet's RSS feed, SKIPPING URLs already in ace_attempts.parquet, then trafilatura -> ACE.
Writes new claims to claims_deep.csv (same schema as the other batches) + appends to ace_attempts.
"""
import os, sys, time, json, datetime
import pandas as pd
import requests
import feedparser
import trafilatura

sys.path.insert(0, "src")
from eval import ace

REPO = "src"
OUTDIR = REPO + "/eval/data/survey_claims"
ATTEMPTS = REPO + "/eval/data/ace_attempts.parquet"
BR = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")}
TIMEOUT = 12
MIN_CHARS = 400
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
ONLY = set(sys.argv[2].split("|")) if len(sys.argv) > 2 else None

# thin-cell outlets -> (cell, RSS feed)
FEEDS = {
    "Mother Jones":         ("M-L", "https://www.motherjones.com/feed/"),
    "Daily Kos":            ("U-L", "https://www.dailykos.com/blogs/main.rss"),
    "MSNBC":                ("U-L", "https://www.ms.now/feed"),
    "Fox News":             ("M-R", "https://moxie.foxnews.com/google-publisher/latest.xml"),
    "New York Post":        ("M-R", "https://nypost.com/feed/"),
    "Townhall":             ("M-R", "https://townhall.com/feed/"),
    "The Washington Times": ("M-R", "https://www.washingtontimes.com/rss/headlines/news/"),
}


def get_entries(url, sess):
    try:
        r = sess.get(url, timeout=TIMEOUT)
        return feedparser.parse(r.content).entries if r.status_code == 200 and r.content else []
    except Exception:
        return []


def fetch_body(url, sess):
    try:
        html = sess.get(url, timeout=TIMEOUT, allow_redirects=True).text
        return trafilatura.extract(html, include_comments=False, include_tables=True) if html else None
    except Exception:
        return None


def main():
    seen = set(pd.read_parquet(ATTEMPTS)["url"]) if os.path.exists(ATTEMPTS) else set()
    sess = requests.Session(); sess.headers.update(BR)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    targets = {o: v for o, v in FEEDS.items() if ONLY is None or o in ONLY}
    attempts, claims = [], []
    for oi, (outlet, (cell, feed)) in enumerate(targets.items(), 1):
        entries = get_entries(feed, sess)
        fresh = [(getattr(e, "link", ""), getattr(e, "title", "")) for e in entries]
        fresh = [(u, t) for u, t in fresh if u and u not in seen][:N]
        got = 0
        for url, headline in fresh:
            seen.add(url)
            body = fetch_body(url, sess)
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
        print(f"[{oi}/{len(targets)}] {outlet:22} ({cell}) {len(fresh)} new arts, {got} fetched -> {nc} claims", flush=True)

    at = pd.DataFrame(attempts)
    keep = [c for c in at.columns if c != "cell"]
    save = at[keep].copy()
    if os.path.exists(ATTEMPTS):
        old = pd.read_parquet(ATTEMPTS)
        save = pd.concat([old, save]).drop_duplicates(subset="url", keep="last").reset_index(drop=True)
    save.to_parquet(ATTEMPTS, index=False)
    # append to claims_deep.csv (dedup on url+claim across re-runs)
    cl = pd.DataFrame(claims)
    dp = OUTDIR + "/claims_deep.csv"
    if os.path.exists(dp) and len(cl):
        cl = pd.concat([pd.read_csv(dp), cl]).drop_duplicates(subset=["url", "claim"]).reset_index(drop=True)
    cl.to_csv(dp, index=False)
    if len(cl):
        print("\nnew claims per cell:", flush=True)
        print(cl.groupby("cell").size().to_string(), flush=True)
    print(f"\nwrote {len(claims)} claims this run -> {dp}", flush=True)


if __name__ == "__main__":
    main()
