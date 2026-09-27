"""RSS -> article -> ACE claim-extraction smoke over the 32 free-readable outlets.

For each free outlet: pull N recent articles from its RSS feed, fetch the body (requests+trafilatura),
run ACE on the body, persist every attempt, and collect the extracted claims. Reports claim yield per
reliability x bias cell so we can judge the full harvest. Sequential + timeout'd. PAID API (ACE).
"""
import os, sys, time, json, datetime
import pandas as pd
import requests
import feedparser
import trafilatura

sys.path.insert(0, "src")
from eval import ace

REPO = "src"
RAW = REPO + "/eval/data/survey_claims/source_accessibility_full.xlsx"
OUTDIR = REPO + "/eval/data/survey_claims"
ATTEMPTS = REPO + "/eval/data/ace_attempts.parquet"
BR = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")}
TIMEOUT = 12
MIN_CHARS = 400
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ONLY = set(sys.argv[2].split("|")) if len(sys.argv) > 2 else None

FREE = ["AlterNet", "Jacobin", "Los Angeles Times", "NBC News", "Salon", "The Atlantic",
        "The Daily Beast", "The Guardian", "The Intercept", "Time", "Vox", "Business Insider",
        "NPR", "ProPublica", "City Journal", "HotAir", "NewsBusters", "Reason",
        "The American Conservative", "The Daily Caller", "Washington Examiner", "Mother Jones",
        "Fox News", "New York Post", "Townhall", "Daily Kos", "MSNBC", "Breitbart",
        "One America News", "The Blaze", "The Federalist", "The Gateway Pundit"]


def get_entries(url, sess):
    try:
        r = sess.get(url, timeout=TIMEOUT)
        if r.status_code != 200 or not r.content:
            return []
        return feedparser.parse(r.content).entries
    except Exception:
        return []


def fetch_body(url, sess):
    try:
        html = sess.get(url, timeout=TIMEOUT, allow_redirects=True).text
        return trafilatura.extract(html, include_comments=False, include_tables=True) if html else None
    except Exception:
        return None


def main():
    ba = pd.read_excel(RAW, sheet_name="by_article")
    feedmap, cellmap = {}, {}
    for outlet, g in ba.groupby("outlet"):
        feeds = [f for f in g["feed"].tolist() if str(f).startswith("http")]
        if feeds:
            feedmap[outlet] = feeds[0]
            cellmap[outlet] = g["cell"].iloc[0]
    targets = [o for o in FREE if o in feedmap and (ONLY is None or o in ONLY)]

    sess = requests.Session(); sess.headers.update(BR)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    attempts, claims = [], []
    for oi, outlet in enumerate(targets, 1):
        cell = cellmap.get(outlet, "?")
        entries = get_entries(feedmap[outlet], sess)[:N]
        got = 0
        for e in entries:
            url = getattr(e, "link", "")
            headline = getattr(e, "title", "")
            body = fetch_body(url, sess)
            rec = {"source": outlet, "cell": cell, "url": url, "headline": headline,
                   "attempted_at": now, "body_chars": len(body) if body else 0}
            if not body or len(body) < MIN_CHARS:
                rec.update({"status": "fetch_failed", "n_claims": 0, "claims_json": "[]"})
            else:
                try:
                    r = ace.extract(body, headline=headline)
                    n = r.get("n", len(r.get("claims", [])))
                    cl = r.get("claims", [])
                except Exception as ex:
                    n, cl = 0, []
                    rec["ace_error"] = str(ex)[:120]
                rec.update({"status": "none" if n == 0 else "claims", "n_claims": n,
                            "claims_json": json.dumps(cl, ensure_ascii=False)})
                got += 1
                for c in cl:
                    claims.append({"outlet": outlet, "cell": cell, "url": url, "headline": headline,
                                   "claim": c.get("claim"), "risk_reason": c.get("risk_reason"),
                                   "quote": c.get("quote")})
            attempts.append(rec)
            time.sleep(0.4)
        nc = sum(a["n_claims"] for a in attempts if a["source"] == outlet)
        print(f"[{oi}/{len(targets)}] {outlet:24} ({cell}) fetched {got}/{len(entries)} -> {nc} claims", flush=True)

    at = pd.DataFrame(attempts)
    # persist attempts (append + dedup by url)
    keep = [c for c in at.columns if c != "cell"]
    save = at[keep].copy()
    if os.path.exists(ATTEMPTS):
        old = pd.read_parquet(ATTEMPTS)
        save = pd.concat([old, save]).drop_duplicates(subset="url", keep="last").reset_index(drop=True)
    save.to_parquet(ATTEMPTS, index=False)
    cl = pd.DataFrame(claims)
    cl.to_csv(OUTDIR + "/claims_smoke.csv", index=False)

    print("\n=== per-cell claim yield ===", flush=True)
    at["has_claim"] = at["status"] == "claims"
    g = at.groupby("cell").agg(articles=("url", "count"),
                               fetched=("status", lambda s: (s != "fetch_failed").sum()),
                               with_claims=("has_claim", "sum"),
                               claims=("n_claims", "sum"))
    g["yield"] = (g["with_claims"] / g["fetched"].clip(lower=1)).round(2)
    print(g.to_string(), flush=True)
    tot_f = int((at["status"] != "fetch_failed").sum())
    print(f"\nTOTAL: {len(at)} articles, {tot_f} fetched, {len(cl)} claims from "
          f"{int(at['has_claim'].sum())} articles (yield {round(at['has_claim'].sum()/max(tot_f,1),2)})", flush=True)
    print(f"wrote {ATTEMPTS} + {OUTDIR}/claims_smoke.csv", flush=True)


if __name__ == "__main__":
    main()
