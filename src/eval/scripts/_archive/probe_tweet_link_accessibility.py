"""Article accessibility audit: per source, can we read the linked article for FREE?

For each captured outlet, sample its tweets that carry a t.co link, resolve the link to the real
article URL, and try the free path (trafilatura). Classify each article readable / short-paywalled /
fetch_failed / no_link, then summarize per source so we know which outlets need a paywall tool (Jina)
and which we can harvest for free right now. Sequential + polite (concurrent fetching gets blocked).
"""
import os, re, sys, time
import pandas as pd
import requests
import trafilatura

REPO = "src"
ORIG = REPO + "/eval/data/survey_claims/originals.csv"
RATINGS = REPO + "/eval/data/us_source_ratings.csv"
OUTDIR = REPO + "/eval/data/survey_claims"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
TCO = re.compile(r"https?://t\.co/\w+")
READABLE_MIN = 500  # chars of extracted body to count as a real, free-readable article

N_PER_OUTLET = int(sys.argv[1]) if len(sys.argv) > 1 else 12
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None  # smoke: restrict to outlets


def domain(url):
    m = re.match(r"https?://([^/]+)", url or "")
    return (m.group(1).lower().replace("www.", "") if m else "")


def resolve_tco(full_text, sess):
    """Return the first t.co in the tweet that resolves to a NON-twitter article URL."""
    for tco in TCO.findall(full_text or ""):
        try:
            r = sess.get(tco, allow_redirects=True, timeout=15, stream=True)
            final = r.url
            r.close()
            d = domain(final)
            if d and not any(x in d for x in ("x.com", "twitter.com", "t.co")):
                return final, d
        except Exception:
            continue
    return None, None


def fetch_body(url):
    dl = trafilatura.fetch_url(url)
    if not dl:
        return None
    return trafilatura.extract(dl, include_comments=False, include_tables=True)


def main():
    orig = pd.read_csv(ORIG)
    dom = dict(zip(pd.read_csv(RATINGS)["outlet"], pd.read_csv(RATINGS)["domain"]))
    links = orig[(orig["has_tco"]) & (orig["lang"] == "en")].copy()
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})

    rows = []
    outlets = [o for o in orig["outlet"].unique() if (ONLY is None or o in ONLY)]
    for oi, outlet in enumerate(outlets, 1):
        cell = orig[orig["outlet"] == outlet]["cell"].iloc[0]
        sub = links[links["outlet"] == outlet].head(N_PER_OUTLET)
        print(f"[{oi}/{len(outlets)}] {outlet} ({cell}) — testing {len(sub)} links", flush=True)
        for _, t in sub.iterrows():
            url, d = resolve_tco(t["full_text"], sess)
            if not url:
                rows.append(dict(outlet=outlet, cell=cell, tweet_id=t["id"], resolved_url="",
                                 art_domain="", body_chars=0, status="no_link"))
                print("      · no article link", flush=True)
                time.sleep(0.4)
                continue
            try:
                body = fetch_body(url)
            except Exception:
                body = None
            n = len(body) if body else 0
            status = "readable" if n >= READABLE_MIN else ("short_paywalled" if n > 0 else "fetch_failed")
            rows.append(dict(outlet=outlet, cell=cell, tweet_id=t["id"], resolved_url=url,
                             art_domain=d or dom.get(outlet, ""), body_chars=n, status=status))
            print(f"      · {status:16} {n:6}ch  {d}", flush=True)
            time.sleep(0.8)  # polite, sequential

    art = pd.DataFrame(rows)
    # per-source summary
    def verdict(g):
        rate = (g["status"] == "readable").mean()
        return "FREE" if rate >= 0.7 else ("PARTIAL" if rate >= 0.3 else "PAYWALLED")
    summ = []
    for outlet, g in art.groupby("outlet"):
        rate = (g["status"] == "readable").mean()
        summ.append(dict(
            outlet=outlet, cell=g["cell"].iloc[0], art_domain=g["art_domain"].mode().iloc[0] if len(g) else "",
            n_tested=len(g), n_readable=int((g["status"] == "readable").sum()),
            readable_rate=round(rate, 2), median_chars=int(g["body_chars"].median()),
            n_short_paywalled=int((g["status"] == "short_paywalled").sum()),
            n_fetch_failed=int((g["status"] == "fetch_failed").sum()),
            n_no_link=int((g["status"] == "no_link").sum()),
            verdict=verdict(g), needs_paywall_tool=("no" if rate >= 0.7 else "yes"),
        ))
    summ = pd.DataFrame(summ).sort_values("readable_rate", ascending=False)

    os.makedirs(OUTDIR, exist_ok=True)
    tag = "smoke" if ONLY else "full"
    xlsx = f"{OUTDIR}/source_accessibility_{tag}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        summ.to_excel(w, sheet_name="by_source", index=False)
        art.to_excel(w, sheet_name="by_article", index=False)
    print("\n=== per-source accessibility ===", flush=True)
    print(summ.to_string(index=False), flush=True)
    print(f"\nwrote {xlsx}", flush=True)


if __name__ == "__main__":
    main()
