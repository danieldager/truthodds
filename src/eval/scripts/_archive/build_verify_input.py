"""Shape claim_pool.csv into the schema pipeline/verify_text.py consumes.

verify_text(raw_text, block_factcheck=True, date_ceiling=claim_date) per claim, keyed by claim_id.
For the survey the CLAIM *is* the synthetic-tweet text, so raw_context = the claim (framing intact,
NOT further de-framed). Adds a per-claim date ceiling (URL-derived where the article URL carries a date,
else the harvest date) so an FC-blocked run is a fair 'live, no-fact-check-yet' simulation, plus the
grouping metadata (side x reliability) that defines the experiment cells.
"""
import re, hashlib
from urllib.parse import urlparse
import pandas as pd

REPO = "src"
POOL = REPO + "/eval/data/survey_claims/claim_pool.csv"
ATTEMPTS = REPO + "/eval/data/ace_attempts.parquet"
OUT = REPO + "/eval/data/survey_claims/claim_pool_verify.parquet"
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def valid(y, mo, d):
    return 2000 <= y <= 2035 and 1 <= mo <= 12 and 1 <= d <= 31


def url_date(u):
    u = str(u)
    m = re.search(r"/(20\d{2})/(\d{2})/(\d{2})[/-]", u)          # /2026/07/03/
    if m and valid(*(int(m.group(i)) for i in (1, 2, 3))):
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"/(20\d{2})/(\d{2})(\d{2})[/-]", u)            # /2026/0703/ (csm)
    if m and valid(int(m.group(1)), int(m.group(2)), int(m.group(3))):
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"/(20\d{2})/([a-z]{3})/(\d{1,2})[/-]", u, re.I)  # /2026/jul/3/ (washtimes)
    if m and m.group(2).lower() in MONTHS:
        return f"{m.group(1)}-{MONTHS[m.group(2).lower()]:02d}-{int(m.group(3)):02d}"
    return None


def main():
    d = pd.read_csv(POOL)
    # harvest date per url (fallback ceiling) from ace_attempts.attempted_at
    at = pd.read_parquet(ATTEMPTS)[["url", "attempted_at"]].drop_duplicates("url")
    hdate = dict(zip(at["url"], at["attempted_at"].astype(str).str[:10]))

    rows = []
    for _, r in d.iterrows():
        claim = str(r["claim"]).strip()
        url = str(r["url"])
        ud = url_date(url)
        cd = ud or hdate.get(url) or "2026-07-03"
        tier, _, side = str(r["cell"]).partition("-")
        rows.append({
            "claim_id": hashlib.md5(claim.encode("utf-8")).hexdigest()[:12],
            "raw_context": claim,          # what verify_text verifies (= the synthetic-tweet text)
            "claim_date": cd,              # date ceiling for the FC-blocked live simulation
            "claim_date_source": "url" if ud else "harvest",
            "outlet": r["outlet"],
            "domain": urlparse(url).netloc.replace("www.", ""),
            "cell": r["cell"],
            "side": side,                  # L / R / C  -> the political-side experiment axis
            "reliability": tier,           # R / M / U
            "source_url": url,
            "headline": r.get("headline", ""),
            "risk_reason": r.get("risk_reason", ""),
            "extraction_source": r.get("extraction_source", "body"),  # body | headline
            "batch": r.get("batch", ""),
        })
    out = pd.DataFrame(rows).drop_duplicates("claim_id").reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    out.to_csv(OUT.replace(".parquet", ".csv"), index=False)

    print(f"wrote {len(out)} claims -> {OUT}", flush=True)
    print(f"\nclaim_date source: {dict(out.claim_date_source.value_counts())}", flush=True)
    print(f"date range: {out.claim_date.min()} .. {out.claim_date.max()}", flush=True)
    print("\nby side x reliability (the experiment grid):", flush=True)
    print(out.groupby(["side", "reliability"]).size().unstack(fill_value=0).to_string(), flush=True)
    print("\nsample rows:", flush=True)
    print(out[["claim_id", "claim_date", "claim_date_source", "side", "reliability", "raw_context"]]
          .head(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
