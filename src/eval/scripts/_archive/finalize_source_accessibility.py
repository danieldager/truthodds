"""Turn the raw audit into an HONEST per-source artifact.

Only RSS-feed-sourced rows are trustworthy (real article URLs). The homepage-scrape fallback grabbed
asset/hub URLs, so its verdicts are unreliable -> those outlets are UNDETERMINED, not paywalled.
Output: a clean workbook with a per-source verdict, a grid of harvest-now-free outlets, and the raw
evidence. Also prints the actionable summary.
"""
import pandas as pd

REPO = "src"
RAW = REPO + "/eval/data/survey_claims/source_accessibility_full.xlsx"
OUT = REPO + "/eval/data/survey_claims/source_accessibility.xlsx"
READABLE = 1000

art = pd.read_excel(RAW, sheet_name="by_article")


def src_type(feed):
    f = str(feed)
    return "rss" if f.startswith("http") else ("homepage" if f == "homepage" else "none")


art["src_type"] = art["feed"].map(src_type)
rows = []
for outlet, g in art.groupby("outlet", sort=False):
    cell, domain = g["cell"].iloc[0], g["domain"].iloc[0]
    st = g["src_type"].iloc[0]
    tested = g[g["status"].isin(["readable", "short", "failed"])]
    if st != "rss" or not len(tested):
        # feed discovery failed or only homepage-junk URLs -> accessibility UNKNOWN
        rows.append(dict(outlet=outlet, cell=cell, domain=domain, article_source=st,
                         n=len(tested), readable_rate=0.0, median_chars=0,
                         verdict="UNDETERMINED", harvest_now="unknown (need real feed/sitemap)"))
        continue
    rate = (tested["status"] == "readable").mean()
    med = int(tested["body_chars"].median())
    if rate >= 0.6 and med >= READABLE:
        v, hn = "FREE", "yes — trafilatura"
    elif med < 500:
        v, hn = "PAYWALLED", "no — needs Jina/render"
    else:
        v, hn = "PARTIAL", "maybe — metered"
    rows.append(dict(outlet=outlet, cell=cell, domain=domain, article_source="rss",
                     n=len(tested), readable_rate=round(rate, 2), median_chars=med,
                     verdict=v, harvest_now=hn))
src = pd.DataFrame(rows)
order = {"FREE": 0, "PARTIAL": 1, "PAYWALLED": 2, "UNDETERMINED": 3}
src = src.sort_values(["verdict", "readable_rate"], key=lambda c: c.map(order) if c.name == "verdict" else c,
                      ascending=[True, False])

# grid of harvest-now-free outlets by reliability x bias cell
free = src[src["verdict"] == "FREE"]
tiers, sides = ["R", "M", "U"], ["L", "C", "R"]
grid_rows = []
for t in tiers:
    for s in sides:
        cell = f"{t}-{s}"
        outs = sorted(free[free["cell"] == cell]["outlet"].tolist())
        grid_rows.append(dict(cell=cell, n_free=len(outs), free_outlets=", ".join(outs)))
grid = pd.DataFrame(grid_rows)

with pd.ExcelWriter(OUT, engine="openpyxl") as w:
    src.to_excel(w, sheet_name="by_source", index=False)
    grid.to_excel(w, sheet_name="grid_free_now", index=False)
    art[["outlet", "cell", "domain", "src_type", "art_url", "body_chars", "status"]].to_excel(
        w, sheet_name="by_article_raw", index=False)

vc = src["verdict"].value_counts().to_dict()
print("VERDICTS:", vc, "\n")
print("=== HARVEST-NOW-FREE outlets per reliability x bias cell ===")
print(grid.to_string(index=False), "\n")
print("=== TRUE PAYWALLED (RSS-confirmed, free path fails -> Jina) ===")
print(src[src.verdict == "PAYWALLED"][["outlet", "cell", "median_chars"]].to_string(index=False), "\n")
print("=== UNDETERMINED (no usable feed found; NOT necessarily paywalled) ===")
print(", ".join(src[src.verdict == "UNDETERMINED"]["outlet"].tolist()))
print(f"\nFREE now: {vc.get('FREE',0)}  |  wrote {OUT}")
