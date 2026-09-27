"""Per-source summary of the shortlist harvest for the survey claim pool (#13/#14).
One row per outlet: cell, reliability tier, political orientation, NewsGuard score+rating,
Tranco traffic rank, fetch method, articles processed (with breakdown), and claims extracted.
Writes survey_claims/sources_pull_summary.csv (tracked deliverable; NewsGuard-derived -> internal/private only).
"""
import pandas as pd
BASE = "src/eval/data/survey_claims"
NG = "src/eval/data/newsguard/label-full-metadata-20241215.csv"
ATT = "src/eval/data/ace_attempts.parquet"

sl = pd.read_csv(f"{BASE}/source_shortlist.csv")            # domain, cell, ng_score, tranco, access, note
cl = pd.read_parquet(f"{BASE}/shortlist_claims.parquet")    # per-claim (this pull)
at = pd.read_parquet(ATT)

# NewsGuard score + T/N rating per domain (authoritative, fills shortlist gaps)
ng = pd.read_csv(NG, usecols=["Domain", "Rating", "Score"])
ng["dom"] = ng.Domain.str.lower().str.replace(r"^www\.", "", regex=True)
ng = ng.drop_duplicates("dom").set_index("dom")
ng_rating, ng_score = ng["Rating"], ng["Score"]

# this pull's attempts = ace_attempts rows whose source is a shortlist DOMAIN
# (older harvests logged source=outlet-NAME, so this cleanly isolates the new run)
doms = set(sl.domain)
mine = at[at.source.isin(doms)]
status = mine.groupby("source").status.value_counts().unstack(fill_value=0)
for c in ["claims", "none", "off_topic", "fetch_failed"]:
    if c not in status:
        status[c] = 0
status["n_articles"] = status[["claims", "none", "off_topic", "fetch_failed"]].sum(axis=1)

n_claims = cl.groupby("domain").size().rename("n_claims")

out = sl.copy()
out["reliability"] = out.cell.str.split("-").str[0]
out["orientation"] = out.cell.str.split("-").str[1]
out["newsguard_score"] = out.domain.map(ng_score)
out["newsguard_rating"] = out.domain.map(ng_rating)
out = out.rename(columns={"tranco": "tranco_rank", "access": "fetch_method"})
out = out.merge(status.reset_index().rename(columns={"source": "domain"}), on="domain", how="left")
out = out.merge(n_claims.reset_index(), on="domain", how="left")
out = out.rename(columns={"claims": "articles_with_claims", "none": "articles_no_claims",
                          "off_topic": "articles_off_topic", "fetch_failed": "articles_fetch_failed"})
for c in ["n_articles", "n_claims", "articles_with_claims", "articles_no_claims", "articles_off_topic", "articles_fetch_failed"]:
    out[c] = out[c].fillna(0).astype(int)

cols = ["cell", "reliability", "orientation", "domain", "newsguard_score", "newsguard_rating",
        "tranco_rank", "fetch_method", "n_articles", "articles_with_claims", "articles_no_claims",
        "articles_off_topic", "articles_fetch_failed", "n_claims"]
out = out[cols].sort_values(["cell", "n_claims"], ascending=[True, False]).reset_index(drop=True)
out.to_csv(f"{BASE}/sources_pull_summary.csv", index=False)
print(f"wrote {len(out)} sources -> sources_pull_summary.csv")
print("claims by cell:", cl.groupby(cl.cell).size().to_dict())
print(out.to_string(index=False))
