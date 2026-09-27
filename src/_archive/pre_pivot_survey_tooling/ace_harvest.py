"""Run ACE over a batch of articles and PERSIST every attempt (for posterity).

For each article we attempt, we log source + url + headline + the ACE result (claims or NONE),
including fetch failures, to a parquet. This is the durable record of what was harvested and what
ACE returned, so nothing is lost and the NONE-rate per source is auditable.
"""
import datetime
import json
import os

import pandas as pd
import trafilatura

from eval import ace

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "data", "ace_attempts.parquet")


def fetch_body(url: str) -> str | None:
    dl = trafilatura.fetch_url(url)
    if not dl:
        return None
    return trafilatura.extract(dl, include_comments=False, include_tables=True)


def process(articles: list[dict], out_path: str = DEFAULT_OUT, min_chars: int = 200) -> pd.DataFrame:
    """articles: list of {source, url, headline, ...optional meta}. Persists all attempts.

    status ∈ {claims, none, fetch_failed}. Appends to out_path, deduped by url (keep latest).
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")
    rows = []
    for a in articles:
        url = a.get("url", "")
        body = fetch_body(url) if url else None
        rec = {"source": a.get("source", ""), "url": url, "headline": a.get("headline", ""),
               "attempted_at": now, "body_chars": len(body) if body else 0}
        if not body or len(body) < min_chars:
            rec.update({"status": "fetch_failed", "n_claims": 0, "claims_json": "[]"})
        else:
            r = ace.extract(body, headline=a.get("headline"))
            rec.update({"status": "none" if r["n"] == 0 else "claims",
                        "n_claims": r["n"], "claims_json": json.dumps(r["claims"], ensure_ascii=False)})
        rows.append(rec)
    df = pd.DataFrame(rows)
    if os.path.exists(out_path):
        old = pd.read_parquet(out_path)
        df = pd.concat([old, df]).drop_duplicates(subset="url", keep="last").reset_index(drop=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
