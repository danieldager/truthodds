"""POC: mine corrected (true) counter-facts from refutation articles.

Picks 10 Refuted claims spread across publishers, fetches the fact-check
article, asks the LLM to extract the verified true correction (if any),
and prints results side-by-side. Cost: ~$0.0005 + 10 HTTP fetches.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import requests
import trafilatura

from eval.harmonize import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

PARQUET = Path("eval/data/eval_v1.parquet")
N_PER_PUBLISHER = 2
PUBLISHERS = ["snopes.com", "factcheck.afp.com", "fullfact.org", "newschecker.in", "politifact.com"]
MAX_BODY_CHARS = 6000

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

SYSTEM = (
    "You read a fact-check article that refuted a claim, and extract the "
    "verified TRUE counter-fact that the article establishes (if any).\n\n"
    "Return JSON:\n"
    '  {"correction": "<single concise factual sentence, or null>",\n'
    '   "reasoning": "<one sentence on what you found>"}\n\n'
    "Rules:\n"
    "- The correction must be a standalone factual statement, verifiable "
    "independently (e.g., 'The 2020 Ohio primary was delayed by Gov. DeWine, "
    "not cancelled by Dr. Amy Acton').\n"
    "- Return null if no clean counter-fact exists "
    "(e.g., the article only says 'this image is AI-generated' with no "
    "substantive true statement to extract).\n"
    "- Do NOT include hedges like 'according to the article'."
)


def fetch_article(url: str) -> str | None:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        if r.status_code != 200:
            return None
        return trafilatura.extract(r.text, include_comments=False, include_tables=False)
    except Exception:
        return None


def extract_correction(claim: str, article: str) -> dict:
    body = article[:MAX_BODY_CHARS]
    user_msg = (
        f"REFUTED CLAIM:\n{claim}\n\n"
        f"FACT-CHECK ARTICLE (truncated):\n{body}"
    )
    r = requests.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    r.raise_for_status()
    return json.loads(r.json()["choices"][0]["message"]["content"])


def main() -> None:
    df = pl.read_parquet(PARQUET).filter(
        (pl.col("harmonised_label") == "Refuted")
        & pl.col("publisher_site").is_in(PUBLISHERS)
    )
    sample = (
        df.group_by("publisher_site")
        .head(N_PER_PUBLISHER)
        .sort("publisher_site")
    )
    print(f"Mining {len(sample)} claims ({N_PER_PUBLISHER} per publisher × {len(PUBLISHERS)})")
    print()

    for i, row in enumerate(sample.iter_rows(named=True), 1):
        print("=" * 80)
        print(f"[{i}] {row['publisher_site']}  {row['review_date']}")
        print(f"CLAIM: {row['claim_text']}")
        print(f"ORIGINAL RATING: {row['original_rating']}")
        print(f"URL: {row['review_url']}")

        article = fetch_article(row["review_url"])
        if not article:
            print("  → fetch failed")
            continue
        print(f"  (scraped {len(article)} chars)")

        try:
            result = extract_correction(row["claim_text"] or "", article)
        except Exception as e:
            print(f"  → LLM error: {e}")
            continue

        print(f"CORRECTION: {result.get('correction')}")
        print(f"REASONING:  {result.get('reasoning')}")
        print()


if __name__ == "__main__":
    main()
