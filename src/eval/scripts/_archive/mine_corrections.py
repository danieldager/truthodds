"""Mine corrected (true) counter-facts from refutation articles.

Produces a parquet of synthetic Supported claims that can later be merged
into the eval set. Each row carries `synthetic=True` and `source_url` (the
article that was mined) so the pipeline can blocklist that URL during
evidence retrieval (avoids circular reasoning).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl
import requests
import trafilatura
from tqdm import tqdm

from eval.harmonize import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

IN_PARQUET = Path("eval/data/eval_v1.parquet")
OUT_PARQUET = Path("eval/data/mined_corrections_v1.parquet")

EXCLUDE_PUBLISHERS = {"factcheck.afp.com"}  # anti-bot, fetch fails
N_SAMPLE = 50
MAX_BODY_CHARS = 6000

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

SYSTEM = (
    "You read a fact-check article that refuted a claim, and extract the "
    "verified counter-fact the article establishes (if any).\n\n"
    "Return JSON:\n"
    '  {"correction": "<single concise factual sentence, or null>",\n'
    '   "is_positive_assertion": true | false,\n'
    '   "reasoning": "<one short sentence>"}\n\n'
    "Rules:\n"
    "- The correction must be a standalone factual sentence that someone "
    "could independently verify. Examples of acceptable forms:\n"
    "  * Positive: 'The 2020 Ohio primary was delayed by Gov. DeWine.'\n"
    "  * Specific negation: 'Thom Tillis remains a U.S. Senator until "
    "January 2027.' (negates the false claim with a verifiable fact)\n"
    "- REJECT (return null) if the article's only finding is a media-forensic "
    "judgement with no substantive counter-fact (e.g., 'this image is "
    "AI-generated', 'the video is doctored', 'no evidence supports the "
    "claim' with no positive alternative).\n"
    "- REJECT vague generalities ('the situation is complicated').\n"
    "- `is_positive_assertion` is true if the correction states what IS the "
    "case (positive fact); false if it states what is NOT the case "
    "(verifiable negation).\n"
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
    user_msg = f"REFUTED CLAIM:\n{claim}\n\nFACT-CHECK ARTICLE (truncated):\n{body}"
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
    df = pl.read_parquet(IN_PARQUET)
    pool = (
        df.filter(
            (pl.col("harmonised_label") == "Refuted")
            & ~pl.col("publisher_site").is_in(EXCLUDE_PUBLISHERS)
        )
        # dedup by claim text (URL dedup happened earlier, but Newschecker has /_next/ duplicates)
        .unique(subset=["claim_text"])
    )
    print(f"Eligible Refuted claims (non-AFP, deduped by text): {len(pool)}", flush=True)

    sample = pool.sample(N_SAMPLE, seed=42, with_replacement=False).sort("publisher_site")
    print(f"Sampling {len(sample)}", flush=True)

    rows = []
    stats = {"fetch_fail": 0, "llm_fail": 0, "null": 0, "positive": 0, "negation": 0}

    for row in tqdm(sample.iter_rows(named=True), total=len(sample), desc="mine"):
        article = fetch_article(row["review_url"])
        if not article:
            stats["fetch_fail"] += 1
            continue
        try:
            res = extract_correction(row["claim_text"] or "", article)
        except Exception as e:
            stats["llm_fail"] += 1
            print(f"  [llm fail] {row['publisher_site']}: {e}", file=sys.stderr)
            continue
        correction = (res.get("correction") or "").strip()
        if not correction or correction.lower() == "null":
            stats["null"] += 1
            continue
        is_pos = bool(res.get("is_positive_assertion"))
        stats["positive" if is_pos else "negation"] += 1

        rows.append(
            {
                "claim_text": correction,
                "claimant": None,
                "claim_date": row["review_date"],
                "publisher_site": row["publisher_site"],
                "publisher_name": row["publisher_name"],
                "review_url": row["review_url"],  # for traceability
                "review_title": row["review_title"],
                "review_date": row["review_date"],
                "language_code": row["language_code"],
                "original_rating": "[mined from refutation]",
                "rule_label": None,
                "llm_label": "Supported",
                "harmonised_label": "Supported",
                "harmonisation_source": "mined",
                "binary_label": "pass",
                "synthetic": True,
                "source_url": row["review_url"],  # for circular-evidence blocklist
                "is_positive_assertion": is_pos,
                "mined_from_claim": row["claim_text"],  # the original refuted claim
                "reasoning": res.get("reasoning"),
            }
        )
        time.sleep(0.3)  # be polite

    print(f"\nStats: {stats}")
    if not rows:
        print("No rows extracted; nothing written.")
        return

    out = pl.DataFrame(rows)
    out.write_parquet(OUT_PARQUET)
    print(f"Wrote {OUT_PARQUET} ({len(out)} rows)")
    print()
    print("=== Sample of mined corrections ===")
    for r in out.sample(min(5, len(out)), seed=1).iter_rows(named=True):
        print(f"\n[{r['publisher_site']}] is_positive={r['is_positive_assertion']}")
        print(f"  ORIG: {r['mined_from_claim']}")
        print(f"  MINED: {r['claim_text']}")


if __name__ == "__main__":
    main()
