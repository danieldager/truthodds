"""Load AVeriTeC dev claims for Tier 3 verification grading.

Loads pminervini/averitec dev split (~500 claims) via HF datasets, samples N
reproducibly, dumps to data/claims_n{N}.parquet. The `claim_id` is a stable
sha1[:16] hash of the original_claim_url so downstream parquets can join
cleanly.
"""
from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import polars as pl
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    out = args.output or Path(__file__).parent / "data" / f"claims_n{args.n}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    print("loading pminervini/averitec dev split")
    ds = load_dataset("pminervini/averitec", split="dev")
    print(f"total claims in dev: {len(ds)}")

    indices = list(range(len(ds)))
    random.seed(args.seed)
    random.shuffle(indices)
    sample = [ds[i] for i in indices[: args.n]]

    rows = []
    for ex in sample:
        # claim_id must be UNIQUE per claim. Several AVeriTeC claims share one
        # fact-check URL, so hashing the URL alone collides — include the claim text.
        url_key = ex.get("original_claim_url") or ""
        rows.append({
            "claim_id": stable_id(f"{ex.get('claim', '')}||{url_key}"),
            "claim_text": ex.get("claim", ""),
            "gold_label": ex.get("label", ""),
            "claim_date": ex.get("claim_date") or None,
            "speaker": ex.get("speaker") or None,
            "fact_checking_article": ex.get("fact_checking_article") or None,
            "gold_justification": ex.get("justification") or "",
            "original_claim_url": ex.get("original_claim_url") or None,
        })

    df = pl.DataFrame(rows)
    # The AVeriTeC sample can contain a claim twice (same text + URL); drop exact dups so
    # claim_id is unique and downstream joins are 1:1.
    before = df.height
    df = df.unique(subset="claim_id", keep="first", maintain_order=True)
    if df.height < before:
        print(f"dropped {before - df.height} duplicate claim(s)")
    df.write_parquet(out)
    print(f"wrote {df.height} rows -> {out}")
    print("\nlabel distribution:")
    for row in df["gold_label"].value_counts(sort=True).iter_rows(named=True):
        print(f"  {row['gold_label']:<40} {row['count']}")


if __name__ == "__main__":
    main()
