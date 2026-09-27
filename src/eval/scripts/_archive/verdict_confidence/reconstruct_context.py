"""Reconstruct a verdict-NEUTRAL "contextualized claim" per eval_v1 row, so the eval input
resembles what production extraction preserves from a real post — restoring the subject /
timeframe / entities / implied causation the distillation dropped, WITHOUT leaking the
fact-checker's conclusion.

eval_v1's real posts are unrecoverable (the FCT API never carried them); the fact-check itself
is the best surviving record of what the post asserted. The strict guardrail: reconstruct what
the POST CLAIMED, never the fact-checker's correction (no "but actually", no true/false/misleading).

Pilot: the 6 hand-dissected cases + a soft_flag dev sample → eyeball faithfulness + leakage.
Run (from src/): uv run python -m eval.scripts.verdict_confidence.reconstruct_context -n 6
"""
from __future__ import annotations

import argparse
import json
import re

import polars as pl
from openai import OpenAI

from config import VERIFICATION_API_KEY, VERIFICATION_BASE_URL, VERIFICATION_MODEL

_client = OpenAI(base_url=VERIFICATION_BASE_URL, api_key=VERIFICATION_API_KEY)
SPLITS = "eval/scripts/verdict_confidence/data/eval_v1_splits.parquet"
HAND = ["ev2215", "ev1229", "ev716", "ev1689", "ev569", "ev1630"]

RECON_SYSTEM = """You reconstruct the CONTEXT a social-media claim was made in, so a distilled fact-check claim resembles what an automated system extracts from a real post in production.

Given the distilled CLAIM, who made it (claimant), the date, and the fact-checker's HEADLINE, produce ONE "contextualized claim": the claim restored with the situational context the original POST carried but distillation dropped — the specific SUBJECT/scope, TIMEFRAME/recency, named ENTITIES, and any implied comparison or causation the post asserted.

STRICT RULES:
- Reconstruct what the POST ASSERTED, never the fact-checker's conclusion. Do NOT state or imply whether the claim is true, false, misleading, exaggerated, or lacking context. No corrections, no "but actually", no hedging, no rating words.
- Use the HEADLINE only to recover the subject/timeframe/entities; strip all of its verdict language.
- Add only context implied by the claim + claimant + date; invent no new facts or numbers.
- One declarative sentence, in the voice of the original poster.

Output strict JSON: {"contextualized_claim": "...", "added_context": "<what you restored, or 'none'>"}"""


def reconstruct(claim: str, claimant: str, date: str, title: str) -> dict:
    msg = [{"role": "system", "content": RECON_SYSTEM},
           {"role": "user", "content": (f"CLAIM: {claim}\nCLAIMANT: {claimant or 'unknown'}\n"
                                        f"DATE: {date or 'unknown'}\nFACT-CHECKER HEADLINE: {title}\n\n"
                                        f"Reconstruct the contextualized claim.")}]
    raw = _client.chat.completions.create(model=VERIFICATION_MODEL, messages=msg,
        response_format={"type": "json_object"}, temperature=0.2, max_tokens=1200
        ).choices[0].message.content or ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    d = json.loads(m.group(0)) if m else {}
    return {"contextualized_claim": str(d.get("contextualized_claim", "")).strip(),
            "added_context": str(d.get("added_context", "")).strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--sample", type=int, default=6, help="# soft_flag dev claims after the 6")
    args = ap.parse_args()
    df = pl.read_parquet(SPLITS)

    rows = df.filter(pl.col("claim_id").is_in(HAND)).to_dicts()
    rows += df.filter((pl.col("split") == "dev") & pl.col("soft_flag")
                      & ~pl.col("claim_id").is_in(HAND)).head(args.sample).to_dicts()
    for r in rows:
        out = reconstruct(r["claim_text"], r["claimant"], r["claim_date"], r["review_title"] or "")
        tag = "HAND" if r["claim_id"] in HAND else "dev "
        print("=" * 96)
        print(f"{tag} {r['claim_id']}  ({r['original_rating'][:22]})")
        print(f"  original : {r['claim_text']}")
        print(f"  + context: {out['contextualized_claim']}")
        print(f"  restored : {out['added_context']}")
        print(f"  headline : {r['review_title']}")  # for leakage eyeball (NOT given to the verifier)


if __name__ == "__main__":
    main()
