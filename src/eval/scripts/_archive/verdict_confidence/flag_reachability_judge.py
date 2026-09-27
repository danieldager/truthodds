"""LLM judge: is a FLAG reachable from the atomic claim + web evidence, or does it need the
raw post's framing? Blind to our verifier's verdict (avoids circularity) — sees only the claim
+ the fact-checker's stated reason. Three buckets:

  truth_determinable    — flag follows from the claim's own truth/accuracy, checkable from the
                          claim + web evidence (INCLUDING implied causation, authenticity/timing,
                          and whether an attributed statement was really made — all present in the
                          claim). A competent verifier (good retrieval, refutation-seeking) reaches it.
  extraction_recoverable— claim is literally true; flag hinges on a QUALIFIER the post had but the
                          distillation dropped ("under Labour", "vs last year"). Better extraction
                          would make it truth_determinable.
  needs_raw_post        — flag depends on how the post PRESENTS true facts (old event shown as
                          recent, sarcasm, juxtaposition, miscaption, spin) — unrecoverable.

Pilot: run on the 6 hand-dissected claims (compare to our labels) + a sample of soft_flag dev
claims. Run (from src/):
  uv run python -m eval.scripts.verdict_confidence.flag_reachability_judge -n 15
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
BUCKETS = {"truth_determinable", "extraction_recoverable", "needs_raw_post"}

# Our hand labels for the 6 dissected pass-side cases (clog/150626).
HAND = {"ev2215": "extraction_recoverable", "ev1229": "truth_determinable",
        "ev716": "truth_determinable", "ev1689": "truth_determinable",
        "ev569": "truth_determinable", "ev1630": "needs_raw_post"}

JUDGE_SYSTEM = """You are auditing a fact-checking EVALUATION dataset. Each row is a CLAIM a professional fact-checker reviewed and flagged, plus the fact-checker's headline and rating. The claim is a distilled statement; the original social-media POST is NOT available.

Our system verifies the CLAIM against web evidence and flags the post if the claim is false/misleading. Decide whether that system could reach the fact-checker's flag from the CLAIM + web evidence alone. Reason in TWO steps:

STEP 1 — On its OWN terms, is the CLAIM actually false, inaccurate, or internally misleading? A careful verifier (good sources, refutations, primary sources) checks not just surface wording but every assertion the claim makes — including an implied CAUSATION ("X rose AFTER Y" ⇒ was it due to Y?), the authenticity/timing of a described video/event, whether an attributed statement was really made, and whether ANY component of a multi-part claim is false. Answer true (claim is false/misleading on its own terms) or false (claim is essentially TRUE as stated).

STEP 2 — pick the bucket:
- "truth_determinable": STEP 1 = true. Verifying the claim REACHES the flag.
- "extraction_recoverable": STEP 1 = false (claim is essentially TRUE), but it was flagged because a concrete QUALIFIER the original post contained — a subject, timeframe, or comparison ("under Labour", "in 2025") — was dropped in distillation; restoring it would make the claim checkably false.
- "needs_raw_post": STEP 1 = false (claim is essentially TRUE), and the flag depends on how the post PRESENTS it (framing, juxtaposition, sarcasm, implied recency/context not reducible to a single qualifier) — unrecoverable from the claim or web evidence.

CRITICAL: if the claim is essentially TRUE as stated, it is NEVER truth_determinable — a truth-verifier would PASS it. Output strict JSON: {"step1_claim_false": <true|false>, "bucket": "<one of the three>", "rationale": "<one sentence>"}"""


def judge(claim: str, title: str, rating: str) -> dict:
    msg = [{"role": "system", "content": JUDGE_SYSTEM},
           {"role": "user", "content": (f"CLAIM: {claim}\n\nFACT-CHECKER HEADLINE: {title}\n"
                                        f"FACT-CHECKER RATING: {rating}\n\nClassify.")}]
    raw = _client.chat.completions.create(model=VERIFICATION_MODEL, messages=msg,
        response_format={"type": "json_object"}, temperature=0.1, max_tokens=1500
        ).choices[0].message.content or ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    d = json.loads(m.group(0)) if m else {}
    b = str(d.get("bucket", "")).strip()
    return {"bucket": b if b in BUCKETS else "?", "rationale": str(d.get("rationale", "")).strip()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--sample", type=int, default=15, help="# soft_flag dev claims to sample")
    args = ap.parse_args()
    df = pl.read_parquet(SPLITS)

    print("=== 6 hand-dissected cases (judge vs our hand label) ===")
    hits = 0
    for cid, hand in HAND.items():
        r = df.filter(pl.col("claim_id") == cid).to_dicts()[0]
        out = judge(r["claim_text"], r["review_title"] or "", r["original_rating"] or "")
        ok = "✓" if out["bucket"] == hand else "✗"
        hits += out["bucket"] == hand
        print(f"  {cid} {ok}  hand={hand:22} judge={out['bucket']:22}")
        print(f"       claim: {r['claim_text'][:75]}")
        print(f"       why:   {out['rationale']}")
    print(f"  agreement with hand labels: {hits}/{len(HAND)}\n")

    sample = df.filter((pl.col("split") == "dev") & pl.col("soft_flag")
                       & ~pl.col("claim_id").is_in(list(HAND))).head(args.sample)
    print(f"=== {sample.height} soft_flag DEV claims (eyeball) ===")
    counts: dict[str, int] = {}
    for r in sample.to_dicts():
        out = judge(r["claim_text"], r["review_title"] or "", r["original_rating"] or "")
        counts[out["bucket"]] = counts.get(out["bucket"], 0) + 1
        print(f"  [{out['bucket']:22}] {r['original_rating'][:14]:14} {r['claim_text'][:62]}")
        print(f"       {out['rationale']}")
    print(f"\nbucket counts (sample): {counts}")


if __name__ == "__main__":
    main()
