"""Seed loader + controlled-variant generator for the Tier-1 cache eval.

See README.md for the protocol and src/docs/tier1_cache_design.md for the cache
design this evaluates.

Seeds = correctly-verified AVeriTeC claims (the trusted "cache" contents). For
each seed we generate variants at controlled distance, each gold-labeled:
  equivalent  (should HIT): paraphrase / synonym / reorder / filler  -> same verdict
  adversarial (must MISS):  negation / number / entity / scope        -> verdict flips
The adversarial axes are NEAR in embedding space but FLIP the verdict — the
safety-critical cases. Generation is LLM (per-axis strict prompt); rules validate.

  uv run python -m eval.scripts.cache_eval.generator \
      -c eval/scripts/verification_grading/data/claims_n100.parquet \
      -v eval/scripts/verification_grading/data/verdicts_n100_tavily.parquet \
      -o eval/scripts/cache_eval/data/variants.parquet
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]  # -> src/
sys.path.insert(0, str(ROOT))
from openai import OpenAI  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from pipeline.embedding import embed  # noqa: E402

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
_FENCE = re.compile(r"^```.*?\n|```$", re.DOTALL)
_DIGITS = re.compile(r"\d+(?:[.,]\d+)?")
_client: OpenAI | None = None

# axis -> (is_equivalent, instruction). Equivalent variants must HIT the cache;
# adversarial variants must MISS it (their verdict differs from the seed's).
AXES: dict[str, tuple[bool, str]] = {
    "paraphrase": (True,
        "Rewrite the claim completely differently in wording but with EXACTLY the same "
        "meaning and truth conditions. Keep all numbers, named entities, negations, and "
        "scope identical."),
    "synonym": (True,
        "Rewrite the claim swapping key words for synonyms, keeping the meaning and truth "
        "identical. Keep numbers, named entities, negations, and scope identical."),
    "reorder": (True,
        "Restructure the claim (e.g. active<->passive voice, or front a different clause) "
        "without changing meaning or truth. Keep numbers, named entities, negations, and "
        "scope identical."),
    "filler": (True,
        "Add hedge/filler words (e.g. 'reportedly', 'as it turns out') without changing the "
        "core factual assertion or its truth. Keep numbers, named entities, negations, and "
        "scope identical."),
    "negation": (False,
        "Produce the logical NEGATION of the main factual assertion so its truth value FLIPS. "
        "Change only what is needed to negate it; keep the same entities, numbers, and topic. "
        "If the claim cannot be sensibly negated, reply exactly: N/A"),
    "number": (False,
        "Change a quantity / number / percentage / year in the claim to a DIFFERENT plausible "
        "value so the claim becomes factually different. Change nothing else. "
        "If there is no number to change, reply exactly: N/A"),
    "entity": (False,
        "Swap one named entity (person / place / organization) for a DIFFERENT real one of the "
        "same type, so the claim is about someone/something else. Change nothing else. "
        "If there is no named entity to swap, reply exactly: N/A"),
    "scope": (False,
        "Change the scope/quantifier so the meaning changes (e.g. some->all, a few->every, or "
        "flip a comparative's direction). Change nothing else. "
        "If there is no quantifier/scope to change, reply exactly: N/A"),
}

SYSTEM = ("You generate evaluation variants of factual claims for a fact-checking cache test. "
          "Output ONLY the single variant sentence, or exactly 'N/A' if the instruction does "
          "not apply. No quotes, no commentary.")


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://api.groq.com/openai/v1",
                         api_key=os.environ["GROQ_API_KEY"])
    return _client


def make_variant(claim: str, axis: str, model: str) -> str | None:
    """LLM-generate the `axis` variant of `claim`; None if N/A or unchanged."""
    _, instruction = AXES[axis]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"CLAIM: {claim}\n\nINSTRUCTION: {instruction}"}]
    for attempt in range(5):
        try:
            r = get_client().chat.completions.create(
                model=model, messages=msgs, temperature=0.4, max_tokens=300)
            txt = _FENCE.sub("", (r.choices[0].message.content or "").strip()).strip().strip('"')
            if not txt or txt.upper() == "N/A":
                return None
            if txt.strip().lower() == claim.strip().lower():
                return None  # unchanged -> useless variant
            return txt
        except Exception:  # noqa: BLE001
            if attempt == 4:
                return None
            time.sleep(min(2 ** attempt, 20.0))
    return None


def is_valid(claim: str, variant: str, axis: str) -> bool:
    """Lightweight per-axis sanity check on a generated variant.

    Catches the obvious generation failures (esp. an adversarial 'number' edit that
    didn't actually change a number). Equivalence/non-equivalence correctness is the
    gate's job at scoring time; this just filters junk.
    """
    if axis == "number":
        return set(_DIGITS.findall(claim)) != set(_DIGITS.findall(variant))
    return True


def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard — the lexical-distance companion to cosine."""
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def load_seed(claims_path: Path, verdicts_path: Path) -> pl.DataFrame:
    """Correctly-verified AVeriTeC claims = the trusted cache contents.

    Join on claim_id; keep rows where verdict_4class == gold_label.
    Returns: claim_id, claim_text, verdict (the agreed 4-class label).
    """
    c = pl.read_parquet(claims_path).select(["claim_id", "claim_text", "gold_label"])
    v = pl.read_parquet(verdicts_path).select(["claim_id", "verdict_4class"])
    j = c.join(v, on="claim_id", how="inner")
    correct = j.filter(pl.col("verdict_4class") == pl.col("gold_label"))
    return correct.select(["claim_id", "claim_text", pl.col("gold_label").alias("verdict")])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--claims", type=Path, required=True)
    ap.add_argument("-v", "--verdicts", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-w", "--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap #seeds (smoke test); 0 = all")
    args = ap.parse_args()

    seed = load_seed(args.claims, args.verdicts)
    if args.limit:
        seed = seed.head(args.limit)
    print(f"seeds (correctly-verified): {seed.height}  | axes: {len(AXES)}")

    jobs = [(r["claim_id"], r["claim_text"], r["verdict"], axis)
            for r in seed.iter_rows(named=True) for axis in AXES]

    rows: list[dict] = []
    n_na = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(make_variant, text, axis, args.model): (cid, text, verdict, axis)
                for (cid, text, verdict, axis) in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            cid, text, verdict, axis = futs[fut]
            variant = fut.result()
            if variant is None or not is_valid(text, variant, axis):
                n_na += 1
            else:
                rows.append({
                    "seed_claim_id": cid, "seed_text": text, "verdict": verdict,
                    "axis": axis, "gold_equivalent": AXES[axis][0],
                    "variant_text": variant,
                    "cosine": float(embed(text) @ embed(variant)),
                    "jaccard": jaccard(text, variant),
                })
            if i % 25 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] generated  (skipped/N-A: {n_na})")

    out = pl.DataFrame(rows).sort(["seed_claim_id", "axis"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out)
    print(f"\nwrote {out.height} variants -> {args.out}")
    eq = out.filter(pl.col("gold_equivalent"))
    ad = out.filter(~pl.col("gold_equivalent"))
    print(f"  equivalent (should HIT): {eq.height}  | mean cosine {eq['cosine'].mean():.3f}")
    print(f"  adversarial (must MISS): {ad.height}  | mean cosine {ad['cosine'].mean():.3f}")
    print("  per-axis counts:", out["axis"].value_counts().sort("axis").to_dicts())


if __name__ == "__main__":
    main()
