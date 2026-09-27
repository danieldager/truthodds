"""FN-rate sample check for v4 extraction.

The v4 design judges only extracted claims (has_claim=true posts), so FN
(misinformation candidates the extractor missed) is unmeasurable from the
main pipeline. This script fills that gap: it samples N posts where the
extractor returned has_claim=false, sends each to qwen3-32b with a
post-level "is there a missed misinformation candidate?" prompt, and
reports the rate.

Uses the SAME misinformation criterion as the extractor (imported from
prompts.py) to keep extractor / judge / FN-check definitionally aligned.

Output parquet (one row per sampled post):
    post_id, text,
    missed_misinfo_candidate (bool),
    missed_candidate_claim (str — the judge's proposed claim if any),
    rationale (str),
    raw_response (str),
    latency_s (float),
    error (str | null)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from eval.scripts.extraction_grading.prompts import _MISINFO_CRITERION, _clean  # noqa: E402

load_dotenv(ROOT / ".env")

DEFAULT_JUDGE_MODEL = "qwen/qwen3-32b"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
    return _client


FN_CHECK_SYSTEM = f"""You are an expert fact-checker reviewing whether another system MISSED a misinformation-candidate claim in a social media post. The other system returned has_claim=false (no extraction). Your job is to independently judge whether the post contains at least one misinformation-candidate claim per the criterion below.

{_MISINFO_CRITERION}

Answer two things:
1. missed_misinfo_candidate: Does the post contain at least one misinformation-candidate claim that the extractor should have produced? (true/false)
2. missed_candidate_claim: If yes, what claim should have been extracted (one sentence, in the form an extractor would produce — decontextualized, no pronouns)? If no, empty string.

Return strict JSON only, with this shape and nothing else:
{{"missed_misinfo_candidate": <true|false>, "missed_candidate_claim": "<string>", "rationale": "<one-sentence reason>"}}

Be conservative: only return missed_misinfo_candidate=true if you are confident the post contains a clear misinformation-candidate claim per all four criteria above. The extractor is designed to lean false on borderline cases — that is by design, not a defect. Only flag clear misses."""


def _call_llm(messages, model: str) -> str:
    resp = get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=4000,
    )
    return resp.choices[0].message.content or ""


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    raise ValueError(f"not coercible to bool: {value!r}")


def _parse_fn_response(raw: str) -> dict:
    cleaned = _clean(raw)
    if not cleaned:
        raise ValueError("empty response")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_OBJECT.search(cleaned)
        if not m:
            raise ValueError(f"no JSON object: {cleaned[:200]!r}")
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError(f"expected object, got {type(data).__name__}")
    return {
        "missed_misinfo_candidate": _coerce_bool(data.get("missed_misinfo_candidate")),
        "missed_candidate_claim": str(data.get("missed_candidate_claim", "")),
        "rationale": str(data.get("rationale", "")),
    }


def fn_check_one(post_id: str, post: str, model: str) -> dict:
    t0 = time.time()
    raw = ""
    parsed = {"missed_misinfo_candidate": False, "missed_candidate_claim": "", "rationale": ""}
    error: str | None = None
    messages = [
        {"role": "system", "content": FN_CHECK_SYSTEM},
        {"role": "user", "content": f"POST:\n{post}"},
    ]
    try:
        raw = _call_llm(messages, model)
        parsed = _parse_fn_response(raw)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    return {
        "post_id": post_id,
        "text": post,
        "missed_misinfo_candidate": parsed["missed_misinfo_candidate"],
        "missed_candidate_claim": parsed["missed_candidate_claim"],
        "rationale": parsed["rationale"],
        "raw_response": raw,
        "latency_s": round(time.time() - t0, 3),
        "error": error,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-p", "--posts", type=Path,
        default=DEFAULT_DATA_DIR / "posts_n1000.parquet",
    )
    ap.add_argument(
        "-e", "--extractions", type=Path,
        default=DEFAULT_DATA_DIR / "extractions_n1000.parquet",
    )
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("-n", "--sample", type=int, default=100,
                    help="how many has_claim=false posts to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-w", "--workers", type=int, default=4)
    ap.add_argument("-m", "--model", default=DEFAULT_JUDGE_MODEL)
    args = ap.parse_args()

    if not args.posts.exists():
        raise SystemExit(f"posts parquet not found: {args.posts}")
    if not args.extractions.exists():
        raise SystemExit(f"extractions parquet not found: {args.extractions}")

    # Default output path: data/fn_check_n{sample}_of_{N}.parquet
    if args.output is None:
        m = re.search(r"posts_n(\d+)", args.posts.stem)
        n_full = m.group(1) if m else "all"
        args.output = args.posts.parent / f"fn_check_n{args.sample}_of_n{n_full}.parquet"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    posts = pl.read_parquet(args.posts).select(["post_id", "text"])
    extractions = pl.read_parquet(args.extractions).select(["post_id", "has_claim"])
    joined = posts.join(extractions, on="post_id", how="inner")
    neg = joined.filter(~pl.col("has_claim"))
    print(f"has_claim=false pool: {neg.height} posts (of {joined.height} extracted)")
    if neg.height == 0:
        raise SystemExit("no has_claim=false posts to sample from")

    sample_size = min(args.sample, neg.height)
    # Reproducible sample via post_id sort + seeded shuffle
    rng = random.Random(args.seed)
    pool = neg.sort("post_id").to_dicts()
    rng.shuffle(pool)
    sample = pool[:sample_size]

    print(f"sampling {sample_size} posts; output: {args.output}")
    print(f"model: {args.model}, workers: {args.workers}")

    t0 = time.time()
    results: list[dict] = []
    n_err = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool_ex:
        futures = {
            pool_ex.submit(fn_check_one, r["post_id"], r["text"], args.model): r["post_id"]
            for r in sample
        }
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if r["error"]:
                n_err += 1
                print(f"  [{i}/{sample_size}] {r['post_id'][:8]} ERROR: {r['error']}")
            elif i % 10 == 0 or i == sample_size:
                print(f"  [{i}/{sample_size}] done")

    df = pl.DataFrame(results, schema={
        "post_id": pl.String,
        "text": pl.String,
        "missed_misinfo_candidate": pl.Boolean,
        "missed_candidate_claim": pl.String,
        "rationale": pl.String,
        "raw_response": pl.String,
        "latency_s": pl.Float64,
        "error": pl.String,
    })
    df.write_parquet(args.output)

    n_missed = int(df["missed_misinfo_candidate"].fill_null(False).sum())
    rate = n_missed / sample_size
    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.1f}s. errors: {n_err}/{sample_size}")
    print(f"missed_misinfo_candidate=true: {n_missed}/{sample_size} ({rate:.1%})")
    print(f"  -> estimated FN rate on the has_claim=false pool: {rate:.1%}")
    print(f"  -> projected absolute FNs across {neg.height} pool: ~{rate * neg.height:.0f}")
    print(f"wrote {df.height} rows -> {args.output}")


if __name__ == "__main__":
    main()
