"""Run claim extraction on a posts_n{N}.parquet, write extractions_n{N}.parquet.

One LLM call per post (`openai/gpt-oss-120b` on Groq). Returns a JSON object
`{"has_claim": bool, "claims": [...]}`. See `prompts.EXTRACTION_SYSTEM`.

Output schema (one row per post_id):
    post_id      : str
    has_claim    : bool
    n_claims     : int
    claims       : list[str]
    raw_response : str
    latency_s    : float
    error        : str | null

Resumable: if the output parquet exists, post_ids already extracted are
skipped and new rows are appended. Per-row exceptions (API errors, JSON
parse failures) are recorded in the `error` column with raw_response saved
for audit.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from openai import OpenAI  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from eval.scripts.extraction_grading.prompts import (  # noqa: E402
    build_messages,
    parse_extraction,
)

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "openai/gpt-oss-120b"
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


OUTPUT_SCHEMA = {
    "post_id": pl.String,
    "has_claim": pl.Boolean,
    "n_claims": pl.Int64,
    "claims": pl.List(pl.String),
    "raw_response": pl.String,
    "latency_s": pl.Float64,
    "error": pl.String,
}


def call_llm(post: str, model: str) -> str:
    resp = get_client().chat.completions.create(
        model=model,
        messages=build_messages(post),
        temperature=0.1,
        max_tokens=4000,  # reasoning model headroom
    )
    return resp.choices[0].message.content or ""


def run_one(post_id: str, post: str, model: str) -> dict:
    t0 = time.time()
    raw = ""
    has_claim = False
    claims: list[str] = []
    error: str | None = None
    try:
        raw = call_llm(post, model)
        parsed = parse_extraction(raw)
        has_claim = parsed["has_claim"]
        claims = parsed["claims"]
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    return {
        "post_id": post_id,
        "has_claim": has_claim,
        "n_claims": len(claims),
        "claims": claims,
        "raw_response": raw,
        "latency_s": round(time.time() - t0, 3),
        "error": error,
    }


def _infer_default_output(input_path: Path) -> Path:
    """data/posts_n50.parquet -> data/extractions_n50.parquet."""
    m = re.search(r"posts_n(\d+)", input_path.stem)
    if m:
        return input_path.parent / f"extractions_n{m.group(1)}.parquet"
    return input_path.parent / f"extractions_{input_path.stem}.parquet"


def load_existing(path: Path) -> tuple[pl.DataFrame, set[str]]:
    if not path.exists():
        empty = pl.DataFrame(schema=OUTPUT_SCHEMA)
        return empty, set()
    df = pl.read_parquet(path)
    return df, set(df["post_id"].to_list())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i", "--input", type=Path,
        default=DEFAULT_DATA_DIR / "posts_n2000.parquet",
        help="input posts parquet",
    )
    ap.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output extractions parquet (default: data/extractions_n{N}.parquet)",
    )
    ap.add_argument("-w", "--workers", type=int, default=12)
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="cap rows for smoke testing")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    in_path: Path = args.input
    if not in_path.exists():
        raise SystemExit(f"input parquet not found: {in_path}")
    out_path: Path = args.output or _infer_default_output(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    posts = pl.read_parquet(in_path).select(["post_id", "text"])
    if args.limit is not None:
        posts = posts.head(args.limit)

    existing_df, done = load_existing(out_path)
    todo = posts.filter(~pl.col("post_id").is_in(list(done)))
    print(
        f"input: {in_path} ({len(posts)} rows, limit={args.limit})\n"
        f"output: {out_path} (existing: {len(done)} done, {len(todo)} to do)\n"
        f"model: {args.model}, workers: {args.workers}"
    )
    if len(todo) == 0:
        print("nothing to do.")
        return

    jobs = list(zip(todo["post_id"].to_list(), todo["text"].to_list()))
    t0 = time.time()
    results: list[dict] = []
    n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, pid, txt, args.model) for pid, txt in jobs]
        for j, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if r["error"]:
                n_err += 1
                print(f"  [{j}/{len(jobs)}] {r['post_id']} ERROR: {r['error']}")
            elif j % 20 == 0 or j == len(jobs):
                print(f"  [{j}/{len(jobs)}] done")

    new_df = pl.DataFrame(results, schema=OUTPUT_SCHEMA)
    combined = pl.concat([existing_df, new_df]) if len(existing_df) else new_df
    combined.write_parquet(out_path)

    elapsed = time.time() - t0
    n_has = int(new_df.filter(pl.col("has_claim")).height)
    print(
        f"\ndone in {elapsed:.1f}s. wrote {len(combined)} total rows to {out_path}\n"
        f"  new rows: {len(new_df)}, errors: {n_err}, has_claim=true: {n_has}"
    )


if __name__ == "__main__":
    main()
