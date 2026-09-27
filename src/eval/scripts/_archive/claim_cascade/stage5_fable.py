"""Stage 5: FABLE harm rubric, one call per extracted claim (qwen/qwen3-32b).

Runs on the SAME claim set as the Stage-4 per-claim judge, with the SAME model,
so the headline "4-prong misinfo_candidate vs FABLE harm" comparison isolates
the rubric rather than the model. Each claim is scored on five 1-5 dimensions
(see prompts_scope.FABLE_SYSTEM); the post is supplied for context (mirrors the
judge's post+claim input) but only the claim is scored.

Output:
  data/fable_x862.parquet — one row per (post_id, claim_index):
    post_id, claim_index, claim_text,
    fragmentation, actionability, believability, spread_likelihood,
    exploitativeness, fable_total, fable_checkworthy,
    raw_response, latency_s, error

`fable_total` = sum of the five dimensions (5-25). `fable_checkworthy` =
(fable_total >= --fable-threshold) is DERIVED in code and re-computed across the
whole table on every run, so recalibrating the threshold is a re-run with a new
--fable-threshold (no LLM calls). Error rows default all dims to 0 (total 0,
checkworthy False), raw_response preserved for audit.

Resumable: per-claim rows keyed by (post_id, claim_index). 429 / transient 5xx
retried with exponential backoff. Mirrors extraction_grading/judge.py.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from openai import (  # noqa: E402
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from dotenv import load_dotenv  # noqa: E402

from eval.scripts.claim_cascade.prompts_scope import (  # noqa: E402
    FABLE_CHECKWORTHY_THRESHOLD,
    FABLE_DIMENSIONS,
    build_fable_messages,
    fable_total,
    parse_fable,
)

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "qwen/qwen3-32b"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
TEMPERATURE = 0.1
MAX_TOKENS = 4000  # reasoning-model headroom for <think>...</think>
MAX_RETRIES = 5

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
    return _client


FABLE_SCHEMA = {
    "post_id": pl.String,
    "claim_index": pl.Int64,
    "claim_text": pl.String,
    "fragmentation": pl.Int64,
    "actionability": pl.Int64,
    "believability": pl.Int64,
    "spread_likelihood": pl.Int64,
    "exploitativeness": pl.Int64,
    "fable_total": pl.Int64,
    "fable_checkworthy": pl.Boolean,
    "raw_response": pl.String,
    "latency_s": pl.Float64,
    "error": pl.String,
}


def _retryable(e: Exception) -> bool:
    """429, connection/timeout, and transient 5xx are retried; other 4xx are not.

    APITimeoutError / APIConnectionError are APIError subclasses with no
    status_code, so they must be matched by type, not by `status >= 500`.
    """
    if isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    status = getattr(e, "status_code", None)
    return isinstance(e, APIError) and isinstance(status, int) and status >= 500


def call_llm(messages: list[dict[str, str]], model: str) -> str:
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            if attempt == MAX_RETRIES or not _retryable(e):
                raise
            time.sleep(min(2 ** attempt + random.random(), 30.0))
    raise RuntimeError("unreachable")  # pragma: no cover


def score_one(post_id: str, claim_index: int, claim_text: str,
              post: str, model: str) -> dict:
    t0 = time.time()
    raw = ""
    dims = {k: 0 for k in FABLE_DIMENSIONS}
    total = 0
    error: str | None = None
    try:
        raw = call_llm(build_fable_messages(post, claim_text), model)
        dims = parse_fable(raw)
        total = fable_total(dims)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    return {
        "post_id": post_id,
        "claim_index": claim_index,
        "claim_text": claim_text,
        **dims,
        "fable_total": total,
        "fable_checkworthy": False,  # re-derived from the threshold at write time
        "raw_response": raw,
        "latency_s": round(time.time() - t0, 3),
        "error": error,
    }


def _load_existing(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=FABLE_SCHEMA)
    return pl.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-p", "--posts", type=Path,
                    default=DEFAULT_DATA_DIR / "posts_x862.parquet",
                    help="input posts parquet (post_id, text) — context for scoring")
    ap.add_argument("-e", "--extractions", type=Path,
                    default=DEFAULT_DATA_DIR / "stage3_extractions.parquet",
                    help="input extractions parquet (post_id, has_claim, claims)")
    ap.add_argument("-o", "--output", type=Path,
                    default=DEFAULT_DATA_DIR / "fable_x862.parquet",
                    help="output FABLE parquet (one row per claim)")
    ap.add_argument("--fable-threshold", type=int,
                    default=FABLE_CHECKWORTHY_THRESHOLD,
                    help="fable_total >= threshold => fable_checkworthy (5-25)")
    ap.add_argument("-w", "--workers", type=int, default=12)
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="cap input posts for smoke testing")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    if not args.posts.exists():
        raise SystemExit(f"posts parquet not found: {args.posts}")
    if not args.extractions.exists():
        raise SystemExit(f"extractions parquet not found: {args.extractions}")
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # unique(post_id): guard against a duplicated input post_id multiplying the
    # inner join below into duplicate (post_id, claim_index) jobs within one run.
    posts = (
        pl.read_parquet(args.posts).select(["post_id", "text"])
        .unique(subset=["post_id"], keep="first")
    )
    extractions = (
        pl.read_parquet(args.extractions).select(["post_id", "has_claim", "claims"])
        .unique(subset=["post_id"], keep="first")
    )
    if args.limit is not None:
        posts = posts.head(args.limit)

    # posts ⨝ extractions on post_id; restrict to the selected posts.
    work = posts.join(extractions, on="post_id", how="inner")

    existing = _load_existing(out_path)
    # Successes only — errored claims are re-attempted on re-run (dropped +
    # replaced before the parquet is rewritten).
    ok = existing.filter(pl.col("error").is_null())
    done = set(zip(ok["post_id"].to_list(), ok["claim_index"].to_list()))

    jobs: list[tuple[str, int, str, str]] = []  # (pid, idx, claim, post)
    for row in work.iter_rows(named=True):
        pid = row["post_id"]
        post_text = row["text"]
        for i, claim_text in enumerate(row["claims"] or []):
            if (pid, i) not in done:
                jobs.append((pid, i, claim_text, post_text))

    print(
        f"posts: {args.posts} ({len(posts)} rows, limit={args.limit})\n"
        f"extractions: {args.extractions} ({len(extractions)} rows)\n"
        f"output: {out_path} (existing: {len(done)} claims, to do: {len(jobs)})\n"
        f"model: {args.model}, workers: {args.workers}, "
        f"fable_threshold: {args.fable_threshold}"
    )

    if not jobs and len(existing) == 0:
        print("nothing to do.")
        return

    combined = existing
    if jobs:
        total = len(jobs)
        t0 = time.time()
        results: list[dict] = []
        n_err = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(score_one, pid, idx, claim, post, args.model)
                for pid, idx, claim, post in jobs
            ]
            for j, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                if r["error"]:
                    n_err += 1
                    print(f"  [{j}/{total}] {r['post_id']}/c{r['claim_index']} ERROR: {r['error']}")
                elif j % 20 == 0 or j == total:
                    print(f"  [{j}/{total}] done")
        new_df = pl.DataFrame(results, schema=FABLE_SCHEMA)
        # Drop (post_id, claim_index) pairs we just re-attempted so fresh results
        # replace prior errored rows instead of duplicating the key.
        job_keys = pl.DataFrame(
            {"post_id": [j[0] for j in jobs], "claim_index": [j[1] for j in jobs]},
            schema={"post_id": pl.String, "claim_index": pl.Int64},
        )
        existing_keep = existing.join(job_keys, on=["post_id", "claim_index"], how="anti")
        combined = pl.concat([existing_keep, new_df]) if len(existing_keep) else new_df
        elapsed = time.time() - t0
        print(f"\nscored {len(new_df)} new claims in {elapsed:.1f}s. errors: {n_err}/{total}")

    # Re-derive fable_checkworthy across ALL rows so the column always reflects
    # the current --fable-threshold (recalibration = re-run, no LLM calls).
    combined = combined.with_columns(
        (pl.col("fable_total") >= args.fable_threshold).alias("fable_checkworthy")
    )
    combined.write_parquet(out_path)

    n_cw = int(combined.filter(pl.col("fable_checkworthy")).height)
    print(
        f"wrote {len(combined)} FABLE rows -> {out_path}\n"
        f"  fable_checkworthy=true: {n_cw}/{len(combined)} "
        f"(threshold fable_total>={args.fable_threshold})"
    )


if __name__ == "__main__":
    main()
