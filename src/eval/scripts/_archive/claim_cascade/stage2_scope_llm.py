"""Stage 2: LLM scope gate over ALL posts (openai/gpt-oss-120b on Groq).

One call per post -> {in_scope, topic, reason}. See
`prompts_scope.SCOPE_CONFIRM_SYSTEM`. Run on all 862 (not just the
embedding-positive subset) so we can measure embedding<->LLM agreement; the
LLM is the real scope gate.

Outputs:
  data/scope_x862.parquet  (one row per post_id):
      post_id, in_scope_llm, topic_llm, reason, raw_response, latency_s, error
  data/posts_inscope.parquet  (post_id, text where in_scope_llm) — the Stage-3
      extraction gate. Re-derived from the FULL scope table on every run so it
      always matches the latest scope_x862.parquet.

Resumable: post_ids already scored are skipped and new rows appended. Per-row
API/parse exceptions are captured in `error` (in_scope_llm defaults False, so
errored posts are excluded from the gate). 429 / transient 5xx are retried with
exponential backoff. Mirrors extraction_grading/extract.py.
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
    build_scope_messages,
    parse_scope,
)

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "openai/gpt-oss-120b"
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


OUTPUT_SCHEMA = {
    "post_id": pl.String,
    "in_scope_llm": pl.Boolean,
    "topic_llm": pl.String,
    "reason": pl.String,
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


def run_one(post_id: str, post: str, model: str) -> dict:
    t0 = time.time()
    raw = ""
    in_scope = False
    topic = ""
    reason = ""
    error: str | None = None
    try:
        raw = call_llm(build_scope_messages(post), model)
        parsed = parse_scope(raw)
        in_scope = parsed["in_scope"]
        topic = parsed["topic"]
        reason = parsed["reason"]
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    return {
        "post_id": post_id,
        "in_scope_llm": in_scope,
        "topic_llm": topic,
        "reason": reason,
        "raw_response": raw,
        "latency_s": round(time.time() - t0, 3),
        "error": error,
    }


def load_existing(path: Path) -> tuple[pl.DataFrame, set[str]]:
    """Return (full existing df, set of SUCCESSFULLY-scored post_ids).

    Errored rows are excluded from the done-set so a re-run retries them; they
    are dropped and replaced before the parquet is rewritten (see main()).
    """
    if not path.exists():
        return pl.DataFrame(schema=OUTPUT_SCHEMA), set()
    df = pl.read_parquet(path)
    done = set(df.filter(pl.col("error").is_null())["post_id"].to_list())
    return df, done


def write_gate(scope_df: pl.DataFrame, posts_full: pl.DataFrame, gate_path: Path) -> int:
    """Re-derive posts_inscope.parquet (post_id, text) from the full scope table."""
    gate = (
        scope_df.filter(pl.col("in_scope_llm"))
        .select("post_id")
        .join(posts_full.select(["post_id", "text"]), on="post_id", how="inner")
        .select(["post_id", "text"])
    )
    gate.write_parquet(gate_path)
    return len(gate)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", type=Path,
                    default=DEFAULT_DATA_DIR / "posts_x862.parquet",
                    help="input posts parquet (post_id, text)")
    ap.add_argument("-o", "--output", type=Path,
                    default=DEFAULT_DATA_DIR / "scope_x862.parquet",
                    help="output scope parquet")
    ap.add_argument("--gate-output", type=Path,
                    default=DEFAULT_DATA_DIR / "posts_inscope.parquet",
                    help="output in-scope gate parquet (post_id, text)")
    ap.add_argument("-w", "--workers", type=int, default=12)
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="cap rows for smoke testing")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    in_path: Path = args.input
    if not in_path.exists():
        raise SystemExit(f"input parquet not found: {in_path}")
    out_path: Path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    posts_full = pl.read_parquet(in_path).select(["post_id", "text"])
    posts = posts_full.head(args.limit) if args.limit is not None else posts_full

    existing_df, done = load_existing(out_path)
    todo = posts.filter(~pl.col("post_id").is_in(list(done)))
    print(
        f"input: {in_path} ({len(posts)} rows, limit={args.limit})\n"
        f"output: {out_path} (existing: {len(done)} done, {len(todo)} to do)\n"
        f"gate: {args.gate_output}\n"
        f"model: {args.model}, workers: {args.workers}"
    )

    combined = existing_df
    if len(todo) > 0:
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
        # Drop rows we just re-attempted (previously-errored post_ids in `todo`)
        # so fresh results replace them instead of duplicating the key.
        retried = set(todo["post_id"].to_list())
        existing_keep = existing_df.filter(~pl.col("post_id").is_in(list(retried)))
        combined = pl.concat([existing_keep, new_df]) if len(existing_keep) else new_df
        combined.write_parquet(out_path)

        elapsed = time.time() - t0
        n_scope = int(new_df.filter(pl.col("in_scope_llm")).height)
        print(
            f"\ndone in {elapsed:.1f}s. wrote {len(combined)} total rows to {out_path}\n"
            f"  new rows: {len(new_df)}, errors: {n_err}, in_scope_llm=true (new): {n_scope}"
        )
    else:
        print("scope: nothing to do (all posts already scored).")

    if len(combined) == 0:
        print("no scope rows yet — gate not written.")
        return

    n_gate = write_gate(combined, posts_full, args.gate_output)
    n_in_scope_total = int(combined.filter(pl.col("in_scope_llm")).height)
    if n_gate < n_in_scope_total:
        print(
            f"  WARNING: {n_in_scope_total - n_gate} in-scope post(s) are missing "
            f"from --input ({in_path}) and were dropped from the gate. "
            f"--input must be a superset of the scored posts."
        )
    print(
        f"gate: wrote {n_gate} in-scope posts to {args.gate_output} "
        f"({n_in_scope_total}/{len(combined)} scored posts in scope)"
    )


if __name__ == "__main__":
    main()
