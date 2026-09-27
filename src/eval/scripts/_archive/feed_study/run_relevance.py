"""Run the Relevance (public-import) filter in one of three input modes.

  --mode raw    judge each RAW POST.            input: posts parquet. one row/post.
  --mode claim  judge each CLAIM (no post).     input: a claimify parquet's claims.
  --mode both   judge each CLAIM with post ctx. input: a claimify parquet's claims.

Mirrors run_fable.py (client, retry, ThreadPoolExecutor, resumable parquet). The
filter is independent/composable; the same prompt judges post or claim.

  uv run python -m eval.scripts.feed_study.run_relevance --mode raw  -m openai/gpt-oss-120b
  uv run python -m eval.scripts.feed_study.run_relevance --mode both -m qwen/qwen3-32b \
      --claims data/sample1000/claimify_gpt-oss-120b.parquet --outdir data/sample1000

Output: <outdir>/relevance_<mode>_<modeltag>.parquet. Resumable on (post_id, claim_index).
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

ROOT = Path(__file__).resolve().parents[3]  # -> src/
sys.path.insert(0, str(ROOT))

from openai import (  # noqa: E402
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from dotenv import load_dotenv  # noqa: E402

from eval.scripts.feed_study.prompts_relevance import (  # noqa: E402
    build_relevance_both,
    build_relevance_messages,
    parse_relevance,
)

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
TEMPERATURE = 0.1
MAX_TOKENS = 4000
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


RELEVANCE_SCHEMA = {
    "post_id": pl.String,
    "claim_index": pl.Int64,
    "unit_text": pl.String,
    "model": pl.String,
    "mode": pl.String,
    "in_scope": pl.Boolean,
    "domain": pl.String,
    "reason": pl.String,
    "raw_response": pl.String,
    "latency_s": pl.Float64,
    "error": pl.String,
}


def _retryable(e: Exception) -> bool:
    if isinstance(e, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    status = getattr(e, "status_code", None)
    return isinstance(e, APIError) and isinstance(status, int) and status >= 500


def call_llm(messages: list[dict[str, str]], model: str) -> str:
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = get_client().chat.completions.create(
                model=model, messages=messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            if attempt == MAX_RETRIES or not _retryable(e):
                raise
            time.sleep(min(2 ** attempt + random.random(), 30.0))
    raise RuntimeError("unreachable")  # pragma: no cover


def score_one(post_id: str, claim_index: int, unit_text: str, post: str,
              mode: str, model: str) -> dict:
    t0 = time.time()
    raw = ""
    out = {"in_scope": None, "domain": None, "reason": None}
    error: str | None = None
    try:
        msgs = (build_relevance_both(post, unit_text) if mode == "both"
                else build_relevance_messages(unit_text))
        raw = call_llm(msgs, model)
        out = parse_relevance(raw)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    return {
        "post_id": post_id,
        "claim_index": claim_index,
        "unit_text": unit_text,
        "model": model,
        "mode": mode,
        "in_scope": out["in_scope"],
        "domain": out["domain"],
        "reason": out["reason"],
        "raw_response": raw,
        "latency_s": round(time.time() - t0, 3),
        "error": error,
    }


def _build_jobs_raw(posts: pl.DataFrame) -> list[tuple]:
    jobs = []
    for r in posts.iter_rows(named=True):
        text = (r["text"] or "").strip()
        if text:
            jobs.append((r["post_id"], -1, text, text))
    return jobs


def _build_jobs_claim(posts: pl.DataFrame, claims_path: Path) -> list[tuple]:
    if not claims_path.exists():
        raise SystemExit(f"--claims parquet not found: {claims_path}")
    cl = pl.read_parquet(claims_path).select(["post_id", "claims"])
    post_text = {r["post_id"]: (r["text"] or "") for r in posts.iter_rows(named=True)}
    jobs = []
    for r in cl.iter_rows(named=True):
        pid = r["post_id"]
        for i, claim in enumerate(r["claims"] or []):
            c = (claim or "").strip()
            if c:
                jobs.append((pid, i, c, post_text.get(pid, "")))
    return jobs


def _load_existing(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=RELEVANCE_SCHEMA)
    return pl.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["raw", "claim", "both"], required=True)
    ap.add_argument("-p", "--posts", type=Path, default=DEFAULT_DATA_DIR / "posts_x862.parquet")
    ap.add_argument("--claims", type=Path, default=None,
                    help="claim/both modes: a claimify_<model>.parquet")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-w", "--workers", type=int, default=12)
    ap.add_argument("-n", "--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.posts.exists():
        raise SystemExit(f"posts parquet not found: {args.posts}")
    if args.mode in ("claim", "both") and args.claims is None:
        raise SystemExit(f"--mode {args.mode} requires --claims <claimify parquet>")

    tag = args.model.split("/")[-1]
    out_path: Path = args.output or (args.outdir / f"relevance_{args.mode}_{tag}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    posts = (pl.read_parquet(args.posts).select(["post_id", "text"])
             .unique(subset=["post_id"], keep="first", maintain_order=True))
    all_jobs = (_build_jobs_raw(posts) if args.mode == "raw"
                else _build_jobs_claim(posts, args.claims))
    if args.limit is not None:
        all_jobs = all_jobs[:args.limit]

    existing = _load_existing(out_path)
    ok = existing.filter(pl.col("error").is_null())
    done = set(zip(ok["post_id"].to_list(), ok["claim_index"].to_list()))
    jobs = [j for j in all_jobs if (j[0], j[1]) not in done]

    print(f"mode: {args.mode} | posts: {len(posts)} | model: {args.model}\n"
          f"output: {out_path} (existing ok: {len(done)}, to do: {len(jobs)})\n"
          f"workers: {args.workers}")
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
            futures = [pool.submit(score_one, pid, idx, unit, post, args.mode, args.model)
                       for pid, idx, unit, post in jobs]
            for j, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                if r["error"]:
                    n_err += 1
                    print(f"  [{j}/{total}] {r['post_id']}/c{r['claim_index']} ERROR: {r['error']}")
                elif j % 50 == 0 or j == total:
                    print(f"  [{j}/{total}] done")
        new_df = pl.DataFrame(results, schema=RELEVANCE_SCHEMA)
        job_keys = pl.DataFrame(
            {"post_id": [j[0] for j in jobs], "claim_index": [j[1] for j in jobs]},
            schema={"post_id": pl.String, "claim_index": pl.Int64},
        )
        existing_keep = existing.join(job_keys, on=["post_id", "claim_index"], how="anti")
        combined = pl.concat([existing_keep, new_df]) if len(existing_keep) else new_df
        print(f"\nscored {len(new_df)} units in {time.time() - t0:.1f}s. errors: {n_err}/{total}")

    combined.write_parquet(out_path)
    okc = combined.filter(pl.col("error").is_null())
    n_in = int(okc.filter(pl.col("in_scope")).height)
    print(f"wrote {len(combined)} rows -> {out_path}\n"
          f"  in_scope=true: {n_in}/{len(okc)} ({100*n_in/max(len(okc),1):.1f}%)")


if __name__ == "__main__":
    main()
