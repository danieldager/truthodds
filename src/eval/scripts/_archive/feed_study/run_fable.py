"""Run the FABLE harm rubric in one of two modes.

  --mode post   score each RAW POST (front-filter experiment): how would FABLE
                behave as an up-front harm gate, before any extraction?
                input: posts parquet. one row per post.

  --mode claim  score each NORMALIZED CLAIM from a Claimify run (FABLE's native
                per-claim role). input: a claimify_<model>.parquet. one row per
                (post_id, claim_index).

Both modes use the IDENTICAL prompt (feed_study.fable), so post-mode and
claim-mode scores are directly comparable. Higher total (5-25) = more
potentially harmful / more worth checking. `fable_checkworthy` is DERIVED from
--fable-threshold at write time, so recalibration is a re-run with no LLM calls.

  uv run python -m eval.scripts.feed_study.run_fable --mode post  -m qwen/qwen3-32b
  uv run python -m eval.scripts.feed_study.run_fable --mode claim -m qwen/qwen3-32b \
      --claims eval/scripts/feed_study/data/claimify_gpt-oss-120b.parquet

Output: data/fable_<mode>_<modeltag>.parquet. Resumable; 429/5xx retried.
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

from eval.scripts.feed_study.fable import (  # noqa: E402
    FABLE_DIMENSIONS,
    HARM_DIMS,
    TRIGGER_HARM3_MIN,
    TRIGGER_TOTAL_MIN,
    build_fable_claim_only,
    build_fable_messages,
    build_fable_post_messages,
    fable_total,
    parse_fable,
)

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "qwen/qwen3-32b"
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


# One schema for both modes. In post mode claim_index = -1 and claim_text = the
# post text; in claim mode they identify the claim within its post.
FABLE_SCHEMA = {
    "post_id": pl.String,
    "claim_index": pl.Int64,
    "unit_text": pl.String,
    "model": pl.String,
    "mode": pl.String,
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


def score_one(post_id: str, claim_index: int, unit_text: str, post: str,
              mode: str, model: str) -> dict:
    """Score one unit. post mode: post itself; claim mode: a claim with post context."""
    t0 = time.time()
    raw = ""
    dims = {k: 0 for k in FABLE_DIMENSIONS}
    total = 0
    error: str | None = None
    try:
        if mode == "post":
            msgs = build_fable_post_messages(post)
        elif mode == "claim":
            msgs = build_fable_claim_only(unit_text)
        else:  # both
            msgs = build_fable_messages(post, unit_text)
        raw = call_llm(msgs, model)
        dims = parse_fable(raw)
        total = fable_total(dims)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    return {
        "post_id": post_id,
        "claim_index": claim_index,
        "unit_text": unit_text,
        "model": model,
        "mode": mode,
        **dims,
        "fable_total": total,
        "fable_checkworthy": False,  # re-derived from threshold at write time
        "raw_response": raw,
        "latency_s": round(time.time() - t0, 3),
        "error": error,
    }


def _build_jobs_post(posts: pl.DataFrame) -> list[tuple]:
    """(post_id, claim_index=-1, unit_text=post, post)."""
    jobs = []
    for r in posts.iter_rows(named=True):
        text = (r["text"] or "").strip()
        if text:
            jobs.append((r["post_id"], -1, text, text))
    return jobs


def _build_jobs_claim(posts: pl.DataFrame, claims_path: Path) -> list[tuple]:
    """(post_id, claim_index, claim_text, post) for every claim in a claimify run."""
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
        return pl.DataFrame(schema=FABLE_SCHEMA)
    return pl.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["post", "claim", "both"], required=True,
                    help="post=raw post; claim=claim only (no post); both=post+claim")
    ap.add_argument("-p", "--posts", type=Path,
                    default=DEFAULT_DATA_DIR / "posts_x862.parquet")
    ap.add_argument("--claims", type=Path, default=None,
                    help="claim/both modes: a claimify_<model>.parquet (post_id, claims)")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_DATA_DIR,
                    help="dir for the default-named output (ignored if -o given)")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("--total-min", type=int, default=TRIGGER_TOTAL_MIN,
                    help="flag if fable_total >= this (default 12)")
    ap.add_argument("--harm3-min", type=int, default=TRIGGER_HARM3_MIN,
                    help="flag if frag+action+exploit >= this (default 8)")
    ap.add_argument("-w", "--workers", type=int, default=12)
    ap.add_argument("-n", "--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.posts.exists():
        raise SystemExit(f"posts parquet not found: {args.posts}")
    if args.mode in ("claim", "both") and args.claims is None:
        raise SystemExit(f"--mode {args.mode} requires --claims <claimify parquet>")

    tag = args.model.split("/")[-1]
    out_path: Path = args.output or (args.outdir / f"fable_{args.mode}_{tag}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Full posts table is always needed: it is the post universe for post mode
    # AND the post-context lookup for claim mode. --limit caps JOBS, not posts,
    # so claim-mode jobs never lose their post context.
    posts = (
        pl.read_parquet(args.posts).select(["post_id", "text"])
        .unique(subset=["post_id"], keep="first", maintain_order=True)
    )

    all_jobs = (_build_jobs_post(posts) if args.mode == "post"
                else _build_jobs_claim(posts, args.claims))  # claim + both both iterate claims
    if args.limit is not None:
        all_jobs = all_jobs[:args.limit]

    existing = _load_existing(out_path)
    ok = existing.filter(pl.col("error").is_null())
    done = set(zip(ok["post_id"].to_list(), ok["claim_index"].to_list()))
    jobs = [j for j in all_jobs if (j[0], j[1]) not in done]

    trigger_desc = f"total>={args.total_min} OR harm3>={args.harm3_min}"
    print(
        f"mode: {args.mode} | posts: {len(posts)} | model: {args.model}\n"
        f"output: {out_path} (existing ok: {len(done)}, to do: {len(jobs)})\n"
        f"workers: {args.workers}, trigger: {trigger_desc}"
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
                pool.submit(score_one, pid, idx, unit, post, args.mode, args.model)
                for pid, idx, unit, post in jobs
            ]
            for j, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                if r["error"]:
                    n_err += 1
                    print(f"  [{j}/{total}] {r['post_id']}/c{r['claim_index']} ERROR: {r['error']}")
                elif j % 25 == 0 or j == total:
                    print(f"  [{j}/{total}] done")
        new_df = pl.DataFrame(results, schema=FABLE_SCHEMA)
        job_keys = pl.DataFrame(
            {"post_id": [j[0] for j in jobs], "claim_index": [j[1] for j in jobs]},
            schema={"post_id": pl.String, "claim_index": pl.Int64},
        )
        existing_keep = existing.join(job_keys, on=["post_id", "claim_index"], how="anti")
        combined = pl.concat([existing_keep, new_df]) if len(existing_keep) else new_df
        print(f"\nscored {len(new_df)} units in {time.time() - t0:.1f}s. errors: {n_err}/{total}")

    # Re-derive fable_checkworthy across ALL rows from the stored raw dims, so
    # the column always reflects the current trigger flags (re-deriving a new
    # rule = re-run with new --trigger-* flags, no LLM calls). Error rows have
    # all dims = 0, so they evaluate to False.
    total_hit = pl.col("fable_total") >= args.total_min
    harm3_hit = pl.sum_horizontal([pl.col(k) for k in HARM_DIMS]) >= args.harm3_min
    combined = combined.with_columns((total_hit | harm3_hit).alias("fable_checkworthy"))
    combined.write_parquet(out_path)

    okc = combined.filter(pl.col("error").is_null())
    n_cw = int(okc.filter(pl.col("fable_checkworthy")).height)
    mean_total = float(okc.select(pl.col("fable_total").mean()).item() or 0)
    print(
        f"wrote {len(combined)} rows -> {out_path}\n"
        f"  fable_checkworthy=true: {n_cw}/{len(okc)} ({trigger_desc})\n"
        f"  mean fable_total: {mean_total:.1f}"
    )


if __name__ == "__main__":
    main()
