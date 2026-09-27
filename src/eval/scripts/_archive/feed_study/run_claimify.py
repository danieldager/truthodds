"""Run the Claimify-style pipeline over the feed, one row per post.

Pipeline (see prompts_claimify.py): SELECTION -> DISAMBIGUATION -> DECOMPOSITION.
The pipeline short-circuits (abstains) at the first stage that says "no":
  - selection verifiable=false      -> outcome="no_verifiable_content"
  - disambiguation can=false        -> outcome="abstained_ambiguous"
  - decomposition success           -> outcome="claims"

Model-parameterized so the SAME code produces the primary run and the
different-family MIRROR run for agreement analysis:
  uv run python -m eval.scripts.feed_study.run_claimify -m openai/gpt-oss-120b
  uv run python -m eval.scripts.feed_study.run_claimify -m qwen/qwen3-32b

Output: data/claimify_<modeltag>.parquet — one row per post:
  post_id, model, outcome, stage_reached,
  verifiable, cleaned, can_disambiguate, confidence, decontextualized,
  claims (list[str]), n_claims,
  selection_raw, disambig_raw, decomp_raw, latency_s, error

Resumable: rows keyed by post_id; successful rows are skipped on re-run, errored
rows are re-attempted. 429 / transient 5xx retried with exponential backoff.
Mirrors the archived claim_cascade runners.
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

from eval.scripts.feed_study.prompts_claimify import (  # noqa: E402
    build_decomposition_messages,
    build_disambiguation_messages,
    build_selection_messages,
    parse_decomposition,
    parse_disambiguation,
    parse_selection,
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


CLAIMIFY_SCHEMA = {
    "post_id": pl.String,
    "model": pl.String,
    "outcome": pl.String,
    "stage_reached": pl.String,
    "verifiable": pl.Boolean,
    "cleaned": pl.String,
    "can_disambiguate": pl.Boolean,
    "confidence": pl.String,
    "decontextualized": pl.String,
    "claims": pl.List(pl.String),
    "n_claims": pl.Int64,
    "selection_raw": pl.String,
    "disambig_raw": pl.String,
    "decomp_raw": pl.String,
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


def _blank_row(post_id: str, model: str) -> dict:
    """Row skeleton with all-null/empty stage fields; callers fill what they reach."""
    return {
        "post_id": post_id,
        "model": model,
        "outcome": "",
        "stage_reached": "",
        "verifiable": None,
        "cleaned": None,
        "can_disambiguate": None,
        "confidence": None,
        "decontextualized": None,
        "claims": [],
        "n_claims": 0,
        "selection_raw": None,
        "disambig_raw": None,
        "decomp_raw": None,
        "latency_s": 0.0,
        "error": None,
    }


def run_pipeline(post_id: str, post: str, model: str) -> dict:
    """Run the 3 stages, short-circuiting on the first abstain. One row out."""
    t0 = time.time()
    row = _blank_row(post_id, model)
    try:
        # Stage 1 — Selection
        sel_raw = call_llm(build_selection_messages(post), model)
        row["selection_raw"] = sel_raw
        sel = parse_selection(sel_raw)
        row["verifiable"] = sel["verifiable"]
        row["cleaned"] = sel["cleaned"]
        row["stage_reached"] = "selection"
        if not sel["verifiable"]:
            row["outcome"] = "no_verifiable_content"
            return row

        # Stage 2 — Disambiguation (hybrid / confidence-flagged)
        dis_raw = call_llm(build_disambiguation_messages(post, sel["cleaned"]), model)
        row["disambig_raw"] = dis_raw
        dis = parse_disambiguation(dis_raw)
        row["can_disambiguate"] = dis["can_disambiguate"]
        row["confidence"] = dis["confidence"]
        row["decontextualized"] = dis["decontextualized"]
        row["stage_reached"] = "disambiguation"
        if not dis["can_disambiguate"]:
            row["outcome"] = "abstained_ambiguous"
            return row

        # Stage 3 — Decomposition
        dec_raw = call_llm(build_decomposition_messages(dis["decontextualized"]), model)
        row["decomp_raw"] = dec_raw
        dec = parse_decomposition(dec_raw)
        row["claims"] = dec["claims"]
        row["n_claims"] = len(dec["claims"])
        row["stage_reached"] = "decomposition"
        row["outcome"] = "claims"
    except Exception as e:  # noqa: BLE001
        row["error"] = f"{type(e).__name__}: {e}"
    finally:
        row["latency_s"] = round(time.time() - t0, 3)
    return row


def _model_tag(model: str) -> str:
    """openai/gpt-oss-120b -> gpt-oss-120b ; qwen/qwen3-32b -> qwen3-32b."""
    return model.split("/")[-1]


def _load_existing(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=CLAIMIFY_SCHEMA)
    return pl.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-p", "--posts", type=Path,
                    default=DEFAULT_DATA_DIR / "posts_x862.parquet",
                    help="input posts parquet (post_id, text)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output parquet (default: <outdir>/claimify_<modeltag>.parquet)")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_DATA_DIR,
                    help="dir for the default-named output (ignored if -o given)")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-w", "--workers", type=int, default=12)
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="cap input posts for smoke testing")
    args = ap.parse_args()

    if not args.posts.exists():
        raise SystemExit(f"posts parquet not found: {args.posts}")
    out_path: Path = args.output or (args.outdir / f"claimify_{_model_tag(args.model)}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    posts = (
        pl.read_parquet(args.posts).select(["post_id", "text"])
        .unique(subset=["post_id"], keep="first", maintain_order=True)
    )
    if args.limit is not None:
        posts = posts.head(args.limit)

    existing = _load_existing(out_path)
    ok = existing.filter(pl.col("error").is_null())
    done = set(ok["post_id"].to_list())

    jobs = [
        (row["post_id"], row["text"])
        for row in posts.iter_rows(named=True)
        if row["post_id"] not in done and (row["text"] or "").strip()
    ]

    print(
        f"posts: {args.posts} ({len(posts)} rows, limit={args.limit})\n"
        f"output: {out_path} (existing ok: {len(done)}, to do: {len(jobs)})\n"
        f"model: {args.model}, workers: {args.workers}"
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
            futures = [pool.submit(run_pipeline, pid, text, args.model) for pid, text in jobs]
            for j, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                results.append(r)
                if r["error"]:
                    n_err += 1
                    print(f"  [{j}/{total}] {r['post_id']} ERROR: {r['error']}")
                elif j % 25 == 0 or j == total:
                    print(f"  [{j}/{total}] done")
        new_df = pl.DataFrame(results, schema=CLAIMIFY_SCHEMA)
        # Replace any prior errored rows for these post_ids with fresh results.
        job_keys = pl.DataFrame({"post_id": [j[0] for j in jobs]}, schema={"post_id": pl.String})
        existing_keep = existing.join(job_keys, on="post_id", how="anti")
        combined = pl.concat([existing_keep, new_df]) if len(existing_keep) else new_df
        print(f"\nran {len(new_df)} posts in {time.time() - t0:.1f}s. errors: {n_err}/{total}")

    combined.write_parquet(out_path)

    # Outcome funnel — the headline for one model's run.
    n = len(combined)
    okc = combined.filter(pl.col("error").is_null())
    by_outcome = dict(
        okc.group_by("outcome").len().sort("len", descending=True)
        .iter_rows()
    )
    n_claims_total = int(okc.select(pl.col("n_claims").sum()).item() or 0)
    print(
        f"wrote {n} rows -> {out_path}\n"
        f"  outcome funnel (ok rows): {by_outcome}\n"
        f"  total claims extracted: {n_claims_total}"
    )


if __name__ == "__main__":
    main()
