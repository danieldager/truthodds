"""LLM-as-judge for extraction outputs from extract.py — v4.

Per post, sends K judge calls to `qwen/qwen3-32b` on Groq, where K is the
number of claims the extractor produced. Each call scores ONE claim on:

  * fidelity (1-5)
  * decontextualized (1-5)
  * verifiability (1-5)
  * misinfo_candidate (bool)

The bool `misinfo_candidate` replaces v3's separate detection judge: the
per-claim judge applies the same 4-criterion misinformation filter the
extractor uses, so disagreement (extractor=true, judge=false) is the
cleanest precision signal.

Reference-free — no gold shown. Reasoning model: max_tokens=4000 to leave
room for `<think>...</think>` blocks; `prompts.parse_per_claim` strips them.

Output:
  data/judgments_per_claim_n{N}.parquet — one row per (post_id, claim_index):
    post_id, claim_index, claim_text,
    fidelity, decontextualized, verifiability, misinfo_candidate,
    raw_response, latency_s, error

Resumable: per-claim rows keyed by (post_id, claim_index). Per-row
exceptions captured in `error`, scores default to 0 / bool to false,
raw_response preserved for audit.
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
    build_per_claim_messages,
    parse_per_claim,
)

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


PER_CLAIM_SCHEMA = {
    "post_id": pl.String,
    "claim_index": pl.Int64,
    "claim_text": pl.String,
    "fidelity": pl.Int64,
    "decontextualized": pl.Int64,
    "verifiability": pl.Int64,
    "misinfo_candidate": pl.Boolean,
    "raw_response": pl.String,
    "latency_s": pl.Float64,
    "error": pl.String,
}


def _call_llm(messages: list[dict[str, str]], model: str) -> str:
    resp = get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=4000,  # reasoning model headroom for <think>...</think>
    )
    return resp.choices[0].message.content or ""


def judge_per_claim(post_id: str, claim_index: int, claim_text: str,
                    post: str, model: str) -> dict:
    t0 = time.time()
    raw = ""
    scores = {"fidelity": 0, "decontextualized": 0, "verifiability": 0,
              "misinfo_candidate": False}
    error: str | None = None
    try:
        raw = _call_llm(build_per_claim_messages(post, claim_text), model)
        scores = parse_per_claim(raw)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    return {
        "post_id": post_id,
        "claim_index": claim_index,
        "claim_text": claim_text,
        "fidelity": scores["fidelity"],
        "decontextualized": scores["decontextualized"],
        "verifiability": scores["verifiability"],
        "misinfo_candidate": scores["misinfo_candidate"],
        "raw_response": raw,
        "latency_s": round(time.time() - t0, 3),
        "error": error,
    }


def _infer_default_output(extractions_path: Path) -> Path:
    """data/extractions_n50.parquet -> data/judgments_per_claim_n50.parquet."""
    m = re.search(r"extractions_n(\d+)", extractions_path.stem)
    suffix = f"n{m.group(1)}" if m else extractions_path.stem
    return extractions_path.parent / f"judgments_per_claim_{suffix}.parquet"


def _load_existing(path: Path, schema: dict) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=schema)
    return pl.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-p", "--posts", type=Path,
        default=DEFAULT_DATA_DIR / "posts_n2000.parquet",
        help="input posts parquet (post_id, text)",
    )
    ap.add_argument(
        "-e", "--extractions", type=Path,
        default=DEFAULT_DATA_DIR / "extractions_n2000.parquet",
        help="input extractions parquet (post_id, has_claim, claims, ...)",
    )
    ap.add_argument(
        "--per-claim-output", type=Path, default=None,
        help="output per-claim judgments parquet "
             "(default: data/judgments_per_claim_n{N}.parquet)",
    )
    ap.add_argument("-w", "--workers", type=int, default=12)
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="cap input posts for smoke testing")
    ap.add_argument("-m", "--model", default=DEFAULT_JUDGE_MODEL)
    args = ap.parse_args()

    if not args.posts.exists():
        raise SystemExit(f"posts parquet not found: {args.posts}")
    if not args.extractions.exists():
        raise SystemExit(f"extractions parquet not found: {args.extractions}")

    per_claim_out = args.per_claim_output or _infer_default_output(args.extractions)
    per_claim_out.parent.mkdir(parents=True, exist_ok=True)

    posts = pl.read_parquet(args.posts).select(["post_id", "text"])
    extractions = pl.read_parquet(args.extractions).select(
        ["post_id", "has_claim", "claims"]
    )
    if args.limit is not None:
        posts = posts.head(args.limit)

    # Join posts ⨝ extractions on post_id; restrict to posts the user selected.
    work = posts.join(extractions, on="post_id", how="inner")

    existing_per_claim = _load_existing(per_claim_out, PER_CLAIM_SCHEMA)
    done_per_claim = set(
        zip(
            existing_per_claim["post_id"].to_list(),
            existing_per_claim["claim_index"].to_list(),
        )
    )

    # Build job list — only per-claim, no detection.
    per_claim_jobs: list[tuple[str, int, str, str]] = []  # (pid, idx, claim, post)
    for row in work.iter_rows(named=True):
        pid = row["post_id"]
        post_text = row["text"]
        claims = row["claims"] or []
        for i, claim_text in enumerate(claims):
            if (pid, i) not in done_per_claim:
                per_claim_jobs.append((pid, i, claim_text, post_text))

    print(
        f"posts: {args.posts} ({len(posts)} rows, limit={args.limit})\n"
        f"extractions: {args.extractions} ({len(extractions)} rows)\n"
        f"per-claim output: {per_claim_out} "
        f"(existing: {len(done_per_claim)}, to do: {len(per_claim_jobs)})\n"
        f"model: {args.model}, workers: {args.workers}"
    )

    if not per_claim_jobs:
        print("nothing to do.")
        return

    total = len(per_claim_jobs)
    t0 = time.time()
    per_claim_results: list[dict] = []
    n_err = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(judge_per_claim, pid, idx, claim_text, post_text, args.model): (pid, idx)
            for pid, idx, claim_text, post_text in per_claim_jobs
        }
        completed = 0
        for fut in as_completed(futures):
            r = fut.result()
            completed += 1
            per_claim_results.append(r)
            if r["error"]:
                n_err += 1
                print(f"  [{completed}/{total}] {r['post_id']}/c{r['claim_index']} ERROR: {r['error']}")
            elif completed % 20 == 0 or completed == total:
                print(f"  [{completed}/{total}] done")

    # Write per-claim parquet.
    if per_claim_results:
        new_df = pl.DataFrame(per_claim_results, schema=PER_CLAIM_SCHEMA)
        combined = (
            pl.concat([existing_per_claim, new_df])
            if len(existing_per_claim) else new_df
        )
        combined.write_parquet(per_claim_out)
        print(f"wrote {len(combined)} per-claim rows ({len(new_df)} new) "
              f"-> {per_claim_out}")

    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.1f}s. errors: {n_err}/{total}")


if __name__ == "__main__":
    main()
