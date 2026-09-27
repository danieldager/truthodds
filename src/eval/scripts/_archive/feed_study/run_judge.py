"""3rd-model (llama-3.3-70b-versatile) judge runner: claim-set comparison + quality.

  --mode compare  one call per post where BOTH models extracted claims; compares
                  the two claim sets. needs --claims-a (gpt) and --claims-b (qwen).
  --mode quality  one call per post with claims, auditing one model's extraction.
                  needs --claims and --extractor.

Mirrors run_fable.py scaffolding. Output <outdir>/judge_compare.parquet or
judge_quality_<extractor>.parquet. Resumable on post_id.

  uv run python -m eval.scripts.feed_study.run_judge --mode compare \
      --claims-a data/sample1000/claimify_gpt-oss-120b.parquet \
      --claims-b data/sample1000/claimify_qwen3-32b.parquet -p data/posts_sample1000.parquet --outdir data/sample1000
  uv run python -m eval.scripts.feed_study.run_judge --mode quality \
      --claims data/sample1000/claimify_gpt-oss-120b.parquet --extractor gpt-oss-120b \
      -p data/posts_sample1000.parquet --outdir data/sample1000
"""

from __future__ import annotations

import argparse
import json
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

from eval.scripts.feed_study.prompts_judge import (  # noqa: E402
    build_compare_messages,
    build_quality_messages,
    parse_compare,
    parse_quality,
)

load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
TEMPERATURE = 0.1
MAX_TOKENS = 4000
MAX_RETRIES = 5

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://api.groq.com/openai/v1",
                         api_key=os.environ["GROQ_API_KEY"])
    return _client


COMPARE_SCHEMA = {
    "post_id": pl.String, "judge_model": pl.String,
    "n_a": pl.Int64, "n_b": pl.Int64,
    "aligned_json": pl.String, "only_a_json": pl.String, "only_b_json": pl.String,
    "agreement": pl.String, "more_complete": pl.String, "notes": pl.String,
    "raw_response": pl.String, "latency_s": pl.Float64, "error": pl.String,
}

QUALITY_SCHEMA = {
    "post_id": pl.String, "extractor_model": pl.String, "judge_model": pl.String,
    "n_claims": pl.Int64, "claim_scores_json": pl.String,
    "coverage": pl.Int64, "flag": pl.String, "reason": pl.String,
    "raw_response": pl.String, "latency_s": pl.Float64, "error": pl.String,
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


def compare_one(post_id, post, claims_a, claims_b, model) -> dict:
    t0 = time.time()
    raw, error = "", None
    out = {"aligned_pairs": [], "only_a": [], "only_b": [],
           "agreement": None, "more_complete": None, "notes": None}
    try:
        raw = call_llm(build_compare_messages(post, claims_a, claims_b), model)
        out = parse_compare(raw)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    return {
        "post_id": post_id, "judge_model": model,
        "n_a": len(claims_a), "n_b": len(claims_b),
        "aligned_json": json.dumps(out["aligned_pairs"]),
        "only_a_json": json.dumps(out["only_a"]), "only_b_json": json.dumps(out["only_b"]),
        "agreement": out["agreement"], "more_complete": out["more_complete"],
        "notes": out["notes"],
        "raw_response": raw, "latency_s": round(time.time() - t0, 3), "error": error,
    }


def quality_one(post_id, post, claims, extractor, model) -> dict:
    t0 = time.time()
    raw, error = "", None
    out = {"claim_scores": [], "coverage": None, "flag": None, "reason": None}
    try:
        raw = call_llm(build_quality_messages(post, claims), model)
        out = parse_quality(raw)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"
    return {
        "post_id": post_id, "extractor_model": extractor, "judge_model": model,
        "n_claims": len(claims), "claim_scores_json": json.dumps(out["claim_scores"]),
        "coverage": out["coverage"], "flag": out["flag"], "reason": out["reason"],
        "raw_response": raw, "latency_s": round(time.time() - t0, 3), "error": error,
    }


def _post_text(posts: pl.DataFrame) -> dict:
    return {r["post_id"]: (r["text"] or "") for r in posts.iter_rows(named=True)}


def _claims_map(path: Path) -> dict:
    cl = pl.read_parquet(path).select(["post_id", "claims"])
    return {r["post_id"]: [c for c in (r["claims"] or []) if (c or "").strip()]
            for r in cl.iter_rows(named=True)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["compare", "quality"], required=True)
    ap.add_argument("-p", "--posts", type=Path, default=DEFAULT_DATA_DIR / "posts_sample1000.parquet")
    ap.add_argument("--claims-a", type=Path, default=None, help="compare: gpt claimify")
    ap.add_argument("--claims-b", type=Path, default=None, help="compare: qwen claimify")
    ap.add_argument("--claims", type=Path, default=None, help="quality: one claimify parquet")
    ap.add_argument("--extractor", type=str, default=None, help="quality: extractor model tag")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("-w", "--workers", type=int, default=12)
    ap.add_argument("-n", "--limit", type=int, default=None)
    args = ap.parse_args()

    if not args.posts.exists():
        raise SystemExit(f"posts parquet not found: {args.posts}")
    posts = pl.read_parquet(args.posts).select(["post_id", "text"]).unique(
        subset=["post_id"], keep="first", maintain_order=True)
    ptext = _post_text(posts)
    schema = COMPARE_SCHEMA if args.mode == "compare" else QUALITY_SCHEMA

    if args.mode == "compare":
        if not (args.claims_a and args.claims_b):
            raise SystemExit("--mode compare requires --claims-a and --claims-b")
        a, b = _claims_map(args.claims_a), _claims_map(args.claims_b)
        both = [pid for pid in a if a.get(pid) and b.get(pid)]
        all_jobs = [(pid, ptext.get(pid, ""), a[pid], b[pid]) for pid in both]
        out_path = args.output or (args.outdir / "judge_compare.parquet")
    else:
        if not (args.claims and args.extractor):
            raise SystemExit("--mode quality requires --claims and --extractor")
        c = _claims_map(args.claims)
        all_jobs = [(pid, ptext.get(pid, ""), claims) for pid, claims in c.items() if claims]
        out_path = args.output or (args.outdir / f"judge_quality_{args.extractor}.parquet")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.limit is not None:
        all_jobs = all_jobs[:args.limit]

    existing = pl.read_parquet(out_path) if out_path.exists() else pl.DataFrame(schema=schema)
    ok = existing.filter(pl.col("error").is_null())
    done = set(ok["post_id"].to_list())
    jobs = [j for j in all_jobs if j[0] not in done]

    print(f"mode: {args.mode} | judge: {args.model} | jobs: {len(all_jobs)} "
          f"(existing ok: {len(done)}, to do: {len(jobs)})\noutput: {out_path}")
    if not jobs and len(existing) == 0:
        print("nothing to do.")
        return

    combined = existing
    if jobs:
        t0 = time.time()
        results, n_err, total = [], 0, len(jobs)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            if args.mode == "compare":
                futs = [pool.submit(compare_one, pid, pt, ca, cb, args.model)
                        for pid, pt, ca, cb in jobs]
            else:
                futs = [pool.submit(quality_one, pid, pt, cl, args.extractor, args.model)
                        for pid, pt, cl in jobs]
            for j, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                if r["error"]:
                    n_err += 1
                    print(f"  [{j}/{total}] {r['post_id']} ERROR: {r['error']}")
                elif j % 50 == 0 or j == total:
                    print(f"  [{j}/{total}] done")
        new_df = pl.DataFrame(results, schema=schema)
        keep = existing.join(pl.DataFrame({"post_id": [j[0] for j in jobs]},
                                          schema={"post_id": pl.String}),
                             on="post_id", how="anti")
        combined = pl.concat([keep, new_df]) if len(keep) else new_df
        print(f"\njudged {len(new_df)} posts in {time.time() - t0:.1f}s. errors: {n_err}/{total}")

    combined.write_parquet(out_path)
    okc = combined.filter(pl.col("error").is_null())
    if args.mode == "compare":
        dist = dict(okc.group_by("agreement").len().iter_rows())
        print(f"wrote {len(combined)} rows -> {out_path}\n  agreement dist: {dist}")
    else:
        dist = dict(okc.group_by("flag").len().iter_rows())
        print(f"wrote {len(combined)} rows -> {out_path}\n  flag dist: {dist}")


if __name__ == "__main__":
    main()
