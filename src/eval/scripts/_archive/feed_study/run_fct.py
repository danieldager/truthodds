"""Tier-2: Google Fact Check Tools lookup on the novel (deduped) claims.

Queries `claims:search` for each novel-claim representative (from dedup_claims.py)
to measure how many already have a published ClaimReview — i.e. resolve at Tier-2
and skip the expensive Tier-3 verification. Self-contained (key from env, FCT
client logic mirrors the archived stage7_fct.py); resumable on cluster_id; backs
off on 429/5xx with a quota circuit-breaker.

  uv run python -m eval.scripts.feed_study.run_fct \
      --claims eval/scripts/feed_study/data/sample1000/novel_claims_gpt-oss-120b.parquet
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import requests

ROOT = Path(__file__).resolve().parents[3]  # -> src/
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

FCT_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
_LANG_REMAP = {"in": "id", "iw": "he"}
_NON_LANG = {"und", "zxx", "qme", "qst", "qht", "qam", "qct", "art", ""}
_KEY_RE = re.compile(r"key=[^&\s]+")

_QUOTA_ABORT_AFTER = 8
_lock = threading.Lock()
_consec_429 = 0
_tripped = threading.Event()
_session: requests.Session | None = None
_SLEEP = 0.0  # per-call throttle (s) to respect the FCT rate limit; set via --sleep

FCT_SCHEMA = {
    "cluster_id": pl.Int64, "representative": pl.String, "lang": pl.String,
    "n_members": pl.Int64, "fct_match": pl.Boolean, "fct_publisher": pl.String,
    "fct_rating": pl.String, "fct_url": pl.String, "n_results": pl.Int64,
    "latency_s": pl.Float64, "error": pl.String,
}


def get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"X-goog-api-key": os.environ["GOOGLE_FCTAPI_KEY"]})
        _session = s
    return _session


def to_language_code(lang: str | None) -> str | None:
    if not lang:
        return None
    code = lang.strip().lower()
    if code in _NON_LANG:
        return None
    return _LANG_REMAP.get(code, code)


def query_fct(claim_text: str, language_code: str | None, max_retries: int = 3) -> dict:
    params: dict[str, object] = {"query": claim_text, "pageSize": 5}
    if language_code:
        params["languageCode"] = language_code
    last = None
    for attempt in range(max_retries):
        resp = get_session().get(FCT_ENDPOINT, params=params, timeout=20)
        if resp.status_code == 200:
            with _lock:
                globals()["_consec_429"] = 0
            return resp.json()
        if resp.status_code == 429:
            with _lock:
                globals()["_consec_429"] += 1
                if _consec_429 >= _QUOTA_ABORT_AFTER:
                    _tripped.set()
        if resp.status_code == 429 or resp.status_code >= 500:
            last = RuntimeError(f"HTTP {resp.status_code}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"HTTP {resp.status_code} (non-retryable)")
    raise last or RuntimeError("FCT request failed")


def parse_fct(data: dict) -> dict:
    claims = data.get("claims") or []
    reviews = (claims[0].get("claimReview") or []) if claims else []
    if not reviews:
        return {"fct_match": False, "fct_publisher": "", "fct_rating": "", "fct_url": "",
                "n_results": len(claims)}
    r = reviews[0]
    pub = r.get("publisher") or {}
    return {"fct_match": True, "fct_publisher": pub.get("site") or pub.get("name") or "",
            "fct_rating": r.get("textualRating") or "", "fct_url": r.get("url") or "",
            "n_results": len(claims)}


def run_one(cid: int, rep: str, lang: str, n_members: int) -> dict:
    t0 = time.time()
    error = None
    p = {"fct_match": False, "fct_publisher": "", "fct_rating": "", "fct_url": "", "n_results": 0}
    if _tripped.is_set():
        error = "aborted: FCT quota circuit breaker tripped"
    else:
        try:
            if _SLEEP:
                time.sleep(_SLEEP)
            p = parse_fct(query_fct(rep, to_language_code(lang)))
        except Exception as e:  # noqa: BLE001
            error = _KEY_RE.sub("key=REDACTED", f"{type(e).__name__}: {e}")
    return {"cluster_id": cid, "representative": rep, "lang": lang, "n_members": n_members,
            **p, "latency_s": round(time.time() - t0, 3), "error": error}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claims", type=Path, required=True, help="novel_claims_<ext>.parquet")
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("-w", "--workers", type=int, default=6)
    ap.add_argument("--sleep", type=float, default=0.0, help="per-call delay (s) to respect FCT rate limit")
    ap.add_argument("-n", "--limit", type=int, default=None)
    args = ap.parse_args()
    globals()["_SLEEP"] = args.sleep

    nc = pl.read_parquet(args.claims)
    out_path = args.output or args.claims.with_name(args.claims.stem.replace("novel_claims", "fct") + ".parquet")
    jobs = [(r["cluster_id"], r["representative"], r["lang"], r["n_members"])
            for r in nc.iter_rows(named=True)]
    if args.limit is not None:
        jobs = jobs[:args.limit]

    existing = pl.read_parquet(out_path) if out_path.exists() else pl.DataFrame(schema=FCT_SCHEMA)
    done = set(existing.filter(pl.col("error").is_null())["cluster_id"].to_list())
    jobs = [j for j in jobs if j[0] not in done]
    print(f"novel claims: {nc.height} | to query: {len(jobs)} (done: {len(done)}) -> {out_path}")
    if not jobs and len(existing) == 0:
        print("nothing to do.")
        return

    combined = existing
    if jobs:
        t0 = time.time(); results = []; n_err = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_one, *j) for j in jobs]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result(); results.append(r)
                if r["error"]:
                    n_err += 1
                elif i % 50 == 0 or i == len(jobs):
                    print(f"  [{i}/{len(jobs)}] done")
        new = pl.DataFrame(results, schema=FCT_SCHEMA)
        keep = existing.join(pl.DataFrame({"cluster_id": [j[0] for j in jobs]},
                                          schema={"cluster_id": pl.Int64}), on="cluster_id", how="anti")
        combined = pl.concat([keep, new]) if len(keep) else new
        print(f"\nqueried {len(new)} in {time.time()-t0:.1f}s. errors: {n_err}/{len(jobs)}"
              + ("  (quota breaker TRIPPED)" if _tripped.is_set() else ""))

    combined.write_parquet(out_path)
    okc = combined.filter(pl.col("error").is_null())
    n_match = int(okc.filter(pl.col("fct_match")).height)
    print(f"wrote {len(combined)} rows -> {out_path}\n"
          f"  FCT match (existing fact-check): {n_match}/{len(okc)} ({100*n_match/max(len(okc),1):.1f}%)")


if __name__ == "__main__":
    main()
