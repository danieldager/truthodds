"""Stage 7 — Google Fact Check Tools cross-check on check-worthy claims.

Runs ONLY on the UNION of claims flagged check-worthy by EITHER rubric:

  * ``misinfo_candidate == True``  (Stage 4, the 4-prong per-claim judge)
  * ``fable_checkworthy == True``  (Stage 5, the FABLE harm rubric)

For each such claim we query FCT ``claims:search`` with the claim text and a
``languageCode`` derived from the post's ``lang``. This is the Tier-2
short-circuit measurement: how many check-worthy claims already have a
published fact-check vs. need novel verification.

Output (``data/fct_x862.parquet``), one row per check-worthy claim, keyed
``(post_id, claim_index)``:

    post_id, claim_index, claim_text, lang,
    fct_match (bool), fct_publisher, fct_rating, fct_url,
    n_results, latency_s, error

``fct_match`` is true iff the API returned at least one indexed ClaimReview
for the top result; ``fct_publisher``/``fct_rating``/``fct_url`` come from that
review. ``lang``/``n_results``/``latency_s``/``error`` are audit columns.

Resumable: existing rows whose call SUCCEEDED (``error`` is null) are skipped;
errored rows are retried and overwrite their prior row (dedup keep-last). On
write the table is filtered to the *current* union, so it always covers
EXACTLY the union (no stale rows). ThreadPool over the (small) union set;
backs off on HTTP 429/5xx, with a circuit breaker that aborts the run if the
quota looks exhausted (sustained 429s) rather than hammering the API.
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import FCTAPI_ENDPOINT, FCTAPI_KEY  # noqa: E402

DEFAULT_DATA_DIR = Path(__file__).parent / "data"

FCT_SCHEMA = {
    "post_id": pl.String,
    "claim_index": pl.Int64,
    "claim_text": pl.String,
    "lang": pl.String,
    "fct_match": pl.Boolean,
    "fct_publisher": pl.String,
    "fct_rating": pl.String,
    "fct_url": pl.String,
    "n_results": pl.Int64,
    "latency_s": pl.Float64,
    "error": pl.String,
}

# Twitter lang codes that are not BCP-47 (remap), and codes that carry no
# linguistic content (omit languageCode entirely → search all languages).
_LANG_REMAP = {"in": "id", "iw": "he"}
_NON_LANG = {"und", "zxx", "qme", "qst", "qht", "qam", "qct", "art", ""}

_KEY_RE = re.compile(r"key=[^&\s]+")

# --- Circuit breaker: stop hammering a quota-exhausted API. ----------------
_QUOTA_ABORT_AFTER = 8           # consecutive 429s across threads → abort
_breaker_lock = threading.Lock()
_consecutive_429 = 0
_quota_tripped = threading.Event()

_session: requests.Session | None = None


def get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        # Send the key as a header, never a query param — keeps it out of any
        # exception message / logged URL / persisted error column.
        s.headers.update({"X-goog-api-key": FCTAPI_KEY})
        _session = s
    return _session


def _redact(msg: str) -> str:
    return _KEY_RE.sub("key=REDACTED", msg)


def _reset_breaker() -> None:
    global _consecutive_429
    with _breaker_lock:
        _consecutive_429 = 0
    _quota_tripped.clear()


def _note_429() -> None:
    global _consecutive_429
    with _breaker_lock:
        _consecutive_429 += 1
        if _consecutive_429 >= _QUOTA_ABORT_AFTER:
            _quota_tripped.set()


def _note_ok() -> None:
    global _consecutive_429
    with _breaker_lock:
        _consecutive_429 = 0


def to_language_code(lang: str | None) -> str | None:
    """Map a Twitter `lang` to a BCP-47 languageCode, or None to skip the filter."""
    if not lang:
        return None
    code = lang.strip().lower()
    if code in _NON_LANG:
        return None
    return _LANG_REMAP.get(code, code)


def query_fct(claim_text: str, language_code: str | None,
              max_retries: int = 3) -> dict:
    """GET claims:search. Retries on 429 / 5xx with exponential backoff.

    The API key travels in the session header, so neither the URL nor any
    raised exception contains it.
    """
    params: dict[str, object] = {"query": claim_text, "pageSize": 5}
    if language_code:
        params["languageCode"] = language_code
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        resp = get_session().get(FCTAPI_ENDPOINT, params=params, timeout=20)
        if resp.status_code == 200:
            _note_ok()
            return resp.json()
        if resp.status_code == 429:
            _note_429()
        if resp.status_code == 429 or resp.status_code >= 500:
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
        # Non-retryable (e.g. 403): surface status only, never the URL/body.
        raise RuntimeError(f"HTTP {resp.status_code} (non-retryable)")
    raise last_exc or RuntimeError("FCT request failed")


def parse_fct(data: dict) -> dict:
    """Extract the headline match fields from a claims:search response.

    fct_match is true iff the top claim carries at least one ClaimReview; a
    returned claim with an empty claimReview list is NOT a match.
    """
    claims = data.get("claims") or []
    reviews = (claims[0].get("claimReview") or []) if claims else []
    if not reviews:
        return {
            "fct_match": False, "fct_publisher": "", "fct_rating": "",
            "fct_url": "", "n_results": len(claims),
        }
    review = reviews[0]
    publisher = review.get("publisher") or {}
    return {
        "fct_match": True,
        "fct_publisher": publisher.get("site") or publisher.get("name") or "",
        "fct_rating": review.get("textualRating") or "",
        "fct_url": review.get("url") or "",
        "n_results": len(claims),
    }


def run_one(post_id: str, claim_index: int, claim_text: str, lang: str) -> dict:
    t0 = time.time()
    error: str | None = None
    parsed = {
        "fct_match": False, "fct_publisher": "", "fct_rating": "",
        "fct_url": "", "n_results": 0,
    }
    if _quota_tripped.is_set():
        error = "aborted: FCT quota circuit breaker tripped"
    else:
        try:
            data = query_fct(claim_text, to_language_code(lang))
            parsed = parse_fct(data)
        except Exception as e:  # noqa: BLE001
            error = _redact(f"{type(e).__name__}: {e}")
    return {
        "post_id": post_id,
        "claim_index": claim_index,
        "claim_text": claim_text,
        "lang": lang or "",
        **parsed,
        "latency_s": round(time.time() - t0, 3),
        "error": error,
    }


def build_union(stage4: Path, stage5: Path, posts: Path) -> pl.DataFrame:
    """Union of claims flagged check-worthy by EITHER rubric, with post lang.

    Columns: post_id, claim_index, claim_text, lang.
    """
    s4 = pl.read_parquet(stage4).select(
        ["post_id", "claim_index", "claim_text", "misinfo_candidate"]
    )
    s5 = pl.read_parquet(stage5).select(
        ["post_id", "claim_index",
         pl.col("claim_text").alias("claim_text_fable"), "fable_checkworthy"]
    )
    # Full join so a claim present in only one stage is still considered; the
    # two stages run on the same claim set, so this is defensive, not lossy.
    claims = s4.join(s5, on=["post_id", "claim_index"], how="full", coalesce=True)
    claims = claims.with_columns(
        pl.coalesce(["claim_text", "claim_text_fable"]).alias("claim_text"),
        pl.col("misinfo_candidate").fill_null(False),
        pl.col("fable_checkworthy").fill_null(False),
    )
    union = claims.filter(
        pl.col("misinfo_candidate") | pl.col("fable_checkworthy")
    )

    lang = pl.read_parquet(posts).select(["post_id", "lang"])
    union = union.join(lang, on="post_id", how="left")
    return union.select(["post_id", "claim_index", "claim_text", "lang"]).sort(
        ["post_id", "claim_index"]
    )


def _load_existing(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=FCT_SCHEMA)
    return pl.read_parquet(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="dir holding the stage parquets (default: ./data)")
    ap.add_argument("--stage4", type=Path, default=None,
                    help="override: stage4 judgments parquet")
    ap.add_argument("--stage5", type=Path, default=None,
                    help="override: stage5 fable parquet")
    ap.add_argument("--posts", type=Path, default=None,
                    help="override: stage0 posts parquet (for lang)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="override: output fct parquet")
    ap.add_argument("-w", "--workers", type=int, default=6,
                    help="thread pool size (back off if FCT returns 429)")
    ap.add_argument("-n", "--limit", type=int, default=None,
                    help="cap union claims for smoke testing")
    args = ap.parse_args()

    d = args.data_dir
    stage4 = args.stage4 or d / "stage4_judgments.parquet"
    stage5 = args.stage5 or d / "fable_x862.parquet"
    posts = args.posts or d / "posts_x862.parquet"
    out = args.output or d / "fct_x862.parquet"

    for required in (stage4, stage5, posts):
        if not required.exists():
            raise SystemExit(f"required parquet missing: {required}")
    if not FCTAPI_KEY:
        raise SystemExit("GOOGLE_FCTAPI_KEY is empty; set it in src/.env")
    out.parent.mkdir(parents=True, exist_ok=True)
    _reset_breaker()

    union_full = build_union(stage4, stage5, posts)
    union = union_full.head(args.limit) if args.limit is not None else union_full

    existing = _load_existing(out)
    # Only SUCCESSFUL rows count as done; errored rows are retried.
    done_df = existing.filter(pl.col("error").is_null())
    done = set(zip(done_df["post_id"].to_list(), done_df["claim_index"].to_list()))
    jobs = [
        (r["post_id"], r["claim_index"], r["claim_text"], r["lang"])
        for r in union.iter_rows(named=True)
        if (r["post_id"], r["claim_index"]) not in done
    ]

    print(
        f"stage4: {stage4}\nstage5: {stage5}\nposts: {posts}\n"
        f"output: {out} (done-ok: {len(done)}, union: {union.height}, "
        f"to do: {len(jobs)})\nworkers: {args.workers}"
    )
    if not jobs:
        # Still re-write to drop any stale rows no longer in the union.
        _finalize(existing, union_full, out, wrote_new=0)
        return

    total = len(jobs)
    t0 = time.time()
    results: list[dict] = []
    n_err = 0
    n_match = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, pid, idx, claim, lang): (pid, idx)
            for pid, idx, claim, lang in jobs
        }
        for completed, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if r["error"]:
                n_err += 1
                print(f"  [{completed}/{total}] {r['post_id']}/c{r['claim_index']} "
                      f"ERROR: {r['error']}")
            else:
                n_match += int(r["fct_match"])
                if completed % 20 == 0 or completed == total:
                    print(f"  [{completed}/{total}] done ({n_match} matches so far)")

    if _quota_tripped.is_set():
        print("\n!! FCT quota circuit breaker tripped — remaining claims were "
              "skipped (recorded as errors). Re-run later to retry them.")

    new_df = pl.DataFrame(results, schema=FCT_SCHEMA)
    combined = pl.concat([existing, new_df]) if existing.height else new_df
    elapsed = time.time() - t0
    _finalize(combined, union_full, out, wrote_new=len(new_df), elapsed=elapsed,
              n_err=n_err, total=total)


def _finalize(combined: pl.DataFrame, union_full: pl.DataFrame, out: Path,
              wrote_new: int, elapsed: float | None = None,
              n_err: int = 0, total: int = 0) -> None:
    """Dedup keep-last, restrict to the current union, write, report."""
    keys = ["post_id", "claim_index"]
    combined = combined.unique(subset=keys, keep="last")
    # EXACTLY the union: drop any stale row whose claim is no longer check-worthy.
    combined = combined.join(union_full.select(keys), on=keys, how="semi")
    combined = combined.sort(keys)
    combined.write_parquet(out)
    total_match = int(combined.filter(pl.col("fct_match")).height)
    n_pending = int(combined.filter(pl.col("error").is_not_null()).height)
    tail = "" if elapsed is None else f"done in {elapsed:.1f}s. "
    print(
        f"\n{tail}wrote {combined.height} rows ({wrote_new} new) -> {out}\n"
        f"  union: {union_full.height} | errored/pending rows: {n_pending}"
        + (f" | this-run errors: {n_err}/{total}" if total else "")
        + f" | fct_match: {total_match}/{combined.height}"
    )


if __name__ == "__main__":
    main()
