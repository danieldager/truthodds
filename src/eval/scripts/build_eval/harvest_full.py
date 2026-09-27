"""Exhaustive Google Fact Check Tools harvest — extend fc-gold as deep as GFC allows.

Unlike harvest_incremental.py (38-day rolling window, 8 publishers), this pulls each
publisher's FULL history (no maxAgeDays; paginate until the token runs out), over an
extended roster (curated 8 + the international fact-checkers the verify loop already
trusts), in EN + FR. The review/claim dates are kept intact — harvest depth is the
TEMPORAL dimension for Truth Odds constant fitting (Daniel 2026-07-23): p̂'s can be
conditioned on claim age once the pool spans years.

Raw pull only — harmonization (veracity 1–5 + nee tag) is a separate pass so the
mapping can be reviewed before any LLM fallback spend.

  uv run python -m eval.scripts.build_eval.harvest_full --smoke      # 3 pages/pub, 2 pubs
  uv run python -m eval.scripts.build_eval.harvest_full              # the real pull (GATED)

Outputs:
  eval/data/fc_harvest_full.parquet      raw deduped pull (10-field projection + meta)
  eval/data/fc_harvest_full_state.json   per-(publisher,lang) progress — resumable;
                                         completed pairs are skipped on re-run
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.harvest import fetch_claims_raw
from pipeline.credibility import TRUSTED_FACTCHECKERS

DATA_DIR = Path("eval/data")
OUT = DATA_DIR / "fc_harvest_full.parquet"
STATE = DATA_DIR / "fc_harvest_full_state.json"
MASTER = DATA_DIR / "eval_master.parquet"  # dedup against the existing rolling master

# Curated 8 (harvest_incremental) + trusted internationals (pipeline/credibility.py,
# domain-style entries only). GFC indexes a subset; a publisher it doesn't index
# costs one request and returns empty.
CURATED_PUBS = {
    "snopes.com", "factcheck.afp.com", "newschecker.in", "verafiles.org",
    "rumorscanner.com", "politifact.com", "factcheck.org", "fullfact.org",
}
ROSTER = sorted(CURATED_PUBS | {d for d in TRUSTED_FACTCHECKERS if "/" not in d}
                | {"factuel.afp.com"})

LANGS = ["en", "fr"]
PAGE_SIZE = 100
SLEEP_S = 0.4          # politeness between pages (free API; no reason to hammer)
MAX_PAGES = 400        # hard runaway cap per (publisher, lang) = 40k claims


def pull_publisher(site: str, lang: str, max_pages: int) -> list[dict]:
    """Full-history paginated pull for one (publisher, language)."""
    rows, token, pages = [], None, 0
    while pages < max_pages:
        resp = fetch_claims_raw(
            review_publisher_site_filter=site, language_code=lang,
            page_size=PAGE_SIZE, page_token=token,
        )
        pages += 1
        for c in resp.get("claims", []):
            for r in c.get("claimReview", []):
                rows.append({
                    "claim_text": c.get("text"),
                    "claimant": c.get("claimant"),
                    "claim_date": c.get("claimDate"),
                    "publisher_site": (r.get("publisher") or {}).get("site") or site,
                    "publisher_name": (r.get("publisher") or {}).get("name"),
                    "review_url": r.get("url"),
                    "review_title": r.get("title"),
                    "review_date": r.get("reviewDate"),
                    "language_code": r.get("languageCode") or lang,
                    "original_rating": r.get("textualRating"),
                })
        token = resp.get("nextPageToken")
        if not token:
            break
        time.sleep(SLEEP_S)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 publishers x 3 pages, EN only — pagination + date-depth check")
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES)
    args = ap.parse_args()

    roster = ["politifact.com", "boomlive.in"] if args.smoke else ROSTER
    langs = ["en"] if args.smoke else LANGS
    max_pages = 3 if args.smoke else args.max_pages

    state = json.loads(STATE.read_text()) if STATE.exists() and not args.smoke else {}
    existing = pl.read_parquet(OUT) if OUT.exists() and not args.smoke else None
    seen = set(existing["review_url"].to_list()) if existing is not None else set()
    if MASTER.exists():
        pass  # master rows are NOT excluded — they lack the full-history context; dedup happens at merge

    all_new = []
    t0 = time.time()
    for site in roster:
        for lang in langs:
            key = f"{site}|{lang}"
            if state.get(key, {}).get("done"):
                print(f"[skip] {key} (done, {state[key]['n']} rows)", flush=True)
                continue
            t1 = time.time()
            try:
                rows = pull_publisher(site, lang, max_pages)
            except Exception as e:
                print(f"[FAIL] {key}: {e}", flush=True)
                continue
            fresh = [r for r in rows if r.get("review_url") and r["review_url"] not in seen]
            seen.update(r["review_url"] for r in fresh)
            all_new.extend(fresh)
            dates = sorted(d for d in (r.get("review_date") for r in rows) if d)
            span = f"{dates[0][:10]} .. {dates[-1][:10]}" if dates else "no dates"
            print(f"[done] {key}: {len(rows)} rows ({len(fresh)} new) in "
                  f"{time.time()-t1:.0f}s | review-date span: {span}", flush=True)
            if not args.smoke:
                state[key] = {"done": True, "n": len(rows),
                              "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                STATE.write_text(json.dumps(state, indent=1))

    if not all_new:
        print("nothing new pulled")
        return
    df = pl.DataFrame(all_new)
    if args.smoke:
        print(f"\nSMOKE: {len(df)} rows total — not written. Rating spread:")
        print(df.group_by("original_rating").len().sort("len", descending=True).head(20))
        return
    if existing is not None:
        df = pl.concat([existing, df], how="diagonal")
    df = df.unique(subset=["review_url"], keep="first")
    df.write_parquet(OUT)
    print(f"\nwrote {OUT}: {len(df)} rows total (+{len(all_new)} this run, "
          f"{time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
