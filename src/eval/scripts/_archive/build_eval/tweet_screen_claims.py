"""E2 claim screen — re-audit tweet claims for ill-posedness (Daniel 2026-08-04).

Reuses screen_claims.screen() UNMODIFIED; only the input adapter differs, same
pattern as tweet_urn_run.py. Written because a regex screen was not good enough:
a claim-initial pronoun regex returned 2.1%, but broadening it to leading definite
descriptions ("The bill would confiscate assets", "The plan is to control 100
percent of Gaza") raised the union to 4.1%, and "any third-person pronoun
anywhere" hits 22.5%. Regexes bound the answer between 4.1% and 22.5% and cannot
narrow it — the class is semantic, not lexical.

The screen sees CLAIM + DESCRIPTION(=post text) + DATE, so its verdict answers the
operative question directly: is the referent nameable FROM THE POST? A claim the
screen still calls unresolved_referent with the post in hand cannot be fixed by a
within-post coreference pass, which settles whether resolution or exclusion is the
right treatment.

Taxonomy transfer is UNTESTED: SCREEN_SYS was written for fc-gold claims, where
`misattributed_media` and the context_leak check are about a fact-check verdict
that tweets do not have. Read the non-ok cases before trusting the counts.

  uv run python -m eval.scripts.build_eval.tweet_screen_claims --smoke
  uv run python -m eval.scripts.build_eval.tweet_screen_claims

Writes eval/data/survey_claims/tweet_claim_screen.parquet + an HTML page.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval.screen_claims import build_page, screen
from eval.scripts.build_eval.tweet_urn_run import adapt, load_claims

OUT = Path("eval/data/survey_claims/tweet_claim_screen.parquet")
PAGE = Path("eval/data/survey_claims/tweet_claim_screen.html")
OUT_NC = Path("eval/data/survey_claims/tweet_claim_screen_nocontext.parquet")
PAGE_NC = Path("eval/data/survey_claims/tweet_claim_screen_nocontext.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="50 claims, cost probe")
    ap.add_argument("--workers", type=int, default=8)
    # The WITH-context pass answers the QUERY step's question (query-v3 receives the
    # post). READ receives claim + date ONLY (evidence_urn_run.py:648), so the
    # ill-posed-for-READ population is the DELTA between the two passes. Run both.
    ap.add_argument("--no-context", action="store_true",
                    help="screen the claim ALONE, as READ sees it")
    args = ap.parse_args()

    df = load_claims().filter(pl.col("verify_eligible")).sort("claim_id")
    rows = []
    for r in df.iter_rows(named=True):
        a = adapt(r)
        rows.append({
            "review_url": a["review_url"],
            "claim_for_screen": a["claim_text"],
            # the post text IS the context (context_ok=True for every tweet row)
            "ctx_for_screen": "" if args.no_context else (a["x_context"] or "").strip(),
            "date_for_screen": a["claim_date"],
            "bin": r["bin"], "post_id": r["post_id"], "domain": r["domain"],
        })
    if args.smoke:
        rows = rows[::max(1, len(rows) // 50)][:50]
    print(f"screening {len(rows)} claims | {args.workers} workers", flush=True)

    lock, state = threading.Lock(), {"n": 0, "cost": 0.0}
    out, t0 = [], time.time()

    def work(row):
        v, cost = screen(row)
        with lock:
            state["n"] += 1
            state["cost"] += cost
            out.append({**row, **v})
            if state["n"] % 200 == 0 or state["n"] == len(rows):
                el = (time.time() - t0) / 60
                print(f"  {state['n']}/{len(rows)} | ${state['cost']:.3f} | {el:.1f}m | "
                      f"proj ${state['cost']/state['n']*len(rows):.2f}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, rows))

    res = pl.DataFrame(out)
    if not args.smoke:
        o, pg = (OUT_NC, PAGE_NC) if args.no_context else (OUT, PAGE)
        res.write_parquet(o)
        pg.write_text(build_page(res))
        print(f"\nwrote {o} and {pg}")
    print(res["verdict"].value_counts().sort("count", descending=True))
    # per-bin, because a post-level exclusion rule that fires on ANY bad claim
    # scales with claims/post -- which is itself the bin gradient (2.88 -> 1.54).
    print(res.group_by("bin").agg(
        pl.len().alias("claims"),
        (pl.col("verdict") != "ok").sum().alias("not_ok"),
        pl.col("post_id").n_unique().alias("posts"),
        pl.col("post_id").filter(pl.col("verdict") != "ok").n_unique().alias("posts_hit"),
    ).with_columns(
        (pl.col("not_ok") / pl.col("claims")).round(3).alias("claim_rate"),
        (pl.col("posts_hit") / pl.col("posts")).round(3).alias("post_excl_rate"),
    ).sort("bin"))
    print(f"total ${state['cost']:.3f} | {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
