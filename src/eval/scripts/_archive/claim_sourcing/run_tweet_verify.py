"""Run the v6 verify loop over a handoff claims parquet (the dev-500 by default).

Thin driver wiring the three existing pieces together: `load_verify_posts()` (the handoff
contract in build_verify_input.py) -> `harness.run_posts()` (checkpointed sharded JSONL,
resume-safe, live progress/ETA) -> `verify_post()` (pipeline/verify_tweet_claims.py).
Only posts with >=1 checkworthy claim run. All loop analytics live in the records:
per-round ledger snapshots + provider + docs-read flags (`rounds`), `close_round`,
`stopped`, `tries`, per-claim `claim_id` for joins back to flags/topics/NG strata.

  cd src && uv run python eval/scripts/claim_sourcing/run_tweet_verify.py \
      [-i eval/data/survey_claims/dev500_claims.parquet] \
      [-o eval/data/survey_claims/verify_v6_dev500] \
      [--sample N] [--seed 42] [--k 3] [--name smoke]

--sample N draws a stratified sample across NewsGuard bands (uses ng_score from the claims
parquet; strata never reach the prompts). Re-running with the same -o resumes: checkpointed
posts are skipped.
"""
import argparse, asyncio, random, sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from build_verify_input import load_verify_posts
from pipeline.harness import run_posts
from pipeline.pools import OrchestrationConfig, make_pools
from pipeline.verify_tweet_claims import verify_post


def _band(s):
    return "0-30" if s < 30 else "30-50" if s < 50 else "50-70" if s < 70 else "70-90" if s < 90 else "90-100"


def stratified_sample(posts, claims_parquet, n, seed):
    df = pd.read_parquet(claims_parquet)
    ng = {str(pid): float(g.iloc[0].get("ng_score") or 0) for pid, g in df.groupby("post_id")}
    by_band = defaultdict(list)
    for p in posts:
        by_band[_band(ng.get(p["post_id"], 0.0))].append(p)
    rng = random.Random(seed)
    for band in by_band.values():
        rng.shuffle(band)
    picked, i = [], 0
    while len(picked) < n and any(by_band.values()):     # round-robin across bands
        for band in sorted(by_band):
            if by_band[band] and len(picked) < n:
                picked.append(by_band[band].pop())
        i += 1
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default=str(SRC / "eval/data/survey_claims/dev500_claims.parquet"))
    ap.add_argument("-o", "--out-dir", default=str(SRC / "eval/data/survey_claims/verify_v6_dev500"))
    ap.add_argument("--sample", type=int, default=0, help="stratified sample size across NG bands (0 = all)")
    ap.add_argument("--posts", default=None, help="comma-separated post_ids: run ONLY these")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=0, help="posts in flight (0 = pool default)")
    ap.add_argument("--name", default="verify")
    args = ap.parse_args()

    posts = [p for p in load_verify_posts(args.input) if any(c["cw"] for c in p["claims"])]
    print(f"{len(posts)} verify-path posts in {args.input}")
    if args.posts:
        want = {s.strip() for s in args.posts.split(",")}
        posts = [p for p in posts if p["post_id"] in want]
        print(f"--posts filter: {len(posts)} of {len(want)} requested")
    if args.sample:
        posts = stratified_sample(posts, args.input, args.sample, args.seed)
        print(f"stratified sample: {len(posts)} posts "
              f"({', '.join(sorted(p['handle'] for p in posts))})")

    cfg = OrchestrationConfig(**({"k_posts": args.k} if args.k else {}))

    async def _run():
        pools = make_pools(cfg)
        try:
            return await run_posts(posts, verify_post, pools, args.out_dir, name=args.name)
        finally:
            await pools.close()

    print(asyncio.run(_run()))


if __name__ == "__main__":
    main()
