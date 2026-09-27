"""Run the v6 tweet-verify loop on the AVeriTeC dev split (GitHub issue #16) and score it
against gold — the same dataset ClaimCheck reports 76.4% on (Qwen3-4B, live Serper).

Each AVeriTeC claim becomes a single-claim "post" (claim text = post text, claim_date = post
date, the original claim URL's domain = origin for the exclude-origin rule). PARITY MODE
(2026-07-15, verified against idirlab/claimcheck code): DATE CEILING ON by default — their
web_search.py caps Serper at the claim date (tbs cd_max) — and Exa ON (Daniel). Scoring runs
over the FULL 500-row gold (duplicates share a claim_id and inherit its prediction).

Two scorings are printed (no extra LLM calls):
  raw       — our labels as-is: unsupported -> Not Enough Evidence (our v6 semantics)
  claimcheck-convention — unsupported -> Refuted (ClaimCheck maps "no supporting evidence
              found" to Refuted, which games AVeriTeC's fabrication-heavy Refuted class)

  cd src && uv run python eval/scripts/verification_grading/run_averitec_benchmark.py \
      [-i eval/scripts/verification_grading/data/claims_dev_full.parquet] \
      [-o eval/data/survey_claims/verify_v6_averitec] [--limit N] [--k 0] [--score-only]
"""
import argparse, asyncio, dataclasses, json, glob, sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

import pandas as pd

from pipeline.harness import run_posts
from pipeline.pools import OrchestrationConfig, make_pools
from pipeline.verify_tweet_claims import verify_post, CFG

GOLD = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
RAW_MAP = {"supported": "Supported", "refuted": "Refuted",
           "unsupported": "Not Enough Evidence", "conflicting": "Conflicting Evidence/Cherrypicking"}


def _iso(d):
    if not d:
        return None
    try:                                    # AVeriTeC dates are D-M-Y
        day, month, year = str(d).split("-")
        return f"{year}-{int(month):02d}-{int(day):02d}"
    except ValueError:
        return str(d)[:10]


def build_posts(df):
    posts = []
    for r in df.itertuples():
        url = r.original_claim_url if isinstance(r.original_claim_url, str) else ""
        dom = urlparse(url).netloc.lower().removeprefix("www.") if url else ""
        posts.append({"id": r.claim_id, "post_id": r.claim_id, "url": url or None,
                      "handle": (r.speaker if isinstance(r.speaker, str) and r.speaker else "claimant")[:40],
                      "domain": dom,
                      "date": _iso(r.claim_date), "text": r.claim_text,
                      "claims": [{"c": r.claim_text, "t": "assertion", "cw": True,
                                  "claim_id": r.claim_id}]})
    return posts


def score(out_dir, df):
    preds = {}
    for f in glob.glob(str(Path(out_dir) / "*.jsonl")):
        for line in open(f):
            d = json.loads(line)
            if d.get("ok"):
                preds[d["post_id"]] = d["result"]["ledger"].get("1")
    # score over EVERY gold row — duplicates share a claim_id and inherit its prediction
    rows = [(r.claim_id, preds.get(r.claim_id), r.gold_label)
            for r in df.itertuples() if preds.get(r.claim_id)]
    print(f"\nscoring {len(rows)} gold rows ({len(preds)} unique claims verified)")
    for name, mapping in (("raw", RAW_MAP),
                          ("claimcheck-convention", {**RAW_MAP, "unsupported": "Refuted"})):
        pairs = [(mapping.get(lab), g) for _, lab, g in rows]
        acc = sum(p == g for p, g in pairs) / max(1, len(pairs))
        f1s = []
        for cls in GOLD:
            tp = sum(1 for p, g in pairs if p == cls and g == cls)
            fp = sum(1 for p, g in pairs if p == cls and g != cls)
            fn = sum(1 for p, g in pairs if p != cls and g == cls)
            f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
        print(f"\n[{name}] accuracy {acc:.3f}  macro-F1 {sum(f1s)/len(f1s):.3f}  "
              f"(ClaimCheck reference: 0.764 acc)")
        conf = Counter((g, p) for p, g in pairs)
        print("  gold -> pred counts:")
        for g in GOLD:
            line_ = "  ".join(f"{p.split()[0][:4]}={conf.get((g, p), 0):3d}" for p in GOLD)
            print(f"    {g[:35]:37s} {line_}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default=str(SRC / "eval/scripts/verification_grading/data/claims_dev_500_gold.parquet"))
    ap.add_argument("-o", "--out-dir", default=str(SRC / "eval/data/survey_claims/verify_v7_averitec"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=0, help="posts in flight (0 = pool default)")
    ap.add_argument("--no-exa", action="store_true", help="disable the Exa escalation round")
    ap.add_argument("--no-date-ceiling", action="store_true", help="disable claim-date search ceiling")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)
    if not args.score_only:
        posts = build_posts(df.drop_duplicates("claim_id"))
        cfg = dataclasses.replace(CFG, exa_enabled=not args.no_exa,
                                  date_ceiling=not args.no_date_ceiling)
        pcfg = OrchestrationConfig(**({"k_posts": args.k} if args.k else {}))

        async def _run():
            pools = make_pools(pcfg)
            try:
                return await run_posts(posts, lambda p, pl: verify_post(p, pl, cfg),
                                       pools, args.out_dir, name="averitec")
            finally:
                await pools.close()

        print(asyncio.run(_run()))
    score(args.out_dir, pd.read_parquet(args.input))


if __name__ == "__main__":
    main()
