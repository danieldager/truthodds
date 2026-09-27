"""Draw the LOCKED dev-500-B subset: 10 posts/source from the 50-source harvest, same
eligibility pipeline as the original sampler, DISJOINT from dev-500a and eval-5k.
Purpose (Daniel 2026-07-16): generalization / overfitting test — the extraction+verify
chain was iterated on dev-500a; B is run and audited with NO changes.

  cd src && uv run python eval/scripts/claim_sourcing/draw_dev500b.py [--seed 20260716]
"""
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sample_survey_subsets import _KEEP_LANGS, _PROMO_RE, dedupe_same_story, systematic_draw

SRC = Path(__file__).resolve().parents[3]
DATA = SRC / "eval/data/survey_claims"
TWEETS = DATA / "outlet_tweets.parquet"
ROSTER = DATA / "source_roster.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--dupe-threshold", type=float, default=0.6)
    ap.add_argument("--dev-n", type=int, default=10)
    args = ap.parse_args()
    import random
    rng = random.Random(args.seed)

    df = pd.read_parquet(TWEETS)
    src_hash = hashlib.sha256(TWEETS.read_bytes()).hexdigest()[:16]
    taken = set(pd.read_parquet(DATA / "survey_dev_500.parquet").post_id.astype(str)) | \
            set(pd.read_parquet(DATA / "survey_eval_5k.parquet").post_id.astype(str))
    roster = pd.read_csv(ROSTER)
    meta_cols = roster[["domain", "ng_score", "lean", "register", "followers"]]

    parts, funnel = [], []
    for domain, g in df.groupby("domain", sort=True):
        g = g[g.lang.isin(_KEEP_LANGS)
              & ~g.is_quote.astype(bool) & ~g.text.str.contains(_PROMO_RE)]
        g = g[~g.post_id.astype(str).isin(taken)]
        g, _ = dedupe_same_story(g, args.dupe_threshold)
        dv = systematic_draw(g, args.dev_n, rng)
        funnel.append({"domain": domain, "usable_disjoint": len(g), "drawn": len(dv)})
        parts.append(dv)

    dv = pd.concat(parts).merge(meta_cols, on="domain", how="left")
    assert not set(dv.post_id.astype(str)) & taken, "overlap with dev-500a/eval-5k!"
    assert dv.post_id.is_unique
    out = DATA / "survey_dev_500b.parquet"
    dv.to_parquet(out, index=False)
    out.chmod(0o444)  # locked, like its siblings
    prov = {"seed": args.seed, "dupe_threshold": args.dupe_threshold, "dev_n": args.dev_n,
            "source_parquet_sha256_16": src_hash, "disjoint_from": ["survey_dev_500", "survey_eval_5k"],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": len(dv), "funnel": funnel}
    (DATA / "survey_dev_500b_provenance.json").write_text(json.dumps(prov, indent=2))
    print(pd.DataFrame(funnel).to_string(index=False))
    print(f"\ndev-500-B: {len(dv)} rows -> {out} (locked read-only)")


if __name__ == "__main__":
    main()
