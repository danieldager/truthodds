"""Score an extraction claims parquet against the gold calibration cases.

The gold file (`extraction_gold_cases.json`) holds hand-ruled posts (Daniel's verify-audit
rulings + the PI's review) with machine-checkable expectations. Small and directional — read the
per-case output, don't tune to the pass count.

  cd src && uv run python eval/scripts/claim_sourcing/check_extraction_gold.py \
      [-i eval/data/survey_claims/dev500_claims.parquet] \
      [--gold eval/data/survey_claims/extraction_gold_cases.json]

Check kinds (patterns are case-insensitive regex over claim text; optional "type" restricts):
  has_claim               >=1 extracted claim matches
  has_checkworthy         >=1 cw=True claim matches
  no_checkworthy_claims   the post has NO cw=True claim
  content_not_assertion   no assertion-typed claim matches (any cw)
  content_not_checkworthy no cw=True claim matches
  content_not_attribution no attribution-typed claim matches
"""
import argparse, json, re
from pathlib import Path
import pandas as pd

SRC = Path(__file__).resolve().parents[3]


def run_check(chk, claims):
    pat = re.compile(chk.get("pattern", ""), re.I) if chk.get("pattern") else None
    typ = chk.get("type")
    pool = [c for c in claims if not typ or c.get("type") == typ]
    hits = [c for c in pool if pat is None or pat.search(c["claim"] or "")]
    kind = chk["kind"]
    if kind == "has_claim":
        return bool(hits)
    if kind == "has_checkworthy":
        return any(c["checkworthy"] for c in hits)
    if kind == "no_checkworthy_claims":
        return not any(c["checkworthy"] for c in claims)
    if kind == "content_not_assertion":
        return not any(c["type"] == "assertion" for c in hits)
    if kind == "content_not_checkworthy_assertion":
        # the content may survive as an attribution or a non-checkworthy claim; what it must
        # never be is a BARE assertion entering the verify ledger
        return not any(c["type"] == "assertion" and c["checkworthy"] for c in hits)
    if kind == "content_not_checkworthy":
        return not any(c["checkworthy"] for c in hits)
    if kind == "content_not_attribution":
        return not any(c["type"] == "attribution" for c in hits)
    raise ValueError(f"unknown check kind: {kind}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default=str(SRC / "eval/data/survey_claims/dev500_claims.parquet"))
    ap.add_argument("--gold", default=str(SRC / "eval/data/survey_claims/extraction_gold_cases.json"))
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    if "post_id" not in df.columns and "url" in df.columns:  # normalize_tweet_claims parquet has url only
        df["post_id"] = df.url.str.rsplit("/", n=1).str[-1]
    manifest_path = Path(args.input).with_name(Path(args.input).stem + "_posts_manifest.parquet")
    in_set = (set(pd.read_parquet(manifest_path).post_id.astype(str))
              if manifest_path.exists() else None)
    gold = json.loads(Path(args.gold).read_text())
    npass = ntot = nskip = 0
    for case in gold["cases"]:
        if in_set is not None and case["post_id"] not in in_set:
            print(f"\n== {case['case']}  (@{case['post_id']}, {case['source']})")
            print("   -- SKIPPED: post not in this run's input set --")
            nskip += 1
            continue
        claims = df[df.post_id.astype(str) == case["post_id"]][["claim", "type", "checkworthy"]].to_dict("records")
        print(f"\n== {case['case']}  (@{case['post_id']}, {case['source']})")
        print(f"   {case['ruling']}")
        for c in claims:
            cw = "cw=True " if c["checkworthy"] else "cw=False"
            typ = c["type"] if isinstance(c["type"], str) else "REMOVED"
            print(f"     [{typ:11s} {cw}] {c['claim'][:110]}")
        if not claims:
            print("     (no claims extracted for this post)")
        if not case["checks"]:
            print("   -- discuss-only, no checks --")
            continue
        for chk in case["checks"]:
            ok = run_check(chk, claims)
            npass += ok; ntot += 1
            lab = " ".join(f"{k}={v}" for k, v in chk.items() if k != "kind")
            print(f"   {'PASS' if ok else 'FAIL'}  {chk['kind']} {lab}")
    skip_note = f"; {nskip} case(s) skipped (post not in input set)" if nskip else ""
    print(f"\n{npass}/{ntot} checks pass{skip_note}  (directional — read the cases, don't tune to the count)")


if __name__ == "__main__":
    main()
