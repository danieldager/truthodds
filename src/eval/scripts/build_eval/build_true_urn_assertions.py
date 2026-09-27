"""Build tweet_corpus/true_urn_assertions.parquet — the 768-post / 1,225-claim
true-outlet population behind the wire_true two-urn transfer result (read by
wire_urn_run.py and outlet_aliases.py; the archived outlet_urn_run.py too).

It is a DETERMINISTIC parity screen over the cached extracted-claims file
`tweet_corpus/true_urn_claims_v47.parquet` (3,225 claims from the reputable-outlet
harvest, extract --voice outlet, v47). The screen is
`timeline_urn_run.parity` = assertion AND checkworthy, with the attribution-content
claims (`paired_content`) dropped, preserving the v47 row order:

    keep type == "assertion" AND checkworthy AND NOT paired_content

Row order and all 31 columns are taken verbatim from v47 (no re-sort). Reproduces
the pinned file frame-for-frame (all 1,225 rows / 31 columns / dtypes / order
identical; the on-disk parquet bytes differ only in pyarrow container metadata).

REPRODUCIBILITY: this builder is $0 — it reproduces the assertions file from the
CACHED extracted claims. The step UPSTREAM of it (the reputable-outlet SocialData
harvest -> true_urn_outlet_posts.parquet -> LLM extraction @ voice=outlet ->
true_urn_claims_v47.parquet) needs paid SocialData + DeepInfra calls and is NOT
reproducible at $0; true_urn_claims_v47.parquet is the frozen cache of that step.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

SRC = Path(".")
TC = SRC / "eval/data/tweet_corpus"
CLAIMS_V47 = TC / "true_urn_claims_v47.parquet"
OUT = TC / "true_urn_assertions.parquet"


def build(claims_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(claims_path)
    mask = (df["type"] == "assertion") & (df["checkworthy"] == True) & (df["paired_content"] == False)  # noqa: E712
    return df[mask].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", type=Path, default=CLAIMS_V47)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--force", action="store_true",
                    help="overwrite the pinned true_urn_assertions.parquet")
    ap.add_argument("--verify", action="store_true",
                    help="build to memory and assert frame-equality with the pinned file; write nothing")
    a = ap.parse_args()

    built = build(a.claims)

    if a.verify:
        pinned = pd.read_parquet(OUT)
        assert built.equals(pinned), "rebuilt frame != pinned true_urn_assertions.parquet"
        print(f"VERIFY OK: {len(built)} rows / {built.shape[1]} cols frame-identical to {OUT} "
              f"({built['post_id'].nunique()} posts)")
        return

    if a.out.resolve() == OUT.resolve() and not a.force:
        sys.exit(f"refusing to overwrite the pinned {OUT}: pass --force to regenerate, "
                 f"or --out <path> to write elsewhere (or --verify to only check)")
    built.to_parquet(a.out, index=False)
    print(f"wrote {a.out} ({len(built)} rows / {built.shape[1]} cols / "
          f"{built['post_id'].nunique()} posts)")


if __name__ == "__main__":
    main()
