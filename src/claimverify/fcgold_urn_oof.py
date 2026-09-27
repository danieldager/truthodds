"""Persist the urn's out-of-fold scores on the pinned fc-gold population (n=3,274).

The published ladder numbers (eval/data/urn_runs/e1_ctx/model_ladder/ladder.json) are
computed in memory and never stored per claim. This writes them once, keyed by review_url,
so the loop-vs-urn comparison has a fixed urn side.

  cd src
  uv run python -m claimverify.fcgold_urn_oof

Protocol is model_ladder's, unchanged: 5 folds by blake2b(review_url) % 5, weights refit
per fold on the other four, mixed (veracity 3) claims scored but out of every fit.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from claimverify.config import SRC

sys.path.insert(0, str(SRC / "eval/scripts/build_eval"))
os.environ.setdefault("GOLD_EXCLUSIONS", "none")   # the pinned n=3,274 keeps the 63 gold media ids
import model_ladder  # noqa: E402

OUT = SRC / "eval/data/urn_runs/e1_ctx/model_ladder/oof_scores.parquet"
LADDER = SRC / "eval/data/urn_runs/e1_ctx/model_ladder/ladder.json"


def main() -> None:
    rows = [r for r in model_ladder.load_docs() if r["subtype"] != model_ladder.SET_ASIDE]
    y = np.array([r["y"] for r in rows])
    mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    df = pd.DataFrame({"review_url": [r["review_url"] for r in rows], "gold_true": y.astype(bool),
                       "mid": mid, "fold": fold, "subtype": [r["subtype"] for r in rows],
                       "n_docs": [len(r["docs"]) for r in rows]})
    ladder = json.loads(LADDER.read_text())
    for model, col in (("7-flag", "score_7flag"), ("3-voice", "score_3voice")):
        C = model_ladder.count_matrix(rows, model)
        s = model_ladder.oof_scores(C, y, mid, fold)
        df[col] = s
        auc = model_ladder.auc_np(s[y == 1], s[y == 0])
        pub = ladder["models"][model].get("auc_oof")
        print(f"{model}: n={len(s)} oof AUC {auc:.4f} (ladder.json {pub})")
    OUT.write_bytes(b"")
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT} n={len(df)} true={int(y.sum())} false={int((1 - y).sum())}")


if __name__ == "__main__":
    main()
