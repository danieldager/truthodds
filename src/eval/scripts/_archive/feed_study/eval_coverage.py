"""Score our extractor against Claimify's BingCheck coverage gold (binary).

GOLD: microsoft/claimify-dataset — 6,490 sentences from 396 long-form Bing Chat
answers, each labeled `contains_factual_claim` (bool) by 3 MS Research annotators.
This is a CLONE-VALIDATION: run our Claimify reimplementation (run_claimify.py) on
the sentences, compare to Claimify's published numbers.

Two metrics (we have binary labels only — no gold claims, so no element-level):
  SELECTION  — our `verifiable` bool vs the gold label. The FAIR comparison:
               robust to the missing surrounding context (we feed bare sentences),
               and the direct analog of Claimify's sentence-level coverage.
  COVERAGE   — did the full pipeline extract >=1 claim (outcome='claims'). Faithful
               to Claimify's "verifiable if >=1 claim extracted" definition, BUT
               understated here because bare sentences inflate disambiguation
               abstains (we strip the context Claimify provided). Reported with caveat.

Claimify's published English numbers (gpt-4o, sentence-level, Table 2):
  accuracy 91.8 | macro-F1 91.2  (element-level macro-F1 83.7; entailment 99.0)

  uv run python -m eval.scripts.feed_study.eval_coverage \
      --claimify data/bingcheck/claimify_gpt-oss-120b.parquet \
      --gold data/bingcheck/gold_sample1000.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

DATA = Path(__file__).parent / "data" / "bingcheck"
CLAIMIFY_ACC, CLAIMIFY_MACROF1 = 91.8, 91.2  # their published sentence-level (gpt-4o)


def prf(pred: list[bool], gold: list[bool]) -> dict:
    tp = sum(1 for p, g in zip(pred, gold) if p and g)
    fp = sum(1 for p, g in zip(pred, gold) if p and not g)
    fn = sum(1 for p, g in zip(pred, gold) if not p and g)
    tn = sum(1 for p, g in zip(pred, gold) if not p and not g)
    n = len(gold)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    Pn = tn / (tn + fn) if tn + fn else 0.0
    Rn = tn / (tn + fp) if tn + fp else 0.0
    F1n = 2 * Pn * Rn / (Pn + Rn) if Pn + Rn else 0.0
    return {"P": P, "R": R, "F1": F1, "macroF1": (F1 + F1n) / 2,
            "acc": (tp + tn) / n, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--claimify", type=Path, default=DATA / "claimify_gpt-oss-120b.parquet")
    ap.add_argument("--gold", type=Path, default=DATA / "gold_sample1000.parquet")
    args = ap.parse_args()

    cl = pl.read_parquet(args.claimify).filter(pl.col("error").is_null())
    gold = pl.read_parquet(args.gold)
    df = cl.join(gold, on="post_id", how="inner")
    n = df.height
    g = df["gold"].to_list()
    npos = sum(g)

    print("=" * 72)
    print(f"COVERAGE / SELECTION vs Claimify BingCheck gold  (n={n} sentences, "
          f"{npos} have a claim = {100*npos/n:.0f}%)")
    print(f"Claimify published (gpt-4o, sentence-level): acc {CLAIMIFY_ACC} | macro-F1 {CLAIMIFY_MACROF1}")
    print("=" * 72)

    # outcome funnel (what our pipeline did)
    funnel = dict(df.group_by("outcome").len().sort("len", descending=True).iter_rows())
    print(f"\nour pipeline outcomes: {funnel}")
    abst = funnel.get("abstained_ambiguous", 0)
    print(f"  disambiguation-abstain rate: {100*abst/n:.0f}%  "
          f"(bare sentences strip context -> inflates abstains; ~8% on our own posts)")

    # PRIMARY: Selection verifiable vs gold (fair, context-robust)
    sel = df["verifiable"].fill_null(False).to_list()
    m = prf(sel, g)
    print(f"\n[PRIMARY] SELECTION verifiable vs gold  (the fair comparison):")
    print(f"  acc={100*m['acc']:.1f}  macro-F1={100*m['macroF1']:.1f}  "
          f"P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f}  (TP{m['tp']}/FP{m['fp']}/FN{m['fn']}/TN{m['tn']})")
    print(f"  vs Claimify: acc {100*m['acc']:.1f} vs {CLAIMIFY_ACC}  |  "
          f"macro-F1 {100*m['macroF1']:.1f} vs {CLAIMIFY_MACROF1}")

    # SECONDARY: full-pipeline coverage (>=1 claim) vs gold — understated by abstains
    cov = (df["outcome"] == "claims").to_list()
    mc = prf(cov, g)
    print(f"\n[SECONDARY] FULL-PIPELINE coverage (>=1 claim extracted) vs gold:")
    print(f"  acc={100*mc['acc']:.1f}  macro-F1={100*mc['macroF1']:.1f}  "
          f"P={mc['P']:.3f} R={mc['R']:.3f} F1={mc['F1']:.3f}")
    print(f"  NOTE: understated — {100*abst/n:.0f}% abstained for lack of context, "
          f"not because the claim wasn't recognized (see SELECTION above).")

    # our claim-count distribution (flavor of multi-claim; no gold counts to compare)
    claimed = df.filter(pl.col("outcome") == "claims")
    if claimed.height:
        dist = dict(claimed.group_by("n_claims").len().sort("n_claims").iter_rows())
        mean_nc = claimed["n_claims"].mean()
        print(f"\nour claims-per-sentence (extracted sentences only, n={claimed.height}): "
              f"mean={mean_nc:.2f}  dist={dist}")
        print("  (no gold claim counts in the release -> can't score multi-claim; descriptive only)")


if __name__ == "__main__":
    main()
