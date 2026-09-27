"""Refit the truth-odds urn on the v7qa reader, so all three frozen populations
are read by one reader.

The shipped 7-flag weights and the rating cuts (`populations/refit_results.json`,
`headline_metrics.json`) were fitted on read-v5 Flash reads. Under v7qa the flag
"4" means a narrower thing ("the document points toward the claim without
establishing it") and its read-v5 weight no longer applies — the mixed-reader fit
of 2026-09-10 put w_4 at ~0, which was a units mismatch, not a fact about truth.
This script re-runs the same three fits on the v7qa re-reads:

  (a) 7-flag weights on fc-gold, cluster-disjoint 5-fold oof, clustered bootstrap,
      nested 2% operating point (the `graded_urn_7flag` cell);
  (b) two-urn de-mix cn_false vs x_feed at the headline eps, transferred to fc-gold
      with fixed weights (the `two_urn_7flag` cell);
  (c) the four rating cuts on fc-gold oof scores under the new (a) weights, nested
      in the same folds (clog/090926 11:40, scale as fixed 2026-09-09 13:20:
      cut_1 = 2% false-alarm, cut_2 = 5% budget, cut_4 = 10% miss, cut_5 = 2% miss).

Nothing here re-implements the model: fit_urn / graded_urn / fit_two_urn are
imported. The only new input is the substituted run files written by
`substitute_reader_flags.py`.

    uv run python -m eval.scripts.build_eval.refit_v7qa \
        --sub-dir eval/data/reader_lab/refit_v7qa/substituted \
        --out eval/data/populations/refit_results_v7qa.json

WIRE_TRUE variant — swap the two-urn TRUE side from x_feed to the reputable-outlet
wire_true reads (population=None: there is no frozen parquet for this population,
its filter is "everything in the run file"). This is the committed writer for
`populations/refit_v7qa_wire_true.json` (two-urn cn_false vs wire_true -> fc_gold,
eps 0.10 AUC 0.8171; clog/110926.md 10:25):

    uv run python -m eval.scripts.build_eval.refit_v7qa \
        --sub-dir eval/data/reader_lab/refit_v7qa/substituted \
        --true-file wire_true__scores.jsonl --true-population none \
        --out eval/data/populations/refit_v7qa_wire_true.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.scripts.build_eval import fit_two_urn as f2  # noqa: E402
from eval.scripts.build_eval import fit_urn, graded_urn  # noqa: E402

POP = Path("eval/data/populations")
FLAGS = graded_urn.FLAGS                      # display order 5 4 3 X I 2 1
EPS_HEADLINE = 0.10
READER_LABEL = ["v7qa gpt-oss-120b low @ DeepInfra"]   # --reader overrides it
# the read-v5 rows this refit is measured against (populations/refit_results.json)
BASELINE = {"graded_urn_7flag": {"auc": 0.8559, "recall": 0.4003},
            "two_urn_7flag": {"auc": 0.8453, "recall": 0.3951},
            # clog/090926.md 11:40. The FALSE-ALARM pair has a pinned artefact behind it:
            # graded_metrics_clustered.json's thresholds_by_fold are
            # [-3.9619, -3.8865, -4.13, -4.2101, -4.2198], which rating_cuts reproduces
            # EXACTLY on read-v5 flags and whose mean is -4.0817 (the clog quotes -4.08).
            # The MISS pair has none — budget_bands.py was a scratch script and is gone —
            # and the nested protocol gives +0.126 / +5.725 against the quoted +0.03 / +5.56.
            # Flagged for Daniel; the nested definition is what ships.
            "cuts": {"cut_1": -4.079, "cut_2": -2.075, "cut_4": 0.03, "cut_5": 5.56}}


# ---- the four rating cuts --------------------------------------------------
# Definition (clog/090926.md 11:40, scale fixed 2026-09-09 13:20). Each cut is a
# NESTED threshold in exactly the sense fit_urn.nested_threshold already means it:
# for each outer fold the four training folds are scored out-of-fold among
# themselves (an inner refit per inner fold), the threshold is picked on THOSE
# scores, and the cut quoted is the MEAN of the five per-fold thresholds. That is
# what produced the pinned -4.079 / -2.075 / 0.034 / 5.560, and the pinned
# graded cell's own thresholds_by_fold average to -4.0817.
#   cut_1 = 2% false alarm   (gold-TRUE claims at or below it)
#   cut_2 = 5% false alarm
#   cut_4 = 10% miss         (gold-FALSE claims strictly above it)
#   cut_5 = 2% miss
def _thr_false_alarm(pairs, budget):
    """Largest threshold whose FPR stays inside the budget (fit_urn's own choice)."""
    return fit_urn.recall_at_fpr(pairs, budget)[2]


def _thr_miss(pairs, budget):
    """Smallest threshold whose MISS rate (gold-FALSE strictly above it) stays
    inside the budget. The claims above the cut are the supported band."""
    falses = sorted(s for s, y in pairs if y == 0)
    n = len(falses)
    for t in sorted({s for s, _ in pairs}):
        if (n - bisect.bisect_right(falses, t)) / n <= budget:
            return float(t)
    return float("inf")


CUTS = {"cut_1": (_thr_false_alarm, 0.02), "cut_2": (_thr_false_alarm, 0.05),
        "cut_4": (_thr_miss, 0.10), "cut_5": (_thr_miss, 0.02)}


def rating_cuts(rows):
    """The four cuts, nested over the cluster-disjoint folds. Per outer fold the
    threshold is chosen on the INNER out-of-fold scores of the four training folds
    (fit_urn.oof_np, so the weights are refit inside), and the cut is their mean."""
    C7 = np.array([graded_urn.flag_counts(r) for r in rows], float).T
    y = np.array([r["y"] for r in rows])
    mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    idx = np.arange(C7.shape[1])
    inner = {}
    for k in range(fit_urn.K_FOLDS):
        tr = idx[fold != k]
        s_in = fit_urn.oof_np(C7, y, mid, fold, tr)
        inner[k] = list(zip(s_in.tolist(), y[tr].tolist()))
    oof = graded_urn.oof_graded(rows)
    falses = sorted(s for s, yy in oof if yy == 0)
    trues = sorted(s for s, yy in oof if yy == 1)
    out = {}
    for name, (chooser, budget) in CUTS.items():
        by_fold = [float(chooser(inner[k], budget)) for k in range(fit_urn.K_FOLDS)]
        cut = float(np.mean(by_fold))
        out[name] = {
            "kind": "false_alarm" if chooser is _thr_false_alarm else "miss",
            "budget": budget, "cut": cut, "by_fold": by_fold,
            "fold_min": min(by_fold), "fold_max": max(by_fold),
            "share_false_below": bisect.bisect_right(falses, cut) / len(falses),
            "share_true_below": bisect.bisect_right(trues, cut) / len(trues),
            "baseline_readv5": BASELINE["cuts"][name],
        }
    return out, oof


# ---- (a) 7-flag refit on fc-gold ------------------------------------------
def fit_gold(rows):
    oof7 = graded_urn.oof_graded(rows)
    oof3 = fit_urn.out_of_fold(rows)
    w7 = graded_urn.fit_graded(rows)
    auc7 = fit_urn.auc(oof7)
    C7 = np.array([graded_urn.flag_counts(r) for r in rows], float).T
    nest = fit_urn.nested_threshold(C7, np.array([r["y"] for r in rows]),
                                    np.array([r["mid"] for r in rows]),
                                    np.array([r["fold"] for r in rows]))
    print(f"  bootstrap ({fit_urn.BOOT_REPS} reps, seed {fit_urn.BOOT_SEED}) ...", flush=True)
    ci, dauc_ci, deff, _ = graded_urn.bootstrap(rows, fit_urn.BOOT_REPS, fit_urn.BOOT_SEED)
    n_true = sum(r["y"] for r in rows)
    return {
        "variant": "graded_urn_7flag", "fit": "refit on fc-gold labels, v7qa reads",
        "population": "fc_gold", "reader": READER_LABEL[0],
        "n": len(rows), "n_true": n_true, "n_false": len(rows) - n_true,
        "docs": int(C7.sum()),
        "auc": {"point": auc7, "ci95_cluster": deff["auc_ci95"],
                "ci95_row": deff["auc_ci95_row"],
                "insample": fit_urn.auc([(graded_urn.score_graded(r, w7), r["y"]) for r in rows]),
                "three_voice_point": fit_urn.auc(oof3),
                "delta_vs_3voice": auc7 - fit_urn.auc(oof3),
                "delta_ci95_cluster": list(dauc_ci),
                "baseline_readv5": BASELINE["graded_urn_7flag"]["auc"]},
        "nested_recall_at_2pct_fpr": {"recall": nest["recall"], "fpr": nest["fpr"],
                                      "thresholds_by_fold": nest["thresholds_by_fold"],
                                      "threshold_spread": nest["threshold_spread"],
                                      "baseline_readv5": BASELINE["graded_urn_7flag"]["recall"]},
        "weights": {k: w7[k] for k in FLAGS},
        "weights_ci": {k: list(ci[k]) for k in FLAGS},
    }, oof7


# ---- (b) two-urn de-mix, transferred to fc-gold ---------------------------
def fit_two(false_rows, true_rows, gold, eps_list):
    pf3, pm3 = f2.rates(false_rows), f2.rates(true_rows)
    pf7, pm7 = f2.flag_rates(false_rows), f2.flag_rates(true_rows)
    docs_m = sum(sum(r["flags"].values()) for r in true_rows)
    docs_f = sum(sum(r["flags"].values()) for r in false_rows)
    print(f"  bootstrap ({f2.BOOT_REPS} reps, seed {f2.BOOT_SEED}) ...", flush=True)
    boot = f2.bootstrap_weights(false_rows, true_rows, eps_list)
    cells = []
    for eps in eps_list:
        w3 = f2.demix_weights(pm3, pf3, eps)
        w7 = f2.demix_flag_weights(pm7, pf7, eps, (docs_m, docs_f))
        if w3 is None or w7 is None:
            continue
        ev7 = f2.transfer_eval7(w7, gold); ev7.pop("pairs", None)
        b = boot["by_eps"][eps]
        cells.append({
            "variant": "two_urn_7flag", "eps": eps,
            "fit": f"label-free two-urn de-mix, eps={eps}; transfer, no refit; v7qa reads",
            "population": "cn_false + x_feed -> fc_gold",
            "n_fit_false": len(false_rows), "n_fit_true": len(true_rows), "n_gold": len(gold),
            "docs_false": docs_f, "docs_true": docs_m,
            "flag_rates_false": pf7, "flag_rates_true_mix": pm7,
            "auc": {"point": ev7["auc"],
                    "baseline_readv5": BASELINE["two_urn_7flag"]["auc"]},
            "nested_recall_at_2pct_fpr": {
                "recall": ev7["recall_at_2pct_fpr"], "fpr": ev7["fpr"],
                "thresholds_by_fold": ev7["thresholds_by_fold"],
                "threshold_spread": ev7["threshold_spread"],
                "baseline_readv5": BASELINE["two_urn_7flag"]["recall"]},
            "weights": w7, "weights_ci": b["weights_ci7"],
            "three_voice": {"weights": w3, "weights_ci": b["weights_ci"],
                            "fc_gold_transfer": f2.transfer_eval(w3, gold)},
        })
    return cells


def flag_table(name, rows):
    tot = sum(sum(r["flags"].values()) for r in rows)
    return {"population": name, "n_claims": len(rows), "docs_padded": tot,
            "rates": {k: sum(r["flags"].get(k, 0) for r in rows) / tot for k in FLAGS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", required=True,
                    help="directory of substitute_reader_flags.py output")
    ap.add_argument("--out", default=str(POP / "refit_results_v7qa.json"))
    ap.add_argument("--eps", type=float, nargs="*", default=[0.0, 0.05, 0.10, 0.15])
    ap.add_argument("--gold-only", action="store_true",
                    help="fit only (a) and (c) on fc-gold; skip the two-urn cell. For a "
                         "reader variant re-read on fc-gold alone — the two-urn fit would "
                         "otherwise mix that reader with another one on cn_false/x_feed.")
    ap.add_argument("--reader", default="v7qa gpt-oss-120b low @ DeepInfra",
                    help="reader label recorded in the output json")
    ap.add_argument("--true-file", default="x_feed__scores.jsonl",
                    help="two-urn TRUE-side reads file inside --sub-dir "
                         "(e.g. wire_true__scores.jsonl for the wire_true variant)")
    ap.add_argument("--true-population", default="x_feed.parquet",
                    help="frozen population parquet name for the TRUE urn, or 'none' "
                         "for a population-free urn (wire_true has no frozen parquet)")
    a = ap.parse_args()
    READER_LABEL[0] = a.reader
    sub = Path(a.sub_dir)

    gold = fit_urn.load_headline(sub / "fc_gold__results-00.jsonl",
                                 population=fit_urn.load_population(POP / "fc_gold.parquet"))
    false_rows, true_rows = [], []
    if not a.gold_only:
        false_rows = f2.load_urn([sub / "cn_false__scores.jsonl", sub / "cn_false__scores_ext.jsonl"],
                                 f2.C2 / "fit_exclusions.json",
                                 fit_urn.load_population(POP / "cn_false.parquet", "false"))
        true_pop = (None if a.true_population.lower() == "none"
                    else fit_urn.load_population(POP / a.true_population, "true"))
        true_rows = f2.load_urn([sub / a.true_file], population=true_pop)
    print(f"fc_gold {len(gold)} | cn_false {len(false_rows)} | x_feed {len(true_rows)}", flush=True)

    tables = [flag_table("fc_gold", gold)] + ([] if a.gold_only else
             [flag_table("cn_false", false_rows), flag_table("x_feed", true_rows)])
    print(f"\nflag rates per document (pad-to-10), {a.reader}:")
    print(f"{'population':<10} {'claims':>7} {'docs':>7} " + " ".join(f"{k:>7}" for k in FLAGS))
    for t in tables:
        print(f"{t['population']:<10} {t['n_claims']:>7} {t['docs_padded']:>7} "
              + " ".join(f"{t['rates'][k]:>7.4f}" for k in FLAGS))

    print("\n(a) 7-flag refit on fc-gold", flush=True)
    cell_a, _ = fit_gold(gold)
    print(f"  AUC oof {cell_a['auc']['point']:.4f} "
          f"[{cell_a['auc']['ci95_cluster'][0]:.4f}, {cell_a['auc']['ci95_cluster'][1]:.4f}] "
          f"(read-v5 {BASELINE['graded_urn_7flag']['auc']:.4f})   "
          f"recall@2% nested {cell_a['nested_recall_at_2pct_fpr']['recall']:.4f} "
          f"(read-v5 {BASELINE['graded_urn_7flag']['recall']:.4f})")
    for k in FLAGS:
        ci = cell_a["weights_ci"][k]
        print(f"    {k}: {cell_a['weights'][k]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  "
              f"{graded_urn.FLAG_DESC[k]}")

    cells_b = []
    if not a.gold_only:
        print("\n(b) two-urn de-mix cn_false vs x_feed -> fc-gold", flush=True)
        cells_b = fit_two(false_rows, true_rows, gold, a.eps)
    for c in cells_b:
        print(f"  eps {c['eps']:.2f}  AUC {c['auc']['point']:.4f}  "
              f"recall@2% {c['nested_recall_at_2pct_fpr']['recall']:.4f}   "
              + " ".join(f"{k} {c['weights'][k]:+.2f}" for k in FLAGS))
    for c in cells_b:
        if abs(c["eps"] - EPS_HEADLINE) < 1e-9:
            for k in FLAGS:
                ci = c["weights_ci"][k]
                print(f"    {k}: {c['weights'][k]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    print("\n(c) rating cuts on fc-gold oof scores under the (a) weights", flush=True)
    cuts, _ = rating_cuts(gold)
    for name in ("cut_1", "cut_2", "cut_4", "cut_5"):
        c = cuts[name]
        print(f"  {name} ({c['kind']} {c['budget']:.0%}): {c['cut']:+.3f}  "
              f"folds [{c['fold_min']:+.3f}, {c['fold_max']:+.3f}]  "
              f"(read-v5 {c['baseline_readv5']:+.3f})  "
              f"false below {c['share_false_below']:.3f} / true below {c['share_true_below']:.3f}")

    out = {"generated": "refit_v7qa.py",
           "reader": {"prompt": "v7qa", "mapper": "map_qa",
                      "model": "openai/gpt-oss-120b", "reasoning": "low",
                      "provider": "deepinfra", "json_mode": False},
           "substituted_runs": str(sub),
           "populations_manifest": str(POP / "manifest.json"),
           "pad_to": fit_urn.PAD_TO, "folds": fit_urn.K_FOLDS, "fold_key": "cluster_id",
           "eps_headline": EPS_HEADLINE,
           "bootstrap": {"reps": fit_urn.BOOT_REPS, "seed": fit_urn.BOOT_SEED,
                         "cluster_design": "cluster_id (fc-gold) / post_id within urn (two-urn)"},
           "baseline_readv5": "eval/data/populations/refit_results.json",
           "flag_rates": tables,
           "cells": [cell_a] + cells_b,
           "rating_cuts": cuts}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
