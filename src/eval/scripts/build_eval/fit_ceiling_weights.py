"""Fit the FROZEN survey instrument's training-mode ("ceiling") weights, and
ship them to the production headline file.

Produces `eval/data/urn_runs/e1_ctx/weights_v5_ceiling_bal3000.json`: the
seven-flag graded-urn weights + nested 2%-FPR boundary that the CN-survey pool
is scored under (frozen 2026-09-18). "Ceiling" = the training-mode read set
(cached read-v5 @ DeepSeek-V4-Flash reads in e1_ctx/results-00.jsonl), fit on
the balanced 3,000-claim slice fc_gold_bal3000.

    uv run python -m eval.scripts.build_eval.fit_ceiling_weights            # -> pinned weights file (guarded)
    uv run python -m eval.scripts.build_eval.fit_ceiling_weights --out /tmp/w.json  # gate to temp
    uv run python -m eval.scripts.build_eval.fit_ceiling_weights --ship --force     # regenerate weights file + headline_metrics.json

--ship also writes the SAME fit into `headline_metrics.json`, the file every
production consumer reads, in that file's nested `overall` schema (Daniel
2026-09-21: the frozen read-v5 seven-flag instrument is now THE shipped tool;
the six-flag read-v6.1 fit it replaced is archived as
`headline_metrics_v61_sixflag_2026-09-21.json`). The shipped weights / boundary /
AUC / recall / CI / n / read-hash are the exact same numbers as the flat weights
file (gate: equal to 1e-9); the extra fields (weights_ci, in-sample point,
3-voice delta) are computed the same way graded_urn --ship computes them.

$0 -- refits cached reads, no API calls. All fitting reuses graded_urn
(fit_graded / oof_graded / score_graded / bootstrap, seven-flag) and fit_urn
(nested_threshold, recall_at_fpr, out_of_fold, auc); nothing is re-implemented.
The threshold is the nested operating point: the mean of the five per-fold
thresholds, each picked at FPR <= 2% on that fold's inner out-of-fold scores
(the same quantity graded_urn --ship writes as headline_metrics.threshold).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from eval.scripts.build_eval import fit_urn, graded_urn

RESULTS = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
POPULATION = Path("eval/data/populations/fc_gold_bal3000.parquet")
OUT = Path("eval/data/urn_runs/e1_ctx/weights_v5_ceiling_bal3000.json")
READ_HASH = "92404a300e14"
DATE = "2026-09-16"


def load_rows(results: Path, population: Path) -> list[dict]:
    graded_urn.FLAGS = graded_urn.FLAGS7            # seven-flag (5/4/3/X/I/2/1)
    pop = fit_urn.load_population(population)
    return fit_urn.load_headline(results, population=pop)


def build(rows: list[dict], results: Path, population: Path) -> dict:
    """The flat weights-file dict (weights + nested boundary + silence rule)."""
    n = len(rows)
    n_true = sum(r["y"] for r in rows)

    w7 = graded_urn.fit_graded(rows)
    auc_oof = fit_urn.auc(graded_urn.oof_graded(rows))

    y = np.array([r["y"] for r in rows])
    mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    C7 = np.array([graded_urn.flag_counts(r) for r in rows], dtype=float).T
    nest = fit_urn.nested_threshold(C7, y, mid, fold)
    thr = float(np.mean(nest["thresholds_by_fold"]))

    _, _, deff, _ = graded_urn.bootstrap(rows, fit_urn.BOOT_REPS, fit_urn.BOOT_SEED)

    # Silence rule: a claim whose reads carry no direction (all-I) is never
    # flagged (its score sits above the boundary), so overriding silent claims to
    # PASS is a no-op at the operating point; forcing them to pass DOES collapse
    # the ranking AUC, which is what the "keep the plain urn" decision measures.
    s = fit_urn.oof_np(C7, y, mid, fold)
    silent = np.array([sum(graded_urn.flag_counts(r)) == r["flags"].get("I", 0) for r in rows])
    s_pass = s.copy()
    s_pass[silent] = np.inf
    auc_silent = fit_urn.auc(list(zip(s_pass.tolist(), y.tolist())))

    return {
        "name": "read-v5 SEVEN-flag graded urn — bal3000 ceiling refit",
        "date": DATE,
        "slice": str(population),
        "reader": "read-v5 @ deepseek-ai/DeepSeek-V4-Flash (ceiling reads)",
        "reads": str(results),
        "read_hash": READ_HASH,
        "flags": list(graded_urn.FLAGS7),
        "weights": {k: w7[k] for k in graded_urn.FLAGS7},
        "threshold": thr,
        "threshold_rule": "nested: mean of the five per-fold thresholds at FPR<=2%",
        "thresholds_by_fold": nest["thresholds_by_fold"],
        "n": n,
        "n_true": n_true,
        "n_false": n - n_true,
        "auc_oof": auc_oof,
        "auc_ci95": deff["auc_ci95"],
        "recall_at_2pct_fpr": nest["recall"],
        "fpr": nest["fpr"],
        "pad_to": fit_urn.PAD_TO,
        "folds": fit_urn.K_FOLDS,
        "fold_key": "cluster_id",
        "smoothing": "+1 per flag / +7 on totals",
        "silence_rule": {
            "recall_at_2pct_fpr": nest["recall"],
            "auc_oof_silent_as_pass": auc_silent,
            "n_silent_true": int((silent & (y == 1)).sum()),
            "n_silent_false": int((silent & (y == 0)).sum()),
        },
    }


def ship_headline(rows: list[dict], results: Path, population: Path, wf: dict) -> Path:
    """Write the frozen fit into headline_metrics.json in its nested `overall`
    schema. The weights / boundary / AUC / recall / CI / n / read-hash are the
    SAME numbers as the flat weights file `wf` (build()); the extra fields are
    computed exactly as graded_urn --ship computes them. Provenance is the
    2026-09-21 frozen ship (Daniel), superseding the six-flag read-v6.1 file."""
    from eval.scripts.build_eval import evidence_urn_run as eur
    from eval.prompt_hash import prompt_hash

    FLAGS = graded_urn.FLAGS7
    w7 = wf["weights"]
    n, n_true = wf["n"], wf["n_true"]

    oof7 = graded_urn.oof_graded(rows)
    auc_insample = fit_urn.auc([(graded_urn.score_graded(r, w7), r["y"]) for r in rows])
    oof3 = fit_urn.out_of_fold(rows)
    auc3 = fit_urn.auc(oof3)

    ci, dauc_ci, deff, _ = graded_urn.bootstrap(rows, fit_urn.BOOT_REPS, fit_urn.BOOT_SEED)

    y = np.array([r["y"] for r in rows]); mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    C7 = np.array([graded_urn.flag_counts(r) for r in rows], dtype=float).T
    nest = fit_urn.nested_threshold(C7, y, mid, fold)
    thr = float(np.mean(nest["thresholds_by_fold"]))
    lr_plus = nest["recall"] / nest["fpr"] if nest["fpr"] else float("inf")
    rec_2, fpr_2, thr_2 = fit_urn.recall_at_fpr(oof7, 0.02)
    rec3_2, fpr3_2, _ = fit_urn.recall_at_fpr(oof3, 0.02)

    shipped = {
        "model": f"{len(FLAGS)}-flag graded (one fitted weight per reader flag; reader read-v5)",
        "shipped": "2026-09-21",
        "decided": "2026-09-21",
        "supersedes": "headline_metrics_v61_sixflag_2026-09-21.json",
        "baseline_3voice": str(fit_urn.E1_METRICS_CLUSTERED),
        "sibling": str(OUT),
        "input": str(results),
        "population": str(population),
        "population_manifest": "eval/data/populations/manifest.json",
        "slots": "returned",
        "pad_to": fit_urn.PAD_TO,
        "demote_blanket": False,
        "media_axis_excluded": "in the frozen population",
        "folds": fit_urn.K_FOLDS,
        "fold_key": "cluster_id",
        "smoothing": wf["smoothing"],
        "flags": list(FLAGS),
        "flag_desc": graded_urn.FLAG_DESC,
        "prompt_versions": {"query": eur.QUERY_PROMPT_V, "read": eur.READ_PROMPT_V,
                            "clean": eur.CLEAN_V, "prep": eur.PREP_V},
        "prompt_hashes": {"query": prompt_hash(eur.QUERY_SYS), "read": wf["read_hash"]},
        "prompt_hash_note": ("the E1 run predates prompt-hash stamping (Phase 1, 2026-09-08); "
                             "these hash the current text of the versions the run rows name"),
        "frozen_note": ("the FROZEN survey instrument (Daniel 2026-09-21): read-v5 @ "
                        "DeepSeek-V4-Flash, seven flags, query-v3 -> Serper top-10, scored "
                        "with weights_v5_ceiling_bal3000.json on fc_gold_bal3000. This file "
                        "mirrors that flat weights file; regenerate both with "
                        "`fit_ceiling_weights --ship --force`."),
        "bootstrap": {"reps": fit_urn.BOOT_REPS, "seed": fit_urn.BOOT_SEED, "design": "cluster_id"},
        "overall": {
            "n": n, "n_true": n_true, "n_false": n - n_true,
            "auc_oof": wf["auc_oof"], "auc_insample": auc_insample,
            "auc_ci95": wf["auc_ci95"], "auc_ci95_row": deff["auc_ci95_row"],
            "recall_at_2pct_fpr": wf["recall_at_2pct_fpr"], "fpr": wf["fpr"],
            "lr_plus": lr_plus,
            "threshold": wf["threshold"],
            "threshold_rule": ("nested: mean of the five per-fold thresholds, each picked at "
                               "FPR <= 2% on that fold's inner out-of-fold scores"),
            "thresholds_by_fold": nest["thresholds_by_fold"],
            "threshold_spread": nest["threshold_spread"],
            "insample": {"recall": rec_2, "fpr": fpr_2, "threshold": thr_2,
                         "lr_plus": rec_2 / fpr_2 if fpr_2 else float("inf"),
                         "baseline_recall": rec3_2, "baseline_fpr": fpr3_2},
            "weights": w7,
            "weights_ci": {k: list(ci[k]) for k in FLAGS},
            "delta_auc_oof_vs_3voice": wf["auc_oof"] - auc3,
            "delta_auc_ci95": list(dauc_ci),
        },
        "silence_rule": wf["silence_rule"],
    }
    p = fit_urn.E1_METRICS_PATH
    p.write_text(json.dumps(shipped, indent=2))
    print(f"\nSHIPPED {len(FLAGS)}-flag frozen constants -> {p}\n"
          f"  n {n}  AUC {wf['auc_oof']:.6f}  recall@2%FPR {wf['recall_at_2pct_fpr']:.6f} "
          f"(FPR {wf['fpr']:.6f})  threshold {wf['threshold']:+.6f}  read {wf['read_hash']}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--population", type=Path, default=POPULATION)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--ship", action="store_true",
                    help="also write the fit into headline_metrics.json (the production "
                         "headline file) in its nested `overall` schema")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting the pinned frozen weights file / headline file")
    args = ap.parse_args()

    if args.out.resolve() == OUT.resolve() and not args.force:
        sys.exit(f"refusing to overwrite the pinned frozen weights file {OUT}: pass --force to "
                 f"regenerate it in place (with --ship, also headline_metrics.json), or --out "
                 f"PATH to write elsewhere (the $0 reproduction gate).")

    rows = load_rows(args.results, args.population)
    out = build(rows, args.results, args.population)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}\n  n {out['n']} (T {out['n_true']} / F {out['n_false']})  "
          f"AUC {out['auc_oof']:.6f}  recall@2%FPR {out['recall_at_2pct_fpr']:.4f}  "
          f"threshold {out['threshold']:+.6f}")

    if args.ship:
        ship_headline(rows, args.results, args.population, out)


if __name__ == "__main__":
    main()
