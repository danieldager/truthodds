"""Graded Truth Odds: one fitted weight per read-v5 flag (5/4/3/X/I/2/1).

The committed 3-voice urn collapses the reader's seven flags into
support / refute / silent before fitting, which forces flag 2 ("points
against") to carry flag 1's full refutation weight and prices X ("on-claim
context") at the silent channel's value. This script fits the uncollapsed
model on the SAME population, folds and conventions as fit_urn (imported,
not duplicated) and reports what the collapse costs.

    uv run python -m eval.scripts.build_eval.graded_urn

$0 -- refits the saved E1 reads. Inherited from fit_urn: out-of-fold
discipline (5 folds by blake2b of cluster_id, cluster-disjoint), the eval
convention (mixed counts as FALSE at eval, excluded from the fit), media-axis
exclusion, N = documents actually returned, the cluster bootstrap (with the row
bootstrap beside it for the design effect) and the nested operating point
(2026-09-08, see fit_urn). Laplace smoothing is +1 per flag count and +7 on
totals (seven categories). Baseline numbers are READ from
headline_metrics_clustered.json, never hardcoded; the 3-voice oof AUC is
re-derived and asserted against it before anything is trusted.

--ship writes the fitted constants to headline_metrics.json, the file every
production consumer reads (Daniel 2026-09-14, closing the 2026-08-21 decision).
From that point headline_metrics.json is SEVEN-FLAG on the frozen population:
`overall.weights` is keyed by flag (5/4/3/X/I/2/1), not by voice (n_t/n_f/n_e),
and `overall.threshold` is the nested operating point (the mean of the five
per-fold thresholds, the same quantity refit_v7qa.rating_cuts calls cut_1).
The 3-voice fit stays reachable: headline_metrics_clustered.json on the same
frozen population, headline_metrics_3voice_2026-09-14.json as the shipped file
it replaced.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval import fit_urn
from eval.scripts.build_eval.e1_figures import roc, style, save, _auc_np

FLAGS7 = ("5", "4", "3", "X", "I", "2", "1")   # display order, support -> refute
FLAGS6 = ("5", "4", "X", "I", "2", "1")        # six-flag: contested/mixed (3) folded into X
FLAGS = FLAGS7                                  # active set; --six-flag switches it to FLAGS6
FLAG_DESC = {"5": "states / establishes the claim", "4": "points toward it",
             "3": "contested / mixed", "X": "on-claim context, no direction",
             "I": "irrelevant", "2": "points against it", "1": "contradicts / disproves"}
RESULTS = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
OUTDIR = Path("eval/data/urn_runs/e1_ctx")
FIGDIR = OUTDIR / "figures"
TAG = "_clustered"          # sibling outputs until the cluster design is pinned
BOOT_REPS, BOOT_SEED = fit_urn.BOOT_REPS, fit_urn.BOOT_SEED
NEGATION = re.compile(r"\b(not|never|no longer|fail(?:ed|s|ure)?|reject(?:ed|s)?|"
                      r"voted down|struck down|defeat(?:ed)?|block(?:ed)?|halt(?:ed)?|"
                      r"overturn(?:ed)?|den(?:ied|ies)|off the table)\b", re.I)


def flag_counts(row: dict) -> list[int]:
    """Per-flag document counts in the active FLAGS order. In six-flag mode the
    contested/mixed count (3) is folded into on-claim context (X)."""
    f = row["flags"]
    if FLAGS is FLAGS6:
        return [f.get("X", 0) + f.get("3", 0) if k == "X" else f.get(k, 0) for k in FLAGS]
    return [f.get(k, 0) for k in FLAGS]


def fit_graded(rows: list[dict]) -> dict[str, float]:
    """Per-flag log-LR, Laplace-smoothed (+1 per flag / +K on the totals, K =
    len(FLAGS)). Mixed claims never enter."""
    rows = [r for r in rows if not r.get("mid")]
    K = len(FLAGS)
    t = collections.Counter(); f = collections.Counter()
    for r in rows:
        counts = dict(zip(FLAGS, flag_counts(r)))
        (t if r["y"] == 1 else f).update(counts)
    tot_t, tot_f = sum(t.values()), sum(f.values())
    return {k: math.log(((t[k] + 1) / (tot_t + K)) / ((f[k] + 1) / (tot_f + K)))
            for k in FLAGS}


def score_graded(row: dict, w: dict[str, float]) -> float:
    return sum(w[k] * n for k, n in zip(FLAGS, flag_counts(row)))


def oof_graded(rows: list[dict]) -> list[tuple[float, int]]:
    scored = []
    for fold in range(fit_urn.K_FOLDS):
        train = [r for r in rows if r["fold"] != fold]
        test = [r for r in rows if r["fold"] == fold]
        w = fit_graded(train)
        scored += [(score_graded(r, w), r["y"]) for r in test]
    return scored


def bootstrap(rows: list[dict], reps: int, seed: int) -> tuple[dict, tuple, dict, tuple]:
    """Cluster bootstrap (resample cluster_id; the headline) beside the old
    class-stratified row bootstrap, same seed: per-flag weight CIs + paired oof
    dAUC (7-flag - 3-voice). Returns (weight CIs, dAUC CI, design effects
    var cluster / var row, row-design dAUC CI).

    Weights refit inside every replicate. dAUC replicates the full oof protocol
    (per-fold refit, mixed excluded from fits, scored as FALSE at eval) so the
    interval speaks about the number the report headlines.
    """
    C7 = np.array([flag_counts(r) for r in rows], dtype=float).T          # 7 x n
    C3 = np.array([[r["n_t"], r["n_f"], r["n_e"]] for r in rows], float).T  # 3 x n
    y = np.array([r["y"] for r in rows]); mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows]); cl = fit_urn.cluster_index(rows)

    res = {}
    for design in ("cluster", "row"):
        rng = np.random.default_rng(seed)
        w_reps, deltas, a7s = [], [], []
        for _ in range(reps):
            idx = fit_urn.boot_idx(rng, y, cl, design == "cluster")
            w_reps.append(fit_urn.fit_np(C7, y, mid, idx))
            s7 = fit_urn.oof_np(C7, y, mid, fold, idx)
            s3 = fit_urn.oof_np(C3, y, mid, fold, idx)
            yy = y[idx]
            a7 = _auc_np(s7[yy == 1], s7[yy == 0])
            a7s.append(a7)
            deltas.append(a7 - _auc_np(s3[yy == 1], s3[yy == 0]))
        res[design] = (np.array(w_reps), np.array(deltas), np.array(a7s))
    W, D, A = res["cluster"]; Wr, Dr, Ar = res["row"]
    ci = {k: tuple(np.percentile(W[:, i], [2.5, 97.5])) for i, k in enumerate(FLAGS)}
    deff = {"dauc": float(D.var() / Dr.var()),
            "auc": float(A.var() / Ar.var()),
            "auc_ci95": [float(x) for x in np.percentile(A, [2.5, 97.5])],
            "auc_ci95_row": [float(x) for x in np.percentile(Ar, [2.5, 97.5])],
            "weights": {k: float(W[:, i].var() / Wr[:, i].var()) for i, k in enumerate(FLAGS)}}
    return ci, tuple(np.percentile(D, [2.5, 97.5])), deff, tuple(np.percentile(Dr, [2.5, 97.5]))


def run_prompt_versions(path: Path) -> dict:
    """The prompt versions stamped on the run's rows, asserted uniform."""
    seen = {json.dumps(json.loads(l).get("prompts"), sort_keys=True) for l in path.open()}
    assert len(seen) == 1, f"{path} carries {len(seen)} different prompt sets"
    return json.loads(seen.pop())


def ship(out: dict, rows: list[dict], population: Path | None) -> Path:
    """Write the fitted 7-flag constants to headline_metrics.json.

    Same numbers as graded_metrics{TAG}.json, re-keyed into the shape the
    consumers read: a top-level provenance block and one `overall` cell with
    weights, a single deployable threshold, n, AUC and recall.

    The threshold is the NESTED operating point, defined as the mean of the five
    per-fold thresholds (fit_urn.nested_threshold picks each on the four training
    folds' inner out-of-fold scores). That is the definition refit_v7qa.rating_cuts
    uses for cut_1, so the shipped flag boundary and the survey's rating-1 cut are
    the same quantity.
    """
    from eval.scripts.build_eval import evidence_urn_run as eur

    nest = out["recall_at_2pct_fpr"]
    thr = float(np.mean(nest["thresholds_by_fold"]))
    n_true = out["n_true"]
    six = FLAGS is FLAGS6
    pv = run_prompt_versions(Path(out["input"])) or {
        "query": eur.QUERY_PROMPT_V, "read": eur.READ_PROMPT_V,
        "clean": eur.CLEAN_V, "prep": eur.PREP_V}
    shipped = {
        "model": f"{len(FLAGS)}-flag graded (one fitted weight per reader flag; "
                 f"reader {pv.get('read')})",
        "shipped": "2026-09-14",
        "decided": "2026-08-21" if not six else "2026-09-14",
        "supersedes": ("headline_metrics_7flag_2026-09-14.json" if six
                       else "headline_metrics_3voice_2026-09-14.json"),
        "baseline_3voice": str(fit_urn.E1_METRICS_CLUSTERED),
        "sibling": str(OUTDIR / f"graded_metrics{TAG}.json"),
        "input": out["input"],
        "population": out["population"],
        "population_manifest": "eval/data/populations/manifest.json",
        "slots": "returned",
        "pad_to": fit_urn.PAD_TO,
        "demote_blanket": False,
        "media_axis_excluded": None if out["keep_media_axis"] else "in the frozen population",
        "folds": out["folds"],
        "fold_key": "cluster_id",
        "smoothing": out["smoothing"],
        "flags": list(FLAGS),
        "flag_desc": FLAG_DESC,
        "prompt_versions": pv,
        "prompt_hashes": dict(eur.PROMPT_HASHES),
        "prompt_hash_note": ("the E1 run predates prompt-hash stamping (Phase 1, "
                             "2026-09-08); these hash the current text of the "
                             "versions the run rows name"),
        "bootstrap": out["bootstrap"],
        "overall": {
            "n": out["n"], "n_true": n_true, "n_false": out["n"] - n_true,
            "auc_oof": out["auc_oof"], "auc_insample": out["auc_insample"],
            "auc_ci95": out["auc_ci95"], "auc_ci95_row": out["auc_ci95_row"],
            "recall_at_2pct_fpr": nest["recall"], "fpr": nest["fpr"],
            "lr_plus": nest["lr_plus"],
            "threshold": thr,
            "threshold_rule": ("nested: mean of the five per-fold thresholds, each "
                               "picked at FPR <= 2% on that fold's inner out-of-fold "
                               "scores"),
            "thresholds_by_fold": nest["thresholds_by_fold"],
            "threshold_spread": nest["threshold_spread"],
            "insample": nest["insample"],
            "weights": out["weights"],
            "weights_ci": out["weights_ci"],
            "delta_auc_oof_vs_3voice": out["delta_auc_oof_vs_3voice"],
            "delta_auc_ci95": out["delta_auc_ci95"],
        },
    }
    p = fit_urn.E1_METRICS_PATH
    p.write_text(json.dumps(shipped, indent=2))
    print(f"\nSHIPPED {len(FLAGS)}-flag constants -> {p}\n"
          f"  n {out['n']}  AUC {out['auc_oof']:.4f}  recall@2%FPR {nest['recall']:.4f} "
          f"(FPR {nest['fpr']:.4f})  threshold {thr:+.4f}")
    return p


def main() -> None:
    global TAG, FLAGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=Path, default=None,
                    help="parquet with a claim_id column: restrict to those claims")
    ap.add_argument("--results", type=Path, default=RESULTS,
                    help="run jsonl to fit on (default: the cached E1 read-v5 reads). "
                         "A substituted read-v6 / read-v5-rerun file goes here.")
    ap.add_argument("--six-flag", action="store_true",
                    help="fit six flags (5/4/X/I/2/1), folding contested/mixed (3) into X")
    ap.add_argument("--keep-media-axis", action="store_true",
                    help="keep media-provenance claims (fit_urn --keep-media-axis)")
    ap.add_argument("--metrics", type=Path, default=fit_urn.E1_METRICS_CLUSTERED,
                    help="3-voice sibling JSON the trust gate reproduces")
    ap.add_argument("--tag", default=TAG, help="suffix for the output JSON / figures")
    ap.add_argument("--ship", action="store_true",
                    help="also write the fitted constants to headline_metrics.json, "
                         "the file the production consumers read")
    args = ap.parse_args()
    TAG = args.tag
    if args.six_flag:
        FLAGS = FLAGS6
    default_reads = args.results.resolve() == RESULTS.resolve()
    pop = fit_urn.load_population(args.population)
    rows = (fit_urn.load(args.results, None, False, fit_urn.load_judged_axis(), pop)
            if args.keep_media_axis
            else fit_urn.load_headline(args.results, population=pop))
    hm = json.loads(args.metrics.read_text())["overall"]

    # Trust gate: on the cached E1 reads this loader must reproduce the shipped
    # 3-voice headline exactly. On a substituted read set (v6 / a v5 rerun) the
    # 3-voice AUC legitimately differs, so the gate only prints.
    oof3 = fit_urn.out_of_fold(rows)
    auc3 = fit_urn.auc(oof3)
    if default_reads:
        assert abs(auc3 - hm["auc_oof"]) < 1e-9, f"3-voice oof AUC {auc3} != headline {hm['auc_oof']}"
        print(f"3-voice reproduction OK: oof AUC {auc3:.6f} == headline")
    else:
        print(f"3-voice oof AUC on {args.results}: {auc3:.6f} (headline {hm['auc_oof']:.6f}, "
              f"different read set -- not asserted)")

    n_true = sum(r["y"] for r in rows)
    print(f"claims {len(rows)} (T {n_true} / F {len(rows) - n_true}), "
          f"docs {sum(sum(flag_counts(r)) for r in rows)}")

    oof7 = oof_graded(rows)
    w7 = fit_graded(rows)
    auc7_oof = fit_urn.auc(oof7)
    auc7_ins = fit_urn.auc([(score_graded(r, w7), r["y"]) for r in rows])

    # in-sample threshold choice (the pre-2026-09-08 headline), kept for comparison
    budget_matched = hm["insample"]["fpr"]   # the 3-voice achieved in-sample operating point
    rec_m, fpr_m, thr_m = fit_urn.recall_at_fpr(oof7, budget_matched)
    rec_2, fpr_2, thr_2 = fit_urn.recall_at_fpr(oof7, 0.02)
    rec3_2, fpr3_2, _ = fit_urn.recall_at_fpr(oof3, 0.02)
    # nested threshold choice: the headline
    C7 = np.array([flag_counts(r) for r in rows], dtype=float).T
    nest = fit_urn.nested_threshold(C7, np.array([r["y"] for r in rows]),
                                    np.array([r["mid"] for r in rows]),
                                    np.array([r["fold"] for r in rows]))

    print(f"\n7-flag  oof AUC {auc7_oof:.4f} / in-sample {auc7_ins:.4f}   "
          f"(3-voice {auc3:.4f} / {hm['auc_insample']:.4f})")
    print(f"recall@FPR<={budget_matched:.4f} (matched, in-sample thr): {rec_m:.4f} (FPR {fpr_m:.4f})   "
          f"3-voice {hm['insample']['recall_at_2pct_fpr']:.4f}")
    print(f"recall@FPR<=2% in-sample thr: {rec_2:.4f} (FPR {fpr_2:.4f}, thr {thr_2:.3f})   "
          f"3-voice {rec3_2:.4f} (FPR {fpr3_2:.4f})")
    print(f"recall@FPR<=2% NESTED: {nest['recall']:.4f} (FPR {nest['fpr']:.4f}, "
          f"thr by fold {[round(t, 3) for t in nest['thresholds_by_fold']]})   "
          f"3-voice {hm['recall_at_2pct_fpr']:.4f} (FPR {hm['fpr']:.4f})")
    lr7 = nest["recall"] / nest["fpr"] if nest["fpr"] else float("inf")

    print(f"\nbootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}, cluster + row) ...")
    w_ci, dauc_ci, deff, dauc_ci_row = bootstrap(rows, BOOT_REPS, BOOT_SEED)
    for k in FLAGS:
        print(f"  {k}: {w7[k]:+.3f}  [{w_ci[k][0]:+.3f}, {w_ci[k][1]:+.3f}]  "
              f"deff {deff['weights'][k]:.2f}  {FLAG_DESC[k]}")
    print(f"  dAUC oof (7-flag - 3-voice): {auc7_oof - auc3:+.4f}  "
          f"[{dauc_ci[0]:+.4f}, {dauc_ci[1]:+.4f}] cluster / "
          f"[{dauc_ci_row[0]:+.4f}, {dauc_ci_row[1]:+.4f}] row   design effect {deff['dauc']:.2f}")

    # Calibration: oof score deciles -> empirical share gold-FALSE, vs the
    # posterior the score implies under the corpus prior.
    pi = n_true / len(rows)
    prior = math.log(pi / (1 - pi))
    svals = np.array([s for s, _ in oof7]); yvals = np.array([yy for _, yy in oof7])
    edges = np.quantile(svals, np.linspace(0, 1, 11))
    calib = []
    for i in range(10):
        hi_ok = (svals <= edges[i + 1]) if i == 9 else (svals < edges[i + 1])
        m = (svals >= edges[i]) & hi_ok
        if not m.any():
            continue
        # P(F|s) = 1 / (1 + prior_odds * e^s) = 1 / (1 + e^(log-prior-odds + s))
        implied = float(np.mean(1 / (1 + np.exp(prior + svals[m]))))
        calib.append({"decile": i + 1, "n": int(m.sum()),
                      "score_lo": float(edges[i]), "score_hi": float(edges[i + 1]),
                      "mean_score": float(svals[m].mean()),
                      "implied_p_false": implied,
                      "empirical_p_false": float((yvals[m] == 0).mean())})

    # Residuals at the 2% operating point (feeds B4).
    scored = sorted(((score_graded(r, w7), r) for r in rows), key=lambda x: x[0])
    fps = [(s, r) for s, r in scored if r["y"] == 1 and s <= thr_2]
    fns = [(s, r) for s, r in reversed(scored) if r["y"] == 0]
    neg_share_fp = sum(bool(NEGATION.search(r["claim"])) for _, r in fps) / max(1, len(fps))
    neg_share_all = sum(bool(NEGATION.search(r["claim"])) for r in rows) / len(rows)
    top = lambda lst: [{"score": round(s, 2), "claim": r["claim"][:140],
                        "flags": r["flags"], "url": r["review_url"]} for s, r in lst[:10]]
    print(f"\nflagged gold-TRUE at 2% budget: {len(fps)}; negation-pattern share "
          f"{neg_share_fp:.2f} vs {neg_share_all:.2f} corpus-wide")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    figs = []

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for pairs, a, label, c, lw in ((oof7, auc7_oof, "7-flag graded", "0.10", 1.8),
                                   (oof3, auc3, "3-voice (shipped)", "0.55", 1.4)):
        x, yv = roc(pairs)
        ax.plot(x, yv, "-", color=c, linewidth=lw, label=f"{label}  AUC {a:.3f}")
    ax.plot([0, 1], [0, 1], "-", color="0.85", linewidth=1, zorder=0)
    ax.set_xlabel("FPR (gold-TRUE flagged)"); ax.set_ylabel("recall (gold-FALSE flagged)")
    ax.set_title("Out-of-fold ROC — graded vs collapsed", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right"); style(ax)
    figs.append(save(fig, FIGDIR, f"graded_roc{TAG}.png"))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ys = np.arange(len(FLAGS))[::-1]
    for yp, k in zip(ys, FLAGS):
        lo, hi = w_ci[k]
        ax.plot([lo, hi], [yp, yp], "-", color="0.35", linewidth=1.4)
        ax.plot([w7[k]], [yp], "o", color="0.15", markersize=7,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax.annotate(f"{w7[k]:+.2f}", (w7[k], yp + 0.28), fontsize=8, color="0.25",
                    ha="center")
    ax.axvline(0, color="0.80", linewidth=1)
    ax.set_yticks(ys, [f"{k}  {FLAG_DESC[k]}" for k in FLAGS], fontsize=8.5)
    ax.set_xlabel("fitted weight (log likelihood-ratio per document)")
    ax.set_title("Per-flag weights, 95% bootstrap CI", fontsize=10); style(ax)
    figs.append(save(fig, FIGDIR, f"graded_weights{TAG}.png"))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bins = np.linspace(min(svals), max(svals), 45)
    ax.hist([svals[yvals == 1], svals[yvals == 0]], bins=bins, density=True,
            label=[f"gold TRUE (n={int((yvals==1).sum()):,})",
                   f"gold FALSE (n={int((yvals==0).sum()):,})"],
            color=["#8fb0c2", "#a05c46"], edgecolor="white", linewidth=0.4,
            histtype="bar")
    ax.axvline(thr_2, color="0.10", linewidth=1.2, linestyle="--")
    ax.text(thr_2 - 0.4, ax.get_ylim()[1] * 0.92, f"flag ≤ {thr_2:.2f}",
            fontsize=8, color="0.25", ha="right")
    ax.set_xlabel("graded score (nats)"); ax.set_ylabel("density")
    ax.set_title("Out-of-fold score distribution by gold label", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5); style(ax)
    figs.append(save(fig, FIGDIR, f"graded_scores{TAG}.png"))

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], "-", color="0.85", linewidth=1, zorder=0)
    ax.plot([c["implied_p_false"] for c in calib],
            [c["empirical_p_false"] for c in calib],
            "o-", color="0.15", linewidth=1.4, markersize=6,
            markeredgecolor="white", markeredgewidth=1.0)
    ax.set_xlabel("implied P(false) under corpus prior"); ax.set_ylabel("empirical share gold-FALSE")
    ax.set_title("Reliability by score decile", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); style(ax)
    figs.append(save(fig, FIGDIR, f"graded_calibration{TAG}.png"))

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    budgets = np.linspace(0.0, 0.05, 101)
    for pairs, label, c in ((oof7, "7-flag graded", "0.10"), (oof3, "3-voice (shipped)", "0.55")):
        x, yv = roc(pairs)
        recs = [yv[x <= b].max() if (x <= b).any() else 0.0 for b in budgets]
        ax.plot(budgets * 100, recs, "-", color=c, linewidth=1.6, label=label)
    ax.axvline(2.0, color="0.55", linewidth=1, linestyle=":")
    ax.text(2.05, 0.02, "2% FPR budget", fontsize=7.5, color="0.45", rotation=90)
    ax.set_xlabel("FPR budget (%)"); ax.set_ylabel("recall of gold-FALSE")
    ax.set_title("Recall vs false-alarm budget", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right"); style(ax)
    figs.append(save(fig, FIGDIR, f"graded_recall_budget{TAG}.png"))

    out = {
        "input": str(args.results), "folds": fit_urn.K_FOLDS,
        "smoothing": f"+1 count / +{len(FLAGS)} total",
        "n": len(rows), "n_true": n_true,
        "auc_oof": auc7_oof, "auc_insample": auc7_ins,
        "delta_auc_oof_vs_3voice": auc7_oof - auc3,
        "delta_auc_ci95": list(dauc_ci),
        "delta_auc_ci95_row": list(dauc_ci_row),
        "auc_ci95": deff["auc_ci95"], "auc_ci95_row": deff["auc_ci95_row"],
        "design_effect": deff,
        "recall_at_matched_fpr": {"budget": budget_matched, "recall": rec_m, "fpr": fpr_m,
                                  "threshold": thr_m,
                                  "baseline_recall": hm["insample"]["recall_at_2pct_fpr"],
                                  "note": "in-sample threshold choice on both sides"},
        "recall_at_2pct_fpr": {"recall": nest["recall"], "fpr": nest["fpr"],
                               "thresholds_by_fold": nest["thresholds_by_fold"],
                               "threshold_spread": nest["threshold_spread"],
                               "lr_plus": lr7, "baseline_recall": hm["recall_at_2pct_fpr"],
                               "baseline_fpr": hm["fpr"], "baseline_lr_plus": hm["lr_plus"],
                               "insample": {"recall": rec_2, "fpr": fpr_2, "threshold": thr_2,
                                            "lr_plus": rec_2 / fpr_2 if fpr_2 else float("inf"),
                                            "baseline_recall": rec3_2, "baseline_fpr": fpr3_2}},
        "weights": {k: w7[k] for k in FLAGS},
        "weights_ci": {k: list(w_ci[k]) for k in FLAGS},
        "flag_desc": FLAG_DESC,
        "baseline": {"auc_oof": auc3, "auc_insample": hm["auc_insample"],
                     "thresholds_by_fold": hm["thresholds_by_fold"], "weights": hm["weights"],
                     "weights_ci": hm["weights_ci"]},
        "calibration": calib,
        "residuals": {"flagged_gold_true_at_2pct": len(fps),
                      "negation_share_flagged_true": neg_share_fp,
                      "negation_share_corpus": neg_share_all,
                      "worst_false_positives": top(fps if len(fps) else []),
                      "worst_false_negatives": top(fns)},
        "bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED, "design": "cluster_id"},
        "population": str(args.population) if args.population else None,
        "keep_media_axis": args.keep_media_axis,
    }
    (OUTDIR / f"graded_metrics{TAG}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUTDIR / f'graded_metrics{TAG}.json'}\n{len(figs)} figures -> {FIGDIR}")

    if args.ship:
        ship(out, rows, args.population)


if __name__ == "__main__":
    main()
