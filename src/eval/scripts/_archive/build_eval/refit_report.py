"""Collect every frozen-population refit into one JSON, and draw the
gold-fit vs two-urn-fit weight replication figure with intervals.

Reads only the sibling `_clustered` outputs the fitting scripts wrote; it never
refits. Run after fit_urn / graded_urn / model_ladder / fit_two_urn.

    uv run python -m eval.scripts.build_eval.refit_report
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POP = Path("eval/data/populations")
E1 = Path("eval/data/urn_runs/e1_ctx")
TL = Path("eval/data/urn_runs/true_timeline")
FIG = POP / "figures"
EPS_HEADLINE = 0.1
FLAGS = ("5", "4", "3", "X", "I", "2", "1")     # display order
VOICES = ("n_t", "n_f", "n_e")
VOICE_LABEL = {"n_t": "supports", "n_f": "refutes", "n_e": "silent"}


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _cell(**kw) -> dict:
    return kw


def collect() -> dict:
    out = {"populations_manifest": str(POP / "manifest.json"),
           "eps_headline": EPS_HEADLINE,
           # Every claim is scored over ten slots; slots with no read flag are
           # silent documents (fit_urn.PAD_TO, Daniel 2026-09-08).
           "pad_to": 10,
           "bootstrap": {"reps": 2000, "seed": 707,
                         "cluster_design": "cluster_id (fc-gold) / post_id within urn (two-urn)"},
           "cells": []}

    # ---- fit_urn, 3-voice, refit on gold labels ----------------------------
    for pop, f in (("fc_gold", E1 / "headline_metrics_clustered.json"),):
        d = _load(f)["overall"]
        out["cells"].append(_cell(
            variant="fit_urn_3voice", fit="refit on fc-gold labels",
            population=pop, population_file=str(POP / f"{pop}.parquet"),
            source_json=str(f), n=d["n"], n_true=d["n_true"],
            n_false=d["n"] - d["n_true"], n_clusters=d["n_clusters"],
            auc={"point": d["auc_oof"], "ci95_cluster": d["auc_ci95"],
                 "ci95_row": d["auc_ci95_row"], "design_effect": d["design_effect_auc"],
                 "insample": d["auc_insample"]},
            nested_recall_at_2pct_fpr={"recall": d["recall_at_2pct_fpr"], "fpr": d["fpr"],
                                       "thresholds_by_fold": d["thresholds_by_fold"],
                                       "threshold_spread": d["threshold_spread"]},
            insample_recall_at_2pct_fpr=d["insample"],
            weights=d["weights"], weights_ci=d["weights_ci"],
            weights_ci_row=d["weights_ci_row"]))

    # ---- graded_urn, 7-flag, refit on gold labels --------------------------
    for pop, f in (("fc_gold", E1 / "graded_metrics_clustered.json"),):
        d = _load(f)
        r = d["recall_at_2pct_fpr"]
        out["cells"].append(_cell(
            variant="graded_urn_7flag", fit="refit on fc-gold labels",
            population=pop, population_file=str(POP / f"{pop}.parquet"),
            source_json=str(f), n=d["n"], n_true=d["n_true"],
            n_false=d["n"] - d["n_true"],
            auc={"point": d["auc_oof"], "ci95_cluster": d["auc_ci95"],
                 "ci95_row": d["auc_ci95_row"], "insample": d["auc_insample"],
                 "delta_vs_3voice": d["delta_auc_oof_vs_3voice"],
                 "delta_ci95_cluster": d["delta_auc_ci95"]},
            nested_recall_at_2pct_fpr={"recall": r["recall"], "fpr": r["fpr"],
                                       "thresholds_by_fold": r["thresholds_by_fold"],
                                       "threshold_spread": r["threshold_spread"]},
            insample_recall_at_2pct_fpr=r["insample"],
            weights=d["weights"], weights_ci=d["weights_ci"]))

    # ---- model_ladder, three nested models ---------------------------------
    for pop, f in (("fc_gold", E1 / "model_ladder/ladder_clustered.json"),):
        d = _load(f)
        p = d["population"]
        for m, res in d["models"].items():
            out["cells"].append(_cell(
                variant=f"model_ladder_{m}", fit="refit on fc-gold labels",
                population=pop, population_file=str(POP / f"{pop}.parquet"),
                source_json=str(f), n=p["n"], n_true=p["n_true"], n_false=p["n_false"],
                n_set_aside_mixed=p["set_aside_n"],
                auc={"point": res["auc_oof"], "ci95_cluster": res["auc_ci"],
                     "ci95_row": res["auc_ci_row"], "design_effect": res["design_effect_auc"]},
                nested_recall_at_2pct_fpr={"recall": res["recall_2pct"], "fpr": res["fpr_2pct"],
                                           "thresholds_by_fold": res["thresholds_by_fold"],
                                           "threshold_spread": res["threshold_spread"]},
                insample_recall_at_2pct_fpr=res["insample"],
                weights=res["weights"], weights_ci=res["weights_ci"]))

    # ---- fit_two_urn, label-free, transferred to fc-gold -------------------
    for pop, f in (("fc_gold", TL / "two_urn_fit_clustered.json"),):
        d = _load(f)
        for key, variant in (("fits", "two_urn_3voice"), ("fits7", "two_urn_7flag")):
            fit = next(x for x in d[key] if abs(x["eps"] - EPS_HEADLINE) < 1e-9)
            ev = fit["fc_gold_transfer"]
            out["cells"].append(_cell(
                variant=variant,
                fit=f"label-free two-urn de-mix, eps={EPS_HEADLINE}; transfer, no refit",
                population=f"cn_false + x_feed -> {pop}",
                population_file=[str(POP / "cn_false.parquet"),
                                 str(POP / "x_feed.parquet"),
                                 str(POP / f"{pop}.parquet")],
                source_json=str(f),
                n=d["n_gold"], n_fit_false=d["n_false"], n_fit_true=d["n_true"],
                n_posts_false=d["bootstrap"]["n_posts_false"],
                n_posts_true=d["bootstrap"]["n_posts_true"],
                auc={"point": ev["auc"], "ci95_cluster": None, "ci95_row": None,
                     "note": "transfer AUC has no bootstrap in fit_two_urn; "
                             "the urn bootstrap covers the weights only"},
                nested_recall_at_2pct_fpr={"recall": ev["recall_at_2pct_fpr"], "fpr": ev["fpr"],
                                           "thresholds_by_fold": ev["thresholds_by_fold"],
                                           "threshold_spread": ev["threshold_spread"]},
                insample_recall_at_2pct_fpr=ev["insample"],
                weights=fit["weights"], weights_ci=fit["weights_ci"],
                design_effect_weights=fit["design_effect_weights"]))
    return out


# ---- replication figure ----------------------------------------------------
def _panel(keys, labels, gold_w, gold_ci, urn_w, urn_ci, title, stem):
    x = list(range(len(keys)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for dx, (w, ci, lab, mk, fc) in enumerate((
            (gold_w, gold_ci, "fitted on fc-gold labels", "o", "white"),
            (urn_w, urn_ci, f"two-urn, label-free (eps={EPS_HEADLINE})", "o", "black"))):
        xs = [i + (dx - 0.5) * 0.16 for i in x]
        lo = [w[k] - ci[k][0] for k in keys]
        hi = [ci[k][1] - w[k] for k in keys]
        ax.errorbar(xs, [w[k] for k in keys], yerr=[lo, hi], fmt=mk, ms=6,
                    mfc=fc, mec="black", color="black", capsize=3, lw=1.1,
                    ls="none", label=lab)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("read flag" if len(keys) == 7 else "voice")
    ax.set_ylabel("log rate ratio per document")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG / stem}.png / .svg")


def figures() -> None:
    g7 = _load(E1 / "graded_metrics_clustered.json")
    g3 = _load(E1 / "headline_metrics_clustered.json")["overall"]
    tu = _load(TL / "two_urn_fit_clustered.json")
    u7 = next(x for x in tu["fits7"] if abs(x["eps"] - EPS_HEADLINE) < 1e-9)
    u3 = next(x for x in tu["fits"] if abs(x["eps"] - EPS_HEADLINE) < 1e-9)

    for keys, labels, gw, gci, uw, uci, title, stem in (
        (FLAGS, list(FLAGS), g7["weights"], g7["weights_ci"], u7["weights"], u7["weights_ci"],
         "Per-flag weights: gold-label fit vs label-free two-urn fit", "weights_replicate_ci"),
        (VOICES, [VOICE_LABEL[v] for v in VOICES], g3["weights"], g3["weights_ci"],
         u3["weights"], u3["weights_ci"],
         "Per-voice weights: gold-label fit vs label-free two-urn fit",
         "weights_replicate_ci_3voice"),
    ):
        _panel(keys, labels, gw, gci, uw, uci, title, stem)
        payload = {"stem": stem, "eps_headline": EPS_HEADLINE,
                   "order": list(keys),
                   "gold_fit": {"source": str(E1 / ("graded_metrics_clustered.json"
                                                    if len(keys) == 7
                                                    else "headline_metrics_clustered.json")),
                                "population": "fc_gold",
                                "weights": {k: gw[k] for k in keys},
                                "weights_ci95_cluster": {k: gci[k] for k in keys}},
                   "two_urn_fit": {"source": str(TL / "two_urn_fit_clustered.json"),
                                   "population": "cn_false + x_feed",
                                   "weights": {k: uw[k] for k in keys},
                                   "weights_ci95_cluster": {k: uci[k] for k in keys}}}
        (FIG / f"{stem}.json").write_text(json.dumps(payload, indent=2))
        print(f"wrote {FIG / stem}.json")


def main() -> None:
    res = collect()
    (POP / "refit_results.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {POP / 'refit_results.json'} ({len(res['cells'])} cells)")
    figures()


if __name__ == "__main__":
    main()
