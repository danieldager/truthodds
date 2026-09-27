"""Derive every number and figure for the AVeriTeC results artifact from run files.

Nothing on the page is hand-copied. This script reads compare_averitec's comparison.json,
bar_counterfactual's counterfactual.json, the per-arm scored parquets, the urn per-claim
scores and the gold parquet, then emits stats.json plus monochrome figures;
averitec_artifact_html.py renders the page from those.

  uv run python -m claimverify.averitec_artifact_stats --arms top3,all10

A requested arm that is not in comparison.json is skipped and recorded in
stats.json under arms_missing, so the page can be built before every arm has run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from claimverify.config import SRC  # noqa: E402
from claimverify.score_averitec import BINARY, DEFAULT_GOLD  # noqa: E402

RUN = SRC / "eval/data/claimverify_runs/averitec_dev"
DEFAULT_COMPARISON = RUN / "comparison/comparison.json"
DEFAULT_COUNTERFACTUAL = RUN / "comparison/counterfactual.json"
DEFAULT_URN = SRC / "eval/data/urn_runs/averitec_dev/scores.scored.parquet"
DEFAULT_OUT = RUN / "artifact"

URN_MODELS = {"3-voice": "s3", "7-flag": "s7"}
ARM_LABEL = {"noceil": "loop, no ceiling, their condition",
             "top3_noceil": "loop, no ceiling, their condition",
             "top3": "loop, top 3 read", "all10": "loop, all 10 read"}
ARM_SHORT = {"noceil": "no ceiling", "top3_noceil": "no ceiling",
             "top3": "top 3", "all10": "all 10"}
# figure 1 order: the paper first, then ceiling-off, then the two ceiling-on arms
ARM_ORDER = ["noceil", "top3_noceil", "top3", "all10"]

INK, MUT, LIGHT = "#1a1a1a", "#666666", "#bbbbbb"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                     "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "svg.fonttype": "none"})


def arm_label(name: str) -> str:
    return ARM_LABEL.get(name, f"loop, {name}")


def roc(scores: np.ndarray, gold_flag: np.ndarray) -> dict:
    """FPR / flag-recall curve for flag = score <= threshold, positives = gold flag."""
    flag, pas = scores[gold_flag], scores[~gold_flag]
    thrs = np.concatenate(([-np.inf], np.unique(scores), [np.inf]))
    return {"fpr": [float((pas <= t).mean()) for t in thrs],
            "recall": [float((flag <= t).mean()) for t in thrs]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="top3,all10",
                    help="comma separated arm names as passed to compare_averitec")
    ap.add_argument("--comparison", default=str(DEFAULT_COMPARISON))
    ap.add_argument("--counterfactual", default=str(DEFAULT_COUNTERFACTUAL))
    ap.add_argument("--urn", default=str(DEFAULT_URN))
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    C = json.loads(Path(a.comparison).read_text())
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    wanted = [n for n in a.arms.split(",") if n.strip()]
    present = [n for n in wanted if n in C["arms"]]
    missing = [n for n in wanted if n not in C["arms"]]
    ordered = ([n for n in ARM_ORDER if n in present]
               + [n for n in present if n not in ARM_ORDER])
    if not ordered:
        raise SystemExit(f"no requested arm found in {a.comparison}: {wanted}")

    def rel(p) -> str:
        p = Path(p)
        return str(p.relative_to(SRC)) if p.is_absolute() and p.is_relative_to(SRC) else str(p)

    S: dict = {"generated_from": rel(a.comparison),
               "comparison_generated_utc": C["generated_utc"],
               "arms_present": ordered, "arms_missing": missing,
               "arm_labels": {n: arm_label(n) for n in ordered},
               "arm_short": {n: ARM_SHORT.get(n, n) for n in ordered}}

    # ---------- dataset, models, conventions ----------
    gold = pd.read_parquet(a.gold).copy()
    gold["claim_id"] = gold["claim_id"].astype(str)
    mans = {n: json.loads((SRC / C["arms"][n]["dir"] / "manifest.json").read_text())
            for n in ordered}
    # prose about the ceiling and the loop's shape must come from a ceiling-on arm
    first = next((n for n in ordered if mans[n]["config"]["date_ceiling"]), ordered[0])
    man = mans[first]
    urn_man_path = Path(C["urn"]["path"]).parent / "manifest.json"
    urn_man = json.loads((urn_man_path if urn_man_path.is_absolute()
                          else SRC / urn_man_path).read_text())
    ladder_path = Path(C["urn"]["ladder"])
    ladder = json.loads((ladder_path if ladder_path.is_absolute() else SRC / ladder_path).read_text())

    S["meta"] = {
        "n_gold_rows": C["n_gold_rows"], "n_gold_claims": C["n_gold_claims"],
        "bootstrap": C["bootstrap"],
        "loop_version": man["loop_version"],
        "loop_model": man["model"], "urn_model": urn_man["model"],
        "loop_date_ceiling": man["config"]["date_ceiling"],
        "loop_origin_exclusion": man["config"]["origin_exclusion"],
        "urn_ceiling_policy": urn_man["ceiling_policy"],
        "urn_exclusion_policy": urn_man["exclusion_policy"],
        "urn_workers": urn_man["workers"],
        "urn_fit_population_n": ladder["population"]["n"],
        "urn_ladder": rel(ladder_path), "urn_scores": rel(a.urn), "gold": rel(a.gold),
    }
    S["gold"] = {"refuted_share": float((gold["gold_label"] == "Refuted").mean()),
               "n_rows": int(len(gold))}
    S["convention"] = {
        "four_class": "ClaimCheck convention, unsupported counts as Refuted",
        "binary_headline": "A",
        "binary_maps": C["binary_maps"],
    }
    S["paper"] = {"accuracy": C["paper"]["four_class_convention_accuracy"],
                  "n": C["paper"]["n"], "ci": C["paper"]["ci"],
                  "note": C["paper"]["note"]}

    # ---------- section 2: four-class accuracy + the ceiling ----------
    S["four_class"] = {n: {"accuracy": C["arms"][n]["four_class_convention"]["accuracy"]["value"],
                           "ci": C["arms"][n]["four_class_convention"]["accuracy"]["ci"],
                           "n": C["arms"][n]["four_class_convention"]["accuracy"]["n"],
                           "delta_vs_paper": C["arms"][n]["four_class_convention"]["delta_vs_paper"]}
                       for n in ordered}

    # the paper sends the ceiling as DD/MM/YYYY; Google reads it as M/D/YYYY, so a day past 12
    # is not a valid month and the filter is dropped. AVeriTeC dates are D-M-YYYY.
    day = gold["claim_date"].astype(str).str.split("-").str[0].astype(int)
    dropped_rows = int((day > 12).sum())
    S["ceiling"] = {
        "paper_dropped_rows": dropped_rows,
        "paper_dropped_share_rows": dropped_rows / len(gold),
        "paper_dropped_share_claims": float(
            gold.assign(d=day).drop_duplicates("claim_id")["d"].gt(12).mean()),
        "derivation": "claim dates with day > 12 cannot be read as a month, so Google drops "
                      "the paper's DD/MM/YYYY ceiling entirely",
        "arms": {},
    }
    for n in ordered:
        cel = C["arms"][n]["ceiling"]
        srp, exa = cel.get("serper") or {}, cel.get("exa") or {}
        S["ceiling"]["arms"][n] = {
            "serper_searches": srp.get("searches"),
            "serper_date_instrumented": srp.get("date_instrumented_searches"),
            "share_searches_with_post_ceiling_hit": srp.get("share_searches_with_post_ceiling_hit"),
            "share_hits_post_ceiling": srp.get("share_hits_post_ceiling"),
            "share_hits_undated": srp.get("share_hits_undated"),
            "exa_searches": exa.get("searches"),
            "exa_date_instrumented": exa.get("date_instrumented_searches"),
            "date_ceiling_on": mans[n]["config"]["date_ceiling"],
        }
    S["ceiling"]["reference_arm"] = next(
        (n for n in ordered
         if S["ceiling"]["arms"][n]["share_searches_with_post_ceiling_hit"] is not None
         and S["ceiling"]["arms"][n]["date_ceiling_on"]),
        ordered[0])

    # ---------- section 3: urn ROC on binary map A ----------
    passes_A, dropped_A = BINARY["A"]
    gold_flag = ~gold["gold_label"].isin(passes_A).to_numpy()
    urn = pd.read_parquet(a.urn)[["claim_id", *URN_MODELS.values()]].copy()
    urn["claim_id"] = urn["claim_id"].astype(str)
    joined = gold[["claim_id"]].merge(urn, on="claim_id", how="left")
    covered = joined["s3"].notna().to_numpy()

    S["urn"] = {"n_covered_rows": int(covered.sum()), "n_gold_rows": len(gold),
                "coverage_note": C["urn"]["coverage_note"], "models": {}}
    for model, col in URN_MODELS.items():
        m = C["urn"]["models"][model]
        s = joined[col].to_numpy(float)
        S["urn"]["models"][model] = {
            "score_column": col,
            "auc": m["auc"]["A"]["value"], "auc_ci": m["auc"]["A"]["ci"],
            "auc_fc_gold": ladder["models"][model]["auc_oof"],
            "auc_fc_gold_ci": ladder["models"][model]["auc_ci"],
            "threshold_2pct": m["threshold_2pct"],
            "fitted_fpr": m["fitted"]["A"]["fpr"]["value"],
            "fitted_recall": m["fitted"]["A"]["flag_recall"]["value"],
            "fitted_recall_ci": m["fitted"]["A"]["flag_recall"]["ci"],
            "roc": roc(s[covered], gold_flag[covered]),
        }

    S["operating_points"] = {
        n: {"fpr": C["arms"][n]["binary"]["A"]["fpr"]["value"],
            "recall": C["arms"][n]["binary"]["A"]["flag_recall"]["value"],
            "recall_ci": C["arms"][n]["binary"]["A"]["flag_recall"]["ci"],
            "accuracy": C["arms"][n]["binary"]["A"]["accuracy"]["value"],
            "precision": C["arms"][n]["binary"]["A"]["precision"]["value"]}
        for n in ordered}

    # ---------- section 4: matched comparison ----------
    S["matched"] = {}
    for n in ordered:
        b = C["arms"][n]["binary"]["A"]
        row = {"fpr": b["fpr"]["value"], "label": arm_label(n),
               "systems": {"loop": {"recall": b["flag_recall"]["value"],
                                    "ci": b["flag_recall"]["ci"],
                                    "fpr": b["fpr"]["value"]}},
               "mcnemar": {}}
        for model in ("7-flag", "3-voice"):
            q = C["urn"]["models"][model]["matched"].get(n, {}).get("A")
            if q is None:
                continue
            row["systems"][f"urn {model}"] = {"recall": q["flag_recall"]["value"],
                                              "ci": q["flag_recall"]["ci"],
                                              "fpr": q["fpr"]["value"]}
            key = f"{n} vs urn {model} at matched FPR on binary A"
            if key in C["mcnemar"]:
                row["mcnemar"][f"urn {model}"] = C["mcnemar"][key]
        S["matched"][n] = row

    # maps B and C, for the caveat that the ordering does not change
    S["maps_bc"] = {}
    for v in ("B", "C"):
        entry = {"arms": {}, "urn": {}}
        for n in ordered:
            entry["arms"][n] = C["arms"][n]["binary"][v]["flag_recall"]["value"]
            for model in ("7-flag", "3-voice"):
                q = C["urn"]["models"][model]["matched"].get(n, {}).get(v)
                if q is not None:
                    entry["urn"].setdefault(model, {})[n] = q["flag_recall"]["value"]
        entry["loop_leads_7flag"] = {
            n: entry["arms"][n] >= entry["urn"].get("7-flag", {}).get(n, -1.0)
            for n in ordered}
        S["maps_bc"][v] = entry

    S["mcnemar_arms"] = {k: v for k, v in C["mcnemar"].items() if " vs urn " not in k}

    # ---------- section 5: where the gap comes from, the bar and the ceiling ----------
    cfp = Path(a.counterfactual)
    if cfp.exists():
        CF = json.loads(cfp.read_text())
        S["counterfactual"] = {
            "generated_from": rel(cfp), "definition": CF["counterfactual"],
            "bootstrap": CF["bootstrap"], "arms": {}}
        for n in ordered:
            e = CF["arms"].get(n)
            if e is None:
                continue
            an, lk = e["unsupported_anatomy"], e["leak"]
            S["counterfactual"]["arms"][n] = {
                "actual_accuracy": e["actual"]["four_class_convention"]["value"],
                "no_bar_accuracy": e["cf_a_no_bar"]["four_class_convention"]["value"],
                "no_bar_accuracy_ci": e["cf_a_no_bar"]["four_class_convention"]["ci"],
                "actual_recall": e["actual"]["binary_A"]["flag_recall"]["value"],
                "no_bar_recall": e["cf_a_no_bar"]["binary_A"]["flag_recall"]["value"],
                "actual_fpr": e["actual"]["binary_A"]["fpr"]["value"],
                "no_bar_fpr": e["cf_a_no_bar"]["binary_A"]["fpr"]["value"],
                "n_flipped": e["n_flipped_by_cf_a"],
                "n_unsupported": an["n_unsupported"], "bar_refusal": an["bar_refusal"],
                "retrieval_miss": an["retrieval_miss"], "other": an["other"],
                "stance_docs": lk["stance_docs"],
                "fact_check_share": lk["fact_check_share"],
                "post_claim_share": lk["post_claim_share"],
                "date_ceiling_on": mans[n]["config"]["date_ceiling"]}
        S["counterfactual"]["best_no_bar"] = max(
            S["counterfactual"]["arms"],
            key=lambda n: S["counterfactual"]["arms"][n]["no_bar_accuracy"], default=None)
    else:
        S["counterfactual"] = None

    # ---------- section 6: speed and cost ----------
    rows = []
    for n in ordered:
        sc = C["arms"][n]["speed_cost"]
        rows.append({"system": arm_label(n),
                     "llm_calls": sc["llm_calls_per_claim"],
                     "serper": sc["serper_calls_per_claim"], "exa": sc["exa_calls_per_claim"],
                     "scrapes": sc["scrapes_per_claim"],
                     "prompt_tok": sc["tokens_per_claim"].get("prompt"),
                     "completion_tok": sc["tokens_per_claim"].get("completion"),
                     "cost_usd": sc["cost_usd_per_claim"],
                     "throughput": sc["throughput_claims_per_min"],
                     "concurrency": sc["concurrency_k_claims"],
                     "concurrency_unit": "claims in flight",
                     "llm_p50_s": sc["llm_latency_s_p50_uncached"]})
    u = C["urn"]["speed_cost"]
    rows.append({"system": "urn", "llm_calls": u["llm_calls_per_claim"],
                 "serper": u["search_calls_per_claim"], "exa": 0.0,
                 "scrapes": u["scrapes_per_claim"],
                 "prompt_tok": u["tokens_per_claim"]["prompt"],
                 "completion_tok": u["tokens_per_claim"]["completion"],
                 "cost_usd": u["cost_usd_per_claim"],
                 "throughput": u["throughput_claims_per_min"],
                 "concurrency": u["workers"], "concurrency_unit": "workers",
                 "llm_p50_s": u["llm_latency_s_p50_uncached"]})
    S["speed_cost"] = {"rows": rows,
                       "loop_wall_source": C["arms"][first]["speed_cost"]["wall_source"],
                       "urn_wall_source": u["wall_source"]}
    S["depth"] = {"loop_serper_stages": man["config"]["tries_per_claim"],
                  "loop_exa_stages": 1 if man["config"]["exa_enabled"] else 0,
                  "loop_stages": man["config"]["tries_per_claim"]
                  + (1 if man["config"]["exa_enabled"] else 0),
                  "loop_pages_per_round": {n: mans[n]["config"]["pages_per_round"]
                                           for n in ordered},
                  "urn_searches_per_claim": u["search_calls_per_claim"],
                  "urn_docs_per_claim": u["docs_per_claim"]}

    (out / "stats.json").write_text(json.dumps(S, indent=1, default=float))
    print("stats.json written", out / "stats.json")
    if missing:
        print("arms not in comparison.json, skipped:", ", ".join(missing))

    # =================== FIGURES ===================
    F = out / "figures"

    # --- figure 1: four-class accuracy, horizontal bars with CI whiskers ---
    labels = ["ClaimCheck paper, as reported"] + [arm_label(n) for n in ordered]
    vals = [S["paper"]["accuracy"]] + [S["four_class"][n]["accuracy"] for n in ordered]
    cis = [None] + [S["four_class"][n]["ci"] for n in ordered]
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(labels) + 1.1))
    y = np.arange(len(labels))[::-1]
    for yi, v, ci, lab in zip(y, vals, cis, labels):
        face = "none" if ci is None else LIGHT
        ax.barh(yi, v, height=0.55, color=face, edgecolor=INK if ci is None else "none",
                linewidth=1.1)
        if ci is not None:
            ax.plot([ci[0], ci[1]], [yi, yi], color=INK, lw=1.2)
            for e in ci:
                ax.plot([e, e], [yi - 0.12, yi + 0.12], color=INK, lw=1.2)
        ax.text(max(v, ci[1] if ci else v) + 0.014, yi, f"{v:.3f}", va="center", color=INK,
                fontsize=10, fontfamily="monospace")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("four-class accuracy, ClaimCheck convention, 500 gold rows")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(F / "fig_four_class.svg")
    plt.close(fig)

    # --- figure 2: urn ROC with the loop operating points ---
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot([0, 1], [0, 1], color=LIGHT, lw=1, ls="--", zorder=0)
    for model, style, colr, lw in (("7-flag", "-", INK, 1.7), ("3-voice", "-", MUT, 1.2)):
        m = S["urn"]["models"][model]
        ax.plot(m["roc"]["fpr"], m["roc"]["recall"], style, color=colr, lw=lw,
                label=f"urn {model}, AUC {m['auc']:.3f} "
                      f"[{m['auc_ci'][0]:.3f}, {m['auc_ci'][1]:.3f}]")
    lead = {"arrowstyle": "-", "color": LIGHT, "lw": 0.8, "shrinkA": 2, "shrinkB": 5}
    m7 = S["urn"]["models"]["7-flag"]
    ax.scatter([m7["fitted_fpr"]], [m7["fitted_recall"]], marker="D", s=55, color=INK,
               zorder=4)
    ax.annotate(f"7-flag at the fitted 2% threshold\nFPR {m7['fitted_fpr']:.3f}, "
                f"recall {m7['fitted_recall']:.3f}",
                (m7["fitted_fpr"], m7["fitted_recall"]), textcoords="offset points",
                xytext=(18, -30), fontsize=10, color=INK, arrowprops=lead)
    marks = ["o", "s", "^", "v"]
    for i, n in enumerate(ordered):
        p = S["operating_points"][n]
        ax.scatter([p["fpr"]], [p["recall"]], marker=marks[i % len(marks)], s=70,
                   facecolors="white", edgecolors=INK, linewidths=1.4, zorder=5)
        ax.annotate(f"{arm_label(n)}\nFPR {p['fpr']:.3f}, recall {p['recall']:.3f}",
                    (p["fpr"], p["recall"]), textcoords="offset points",
                    xytext=(40, -150 - 44 * i), fontsize=10, color=INK,
                    arrowprops=lead)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("false positive rate on gold Supported claims")
    ax.set_ylabel("flag recall")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(F / "fig_roc.svg")
    plt.close(fig)

    # --- figure 3: matched comparison, paired bars per arm ---
    groups = [(n, S["matched"][n]) for n in ordered]
    sysnames = list(dict.fromkeys(k for _, g in groups for k in g["systems"]))
    fig, ax = plt.subplots(figsize=(11, 4.9))
    wd = 0.8 / max(1, len(sysnames))
    colors = {0: LIGHT, 1: INK, 2: MUT}
    for j, sysname in enumerate(sysnames):
        xs, ys, los, his = [], [], [], []
        for i, (_, g) in enumerate(groups):
            e = g["systems"].get(sysname)
            if e is None:
                continue
            xs.append(i + (j - (len(sysnames) - 1) / 2) * wd)
            ys.append(e["recall"])
            los.append(e["recall"] - e["ci"][0])
            his.append(e["ci"][1] - e["recall"])
        legend = sysname if sysname.startswith("urn") else "the loop, at its own FPR"
        ax.bar(xs, ys, wd * 0.9, color=colors.get(j, MUT), label=legend,
               yerr=[los, his], error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
        for x, v in zip(xs, ys):
            ax.text(x, 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                    color=INK if j == 0 else "white", fontfamily="monospace", rotation=90)
    for i, (n, g) in enumerate(groups):
        lines = [f"FPR held at {g['fpr']:.3f}"] + [
            f"McNemar vs {k} p={v['p_exact']:.3f}" for k, v in g["mcnemar"].items()]
        for j, line in enumerate(lines):
            ax.text(i, -0.115 - 0.058 * j, line, ha="center", va="top", fontsize=8.5,
                    color=MUT, fontfamily="sans-serif" if j == 0 else "monospace",
                    transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g["label"] for _, g in groups])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("flag recall at the loop's own false positive rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(F / "fig_matched.svg")
    plt.close(fig)

    print("figures written:", len(list(F.glob("fig_*.svg"))), "->", F)


if __name__ == "__main__":
    main()
