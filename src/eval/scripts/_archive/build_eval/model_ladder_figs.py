"""Figures for the Score Ladder artifact, drawn from ladder.json and transfer.json.

    uv run python -m eval.scripts.build_eval.model_ladder_figs

ONE drawing implementation, used twice. The gold-fitted ladder and the urn-fitted
transfer ladder get identical treatment so the two pairs can be read against each
other without the style itself being a variable.

  fig_roc.svg / fig_roc_transfer.svg
      the whole ROC on the left, the 0 to 10% false alarm region on the right with
      the 2% operating point marked. One curve per model.
  fig_weights.svg / fig_weights_transfer.svg
      three stacked panels on ONE shared log-likelihood-ratio axis, dot at the
      fitted weight, whisker at the 95% bootstrap interval. A bucket with fewer
      than MIN_SUP documents in either gold class is drawn hollow, so a wide
      interval reads as thin evidence rather than as a measured null. On the
      transfer figure a tick marks where the gold fit puts the same bucket.

House style, near monochrome, 11in wide so every figure renders at one width,
svg.fonttype none.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = Path("eval/data/urn_runs/e1_ctx/model_ladder")
LAD = json.loads((A / "ladder.json").read_text())
MODELS = ("3-voice", "7-flag", "28-cell")
MIN_SUP = 30

INK, MUT, LIGHT = "#1a1a1a", "#666666", "#bbbbbb"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                     "axes.edgecolor": MUT, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "svg.fonttype": "none"})

TITLE = {"3-voice": "3 buckets, support / refute / silent",
         "7-flag": "7 buckets, one per read flag",
         "28-cell": "28 buckets, read flag by source tier"}
STROKE = {"3-voice": ("#a6a6a6", 2.4, "-"), "7-flag": ("#5a5a5a", 1.7, "--"),
          "28-cell": (INK, 1.7, "-")}
NUDGE = {"3-voice": (10, 7), "7-flag": (10, -15), "28-cell": (12, 9)}
FLAG_DESC = LAD["flag_desc"]
SUPPORT = LAD["models"]   # document support is a property of the eval corpus, shared


def auc_of(v: dict) -> float:
    return v["auc_oof"] if "auc_oof" in v else v["auc"]


def draw_roc(M: dict, out: Path, xlab: str = "false alarm rate, gold-true claims flagged",
             ylab: str = "recall, gold-false claims flagged") -> None:
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11, 4.8),
                                  gridspec_kw={"width_ratios": [1.22, 1], "wspace": 0.22})
    for m in MODELS:
        x, y = M[m]["roc"]
        c, lw, ls = STROKE[m]
        lo, hi = M[m]["auc_ci"]
        ax.plot(x, y, ls, color=c, lw=lw,
                label=f"{m}   AUC {auc_of(M[m]):.3f}  [{lo:.3f}, {hi:.3f}]")
        axz.plot(x, y, ls, color=c, lw=lw)
        axz.scatter([M[m]["fpr_2pct"]], [M[m]["recall_2pct"]], s=44, color=c,
                    edgecolors="white", linewidths=0.9, zorder=3)
        axz.annotate(f"{M[m]['recall_2pct']*100:.1f}%",
                     (M[m]["fpr_2pct"], M[m]["recall_2pct"]),
                     xytext=NUDGE[m], textcoords="offset points", fontsize=10, color=c)
    ax.plot([0, 1], [0, 1], "-", color="#e4e1db", lw=1, zorder=0)
    ax.axvline(0.02, color=LIGHT, lw=1, ls=":")
    ax.text(0.032, 0.955, "2% false alarm budget", color=MUT, fontsize=10)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower right", fontsize=10, borderaxespad=0.8)
    ax.set_title("the whole curve", fontsize=11, loc="left", color=MUT, pad=6)

    axz.axvline(0.02, color=LIGHT, lw=1, ls=":")
    axz.set_xlim(0, 0.10); axz.set_ylim(0, 0.64)
    axz.text(0.023, 0.615, "2%", color=MUT, fontsize=10)
    axz.set_xlabel("false alarm rate")
    axz.set_title("the part we operate in, up to 10%", fontsize=11, loc="left",
                  color=MUT, pad=6)
    for a in (ax, axz):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)


def draw_weights(M: dict, out: Path, compare_key: str | None = None) -> None:
    counts = [len(M[m]["weights"]) for m in MODELS]
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(11, 9.6),
                             gridspec_kw={"height_ratios": counts}, sharex=True)
    vals = [v for m in MODELS for v in M[m]["weight_ci"].values()]
    if compare_key:
        vals += [[w, w] for m in MODELS for w in M[m][compare_key].values()]
    lo_all, hi_all = min(v[0] for v in vals), max(v[1] for v in vals)
    pad = 0.06 * (hi_all - lo_all)

    for ax, m in zip(axes, MODELS):
        chans = list(M[m]["weights"])
        if m == "28-cell":   # faint band per flag group, four tiers each
            for g in range(0, len(chans), 8):
                ax.axhspan(g - 0.5, min(g + 3.5, len(chans) - 0.5),
                           color="#f4f2ee", zorder=-1, lw=0)
        ax.axvline(0, color=LIGHT, lw=1, zorder=0)
        for i, c in enumerate(chans):
            w = M[m]["weights"][c]
            a, b = M[m]["weight_ci"][c]
            sup = SUPPORT[m]["support"][c]
            thin = min(sup["docs_true"], sup["docs_false"]) < MIN_SUP
            ax.plot([a, b], [i, i], color=LIGHT, lw=1.2, zorder=1, solid_capstyle="butt")
            if compare_key:
                g = M[m][compare_key][c]
                ax.plot([g, g], [i - 0.34, i + 0.34], color=MUT, lw=1.3, zorder=2)
            ax.scatter([w], [i], s=46, zorder=3,
                       **({"facecolors": "white", "edgecolors": INK, "linewidths": 1.1}
                          if thin else {"color": INK}))
        labs = [f"{c}  {FLAG_DESC[c]}" if m == "7-flag" else c for c in chans]
        ax.set_yticks(range(len(chans))); ax.set_yticklabels(labs, fontsize=10)
        ax.invert_yaxis()
        ax.set_ylim(len(chans) - 0.4, -0.6)
        ax.set_xlim(lo_all - pad, hi_all + pad)
        ax.set_title(TITLE[m], fontsize=11, loc="left", color=INK, pad=6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("log likelihood ratio, one evidence document")

    handles = [axes[0].scatter([], [], s=46, color=INK,
                               label="30 or more documents in both gold classes"),
               axes[0].scatter([], [], s=46, facecolors="white", edgecolors=INK,
                               linewidths=1.1, label="thinner than that on one side")]
    if compare_key:
        handles.append(plt.Line2D([], [], color=MUT, lw=1.3,
                                  label="where the gold fit puts it"))
    fig.legend(handles=handles, frameon=False, fontsize=10, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.55, 0.0))
    fig.subplots_adjust(left=0.26, right=0.985, top=0.968, bottom=0.085, hspace=0.17)
    fig.savefig(out); plt.close(fig)


FITLAB = {"gold": "fitted on gold", "urn": "fitted on the urns", "pool": "fitted on both"}
EVLAB = {"gold": "scored on fc gold", "urn": "scored on the urns"}


def draw_matrix(C: dict, out: Path) -> None:
    """Who wins where. One panel per eval, one row per fit, three models a row."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.30})
    off = {"3-voice": -0.24, "7-flag": 0.0, "28-cell": 0.24}
    for ax, ev in zip(axes, ("gold", "urn")):
        for i, fit in enumerate(("gold", "urn", "pool")):
            for m in MODELS:
                c = C[f"{fit}->{ev}"][m]
                if not c:
                    continue
                col = STROKE[m][0]
                yy = i + off[m]
                lo, hi = c["auc_ci"]
                ax.plot([lo, hi], [yy, yy], color=LIGHT, lw=1.2, zorder=1,
                        solid_capstyle="butt")
                ax.scatter([c["auc"]], [yy], s=44, color=col, zorder=2,
                           edgecolors="white", linewidths=0.7)
        ax.set_yticks(range(3)); ax.set_yticklabels([FITLAB[f] for f in ("gold", "urn", "pool")],
                                                    fontsize=10)
        ax.set_ylim(2.6, -0.6)
        ax.set_title(EVLAB[ev], fontsize=11, loc="left", color=MUT, pad=6)
        ax.set_xlabel("AUC")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#f0eee9", lw=0.8)
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=STROKE[m][0], ms=7,
                          markeredgecolor="white", label=m) for m in MODELS]
    fig.legend(handles=handles, frameon=False, fontsize=10, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.90, bottom=0.30)
    fig.savefig(out); plt.close(fig)


draw_roc(LAD["models"], A / "fig_roc.svg")
draw_weights(LAD["models"], A / "fig_weights.svg")
print("wrote fig_roc.svg, fig_weights.svg")

tr = A / "transfer.json"
if tr.exists():
    T = json.loads(tr.read_text())["models"]
    draw_roc(T, A / "fig_roc_transfer.svg")
    draw_weights(T, A / "fig_weights_transfer.svg", compare_key="gold_weights")
    print("wrote fig_roc_transfer.svg, fig_weights_transfer.svg")

cr = A / "cross.json"
if cr.exists():
    C = json.loads(cr.read_text())["cells"]
    draw_roc(C["gold->urn"], A / "fig_roc_urn.svg",
             xlab="false alarm rate, timeline claims flagged (corrected)",
             ylab="recall, community-noted false claims flagged")
    draw_matrix(C, A / "fig_matrix.svg")
    print("wrote fig_roc_urn.svg, fig_matrix.svg")


def draw_eps_bands(res: dict, out: Path) -> None:
    """Per-band false share under both weight sets, with exact intervals.

    Two panels so the two cuts can be read against each other. The bar behind each
    point is the share of the urn sitting in that band, on its own right axis, so
    the reader can see that the well-populated bands are the ones with tight
    intervals rather than having to infer it from the counts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"wspace": 0.26})
    panels = (("two urn weights, bands already in use", res["bands_two_urn"]),
              ("gold fitted weights, frame percentile matched cuts", res["bands_gold"]))
    hi = max(r["cp"][1] for _, bt in panels for r in bt["rows"] if r["decided"])
    for ax, (title, bt) in zip(axes, panels):
        rows = bt["rows"]
        x = list(range(len(rows)))
        axb = ax.twinx()
        axb.bar(x, [r["urn_pct"] for r in rows], width=0.62, color="#ececec",
                edgecolor="none", zorder=0)
        axb.set_ylim(0, 100)
        axb.set_yticks([0, 25, 50])
        axb.tick_params(axis="y", colors=MUT, labelsize=9)
        axb.set_ylabel("share of urn, %", color=MUT, fontsize=9)
        for s in axb.spines.values():
            s.set_visible(False)
        for i, r in enumerate(rows):
            if not r["decided"]:
                continue
            lo, up = r["cp"]
            ax.plot([i, i], [lo, up], color=MUT, lw=1.4, zorder=3,
                    solid_capstyle="butt")
            ax.plot([i], [r["eps"]], "o", ms=6, color=INK, zorder=4)
            ax.annotate(f"{r['F']}/{r['decided']}", (i, up), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=9, color=MUT)
        ax.set_zorder(axb.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels([r["band"] for r in rows], fontsize=9, rotation=20,
                           ha="right")
        ax.set_ylim(0, hi * 1.12)
        ax.set_ylabel("false share of decided claims")
        ax.set_xlabel("score band, low to high")
        ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=14)
        ax.axhline(res["eps"], color=LIGHT, lw=1.1, ls=":", zorder=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].annotate(f"whole draw, {res['eps']:.3f}",
                     (len(panels[0][1]["rows"]) - 0.5, res["eps"]),
                     textcoords="offset points", xytext=(-4, 5), ha="right",
                     fontsize=9, color=MUT)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.86, bottom=0.22, wspace=0.34)
    fig.savefig(out); plt.close(fig)


ep = Path("eval/data/eps_audit/pooled/result.json")
if ep.exists():
    draw_eps_bands(json.loads(ep.read_text()), A / "fig_eps_bands.svg")
    print("wrote fig_eps_bands.svg")
