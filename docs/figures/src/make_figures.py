"""Draw every README figure from figdata.json (built by extract_figdata.py) plus the
constants below. $0, no network.   uv run --with matplotlib python docs/figures/src/make_figures.py"""
import json, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT = HERE.parent
D = json.loads((HERE / "figdata.json").read_text())

# Reader comparison, 500 fact-checked claims (250 true / 250 false), same pages for every
# reader, weights refitted per reader, out of fold. AUCs as published in the frozen
# evaluation (src/eval/data/urn_runs/final_instrument/final_instrument.md, reader table);
# extract_figdata.py reproduces them to within 0.005.
READERS = [("DeepSeek V4 Flash", 0.866), ("DeepSeek V4 Pro", 0.851), ("Kimi K2.6", 0.848)]
# Measured spend per 1,000 page reads, from the run logs of the same documents:
# Pro 4,739 reads at $0.001977 and Kimi 4,739 at $0.012469 (reader_lab/sub500/*.log);
# Flash 4,556 reads at $0.000139 (reader_lab/requestion/flash_v5b.log).
PRICE = {"DeepSeek V4 Flash": 0.139, "DeepSeek V4 Pro": 1.977, "Kimi K2.6": 12.469}
# Missed false claims by cause (1,500 false claims; boundary at 2% false positives).
# Hand-read split of misses with a refuting page, 60 per condition (research log 2026-09-16).
MISSES = {"evidence dated\nbefore the fact-check": {"no refuting page found": 656, "reader misjudged": 62, "other": 157},
          "today's web": {"no refuting page found": 566, "reader misjudged": 88, "other": 262}}
AVERITEC_AUC = 0.869

# dataviz reference palette (light surface)
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
NEUTRAL = "#b8b7b0"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
W, DPI = 8.0, 200
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.labelsize": 11, "axes.labelcolor": INK2, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5, "xtick.color": INK2, "ytick.color": INK2, "legend.fontsize": 10.5,
    "legend.frameon": False, "text.color": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "figure.facecolor": "white", "savefig.facecolor": "white"})
MINUS = lambda s: s.replace("-", "−")


def frame(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.get_layout_engine().set(w_pad=0.2, h_pad=0.2); fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)


def fig_roc():
    fig, ax = plt.subplots(figsize=(W, 5.2), layout="constrained")
    for key, c, lab in (("before", BLUE, "evidence dated before the fact-check"), ("today", ORANGE, "today's web")):
        r = D["roc"][key]
        ax.plot(r["fpr"], r["tpr"], color=c, linewidth=2, label=f"{lab} (AUC {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("share of true claims wrongly flagged")
    ax.set_ylabel("share of false claims caught")
    ax.set_title("Telling true claims from false ones")
    ax.legend(loc="lower right")
    ax.text(0.98, 0.30, f"Unseen benchmark (AVeriTeC), same weights: AUC {AVERITEC_AUC:.3f}",
            ha="right", va="bottom", fontsize=10, color=INK2, transform=ax.transAxes)
    frame(ax, "both")
    save(fig, "roc.png")


def fig_weights():
    names = {"5": "states the claim as fact", "4": "points toward it", "3": "contested or mixed",
             "X": "on topic, no direction", "I": "irrelevant", "2": "points against it",
             "1": "contradicts the claim"}
    flags = list(D["weights"])
    fig, ax = plt.subplots(figsize=(W, 4.4), layout="constrained")
    for i, k in enumerate(flags):
        w, (lo, hi) = D["weights"][k]["w"], D["weights"][k]["ci"]
        y = len(flags) - 1 - i
        c = BLUE if w > 0 else RED
        ax.plot([lo, hi], [y, y], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(w, y, "o", color=c, markersize=8, markeredgecolor="white", markeredgewidth=2, zorder=3)
        ax.text(hi + 0.12 if w > 0 else lo - 0.12, y, MINUS(f"{w:+.2f}"), va="center",
                ha="left" if w > 0 else "right", fontsize=10, color=INK2)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(range(len(flags)), [names[k] for k in reversed(flags)])
    ax.set_xlim(-3.3, 3.8)
    ax.set_xlabel("weight per page (log-odds; right = evidence the claim is true)")
    ax.set_title("What each flag is worth")
    frame(ax, "x")
    ax.spines["left"].set_visible(False)
    save(fig, "weights.png")


def fig_score_by_verdict():
    labels = [(1, "false"), (2, "mostly\nfalse"), (3, "mixed"), (4, "mostly\ntrue"), (5, "true")]
    colors = [RED, "#f0a3a2", NEUTRAL, "#86b6ef", "#256abf"]
    data = [[r["s"] for r in D["score_by_verdict"] if r["v"] == v] for v, _ in labels]
    thr = D["threshold"]
    fig, ax = plt.subplots(figsize=(W, 4.8), layout="constrained")
    bp = ax.boxplot(data, tick_labels=[f"{n}\nn={len(x):,}" for (_, n), x in zip(labels, data)],
                    widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(linewidth=2, color=INK), whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED))
    for p, c in zip(bp["boxes"], colors):
        p.set_facecolor(c); p.set_edgecolor("white"); p.set_linewidth(2)
    ax.hlines(thr, 0.5, 5.45, color=INK, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.set_xlim(0.5, 6.2)
    ax.text(5.5, thr, "nudge\nboundary", va="center", ha="left", fontsize=10, color=INK)
    ax.set_ylabel("score (log-odds)")
    ax.set_xlabel("fact-checker's verdict", labelpad=8)
    ax.set_title("Scores by the fact-checker's verdict")
    frame(ax)
    save(fig, "score_by_veracity.png")


def fig_readers():
    names = [n for n, _ in READERS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.8), layout="constrained")
    y = range(len(names))[::-1]
    for yi, (n, auc) in zip(y, READERS):
        a.plot(auc, yi, "o", color=BLUE, markersize=9, markeredgecolor="white", markeredgewidth=2)
        a.text(auc + 0.004, yi, f"{auc:.3f}", va="center", fontsize=10.5, color=INK2)
    a.set_yticks(list(y), names); a.set_xlim(0.80, 0.90); a.set_ylim(-0.6, 2.6)
    a.set_title("AUC on the same 500 claims", fontsize=11.5)
    frame(a, "x"); a.spines["left"].set_visible(False)
    for yi, n in zip(y, names):
        b.barh(yi, PRICE[n], color=BLUE if n == names[0] else NEUTRAL, height=0.5)
        b.text(PRICE[n] + 0.25, yi, f"${PRICE[n]:.2f}", va="center", fontsize=10.5, color=INK2)
    b.set_yticks(list(y), names); b.set_xlim(0, 15); b.set_ylim(-0.6, 2.6)
    b.set_title("US$ per 1,000 pages read", fontsize=11.5)
    frame(b, "x"); b.spines["left"].set_visible(False)
    fig.suptitle("A bigger reader does not score better, only costs more", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    save(fig, "readers.png")


def fig_misses():
    cats = [("no refuting page found", BLUE), ("reader misjudged", ORANGE), ("other", NEUTRAL)]
    fig, ax = plt.subplots(figsize=(W, 4.6), layout="constrained")
    for i, (cond, parts) in enumerate(MISSES.items()):
        base, tot = 0, sum(parts.values())
        for cat, c in cats:
            v = parts[cat]
            ax.bar(i, v, bottom=base, width=0.5, color=c, edgecolor="white", linewidth=2,
                   label=cat if i == 0 else None)
            ax.text(i + 0.29, base + v / 2, f"{v / tot:.0%} ({v})", va="center", fontsize=10, color=INK2)
            base += v
        ax.text(i, tot + 18, f"{tot} missed", ha="center", fontsize=10.5, color=INK)
    ax.set_xticks([0, 1], list(MISSES)); ax.set_xlim(-0.5, 2.6); ax.set_ylim(0, 1000)
    ax.set_ylabel("false claims missed (of 1,500)")
    ax.set_title("Why false claims are missed")
    ax.legend(loc="upper right", ncol=1)
    frame(ax)
    save(fig, "missdecomp.png")


if __name__ == "__main__":
    fig_roc(); fig_weights(); fig_score_by_verdict(); fig_readers(); fig_misses()
