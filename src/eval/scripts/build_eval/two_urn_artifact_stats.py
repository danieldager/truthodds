"""Derive every number and figure for the two-urn results artifact from run files.

Nothing in the artifact is hand-copied. This script reads the run outputs and
emits stats.json plus monochrome figures; the artifact HTML is generated from
those. Rerun after any refit and republish.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from eval.scripts.build_eval.fit_urn import load_judged_axis, MEDIA_AXIS  # noqa: E402

RUN = SRC / "eval/data/urn_runs"
OUT = RUN / "synth_expt/artifact"


def main():
    OUT.mkdir(exist_ok=True)
    FIT = json.loads((RUN / "true_timeline/two_urn_fit.json").read_text())
    EPS = 0.10

    INK, MUT, LIGHT = "#1a1a1a", "#666666", "#bbbbbb"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                         "axes.edgecolor": MUT, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "svg.fonttype": "none"})

    S = {}

    def w7():
        for f in FIT["fits7"]:
            if abs(f["eps"] - EPS) < 1e-9 and f.get("weights"):
                return f["weights"]

    W7 = w7()

    def flags_of(rec):
        return [d["read"]["direction"] for d in rec.get("results") or []
                if d.get("read") and d["read"].get("direction")]

    def s7_of(flags):
        return sum(W7.get(f, 0.0) for f in flags)

    # ---------- datasets ----------
    u = pl.read_parquet(SRC / "eval/data/tweet_corpus/timeline_urn.parquet")
    par = u.filter(pl.col("checkworthy") & (pl.col("type") == "assertion") & (pl.col("lang") == "en"))
    S["tl"] = {"claims": u.height, "cw": u.filter("checkworthy").height,
               "posts": u["post_id"].n_unique(), "parity": par.height,
               "parity_posts": par["post_id"].n_unique(),
               "frames": {r["frame"]: r["len"] for r in par.group_by("frame").len().iter_rows(named=True)},
               "topics": {r["topic"]: r["len"] for r in
                          par.group_by("topic").len().sort("len", descending=True).head(8).iter_rows(named=True)}}

    excl = {e["claim"][:80] for e in json.loads((RUN / "c2_false/fit_exclusions.json").read_text())}
    cn_claims, cn_topics = [], collections.Counter()
    for p in ("scores.jsonl", "scores_ext.jsonl"):
        for line in (RUN / "c2_false" / p).open():
            r = json.loads(line)
            if (r.get("claim_text") or "")[:80] in excl:
                continue
            fl = flags_of(r)
            if fl:
                cn_claims.append(fl)
                cn_topics[r.get("topic") or "?"] += 1
    S["cn"] = {"claims": len(cn_claims), "docs": sum(len(f) for f in cn_claims),
               "topics": dict(cn_topics.most_common(8))}
    tl_scored = []
    for line in (RUN / "true_timeline/scores.jsonl").open():
        fl = flags_of(json.loads(line))
        if fl:
            tl_scored.append(fl)
    S["tl"]["scored"] = len(tl_scored)
    S["tl"]["docs"] = sum(len(f) for f in tl_scored)

    # flag mixes
    def mix(claimflags):
        c = collections.Counter()
        for fl in claimflags:
            c.update(fl)
        t = sum(c.values())
        return {"sup": (c["5"] + c["4"]) / t, "ref": (c["1"] + c["2"]) / t,
                "sil": (c["3"] + c["X"] + c["I"]) / t}
    S["mix"] = {"timeline": mix(tl_scored), "cn": mix(cn_claims)}

    # ---------- weights + eps ----------
    S["weights7"] = {str(f["eps"]): f["weights"] for f in FIT["fits7"] if f.get("weights")}
    gm = json.loads((RUN / "e1_ctx/graded_metrics.json").read_text())
    gold_w = gm["weights"] if "weights" in gm else gm.get("overall", {}).get("weights", {})
    S["gold_weights7"] = gold_w
    S["transfer"] = {str(f["eps"]): {"auc": f.get("auc"), "rec2": f.get("recall_at_2pct_fpr")}
                     for f in FIT["fits7"] if f.get("weights")}
    S["e1_graded_baseline"] = {"auc": gm.get("auc") or gm.get("overall", {}).get("auc"),
                               "rec2": gm.get("recall_at_2pct_fpr") or gm.get("overall", {}).get("recall_at_2pct_fpr")}

    # ---------- gold rows ----------
    axis = load_judged_axis()
    gold = []
    for line in (RUN / "e1_ctx/results-00.jsonl").open():
        r = json.loads(line)
        v = r.get("veracity")
        if v not in (1, 2, 3, 4, 5):
            continue
        if (axis.get(r["review_url"]) or "untagged") == MEDIA_AXIS:
            continue
        fl = flags_of(r)
        if fl:
            gold.append({"v": v, "s": s7_of(fl), "flags": fl,
                         "fc": sum(1 for d in r.get("results") or [] if d.get("fc_domain"))})
    S["gold_n"] = {"total": len(gold), "true": sum(1 for r in gold if r["v"] >= 4),
                   "false_mixed": sum(1 for r in gold if r["v"] <= 3)}
    trues = sorted(r["s"] for r in gold if r["v"] >= 4)
    S["recall_by_veracity"] = {}
    for b in (0.005, 0.01, 0.02):
        thr = trues[max(0, int(b * len(trues)) - 1)]
        row = {"thr": thr, "fpr": sum(1 for s in trues if s <= thr) / len(trues)}
        for lv in (1, 2, 3):
            sub = [r for r in gold if r["v"] == lv]
            row[f"v{lv}"] = sum(1 for r in sub if r["s"] <= thr) / len(sub)
        S["recall_by_veracity"][f"{b:.1%}"] = row

    # miss anatomy at 2% thr
    thr2 = trues[max(0, int(0.02 * len(trues)) - 1)]
    v1 = [r for r in gold if r["v"] == 1]
    miss = [r for r in v1 if r["s"] > thr2]
    S["miss"] = {"n_v1": len(v1), "missed": len(miss),
                 "zero_refute": sum(1 for r in miss if not any(f in ("1", "2") for f in r["flags"])) / len(miss),
                 "all_silent": sum(1 for r in miss if all(f in ("3", "X", "I") for f in r["flags"])) / len(miss),
                 "fc_seen": sum(1 for r in miss if r["fc"] > 0) / len(miss)}

    # ---------- band audit ----------
    ba = [json.loads(l) for l in (RUN / "true_timeline/band_audit.jsonl").open()]
    S["band_audit"] = []
    for t, name in ((-6.659, "0.5%"), (-4.949, "1%"), (-4.047, "2%")):
        sub = [r for r in ba if r["s7"] <= t]
        c = collections.Counter(r["verdict"] for r in sub)
        n = len(sub)
        S["band_audit"].append({"name": name, "thr": t, "n": n, "false": c["false"],
                                "true": c["true"], "unsure": c["unsure"],
                                "p_decided": c["false"] / max(c["false"] + c["true"], 1),
                                "p_floor": c["false"] / max(n, 1)})

    # ---------- synthesis gold confusion ----------
    sv = [json.loads(l) for l in (RUN / "e1_ctx/synth_verdicts.jsonl").open()]
    conf = collections.defaultdict(collections.Counter)
    for r in sv:
        conf["T" if r["veracity"] >= 4 else "F"][r["verdict"]] += 1
    S["synth_confusion"] = {k: {vv: c / sum(cnt.values()) for vv, c in cnt.items()}
                            for k, cnt in conf.items()}
    S["synth_v1_recall"] = sum(1 for r in sv if r["veracity"] == 1 and r["verdict"] == "false") / \
        sum(1 for r in sv if r["veracity"] == 1)

    # ---------- experiment policies ----------
    rows = {json.loads(l)["review_url"]: json.loads(l) for l in (RUN / "synth_expt/sample.jsonl").open()}
    arms = {}
    for a in ("serper_v1", "serper_pro", "exa_v1", "exa_pro"):
        arms[a] = {json.loads(l)["review_url"]: json.loads(l) for l in (RUN / f"synth_expt/arm_{a}.jsonl").open()}
    WPOP = {"cn_false": {"flag": 585, "cliff": 685, "pass": 699},
            "gold_false": {"flag": 850, "cliff": 873, "pass": 474},
            "gold_true": {"flag": 35, "cliff": 503, "pass": 964},
            "timeline": {"flag": 113, "cliff": 413, "pass": 1473}}
    def esc(base, fb):
        return {u: (arms[fb][u] if (b["verdict"] == "unsure" and u in arms[fb]) else b)
                for u, b in arms[base].items()}
    POL = {"flash": arms["serper_v1"], "pro": arms["serper_pro"],
           "flash_exa": esc("serper_v1", "exa_v1"), "pro_exa": esc("serper_pro", "exa_v1"),
           "pro_exa_pro": esc("serper_pro", "exa_pro")}
    def rate(pop, verd, want="false", bands=("flag", "cliff", "pass")):
        num = den = 0.0
        for b, wgt in WPOP[pop].items():
            if b not in bands:
                den += wgt
                continue
            us = [u for u, r in rows.items() if r["population"] == pop and r["band"] == b]
            num += wgt * sum(1 for u in us if verd.get(u, {}).get("verdict") == want) / len(us)
            den += wgt
        return num / den
    S["policies"] = {n: {"cn": rate("cn_false", v), "gold_f": rate("gold_false", v),
                         "gold_t_fpr": rate("gold_true", v)} for n, v in POL.items()}
    S["dial"] = []
    for bands, label in ((("flag",), "flag band only"), (("flag", "cliff"), "flag + cliff"),
                         (("flag", "cliff", "pass"), "all bands")):
        v = POL["pro_exa"]
        S["dial"].append({"label": label, "fpr": rate("gold_true", v, bands=bands),
                          "cn": rate("cn_false", v, bands=bands),
                          "gold_f": rate("gold_false", v, bands=bands)})

    json.dump(S, (OUT / "stats.json").open("w"), indent=1)
    print("stats.json written")

    # ================= FIGURES =================
    FLAGS = ["5", "4", "3", "2", "1", "X", "I"]

    # weights dot plot
    fig, ax = plt.subplots(figsize=(11, 3.2))
    xs = range(len(FLAGS))
    two = [W7[f] for f in FLAGS]
    gw = [gold_w.get(f, 0) for f in FLAGS]
    ax.axhline(0, color=LIGHT, lw=1)
    ax.scatter(xs, gw, marker="o", s=70, facecolors="none", edgecolors=INK, label="fitted on fc gold labels")
    ax.scatter(xs, two, marker="o", s=70, color=INK, label="fitted from the two urns, no labels")
    for x, a, b in zip(xs, two, gw):
        ax.plot([x, x], [a, b], color=LIGHT, lw=1, zorder=0)
    ax.set_xticks(list(xs)); ax.set_xticklabels(FLAGS)
    ax.set_ylabel("log likelihood ratio"); ax.set_xlabel("evidence flag")
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_weights.svg"); plt.close(fig)

    # eps sweep (weight of flag 1 and 5 vs eps + transfer recall)
    fig, ax = plt.subplots(figsize=(11, 3))
    epss = sorted(float(e) for e in S["weights7"])
    for f, style in (("5", "-"), ("1", "--")):
        ax.plot(epss, [S["weights7"][f"{e}" if e != int(e) else f"{e:.1f}"][f] if False else S["weights7"][str(e)][f] for e in epss],
                style, color=INK, marker="o", ms=4, label=f"flag {f}")
    ax.axhline(0, color=LIGHT, lw=1)
    ax.set_xlabel("assumed share of false claims in the feed urn (epsilon)")
    ax.set_ylabel("weight")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_eps.svg"); plt.close(fig)

    # s7 histogram with bands
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ss = [s7_of(f) for f in tl_scored]
    ax.hist(ss, bins=80, color=LIGHT, edgecolor="white")
    for thr, lbl in ((-4.05, "flag"), (-2.0, "cliff")):
        ax.axvline(thr, color=INK, lw=1.2, ls="--")
    ax.text(-11, ax.get_ylim()[1]*0.85, "flag band", color=INK)
    ax.text(-3.85, ax.get_ylim()[1]*0.85, "cliff", color=INK)
    ax.text(4, ax.get_ylim()[1]*0.85, "pass band", color=INK)
    ax.set_xlabel("claim score on the timeline"); ax.set_ylabel("claims")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_hist.svg"); plt.close(fig)

    # recall by veracity bars
    fig, ax = plt.subplots(figsize=(11, 3.2))
    budgets = list(S["recall_by_veracity"])
    xpos = range(len(budgets))
    wd = 0.25
    for i, (lv, lbl, col) in enumerate(((1, "obvious false", INK), (2, "mostly false", MUT), (3, "mixed", LIGHT))):
        ax.bar([x + (i - 1) * wd for x in xpos],
               [S["recall_by_veracity"][b][f"v{lv}"] for b in budgets], wd, color=col, label=lbl)
    ax.set_xticks(list(xpos)); ax.set_xticklabels([f"false alarms under {b}" for b in budgets])
    ax.set_ylabel("share caught"); ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_recall.svg"); plt.close(fig)

    # escalation frontier
    fig, ax = plt.subplots(figsize=(11, 3.6))
    labels = {"flash": "Flash", "pro": "Pro", "flash_exa": "Flash then Exa",
              "pro_exa": "Pro then Exa", "pro_exa_pro": "Pro then Exa with Pro"}
    for n, p in S["policies"].items():
        ax.scatter(p["gold_t_fpr"], p["cn"], s=60, color=INK)
        ax.annotate(labels[n], (p["gold_t_fpr"], p["cn"]), textcoords="offset points",
                    xytext=(8, -3), fontsize=10, color=INK)
    ax.set_xlabel("false flags on true claims"); ax.set_ylabel("community note falses caught")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_frontier.svg"); plt.close(fig)
    print("figures written:", len(list(OUT.glob("fig_*.svg"))))


if __name__ == "__main__":
    main()
