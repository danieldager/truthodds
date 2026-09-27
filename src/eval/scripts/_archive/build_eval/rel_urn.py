"""EXPLORATORY — does the urn improve if a voice carries WHO said it?

    uv run python -m eval.scripts.build_eval.rel_urn \
        -i eval/data/urn_runs/e1_ctx/results-00.jsonl -o eval/data/urn_runs/e1_ctx/figures

The committed model has three voices — supporting, refuting, silent — with one
fitted log-LR each, blind to the source. This splits every voice by the source's
reliability class and refits.

NOT production and NOT the Thursday headline (Daniel 2026-08-05). It was conceived
after the primary metric was committed, so it is a hypothesis with a bootstrap
behind it, not a validated improvement. Recorded because the direction of the
effect is interesting: reliability prices SUPPORT and does nothing for REFUTATION.

Uses `rel` (4-class, 100% coverage) rather than `ng` (NewsGuard numeric, on 38% of
reads, median 95, only 210 UNRELIABLE) — a numeric weighting there would be mostly
imputation. Silent voices carry a reliability class too: silence from a rated
source is a different observation from silence from an unrated one.

Everything is out-of-fold on the same blake2b folds as the headline, and the
3-channel arm reproduces the banked 0.8498 as a check that the generic fitter is
the same estimator.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.scripts.build_eval import fit_urn

DIR = {"supports": "S", "refutes": "R"}         # anything else is a silent voice
RATED = ("PRIMARY", "RELIABLE")
BOOT_REPS, BOOT_SEED = 400, 707
VOICES = [("S", "supporting"), ("R", "refuting"), ("E", "silent")]
DPI = 120

SCHEMES = {                                     # channel key -> reliability suffix
    "direction only": lambda rel: "",
    "+ rated vs not": lambda rel: "|rated" if rel in RATED else "|unrated",
    "+ primary / reliable / rest": lambda rel: "|" + (rel if rel in RATED else "rest"),
    "+ full rel field": lambda rel: "|" + (rel or "NONE"),
}


def load(path: Path) -> list[dict]:
    recs = []
    for line in path.open():
        r = json.loads(line)
        if r.get("veracity") not in (1, 2, 4, 5):
            continue
        docs = [(DIR.get(v, "E"), d.get("rel"), bool(d.get("fc_domain")), bool(d.get("mirror")))
                for d in r.get("results") or []
                if (v := fit_urn.FLAG_TO_VOICE.get((d.get("read") or {}).get("direction")))]
        if docs:
            recs.append({"y": 1 if r["veracity"] >= 4 else 0, "docs": docs,
                         "fold": fit_urn.fold_of(r["review_url"], fit_urn.K_FOLDS)})
    return recs


def build(recs: list[dict], key, drop_fc: bool = False) -> list[dict]:
    rows = []
    for r in recs:
        c = collections.Counter()
        for dirn, rel, fc, mir in r["docs"]:
            if not (drop_fc and (fc or mir)):
                c[dirn + key(rel)] += 1
        if c:
            rows.append({"y": r["y"], "c": c, "fold": r["fold"]})
    return rows


def fit(rows: list[dict], chans: list[str]) -> dict[str, float]:
    """fit_urn.fit generalised to an arbitrary channel set. Same Laplace form."""
    tot_t = sum(v for r in rows if r["y"] == 1 for v in r["c"].values())
    tot_f = sum(v for r in rows if r["y"] == 0 for v in r["c"].values())
    k = len(chans)
    if not tot_t or not tot_f:
        return {ch: 0.0 for ch in chans}
    return {ch: math.log(((sum(r["c"][ch] for r in rows if r["y"] == 1) + 1) / (tot_t + k))
                         / ((sum(r["c"][ch] for r in rows if r["y"] == 0) + 1) / (tot_f + k)))
            for ch in chans}


def out_of_fold(rows: list[dict]) -> list[tuple[float, int]]:
    chans = sorted({ch for r in rows for ch in r["c"]})
    oof = []
    for k in range(fit_urn.K_FOLDS):
        tr = [r for r in rows if r["fold"] != k]
        te = [r for r in rows if r["fold"] == k]
        if not tr or not te:
            continue
        w = fit(tr, chans)
        oof += [(sum(n * w[ch] for ch, n in r["c"].items()), r["y"]) for r in te]
    return oof


def report(rows: list[dict], label: str) -> float:
    oof = out_of_fold(rows)
    chans = sorted({ch for r in rows for ch in r["c"]})
    w = fit(rows, chans)
    ins = [(sum(n * w[ch] for ch, n in r["c"].items()), r["y"]) for r in rows]
    a, ai = fit_urn.auc(oof), fit_urn.auc(ins)
    rec, _, _ = fit_urn.recall_at_fpr(oof, 0.02)
    print(f"  {label:30s} {len(chans):2d} ch   AUC {a:.4f} oof / {ai:.4f} in-sample "
          f"(gap {ai - a:+.4f})   recall@2%FPR {rec:.3f}")
    return a


def paired_delta(recs: list[dict], rng) -> tuple[np.ndarray, np.ndarray]:
    """Same claim resample scored by both models -- they share the claims, so
    independent intervals would overstate the uncertainty on the difference.

    Returns (delta AUC, delta recall@2%FPR). Recall is reported with its own
    interval because it is read off a single threshold in the tail and is far
    noisier than AUC: a one-point move there is not self-evidently a gain.
    """
    a = build(recs, SCHEMES["direction only"])
    b = build(recs, SCHEMES["+ rated vs not"])
    idx = np.arange(len(a))
    d_auc, d_rec = [], []
    for _ in range(BOOT_REPS):
        take = rng.choice(idx, len(idx))
        oa = out_of_fold([a[i] for i in take])
        ob = out_of_fold([b[i] for i in take])
        if len({y for _, y in oa}) < 2:
            continue
        d_auc.append(fit_urn.auc(ob) - fit_urn.auc(oa))
        d_rec.append(fit_urn.recall_at_fpr(ob, 0.02)[0]
                     - fit_urn.recall_at_fpr(oa, 0.02)[0])
    return np.array(d_auc), np.array(d_rec)


def figure(out: Path, w6: dict[str, float], a3: float, a6: float, d: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(VOICES))
    for i, (suffix, label, col) in enumerate(
            [("|rated", "rated source", "0.25"), ("|unrated", "unrated source", "0.74")]):
        vals = [w6[v + suffix] for v, _ in VOICES]
        ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=col, edgecolor="white",
               linewidth=1.0, label=label)
        for xi, v in zip(x + (i - 0.5) * 0.36, vals):
            ax.text(xi, v + (0.10 if v > 0 else -0.10), f"{v:+.2f}", ha="center",
                    va="bottom" if v > 0 else "top", fontsize=8.5)
    ax.axhline(0, color="0.2", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in VOICES], fontsize=9.5)
    ax.set_ylabel("fitted log-likelihood ratio per voice")
    ax.set_ylim(-2.4, 3.0)
    ax.set_title("Reliability prices support, not refutation", fontsize=11)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.annotate(
        f"Splitting each voice by whether the source is rated: AUC {a3:.3f} → {a6:.3f}, "
        f"paired ΔAUC {d.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, "
        f"{np.percentile(d, 97.5):+.3f}].\nExploratory — not the headline model.",
        xy=(0, -0.20), xycoords="axes fraction", fontsize=7.5, color="0.45", va="top")
    p = out / "e1_reliability.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=Path)
    ap.add_argument("-o", "--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recs = load(args.input)
    print(f"{len(recs)} claims, {sum(len(r['docs']) for r in recs)} voices\n")
    print("every voice split by the source's reliability class:")
    aucs = {}
    for label, key in SCHEMES.items():
        aucs[label] = report(build(recs, key), label)
    # 1e-4, not exact: this fitter sums score terms in dict order while fit_urn sums
    # three named terms, so scores differ in the last float bits and a few AUC ties
    # break the other way. Same estimator.
    assert abs(aucs["direction only"] - 0.8498386934252841) < 1e-4, "baseline drifted"

    print("\nleakage guard — all fact-checker domains and mirrors dropped:")
    guard = {label: report(build(recs, key, drop_fc=True), label)
             for label, key in SCHEMES.items()}

    print(f"\npaired bootstrap, {BOOT_REPS} replicates:")
    d, dr = paired_delta(recs, np.random.default_rng(BOOT_SEED))
    for name, x in (("ΔAUC          ", d), ("Δrecall@2%FPR ", dr)):
        print(f"  {name} mean {x.mean():+.4f}  "
              f"95% CI [{np.percentile(x, 2.5):+.4f}, {np.percentile(x, 97.5):+.4f}]  "
              f"P(Δ>0) = {(x > 0).mean():.2f}")

    rows6 = build(recs, SCHEMES["+ rated vs not"])
    w6 = fit(rows6, sorted({ch for r in rows6 for ch in r["c"]}))
    p = figure(args.out, w6, aucs["direction only"], aucs["+ rated vs not"], d)
    (args.out / "reliability.json").write_text(json.dumps(
        {"input": str(args.input), "auc_oof": aucs, "auc_oof_no_factcheckers": guard,
         "weights_rated_split": w6,
         "paired_delta_auc": {"mean": float(d.mean()),
                              "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                              "p_gt_0": float((d > 0).mean())},
         "paired_delta_recall_at_2pct_fpr": {
             "mean": float(dr.mean()),
             "ci95": [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))],
             "p_gt_0": float((dr > 0).mean())},
         "bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED},
         "figure": str(p), "status": "EXPLORATORY - not the headline model"}, indent=2))


if __name__ == "__main__":
    main()
