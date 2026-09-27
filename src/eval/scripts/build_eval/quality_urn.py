"""Quality-split Truth Odds: do the graded flags gain from knowing WHO said it?

    uv run python -m eval.scripts.build_eval.quality_urn

Interaction ladder over the saved E1 reads ($0), each level an aggregation of one
7-flag x 4-tier cell count matrix (tiers from the per-document `rel` field:
PRIMARY / RELIABLE / UNRELIABLE / UNRATED):

    3-voice              (asserted == headline_metrics_clustered.json)
    7-flag               (the shipped fit, asserted == graded_metrics_clustered.json)
    L1  3-voice x rated/unrated     (rel_urn's committed scheme, sanity anchor)
    L2  7-flag  x rated/unrated
    L3  7-flag  x 4 tiers, HARD shrinkage: a cell with < MIN_CELL docs in either
        class of the training rows falls back to its flag-marginal weight
    L4  7-flag  x 4 tiers, SOFT shrinkage (exploratory): w = lam*cell +
        (1-lam)*flag-marginal with lam = m/(m+MIN_CELL), m = min(class counts)

Conventions inherited from fit_urn / graded_urn: the FROZEN fc-gold population
(eval/data/populations/fc_gold.parquet, media-axis already excluded in it), mixed
counts as FALSE at eval and stays out of every fit, 5 cluster-disjoint folds by
blake2b of cluster_id, pad to fit_urn.PAD_TO silent slots, Laplace +1 per channel
count / +K on totals.
All comparisons out-of-fold; paired stratified bootstrap (weights refit inside
every replicate, shrinkage re-decided inside every fold) for dAUC CIs.

Finer NewsGuard-score bins were CONSIDERED AND DROPPED: `ng` covers only ~37%
of documents and UNRELIABLE holds 249 of 39,368 docs, so numeric bins would be
mostly imputation (same verdict as rel_urn). Recorded in the JSON, not silent.
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval import fit_urn
from eval.scripts.build_eval.graded_urn import FLAGS, FLAG_DESC
from eval.scripts.build_eval.e1_figures import roc, style, save, _auc_np

RESULTS = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
OUTDIR = Path("eval/data/urn_runs/e1_ctx")
POPULATION = Path("eval/data/populations/fc_gold.parquet")
FIGDIR = OUTDIR / "figures"
TIERS = ("PRIMARY", "RELIABLE", "UNRELIABLE", "UNRATED")
RATED = ("PRIMARY", "RELIABLE")
CELLS = [(fl, tr) for fl in FLAGS for tr in TIERS]          # 28, C28 row order
VOICE = {"5": "S", "4": "S", "1": "R", "2": "R", "3": "E", "X": "E", "I": "E"}
MIN_CELL = 200
BOOT_REPS, BOOT_SEED = 2000, 707


def load_docs(population: set[str] | None = None) -> list[dict]:
    """Headline rows with per-document (flag, tier) kept. Must mirror
    fit_urn.load_headline exactly -- asserted in main()."""
    axis = fit_urn.load_judged_axis()
    gold_excl = fit_urn.gold_excluded_ids()   # mirrors fit_urn.load's gold screen
    clusters = fit_urn.load_clusters()
    rows = []
    for line in RESULTS.open():
        r = json.loads(line)
        v = r.get("veracity")
        if v not in (1, 2, 3, 4, 5):
            continue
        if r["review_url"] in gold_excl:
            continue
        if population is not None and r["review_url"] not in population:
            continue
        docs = [(dirn, d.get("rel") or "UNRATED", d.get("domain") or "?",
                 (d.get("read") or {}).get("evidence") or [])
                for d in r.get("results") or []
                if (dirn := (d.get("read") or {}).get("direction")) in FLAGS]
        # Pad rule -- see fit_urn.load. Unflagged slots are silent, unrated,
        # no-domain, no-evidence documents.
        docs += [("I", "UNRATED", "?", [])] * max(0, fit_urn.PAD_TO - len(docs))
        if (axis.get(r["review_url"]) or "untagged") == fit_urn.MEDIA_AXIS:
            continue
        cluster = clusters.get(r["review_url"], r["review_url"])
        rows.append({"y": 1 if v >= 4 else 0, "mid": v == 3, "docs": docs,
                     "claim": r.get("claim_resolved") or r.get("claim_text") or "",
                     "review_url": r["review_url"],
                     "fold": fit_urn.fold_of(cluster, fit_urn.K_FOLDS)})
    return rows


def cell_matrix(rows: list[dict]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CELLS)}
    C = np.zeros((len(CELLS), len(rows)))
    for j, r in enumerate(rows):
        for fl, tr, *_ in r["docs"]:
            C[idx[(fl, tr)], j] += 1
    return C


def aggregator(key) -> tuple[np.ndarray, list[str], np.ndarray]:
    """0/1 matrix folding the 28 cells into this level's channels.
    Returns (A, channel names, parent flag-index per channel)."""
    chans, parent = [], []
    for fl, tr in CELLS:
        ch = key(fl, tr)
        if ch not in chans:
            chans.append(ch)
            parent.append(FLAGS.index(fl))
    A = np.zeros((len(chans), len(CELLS)))
    for i, (fl, tr) in enumerate(CELLS):
        A[chans.index(key(fl, tr)), i] = 1
    return A, chans, np.array(parent)


LEVELS = {  # name -> cell (flag, tier) -> channel key
    "3-voice": lambda fl, tr: VOICE[fl] if VOICE[fl] != "E" else "E",
    "7-flag": lambda fl, tr: fl,
    "L1 voice x rated": lambda fl, tr: f"{VOICE[fl]}|{'rated' if tr in RATED else 'unrated'}",
    "L2 flag x rated": lambda fl, tr: f"{fl}|{'rated' if tr in RATED else 'unrated'}",
    "L2h flag x rated (hard)": lambda fl, tr: f"{fl}|{'rated' if tr in RATED else 'unrated'}",
    "L3 flag x tier (hard)": lambda fl, tr: f"{fl}|{tr}",
    "L4 flag x tier (soft)": lambda fl, tr: f"{fl}|{tr}",
}
# NOTE on the hard rule: min-per-class support < MIN_CELL is INTRINSIC to the
# most informative cells -- a strongly directional flag is by construction rare
# in one class (5|rated has 2,184 TRUE docs and 89 FALSE docs; that imbalance IS
# the signal). So the hard rule guts directional cells at every level (9/14 at
# L2, 22/28 at L3) and its variants are reported as ROBUSTNESS rows, while each
# level's headline is the unshrunk fit (consistent with how 3-voice and 7-flag
# themselves are fitted) with the bootstrap -- which refits weights inside every
# replicate -- carrying the small-cell uncertainty into the CI. The soft rule
# (L4) is the principled middle ground.
SHRINK = {"L2h flag x rated (hard)": "hard",
          "L3 flag x tier (hard)": "hard", "L4 flag x tier (soft)": "soft"}


def fit_w(C: np.ndarray, y: np.ndarray, mid: np.ndarray, idx: np.ndarray,
          parent: np.ndarray | None, mode: str | None,
          C7: np.ndarray | None) -> np.ndarray:
    """Laplace-smoothed per-channel log-LR on the fit rows of idx (mixed out).
    mode hard/soft shrinks under-supported channels toward the 7-flag marginal
    fitted on the SAME rows."""
    keep = idx[~mid[idx]]
    p, n = keep[y[keep] == 1], keep[y[keep] == 0]
    cT, cF = C[:, p].sum(1), C[:, n].sum(1)
    K = C.shape[0]
    w = np.log(((cT + 1) / (cT.sum() + K)) / ((cF + 1) / (cF.sum() + K)))
    if mode:
        w7 = fit_w(C7, y, mid, idx, None, None, None)
        m = np.minimum(cT, cF)
        if mode == "hard":
            w = np.where(m >= MIN_CELL, w, w7[parent])
        else:
            lam = m / (m + MIN_CELL)
            w = lam * w + (1 - lam) * w7[parent]
    return w


def oof_pairs(C, y, mid, fold, parent, mode, C7) -> list[tuple[float, int]]:
    allidx = np.arange(C.shape[1])
    pairs = []
    for k in range(fit_urn.K_FOLDS):
        tr, te = allidx[fold != k], allidx[fold == k]
        w = fit_w(C, y, mid, tr, parent, mode, C7)
        pairs += list(zip(w @ C[:, te], y[te]))
    return pairs


def main() -> None:
    pop = fit_urn.load_population(POPULATION)
    rows = load_docs(pop)
    ref = fit_urn.load_headline(RESULTS, population=pop)
    assert len(rows) == len(ref), (len(rows), len(ref))
    mine = collections.Counter(fl for r in rows for fl, *_ in r["docs"])
    theirs = collections.Counter({k: sum(r["flags"].get(k, 0) for r in ref) for k in FLAGS})
    assert mine == theirs, (mine, theirs)

    # The 3-voice and 7-flag fits on this same frozen population. NOT
    # headline_metrics.json: that is the SHIPPED file and has been 7-flag since
    # 2026-09-14, so the 3-voice arm reads its clustered sibling instead.
    hm = json.loads(fit_urn.E1_METRICS_CLUSTERED.read_text())["overall"]
    gm = json.loads((OUTDIR / "graded_metrics_clustered.json").read_text())

    C28 = cell_matrix(rows)
    y = np.array([r["y"] for r in rows]); mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows])

    aggs = {name: aggregator(key) for name, key in LEVELS.items()}
    A7 = aggs["7-flag"][0]

    print(f"claims {len(rows)} (T {int(y.sum())} / F {int((y == 0).sum())}), "
          f"docs {int(C28.sum())}")

    # ---- ladder, out of fold ------------------------------------------------
    ladder = {}
    for name, (A, chans, parent) in aggs.items():
        C = A @ C28
        pairs = oof_pairs(C, y, mid, fold, parent, SHRINK.get(name), A7 @ C28)
        a = fit_urn.auc(pairs)
        rm, fm, tm = fit_urn.recall_at_fpr(pairs, hm["fpr"])
        r2, f2, t2 = fit_urn.recall_at_fpr(pairs, 0.02)
        w_full = fit_w(C, y, mid, np.arange(len(rows)), parent,
                       SHRINK.get(name), A7 @ C28)
        ladder[name] = {"channels": len(chans), "auc_oof": a,
                        "recall_matched": rm, "fpr_matched": fm,
                        "recall_2pct": r2, "fpr_2pct": f2, "threshold_2pct": t2,
                        "weights": dict(zip(chans, map(float, w_full))),
                        "pairs": pairs}
        print(f"{name:24s} {len(chans):2d} ch  AUC {a:.4f}  "
              f"recall@matched {rm:.3f}  @2% {r2:.3f}")

    assert abs(ladder["3-voice"]["auc_oof"] - hm["auc_oof"]) < 1e-9
    assert abs(ladder["7-flag"]["auc_oof"] - gm["auc_oof"]) < 1e-9
    print("baseline reproductions OK (3-voice == headline, 7-flag == graded)")

    # ---- cell support (fit population, full data) ---------------------------
    fitrows = ~mid
    p, n = np.where(fitrows & (y == 1))[0], np.where(fitrows & (y == 0))[0]
    cT28, cF28 = C28[:, p].sum(1), C28[:, n].sum(1)
    support = {f"{fl}|{tr}": {"docs_true": int(a), "docs_false": int(b),
                              "shrunk_hard": bool(min(a, b) < MIN_CELL)}
               for (fl, tr), a, b in zip(CELLS, cT28, cF28)}
    A2, ch2, _ = aggs["L2 flag x rated"]
    c2T, c2F = (A2 @ C28)[:, p].sum(1), (A2 @ C28)[:, n].sum(1)
    support_l2 = {c: {"docs_true": int(a), "docs_false": int(b),
                      "shrunk_hard": bool(min(a, b) < MIN_CELL)}
                  for c, a, b in zip(ch2, c2T, c2F)}
    n_shrunk = sum(v["shrunk_hard"] for v in support.values())
    print(f"L3 hard shrinkage: {n_shrunk}/28 cells below {MIN_CELL} docs in a class "
          f"-> flag-marginal")

    # ---- paired bootstrap ---------------------------------------------------
    print(f"paired bootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}) ...")
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    names = list(LEVELS)
    Cs = {nm: aggs[nm][0] @ C28 for nm in names}
    deltas = {nm: [] for nm in names}
    for _ in range(BOOT_REPS):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs = {}
        for nm in names:
            C = Cs[nm]; parent = aggs[nm][2]
            s = np.empty(len(idx))
            for k in range(fit_urn.K_FOLDS):
                m = fold[idx] == k
                tr = idx[~m]
                if not m.any() or not len(tr):
                    continue
                w = fit_w(C, y, mid, tr, parent, SHRINK.get(nm), Cs["7-flag"])
                s[m] = w @ C[:, idx[m]]
            yy = y[idx]
            aucs[nm] = _auc_np(s[yy == 1], s[yy == 0])
        for nm in names:
            deltas[nm].append((aucs[nm] - aucs["7-flag"], aucs[nm] - aucs["3-voice"]))
    ci = {nm: {"vs_7flag": [float(q) for q in np.percentile([d[0] for d in deltas[nm]], [2.5, 97.5])],
               "vs_3voice": [float(q) for q in np.percentile([d[1] for d in deltas[nm]], [2.5, 97.5])],
               "d_vs_7flag": ladder[nm]["auc_oof"] - ladder["7-flag"]["auc_oof"],
               "d_vs_3voice": ladder[nm]["auc_oof"] - ladder["3-voice"]["auc_oof"]}
          for nm in names}
    for nm in names:
        c = ci[nm]
        print(f"  {nm:24s} dAUC vs 7-flag {c['d_vs_7flag']:+.4f} "
              f"[{c['vs_7flag'][0]:+.4f}, {c['vs_7flag'][1]:+.4f}]   "
              f"vs 3-voice {c['d_vs_3voice']:+.4f} "
              f"[{c['vs_3voice'][0]:+.4f}, {c['vs_3voice'][1]:+.4f}]")

    # ---- the S|PRIMARY oddity ----------------------------------------------
    # (a) composition: flag mix of supporting docs by tier
    comp = {tr: {fl: int(cT28[CELLS.index((fl, tr))] + cF28[CELLS.index((fl, tr))])
                 for fl in ("5", "4")} for tr in TIERS}
    # (b) L3 UNSHRUNK cell weights for the support flags (full fit, no fallback)
    w28_raw = fit_w(C28, y, mid, np.arange(len(rows)), None, None, None)
    raw = {f"{fl}|{tr}": float(w28_raw[CELLS.index((fl, tr))]) for fl, tr in CELLS}
    # (c) example documents: PRIMARY supports on gold-FALSE claims
    examples = []
    for r in rows:
        if r["y"] == 0 and not r["mid"]:
            for fl, tr, dom, ev in r["docs"]:
                if tr == "PRIMARY" and fl in ("5", "4"):
                    examples.append({"flag": fl, "domain": dom,
                                     "claim": r["claim"][:160],
                                     "url": r["review_url"],
                                     "evidence": [str(e)[:200] for e in ev][:2]})
    print(f"\nPRIMARY supporting docs on gold-FALSE fit claims: {len(examples)}")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    figs = []

    # fig 1: ladder ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    order = names
    ys = np.arange(len(order))[::-1]
    for ax, key, lab in ((axes[0], "auc_oof", "out-of-fold AUC"),
                         (axes[1], "recall_2pct", "recall @ 2% FPR budget")):
        vals = [ladder[nm][key] for nm in order]
        for yp, v, nm in zip(ys, vals, order):
            base = nm in ("3-voice", "7-flag")
            ax.plot([min(vals), v], [yp, yp], "-", color="0.85", linewidth=1)
            ax.plot([v], [yp], "o", color="0.55" if base else "0.15", markersize=8,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3)
            ax.annotate(f"{v:.3f}", (v, yp), fontsize=8, color="0.15",
                        va="center", ha="left", xytext=(6, 0), textcoords="offset points")
        ax.set_yticks(ys, order, fontsize=8.5)
        lo, hi = min(vals), max(vals)
        ax.set_xlim(lo - (hi - lo) * 0.15, hi + (hi - lo) * 0.35)
        ax.set_title(lab, fontsize=10); style(ax)
    fig.suptitle("Interaction ladder — every level out-of-fold", fontsize=10.5)
    fig.tight_layout()
    figs.append(save(fig, FIGDIR, "quality_ladder.png"))

    # fig 2: heatmap ----------------------------------------------------------
    M = np.array([[raw[f"{fl}|{tr}"] for tr in TIERS] for fl in FLAGS])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    v = np.abs(M).max()
    im = ax.imshow(M, cmap="RdGy_r", vmin=-v, vmax=v, aspect="auto")
    for i, fl in enumerate(FLAGS):
        for j, tr in enumerate(TIERS):
            cell = f"{fl}|{tr}"
            s = support[cell]
            shr = s["shrunk_hard"]
            ax.text(j, i, f"{M[i, j]:+.2f}" + ("*" if shr else ""),
                    ha="center", va="center", fontsize=8,
                    color="0.98" if abs(M[i, j]) > v * 0.55 else "0.05")
    ax.set_xticks(range(len(TIERS)), TIERS, fontsize=8.5)
    ax.set_yticks(range(len(FLAGS)), [f"{fl}  {FLAG_DESC[fl]}" for fl in FLAGS], fontsize=8)
    ax.set_title(f"Raw cell weights, flag x tier\n(* = < {MIN_CELL} docs in a class; "
                 "L3-hard falls back to the flag-marginal there)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="log-LR per document")
    fig.tight_layout()
    figs.append(save(fig, FIGDIR, "quality_heatmap.png"))

    # fig 3: oddity panel -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax = axes[0]
    x = np.arange(len(TIERS)); wdt = 0.38
    f5 = [comp[tr]["5"] for tr in TIERS]; f4 = [comp[tr]["4"] for tr in TIERS]
    tot = [a + b for a, b in zip(f5, f4)]
    ax.bar(x - wdt / 2, [a / t if t else 0 for a, t in zip(f5, tot)], wdt,
           color="0.20", label="flag 5 share")
    ax.bar(x + wdt / 2, [a / t if t else 0 for a, t in zip(f4, tot)], wdt,
           color="0.60", label="flag 4 share")
    ax.set_xticks(x, [f"{tr}\nn={t2:,}" for tr, t2 in zip(TIERS, tot)], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Composition of SUPPORTING docs by tier", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8); style(ax)
    ax = axes[1]
    ys2 = np.arange(len(TIERS))[::-1]
    for k, (fl, c) in enumerate((("5", "0.15"), ("4", "0.55"))):
        vals = [raw[f"{fl}|{tr}"] for tr in TIERS]
        ax.plot(vals, ys2 + (0.12 if k else -0.12), "o", color=c, markersize=7,
                markeredgecolor="white", label=f"flag {fl} cell weight (unshrunk)")
    ax.axvline(0, color="0.85", linewidth=1)
    ax.set_yticks(ys2, TIERS, fontsize=8.5)
    ax.set_xlabel("log-LR per document")
    ax.set_title("Support cell weights by tier (raw)", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); style(ax)
    fig.tight_layout()
    figs.append(save(fig, FIGDIR, "quality_oddity.png"))

    out = {
        "input": str(RESULTS), "n": len(rows), "n_true": int(y.sum()),
        "docs": int(C28.sum()), "min_cell": MIN_CELL,
        "ladder": {nm: {k: v for k, v in d.items() if k != "pairs"}
                   for nm, d in ladder.items()},
        "delta_auc_ci95": ci,
        "cell_support": support,
        "cell_support_l2": support_l2,
        "support_composition": comp,
        "raw_cell_weights": raw,
        "primary_support_on_false_examples": examples[:40],
        "ng_bins_verdict": "dropped: ng covers 14,706/39,368 docs (37%), UNRELIABLE "
                           "tier holds 249 docs total — numeric bins would be "
                           "mostly imputation (same verdict as rel_urn)",
        "bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED},
        "figures": [str(f) for f in figs],
    }
    (OUTDIR / "quality_metrics.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUTDIR / 'quality_metrics.json'} + {len(figs)} figures")


if __name__ == "__main__":
    main()
