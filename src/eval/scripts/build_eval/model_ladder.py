"""Model ladder on fc-gold: 3-voice vs 7-flag vs 28-cell, ROC and weights with CIs.

    uv run python -m eval.scripts.build_eval.model_ladder

$0 -- refits the saved E1 reads. One population, one estimator, one convention,
so the three models differ only in how finely a read is bucketed.

Population (Daniel 2026-08-27). fc-gold, media-provenance axis excluded
(fit_urn's default), and the `mixed` rating_subtype ALSO set aside -- the stratum
where the fact-checker graded the post's framing rather than the claim we score
(B3). `unprovable` stays, scored FALSE at eval and out of every fit, which is the
standing veracity-3 convention. The 7-flag AUC on this cut is printed against
the published 0.8620 / 42.2% sensitivity row (row folds, in-sample threshold).

Models. Every read is one document in one bucket. The bucket is
  3-voice   support / refute / silent          (flags 5,4 | 1,2 | 3,X,I)
  7-flag    the read flag itself               (5,4,3,X,I,2,1)
  28-cell   flag x source tier                 (x PRIMARY/RELIABLE/UNRELIABLE/UNRATED)
Each bucket's weight is the Laplace-smoothed log likelihood ratio, +1 per bucket
count and +K on each class total. UNSHRUNK at every level -- no cell borrows from
its flag marginal -- so a thin cell's interval reflects only its own evidence.

Intervals. 2,000-rep CLUSTER bootstrap (resample cluster_id), seed 707, with the
old within-class row bootstrap run beside it for the design effect (2026-09-08).
The three models share every replicate, so the dAUC intervals are paired. Weights
are refit inside each replicate; the AUC replicates redo the whole out-of-fold
protocol. Folds are cluster-disjoint (blake2b of cluster_id). The operating point
is nested (threshold picked on the four training folds, applied to the fifth);
the in-sample choice stays under `insample`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval import fit_urn
from eval.scripts.build_eval.graded_urn import FLAGS, FLAG_DESC
from eval.scripts.build_eval.quality_urn import TIERS, VOICE, CELLS

RESULTS = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
# Env-overridable so a re-run can verify the trust gate without overwriting
# the pinned ladder.json (2026-08-28).
OUT = Path(__import__("os").environ.get("MODEL_LADDER_OUT",
                      "eval/data/urn_runs/e1_ctx/model_ladder"))
SET_ASIDE = "mixed"
BOOT_REPS, BOOT_SEED = fit_urn.BOOT_REPS, fit_urn.BOOT_SEED
VOICE_NAME = {"S": "support", "R": "refute", "E": "silent"}
MODELS = ("3-voice", "7-flag", "28-cell")
KEYS = {"3-voice": lambda fl, tr: VOICE_NAME[VOICE[fl]],
        "7-flag": lambda fl, tr: fl,
        "28-cell": lambda fl, tr: f"{fl} | {tr}"}


def load_docs(population: set[str] | None = None,
              keep_media_axis: bool = False) -> list[dict]:
    """Headline rows keeping each document's (flag, tier), plus rating_subtype.

    Mirrors fit_urn.load_headline's population exactly -- asserted in main().
    """
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
        docs = [(d["read"]["direction"], d.get("rel") or "UNRATED")
                for d in r.get("results") or []
                if (d.get("read") or {}).get("direction") in FLAGS]
        # Pad rule -- see fit_urn.load. Slots with no read flag (short result
        # list, empty search, failed fetch) are silent documents from an
        # unrated source, so the ladder sees the same PAD_TO slots per claim
        # that the 3-voice loader does.
        docs += [("I", "UNRATED")] * max(0, fit_urn.PAD_TO - len(docs))
        if (not keep_media_axis
                and (axis.get(r["review_url"]) or "untagged") == fit_urn.MEDIA_AXIS):
            continue
        cluster = clusters.get(r["review_url"], r["review_url"])
        rows.append({"review_url": r["review_url"], "y": 1 if v >= 4 else 0, "mid": v == 3, "docs": docs,
                     "subtype": r.get("rating_subtype") or "?", "cluster": cluster,
                     "fold": fit_urn.fold_of(cluster, fit_urn.K_FOLDS)})
    return rows


def channels(model: str) -> list[str]:
    """Bucket names in display order, support -> silent -> refute."""
    seen = []
    for fl, tr in CELLS:
        ch = KEYS[model](fl, tr)
        if ch not in seen:
            seen.append(ch)
    return seen


def count_matrix(rows: list[dict], model: str) -> np.ndarray:
    chans = channels(model)
    idx = {c: i for i, c in enumerate(chans)}
    C = np.zeros((len(chans), len(rows)))
    for j, r in enumerate(rows):
        for fl, tr in r["docs"]:
            C[idx[KEYS[model](fl, tr)], j] += 1
    return C


def fit_w(C: np.ndarray, y: np.ndarray, mid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Laplace-smoothed per-bucket log-LR on the fit rows of idx. Mixed never
    enters (already dropped); unprovable is excluded here by `mid`."""
    keep = idx[~mid[idx]]
    p, n = keep[y[keep] == 1], keep[y[keep] == 0]
    cT, cF = C[:, p].sum(1), C[:, n].sum(1)
    K = C.shape[0]
    return np.log(((cT + 1) / (cT.sum() + K)) / ((cF + 1) / (cF.sum() + K)))


def oof_scores(C, y, mid, fold) -> np.ndarray:
    s = np.empty(C.shape[1])
    allidx = np.arange(C.shape[1])
    for k in range(fit_urn.K_FOLDS):
        tr, te = allidx[fold != k], allidx[fold == k]
        s[te] = fit_w(C, y, mid, tr) @ C[:, te]
    return s


def auc_np(s_pos: np.ndarray, s_neg: np.ndarray) -> float:
    neg = np.sort(s_neg)
    lo = np.searchsorted(neg, s_pos, side="left")
    hi = np.searchsorted(neg, s_pos, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(s_pos) * len(neg)))


def roc(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x = FPR over gold-TRUE, yv = recall over gold-FALSE. Low score = flagged."""
    trues, falses = np.sort(s[y == 1]), np.sort(s[y == 0])
    thr = np.unique(s)
    fpr = np.searchsorted(trues, thr, side="right") / len(trues)
    rec = np.searchsorted(falses, thr, side="right") / len(falses)
    return np.r_[0.0, fpr, 1.0], np.r_[0.0, rec, 1.0]


def recall_at_fpr(s: np.ndarray, y: np.ndarray, budget: float) -> tuple[float, float, float]:
    pairs = sorted(zip(s.tolist(), y.tolist()))
    return fit_urn.recall_at_fpr(pairs, budget)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=Path, default=None,
                    help="parquet with a claim_id column: restrict to those claims")
    ap.add_argument("--keep-media-axis", action="store_true",
                    help="keep media-provenance claims (fit_urn --keep-media-axis)")
    args = ap.parse_args()
    pop = fit_urn.load_population(args.population)
    rows = load_docs(pop, args.keep_media_axis)
    ref = (fit_urn.load(RESULTS, None, False, fit_urn.load_judged_axis(), pop)
           if args.keep_media_axis else fit_urn.load_headline(RESULTS, population=pop))
    assert len(rows) == len(ref), (len(rows), len(ref))
    print(f"headline population {len(rows)} claims")

    kept = [r for r in rows if r["subtype"] != SET_ASIDE]
    n_set = len(rows) - len(kept)
    y = np.array([r["y"] for r in kept])
    mid = np.array([r["mid"] for r in kept])
    fold = np.array([r["fold"] for r in kept])
    print(f"set aside rating_subtype={SET_ASIDE}: {n_set} -> {len(kept)} claims "
          f"(T {int(y.sum())} / F {int((y == 0).sum())}, "
          f"unprovable still in at eval {int(mid.sum())})")

    C = {m: count_matrix(kept, m) for m in MODELS}
    print(f"documents {int(C['7-flag'].sum()):,}")

    res = {}
    for m in MODELS:
        s = oof_scores(C[m], y, mid, fold)
        a = auc_np(s[y == 1], s[y == 0])
        r2, f2, t2 = recall_at_fpr(s, y, 0.02)
        nest = fit_urn.nested_threshold(C[m], y, mid, fold)
        w = fit_w(C[m], y, mid, np.arange(len(kept)))
        res[m] = {"channels": channels(m), "auc_oof": a,
                  "recall_2pct": nest["recall"], "fpr_2pct": nest["fpr"],
                  "thresholds_by_fold": nest["thresholds_by_fold"],
                  "threshold_spread": nest["threshold_spread"],
                  "insample": {"recall_2pct": r2, "fpr_2pct": f2, "threshold_2pct": t2},
                  "weights": dict(zip(channels(m), map(float, w))),
                  "roc": [x.tolist() for x in roc(s, y)]}
        print(f"{m:9s} {len(channels(m)):2d} buckets  AUC {a:.4f}  recall@2%FPR "
              f"{nest['recall']:.4f} nested (FPR {nest['fpr']:.4f}) / {r2:.4f} in-sample")

    # Trust gate: this cut is the published B3 sensitivity row.
    #
    # THE PINNED POPULATION HAS CHANGED (2026-08-28 evening) AND THESE NUMBERS HAVE
    # NOT. Deliberately. The full-coverage gold media-provenance sweep purged 63 of
    # 1,768 judged gold-FALSE claims (3.56% [2.80, 4.53], majority of 3), so the cut
    # is now 3,211 claims (T 1,502 / F 1,709), not 3,274 (T 1,502 / F 1,772).
    # MEASURED ON THE NEW POPULATION, for Daniel to approve before anything is
    # repinned to it:
    #     3-voice  AUC 0.851887 -> 0.852290   recall@2%FPR 0.339729 -> 0.341720
    #     7-flag   AUC 0.861966 -> 0.862458   recall@2%FPR 0.422122 -> 0.422469
    #     28-cell  AUC 0.868983 -> 0.869475   recall@2%FPR 0.428894 -> 0.430661
    # The assertion below still PASSES, but only just: |0.862458 - 0.8620| = 4.58e-4
    # against a 5e-4 tolerance, i.e. 92% of the budget consumed. It is therefore no
    # longer a meaningful guard on this population. PROPOSED replacement, pending
    # Daniel's sign-off: 0.8625 / 0.4225 with the same tolerances. Do not apply it
    # until he has approved the population change; until then a further drift will
    # correctly fire this gate.
    # To reproduce the pre-gold-purge numbers: GOLD_EXCLUSIONS=none (gold screen off,
    # CN purge kept) or EXTRA_EXCLUSIONS=none (every label-validity screen off).
    # 2026-09-08: folds are now keyed on cluster_id (a re-randomised assignment
    # on this all-singleton population) and the recall headline is nested, so the
    # pinned row-fold numbers are reported as a deviation, not asserted.
    print(f"B3 pin (row folds, in-sample thr): 7-flag AUC 0.8620 / 42.2%  ->  "
          f"{res['7-flag']['auc_oof']:.4f} / {res['7-flag']['insample']['recall_2pct']:.1%} "
          f"in-sample, {res['7-flag']['recall_2pct']:.1%} nested")

    # Per-bucket document support on the fit population.
    fitrows = ~mid
    p, n = np.where(fitrows & (y == 1))[0], np.where(fitrows & (y == 0))[0]
    for m in MODELS:
        res[m]["support"] = {c: {"docs_true": int(a), "docs_false": int(b)}
                             for c, a, b in zip(channels(m), C[m][:, p].sum(1),
                                                C[m][:, n].sum(1))}

    # ---- paired bootstrap ---------------------------------------------------
    print(f"bootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}, cluster + row) ...")
    cl = fit_urn.cluster_index(kept)
    w_reps = {d: {m: [] for m in MODELS} for d in ("cluster", "row")}
    a_reps = {d: {m: [] for m in MODELS} for d in ("cluster", "row")}
    for design in ("cluster", "row"):
        rng = np.random.default_rng(BOOT_SEED)
        for _ in range(BOOT_REPS):
            idx = fit_urn.boot_idx(rng, y, cl, design == "cluster")
            yy = y[idx]
            for m in MODELS:
                w_reps[design][m].append(fit_w(C[m], y, mid, idx))
                s = fit_urn.oof_np(C[m], y, mid, fold, idx)
                a_reps[design][m].append(auc_np(s[yy == 1], s[yy == 0]))

    for m in MODELS:
        W, Wr = np.array(w_reps["cluster"][m]), np.array(w_reps["row"][m])
        lo, hi = np.percentile(W, [2.5, 97.5], axis=0)
        res[m]["weights_ci"] = {c: [float(a), float(b)]
                                for c, a, b in zip(channels(m), lo, hi)}
        res[m]["design_effect_weights"] = {c: float(v) for c, v in
                                           zip(channels(m), W.var(0) / Wr.var(0))}
        A, Ar = np.array(a_reps["cluster"][m]), np.array(a_reps["row"][m])
        res[m]["auc_ci"] = [float(x) for x in np.percentile(A, [2.5, 97.5])]
        res[m]["auc_ci_row"] = [float(x) for x in np.percentile(Ar, [2.5, 97.5])]
        res[m]["design_effect_auc"] = float(A.var() / Ar.var())
    for m in MODELS[1:]:
        d = np.array(a_reps["cluster"][m]) - np.array(a_reps["cluster"]["3-voice"])
        dr = np.array(a_reps["row"][m]) - np.array(a_reps["row"]["3-voice"])
        res[m]["dauc_vs_3voice"] = res[m]["auc_oof"] - res["3-voice"]["auc_oof"]
        res[m]["dauc_ci"] = [float(x) for x in np.percentile(d, [2.5, 97.5])]
        res[m]["dauc_ci_row"] = [float(x) for x in np.percentile(dr, [2.5, 97.5])]
        res[m]["design_effect_dauc"] = float(d.var() / dr.var())

    for m in MODELS:
        print(f"\n{m}  AUC {res[m]['auc_oof']:.4f} "
              f"[{res[m]['auc_ci'][0]:.4f}, {res[m]['auc_ci'][1]:.4f}] cluster / "
              f"[{res[m]['auc_ci_row'][0]:.4f}, {res[m]['auc_ci_row'][1]:.4f}] row  "
              f"deff {res[m]['design_effect_auc']:.2f}")
        if m != "3-voice":
            print(f"  dAUC vs 3-voice {res[m]['dauc_vs_3voice']:+.4f} "
                  f"[{res[m]['dauc_ci'][0]:+.4f}, {res[m]['dauc_ci'][1]:+.4f}] cluster / "
                  f"[{res[m]['dauc_ci_row'][0]:+.4f}, {res[m]['dauc_ci_row'][1]:+.4f}] row  "
                  f"deff {res[m]['design_effect_dauc']:.2f}")
        for c in channels(m):
            lo, hi = res[m]["weights_ci"][c]
            sup = res[m]["support"][c]
            print(f"  {c:22s} {res[m]['weights'][c]:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
                  f"   docs T {sup['docs_true']:5d} / F {sup['docs_false']:5d}")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"population": {"headline": len(rows), "set_aside_subtype": SET_ASIDE,
                              "set_aside_n": n_set, "n": len(kept),
                              "n_true": int(y.sum()), "n_false": int((y == 0).sum()),
                              "n_unprovable_eval_only": int(mid.sum()),
                              "docs": int(C["7-flag"].sum())},
               "boot": {"reps": BOOT_REPS, "seed": BOOT_SEED, "design": "cluster_id"},
               "fold_key": "cluster_id",
               "population_file": str(args.population) if args.population else None,
               "keep_media_axis": args.keep_media_axis,
               "flag_desc": FLAG_DESC, "tiers": list(TIERS), "models": res}
    (OUT / "ladder_clustered.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT / 'ladder_clustered.json'}")


if __name__ == "__main__":
    main()
