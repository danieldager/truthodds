"""Two-urn fit — Truth Odds weights from the CN-FALSE urn vs the de-mixed timeline urn.

The two-urn pivot (docs/logodds_sprint.md): weights are no longer fitted on
fc-gold. FALSE = the CN-false urn (C2 + EXT, band-"false" claims, fit-eligible,
minus fit_exclusions.json). TRUE = the raw timeline urn — a MIXED draw whose
false fraction is eps, de-mixed at the rate level:

    p_mix(voice) = (1-eps) * p_TRUE(voice) + eps * p_FALSE(voice)
    =>  p_TRUE   = (p_mix - eps * p_FALSE) / (1 - eps)

so w_voice = log(p_TRUE / p_FALSE) with p_FALSE measured directly on the FALSE
urn. eps=0 is the naive fit (simulation 2026-08-22: naive at eps<=0.10 costs
~0.000 AUC and attenuates only w_ref). A sweep over eps in {0,3,5,10,15}% is
the sensitivity.

Evaluation is NOT on the fit urns (no gold there): the fitted weights score the
fc-gold E1 population (load_headline, mixed-as-FALSE convention) with NO refit —
a fixed-weight transfer eval against the E1-fitted baseline.

Intervals and operating point (2026-09-08, provenance audit). Every weight
carries a 95% cluster-bootstrap interval: claims are resampled by post_id within
each urn (the timeline urn has up to 26 claims per post; the CN urn is one claim
per post), 2,000 reps, seed 707, with the plain per-claim bootstrap beside it for
the design effect. Thresholds on the transfer test are NESTED over fc-gold's
cluster-disjoint folds (picked on four folds, applied to the fifth, weights
fixed); the in-sample choice stays under `insample`. The thresholds block is
computed at `--eps-headline` (default 0.10, the eps the weights table quotes).

  uv run python -m eval.scripts.build_eval.fit_two_urn            # $0
  uv run python -m eval.scripts.build_eval.fit_two_urn --true-frame search_june

Inputs: urn_runs/c2_false/scores[_ext].jsonl (+fit_exclusions.json),
        urn_runs/true_timeline/scores.jsonl, urn_runs/e1_ctx/results-00.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval import fit_urn  # noqa: E402
from eval.scripts.build_eval.fit_urn import (  # noqa: E402
    BOOT_REPS, BOOT_SEED, FLAG_TO_VOICE, PAD_TO, auc, boot_idx, cluster_index,
    extra_exclusion_paths, load_headline, load_population, nested_fixed, recall_at_fpr)

C2 = Path("eval/data/urn_runs/c2_false")
TL = Path("eval/data/urn_runs/true_timeline")
TRUE_SCORES = Path("eval/data/urn_runs/true_timeline/scores.jsonl")
TL_SCREEN = Path("eval/data/urn_runs/true_timeline/tl_screen_nocontext.parquet")
# future-modality bound (audit 2026-08-26): regex over-catches ~2.4x (8.4% raw vs
# ~3.5% genuine), so excluding every hit BOUNDS the effect of unverifiable-future
# claims rather than measuring it.
import re
FUTURE_RE = re.compile(r"\b(will|would|plans? to|is set to|are set to|expected to|"
                       r"scheduled to|going to|intends? to|targeting a?)\b", re.I)
E1_RESULTS = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
# The THREE-VOICE fit on the frozen population: the like-for-like baseline for the
# 3-voice de-mix printed below. headline_metrics.json is the shipped 7-flag file.
E1_METRICS = Path("eval/data/urn_runs/e1_ctx/headline_metrics_clustered.json")
# Env-overridable so the measured-eps re-run (timeline_eps_audit.py, 2026-08-28)
# can write its own fit without overwriting the pinned two_urn_fit.json.
OUT = Path(__import__("os").environ.get(
    "TWO_URN_OUT", "eval/data/urn_runs/true_timeline/two_urn_fit_clustered.json"))
VOICES = ("n_t", "n_f", "n_e")   # supports / refutes / silent
FLAGS7 = ("5", "4", "3", "2", "1", "X", "I")   # graded refit: one LR per flag


def _counts(rec: dict) -> tuple[dict, dict]:
    c, fl = collections.Counter(), collections.Counter()
    for d in rec.get("results") or []:
        direction = (d.get("read") or {}).get("direction")
        v = FLAG_TO_VOICE.get(direction)
        if v:
            c["n_t" if v == "supports" else "n_f" if v == "refutes" else "n_e"] += 1
            if direction in FLAGS7:
                fl[direction] += 1
    return c, fl


def load_urn(paths: list[Path], exclusions: Path | None = None,
             population: set[str] | None = None) -> list[dict]:
    """scores.jsonl -> one row per claim with voice counts. Zero-doc claims are
    KEPT as PAD_TO silences (same as fit_urn, Daniel 2026-09-08)."""
    excl = set()
    if exclusions and exclusions.exists():
        excl = {e["claim"][:80] for e in json.loads(exclusions.read_text())}
        # Label-validity exclusion files (media_provenance_purge.py, 2026-08-28).
        # ON by default under the majority rule; EXTRA_EXCLUSIONS=none reproduces
        # the pre-purge numbers. Paths are named in fit_urn, once.
        for x in extra_exclusion_paths():
            excl |= {e["claim"][:80] for e in json.loads(x.read_text())
                     if e.get("excluded", True)}
    rows, n_excl = [], 0
    for p in paths:
        for line in p.open():
            r = json.loads(line)
            claim = r.get("claim_resolved") or r.get("claim_text") or ""
            if claim[:80] in excl:
                n_excl += 1
                continue
            if population is not None and r.get("review_url") not in population:
                continue
            c, fl = _counts(r)
            # Pad rule -- see fit_urn.load. Slots with no read flag count as
            # silent documents, so every claim contributes PAD_TO documents to
            # the urn rates and a claim that read nothing stays in as PAD_TO
            # silences.
            n_pad = max(0, PAD_TO - sum(c.values()))
            c["n_e"] += n_pad
            fl["I"] += n_pad
            rows.append({"n_t": c["n_t"], "n_f": c["n_f"], "n_e": c["n_e"],
                         "flags": dict(fl),
                         "claim": claim, "review_url": r.get("review_url"),
                         "post_id": r.get("post_id") or r.get("review_url"),
                         "frame": r.get("frame"), "topic": r.get("topic")})
    if n_excl:
        print(f"  ({paths[0].parent.name}: {n_excl} excluded via {exclusions.name})")
    return rows


def screen_bad_ids() -> set:
    import polars as pl
    if not TL_SCREEN.exists():
        return set()
    d = pl.read_parquet(TL_SCREEN)
    return set(d.filter(pl.col("verdict") != "ok")["claim_id"].to_list())


def dedup(rows: list[dict]) -> list[dict]:
    """Greedy same-fact collapse: token-set Jaccard >= 0.6 keeps the first claim.
    Approximates the audit's eyeballed 54+24 clusters (−6.9% effective n)."""
    kept, sigs = [], []
    for r in rows:
        toks = frozenset(w for w in re.findall(r"[a-z0-9]+", r["claim"].lower()) if len(w) > 2)
        dup = any(len(toks & s) / max(len(toks | s), 1) >= 0.6 for s in sigs)
        if not dup:
            kept.append(r)
            sigs.append(toks)
    return kept


def rates(rows: list[dict]) -> dict[str, float]:
    tot = sum(r["n_t"] + r["n_f"] + r["n_e"] for r in rows)
    return {k: sum(r[k] for r in rows) / tot for k in VOICES}


def demix_weights(p_mix: dict, p_false: dict, eps: float) -> dict[str, float] | None:
    """w = log(p_TRUE/p_FALSE) with p_TRUE de-mixed from the contaminated rates.
    Returns None if eps is large enough to drive any p_TRUE non-positive."""
    w = {}
    for k in VOICES:
        p_t = (p_mix[k] - eps * p_false[k]) / (1 - eps)
        if p_t <= 0:
            return None
        w[k] = math.log(p_t / p_false[k])
    return w


def flag_rates(rows: list[dict]) -> dict[str, float]:
    tot = sum(sum(r["flags"].values()) for r in rows)
    return {k: sum(r["flags"].get(k, 0) for r in rows) / tot for k in FLAGS7}


def demix_flag_weights(p_mix: dict, p_false: dict, eps: float,
                       laplace_docs: tuple[int, int]) -> dict[str, float] | None:
    """Per-flag log-LR with Laplace smoothing (flag '3' is rare enough to need it)."""
    n_m, n_f = laplace_docs
    w = {}
    for k in FLAGS7:
        pm = (p_mix[k] * n_m + 1) / (n_m + len(FLAGS7))
        pf = (p_false[k] * n_f + 1) / (n_f + len(FLAGS7))
        p_t = (pm - eps * pf) / (1 - eps)
        if p_t <= 0:
            return None
        w[k] = math.log(p_t / pf)
    return w


def score7(flags: dict, w: dict[str, float]) -> float:
    return sum(n * w[k] for k, n in flags.items() if k in w)


def _operating_point(pairs: list[tuple[float, int]], folds: list[int], budget: float) -> dict:
    """Nested (headline) and in-sample threshold choice at one FPR budget."""
    rec_i, fpr_i, thr_i = recall_at_fpr(pairs, budget)
    nest = nested_fixed(pairs, folds, budget)
    return {"recall_at_2pct_fpr": nest["recall"], "fpr": nest["fpr"],
            "thresholds_by_fold": nest["thresholds_by_fold"],
            "threshold_spread": nest["threshold_spread"],
            "insample": {"recall_at_2pct_fpr": rec_i, "fpr": fpr_i, "threshold": thr_i}}


def transfer_eval7(w: dict[str, float], gold: list[dict]) -> dict:
    pairs = [(score7(r["flags"], w), 0 if r["mid"] else r["y"]) for r in gold]
    return {"auc": auc(pairs), **_operating_point(pairs, [r["fold"] for r in gold], 0.02),
            "pairs": pairs}


def threshold_table(pairs: list[tuple[float, int]], folds: list[int],
                    budgets=(0.005, 0.01, 0.02, 0.05)) -> list[dict]:
    out = []
    for b in budgets:
        op = _operating_point(pairs, folds, b)
        rec, fpr = op["recall_at_2pct_fpr"], op["fpr"]
        op["insample"]["lr_plus"] = (op["insample"]["recall_at_2pct_fpr"] / op["insample"]["fpr"]
                                     if op["insample"]["fpr"] > 0 else float("inf"))
        out.append({"fpr_budget": b, "recall": rec, "fpr": fpr,
                    "thresholds_by_fold": op["thresholds_by_fold"],
                    "threshold_spread": op["threshold_spread"],
                    "lr_plus": rec / fpr if fpr > 0 else float("inf"),
                    "insample": op["insample"]})
    return out


def transfer_eval(w: dict[str, float], gold: list[dict]) -> dict:
    """Fixed-weight scoring of the fc-gold E1 population (mixed-as-FALSE)."""
    pairs = [(r["n_t"] * w["n_t"] + r["n_f"] * w["n_f"] + r["n_e"] * w["n_e"],
              0 if r["mid"] else r["y"]) for r in gold]
    return {"auc": auc(pairs), **_operating_point(pairs, [r["fold"] for r in gold], 0.02)}


def bootstrap_weights(false_rows: list[dict], true_rows: list[dict], eps_list: list[float],
                      reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> dict:
    """Per-eps 95% intervals on the de-mixed 3-voice and 7-flag weights.
    Cluster design (headline): claims resampled by post_id within each urn.
    Row design (per claim) beside it; design effect = var cluster / var row.
    A replicate where the de-mix is infeasible at some eps is NaN there."""
    CF3 = np.array([[r[k] for k in VOICES] for r in false_rows], float).T
    CT3 = np.array([[r[k] for k in VOICES] for r in true_rows], float).T
    CF7 = np.array([[r["flags"].get(k, 0) for k in FLAGS7] for r in false_rows], float).T
    CT7 = np.array([[r["flags"].get(k, 0) for k in FLAGS7] for r in true_rows], float).T
    clF, clT = cluster_index(false_rows, "post_id"), cluster_index(true_rows, "post_id")
    zF, zT = np.zeros(len(false_rows), int), np.zeros(len(true_rows), int)
    res = {}
    for design in ("cluster", "row"):
        rng = np.random.default_rng(seed)
        w3 = {e: [] for e in eps_list}; w7 = {e: [] for e in eps_list}
        for _ in range(reps):
            if design == "cluster":
                iF, iT = boot_idx(rng, zF, clF, True), boot_idx(rng, zT, clT, True)
            else:
                iF, iT = rng.choice(len(zF), len(zF)), rng.choice(len(zT), len(zT))
            f3, t3 = CF3[:, iF].sum(1), CT3[:, iT].sum(1)
            f7, t7 = CF7[:, iF].sum(1), CT7[:, iT].sum(1)
            p_f = dict(zip(VOICES, f3 / f3.sum())); p_m = dict(zip(VOICES, t3 / t3.sum()))
            pf7 = dict(zip(FLAGS7, f7 / f7.sum())); pm7 = dict(zip(FLAGS7, t7 / t7.sum()))
            for e in eps_list:
                w = demix_weights(p_m, p_f, e)
                w3[e].append([w[k] for k in VOICES] if w else [np.nan] * 3)
                w = demix_flag_weights(pm7, pf7, e, (int(t7.sum()), int(f7.sum())))
                w7[e].append([w[k] for k in FLAGS7] if w else [np.nan] * 7)
        res[design] = ({e: np.array(v) for e, v in w3.items()},
                       {e: np.array(v) for e, v in w7.items()})

    def _summ(W, Wr, keys):
        return ({k: [float(x) for x in np.nanpercentile(W[:, i], [2.5, 97.5])]
                 for i, k in enumerate(keys)},
                {k: float(np.nanvar(W[:, i]) / np.nanvar(Wr[:, i])) for i, k in enumerate(keys)},
                int(np.isnan(W[:, 0]).sum()))
    out = {"reps": reps, "seed": seed, "design": "post_id within urn",
           "n_posts_false": int(clF.max()) + 1, "n_posts_true": int(clT.max()) + 1, "by_eps": {}}
    for e in eps_list:
        ci3, de3, nan3 = _summ(res["cluster"][0][e], res["row"][0][e], VOICES)
        ci7, de7, nan7 = _summ(res["cluster"][1][e], res["row"][1][e], FLAGS7)
        out["by_eps"][e] = {"weights_ci": ci3, "design_effect": de3, "infeasible_reps": nan3,
                            "weights_ci7": ci7, "design_effect7": de7, "infeasible_reps7": nan7}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--true-frame", choices=["all", "feed", "search_june"], default="all")
    ap.add_argument("--eps", type=float, nargs="*", default=[0.0, 0.03, 0.05, 0.10, 0.15])
    ap.add_argument("--eps-headline", type=float, default=0.10,
                    help="eps the thresholds block is computed at (the weights-table eps)")
    ap.add_argument("--population", type=Path, nargs="*", default=None,
                    help="parquet(s) with claim_id + side (false/true): restrict each urn")
    ap.add_argument("--gold-population", type=Path, default=None,
                    help="parquet with a claim_id column: the fc-gold transfer population")
    ap.add_argument("--gold-keep-media-axis", action="store_true",
                    help="keep media-provenance claims in the transfer population")
    args = ap.parse_args()

    def _pop(side: str) -> set[str] | None:
        if not args.population:
            return None
        out: set[str] = set()
        for p in args.population:
            out |= load_population(p, side) or set()
        return out

    false_rows = load_urn([C2 / "scores.jsonl", C2 / "scores_ext.jsonl"],
                          C2 / "fit_exclusions.json", _pop("false"))
    true_rows = load_urn([TRUE_SCORES], population=_pop("true"))
    if args.true_frame != "all":
        true_rows = [r for r in true_rows if r["frame"] == args.true_frame]
    p_f, p_m = rates(false_rows), rates(true_rows)
    print(f"FALSE urn: {len(false_rows)} claims | rates sup {p_f['n_t']:.3f} "
          f"ref {p_f['n_f']:.3f} sil {p_f['n_e']:.3f}")
    print(f"TRUE  urn ({args.true_frame}): {len(true_rows)} claims | rates sup {p_m['n_t']:.3f} "
          f"ref {p_m['n_f']:.3f} sil {p_m['n_e']:.3f}")
    for fr in sorted({r['frame'] for r in true_rows if r.get('frame')}):
        sub = [r for r in true_rows if r["frame"] == fr]
        pr = rates(sub)
        print(f"    {fr:<12} {len(sub):>5} | sup {pr['n_t']:.3f} ref {pr['n_f']:.3f} "
              f"sil {pr['n_e']:.3f}")

    gold_pop = load_population(args.gold_population)
    gold = (fit_urn.load(E1_RESULTS, None, False, fit_urn.load_judged_axis(), gold_pop)
            if args.gold_keep_media_axis
            else load_headline(E1_RESULTS, population=gold_pop))
    e1 = json.loads(E1_METRICS.read_text())["overall"] if E1_METRICS.exists() else None
    if e1:
        print(f"\nE1 baseline (refit-on-gold, oof): AUC {e1['auc_oof']:.3f} "
              f"recall@2%FPR {e1['recall_at_2pct_fpr']:.3f} | weights "
              + " ".join(f"{k} {v:+.3f}" for k, v in e1["weights"].items()))

    print(f"\nbootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}, post_id clusters + per-claim) ...")
    boot = bootstrap_weights(false_rows, true_rows, args.eps)
    print(f"\n{'eps':>5} {'w_sup':>8} {'w_ref':>8} {'w_sil':>8}  (95% cluster CI, design effect) "
          f"| fc-gold transfer (n={len(gold)}): AUC  recall@2% nested (FPR) / in-sample")
    out = {"true_frame": args.true_frame, "n_false": len(false_rows),
           "n_true": len(true_rows), "rates_false": p_f, "rates_mix": p_m,
           "population": [str(p) for p in args.population] if args.population else None,
           "gold_population": str(args.gold_population) if args.gold_population else None,
           "gold_keep_media_axis": args.gold_keep_media_axis, "n_gold": len(gold),
           "bootstrap": {k: v for k, v in boot.items() if k != "by_eps"}, "fits": []}
    for eps in args.eps:
        w = demix_weights(p_m, p_f, eps)
        if w is None:
            print(f"{eps:>5.2f}  de-mix infeasible (p_TRUE <= 0 for some voice)")
            continue
        ev = transfer_eval(w, gold)
        b = boot["by_eps"][eps]
        print(f"{eps:>5.2f} {w['n_t']:>8.3f} {w['n_f']:>8.3f} {w['n_e']:>8.3f}  "
              + " ".join(f"[{b['weights_ci'][k][0]:+.2f},{b['weights_ci'][k][1]:+.2f}]"
                         f" d{b['design_effect'][k]:.1f}" for k in VOICES)
              + f" | {ev['auc']:.3f}  {ev['recall_at_2pct_fpr']:.3f} ({ev['fpr']:.3f}) / "
              f"{ev['insample']['recall_at_2pct_fpr']:.3f}")
        out["fits"].append({"eps": eps, "weights": w, "weights_ci": b["weights_ci"],
                            "design_effect_weights": b["design_effect"],
                            "fc_gold_transfer": ev})

    # Sensitivity cuts on the TRUE side (audit 2026-08-26), all at eps=0.05:
    bad = screen_bad_ids()
    cuts = {
        "base": true_rows,
        "-screen_nonok": [r for r in true_rows if r["review_url"] not in bad],
        "-future_regex": [r for r in true_rows if not FUTURE_RE.search(r["claim"])],
        "-dedup": dedup(true_rows),
    }
    cuts["-all"] = dedup([r for r in cuts["-screen_nonok"] if not FUTURE_RE.search(r["claim"])])
    print(f"\nsensitivity (eps=0.05): cut        n_true   w_sup   w_ref   w_sil   AUC  rec@2%")
    out["sensitivity"] = []
    for name, rows_c in cuts.items():
        if not rows_c:
            continue
        w = demix_weights(rates(rows_c), p_f, 0.05)
        if w is None:
            continue
        ev = transfer_eval(w, gold)
        print(f"{'':>21}{name:<12} {len(rows_c):>5} {w['n_t']:>7.3f} {w['n_f']:>7.3f} "
              f"{w['n_e']:>7.3f}  {ev['auc']:.3f}  {ev['recall_at_2pct_fpr']:.3f}")
        out["sensitivity"].append({"cut": name, "n_true": len(rows_c), "weights": w,
                                   "fc_gold_transfer": ev})
    # ---- 7-flag graded fit (the Phase-3 refit: one LR per flag) ----
    pf7, pm7 = flag_rates(false_rows), flag_rates(true_rows)
    docs_m = sum(sum(r["flags"].values()) for r in true_rows)
    docs_f = sum(sum(r["flags"].values()) for r in false_rows)
    print(f"\n7-flag rates (docs TRUE {docs_m} / FALSE {docs_f}):")
    print("  flag   p_mix   p_false")
    for k in FLAGS7:
        print(f"   {k}    {pm7[k]:.4f}  {pf7[k]:.4f}")
    print(f"\n{'eps':>5} " + " ".join(f"w_{k:>2}" for k in FLAGS7)
          + " | fc-gold: AUC  rec@2% nested / in-sample")
    out["fits7"] = []
    for eps in args.eps:
        w7 = demix_flag_weights(pm7, pf7, eps, (docs_m, docs_f))
        if w7 is None:
            print(f"{eps:>5.2f}  de-mix infeasible")
            continue
        ev = transfer_eval7(w7, gold)
        b = boot["by_eps"][eps]
        print(f"{eps:>5.2f} " + " ".join(f"{w7[k]:+.2f}" for k in FLAGS7)
              + f" |          {ev['auc']:.3f}  {ev['recall_at_2pct_fpr']:.3f} / "
              f"{ev['insample']['recall_at_2pct_fpr']:.3f}")
        print(f"{'':>5} " + " ".join(f"[{b['weights_ci7'][k][0]:+.2f},{b['weights_ci7'][k][1]:+.2f}]"
                                     for k in FLAGS7))
        out["fits7"].append({"eps": eps, "weights": w7, "weights_ci": b["weights_ci7"],
                             "design_effect_weights": b["design_effect7"],
                             "fc_gold_transfer": {k: v for k, v in ev.items() if k != "pairs"}})

    # ---- threshold operating points, 3-voice vs 7-flag, at the headline eps ----
    eps_h = args.eps_headline
    w3 = demix_weights(p_m, p_f, eps_h)
    pairs3 = [(r["n_t"] * w3["n_t"] + r["n_f"] * w3["n_f"] + r["n_e"] * w3["n_e"],
               0 if r["mid"] else r["y"]) for r in gold]
    w7 = demix_flag_weights(pm7, pf7, eps_h, (docs_m, docs_f))
    pairs7 = transfer_eval7(w7, gold)["pairs"]
    folds = [r["fold"] for r in gold]
    print(f"\nthreshold operating points on fc-gold (mixed-as-FALSE, eps={eps_h}), "
          f"nested over cluster-disjoint folds; in-sample in parentheses:")
    print(f"{'model':>8} {'FPR<=':>7} {'recall':>7} {'FPR':>7} {'LR+':>6} {'thr by fold (min..max)':>24}")
    out["thresholds"] = {"eps": eps_h}
    for name, pairs in (("3-voice", pairs3), ("7-flag", pairs7)):
        rows_t = threshold_table(pairs, folds)
        out["thresholds"][name] = rows_t
        for t in rows_t:
            sp = t["threshold_spread"]
            print(f"{name:>8} {t['fpr_budget']:>6.1%} {t['recall']:>7.3f} "
                  f"{t['fpr']:>7.3f} {t['lr_plus']:>6.1f} {sp['min']:>10.3f}..{sp['max']:.3f}"
                  f"   ({t['insample']['recall_at_2pct_fpr']:.3f} at thr {t['insample']['threshold']:.3f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
