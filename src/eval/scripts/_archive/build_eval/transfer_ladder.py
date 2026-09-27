"""The same bucket ladder, but fitted on the two urns and evaluated on fc gold.

    uv run python -m eval.scripts.build_eval.transfer_ladder

$0 -- reuses saved reads on all three corpora. Companion to model_ladder.py,
which fits and evaluates on fc gold. Here NO gold label touches the fit.

Fit. FALSE = the CN-false urn (C2 + EXT, band-false, fit-eligible, minus
fit_exclusions.json). TRUE = the raw timeline urn, which is a MIXED draw whose
false fraction is eps, de-mixed at the rate level exactly as fit_two_urn does

    p_mix = (1-eps) p_TRUE + eps p_FALSE   =>   p_TRUE = (p_mix - eps p_FALSE)/(1-eps)

and w = log(p_TRUE / p_FALSE). Laplace +1 per bucket and +K on each urn's total
before de-mixing, so a thin bucket cannot send a weight to infinity. Unshrunk at
every level, same as model_ladder.

Eval. The fitted weights score the PINNED fc-gold population from model_ladder
(media axis out, rating_subtype=mixed out, n=3,274, unprovable counted FALSE)
with no refit. Fixed-weight transfer, so every gold claim is out of sample and
folds are unnecessary.

Intervals. 2,000-rep bootstrap that resamples all three corpora inside each
replicate -- the FALSE urn's claims, the timeline urn's claims, and the gold
claims within their gold class. The interval therefore carries both the
uncertainty in the fitted weights and the uncertainty in the eval set. Paired
across models by shared replicates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval import fit_urn
from eval.scripts.build_eval import model_ladder as ML
from eval.scripts.build_eval.graded_urn import FLAGS

C2 = Path("eval/data/urn_runs/c2_false")
TL = Path("eval/data/urn_runs/true_timeline")
# Env-overridable so a label-validity refit (media_provenance_purge.py, 2026-08-28)
# can write its own fits without overwriting the pinned ones. Defaults unchanged.
_ENV = __import__("os").environ
OUT = Path(_ENV.get("LADDER_OUT", "eval/data/urn_runs/e1_ctx/model_ladder"))
# Label-validity exclusion files on top of fit_exclusions.json, same
# {"claim": ...} shape plus an "excluded" boolean the aggregation rule set.
# ON by default (media-provenance purge, majority rule, Daniel 2026-08-28);
# EXTRA_EXCLUSIONS=none turns them off. Paths live in fit_urn, once.
EPS_SWEEP = (0.0, 0.05, 0.10, 0.15)
EPS_HEAD = 0.10
BOOT_REPS, BOOT_SEED = 2000, 707
MODELS = ML.MODELS


def load_urn(paths: list[Path], exclusions: Path | None = None) -> list[dict]:
    """One row per claim keeping each document's (flag, tier). Zero-doc claims
    are dropped, matching fit_urn and fit_two_urn."""
    excl = set()
    if exclusions and exclusions.exists():
        excl = {e["claim"][:80] for e in json.loads(exclusions.read_text())}
    if exclusions is not None:      # only the corpus the exclusion files describe
        for x in fit_urn.extra_exclusion_paths():
            add = {e["claim"][:80] for e in json.loads(x.read_text())
                   if e.get("excluded", True)}
            print(f"  (+{len(add)} extra exclusions from {x.name})")
            excl |= add
    rows, n_excl = [], 0
    for p in paths:
        for line in p.open():
            r = json.loads(line)
            claim = r.get("claim_resolved") or r.get("claim_text") or ""
            if claim[:80] in excl:
                n_excl += 1
                continue
            docs = [(d["read"]["direction"], d.get("rel") or "UNRATED")
                    for d in r.get("results") or []
                    if (d.get("read") or {}).get("direction") in FLAGS]
            if not docs:
                continue
            rows.append({"docs": docs, "frame": r.get("frame"), "claim": claim})
    if n_excl:
        print(f"  ({paths[0].parent.name}: {n_excl} excluded via fit_exclusions.json)")
    return rows


def urn_matrix(rows: list[dict], model: str) -> np.ndarray:
    """buckets x claims document counts, in model_ladder's channel order."""
    chans = ML.channels(model)
    idx = {c: i for i, c in enumerate(chans)}
    C = np.zeros((len(chans), len(rows)))
    for j, r in enumerate(rows):
        for fl, tr in r["docs"]:
            C[idx[ML.KEYS[model](fl, tr)], j] += 1
    return C


def demix(cF: np.ndarray, cM: np.ndarray, eps: float) -> np.ndarray | None:
    """Laplace-smoothed rates on each urn, then de-mix the TRUE side.
    Returns None if eps drives any bucket's implied p_TRUE non-positive."""
    K = len(cF)
    pF = (cF + 1) / (cF.sum() + K)
    pM = (cM + 1) / (cM.sum() + K)
    pT = (pM - eps * pF) / (1 - eps)
    if (pT <= 0).any():
        return None
    return np.log(pT / pF)


def main() -> None:
    false_rows = load_urn([C2 / "scores.jsonl", C2 / "scores_ext.jsonl"],
                          C2 / "fit_exclusions.json")
    true_rows = load_urn([TL / "scores.jsonl"])

    gold = [r for r in ML.load_docs() if r["subtype"] != ML.SET_ASIDE]
    y = np.array([r["y"] for r in gold])
    ref = json.loads((OUT / "ladder.json").read_text())["models"]

    print(f"FALSE urn {len(false_rows):,} claims / "
          f"{sum(len(r['docs']) for r in false_rows):,} documents")
    print(f"TIMELINE urn {len(true_rows):,} claims / "
          f"{sum(len(r['docs']) for r in true_rows):,} documents")
    print(f"fc-gold eval {len(gold):,} claims (T {int(y.sum())} / F {int((y == 0).sum())}), "
          f"no refit\n")

    CF = {m: urn_matrix(false_rows, m) for m in MODELS}
    CM = {m: urn_matrix(true_rows, m) for m in MODELS}
    CG = {m: ML.count_matrix(gold, m) for m in MODELS}

    # ---- eps sweep -----------------------------------------------------------
    print(f"{'eps':>5}  " + "  ".join(f"{m:>22}" for m in MODELS))
    sweep = {}
    for eps in EPS_SWEEP:
        cells = []
        for m in MODELS:
            w = demix(CF[m].sum(1), CM[m].sum(1), eps)
            if w is None:
                cells.append("de-mix infeasible")
                sweep.setdefault(eps, {})[m] = None
                continue
            s = w @ CG[m]
            a = ML.auc_np(s[y == 1], s[y == 0])
            r2, f2, t2 = ML.recall_at_fpr(s, y, 0.02)
            entry = {"auc": a, "recall_2pct": r2, "fpr_2pct": f2, "threshold_2pct": t2,
                     "weights": dict(zip(ML.channels(m), map(float, w)))}
            if eps == EPS_HEAD:
                entry["roc"] = [x.tolist() for x in ML.roc(s, y)]
            sweep.setdefault(eps, {})[m] = entry
            cells.append(f"AUC {a:.4f} rec {r2*100:4.1f}%")
        print(f"{eps:>5.2f}  " + "  ".join(f"{c:>22}" for c in cells))

    # ---- bootstrap at the headline eps --------------------------------------
    print(f"\nbootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}, all three corpora resampled) ...")
    rng = np.random.default_rng(BOOT_SEED)
    nF, nM = len(false_rows), len(true_rows)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    a_reps = {m: [] for m in MODELS}
    w_reps = {m: [] for m in MODELS}
    n_infeasible = {m: 0 for m in MODELS}
    for _ in range(BOOT_REPS):
        iF = rng.integers(0, nF, nF)
        iM = rng.integers(0, nM, nM)
        iG = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy = y[iG]
        for m in MODELS:
            w = demix(CF[m][:, iF].sum(1), CM[m][:, iM].sum(1), EPS_HEAD)
            if w is None:      # keep the slot so replicates stay paired across models
                n_infeasible[m] += 1
                a_reps[m].append(np.nan)
                continue
            w_reps[m].append(w)
            s = w @ CG[m][:, iG]
            a_reps[m].append(ML.auc_np(s[yy == 1], s[yy == 0]))

    res = {}
    for m in MODELS:
        h = sweep[EPS_HEAD][m]
        lo, hi = np.nanpercentile(a_reps[m], [2.5, 97.5])
        W = np.array(w_reps[m])
        wlo, whi = np.percentile(W, [2.5, 97.5], axis=0)
        res[m] = dict(h)
        res[m]["auc_ci"] = [float(lo), float(hi)]
        res[m]["weight_ci"] = {c: [float(a), float(b)]
                               for c, a, b in zip(ML.channels(m), wlo, whi)}
        res[m]["gold_weights"] = ref[m]["weights"]
        res[m]["gold_refit_auc"] = ref[m]["auc_oof"]
        res[m]["gold_refit_recall_2pct"] = ref[m]["recall_2pct"]
        res[m]["boot_infeasible"] = n_infeasible[m]
        print(f"\n{m}  transfer AUC {h['auc']:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"recall@2% {h['recall_2pct']*100:.1f}%"
              f"   (refit on gold {ref[m]['auc_oof']:.4f} / {ref[m]['recall_2pct']*100:.1f}%)")
        if m != "3-voice":
            d = np.array(a_reps[m]) - np.array(a_reps["3-voice"])
            dlo, dhi = np.nanpercentile(d, [2.5, 97.5])
            res[m]["dauc_vs_3voice"] = h["auc"] - sweep[EPS_HEAD]["3-voice"]["auc"]
            res[m]["dauc_ci"] = [float(dlo), float(dhi)]
            print(f"  dAUC vs 3-voice {res[m]['dauc_vs_3voice']:+.4f} "
                  f"[{dlo:+.4f}, {dhi:+.4f}]")
        if n_infeasible[m]:
            print(f"  {n_infeasible[m]} of {BOOT_REPS} replicates dropped, de-mix infeasible")
        for c in ML.channels(m):
            g = ref[m]["weights"][c]
            print(f"  {c:22s} {h['weights'][c]:+.3f}  "
                  f"[{res[m]['weight_ci'][c][0]:+.3f}, {res[m]['weight_ci'][c][1]:+.3f}]"
                  f"   gold-fitted {g:+.3f}")

    payload = {"eps_head": EPS_HEAD, "eps_sweep": {str(k): v for k, v in sweep.items()},
               "boot": {"reps": BOOT_REPS, "seed": BOOT_SEED},
               "urns": {"false_claims": len(false_rows),
                        "false_docs": int(CF["7-flag"].sum()),
                        "true_claims": len(true_rows),
                        "true_docs": int(CM["7-flag"].sum())},
               "gold": {"n": len(gold), "n_true": int(y.sum()),
                        "n_false": int((y == 0).sum())},
               "models": res}
    (OUT / "transfer.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT / 'transfer.json'}")


if __name__ == "__main__":
    main()
