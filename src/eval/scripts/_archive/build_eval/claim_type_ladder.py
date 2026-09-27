"""Is CLAIM TYPE a useful stratifier -- diagnostically, and as a scoring change?

    uv run python -m eval.scripts.build_eval.claim_type_ladder                  # ARM 1
    uv run python -m eval.scripts.build_eval.claim_type_ladder --stratifier claim_side

$0 -- refits the saved reads. Two arms, kept strictly apart because they answer
different questions and only one of them is production-legal.

ARM 1, DIAGNOSTIC (`--stratifier judged`). The stratum is `judged_axis`, the
proposition the FACT-CHECKER settled (derive_judged_axis.py). It is derived from
the verdict, so it does NOT exist at production time and can never be a feature.
What it CAN do is answer "which kinds of claim does our instrument actually work
on", which nothing has asked yet. Population: model_ladder's pinned cut with
media_authenticity ADDED BACK as its own stratum (n = 3,274 + 294 = 3,568), so
the excluded stratum's behaviour is visible rather than assumed.

ARM 2, PRODUCTION-LEGAL (`--stratifier claim_side`). The stratum is
`claim_type_side.py`'s label-blind assignment from claim text + date only, which
a deployed system could compute. Same arms, same control, plus the gold-to-urn
transfer (judged_axis has no transfer arm: the urns carry no verdict, so the
axis cannot be derived for them at all).

ARMS. All share model_ladder's out-of-fold protocol, its Laplace convention
(+1 per bucket, +K on each class total, unshrunk, here WITHIN stratum) and its
fit_w / oof_scores / auc_np / recall_at_fpr / fold_of machinery.
  global      the published baseline: one weight vector, model_ladder exactly
  full        every bucket refitted per stratum
  silence     directional weights global, SILENT buckets per stratum
  intercept   CONTROL. Global weights + a per-stratum constant (the claim-level
              log-LR of stratum membership). The reportability run (2026-08-28)
              found the intercept beating the stratified weights outright; if
              that repeats here the "gain" is prior-encoding, not evidence
              behaviour. stratified-vs-intercept is the deliverable, NOT
              stratified-vs-global.
  full_int / silence_int   the stratified arms carrying the same intercept, so
              the only remaining difference is the mechanism.

Intervals. 2,000-rep paired bootstrap, seed 707, resampled within gold class
(and over both urns for the transfer cells). Writes eval/data/claim_type/.
Nothing under eval/data/urn_runs/ is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval import fit_urn
from eval.scripts.build_eval import model_ladder as ML
from eval.scripts.build_eval import transfer_ladder as TLD
from eval.scripts.build_eval.graded_urn import FLAGS

OUT = Path("eval/data/claim_type")
E1 = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
# Gold display order = descending frequency on the pinned cut.
STRATA = ("event_occurrence", "quote_attribution", "statistic_figure",
          "policy_law", "media_authenticity", "causal_effect", "attribute_identity")
MODELS = ("3-voice", "7-flag")
SILENT = {"3-voice": ("silent",), "7-flag": ("3", "X", "I")}
ARM_SPEC = {"global": ("global", False), "full": ("full", False),
            "silence": ("silence", False), "intercept": ("global", True),
            "full_int": ("full", True), "silence_int": ("silence", True)}
ARMS = tuple(ARM_SPEC)
BOOT_REPS = int(os.environ.get("REPS", 2000))
BOOT_SEED = 707
K = fit_urn.K_FOLDS
G = len(STRATA)
S = {s: i for i, s in enumerate(STRATA)}
EPS_HEAD = TLD.EPS_HEAD


# ------------------------------------------------------------------ population
def gold_docs(keep_media: bool = True) -> list[dict]:
    """model_ladder.load_docs, but media_authenticity kept and tagged.

    Asserted in main(): dropping the media rows reproduces model_ladder's cut
    row-for-row, so the pinned instrument is unchanged by the addition."""
    axis = fit_urn.load_judged_axis()
    rows = []
    for line in E1.open():
        r = json.loads(line)
        if r.get("veracity") not in (1, 2, 3, 4, 5):
            continue
        docs = [(d["read"]["direction"], d.get("rel") or "UNRATED")
                for d in r.get("results") or []
                if (d.get("read") or {}).get("direction") in FLAGS]
        if not docs:
            continue
        ax = axis.get(r["review_url"]) or "untagged"
        if not keep_media and ax == fit_urn.MEDIA_AXIS:
            continue
        if (r.get("rating_subtype") or "?") == ML.SET_ASIDE:
            continue
        rows.append({"y": 1 if r["veracity"] >= 4 else 0, "mid": r["veracity"] == 3,
                     "docs": docs, "judged_axis": ax,
                     "key": f"gold:{r['review_url']}",
                     "fold": fit_urn.fold_of(r["review_url"], K)})
    return rows


def load_side_labels() -> dict[str, str]:
    p = OUT / "labels.jsonl"
    lab = {}
    for line in p.open():
        r = json.loads(line)
        lab[r["key"]] = r["claim_type_side"]
    return lab


# ------------------------------------------------------------------ metrics
def recall_at_fpr(s: np.ndarray, y: np.ndarray, budget: float) -> float:
    trues, falses = np.sort(s[y == 1]), np.sort(s[y == 0])
    if not len(trues) or not len(falses):
        return float("nan")
    thr = np.unique(s)
    fpr = np.searchsorted(trues, thr, side="right") / len(trues)
    rec = np.searchsorted(falses, thr, side="right") / len(falses)
    ok = fpr <= budget
    return float(rec[ok].max()) if ok.any() else 0.0


def safe_auc(s: np.ndarray, y: np.ndarray) -> float:
    p, n = s[y == 1], s[y == 0]
    return ML.auc_np(p, n) if len(p) and len(n) else float("nan")


def ci(v) -> list[float]:
    v = np.asarray(v, float)
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])] if np.isfinite(v).any() \
        else [float("nan"), float("nan")]


# ------------------------------------------------------------------ arms
def intercept(y, mid, strat, idx) -> np.ndarray:
    """Claim-level log-LR of stratum membership, Laplace +1 / +G. The control."""
    keep = idx[~mid[idx]]
    nT = np.array([((strat[keep] == g) & (y[keep] == 1)).sum() for g in range(G)])
    nF = np.array([((strat[keep] == g) & (y[keep] == 0)).sum() for g in range(G)])
    return np.log(((nT + 1) / (nT.sum() + G)) / ((nF + 1) / (nF.sum() + G)))


def arm_matrix(w, per, sil_idx, base) -> np.ndarray:
    if base == "full":
        return per
    W = np.repeat(w[:, None], G, 1)
    if base == "silence":
        W[sil_idx, :] = per[sil_idx, :]
    return W


def oof_all(C, y, mid, fold, strat, sil, fit_mask=None) -> dict[str, np.ndarray]:
    """Out-of-fold scores for EVERY arm in one pass (weights fitted once per fold).

    fit_mask restricts which rows may enter a fit -- used to hold the global arm
    to the pinned population while still scoring the media stratum out of sample.
    """
    allidx = np.arange(C.shape[1])
    s = {a: np.empty(C.shape[1]) for a in ARMS}
    for k in range(K):
        tr = allidx[(fold != k) if fit_mask is None else ((fold != k) & fit_mask)]
        te = allidx[fold == k]
        w = ML.fit_w(C, y, mid, tr)
        per = np.stack([ML.fit_w(C, y, mid, tr[strat[tr] == g]) for g in range(G)], 1)
        b = intercept(y, mid, strat, tr)
        for arm, (base, use_int) in ARM_SPEC.items():
            W = arm_matrix(w, per, sil, base)
            v = (W[:, strat[te]] * C[:, te]).sum(0)
            s[arm][te] = v + b[strat[te]] if use_int else v
    return s


# ------------------------------------------------------------------ transfer
def urn_weights(CF, CM, sF, sM, eps):
    w = TLD.demix(CF.sum(1), CM.sum(1), eps)
    if w is None:
        return None, None
    per = []
    for g in range(G):
        wg = TLD.demix(CF[:, sF == g].sum(1), CM[:, sM == g].sum(1), eps)
        per.append(w if wg is None else wg)      # thin stratum falls back to global
    return w, np.stack(per, 1)


def urn_intercept(sF, sM, eps) -> np.ndarray:
    pF = np.array([(sF == g).sum() for g in range(G)], float)
    pM = np.array([(sM == g).sum() for g in range(G)], float)
    pF = (pF + 1) / (pF.sum() + G)
    pM = (pM + 1) / (pM.sum() + G)
    pT = (pM - eps * pF) / (1 - eps)
    return np.log(np.maximum(pT, 1e-9) / pF)


def transfer_all(CG, gstrat, CF, CM, sF, sM, eps, sil):
    w, per = urn_weights(CF, CM, sF, sM, eps)
    if w is None:
        return None
    b = urn_intercept(sF, sM, eps)
    out = {}
    for arm, (base, use_int) in ARM_SPEC.items():
        W = arm_matrix(w, per, sil, base)
        v = (W[:, gstrat] * CG).sum(0)
        out[arm] = v + b[gstrat] if use_int else v
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratifier", choices=["judged", "claim_side"], default="judged")
    ap.add_argument("--population", choices=["all", "pinned"], default="all",
                    help="all = the 3,568 cut with media_authenticity added back as its "
                         "own stratum; pinned = model_ladder's 3,274. Run BOTH: media is "
                         "96%% false, so on `all` the intercept control has a lever that "
                         "is an artefact of the exclusion policy, not of claim type.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.stratifier}_{args.population}"

    pinned = gold_docs(keep_media=False)
    gold = gold_docs(keep_media=True) if args.population == "all" else pinned
    n_media = len(gold) - len(pinned)
    assert len(pinned) == len([r for r in ML.load_docs() if r["subtype"] != ML.SET_ASIDE])
    print(f"population {args.population}: gold {len(gold)} "
          f"(pinned {len(pinned)} + media_authenticity {n_media})")

    if args.stratifier == "judged":
        lab = {r["key"]: r["judged_axis"] for r in gold}
    else:
        lab = load_side_labels()
    strat = np.array([S[lab[r["key"]]] for r in gold])
    is_media_axis = np.array([r["judged_axis"] == fit_urn.MEDIA_AXIS for r in gold])

    # ---- (a) is the production-legal proxy any good? -----------------------
    # Confusion against judged_axis. Only meaningful for the claim-side arm;
    # judged_axis against itself is the identity by construction.
    agree = {}
    if args.stratifier == "claim_side":
        cm = np.zeros((G, G), int)
        for r in gold:
            cm[S[r["judged_axis"]], S[lab[r["key"]]]] += 1
        acc = float(np.trace(cm) / cm.sum())
        print(f"\n(a) claim-side vs judged_axis on gold: overall accuracy "
              f"{acc:.3f} ({int(np.trace(cm))}/{cm.sum()})")
        hdr = "judged_axis vs claim-side"
        print(f"  {hdr:26s}"
              + "".join(f"{s[:6]:>7s}" for s in STRATA)
              + f"{'n':>7}{'recall':>8}{'prec':>7}")
        for i, name in enumerate(STRATA):
            n_i, n_p = cm[i].sum(), cm[:, i].sum()
            rec = cm[i, i] / n_i if n_i else float("nan")
            pre = cm[i, i] / n_p if n_p else float("nan")
            agree[name] = {"n_judged": int(n_i), "n_side": int(n_p),
                           "recall": float(rec), "precision": float(pre),
                           "row": {STRATA[j]: int(cm[i, j]) for j in range(G)}}
            print(f"  {name:26s}" + "".join(f"{cm[i, j]:7d}" for j in range(G))
                  + f"{n_i:7d}{rec:8.3f}{pre:7.3f}")
        agree["_overall"] = {"accuracy": acc, "n": int(cm.sum())}

    y = np.array([r["y"] for r in gold])
    mid = np.array([r["mid"] for r in gold])
    fold = np.array([r["fold"] for r in gold])
    C = {m: ML.count_matrix(gold, m) for m in MODELS}
    sil = {m: np.array([i for i, c in enumerate(ML.channels(m)) if c in SILENT[m]])
           for m in MODELS}

    # ---- (0) the pinned instrument, scored on all 3,568 rows ----------------
    # Global weights are fitted on the pinned population only (media never enters
    # a fit), so `global` here IS the shipped instrument; media rows are scored
    # out of sample by construction.
    pin = ~is_media_axis
    s_pin = {m: oof_all(C[m], y, mid, fold, strat, sil[m], fit_mask=pin)["global"]
             for m in MODELS}
    ref = ML.oof_scores(C["7-flag"][:, pin], y[pin], mid[pin], fold[pin])
    assert np.allclose(s_pin["7-flag"][pin], ref), "global arm is not model_ladder"
    a_pin = safe_auc(s_pin["7-flag"][pin], y[pin])
    r_pin = recall_at_fpr(s_pin["7-flag"][pin], y[pin], 0.02)
    assert abs(a_pin - 0.8620) < 5e-4 and abs(r_pin - 0.422) < 5e-3, (a_pin, r_pin)
    print(f"pinned instrument reproduced on the 3,274 cut: 7-flag AUC {a_pin:.4f} "
          f"/ recall@2% {r_pin:.4f}")

    # ---- (1) THE DIAGNOSTIC TABLE -------------------------------------------
    C7 = C["7-flag"]
    ch7 = ML.channels("7-flag")
    sil_rows = sil["7-flag"]
    i_irr = ch7.index("I")
    tot = C7.sum(0)
    smass = C7[sil_rows, :].sum(0)
    diag = {}
    print(f"\n(1) DIAGNOSTIC: the pinned global 7-flag weights, evaluated WITHIN "
          f"each {args.stratifier} stratum")
    print(f"  {'stratum':20s} {'n':>5} {'P(FALSE)':>9} {'AUC':>7} {'rec@2%':>8} "
          f"{'silent doc':>11} {'all-silent':>11} {'all-irrel':>10}")
    order = []
    for g, name in enumerate(STRATA):
        m_ = strat == g
        if not m_.any():
            continue
        a = safe_auc(s_pin["7-flag"][m_], y[m_])
        d = {"n": int(m_.sum()), "n_true": int(y[m_].sum()),
             "p_false": float((y[m_] == 0).mean()),
             "auc": a, "recall_2pct": recall_at_fpr(s_pin["7-flag"][m_], y[m_], 0.02),
             "silent_doc_share": float(smass[m_].sum() / tot[m_].sum()),
             "all_silent_share": float(((smass[m_] == tot[m_])).mean()),
             "all_irrelevant_share": float((C7[i_irr, m_] == tot[m_]).mean()),
             "docs": int(tot[m_].sum())}
        diag[name] = d
        order.append((a, name))
    for a, name in sorted(order, reverse=True):
        d = diag[name]
        print(f"  {name:20s} {d['n']:5d} {d['p_false']:9.3f} {d['auc']:7.4f} "
              f"{d['recall_2pct']*100:7.1f}% {d['silent_doc_share']*100:10.1f}% "
              f"{d['all_silent_share']*100:10.1f}% {d['all_irrelevant_share']*100:9.1f}%")
    a_all = safe_auc(s_pin["7-flag"], y)
    print(f"  {'ALL':20s} {len(gold):5d} {float((y == 0).mean()):9.3f} "
          f"{a_all:7.4f} {recall_at_fpr(s_pin['7-flag'], y, 0.02)*100:7.1f}% "
          f"{smass.sum()/tot.sum()*100:10.1f}% {float((smass == tot).mean())*100:10.1f}% "
          f"{float((C7[i_irr] == tot).mean())*100:9.1f}%")

    # ---- (2) arm point estimates on the full 3,568 --------------------------
    res = {}
    scores = {}
    for m in MODELS:
        scores[m] = oof_all(C[m], y, mid, fold, strat, sil[m])
        for arm in ARMS:
            s = scores[m][arm]
            res.setdefault(m, {})[arm] = {
                "auc": safe_auc(s, y), "recall_2pct": recall_at_fpr(s, y, 0.02),
                "by_stratum": {name: {"auc": safe_auc(s[strat == g], y[strat == g])}
                               for g, name in enumerate(STRATA) if (strat == g).any()}}

    # ---- (3) transfer, claim-side only --------------------------------------
    tres = {}
    if args.stratifier == "claim_side":
        cn = TLD.load_urn([TLD.C2 / "scores.jsonl", TLD.C2 / "scores_ext.jsonl"],
                          TLD.C2 / "fit_exclusions.json")
        tl = TLD.load_urn([TLD.TL / "scores.jsonl"])
        import eval.scripts.build_eval.claim_type_side as CTS
        ckeys = CTS.urn_rows("cn", [CTS.C2 / "scores.jsonl", CTS.C2 / "scores_ext.jsonl"],
                             CTS.C2 / "fit_exclusions.json")
        tkeys = CTS.urn_rows("tl", [CTS.TL / "scores.jsonl"], None)
        assert [a["claim"] for a in cn] == [b["claim"] for b in ckeys]
        assert [a["claim"] for a in tl] == [b["claim"] for b in tkeys]
        sF = np.array([S[lab[r["key"]]] for r in ckeys])
        sM = np.array([S[lab[r["key"]]] for r in tkeys])
        CF = {m: TLD.urn_matrix(cn, m) for m in MODELS}
        CM = {m: TLD.urn_matrix(tl, m) for m in MODELS}
        print(f"\ntransfer corpora: CN-false {len(cn)} / timeline {len(tl)}")
        print(f"  {'stratum':20s} {'gold %':>7} {'CN %':>7} {'TL %':>7}")
        mix = {}
        for g, name in enumerate(STRATA):
            mix[name] = {"gold": float((strat == g).mean()),
                         "cn": float((sF == g).mean()), "tl": float((sM == g).mean())}
            print(f"  {name:20s} {mix[name]['gold']*100:6.1f}% {mix[name]['cn']*100:6.1f}%"
                  f" {mix[name]['tl']*100:6.1f}%")
        for m in MODELS:
            t = transfer_all(C[m], strat, CF[m], CM[m], sF, sM, EPS_HEAD, sil[m])
            for arm in ARMS:
                tres.setdefault(m, {})[arm] = None if t is None else {
                    "auc": safe_auc(t[arm], y), "recall_2pct": recall_at_fpr(t[arm], y, 0.02)}

    # ---- (4) paired bootstrap ----------------------------------------------
    print(f"\nbootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}) ...", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    a_rep = {m: {a: [] for a in ARMS} for m in MODELS}
    r_rep = {m: {a: [] for a in ARMS} for m in MODELS}
    ta_rep = {m: {a: [] for a in ARMS} for m in MODELS}
    st_rep = {m: {a: {n: [] for n in STRATA} for a in ARMS} for m in MODELS}
    xfer = args.stratifier == "claim_side"
    nF_ = len(cn) if xfer else 0
    nM_ = len(tl) if xfer else 0
    for rep in range(BOOT_REPS):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        yy, mm, ff, ss = y[idx], mid[idx], fold[idx], strat[idx]
        if xfer:
            iF, iM = rng.integers(0, nF_, nF_), rng.integers(0, nM_, nM_)
        for m in MODELS:
            Cm = C[m][:, idx]
            sc = oof_all(Cm, yy, mm, ff, ss, sil[m])
            for arm in ARMS:
                a_rep[m][arm].append(safe_auc(sc[arm], yy))
                r_rep[m][arm].append(recall_at_fpr(sc[arm], yy, 0.02))
                for g, name in enumerate(STRATA):
                    sel = ss == g
                    st_rep[m][arm][name].append(
                        safe_auc(sc[arm][sel], yy[sel]) if sel.any() else np.nan)
            if xfer:
                t = transfer_all(Cm, ss, CF[m][:, iF], CM[m][:, iM],
                                 sF[iF], sM[iM], EPS_HEAD, sil[m])
                for arm in ARMS:
                    ta_rep[m][arm].append(np.nan if t is None else safe_auc(t[arm], yy))
        if (rep + 1) % 250 == 0:
            print(f"  {rep+1}/{BOOT_REPS}", flush=True)

    # ---- (5) report ---------------------------------------------------------
    print(f"\n(2) ARMS on gold, out of fold, n={len(gold)}. The deliverable is the "
          "MINUS-intercept row.")
    for m in MODELS:
        print(f"  {m}")
        for arm in ARMS:
            r = res[m][arm]
            r["auc_ci"] = ci(a_rep[m][arm])
            r["recall_ci"] = ci(r_rep[m][arm])
            line = (f"    {arm:12s} AUC {r['auc']:.4f} [{r['auc_ci'][0]:.4f}, "
                    f"{r['auc_ci'][1]:.4f}]   recall@2% {r['recall_2pct']*100:5.1f}%")
            if arm != "global":
                da = np.array(a_rep[m][arm]) - np.array(a_rep[m]["global"])
                dr = np.array(r_rep[m][arm]) - np.array(r_rep[m]["global"])
                r["dauc"] = r["auc"] - res[m]["global"]["auc"]
                r["dauc_ci"] = ci(da)
                r["drecall"] = r["recall_2pct"] - res[m]["global"]["recall_2pct"]
                r["drecall_ci"] = ci(dr)
                line += (f"\n{'':18s}vs global  dAUC {r['dauc']:+.4f} "
                         f"[{r['dauc_ci'][0]:+.4f}, {r['dauc_ci'][1]:+.4f}]"
                         f"  drecall {r['drecall']*100:+.1f}pp")
            print(line)
        for arm in ("full", "silence", "full_int", "silence_int"):
            d = np.array(a_rep[m][arm]) - np.array(a_rep[m]["intercept"])
            dr = np.array(r_rep[m][arm]) - np.array(r_rep[m]["intercept"])
            lo, hi = ci(d)
            pt = res[m][arm]["auc"] - res[m]["intercept"]["auc"]
            rpt = res[m][arm]["recall_2pct"] - res[m]["intercept"]["recall_2pct"]
            res[m][arm].update({"dauc_vs_intercept": pt, "dauc_vs_intercept_ci": [lo, hi],
                                "drecall_vs_intercept": rpt,
                                "drecall_vs_intercept_ci": ci(dr)})
            print(f"    {arm:12s} MINUS intercept-control: dAUC {pt:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  drecall {rpt*100:+.1f}pp   "
                  f"{'separated' if lo * hi > 0 else 'overlaps zero'}")

    print("\n(3) WITHIN-STRATUM AUC: do per-type weights beat global INSIDE the type?")
    for m in MODELS:
        print(f"  {m}")
        for g, name in enumerate(STRATA):
            if not (strat == g).any():
                continue
            gl = res[m]["global"]["by_stratum"][name]["auc"]
            row = f"    {name:20s} n={int((strat == g).sum()):5d}  global {gl:.4f}"
            for arm in ("full", "full_int"):
                d = (np.array(st_rep[m][arm][name])
                     - np.array(st_rep[m]["global"][name]))
                lo, hi = ci(d)
                pt = res[m][arm]["by_stratum"][name]["auc"] - gl
                res[m][arm]["by_stratum"][name].update(
                    {"dauc_vs_global": pt, "dauc_vs_global_ci": [lo, hi]})
                row += f"   {arm} {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            print(row)

    if xfer:
        print(f"\n(4) TRANSFER: urn-fitted (eps {EPS_HEAD:.2f}), gold-evaluated")
        for m in MODELS:
            print(f"  {m}")
            for arm in ARMS:
                t = tres[m][arm]
                if t is None:
                    print(f"    {arm:12s} de-mix infeasible")
                    continue
                t["auc_ci"] = ci(ta_rep[m][arm])
                line = (f"    {arm:12s} AUC {t['auc']:.4f} [{t['auc_ci'][0]:.4f}, "
                        f"{t['auc_ci'][1]:.4f}]   recall@2% {t['recall_2pct']*100:5.1f}%")
                if arm != "global":
                    lo, hi = ci(np.array(ta_rep[m][arm]) - np.array(ta_rep[m]["global"]))
                    t["dauc"] = t["auc"] - tres[m]["global"]["auc"]
                    t["dauc_ci"] = [lo, hi]
                    line += f"   dAUC {t['dauc']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                if arm in ("full", "silence", "full_int", "silence_int"):
                    lo, hi = ci(np.array(ta_rep[m][arm])
                                - np.array(ta_rep[m]["intercept"]))
                    t["dauc_vs_intercept"] = t["auc"] - tres[m]["intercept"]["auc"]
                    t["dauc_vs_intercept_ci"] = [lo, hi]
                    line += (f"\n{'':18s}minus intercept {t['dauc_vs_intercept']:+.4f} "
                             f"[{lo:+.4f}, {hi:+.4f}]")
                print(line)

    payload = {"stratifier": args.stratifier, "population": args.population, "strata": list(STRATA),
               "boot": {"reps": BOOT_REPS, "seed": BOOT_SEED},
               "population": {"gold": len(gold), "pinned": len(pinned),
                              "media_added_back": n_media,
                              "gold_true": int(y.sum()), "gold_false": int((y == 0).sum())},
               "pinned_check": {"auc": a_pin, "recall_2pct": r_pin},
               "diagnostic": diag, "agreement": agree,
               "gold_arms": res, "transfer_arms": tres}
    p = OUT / f"ladder_{tag}.json"
    p.write_text(json.dumps(payload, indent=1, default=float))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
