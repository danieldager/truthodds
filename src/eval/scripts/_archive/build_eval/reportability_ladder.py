"""Does conditioning the SILENT weight on reportability beat one global number?

    uv run python -m eval.scripts.build_eval.reportability_ladder

$0 -- refits the saved reads. Tests Daniel's hypothesis (2026-08-28): silence is
strong evidence of falsehood when the claim would necessarily have been reported
if true, and uninformative when nothing would have been written either way, so
one fitted w_silent averages two regimes and gets both wrong.

STRATUM. `reportability.py`'s label-blind high/medium/low, assigned before
retrieval from claim text + claim date only.

ARMS, all sharing model_ladder's out-of-fold protocol, its Laplace convention
(+1 per bucket, +K on each class total, unshrunk, now WITHIN stratum) and its
fit_w / oof machinery. Both bucketings, 3-voice and 7-flag. 28-cell skipped.
  global      the published baseline: one weight vector, model_ladder exactly
  silence     HEADLINE. Directional weights global, SILENT buckets per stratum.
              One extra degree of freedom per stratum, on the one bucket whose
              meaning the hypothesis says is claim-dependent.
  full        secondary: every bucket per stratum
  intercept   CONTROL. Global weights plus a per-stratum constant, the claim-level
              log-LR of stratum membership. If `silence` beats `global` but
              `intercept` beats it just as much, the gain is prior-encoding and
              the hypothesis is NOT supported. This comparison is the deliverable.

TRANSFER. The same arms fitted on the two urns with no gold label (CN-false vs
the de-mixed timeline at eps, transfer_ladder's demix verbatim) and scored on
the pinned gold population.

Intervals. 2,000-rep paired bootstrap, seed 707, resampled within gold class
(and over both urns for the transfer cells), so every delta is paired.

Writes eval/data/reportability/ladder.json. Touches nothing under urn_runs/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval import fit_urn
from eval.scripts.build_eval import model_ladder as ML
from eval.scripts.build_eval import transfer_ladder as TLD
from eval.scripts.build_eval import reportability as RP

OUT = Path("eval/data/reportability")
LABELS = OUT / "labels.jsonl"
STRATA = ("high", "medium", "low")
MODELS = ("3-voice", "7-flag")
SILENT = {"3-voice": ("silent",), "7-flag": ("3", "X", "I")}
# arm -> (which weight vectors, whether a per-stratum intercept is added).
# `intercept` is the prior-encoding control: global weights, stratum constant.
# `silence_int` is the hypothesis stated honestly -- a bare per-stratum weight
# vector is only calibrated WITHIN its stratum, so comparing it to a global fit
# on a global ranking metric confounds the mechanism with the miscalibration it
# introduces. silence_int vs intercept is the decisive test.
ARM_SPEC = {"global": ("global", False), "silence": ("silence", False),
            "full": ("full", False), "intercept": ("global", True),
            "silence_int": ("silence", True), "full_int": ("full", True)}
ARMS = tuple(ARM_SPEC)
BOOT_REPS = int(__import__("os").environ.get("REPS", 2000))     # REPS=50 for a shakedown
BOOT_SEED = 707
EPS_HEAD = TLD.EPS_HEAD
K = fit_urn.K_FOLDS
G = len(STRATA)


# ---------------------------------------------------------------- metrics
def recall_at_fpr(s: np.ndarray, y: np.ndarray, budget: float) -> float:
    """Vectorised twin of fit_urn.recall_at_fpr's recall term (asserted in main)."""
    trues, falses = np.sort(s[y == 1]), np.sort(s[y == 0])
    thr = np.unique(s)
    fpr = np.searchsorted(trues, thr, side="right") / len(trues)
    rec = np.searchsorted(falses, thr, side="right") / len(falses)
    ok = fpr <= budget
    return float(rec[ok].max()) if ok.any() else 0.0


def ci(v) -> list[float]:
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]


# ---------------------------------------------------------------- gold arms
def stratum_weights(C, y, mid, strat, idx, model):
    """Global weight vector and the three within-stratum vectors, on rows `idx`."""
    w = ML.fit_w(C, y, mid, idx)
    per = np.stack([ML.fit_w(C, y, mid, idx[strat[idx] == g]) for g in range(G)], 1)
    return w, per                                   # (K,), (K, G)


def arm_matrix(w, per, sil_idx, arm):
    """(K, G) weight matrix, one column per stratum, for one arm."""
    if arm == "full":
        return per
    W = np.repeat(w[:, None], G, 1)
    if arm == "silence":
        W[sil_idx, :] = per[sil_idx, :]
    return W


def intercept(y, mid, strat, idx):
    """Claim-level log-LR of stratum membership, Laplace +1 / +G. The control."""
    keep = idx[~mid[idx]]
    nT = np.array([((strat[keep] == g) & (y[keep] == 1)).sum() for g in range(G)])
    nF = np.array([((strat[keep] == g) & (y[keep] == 0)).sum() for g in range(G)])
    return np.log(((nT + 1) / (nT.sum() + G)) / ((nF + 1) / (nF.sum() + G)))


def oof(C, y, mid, fold, strat, model, arm):
    """Out-of-fold scores for one arm: fit on K-1 folds, score the held-out fold."""
    base, use_int = ARM_SPEC[arm]
    sil = np.array([i for i, c in enumerate(ML.channels(model)) if c in SILENT[model]])
    s = np.empty(C.shape[1])
    allidx = np.arange(C.shape[1])
    for k in range(K):
        tr, te = allidx[fold != k], allidx[fold == k]
        w, per = stratum_weights(C, y, mid, strat, tr, model)
        W = arm_matrix(w, per, sil, base)
        s[te] = (W[:, strat[te]] * C[:, te]).sum(0)
        if use_int:
            s[te] += intercept(y, mid, strat, tr)[strat[te]]
    return s


# ---------------------------------------------------------------- transfer
def urn_weights(CF, CM, sF, sM, eps):
    """Global and per-stratum de-mixed log-LR from the two urns. None if infeasible."""
    w = TLD.demix(CF.sum(1), CM.sum(1), eps)
    if w is None:
        return None, None
    per = []
    for g in range(G):
        wg = TLD.demix(CF[:, sF == g].sum(1), CM[:, sM == g].sum(1), eps)
        per.append(w if wg is None else wg)          # thin stratum falls back to global
    return w, np.stack(per, 1)


def urn_intercept(sF, sM, eps):
    pF = np.array([(sF == g).sum() for g in range(G)], float)
    pM = np.array([(sM == g).sum() for g in range(G)], float)
    pF = (pF + 1) / (pF.sum() + G)
    pM = (pM + 1) / (pM.sum() + G)
    pT = (pM - eps * pF) / (1 - eps)
    return np.log(np.maximum(pT, 1e-9) / pF)


def transfer_scores(CG, gstrat, CF, CM, sF, sM, eps, model, arm):
    base, use_int = ARM_SPEC[arm]
    sil = np.array([i for i, c in enumerate(ML.channels(model)) if c in SILENT[model]])
    w, per = urn_weights(CF, CM, sF, sM, eps)
    if w is None:
        return None
    W = arm_matrix(w, per, sil, base)
    s = (W[:, gstrat] * CG).sum(0)
    return s + urn_intercept(sF, sM, eps)[gstrat] if use_int else s


# ---------------------------------------------------------------- main
def main() -> None:
    lab = {}
    for line in LABELS.open():
        r = json.loads(line)
        lab[r["key"]] = r["reportability"]
    print(f"labels {len(lab)}  [{json.loads(next(LABELS.open()))['prompt_v']}]")

    # ---- populations, each asserted against the script that owns it ----------
    gold = [r for r in ML.load_docs() if r["subtype"] != ML.SET_ASIDE]
    gkeys = RP.gold_rows()
    assert len(gold) == len(gkeys)
    cn = TLD.load_urn([TLD.C2 / "scores.jsonl", TLD.C2 / "scores_ext.jsonl"],
                      TLD.C2 / "fit_exclusions.json")
    tl = TLD.load_urn([TLD.TL / "scores.jsonl"])
    ckeys = RP.urn_rows("cn", [RP.C2 / "scores.jsonl", RP.C2 / "scores_ext.jsonl"],
                        RP.C2 / "fit_exclusions.json")
    tkeys = RP.urn_rows("tl", [RP.TL / "scores.jsonl"], None)
    assert [a["claim"] for a in cn] == [b["claim"] for b in ckeys]
    assert [a["claim"] for a in tl] == [b["claim"] for b in tkeys]

    S = {s: i for i, s in enumerate(STRATA)}
    strat = np.array([S[lab[r["key"]]] for r in gkeys])
    sF = np.array([S[lab[r["key"]]] for r in ckeys])
    sM = np.array([S[lab[r["key"]]] for r in tkeys])

    y = np.array([r["y"] for r in gold])
    mid = np.array([r["mid"] for r in gold])
    fold = np.array([r["fold"] for r in gold])
    C = {m: ML.count_matrix(gold, m) for m in MODELS}
    CF = {m: TLD.urn_matrix(cn, m) for m in MODELS}
    CM = {m: TLD.urn_matrix(tl, m) for m in MODELS}

    # ---- (a) THE CONFOUND CHECK, before anything else -----------------------
    print(f"\ngold {len(gold)} (T {int(y.sum())} / F {int((y == 0).sum())})   "
          f"CN-false {len(cn)}   timeline {len(tl)}")
    print("\n(a) CONFOUND: base rate of FALSE by stratum, and stratum mix per urn")
    print(f"  {'stratum':9s} {'gold n':>7} {'P(FALSE)':>9} {'gold %':>7} "
          f"{'CN %':>7} {'TL %':>7}")
    confound = {}
    for g, name in enumerate(STRATA):
        m_ = strat == g
        pf = float((y[m_] == 0).mean()) if m_.any() else float("nan")
        confound[name] = {"gold_n": int(m_.sum()), "p_false": pf,
                          "gold_share": float(m_.mean()),
                          "cn_share": float((sF == g).mean()),
                          "tl_share": float((sM == g).mean())}
        print(f"  {name:9s} {int(m_.sum()):7d} {pf:9.3f} {m_.mean()*100:6.1f}% "
              f"{(sF == g).mean()*100:6.1f}% {(sM == g).mean()*100:6.1f}%")
    print(f"  {'ALL':9s} {len(gold):7d} {float((y == 0).mean()):9.3f}")

    # ---- point estimates ----------------------------------------------------
    ref = {}
    res = {}
    for m in MODELS:
        for arm in ARMS:
            s = oof(C[m], y, mid, fold, strat, m, arm)
            res.setdefault(m, {})[arm] = {
                "auc": ML.auc_np(s[y == 1], s[y == 0]),
                "recall_2pct": recall_at_fpr(s, y, 0.02)}
            if arm == "global":
                ref[m] = s
    # the global arm must BE model_ladder
    for m in MODELS:
        assert abs(res[m]["global"]["auc"]
                   - ML.auc_np(ML.oof_scores(C[m], y, mid, fold)[y == 1],
                               ML.oof_scores(C[m], y, mid, fold)[y == 0])) < 1e-12
        assert abs(res[m]["global"]["recall_2pct"]
                   - ML.recall_at_fpr(ref[m], y, 0.02)[0]) < 1e-12
    print(f"\nglobal arm reproduces model_ladder: 7-flag AUC "
          f"{res['7-flag']['global']['auc']:.4f} / recall@2% "
          f"{res['7-flag']['global']['recall_2pct']:.4f}")

    # ---- (b) per-stratum silence weights, point estimates -------------------
    sil = {m: np.array([i for i, c in enumerate(ML.channels(m)) if c in SILENT[m]])
           for m in MODELS}
    allidx = np.arange(len(gold))
    wpt = {}
    for m in MODELS:
        w, per = stratum_weights(C[m], y, mid, strat, allidx, m)
        wpt[m] = {"global": w, "per": per}

    # ---- transfer point estimates ------------------------------------------
    tres = {}
    for m in MODELS:
        for arm in ARMS:
            s = transfer_scores(C[m], strat, CF[m], CM[m], sF, sM, EPS_HEAD, m, arm)
            tres.setdefault(m, {})[arm] = None if s is None else {
                "auc": ML.auc_np(s[y == 1], s[y == 0]),
                "recall_2pct": recall_at_fpr(s, y, 0.02)}

    # ---- (d) the practical payoff: HIGH reportability, all-silent -----------
    print("\n(d) claims whose retrieval came back ALL SILENT, by stratum")
    silent_only = {}
    for m in ["7-flag"]:
        tot = C[m].sum(0)
        smass = C[m][sil[m], :].sum(0)
        gsil = (tot > 0) & (smass == tot)
        cnsil_m = CF[m][sil[m], :].sum(0) == CF[m].sum(0)
        tlsil_m = CM[m][sil[m], :].sum(0) == CM[m].sum(0)
        print(f"  {'stratum':9s} {'gold n':>7} {'false rate':>11} | "
              f"{'CN n':>6} {'TL n':>6} {'share CN':>9} {'LR(false)':>10}")
        for g, name in enumerate(STRATA):
            sel = gsil & (strat == g)
            fr = float((y[sel] == 0).mean()) if sel.any() else float("nan")
            nc = int((cnsil_m & (sF == g)).sum())
            nt = int((tlsil_m & (sM == g)).sum())
            share = nc / (nc + nt) if nc + nt else float("nan")
            lr = ((nc / len(cn)) / (nt / len(tl))) if nt else float("inf")
            silent_only[name] = {"gold_n": int(sel.sum()), "gold_false_rate": fr,
                                 "cn_n": nc, "tl_n": nt, "share_cn": share,
                                 "lr_false": lr}
            print(f"  {name:9s} {int(sel.sum()):7d} {fr:11.3f} | {nc:6d} {nt:6d} "
                  f"{share*100:8.1f}% {lr:10.2f}")
        allsel = gsil
        print(f"  {'ALL':9s} {int(allsel.sum()):7d} "
              f"{float((y[allsel] == 0).mean()):11.3f} | "
              f"{int(cnsil_m.sum()):6d} {int(tlsil_m.sum()):6d}")
        silent_only["ALL"] = {"gold_n": int(allsel.sum()),
                              "gold_false_rate": float((y[allsel] == 0).mean()),
                              "cn_n": int(cnsil_m.sum()), "tl_n": int(tlsil_m.sum())}

    # ---- paired bootstrap ---------------------------------------------------
    print(f"\nbootstrap ({BOOT_REPS} reps, seed {BOOT_SEED}) ...", flush=True)
    rng = np.random.default_rng(BOOT_SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    nF_, nM_ = len(cn), len(tl)
    a_rep = {m: {a: [] for a in ARMS} for m in MODELS}
    r_rep = {m: {a: [] for a in ARMS} for m in MODELS}
    ta_rep = {m: {a: [] for a in ARMS} for m in MODELS}
    w_rep = {m: [] for m in MODELS}            # per-stratum weight vectors
    for rep in range(BOOT_REPS):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        iF, iM = rng.integers(0, nF_, nF_), rng.integers(0, nM_, nM_)
        yy, mm, ff, ss = y[idx], mid[idx], fold[idx], strat[idx]
        aa = np.arange(len(idx))
        for m in MODELS:
            Cm = C[m][:, idx]
            _, per = stratum_weights(Cm, yy, mm, ss, aa, m)
            w_rep[m].append(per)
            for arm in ARMS:
                s = oof(Cm, yy, mm, ff, ss, m, arm)
                a_rep[m][arm].append(ML.auc_np(s[yy == 1], s[yy == 0]))
                r_rep[m][arm].append(recall_at_fpr(s, yy, 0.02))
                st = transfer_scores(Cm, ss, CF[m][:, iF], CM[m][:, iM],
                                     sF[iF], sM[iM], EPS_HEAD, m, arm)
                ta_rep[m][arm].append(np.nan if st is None else
                                      ML.auc_np(st[yy == 1], st[yy == 0]))
        if (rep + 1) % 250 == 0:
            print(f"  {rep+1}/{BOOT_REPS}", flush=True)

    # ---- (b) report ---------------------------------------------------------
    print("\n(b) SILENT-bucket weight per stratum (gold-fitted), 95% CI")
    silw = {}
    for m in MODELS:
        W = np.array(w_rep[m])                      # (reps, K, G)
        print(f"  {m}")
        for i in sil[m]:
            ch = ML.channels(m)[i]
            gw = wpt[m]["global"][i]
            for g, name in enumerate(STRATA):
                lo, hi = ci(W[:, i, g])
                pt = wpt[m]["per"][i, g]
                silw.setdefault(m, {}).setdefault(ch, {})[name] = {
                    "w": float(pt), "ci": [lo, hi]}
                print(f"    {ch:8s} {name:7s} {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
                      + (f"     (global {gw:+.3f})" if g == 0 else ""))
            d = W[:, i, S["high"]] - W[:, i, S["low"]]
            dlo, dhi = ci(d)
            dpt = wpt[m]["per"][i, S["high"]] - wpt[m]["per"][i, S["low"]]
            silw[m][ch]["high_minus_low"] = {"d": float(dpt), "ci": [dlo, dhi]}
            print(f"    {ch:8s} HIGH - LOW  {dpt:+.3f}  [{dlo:+.3f}, {dhi:+.3f}]"
                  f"   {'SEPARATED' if dlo * dhi > 0 else 'overlaps zero'}")

    # ---- (c) arm comparison -------------------------------------------------
    print("\n(c) ARMS on gold, out of fold. dAUC / d-recall vs the global baseline")
    for m in MODELS:
        print(f"  {m}")
        for arm in ARMS:
            r = res[m][arm]
            a_ci = ci(a_rep[m][arm])
            line = (f"    {arm:10s} AUC {r['auc']:.4f} [{a_ci[0]:.4f}, {a_ci[1]:.4f}]"
                    f"   recall@2% {r['recall_2pct']*100:5.1f}%")
            res[m][arm]["auc_ci"] = a_ci
            res[m][arm]["recall_ci"] = [x * 1 for x in ci(r_rep[m][arm])]
            if arm != "global":
                da = np.array(a_rep[m][arm]) - np.array(a_rep[m]["global"])
                dr = np.array(r_rep[m][arm]) - np.array(r_rep[m]["global"])
                dal, dah = ci(da)
                drl, drh = ci(dr)
                res[m][arm]["dauc"] = r["auc"] - res[m]["global"]["auc"]
                res[m][arm]["dauc_ci"] = [dal, dah]
                res[m][arm]["drecall"] = r["recall_2pct"] - res[m]["global"]["recall_2pct"]
                res[m][arm]["drecall_ci"] = [drl, drh]
                line += (f"\n{'':16s}dAUC {res[m][arm]['dauc']:+.4f} [{dal:+.4f}, {dah:+.4f}]"
                         f"   drecall {res[m][arm]['drecall']*100:+.1f}pp "
                         f"[{drl*100:+.1f}, {drh*100:+.1f}]")
            print(line)
        # THE DELIVERABLE: does per-stratum silence add anything the per-stratum
        # PRIOR does not already give? Both sides carry the same intercept, so
        # the only difference is the mechanism.
        for arm in ("silence", "full", "silence_int", "full_int"):
            d = np.array(a_rep[m][arm]) - np.array(a_rep[m]["intercept"])
            dr = np.array(r_rep[m][arm]) - np.array(r_rep[m]["intercept"])
            lo, hi = ci(d)
            rlo, rhi = ci(dr)
            pt = res[m][arm]["auc"] - res[m]["intercept"]["auc"]
            rpt = res[m][arm]["recall_2pct"] - res[m]["intercept"]["recall_2pct"]
            res[m][arm]["dauc_vs_intercept"] = pt
            res[m][arm]["dauc_vs_intercept_ci"] = [lo, hi]
            res[m][arm]["drecall_vs_intercept"] = rpt
            res[m][arm]["drecall_vs_intercept_ci"] = [rlo, rhi]
            print(f"    {arm:11s} MINUS intercept-control: dAUC {pt:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  drecall {rpt*100:+.1f}pp "
                  f"[{rlo*100:+.1f}, {rhi*100:+.1f}]"
                  f"   {'separated' if lo * hi > 0 else 'overlaps zero'}")

    print("\n    transfer (urn-fitted, gold-evaluated, eps "
          f"{EPS_HEAD:.2f}), dAUC vs global")
    for m in MODELS:
        print(f"  {m}")
        for arm in ARMS:
            t = tres[m][arm]
            if t is None:
                print(f"    {arm:10s} de-mix infeasible")
                continue
            a_ci = ci(ta_rep[m][arm])
            t["auc_ci"] = a_ci
            line = (f"    {arm:10s} AUC {t['auc']:.4f} [{a_ci[0]:.4f}, {a_ci[1]:.4f}]"
                    f"   recall@2% {t['recall_2pct']*100:5.1f}%")
            if arm != "global":
                d = np.array(ta_rep[m][arm]) - np.array(ta_rep[m]["global"])
                lo, hi = ci(d)
                t["dauc"] = t["auc"] - tres[m]["global"]["auc"]
                t["dauc_ci"] = [lo, hi]
                line += f"   dAUC {t['dauc']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            print(line)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"strata": list(STRATA), "boot": {"reps": BOOT_REPS, "seed": BOOT_SEED},
               "eps_head": EPS_HEAD,
               "population": {"gold": len(gold), "gold_true": int(y.sum()),
                              "gold_false": int((y == 0).sum()),
                              "cn": len(cn), "tl": len(tl)},
               "confound": confound, "silent_weights": silw,
               "gold_arms": res, "transfer_arms": tres,
               "all_silent": silent_only}
    (OUT / "ladder.json").write_text(json.dumps(payload, indent=1, default=float))
    print(f"\nwrote {OUT / 'ladder.json'}")


if __name__ == "__main__":
    main()
