"""Candidate bucket structures.

Every structure is a map  (base counts on the training rows) -> a per-document
weight vector `v` over the base cells, so a claim scores `v @ C[:, claim]`.
That covers partitions (bucket log-LR broadcast back to its cells), shrinkage
(28 weights pulled toward the flag parent), additive log-linear models and L2
logistic regressions on the count vectors alike.

fit(ctx) -> (v, params, k_eff, note)
  ctx carries the training slice and everything a selector may need. Any inner
  cross-validation happens on ctx.tr ONLY, never on the outer test fold.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from bucketopt_core import (NCELL, FLAG_OF, TIER_OF, VOICE_OF, counts, loglr,
                            se_analytic, cellw_from_labels, merge_path,
                            path_cellweights, maxhw, auc_np, K_FOLDS)

INFL = 1.264          # analytic->bootstrap SE inflation, calibrated in diag28.py
HW_MAX = 0.35         # support constraint: 95% half-width on every bucket weight
G_CRIT = 3.84         # chi2, 1 df, p = 0.05
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
M_GRID = (10, 25, 50, 100, 200, 400, 800, 1600)


class Ctx:
    """Training context. C is the base count matrix (nbase x nclaims)."""
    __slots__ = ("C", "y", "mid", "fold", "tr", "parent", "cT", "cF", "frozen")

    def __init__(self, C, y, mid, fold, tr, parent, frozen=None):
        self.C, self.y, self.mid, self.fold, self.tr = C, y, mid, fold, tr
        self.parent = parent
        self.frozen = frozen or {}
        self.cT, self.cF = counts(C, y, mid, tr)

    def sub(self, tr2):
        return Ctx(self.C, self.y, self.mid, self.fold, tr2, self.parent, self.frozen)

    def inner_folds(self):
        """5 inner folds of the training rows, reusing the pinned fold ids."""
        f = self.fold[self.tr]
        return [(self.tr[f != k], self.tr[f == k]) for k in range(K_FOLDS)]

    def inner_auc(self, vs):
        """Out-of-inner-fold AUC for a family of candidate weight vectors,
        vs = callable(sub_ctx) -> (ncand, nbase) matrix. Returns (ncand,) AUCs."""
        s = None
        for itr, ite in self.inner_folds():
            V = vs(self.sub(itr))
            sc = V @ self.C[:, ite]
            s = sc if s is None else np.concatenate([s, sc], axis=1)
        yy = np.concatenate([self.y[ite] for _, ite in self.inner_folds()])
        return np.array([auc_np(row, yy) for row in s])


# ------------------------------------------------------------------ partitions
def _partition(lab):
    def f(ctx):
        v = cellw_from_labels(ctx.cT, ctx.cF, lab)[0]
        K = lab.max() + 1
        aT = np.bincount(lab, ctx.cT, minlength=K)
        aF = np.bincount(lab, ctx.cF, minlength=K)
        return v, loglr(aT, aF), K, ""
    return f


def _lab_from(keys):
    _, lab = np.unique(np.asarray(keys), return_inverse=True)
    return lab.astype(int)


LAB_VOICE = _lab_from([f"v{v}" for v in VOICE_OF])
LAB_FLAG = _lab_from([f"f{f}" for f in FLAG_OF])
LAB_VOICE_TIER = _lab_from([f"v{v}t{t}" for v, t in zip(VOICE_OF, TIER_OF)])
LAB_CELL = np.arange(NCELL)


# ------------------------------------------------- additive log-linear (IPF)
def additive_loglin(ctx):
    """[flag x tier][flag x class][tier x class] log-linear fit of the
    flag x tier x class document table: no three-way interaction, so the
    resulting log-LR is additive, w[f,t] = a_f + b_t. 7+4-1 = 10 free params."""
    nf, nt = FLAG_OF.max() + 1, TIER_OF.max() + 1
    N = np.zeros((nf, nt, 2))
    for i in range(NCELL):
        N[FLAG_OF[i], TIER_OF[i], 0] = ctx.cT[i] + 1.0     # Laplace, as elsewhere
        N[FLAG_OF[i], TIER_OF[i], 1] = ctx.cF[i] + 1.0
    M = np.ones_like(N) * N.sum() / N.size
    for _ in range(200):
        M *= (N.sum(2) / M.sum(2))[:, :, None]
        M *= (N.sum(1) / M.sum(1))[:, None, :]
        M *= (N.sum(0) / M.sum(0))[None, :, :]
    pT = M[:, :, 0] / M[:, :, 0].sum()
    pF = M[:, :, 1] / M[:, :, 1].sum()
    W = np.log(pT / pF)
    v = W[FLAG_OF, TIER_OF]
    a = W[:, 0]                                  # flag effects at tier 0
    b = W[0, :] - W[0, 0]                        # tier offsets
    return v, np.concatenate([a, b[1:]]), nf + nt - 1, ""


# ------------------------------------------------------------------- logistic
def _logit(design, ctx, C):
    """L2 logistic on per-claim counts collapsed by `design` (nfeat x nbase)."""
    keep = ctx.tr[~ctx.mid[ctx.tr]]
    X = (design @ ctx.C[:, keep]).T
    yy = ctx.y[keep]
    m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    m.fit(X, yy)
    return m.coef_[0] @ design, m.coef_[0]


def _design_flag():
    D = np.zeros((FLAG_OF.max() + 1, NCELL))
    D[FLAG_OF, np.arange(NCELL)] = 1
    return D


def _design_cell():
    return np.eye(NCELL)


def _design_add():
    """7 flag counts + 3 tier counts (first tier dropped: collinear with flags)."""
    nf, nt = FLAG_OF.max() + 1, TIER_OF.max() + 1
    D = np.zeros((nf + nt - 1, NCELL))
    D[FLAG_OF, np.arange(NCELL)] = 1
    for t in range(1, nt):
        D[nf + t - 1, TIER_OF == t] = 1
    return D


def make_logit(design_fn, name):
    D = design_fn()

    def f(ctx):
        key = f"C:{name}"
        if key in ctx.frozen:
            C = ctx.frozen[key]
        else:                                    # inner CV on the training rows
            def cand(sub):
                return np.array([_logit(D, sub, c)[0] for c in C_GRID])
            C = C_GRID[int(np.argmax(ctx.inner_auc(cand)))]
        v, coef = _logit(D, ctx, C)
        return v, coef, len(coef), f"C={C}"
    return f, D


# ------------------------------------------------------------------ shrinkage
def _flag_parent_w(ctx):
    aT = np.bincount(LAB_FLAG, ctx.cT, minlength=LAB_FLAG.max() + 1)
    aF = np.bincount(LAB_FLAG, ctx.cF, minlength=LAB_FLAG.max() + 1)
    return loglr(aT, aF)[LAB_FLAG]


def shrink_eb(ctx):
    """Empirical-Bayes partial pooling of the 28 cell weights toward their
    7-flag parent. tau^2 estimated per flag by moments from the spread of its
    cells around the parent, net of sampling variance. No tuning knob."""
    w = loglr(ctx.cT, ctx.cF)
    par = _flag_parent_w(ctx)
    se2 = (INFL * se_analytic(ctx.cT, ctx.cF)) ** 2
    lam = np.empty(NCELL)
    for f in range(FLAG_OF.max() + 1):
        m = FLAG_OF == f
        r2 = ((w[m] - par[m]) ** 2).mean()
        tau2 = max(r2 - se2[m].mean(), 0.0)
        lam[m] = tau2 / (tau2 + se2[m])
    return lam * w + (1 - lam) * par, lam * w + (1 - lam) * par, \
        float(7 + lam.sum()), f"mean lam {lam.mean():.2f}"


def _soft(ctx, M):
    w = loglr(ctx.cT, ctx.cF)
    par = _flag_parent_w(ctx)
    m = np.minimum(ctx.cT, ctx.cF)
    lam = m / (m + M)
    return lam * w + (1 - lam) * par, lam


def shrink_soft_cv(ctx):
    """quality_urn's soft rule, but with M chosen by inner CV instead of 200."""
    if "M" in ctx.frozen:
        M = ctx.frozen["M"]
    else:
        def cand(sub):
            return np.array([_soft(sub, m)[0] for m in M_GRID])
        M = M_GRID[int(np.argmax(ctx.inner_auc(cand)))]
    v, lam = _soft(ctx, M)
    return v, v, float(7 + lam.sum()), f"M={M}"


def shrink_hard200(ctx):
    """quality_urn's shipped hard rule, verbatim, as a reference row."""
    w = loglr(ctx.cT, ctx.cF)
    par = _flag_parent_w(ctx)
    m = np.minimum(ctx.cT, ctx.cF)
    v = np.where(m >= 200, w, par)
    return v, v, int(len(np.unique(np.round(v, 9)))), ""


# ------------------------------------------------------------------- merging
def _support_labels(ctx, within, hw_max):
    """Phase 1: merge until every bucket's inflated 95% half-width <= hw_max.
    Always merges the least-supported allowed split touching a violator."""
    lab = np.arange(NCELL)
    aT, aF = ctx.cT.astype(float).copy(), ctx.cF.astype(float).copy()
    gpar = ctx.parent.astype(float).copy()
    while len(aT) > 2:
        bad = 1.96 * INFL * se_analytic(aT, aF) > hw_max
        if not bad.any():
            break
        from bucketopt_core import g2_matrix
        g = g2_matrix(aT, aF)
        if within:
            g = np.where(gpar[:, None] == gpar[None, :], g, np.inf)
        g = np.where(bad[:, None] | bad[None, :], g, np.inf)
        if not np.isfinite(g).any():
            break                                 # constraint unreachable
        i, j = np.unravel_index(np.argmin(g), g.shape)
        if i > j:
            i, j = j, i
        aT[i] += aT[j]; aF[i] += aF[j]
        if gpar[i] != gpar[j]:
            gpar[i] = np.nan
        lab = np.where(lab == j, i, lab)
        lab = np.where(lab > j, lab - 1, lab)
        keep = np.ones(len(aT), bool); keep[j] = False
        aT, aF, gpar = aT[keep], aF[keep], gpar[keep]
    return lab, aT, aF, gpar


def make_merge_support(within, hw_max=HW_MAX, then_lrt=False):
    def f(ctx):
        lab, aT, aF, gpar = _support_labels(ctx, within, hw_max)
        if then_lrt:                              # phase 2: pool splits the data
            from bucketopt_core import g2_matrix  # cannot tell apart (G2 < 3.84)
            while len(aT) > 2:
                g = g2_matrix(aT, aF)
                if within:
                    g = np.where(gpar[:, None] == gpar[None, :], g, np.inf)
                if not np.isfinite(g).any():
                    break
                i, j = np.unravel_index(np.argmin(g), g.shape)
                if g[i, j] >= G_CRIT:
                    break
                if i > j:
                    i, j = j, i
                aT[i] += aT[j]; aF[i] += aF[j]
                if gpar[i] != gpar[j]:
                    gpar[i] = np.nan
                lab = np.where(lab == j, i, lab)
                lab = np.where(lab > j, lab - 1, lab)
                keep = np.ones(len(aT), bool); keep[j] = False
                aT, aF, gpar = aT[keep], aF[keep], gpar[keep]
        v = cellw_from_labels(ctx.cT, ctx.cF, lab)[0]
        return v, loglr(aT, aF), int(lab.max() + 1), _sig(lab)
    return f


def make_merge_cv(within):
    """Full greedy path, number of buckets chosen by inner CV on the training
    rows. The partition itself is then the training-fit path at that depth."""
    def f(ctx):
        def cand(sub):
            labs, _ = merge_path(sub.cT, sub.cF, sub.parent, within)
            W = path_cellweights(labs, sub.cT, sub.cF)
            out = np.full((NCELL, NCELL), np.nan)
            for lab, row in zip(labs, W):
                out[NCELL - (lab.max() + 1)] = row
            # depths this fold could not reach keep the deepest reached row
            last = W[-1]
            for k in range(NCELL):
                if np.isnan(out[k, 0]):
                    out[k] = last
            return out
        a = ctx.inner_auc(cand)
        k = int(np.argmax(a))
        labs, _ = merge_path(ctx.cT, ctx.cF, ctx.parent, within)
        lab = labs[min(k, len(labs) - 1)]
        v, aT, aF = cellw_from_labels(ctx.cT, ctx.cF, lab)
        return v, loglr(aT, aF), int(lab.max() + 1), _sig(lab)
    return f


def _sig(lab):
    """Compact signature of a partition: cells grouped, for stability checks."""
    from bucketopt_core import CELL_NAME
    groups = {}
    for i, g in enumerate(lab):
        groups.setdefault(int(g), []).append(CELL_NAME[i])
    return " ; ".join("+".join(v) for v in groups.values())


# ------------------------------------------------------------------- registry
def build_registry():
    reg = {}
    reg["3-voice"] = _partition(LAB_VOICE)
    reg["7-flag"] = _partition(LAB_FLAG)
    reg["12-cell (voice x tier)"] = _partition(LAB_VOICE_TIER)
    reg["28-cell (flag x tier)"] = _partition(LAB_CELL)
    reg["additive log-LR (flag+tier)"] = additive_loglin
    reg["logit-7 (L2)"] = make_logit(_design_flag, "flag")[0]
    reg["logit-additive-10 (L2)"] = make_logit(_design_add, "add")[0]
    reg["logit-28 (L2)"] = make_logit(_design_cell, "cell")[0]
    reg["shrink-EB toward flag"] = shrink_eb
    reg["shrink-soft M by inner CV"] = shrink_soft_cv
    reg["shrink-hard M=200 (quality_urn)"] = shrink_hard200
    reg["merge within-flag, support"] = make_merge_support(True)
    reg["merge within-flag, support+LRT"] = make_merge_support(True, then_lrt=True)
    reg["merge any, support"] = make_merge_support(False)
    reg["merge any, support+LRT"] = make_merge_support(False, then_lrt=True)
    reg["merge within-flag, inner-CV"] = make_merge_cv(True)
    reg["merge any, inner-CV"] = make_merge_cv(False)
    return reg
