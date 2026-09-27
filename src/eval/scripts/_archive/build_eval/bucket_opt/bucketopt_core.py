"""Shared loading + estimator core for the bucket-partition optimisation.

$0, local refit of the saved E1 reads. Same pinned population, folds, seed and
bootstrap as model_ladder.py / issue24_2x2.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SRC))

from eval.scripts.build_eval import fit_urn                       # noqa: E402
from eval.scripts.build_eval import model_ladder as ML            # noqa: E402
from eval.scripts.build_eval.graded_urn import FLAGS, FLAG_DESC   # noqa: E402
from eval.scripts.build_eval.quality_urn import TIERS, VOICE, CELLS  # noqa: E402

OUT = Path("eval/data/urn_runs/e1_ctx/model_ladder/bucket_opt")
RESULTS = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
LADDER = Path("eval/data/urn_runs/e1_ctx/model_ladder/ladder.json")
TRANSFER_DIR = Path("eval/data/urn_runs/e1_ctx/model_ladder")
C2 = Path("eval/data/urn_runs/c2_false")
TL = Path("eval/data/urn_runs/true_timeline")

BOOT_REPS = int(os.environ.get("BOOT_REPS", 2000))
BOOT_SEED = 707
K_FOLDS = fit_urn.K_FOLDS
VN = ML.VOICE_NAME
NCELL = len(CELLS)                       # 28
CELL_IDX = {c: i for i, c in enumerate(CELLS)}
CELL_NAME = [f"{fl} | {tr}" for fl, tr in CELLS]   # matches model_ladder 28-cell keys
FLAG_OF = np.array([FLAGS.index(fl) for fl, tr in CELLS])
TIER_OF = np.array([TIERS.index(tr) for fl, tr in CELLS])
VOICE_OF = np.array([("S", "R", "E").index(VOICE[fl]) for fl, tr in CELLS])


def bootstrap() -> None:
    """Process setup the four bucket_opt scripts share. Call first from main()."""
    os.chdir(SRC)
    os.environ.setdefault("GOLD_EXCLUSIONS", "none")   # pinned pre-purge population n=3,274


# --------------------------------------------------------------------------- load
def load_gold(extra: bool = False) -> list[dict]:
    """Pinned gold rows. extra=True also keeps rank / provenance per document."""
    import json
    axis = fit_urn.load_judged_axis()
    gold_excl = fit_urn.gold_excluded_ids()
    rows = []
    for line in RESULTS.open():
        r = json.loads(line)
        v = r.get("veracity")
        if v not in (1, 2, 3, 4, 5) or r["review_url"] in gold_excl:
            continue
        docs = []
        for d in r.get("results") or []:
            dirn = (d.get("read") or {}).get("direction")
            if dirn not in FLAGS:
                continue
            if extra:
                try:
                    rk = int(d.get("rank"))
                except (TypeError, ValueError):
                    rk = 99
                docs.append((dirn, d.get("rel") or "UNRATED", rk,
                             str(d.get("provenance") or "?")))
            else:
                docs.append((dirn, d.get("rel") or "UNRATED"))
        if not docs:
            continue
        if (axis.get(r["review_url"]) or "untagged") == fit_urn.MEDIA_AXIS:
            continue
        rows.append({"review_url": r["review_url"], "y": 1 if v >= 4 else 0,
                     "mid": v == 3, "docs": docs,
                     "subtype": r.get("rating_subtype") or "?",
                     "fold": fit_urn.fold_of(r["review_url"], K_FOLDS)})
    return [r for r in rows if r["subtype"] != ML.SET_ASIDE]


def cell_matrix(rows: list[dict]) -> np.ndarray:
    C = np.zeros((NCELL, len(rows)))
    for j, r in enumerate(rows):
        for d in r["docs"]:
            C[CELL_IDX[(d[0], d[1])], j] += 1
    return C


def load_urns():
    """CN-false urn rows and timeline urn rows, exactly as transfer_ladder does."""
    import json
    def _load(paths, exclusions=None):
        excl = set()
        if exclusions and exclusions.exists():
            excl = {e["claim"][:80] for e in json.loads(exclusions.read_text())}
        if exclusions is not None:
            for x in fit_urn.extra_exclusion_paths():
                excl |= {e["claim"][:80] for e in json.loads(x.read_text())
                         if e.get("excluded", True)}
        out = []
        for p in paths:
            for line in p.open():
                r = json.loads(line)
                claim = r.get("claim_resolved") or r.get("claim_text") or ""
                if claim[:80] in excl:
                    continue
                docs = [(d["read"]["direction"], d.get("rel") or "UNRATED")
                        for d in r.get("results") or []
                        if (d.get("read") or {}).get("direction") in FLAGS]
                if docs:
                    out.append({"docs": docs})
        return out
    return (_load([C2 / "scores.jsonl", C2 / "scores_ext.jsonl"], C2 / "fit_exclusions.json"),
            _load([TL / "scores.jsonl"]))


# ----------------------------------------------------------------- estimator core
def counts(C28, y, mid, idx):
    """(cT, cF) document counts per cell on the fit rows of idx (mid excluded)."""
    keep = idx[~mid[idx]]
    p, n = keep[y[keep] == 1], keep[y[keep] == 0]
    return C28[:, p].sum(1), C28[:, n].sum(1)


def loglr(cT, cF):
    """Laplace-smoothed log-LR, +1 per bucket and +K on each class total."""
    K = len(cT)
    return np.log(((cT + 1) / (cT.sum() + K)) / ((cF + 1) / (cF.sum() + K)))


def se_analytic(cT, cF):
    """Delta-method SE of the Laplace log-LR (per-bucket term dominates)."""
    return np.sqrt(1.0 / (cT + 1) + 1.0 / (cF + 1))


def cellw_from_labels(cT28, cF28, lab):
    """Aggregate cells by integer label vector, fit log-LR, broadcast back to 28."""
    K = lab.max() + 1
    aT = np.bincount(lab, cT28, minlength=K)
    aF = np.bincount(lab, cF28, minlength=K)
    return loglr(aT, aF)[lab], aT, aF


def auc_np(s, y):
    # round before ranking: scores are integer combinations of logs, so two
    # claims that are genuinely tied can differ by ~1e-15 depending on the
    # summation order. Without this the coarse models' AUC wobbles in the 4th
    # decimal purely from how the count matrix was aggregated.
    s = np.round(s, 9)
    return ML.auc_np(s[y == 1], s[y == 0])


def recall_at_fpr(s, y, budget=0.02):
    return ML.recall_at_fpr(np.round(s, 9), y, budget)


# ------------------------------------------------------------------- G2 merge path
def g2_matrix(cT, cF):
    """Pairwise likelihood-ratio statistic for pooling buckets i and j (2x2 on
    class x bucket). Small G2 = the split is not supported by the data."""
    a = cT[:, None] + 0.5; b = cF[:, None] + 0.5
    c = cT[None, :] + 0.5; d = cF[None, :] + 0.5
    n = a + b + c + d
    def xl(x):
        return x * np.log(x)
    g = 2 * (xl(a) + xl(b) + xl(c) + xl(d) + xl(n)
             - xl(a + b) - xl(c + d) - xl(a + c) - xl(b + d))
    np.fill_diagonal(g, np.inf)
    return g


def merge_path(cT, cF, parent, within_parent: bool, g_stop=None):
    """Greedy agglomerative path over the base cells. Returns (labels, g2s):
    labels[k] is the label vector with len(cT)-k buckets, g2s[k] the G2 of the
    merge that produced it. Merges the pair with the smallest G2 among allowed
    pairs (same parent group, if constrained) at every step."""
    nb = len(cT)
    lab = np.arange(nb)
    aT, aF = cT.astype(float).copy(), cF.astype(float).copy()
    gpar = parent.astype(float).copy()
    labs, g2s = [lab.copy()], [0.0]
    while len(aT) > 2:
        g = g2_matrix(aT, aF)
        if within_parent:
            g = np.where(gpar[:, None] == gpar[None, :], g, np.inf)
            if not np.isfinite(g).any():
                break
        i, j = np.unravel_index(np.argmin(g), g.shape)
        gval = float(g[i, j])
        if g_stop is not None and gval >= g_stop:
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
        labs.append(lab.copy()); g2s.append(gval)
    return labs, g2s


def path_cellweights(labs, cT, cF):
    """(len(labs), nbase) matrix of per-cell weights, one row per partition."""
    W = np.empty((len(labs), len(cT)))
    for i, lab in enumerate(labs):
        W[i] = cellw_from_labels(cT, cF, lab)[0]
    return W


def maxhw(lab, cT, cF, infl=1.0):
    K = lab.max() + 1
    aT = np.bincount(lab, cT, minlength=K)
    aF = np.bincount(lab, cF, minlength=K)
    return float((1.96 * infl * se_analytic(aT, aF)).max())
