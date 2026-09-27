"""Document-level loader for the reader-error / ceiling counterfactuals. Mirrors
fit_urn.load_headline() exactly (media-axis exclusion, gold exclusions, pad to 10,
cluster folds) but keeps the per-document real flags so they can be corrected."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval import fit_urn, graded_urn

FLAGS7 = graded_urn.FLAGS7
PAD_TO = fit_urn.PAD_TO
MEDIA_AXIS = fit_urn.MEDIA_AXIS


def load_docs(path: Path, population: set) -> list[dict]:
    """One record per gold-directional claim, keeping the list of real read flags
    (the padding I-docs are recorded as a count, not corrigible)."""
    gold_excl = fit_urn.gold_excluded_ids()
    clusters = fit_urn.load_clusters()
    axis = fit_urn.load_judged_axis()
    out = []
    for line in Path(path).open():
        r = json.loads(line)
        v = r.get("veracity")
        if v not in (1, 2, 3, 4, 5):
            continue
        ru = r.get("review_url")
        if ru in gold_excl:
            continue
        if population is not None and ru not in population:
            continue
        if (axis or {}).get(ru) == MEDIA_AXIS:      # media-axis exclusion (load_headline)
            continue
        real_flags = []
        for d in r.get("results") or []:
            direction = (d.get("read") or {}).get("direction")
            if fit_urn.FLAG_TO_VOICE.get(direction):   # a real read flag
                real_flags.append(direction)
        out.append({
            "y": 1 if v >= 4 else 0,
            "mid": v == 3,
            "veracity": int(v),
            "review_url": ru,
            "real_flags": real_flags,
            "cluster": clusters.get(ru, ru),
            "fold": fit_urn.fold_of(clusters.get(ru, ru), fit_urn.K_FOLDS),
        })
    return out


def aggregate(doc_rows: list[dict]) -> list[dict]:
    """Real flags + pad-to-10 silences -> the 'flags' dict graded_urn.flag_counts wants."""
    rows = []
    for d in doc_rows:
        flags = collections.Counter(d["real_flags"])
        n_pad = max(0, PAD_TO - len(d["real_flags"]))
        flags["I"] += n_pad
        rows.append({"y": d["y"], "mid": d["mid"], "fold": d["fold"],
                     "flags": dict(flags), "cluster": d["cluster"],
                     "review_url": d["review_url"]})
    return rows


def fit_and_score(rows: list[dict]) -> dict:
    """7-flag graded oof AUC, nested recall@2%FPR, operating-point flag counts."""
    C = np.array([graded_urn.flag_counts(r) for r in rows], float).T
    y = np.array([r["y"] for r in rows]); mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    oof = graded_urn.oof_graded(rows)
    auc = fit_urn.auc(oof)
    nest = fit_urn.nested_threshold(C, y, mid, fold, 0.02)
    thr = float(np.mean(nest["thresholds_by_fold"]))
    # operating-point flag counts: reproduce nested flagging to count false/true flagged
    idx = np.arange(C.shape[1]); flagged = np.zeros(len(idx), bool)
    for k in range(fit_urn.K_FOLDS):
        te = fold == k
        if not te.any() or te.all():
            continue
        tr = idx[~te]
        s_in = fit_urn.oof_np(C, y, mid, fold, tr)
        _, _, t = fit_urn.recall_at_fpr(list(zip(s_in.tolist(), y[tr].tolist())), 0.02)
        flagged[te] = (fit_urn.fit_np(C, y, mid, tr) @ C[:, idx[te]]) <= t
    false_flagged = int(flagged[y == 0].sum())   # gold-FALSE flagged (true positives)
    true_flagged = int(flagged[y == 1].sum())    # gold-TRUE flagged (false positives)
    return {"auc": float(auc), "recall": nest["recall"], "fpr": nest["fpr"],
            "threshold": thr, "thresholds_by_fold": nest["thresholds_by_fold"],
            "false_flagged": false_flagged, "true_flagged": true_flagged,
            "n": len(rows), "n_true": int((y == 1).sum()), "n_false": int((y == 0).sum())}
