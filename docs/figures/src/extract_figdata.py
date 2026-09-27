"""Build figdata.json, the only input of make_figures.py. $0, read-only.

Needs the full research tree (the run files are not in this repo): run from its src/ with
    PYTHONPATH=. uv run python <this file> <out.json>
Inputs: e1_ctx/results-00.jsonl (evidence dated before the fact-check),
e1_prodregime/cond_S_v5_full3000_v5.jsonl (today's web), weights_v5_ceiling_bal3000.json,
headline_metrics.json (weight CIs), reader_lab/sub500/*.jsonl (reader comparison)."""
import collections, json, sys
from pathlib import Path
import numpy as np
from eval.scripts.build_eval import fit_urn, graded_urn, fit_ceiling_weights as fcw

E1 = Path("eval/data/urn_runs/e1_ctx")
W = json.loads((E1 / "weights_v5_ceiling_bal3000.json").read_text())
H = json.loads((E1 / "headline_metrics.json").read_text())["overall"]
PROD = Path("eval/data/urn_runs/e1_prodregime/cond_S_v5_full3000_v5.jsonl")

def oof_rows(path):
    rows = fcw.load_rows(path, fcw.POPULATION)
    for k in range(fit_urn.K_FOLDS):
        w = graded_urn.fit_graded([r for r in rows if r["fold"] != k])
        for r in rows:
            if r["fold"] == k: r["score"] = graded_urn.score_graded(r, w)
    return rows

def roc(rows):
    s = np.array([r["score"] for r in rows]); y = np.array([r["y"] for r in rows])
    # a flag = score below the boundary; recall over false claims, FPR over true claims
    thr = np.unique(s)
    tpr = [0.0] + [float(((s <= t) & (y == 0)).sum() / (y == 0).sum()) for t in thr]
    fpr = [0.0] + [float(((s <= t) & (y == 1)).sum() / (y == 1).sum()) for t in thr]
    return {"fpr": [round(v, 4) for v in fpr], "tpr": [round(v, 4) for v in tpr],
            "auc": fit_urn.auc([(r["score"], r["y"]) for r in rows])}

train, prod = oof_rows(fcw.RESULTS), oof_rows(PROD)
out = {"threshold": W["threshold"],
       "roc": {"before": {**roc(train), "ci": W["auc_ci95"]}, "today": {**roc(prod), "ci": [0.914, 0.933]}},
       "weights": {k: {"w": W["weights"][k], "ci": H["weights_ci"][k]} for k in W["flags"]}}

# score by verdict: out-of-fold on the 3,000 fitted claims; mixed claims (never fitted) with the frozen weights
ver = {json.loads(l)["review_url"]: json.loads(l).get("veracity") for l in fcw.RESULTS.open()}
sbv = [{"v": int(ver[r["review_url"]]), "s": r["score"]} for r in train]
mixed = [r for r in fit_urn.load_headline(fcw.RESULTS) if r["mid"]]
sbv += [{"v": 3, "s": graded_urn.score_graded(r, W["weights"])} for r in mixed]
out["score_by_verdict"] = sbv

# reader comparison: 500 claims (250 true / 250 false), same pages, weights refitted per reader
clusters = fit_urn.load_clusters()
def reader_rows(path, key):
    rows = []
    for l in open(path):
        r = json.loads(l); fl = collections.Counter(r[key]); fl["I"] += max(0, 10 - sum(fl.values()))
        rows.append({"y": int(r["group"] == "T"), "mid": False, "flags": dict(fl),
                     "fold": fit_urn.fold_of(clusters.get(r["claim_id"], r["claim_id"]), fit_urn.K_FOLDS)})
    return rows
S = "eval/data/reader_lab/sub500/"
out["readers_recomputed"] = {
    "DeepSeek V4 Flash": fit_urn.auc(graded_urn.oof_graded(reader_rows(S + "v5__DeepSeek-V4-Pro.jsonl", "old_flags"))),
    "DeepSeek V4 Pro": fit_urn.auc(graded_urn.oof_graded(reader_rows(S + "v5__DeepSeek-V4-Pro.jsonl", "new_flags"))),
    "Kimi K2.6": fit_urn.auc(graded_urn.oof_graded(reader_rows(S + "v5__Kimi-K2.6.jsonl", "new_flags")))}
Path(sys.argv[1]).write_text(json.dumps(out))
print({k: v["auc"] for k, v in out["roc"].items()}, out["readers_recomputed"])
