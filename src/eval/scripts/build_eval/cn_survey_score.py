"""Score the UNRESTRAINED CN-survey verification run with the ARM-S six-flag urn (2026-09-16).

Weights and boundary come from `eval/data/urn_runs/e1_prodregime/prodregime_metrics.json`
-> `conditions.S` (the Serper-only arm of the 2026-09-15 production-regime comparison, fitted on
fc_gold_rep1500). They are READ from that file, never retyped, so the grading is traceable.
Scoring convention is fit_urn's: six flags (contested 3 folded into X), every claim padded to
PAD_TO=10 slots with I, score = sum(count x weight), FLAG when score <= boundary.

Writes eval/data/community_notes/survey_scores_2026-09-16.parquet and prints the per-cell table.

  uv run python -m eval.scripts.build_eval.cn_survey_score --run full
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from eval.scripts.build_eval import fit_urn  # noqa: E402
from eval.scripts.build_eval.graded_urn import FLAGS6  # noqa: E402

RUN_DIR = SRC / "eval/data/urn_runs/cn_survey"
S_METRICS = SRC / "eval/data/urn_runs/e1_prodregime/prodregime_metrics.json"
OUT = SRC / "eval/data/community_notes/survey_scores_2026-09-16.parquet"
DIRECTIONAL = ("5", "4", "2", "1")
TRUE_TAIL = 2.0          # "clearly true" cutoff, stated on the table
BOOT_REPS, BOOT_SEED = 2000, 20260916


def arm_s() -> tuple[dict[str, float], float]:
    m = json.loads(S_METRICS.read_text())["conditions"]["S"]
    return {k: float(m["weights"][k]) for k in FLAGS6}, float(m["threshold"])


def flags_of(rec: dict) -> collections.Counter:
    f = collections.Counter()
    for d in rec.get("docs") or []:
        dd = (d.get("read") or {}).get("direction")
        if dd in ("5", "4", "3", "X", "I", "2", "1"):
            f["X" if dd == "3" else dd] += 1
    n = sum(f.values())
    f["I"] += max(0, fit_urn.PAD_TO - n)     # pad rule: missing slots are silent
    return f


def auc_ci(score: np.ndarray, y_false: np.ndarray, post: np.ndarray, rng) -> tuple[float, tuple]:
    """AUC with FALSE as the positive class: a false claim should score LOWER, so the
    discriminant is -score. Cluster bootstrap by post id."""
    def _auc(s, y):
        pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
        if not len(pos) or not len(neg):
            return float("nan")
        lo = np.searchsorted(neg, pos, "left")
        hi = np.searchsorted(neg, pos, "right")
        return float((lo + 0.5 * (hi - lo)).sum() / (len(pos) * len(neg)))
    d = -score
    point = _auc(d, y_false)
    posts = np.unique(post)
    idx_by = {p: np.flatnonzero(post == p) for p in posts}
    boots = []
    for _ in range(BOOT_REPS):
        pick = rng.choice(posts, len(posts), replace=True)
        bi = np.concatenate([idx_by[p] for p in pick])
        a = _auc(d[bi], y_false[bi])
        if not np.isnan(a):
            boots.append(a)
    return point, tuple(np.percentile(boots, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full")
    a = ap.parse_args()

    W, THR = arm_s()
    seen = {}
    for l in (RUN_DIR / f"results_{a.run}.jsonl").open():
        r = json.loads(l)
        seen[r["claim_id"]] = r                      # last-wins
    recs = list(seen.values())
    print(f"loaded {len(recs)} claims from results_{a.run}.jsonl (last-wins)")
    print(f"Arm-S weights {W} | boundary {THR:.4f}  <- {S_METRICS}")

    rows = []
    for r in recs:
        f = flags_of(r)
        score = sum(W[k] * f[k] for k in FLAGS6)
        docs = r.get("docs") or []
        reads = [d for d in docs if d.get("read")]
        top = [d["domain"] for d in reads
               if (d["read"].get("direction") in DIRECTIONAL)][:3]
        if not top:
            top = [d["domain"] for d in docs][:3]
        rows.append({
            "claim_id": r["claim_id"], "post_id": r["post_id"], "claim": r["claim"],
            "lean": r["lean"], "label": r["label"], "month": r["month"],
            "subtopic": r["subtopic"], "post_date": r.get("post_date"),
            "query": r.get("query"), "score": score, "flag": score <= THR,
            "n_docs": len(docs), "n_reads": len(reads),
            **{f"n_{k}": int(f[k]) for k in FLAGS6},
            "n_pad_I": int(f["I"]) - sum(1 for d in reads
                                         if (d["read"].get("direction") == "I")),
            "n_fc_docs": sum(1 for d in docs if d.get("fc_domain")),
            "top_domains": ", ".join(top),
            "dossier": f"eval/data/urn_runs/cn_survey/results_{a.run}.jsonl#{r['claim_id']}",
            "error": r.get("error") or r.get("serper_error") or ""})
    df = pl.DataFrame(rows)
    df.write_parquet(OUT)
    print(f"wrote {OUT}  ({df.height} rows)")

    # ---------------- per-cell table ----------------
    rng = np.random.default_rng(BOOT_SEED)
    lines = []
    lines.append(f"Arm-S six-flag urn, weights from {S_METRICS.relative_to(SRC)} conditions.S: "
                 + " ".join(f"{k} {W[k]:+.4f}" for k in FLAGS6))
    lines.append(f"flag boundary (nested 2% FPR, Arm S) = {THR:.4f}; "
                 f"clearly-true cutoff = score > +{TRUE_TAIL:.0f}; padded to {fit_urn.PAD_TO} slots")
    lines.append("")
    lines.append("| cell | n | p10 | p25 | p50 | p75 | p90 | frac flagged | n<=boundary | n>+2 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    cells = [("left", "false"), ("left", "true"), ("right", "false"), ("right", "true")]
    for lean, lab in cells + [("all", "all")]:
        d = df if lean == "all" else df.filter((pl.col("lean") == lean) & (pl.col("label") == lab))
        s = d["score"].to_numpy()
        q = np.percentile(s, [10, 25, 50, 75, 90])
        name = "ALL" if lean == "all" else f"{lean}-{lab}"
        lines.append(f"| {name} | {len(s)} | " + " | ".join(f"{x:+.2f}" for x in q)
                     + f" | {d['flag'].mean():.3f} | {int((s <= THR).sum())} "
                       f"| {int((s > TRUE_TAIL).sum())} |")
    lines.append("")
    lines.append("| AUC (quasi-label false = positive) | n | n_false | AUC | 95% CI (cluster boot by post) |")
    lines.append("|---|---:|---:|---:|---|")
    for name, d in [("overall", df),
                    ("left", df.filter(pl.col("lean") == "left")),
                    ("right", df.filter(pl.col("lean") == "right"))]:
        s = d["score"].to_numpy()
        y = (d["label"].to_numpy() == "false").astype(int)
        post = d["post_id"].to_numpy()
        pt, (lo, hi) = auc_ci(s, y, post, rng)
        lines.append(f"| {name} | {len(s)} | {int(y.sum())} | {pt:.4f} | [{lo:.4f}, {hi:.4f}] |")
    txt = "\n".join(lines)
    print("\n" + txt)
    return txt


if __name__ == "__main__":
    main()
