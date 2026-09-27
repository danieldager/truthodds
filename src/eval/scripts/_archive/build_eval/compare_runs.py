"""Paired comparison of two urn runs over the SAME claims.

Both runs must cover the same review_urls (guaranteed here by the seeded shuffle
+ identical --limit), so any metric difference is the instrument change and not
sampling luck. Restricts to the intersection before scoring, and breaks the
result down by claim MODE — a mode-aware prompt should move the attribution
claims and leave the assertion claims roughly where they were.

    uv run python -m eval.scripts.build_eval.compare_runs \
        -a eval/data/urn_runs/e1_ctx/results-00.v5-baseline.jsonl \
        -b eval/data/urn_runs/e1_ctx/results-00.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from eval.scripts.build_eval.fit_urn import (FLAG_TO_VOICE, K_FOLDS, auc,
                                             fold_of, out_of_fold,
                                             recall_at_fpr)

DIRECTIONAL = {"5", "4", "1", "2"}


def load(path: Path, keep: set | None = None) -> dict:
    out = {}
    for line in path.open():
        r = json.loads(line)
        if r.get("excluded") or not r.get("results"):
            continue
        if r.get("veracity") not in (1, 2, 4, 5):
            continue
        if keep is not None and r["review_url"] not in keep:
            continue
        c = collections.Counter()
        for d in r["results"]:
            v = FLAG_TO_VOICE.get((d.get("read") or {}).get("direction"))
            if v:
                c[v] += 1
        n = sum(c.values())
        if not n:
            continue
        out[r["review_url"]] = {
            "y": 1 if r["veracity"] >= 4 else 0,
            "n_t": c["supports"], "n_f": c["refutes"],
            "n_e": max(0, n - c["supports"] - c["refutes"]),
            "mode": r.get("claim_mode") or "?",
            "flags": collections.Counter((d.get("read") or {}).get("direction")
                                         for d in r["results"]),
            "fold": fold_of(r["review_url"], K_FOLDS),
        }
    return out


def metrics(rows: list[dict]) -> tuple[float, float, float]:
    if len({r["y"] for r in rows}) < 2 or len(rows) < 20:
        return float("nan"), float("nan"), float("nan")
    oof = out_of_fold(rows)
    rec, fpr, _ = recall_at_fpr(oof, 0.02)
    return auc(oof), rec, fpr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", required=True, type=Path, help="baseline run")
    ap.add_argument("-b", required=True, type=Path, help="new run")
    args = ap.parse_args()

    A0, B0 = load(args.a), load(args.b)
    both = set(A0) & set(B0)
    A = {k: v for k, v in A0.items() if k in both}
    B = {k: v for k, v in B0.items() if k in both}
    print(f"A={len(A0)}  B={len(B0)}  paired intersection={len(both)}\n")

    print(f"{'stratum':28s} {'n':>5s}  {'AUC A':>7s} {'AUC B':>7s} {'delta':>7s}   "
          f"{'recall A':>8s} {'recall B':>8s}")
    strata = [("OVERALL", lambda r: True)]
    for m in sorted({r["mode"] for r in B.values()}):
        strata.append((f"mode={m}", lambda r, m=m: r["mode"] == m))
    for name, pred in strata:
        ka = [v for k, v in A.items() if pred(B[k])]   # stratify by B's mode both sides
        kb = [v for k, v in B.items() if pred(v)]
        aa, ra, _ = metrics(ka)
        ab, rb, _ = metrics(kb)
        d = ab - aa
        print(f"{name:28s} {len(kb):5d}  {aa:7.3f} {ab:7.3f} {d:+7.3f}   "
              f"{ra:8.3f} {rb:8.3f}")

    print("\nflag mix shift (all paired claims):")
    fa, fb = collections.Counter(), collections.Counter()
    for k in both:
        fa.update(A[k]["flags"])
        fb.update(B[k]["flags"])
    tot_a, tot_b = sum(fa.values()), sum(fb.values())
    for f in ("5", "4", "3", "2", "1", "X", "I"):
        print(f"  {f}: {fa[f]:5d} ({fa[f]/tot_a:5.1%})  ->  {fb[f]:5d} ({fb[f]/tot_b:5.1%})")

    print("\nclaims whose urn direction FLIPPED relative to gold:")
    fixed = broke = 0
    for k in both:
        a, b = A[k], B[k]
        wa = (a["y"] == 1 and a["n_t"] < a["n_f"]) or (a["y"] == 0 and a["n_t"] > a["n_f"])
        wb = (b["y"] == 1 and b["n_t"] < b["n_f"]) or (b["y"] == 0 and b["n_t"] > b["n_f"])
        fixed += wa and not wb
        broke += wb and not wa
    print(f"  wrong in A, right in B (fixed):  {fixed}")
    print(f"  right in A, wrong in B (broke):  {broke}")


if __name__ == "__main__":
    main()
