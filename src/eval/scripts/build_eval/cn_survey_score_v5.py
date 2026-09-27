"""Score the CN-survey read-v5 SEVEN-flag re-read with the FROZEN instrument (2026-09-18).

Weights + boundary are READ from `eval/data/urn_runs/e1_ctx/weights_v5_ceiling_bal3000.json`
(the training-mode ceiling refit, produced by fit_ceiling_weights.py), never retyped. This is
the seven-flag (5/4/3/X/I/2/1) companion to cn_survey_score.py, which scores the older six-flag
Arm-S urn; the frozen survey pool uses THIS one. Scoring convention is fit_urn's: every claim
padded to PAD_TO=10 slots with I, score = sum(count x weight), FLAG when score <= boundary
(-4.0115).

The read-v5 reads live in a separate re-read file (cn_survey/reread_v5/), keyed by
(claim_id, doc rank); they are joined back onto the original dossiers (results_full.jsonl,
which carry the prepped docs + claim metadata) with no retrieval rerun.

Writes:
  eval/data/community_notes/survey_scores_2026-09-17_v5.parquet  (six-flag cols + n_3 + prompt_hash)
  eval/data/urn_runs/cn_survey/results_full_v5.jsonl             (dossiers with v5 reads)

  uv run python -m eval.scripts.build_eval.cn_survey_score_v5

$0 -- rescores cached reads, no API calls.
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
from eval.scripts.build_eval.cn_survey_score import auc_ci, BOOT_SEED, DIRECTIONAL, TRUE_TAIL  # noqa: E402

WEIGHTS = SRC / "eval/data/urn_runs/e1_ctx/weights_v5_ceiling_bal3000.json"
DOSS = SRC / "eval/data/urn_runs/cn_survey/results_full.jsonl"
V5READS = SRC / "eval/data/urn_runs/cn_survey/reread_v5/v5__DeepSeek-V4-Flash.jsonl"
OLD = SRC / "eval/data/community_notes/survey_scores_2026-09-16.parquet"
OUT = SRC / "eval/data/community_notes/survey_scores_2026-09-17_v5.parquet"
DOSS_V5 = SRC / "eval/data/urn_runs/cn_survey/results_full_v5.jsonl"
FLAGS7 = ("5", "4", "3", "X", "I", "2", "1")
PROMPT_HASH = "92404a300e14"


def load_weights(path: Path) -> tuple[dict[str, float], float]:
    wj = json.loads(Path(path).read_text())
    return {k: float(wj["weights"][k]) for k in FLAGS7}, float(wj["threshold"])


def build(weights: Path, doss: Path, v5reads: Path, old: Path,
          out: Path, doss_v5: Path) -> pl.DataFrame:
    W, THR = load_weights(weights)

    dossiers = {}                                    # original dossiers, last-wins
    for l in Path(doss).open():
        r = json.loads(l)
        dossiers[r["claim_id"]] = r
    v5 = {}                                          # v5 reads keyed by (claim_id, rank)
    for l in Path(v5reads).open():
        r = json.loads(l)
        for d in r.get("docs") or []:
            v5[(r["claim_id"], d["rank"])] = {"direction": d["new"],
                                              "evidence": d.get("new_evidence") or [],
                                              "reason": d.get("new_reason") or ""}

    rows = []
    n_with_read = 0
    with Path(doss_v5).open("w") as fout:
        for cid, r in dossiers.items():
            f = collections.Counter()
            docs = r.get("docs") or []
            reads = 0
            newdocs = []
            for i, d in enumerate(docs):
                nd = dict(d)
                rd = v5.get((cid, i))
                if d.get("read_status") == "prepped" and rd is not None:
                    nd["read"] = {"direction": rd["direction"], "evidence": rd["evidence"],
                                  "reason": rd["reason"]}
                    if rd["direction"] in FLAGS7:
                        f[rd["direction"]] += 1
                        reads += 1
                newdocs.append(nd)
            if reads:
                n_with_read += 1
            n = sum(f.values())
            f["I"] += max(0, fit_urn.PAD_TO - n)     # pad silent slots
            score = sum(W[k] * f[k] for k in FLAGS7)
            top = [d["domain"] for d in newdocs
                   if (d.get("read") or {}).get("direction") in DIRECTIONAL][:3]
            if not top:
                top = [d["domain"] for d in newdocs][:3]
            n_pad_I = int(f["I"]) - sum(1 for d in newdocs
                                        if (d.get("read") or {}).get("direction") == "I")
            rows.append({
                "claim_id": cid, "post_id": r["post_id"], "claim": r["claim"],
                "lean": r["lean"], "label": r["label"], "month": r["month"],
                "subtopic": r["subtopic"], "post_date": r.get("post_date"),
                "query": r.get("query"), "score": score, "flag": score <= THR,
                "n_docs": len(docs), "n_reads": reads,
                **{f"n_{k}": int(f[k]) for k in FLAGS7},
                "n_pad_I": n_pad_I,
                "n_fc_docs": sum(1 for d in docs if d.get("fc_domain")),
                "top_domains": ", ".join(top),
                "dossier": f"eval/data/urn_runs/cn_survey/results_full_v5.jsonl#{cid}",
                "error": r.get("error") or r.get("serper_error") or "",
                "prompt_hash": PROMPT_HASH})
            rec = {k: r.get(k) for k in ("claim_id", "post_id", "claim", "post_date", "lean",
                                         "label", "month", "subtopic", "group_id", "query")}
            rec["docs"] = newdocs
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # column order: old six-flag cols, insert n_3 after n_4, append prompt_hash
    old_cols = pl.read_parquet(old).columns
    col_order = []
    for c in old_cols:
        col_order.append(c)
        if c == "n_4":
            col_order.append("n_3")
    col_order.append("prompt_hash")
    df = pl.DataFrame(rows).select(col_order)
    df.write_parquet(out)

    print(f"weights {W}\nboundary {THR:.4f}  (from {Path(weights).name})")
    print(f"n scored {df.height} | n with >=1 read {n_with_read}")
    print(f"flagged overall {int(df['flag'].sum())} ({df['flag'].mean() * 100:.1f}%)")

    print("\n| cell | n | flagged | frac | n>+2 | p50 |")
    print("|---|--:|--:|--:|--:|--:|")
    for lean, lab in [("left", "false"), ("left", "true"), ("right", "false"),
                      ("right", "true"), ("all", "all")]:
        d = df if lean == "all" else df.filter((pl.col("lean") == lean) & (pl.col("label") == lab))
        s = d["score"].to_numpy()
        name = "ALL" if lean == "all" else f"{lean}-{lab}"
        print(f"| {name} | {len(s)} | {int(d['flag'].sum())} | {d['flag'].mean():.3f} | "
              f"{int((s > TRUE_TAIL).sum())} | {np.median(s):+.2f} |")

    rng = np.random.default_rng(BOOT_SEED)
    print("\n| AUC (quasi false=pos) | n | n_false | AUC v5 | 95% CI |")
    print("|---|--:|--:|--:|---|")
    for name, d in [("overall", df), ("left", df.filter(pl.col("lean") == "left")),
                    ("right", df.filter(pl.col("lean") == "right"))]:
        s = d["score"].to_numpy()
        y = (d["label"].to_numpy() == "false").astype(int)
        post = d["post_id"].to_numpy()
        pt, (lo, hi) = auc_ci(s, y, post, rng)
        print(f"| {name} | {len(s)} | {int(y.sum())} | {pt:.4f} | [{lo:.4f}, {hi:.4f}] |")
    print(f"\nwrote {out}\nwrote {doss_v5}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--dossiers", type=Path, default=DOSS)
    ap.add_argument("--v5-reads", type=Path, default=V5READS)
    ap.add_argument("--old", type=Path, default=OLD)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dossier-out", type=Path, default=DOSS_V5)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting the pinned frozen outputs")
    a = ap.parse_args()
    for p, pinned in ((a.out, OUT), (a.dossier_out, DOSS_V5)):
        if p.resolve() == pinned.resolve() and not a.force:
            sys.exit(f"refusing to overwrite the pinned frozen file {pinned}: it is the frozen "
                     f"(2026-09-17) survey score set. Pass --force to regenerate it in place, or "
                     f"--out / --dossier-out PATH to write elsewhere (the $0 reproduction gate).")
    build(a.weights, a.dossiers, a.v5_reads, a.old, a.out, a.dossier_out)


if __name__ == "__main__":
    main()
