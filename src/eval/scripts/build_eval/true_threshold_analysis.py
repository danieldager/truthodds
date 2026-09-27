"""Pick the DISPLAY-ONLY "true" threshold for the survey selection board (Daniel 2026-09-25).

The frozen instrument flags a claim FALSE when score <= -4.0115 (weights_v5_ceiling_bal3000.json;
untouched here). Everything above that used to show as "true", including mixed-evidence claims
barely over the line. This script finds a second cut T so the board has three bands:
false (score <= -4.0115), unclear (-4.0115 < score < T), true (score >= T).

Rule, fixed before computing (clog/250926.md 13:05): T = smallest cut on the 0.5 grid [-4, +10]
with (i) dev precision of "true" >= 0.90 on the bal3000 slice (out-of-fold scores, the same scores
the frozen boundary was picked on) AND (ii) the Pro second reader says "false" for <= 10% of survey
pool claims with score >= T (US 2026-09-25 pool union pool656, deduped by claim_id). If (i) and (ii)
pick different cuts, the larger wins.

Three tables are printed:
  A  dev slice, per cut T: precision of true, gold-true retained, share of all claims in the middle band
  B  survey pool, per 2-point score bin above the boundary: second reader false/unclear/true shares and
     the mean contradicting (codes 1+2) and supporting (4+5) reads
  C  reviewers' votes on posts in *_true cells, select vs reject by score bin (small n, indicative)

  uv run python -m eval.scripts.build_eval.true_threshold_analysis
$0 -- refits cached reads (read hash 92404a300e14), no API calls.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from eval.scripts.build_eval import fit_urn, graded_urn
from eval.scripts.build_eval.fit_ceiling_weights import POPULATION, RESULTS, load_rows

SRC = Path(__file__).resolve().parents[3]
DATA = SRC / "eval/data/community_notes"
WEIGHTS = SRC / "eval/data/urn_runs/e1_ctx/weights_v5_ceiling_bal3000.json"
POOLS = [DATA / "survey_pool_us_scored_2026-09-25.parquet", DATA / "survey_pool656_scored_2026-09-22.parquet"]
VOTE_LOOKUP = POOLS + [DATA / "survey_pool500_scored_2026-09-22.parquet"]
OUT = DATA / "true_threshold_2026-09-25.json"
GRID = [x / 2 for x in range(-8, 21)]          # -4.0 .. +10.0 step 0.5
MIN_PREC, MAX_SO_FALSE = 0.90, 0.10


def dev_table(boundary: float) -> tuple[list[dict], str]:
    rows = load_rows(RESULTS, POPULATION)
    y = np.array([r["y"] for r in rows]); mid = np.array([r["mid"] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    C7 = np.array([graded_urn.flag_counts(r) for r in rows], dtype=float).T
    s = fit_urn.oof_np(C7, y, mid, fold)             # out-of-fold, per-fold weights
    W = json.loads(WEIGHTS.read_text())["weights"]
    s_in = np.array([sum(W[k] * c for k, c in zip(graded_urn.FLAGS7, C7[:, i])) for i in range(len(rows))])
    n, n_true = len(y), int(y.sum())
    out = []
    for T in GRID:
        row = {"T": T}
        for tag, sc in (("oof", s), ("insample", s_in)):
            hi = sc >= T
            row[f"prec_true_{tag}"] = float(y[hi].mean()) if hi.any() else None
            row[f"n_above_{tag}"] = int(hi.sum())
            row[f"true_retained_{tag}"] = float((hi & (y == 1)).sum() / n_true)
            row[f"middle_share_{tag}"] = float(((sc > boundary) & (sc < T)).sum() / n)
        out.append(row)
    return out, f"n {n} (T {n_true} / F {n - n_true}), mid rows {int(mid.sum())}"


def load_pool() -> pl.DataFrame:
    cols = ["claim_id", "post_id", "score", "so_verdict", "n_1", "n_2", "n_3", "n_4", "n_5", "n_X", "n_I"]
    d = pl.concat([pl.read_parquet(p).select(cols) for p in POOLS])
    return d.unique("claim_id", keep="first")


def bins_of(boundary: float) -> list[tuple[float, float, str]]:
    edges = [boundary, -2, 0, 2, 4, 6, 8, 10, math.inf]
    return [(a, b, f"({a:+.1f}, {b:+.0f})" if b != math.inf else f"[{a:+.0f}, inf)") for a, b in zip(edges, edges[1:])]


def pool_tables(d: pl.DataFrame, boundary: float) -> tuple[list[dict], list[dict]]:
    by_bin = []
    for a, b, lab in bins_of(boundary):
        g = d.filter((pl.col("score") > a) & (pl.col("score") < b if b != math.inf else pl.lit(True)))
        n = g.height
        vc = {v: int((g["so_verdict"] == v).sum()) for v in ("false", "unclear", "true")}
        by_bin.append({"bin": lab, "n": n, **{f"so_{k}": (v / n if n else None) for k, v in vc.items()},
                       "mean_contra_1_2": float((g["n_1"] + g["n_2"]).mean()) if n else None,
                       "mean_support_4_5": float((g["n_4"] + g["n_5"]).mean()) if n else None,
                       "mean_mixed_3": float(g["n_3"].mean()) if n else None})
    by_cut = []
    for T in GRID:
        g = d.filter(pl.col("score") >= T)
        n = g.height
        by_cut.append({"T": T, "n_above": n,
                       "so_false_share": float((g["so_verdict"] == "false").mean()) if n else None,
                       "so_true_share": float((g["so_verdict"] == "true").mean()) if n else None,
                       "middle_n": d.filter((pl.col("score") > boundary) & (pl.col("score") < T)).height})
    return by_bin, by_cut


def vote_table(votes_dir: Path, boundary: float) -> list[dict]:
    look = pl.concat([pl.read_parquet(p).select("claim_id", "score") for p in VOTE_LOOKUP]).unique("claim_id", keep="first")
    sc = dict(zip(look["claim_id"], look["score"]))
    recs = []
    for f in sorted(votes_dir.glob("*.json")):
        for v in json.loads(f.read_text())["votes"]:
            if not (v.get("cell") or "").endswith("_true") or not v.get("claims"):
                continue
            s = sc.get(v["claims"][0])                 # the post's cell follows its first claim
            if s is not None:
                recs.append({"reviewer": f.stem, "score": s, "vote": v["vote"]})
    out = []
    for a, b, lab in bins_of(boundary):
        g = [r for r in recs if a < r["score"] < b]
        cnt = {k: sum(r["vote"] == k for r in g) for k in ("select", "reject", "flag", "abstain")}
        dec = cnt["select"] + cnt["reject"]
        out.append({"bin": lab, "n": len(g), **cnt, "select_rate": cnt["select"] / dec if dec else None,
                    "by_reviewer": {w: f"{sum(r['vote'] == 'select' for r in g if r['reviewer'] == w)}s/"
                                       f"{sum(r['vote'] == 'reject' for r in g if r['reviewer'] == w)}r"
                                    for w in sorted({r['reviewer'] for r in g})}})
    return out


def f(x, p=3):
    return "-" if x is None else f"{x:.{p}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--votes", type=Path, default=None, help="dir of reviewer vote exports (*.json); table C skipped if absent")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    boundary = float(json.loads(WEIGHTS.read_text())["threshold"])

    dev, devinfo = dev_table(boundary)
    print(f"A. DEV bal3000 ({devinfo}); oof = out-of-fold (used by the rule), insample = frozen weights")
    print("| T | prec true oof | n>=T | true kept | middle share | prec insample | middle insample |")
    for r in dev:
        print(f"| {r['T']:+.1f} | {f(r['prec_true_oof'])} | {r['n_above_oof']} | {f(r['true_retained_oof'])} | "
              f"{f(r['middle_share_oof'])} | {f(r['prec_true_insample'])} | {f(r['middle_share_insample'])} |")

    pool = load_pool()
    by_bin, by_cut = pool_tables(pool, boundary)
    print(f"\nB1. SURVEY POOL (US 0925 union pool656, {pool.height} claims): second reader by score bin above {boundary:.4f}")
    print("| bin | n | SO false | SO unclear | SO true | mean contra(1+2) | mean support(4+5) | mean mixed(3) |")
    for r in by_bin:
        print(f"| {r['bin']} | {r['n']} | {f(r['so_false'], 2)} | {f(r['so_unclear'], 2)} | {f(r['so_true'], 2)} | "
              f"{f(r['mean_contra_1_2'], 2)} | {f(r['mean_support_4_5'], 2)} | {f(r['mean_mixed_3'], 2)} |")
    print("\nB2. SURVEY POOL per cut: second reader 'false' share among claims with score >= T")
    print("| T | n>=T | SO false | SO true | n middle |")
    for r in by_cut:
        print(f"| {r['T']:+.1f} | {r['n_above']} | {f(r['so_false_share'])} | {f(r['so_true_share'])} | {r['middle_n']} |")

    votes = vote_table(a.votes, boundary) if a.votes else []
    if votes:
        print("\nC. REVIEWER VOTES on posts in *_true cells (first claim's score), indicative only")
        print("| bin | n | select | reject | flag | abstain | select/(sel+rej) | per reviewer |")
        for r in votes:
            print(f"| {r['bin']} | {r['n']} | {r['select']} | {r['reject']} | {r['flag']} | {r['abstain']} | "
                  f"{f(r['select_rate'], 2)} | {r['by_reviewer']} |")

    t_dev = next((r["T"] for r in dev if r["prec_true_oof"] is not None and r["prec_true_oof"] >= MIN_PREC), None)
    t_so = next((r["T"] for r in by_cut if r["so_false_share"] is not None and r["so_false_share"] <= MAX_SO_FALSE), None)
    T = max(t_dev, t_so)
    note = ("both criteria agree" if t_dev == t_so else
            f"criteria disagree (dev -> {t_dev:+.1f}, second reader -> {t_so:+.1f}); took the larger")
    print(f"\nRULE: dev precision>= {MIN_PREC} first at T={t_dev:+.1f}; second-reader false<= {MAX_SO_FALSE} first at "
          f"T={t_so:+.1f}  ->  TRUE THRESHOLD {T:+.1f} ({note})")

    a.out.write_text(json.dumps({
        "false_boundary": boundary, "true_threshold": T,
        "rule": (f"smallest T on the 0.5 grid [-4,+10] with dev (bal3000, out-of-fold) precision of true >= "
                 f"{MIN_PREC} AND survey-pool Pro second reader 'false' share <= {MAX_SO_FALSE} among claims "
                 f"with score >= T; larger if they disagree. Display only: the frozen false boundary is unchanged."),
        "rule_outcome": {"dev_T": t_dev, "second_reader_T": t_so, "note": note},
        "sources": {"dev": f"{POPULATION} + {RESULTS} ({devinfo}, read hash 92404a300e14)",
                    "pool": [str(p.relative_to(SRC)) for p in POOLS], "votes": str(a.votes) if a.votes else None},
        "tables": {"dev_by_cut": dev, "pool_by_bin": by_bin, "pool_by_cut": by_cut, "votes_true_cells_by_bin": votes},
    }, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
