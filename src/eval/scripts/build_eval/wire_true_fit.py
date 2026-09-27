"""Provisional two-urn fit with WIRE-TRUE as the TRUE side.

A thin driver over `fit_two_urn`: nothing here re-implements the model. The one
thing it adds is the syndication filter — the TRUE side is read from
`wire_true/results-00_collapsed.jsonl` with `origin_copy` and `syndicated_of`
documents dropped before `fit_two_urn._counts` sees the record, so the pad-to-ten
rule turns every collapsed slot into a silent document exactly as it would a slot
Serper never filled.

FALSE side = cn_false.parquet over c2_false/scores[_ext].jsonl, the same load the
2026-09-08 refit used. Transfer = fc_gold.parquet, fixed weights, no refit.
eps = 0 on wire-true (reputable-outlet claims are treated as pure true here); the
cn+feed fit is recomputed at eps 0 and at its own headline eps 0.10 beside it so
the comparison is apples to apples in one direction and reproduces
`populations/refit_results.json` in the other.

TRUE side is the 2026-09-10 DeepInfra pass (1,185 claims, 10,908 documents, 1 failed
read). The earlier 40-worker Groq pass of the same claims is superseded and its files
are `*.groq-superseded.*` in the same folder.

  uv run python -m eval.scripts.build_eval.wire_true_fit
Writes eval/data/urn_runs/wire_true/fit_provisional.json ($0, re-runnable).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.scripts.build_eval import fit_two_urn as f2  # noqa: E402
from eval.scripts.build_eval.fit_urn import PAD_TO, load_headline, load_population  # noqa: E402

WIRE = Path("eval/data/urn_runs/wire_true/results-00_collapsed.jsonl")
OUT = Path("eval/data/urn_runs/wire_true/fit_provisional.json")
POP = Path("eval/data/populations")


def load_wire(path: Path = WIRE) -> list[dict]:
    """fit_two_urn.load_urn's row shape, with the syndication collapse applied.
    origin_copy and syndicated_of documents are dropped before counting, and the
    pad-to-ten rule refills the freed slot as silent."""
    rows = []
    for line in path.open():
        r = json.loads(line)
        docs = [d for d in (r.get("results") or [])
                if not d.get("origin_copy") and d.get("syndicated_of") is None]
        c, fl = f2._counts({**r, "results": docs})
        n_pad = max(0, PAD_TO - sum(c.values()))
        c["n_e"] += n_pad
        fl["I"] += n_pad
        rows.append({"n_t": c["n_t"], "n_f": c["n_f"], "n_e": c["n_e"], "flags": dict(fl),
                     "claim": r.get("claim_text") or "", "review_url": r.get("review_url"),
                     "post_id": r.get("post_id") or r.get("review_url"),
                     "frame": None, "topic": None,
                     "ng_tier": r.get("ng_tier"), "domain": r.get("domain")})
    return rows


def fit_cell(name: str, true_rows: list[dict], false_rows: list[dict], gold: list[dict],
             eps_list: list[float]) -> list[dict]:
    p_f, p_m = f2.rates(false_rows), f2.rates(true_rows)
    pf7, pm7 = f2.flag_rates(false_rows), f2.flag_rates(true_rows)
    docs_m = sum(sum(r["flags"].values()) for r in true_rows)
    docs_f = sum(sum(r["flags"].values()) for r in false_rows)
    boot = f2.bootstrap_weights(false_rows, true_rows, eps_list)
    cells = []
    for eps in eps_list:
        w3 = f2.demix_weights(p_m, p_f, eps)
        w7 = f2.demix_flag_weights(pm7, pf7, eps, (docs_m, docs_f))
        if w3 is None or w7 is None:
            continue
        ev3 = f2.transfer_eval(w3, gold)
        ev7 = f2.transfer_eval7(w7, gold)
        ev7.pop("pairs", None)
        b = boot["by_eps"][eps]
        cells.append({
            "true_side": name, "eps": eps,
            "n_true": len(true_rows), "n_false": len(false_rows), "n_gold": len(gold),
            "n_posts_true": boot["n_posts_true"], "n_posts_false": boot["n_posts_false"],
            "docs_true": docs_m, "docs_false": docs_f,
            "rates_true_mix": p_m, "rates_false": p_f,
            "flag_rates_true_mix": pm7, "flag_rates_false": pf7,
            "three_voice": {"weights": w3, "weights_ci95_cluster": b["weights_ci"],
                            "design_effect": b["design_effect"], "fc_gold_transfer": ev3},
            "seven_flag": {"weights": w7, "weights_ci95_cluster": b["weights_ci7"],
                           "design_effect": b["design_effect7"], "fc_gold_transfer": ev7},
        })
    return cells


def line(c: dict) -> str:
    t3, t7 = c["three_voice"], c["seven_flag"]
    return (f"{c['true_side']:<12} eps {c['eps']:<5.2f} n_true {c['n_true']:<6} "
            f"w_sup {t3['weights']['n_t']:+.3f} w_ref {t3['weights']['n_f']:+.3f} "
            f"w_sil {t3['weights']['n_e']:+.3f} | 3-voice AUC {t3['fc_gold_transfer']['auc']:.3f} "
            f"rec@2% {t3['fc_gold_transfer']['recall_at_2pct_fpr']:.3f} | "
            f"7-flag AUC {t7['fc_gold_transfer']['auc']:.3f} "
            f"rec@2% {t7['fc_gold_transfer']['recall_at_2pct_fpr']:.3f}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--wire", default=str(WIRE), help="collapsed wire-true file (true side)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    wire_path, out_path = Path(args.wire), Path(args.out)
    C2 = f2.C2
    false_rows = f2.load_urn([C2 / "scores.jsonl", C2 / "scores_ext.jsonl"],
                             C2 / "fit_exclusions.json",
                             load_population(POP / "cn_false.parquet", "false"))
    feed_rows = f2.load_urn([f2.TRUE_SCORES], population=load_population(POP / "x_feed.parquet", "true"))
    wire_rows = load_wire(wire_path)
    gold = load_headline(f2.E1_RESULTS, population=load_population(POP / "fc_gold.parquet"))
    print(f"FALSE cn_false {len(false_rows)} | TRUE x_feed {len(feed_rows)} | "
          f"TRUE wire_true {len(wire_rows)} | gold {len(gold)}", flush=True)

    cells = []
    cells += fit_cell("wire_true", wire_rows, false_rows, gold, [0.0])
    cells += fit_cell("x_feed", feed_rows, false_rows, gold, [0.0, 0.10])
    for c in cells:
        print(line(c))

    per_tier = {}
    for tier in sorted({r["ng_tier"] for r in wire_rows if r.get("ng_tier")}):
        sub = [r for r in wire_rows if r["ng_tier"] == tier]
        p = f2.rates(sub)
        per_tier[tier] = {"n": len(sub), "rates": p,
                          "flag_rates": f2.flag_rates(sub),
                          "weights_eps0": f2.demix_weights(p, f2.rates(false_rows), 0.0)}
    out = {"generated": "wire_true_fit.py",
           "true_side_file": str(wire_path),
           "collapse": "origin_copy or syndicated_of dropped before counting; pad-to-10 refills the slot as silent",
           "false_side": ["eval/data/urn_runs/c2_false/scores.jsonl",
                          "eval/data/urn_runs/c2_false/scores_ext.jsonl"],
           "populations": {"false": str(POP / "cn_false.parquet"),
                           "feed": str(POP / "x_feed.parquet"),
                           "gold": str(POP / "fc_gold.parquet")},
           "pad_to": PAD_TO, "boot_reps": f2.fit_urn.BOOT_REPS, "boot_seed": f2.fit_urn.BOOT_SEED,
           "cells": cells, "wire_true_by_tier": per_tier}
    out_path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
