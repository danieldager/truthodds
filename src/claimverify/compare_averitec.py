"""Final comparison for issue #28: ClaimCheck paper vs our claim-level verifier vs the urn.

  uv run python -m claimverify.compare_averitec \\
      --arm top3=eval/data/claimverify_runs/averitec_dev/top3 \\
      --arm all10=eval/data/claimverify_runs/averitec_dev/all10 \\
      --urn eval/data/urn_runs/averitec_dev/scores.scored.parquet

Everything is measured on the SAME 500 gold rows (491 unique claim_id; the 9 duplicate rows
inherit their claim's prediction, as in score_averitec.py). Three systems:

  paper     ClaimCheck's published 76.4% 4-class accuracy on AVeriTeC dev 500. External
            number, no per-item predictions, so no CI and no paired test.
  verifier  one entry per --arm NAME=DIR; reads <dir>/scored.parquet + <dir>/metrics.json
            (run score_averitec.py first) and <dir>/trace/*.json for uncached LLM latency
            and the date-ceiling caveat.
  urn       --urn <scored.parquet> from score_frozen_urn.py: 3-voice and 7-flag log-odds
            scores, flag = score <= threshold. Reported at the ladder's fitted
            threshold_2pct AND at the threshold matched to each verifier arm's observed FPR
            (the largest threshold whose FPR stays at or below the arm's), plus ROC AUC and
            an FPR sweep. Partial urn coverage is tolerated: urn metrics are restricted to
            covered rows and n is stated everywhere.

Binary maps are score_averitec's: A = Supported passes; B = Supported + Conflicting pass;
C = gold Conflicting rows dropped, Supported passes. CIs are 1,000-resample percentile
bootstraps over the 500 gold rows with one shared index matrix (seed 0), so paired
comparisons see the same resamples. Matched-FPR thresholds are picked on the full sample and
held fixed inside the bootstrap. Writes <out>/comparison.json and <out>/comparison.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from claimverify.config import SRC
from claimverify.score_averitec import BINARY, DEFAULT_GOLD

DEFAULT_URN = SRC / "eval/data/urn_runs/averitec_dev/smoke.scored.parquet"
DEFAULT_LADDER = SRC / "eval/data/urn_runs/e1_ctx/model_ladder/ladder.json"
DEFAULT_OUT = SRC / "eval/data/claimverify_runs/averitec_dev/comparison"
PAPER_ACC = 0.764
URN_MODELS = {"3-voice": "s3", "7-flag": "s7"}
SWEEP_FPRS = (0.02, 0.05, 0.10, 0.20)

# =============================================================================
# bootstrap / metric primitives
# =============================================================================


def boot_index(n: int, reps: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(reps, n))


def _ci(vals: np.ndarray) -> list[float]:
    v = vals[np.isfinite(vals)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [float("nan")] * 2


def rate(num: np.ndarray, den: np.ndarray, idx: np.ndarray | None = None) -> float | np.ndarray:
    """num/den over a boolean mask pair; with idx, the same over every bootstrap resample."""
    if idx is None:
        d = int(den.sum())
        return float(num.sum()) / d if d else float("nan")
    n, d = num[idx].sum(1), den[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, n / np.where(d > 0, d, 1), np.nan)


def with_ci(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> dict:
    return {"value": rate(num, den), "ci": _ci(rate(num, den, idx)), "n": int(den.sum())}


def binary_metrics(sys_flag: np.ndarray, gold_flag: np.ndarray, mask: np.ndarray,
                   idx: np.ndarray) -> dict:
    tp = sys_flag & gold_flag & mask
    return {"n": int(mask.sum()),
            "n_gold_flag": int((gold_flag & mask).sum()),
            "n_gold_pass": int((~gold_flag & mask).sum()),
            "accuracy": with_ci((sys_flag == gold_flag) & mask, mask, idx),
            "flag_recall": with_ci(tp, gold_flag & mask, idx),
            "fpr": with_ci(sys_flag & ~gold_flag & mask, ~gold_flag & mask, idx),
            "precision": with_ci(tp, sys_flag & mask, idx)}


def auc_pass(scores: np.ndarray, gold_pass: np.ndarray) -> float:
    """Mann-Whitney AUC, positives = gold PASS (fit_urn convention: true claims score high)."""
    pos, neg = int(gold_pass.sum()), int((~gold_pass).sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[gold_pass].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auc_with_ci(scores: np.ndarray, gold_pass: np.ndarray, mask: np.ndarray,
                idx: np.ndarray) -> dict:
    boots = np.array([auc_pass(scores[r[mask[r]]], gold_pass[r[mask[r]]]) for r in idx])
    return {"value": auc_pass(scores[mask], gold_pass[mask]), "ci": _ci(boots), "n": int(mask.sum())}


def pick_threshold(scores: np.ndarray, gold_pass: np.ndarray, target_fpr: float) -> float:
    """Largest threshold whose FPR (flag = score <= thr, over gold-pass rows) stays <= target."""
    trues = np.sort(scores[gold_pass])
    if not trues.size or not np.isfinite(target_fpr):
        return float("-inf")
    best = float("-inf")
    for thr in np.unique(scores):
        if np.searchsorted(trues, thr, side="right") / len(trues) <= target_fpr:
            best = float(thr)
    return best


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray, mask: np.ndarray) -> dict:
    b = int((mask & a_correct & ~b_correct).sum())
    c = int((mask & ~a_correct & b_correct).sum())
    n = b + c
    p = 1.0 if n == 0 else min(
        1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"n_pairs": int(mask.sum()), "a_only_correct": b, "b_only_correct": c,
            "n_discordant": n, "p_exact": p}


# =============================================================================
# inputs
# =============================================================================


def run_wall_s(run_dir: Path) -> tuple[float | None, str]:
    """Total wall of the harness run: the finish line of <dir>.log, else None."""
    log = run_dir.parent / f"{run_dir.name}.log"
    if log.exists():
        for line in reversed(log.read_text(errors="replace").splitlines()[-50:]):
            line = line.strip()
            if line.startswith("{") and '"wall_s"' in line:
                try:
                    return float(json.loads(line)["wall_s"]), f"{log.name} finish line"
                except (ValueError, KeyError):
                    pass
    return None, "not recorded"


def scan_traces(trace_dir: Path) -> tuple[list[float], dict]:
    """Uncached LLM call latencies and per-provider search / date-ceiling counts."""
    lat: list[float] = []
    prov: dict[str, dict] = {}
    for p in sorted(trace_dir.glob("*.json")):
        t = json.loads(p.read_text())
        for c in t.get("llm_calls") or []:
            if not c.get("cache_hit") and c.get("latency_s") is not None:
                lat.append(float(c["latency_s"]))
        for s in t.get("searches") or []:
            d = prov.setdefault(s.get("provider") or "unknown", dict.fromkeys(
                ("searches", "cached", "hits", "instrumented", "post_ceiling_hits",
                 "undated_hits", "searches_with_post_ceiling"), 0))
            pc = int(s.get("post_ceiling_hits") or 0)
            d["searches"] += 1
            d["cached"] += int(bool(s.get("cache_hit")))
            d["hits"] += int(s.get("n_results") or 0)
            d["instrumented"] += int("post_ceiling_hits" in s)
            d["post_ceiling_hits"] += pc
            d["undated_hits"] += int(s.get("undated_hits") or 0)
            d["searches_with_post_ceiling"] += int(pc > 0)
    return lat, prov


def arm_speed_cost(run_dir: Path, metrics: dict) -> tuple[dict, dict]:
    sc = metrics["speed_cost"]
    n = max(1, sc["n_claims"])
    wall, wall_src = run_wall_s(run_dir)
    k = (json.loads((run_dir / "manifest.json").read_text()).get("orchestration") or {}
         ).get("k_claims") if (run_dir / "manifest.json").exists() else None
    lat, prov = scan_traces(run_dir / "trace") if (run_dir / "trace").is_dir() else ([], {})
    speed = {
        "n_claims": sc["n_claims"], "wall_s": wall, "wall_source": wall_src,
        "throughput_claims_per_min": (sc["n_claims"] / (wall / 60)) if wall else None,
        "concurrency_k_claims": k,
        "llm_calls_per_claim": sc["llm_calls"] / n,
        "llm_cache_hit_share": sc["llm_cache_hits"] / max(1, sc["llm_calls"]),
        "serper_calls_per_claim": sc["serper_calls"] / n,
        "exa_calls_per_claim": sc["exa_calls"] / n,
        "scrapes_per_claim": sc["scrapes"] / n,
        "tokens_per_claim": {k2: v / n for k2, v in sc["tokens"].items()},
        "cost_usd_per_claim": sc["cost_usd"] / n,
        "cost_usd_total": sc["cost_usd"],
        "llm_latency_s_p50_uncached": float(np.percentile(lat, 50)) if lat else None,
        "n_uncached_llm_calls": len(lat),
        "wall_per_claim_s_p50_at_k100": sc["elapsed_s"]["p50"],
        "wall_per_claim_note": f"per-claim elapsed is concurrency-inflated at K={k}",
    }
    return speed, prov


def urn_speed_cost(raw_path: Path, manifest_path: Path) -> dict:
    if not raw_path.exists():
        return {"note": f"raw records not found at {raw_path}"}
    lat: list[float] = []
    rows = []
    for line in raw_path.open():
        r = json.loads(line)
        if r.get("querygen_latency_s") is not None:
            lat.append(float(r["querygen_latency_s"]))
        docs = r.get("results") or []
        for d in docs:
            for rr in d.get("region_reads") or []:
                if rr.get("latency_s") is not None:
                    lat.append(float(rr["latency_s"]))
        rows.append({"wall_s": r.get("wall_s") or 0.0, "cost": r.get("cost") or 0.0,
                     "llm_calls": r.get("llm_calls") or 0, "search_calls": r.get("search_calls") or 0,
                     "prompt": r.get("prompt_tok") or 0, "cached": r.get("cached_tok") or 0,
                     "completion": r.get("completion_tok") or 0, "docs": len(docs),
                     "scrapes": sum(1 for d in docs if d.get("provenance") == "scrape")})
    df = pd.DataFrame(rows)
    n = max(1, len(df))
    workers = (json.loads(manifest_path.read_text()).get("workers")
               if manifest_path.exists() else None)
    wall = float(df["wall_s"].sum()) / workers if workers else None
    return {
        "n_claims": len(df), "workers": workers,
        "wall_s": wall, "wall_source": "estimated as sum(per-claim wall) / workers",
        "throughput_claims_per_min": (len(df) / (wall / 60)) if wall else None,
        "llm_calls_per_claim": float(df["llm_calls"].mean()),
        "search_calls_per_claim": float(df["search_calls"].mean()),
        "docs_per_claim": float(df["docs"].mean()),
        "scrapes_per_claim": float(df["scrapes"].mean()),
        "tokens_per_claim": {k: float(df[k].mean()) for k in ("prompt", "cached", "completion")},
        "cost_usd_per_claim": float(df["cost"].mean()), "cost_usd_total": float(df["cost"].sum()),
        "llm_latency_s_p50_uncached": float(np.percentile(lat, 50)) if lat else None,
        "n_uncached_llm_calls": len(lat),
        "wall_per_claim_s_p50": float(df["wall_s"].median()),
    }


# =============================================================================
# assembly
# =============================================================================


def gold_frame(gold_path: Path) -> pd.DataFrame:
    g = pd.read_parquet(gold_path).copy()
    g["claim_id"] = g["claim_id"].astype(str)
    g["duplicate"] = g["claim_id"].duplicated()
    g["has_url"] = g["original_claim_url"].notna() & (g["original_claim_url"].astype(str) != "")
    parts = g["claim_date"].astype(str).str.split("-")  # AVeriTeC dates are D-M-YYYY
    g["month"] = parts.str[2] + "-" + parts.str[1].str.zfill(2)
    return g


def map_masks(gold_label: pd.Series) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """variant -> (mask of rows in the map, gold_flag over all rows)."""
    out = {}
    for v, (passes, drop) in BINARY.items():
        out[v] = (~gold_label.isin(drop).to_numpy(), ~gold_label.isin(passes).to_numpy())
    return out


def compare(arms: dict[str, Path], urn_path: Path, gold_path: Path, ladder_path: Path,
            reps: int, seed: int) -> dict:
    g = gold_frame(gold_path)
    n_rows = len(g)
    idx = boot_index(n_rows, reps, seed)
    maps = map_masks(g["gold_label"])
    gold_label = g["gold_label"].to_numpy()

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "gold": str(gold_path), "n_gold_rows": n_rows,
        "n_gold_claims": int(g["claim_id"].nunique()),
        "bootstrap": {"reps": reps, "seed": seed, "unit": "gold row", "interval": "percentile 95%"},
        "binary_maps": {v: {"pass": sorted(p), "gold_dropped": sorted(d)}
                        for v, (p, d) in BINARY.items()},
        "paper": {"system": "ClaimCheck (paper)", "n": 500, "ci": None,
                  "four_class_convention_accuracy": PAPER_ACC,
                  "note": "external number, no per-item predictions: no CI, no paired test, "
                          "and no binary or speed numbers"},
        "arms": {}, "urn": {}, "mcnemar": {}, "slices": {},
    }

    # --- verifier arms -------------------------------------------------------
    arm_state = {}
    for name, run_dir in arms.items():
        scored = pd.read_parquet(run_dir / "scored.parquet")
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert list(scored["claim_id"].astype(str)) == list(g["claim_id"]), \
            f"{name}: scored.parquet row order does not match gold"
        ok = scored["scored"].to_numpy()
        pred_cc = scored["verdict_cc"].fillna("").to_numpy()
        pred_raw = scored["verdict_raw"].fillna("").to_numpy()
        cc_correct = (pred_cc == gold_label) & ok
        entry = {"dir": str(run_dir), "n_scored_rows": int(ok.sum()),
                 "n_scored_claims": metrics["n_gold_claims"] - metrics["n_missing_claims"],
                 "four_class_convention": {
                     "accuracy": with_ci(cc_correct, ok, idx),
                     "paper_accuracy": PAPER_ACC,
                     "delta_vs_paper": rate(cc_correct, ok) - PAPER_ACC},
                 "binary": {}}
        flags = {}
        for v, (mask, gold_flag) in maps.items():
            m = mask & ok
            sys_flag = np.array([str(p) not in BINARY[v][0] for p in pred_raw]) & m
            flags[v] = (sys_flag, m)
            entry["binary"][v] = binary_metrics(sys_flag, gold_flag, m, idx)
        speed, prov = arm_speed_cost(run_dir, metrics)
        entry["speed_cost"] = speed
        entry["ceiling"] = {
            p: {"searches": d["searches"], "cached_searches": d["cached"], "hits": d["hits"],
                "date_instrumented_searches": d["instrumented"],
                **({} if not d["instrumented"] else {
                    "share_searches_with_post_ceiling_hit":
                        d["searches_with_post_ceiling"] / d["instrumented"],
                    "share_hits_post_ceiling": d["post_ceiling_hits"] / max(1, d["hits"]),
                    "share_hits_undated": d["undated_hits"] / max(1, d["hits"])})}
            for p, d in prov.items()}
        out["arms"][name] = entry
        arm_state[name] = {"cc_correct": cc_correct, "ok": ok, "flags": flags}

    # --- urn -----------------------------------------------------------------
    ladder = json.loads(ladder_path.read_text())
    urn_df = pd.read_parquet(urn_path)[["claim_id", *URN_MODELS.values()]].copy()
    urn_df["claim_id"] = urn_df["claim_id"].astype(str)
    joined = g[["claim_id"]].merge(urn_df, on="claim_id", how="left")
    covered = joined["s3"].notna().to_numpy()
    raw_path = urn_path.parent / (urn_path.name.split(".")[0] + ".jsonl")
    out["urn"] = {
        "path": str(urn_path), "raw": str(raw_path), "ladder": str(ladder_path),
        "n_covered_rows": int(covered.sum()),
        "n_covered_claims": int(joined.loc[covered, "claim_id"].nunique()),
        "coverage_note": ("urn metrics are computed on covered rows only"
                          if covered.sum() < n_rows else "full coverage"),
        "speed_cost": urn_speed_cost(raw_path, urn_path.parent / "manifest.json"),
        "models": {},
    }
    urn_state = {}
    for model, col in URN_MODELS.items():
        s = joined[col].fillna(0.0).to_numpy(float)
        thr_fit = float(ladder["models"][model]["threshold_2pct"])
        m_entry = {"score_column": col, "threshold_2pct": thr_fit,
                   "fitted": {}, "matched": {}, "auc": {}, "sweep": {}}
        urn_state[model] = {"scores": s, "ops": {}}
        for v, (mask, gold_flag) in maps.items():
            m = mask & covered
            gold_pass = ~gold_flag
            m_entry["fitted"][v] = {"threshold": thr_fit,
                                    **binary_metrics((s <= thr_fit) & m, gold_flag, m, idx)}
            m_entry["auc"][v] = auc_with_ci(s, gold_pass, m, idx)
            sweep = []
            for t in SWEEP_FPRS:
                thr = pick_threshold(s[m], gold_pass[m], t)
                b = binary_metrics((s <= thr) & m, gold_flag, m, idx)
                sweep.append({"target_fpr": t, "threshold": thr,
                              "fpr": b["fpr"]["value"], "flag_recall": b["flag_recall"]["value"],
                              "accuracy": b["accuracy"]["value"]})
            for arm in arms:
                target = out["arms"][arm]["binary"][v]["fpr"]["value"]
                thr = pick_threshold(s[m], gold_pass[m], target)
                sys_flag = (s <= thr) & m
                b = binary_metrics(sys_flag, gold_flag, m, idx)
                m_entry["matched"].setdefault(arm, {})[v] = {
                    "matched_to_fpr": target, "threshold": thr, **b}
                sweep.append({"target_fpr": target, "threshold": thr, "matched_to_arm": arm,
                              "fpr": b["fpr"]["value"], "flag_recall": b["flag_recall"]["value"],
                              "accuracy": b["accuracy"]["value"]})
                urn_state[model]["ops"][(arm, v)] = (sys_flag, m)
            m_entry["sweep"][v] = sweep
        out["urn"]["models"][model] = m_entry

    # --- paired tests --------------------------------------------------------
    names = list(arms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = arm_state[a]["ok"] & arm_state[b]["ok"]
            out["mcnemar"][f"{a} vs {b} on 4-class convention"] = mcnemar(
                arm_state[a]["cc_correct"], arm_state[b]["cc_correct"], both)
            fa, ma = arm_state[a]["flags"]["A"]
            fb, mb = arm_state[b]["flags"]["A"]
            gold_flag_a = maps["A"][1]
            out["mcnemar"][f"{a} vs {b} on binary A"] = mcnemar(
                fa == gold_flag_a, fb == gold_flag_a, ma & mb)
    gold_flag_a = maps["A"][1]
    for a in names:
        fa, ma = arm_state[a]["flags"]["A"]
        for model in URN_MODELS:
            fu, mu = urn_state[model]["ops"][(a, "A")]
            out["mcnemar"][f"{a} vs urn {model} at matched FPR on binary A"] = mcnemar(
                fa == gold_flag_a, fu == gold_flag_a, ma & mu)

    # --- slices --------------------------------------------------------------
    slice_defs = {"gold_class": g["gold_label"].astype(str),
                  "has_original_claim_url": g["has_url"].map({True: "url", False: "no url"}),
                  "claim_month": g["month"].astype(str)}
    for sname, series in slice_defs.items():
        vals = series.to_numpy()
        out["slices"][sname] = {}
        for key in sorted(pd.unique(vals)):
            sel = vals == key
            row = {"n_rows": int(sel.sum()), "systems": {}}
            for a in names:
                fa, ma = arm_state[a]["flags"]["A"]
                m = ma & sel
                row["systems"][a] = {
                    "n": int(m.sum()),
                    "binA_accuracy": rate((fa == gold_flag_a) & m, m),
                    "binA_flag_recall": rate(fa & gold_flag_a & m, gold_flag_a & m),
                    "four_class_cc_accuracy": rate(
                        arm_state[a]["cc_correct"] & arm_state[a]["ok"] & sel,
                        arm_state[a]["ok"] & sel)}
            for model in URN_MODELS:
                s = urn_state[model]["scores"]
                thr_fit = out["urn"]["models"][model]["threshold_2pct"]
                m = maps["A"][0] & covered & sel
                fu = (s <= thr_fit) & m
                row["systems"][f"urn {model} @fitted"] = {
                    "n": int(m.sum()),
                    "binA_accuracy": rate((fu == gold_flag_a) & m, m),
                    "binA_flag_recall": rate(fu & gold_flag_a & m, gold_flag_a & m)}
                for a in names:
                    fu, mu = urn_state[model]["ops"][(a, "A")]
                    m = mu & sel
                    row["systems"][f"urn {model} @matched:{a}"] = {
                        "n": int(m.sum()),
                        "binA_accuracy": rate((fu == gold_flag_a) & m, m),
                        "binA_flag_recall": rate(fu & gold_flag_a & m, gold_flag_a & m)}
            out["slices"][sname][str(key)] = row
    return out


# =============================================================================
# reporting
# =============================================================================


def _v(d: dict) -> str:
    return f"{d['value']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"


def summary(c: dict) -> str:
    o = [f"AVeriTeC dev {c['n_gold_rows']} rows / {c['n_gold_claims']} claims | "
         f"bootstrap {c['bootstrap']['reps']} reps seed {c['bootstrap']['seed']}",
         "", "4-CLASS (ClaimCheck convention: unsupported counts as Refuted)",
         f"  {'ClaimCheck (paper)':22} {PAPER_ACC:.3f}  (no CI, external)"]
    for name, a in c["arms"].items():
        fc = a["four_class_convention"]
        o.append(f"  {name:22} {_v(fc['accuracy'])}  delta vs paper {fc['delta_vs_paper']:+.3f}"
                 f"  (n={fc['accuracy']['n']})")
    for v in BINARY:
        o.append("")
        o.append(f"BINARY {v}  (pass = {', '.join(sorted(BINARY[v][0]))}"
                 f"{'; gold ' + ', '.join(sorted(BINARY[v][1])) + ' dropped' if BINARY[v][1] else ''})")
        for name, a in c["arms"].items():
            b = a["binary"][v]
            o.append(f"  {name:22} n={b['n']:3d}  acc {_v(b['accuracy'])}  recall "
                     f"{_v(b['flag_recall'])}  FPR {_v(b['fpr'])}  prec {_v(b['precision'])}")
        for model, m in c["urn"]["models"].items():
            f = m["fitted"][v]
            o.append(f"  {'urn ' + model + ' @2%':22} n={f['n']:3d}  acc {_v(f['accuracy'])}"
                     f"  recall {_v(f['flag_recall'])}  FPR {_v(f['fpr'])}"
                     f"  prec {_v(f['precision'])}  thr {f['threshold']:.3f}")
            for arm, per_map in m["matched"].items():
                q = per_map[v]
                o.append(f"  {'urn ' + model + ' ~' + arm:22} n={q['n']:3d}  acc {_v(q['accuracy'])}"
                         f"  recall {_v(q['flag_recall'])}  FPR {_v(q['fpr'])}"
                         f"  prec {_v(q['precision'])}  thr {q['threshold']:.3f}"
                         f" (target {q['matched_to_fpr']:.3f})")
            o.append(f"  {'urn ' + model + ' AUC':22} {_v(m['auc'][v])}")
    u = c["urn"]
    o += ["", f"URN COVERAGE {u['n_covered_rows']}/{c['n_gold_rows']} rows "
              f"({u['n_covered_claims']} claims) - {u['coverage_note']}",
          "", "URN SWEEP (map A, flag = score <= threshold)"]
    for model, m in u["models"].items():
        for s in m["sweep"]["A"]:
            tag = f"match {s['matched_to_arm']}" if s.get("matched_to_arm") else f"FPR {s['target_fpr']:.2f}"
            o.append(f"  {model:8} {tag:14} thr {s['threshold']:8.3f}  FPR {s['fpr']:.3f}  "
                     f"recall {s['flag_recall']:.3f}  acc {s['accuracy']:.3f}")
    o += ["", "MCNEMAR (exact binomial on discordant pairs)"]
    for k, m in c["mcnemar"].items():
        o.append(f"  {k:52} n={m['n_pairs']:3d}  {m['a_only_correct']}/{m['b_only_correct']} "
                 f"discordant {m['n_discordant']}  p={m['p_exact']:.4f}")
    o += ["", "SPEED AND COST PER CLAIM"]
    for name, a in c["arms"].items():
        s = a["speed_cost"]
        tp = f"{s['throughput_claims_per_min']:.1f}/min" if s["throughput_claims_per_min"] else "n/a"
        lat = f"{s['llm_latency_s_p50_uncached']:.1f}s" if s["llm_latency_s_p50_uncached"] else "n/a"
        t = s["tokens_per_claim"]
        o.append(f"  {name:16} {tp:>10}  llm {s['llm_calls_per_claim']:.1f}"
                 f" (cache {s['llm_cache_hit_share']:.0%})  serper {s['serper_calls_per_claim']:.2f}"
                 f"  exa {s['exa_calls_per_claim']:.2f}  scrapes {s['scrapes_per_claim']:.1f}"
                 f"  tok {t.get('prompt', 0):,.0f}p/{t.get('completion', 0):,.0f}c"
                 f"  ${s['cost_usd_per_claim']:.5f}  LLM p50 {lat}"
                 f"  wall/claim p50 {s['wall_per_claim_s_p50_at_k100']:.0f}s at K="
                 f"{s['concurrency_k_claims']}")
    s = u["speed_cost"]
    if s.get("n_claims"):
        tp = f"{s['throughput_claims_per_min']:.1f}/min" if s["throughput_claims_per_min"] else "n/a"
        lat = f"{s['llm_latency_s_p50_uncached']:.1f}s" if s["llm_latency_s_p50_uncached"] else "n/a"
        t = s["tokens_per_claim"]
        o.append(f"  {'urn':16} {tp:>10}  llm {s['llm_calls_per_claim']:.1f}"
                 f"  serper {s['search_calls_per_claim']:.2f}  docs {s['docs_per_claim']:.1f}"
                 f"  scrapes {s['scrapes_per_claim']:.1f}"
                 f"  tok {t['prompt']:,.0f}p/{t['completion']:,.0f}c"
                 f"  ${s['cost_usd_per_claim']:.5f}  LLM p50 {lat}"
                 f"  wall/claim p50 {s['wall_per_claim_s_p50']:.0f}s at {s['workers']} workers")
    o += ["", "DATE CEILING (verifier searches)"]
    for name, a in c["arms"].items():
        for p, d in a["ceiling"].items():
            if not d["date_instrumented_searches"]:
                o.append(f"  {name:8} {p:7} searches {d['searches']:4d}  no date instrumentation")
                continue
            o.append(f"  {name:8} {p:7} searches {d['searches']:4d}  with a post-ceiling hit "
                     f"{d['share_searches_with_post_ceiling_hit']:.1%}  post-ceiling hits "
                     f"{d['share_hits_post_ceiling']:.1%}  undated {d['share_hits_undated']:.1%}")
    return "\n".join(o)


def markdown(c: dict) -> str:
    arms = c["arms"]
    first = next(iter(arms))
    lines = [
        "# AVeriTeC dev 500: paper, claim verifier, log-odds urn", "",
        f"All three systems are scored on the same {c['n_gold_rows']} gold rows "
        f"({c['n_gold_claims']} unique claims, the 9 duplicate rows inherit their claim's "
        "prediction). Intervals are 95% percentile bootstraps over the gold rows, "
        f"{c['bootstrap']['reps']} resamples, seed {c['bootstrap']['seed']}. The ClaimCheck "
        "number is taken from the paper, so it has no interval and cannot enter a paired test.",
    ]
    if c["urn"]["n_covered_rows"] < c["n_gold_rows"]:
        lines += ["", f"Read the urn rows with care: the urn file used here covers only "
                      f"{c['urn']['n_covered_rows']} of the {c['n_gold_rows']} gold rows "
                      f"({c['urn']['n_covered_claims']} claims), so every urn number below is "
                      "computed on that subset and its intervals are wide."]
    lines += [
        "", "## Four-class accuracy, ClaimCheck convention", "",
        "The convention maps our unsupported verdict to Refuted, which is what the paper does "
        "when it finds no support.", "",
        "| system | 4-class accuracy | 95% CI |", "| --- | --- | --- |",
        f"| ClaimCheck (paper) | {PAPER_ACC:.3f} | not reported |"]
    for name, a in arms.items():
        acc = a["four_class_convention"]["accuracy"]
        lines.append(f"| {name} | {acc['value']:.3f} | {acc['ci'][0]:.3f} to {acc['ci'][1]:.3f} |")
    lines += ["", "## Binary flag decision, map A (Supported passes, everything else flags)", "",
              "| system | n | accuracy | flag recall | FPR | precision |", "| --- | --- | --- | --- | --- | --- |"]
    for name, a in arms.items():
        b = a["binary"]["A"]
        lines.append(f"| {name} | {b['n']} | {b['accuracy']['value']:.3f} | "
                     f"{b['flag_recall']['value']:.3f} | {b['fpr']['value']:.3f} | "
                     f"{b['precision']['value']:.3f} |")
    for model, m in c["urn"]["models"].items():
        f = m["fitted"]["A"]
        lines.append(f"| urn {model} at the fitted 2% threshold | {f['n']} | "
                     f"{f['accuracy']['value']:.3f} | {f['flag_recall']['value']:.3f} | "
                     f"{f['fpr']['value']:.3f} | {f['precision']['value']:.3f} |")
        for arm, per_map in m["matched"].items():
            q = per_map["A"]
            lines.append(f"| urn {model} matched to {arm}'s FPR | {q['n']} | "
                         f"{q['accuracy']['value']:.3f} | {q['flag_recall']['value']:.3f} | "
                         f"{q['fpr']['value']:.3f} | {q['precision']['value']:.3f} |")
    u = c["urn"]
    lines += ["", f"Urn coverage is {u['n_covered_rows']} of {c['n_gold_rows']} rows "
                  f"({u['n_covered_claims']} claims); {u['coverage_note']}. "
                  "The urn flags a claim when its log-odds score sits at or below the threshold, "
                  "so a matched comparison holds the false positive rate at or below the "
                  "verifier's and asks what recall is left.", "",
              "## Urn ranking quality", ""]
    for model, m in u["models"].items():
        a = m["auc"]["A"]
        lines.append(f"ROC AUC for {model} on map A is {a['value']:.3f} "
                     f"({a['ci'][0]:.3f} to {a['ci'][1]:.3f}, n={a['n']}).")
    lines += ["", "| model | operating point | threshold | FPR | flag recall |",
              "| --- | --- | --- | --- | --- |"]
    for model, m in u["models"].items():
        for s in m["sweep"]["A"]:
            tag = (f"matched to {s['matched_to_arm']}" if s.get("matched_to_arm")
                   else f"FPR budget {s['target_fpr']:.2f}")
            lines.append(f"| {model} | {tag} | {s['threshold']:.3f} | {s['fpr']:.3f} | "
                         f"{s['flag_recall']:.3f} |")
    lines += ["", "## Paired tests", "",
              "McNemar with an exact binomial on the discordant pairs.", "",
              "| comparison | discordant | split | p |", "| --- | --- | --- | --- |"]
    for k, m in c["mcnemar"].items():
        lines.append(f"| {k} | {m['n_discordant']} | {m['a_only_correct']} / "
                     f"{m['b_only_correct']} | {m['p_exact']:.4f} |")
    lines += ["", "## Speed and cost per claim", "",
              "| system | claims per minute | LLM calls | search calls | scrapes | "
              "tokens in | cost | uncached LLM p50 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, a in arms.items():
        s = a["speed_cost"]
        tp = f"{s['throughput_claims_per_min']:.1f}" if s["throughput_claims_per_min"] else "n/a"
        lat = f"{s['llm_latency_s_p50_uncached']:.1f}s" if s["llm_latency_s_p50_uncached"] else "n/a"
        lines.append(f"| {name} | {tp} | {s['llm_calls_per_claim']:.1f} | "
                     f"{s['serper_calls_per_claim']:.2f} Serper plus {s['exa_calls_per_claim']:.2f} Exa | "
                     f"{s['scrapes_per_claim']:.1f} | {s['tokens_per_claim'].get('prompt', 0):,.0f} | "
                     f"${s['cost_usd_per_claim']:.5f} | {lat} |")
    s = u["speed_cost"]
    if s.get("n_claims"):
        tp = f"{s['throughput_claims_per_min']:.1f}" if s["throughput_claims_per_min"] else "n/a"
        lat = f"{s['llm_latency_s_p50_uncached']:.1f}s" if s["llm_latency_s_p50_uncached"] else "n/a"
        lines.append(f"| urn | {tp} | {s['llm_calls_per_claim']:.1f} | "
                     f"{s['search_calls_per_claim']:.2f} Serper | {s['scrapes_per_claim']:.1f} | "
                     f"{s['tokens_per_claim']['prompt']:,.0f} | ${s['cost_usd_per_claim']:.5f} | {lat} |")
    a0 = arms[first]["speed_cost"]
    lines += ["", f"The verifier's per-claim elapsed time is not a latency number. At K="
                  f"{a0['concurrency_k_claims']} claims in flight the median claim takes "
                  f"{a0['wall_per_claim_s_p50_at_k100']:.0f}s of wall, which is queueing as much "
                  "as work; the per-call number in the table is the median latency of uncached "
                  "LLM calls, which is what a single claim would feel."]
    if u["speed_cost"].get("wall_source"):
        lines += ["", f"Verifier throughput comes from the harness wall clock "
                      f"({a0['wall_source']}); the urn's is {u['speed_cost']['wall_source']}, "
                      f"at {u['speed_cost']['workers']} workers, so treat it as an estimate."]
    lines += ["", "## Date ceiling caveat", ""]
    for name, a in arms.items():
        d = a["ceiling"].get("serper")
        if d and d["date_instrumented_searches"]:
            lines.append(f"For {name}, {d['share_searches_with_post_ceiling_hit']:.1%} of "
                         f"{d['searches']} Serper searches returned at least one hit published "
                         f"after the claim's date ceiling, {d['share_hits_post_ceiling']:.1%} of "
                         f"all hits were post-ceiling, and {d['share_hits_undated']:.1%} of hits "
                         "carried no date at all, so the ceiling is a filter with leaks and not "
                         "a guarantee.")
        e = a["ceiling"].get("exa")
        if e and not e["date_instrumented_searches"]:
            lines.append(f"The {e['searches']} Exa searches in {name} are not date instrumented, "
                         "so their leakage is unmeasured rather than zero.")
    gc = c["slices"]["gold_class"]
    systems = list(next(iter(gc.values()))["systems"])
    lines += ["", "## Slices", "",
              "Binary A flag recall by gold class, on the rows each system covers. The other two "
              "pre-committed slices, whether the claim carries an original claim URL and the "
              "claim's month, are in comparison.json under slices, together with accuracy.", "",
              "| gold class | rows | " + " | ".join(systems) + " |",
              "| --- | --- | " + " | ".join("---" for _ in systems) + " |"]
    for key, row in gc.items():
        cells = []
        for s in systems:
            r = row["systems"].get(s, {})
            v = r.get("binA_flag_recall")
            cells.append("n/a" if v is None or not np.isfinite(v) else f"{v:.3f} (n={r['n']})")
        lines.append(f"| {key} | {row['n_rows']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                    help="verifier arm, repeatable (e.g. --arm top3=.../top3)")
    ap.add_argument("--urn", default=str(DEFAULT_URN), help="score_frozen_urn scored parquet")
    ap.add_argument("--ladder", default=str(DEFAULT_LADDER))
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    arms = {}
    for spec in a.arm:
        name, _, d = spec.partition("=")
        if not d:
            ap.error(f"--arm expects NAME=DIR, got {spec!r}")
        arms[name] = Path(d)
    c = compare(arms, Path(a.urn), Path(a.gold), Path(a.ladder), a.reps, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(c, indent=1, default=float))
    (out / "comparison.md").write_text(markdown(c))
    print(summary(c))
    print(f"\n-> {out / 'comparison.json'}\n-> {out / 'comparison.md'}")


if __name__ == "__main__":
    main()
