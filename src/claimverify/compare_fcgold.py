"""Loop vs log-odds urn on the pinned fc-gold population: do the loop's operating points
sit on the urn's ROC curve?

  cd src
  uv run python -m claimverify.compare_fcgold \
      --run eval/data/claimverify_runs/fc_gold/top3

Population = the urn's out-of-fold score table (one row per review_url, the rows the urn was
scored on). The loop run is joined on claim_id = blake2b(review_url, 8); coverage is reported
and every number is computed on the intersection.

Gold follows the mixed-as-false convention: flag-worthy = gold_true is False (this folds the
veracity-3 "unprovable" rows into flag-worthy; their count is reported). The loop gives ONE
verdict per claim, so it has three operating points (A: only supported passes; B: conflicting
passes too; R: only refuted flags). The urn gives a SCORE per claim, so it has a whole ROC
curve; it flags when score <= threshold. For each loop point the urn is read at the matched
threshold (the largest whose FPR on the full sample stays at or below the loop's observed FPR),
which is what puts both systems at the same false-positive cost, plus at the ladder's fitted
threshold_2pct. Bootstrap CIs are percentile intervals over claims sharing ONE index matrix, so
every difference is paired; matched thresholds are picked once and held fixed inside the
bootstrap. Writes <out>/comparison.json and <out>/comparison.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from claimverify.compare_averitec import (_ci, auc_with_ci, binary_metrics, boot_index,
                                          mcnemar, pick_threshold, rate, scan_traces)
from claimverify.config import SRC
from claimverify.harness import load_records
from claimverify.run_fcgold import claim_id_of

DEFAULT_RUN = SRC / "eval/data/claimverify_runs/fc_gold/top3"
DEFAULT_URN = SRC / "eval/data/urn_runs/e1_ctx/model_ladder/oof_scores.parquet"
DEFAULT_LADDER = SRC / "eval/data/urn_runs/e1_ctx/model_ladder/ladder.json"
# published fitted thresholds (ladder.json threshold_2pct); used only if the ladder is absent
FITTED_FALLBACK = {"7-flag": -3.9536, "3-voice": -4.6256}
URN_MODELS = {"7-flag": "score_7flag", "3-voice": "score_3voice"}
STATUSES = ("supported", "refuted", "unsupported", "conflicting")
POINTS = {"A": ("supported",),
          "B": ("supported", "conflicting"),
          "R": ("supported", "conflicting", "unsupported")}


# =============================================================================
# inputs
# =============================================================================


def join_population(urn_path: Path, run_dir: Path) -> tuple[pd.DataFrame, list[dict], dict]:
    """Urn rows joined to the loop's ok records; returns (rows, results, coverage)."""
    pop = pd.read_parquet(urn_path)
    pop["claim_id"] = [claim_id_of(u) for u in pop["review_url"]]
    recs = load_records(run_dir, ok_only=False)
    ok = {k: v for k, v in recs.items() if v.get("ok")}
    have = pop["claim_id"].isin(ok).to_numpy()
    df = pop[have].reset_index(drop=True)
    missing = pop.loc[~have, "review_url"].astype(str).tolist()
    cov = {"run": str(run_dir), "urn": str(urn_path),
           "n_population": int(len(pop)), "n_loop_records": len(recs), "n_loop_ok": len(ok),
           "n_loop_failed": len(recs) - len(ok), "n_joined": int(len(df)),
           "n_missing": len(missing), "missing_review_urls_first10": missing[:10]}
    return df, [ok[c]["result"] for c in df["claim_id"]], cov


def fitted_thresholds(ladder_path: Path) -> tuple[dict, str]:
    if ladder_path.exists():
        m = json.loads(ladder_path.read_text())["models"]
        return {k: float(m[k]["threshold_2pct"]) for k in URN_MODELS}, str(ladder_path)
    return dict(FITTED_FALLBACK), "hardcoded fallback (ladder.json not found)"


# =============================================================================
# metrics
# =============================================================================


def roc_curve(scores: np.ndarray, gold_flag: np.ndarray) -> dict:
    """Full ROC for flag = score <= threshold, at every distinct threshold."""
    thr = np.concatenate(([float(scores.min()) - 1.0], np.unique(scores)))
    f = np.sort(scores[gold_flag])
    p = np.sort(scores[~gold_flag])
    return {"threshold": thr.tolist(),
            "fpr": (np.searchsorted(p, thr, side="right") / max(1, len(p))).tolist(),
            "recall": (np.searchsorted(f, thr, side="right") / max(1, len(f))).tolist()}


def diff_with_ci(a_flag: np.ndarray, b_flag: np.ndarray, gold_flag: np.ndarray,
                 idx: np.ndarray) -> dict:
    """Paired recall difference a - b on gold-flag claims."""
    boots = rate(a_flag & gold_flag, gold_flag, idx) - rate(b_flag & gold_flag, gold_flag, idx)
    return {"value": rate(a_flag & gold_flag, gold_flag) - rate(b_flag & gold_flag, gold_flag),
            "ci": _ci(boots)}


# =============================================================================
# loop anatomy
# =============================================================================


def anatomy(results: list[dict], gold_flag: np.ndarray, run_dir: Path) -> dict:
    n = max(1, len(results))
    status = np.array([r.get("status") or "" for r in results])
    cross = {s: {"gold_false": int(((status == s) & gold_flag).sum()),
                 "gold_true": int(((status == s) & ~gold_flag).sum())} for s in STATUSES}
    other_status = sorted(set(status) - set(STATUSES))

    unsup = {"bar_refusal": 0, "retrieval_miss_zero_evidence": 0, "other": 0}
    fc_undated_events, fc_undated_claims = 0, 0
    for r in results:
        guards = [g.get("guard") for g in (r.get("guard_events") or [])]
        k = sum(1 for g in guards if g == "fc-undated-dropped")
        fc_undated_events += k
        fc_undated_claims += int(k > 0)
        if r.get("status") == "unsupported":
            if "close-below-bar" in guards:
                unsup["bar_refusal"] += 1
            elif not (r.get("evidence") or []):
                unsup["retrieval_miss_zero_evidence"] += 1
            else:
                unsup["other"] += 1

    ceiling = {}
    trace_dir = run_dir / "trace"
    if trace_dir.is_dir():
        _, prov = scan_traces(trace_dir)
        for p, d in prov.items():
            ceiling[p] = {"searches": d["searches"], "date_instrumented": d["instrumented"],
                          "share_searches_with_post_ceiling_hit":
                              (d["searches_with_post_ceiling"] / d["instrumented"]
                               if d["instrumented"] else None),
                          "share_hits_post_ceiling": d["post_ceiling_hits"] / max(1, d["hits"])}

    return {"status_by_gold": cross, "unexpected_statuses": other_status,
            "unsupported_breakdown": unsup,
            "fc_undated_dropped": {"events": fc_undated_events, "claims": fc_undated_claims},
            "ceiling": ceiling,
            "per_claim": {k: sum(float(r.get(k) or 0.0) for r in results) / n
                          for k in ("llm_calls", "serper_calls", "exa_calls", "cost_usd")},
            "share_stopped_exa_final": sum(1 for r in results if r.get("stopped") == "exa-final") / n}


def sign_flips(df: pd.DataFrame, results: list[dict], gold_flag: np.ndarray) -> dict:
    out = {"gold_true_refuted": [], "gold_false_supported": []}
    for i, r in enumerate(results):
        row = {"claim_id": df["claim_id"].iloc[i], "review_url": df["review_url"].iloc[i],
               "subtype": df["subtype"].iloc[i]}
        if not gold_flag[i] and r.get("status") == "refuted":
            out["gold_true_refuted"].append(row)
        if gold_flag[i] and r.get("status") == "supported":
            out["gold_false_supported"].append(row)
    return out


# =============================================================================
# assembly
# =============================================================================


def compare(run_dir: Path, urn_path: Path, ladder_path: Path, reps: int, seed: int) -> dict:
    df, results, cov = join_population(urn_path, run_dir)
    n = len(df)
    idx = boot_index(n, reps, seed)
    gold_flag = ~df["gold_true"].to_numpy(bool)
    gold_pass = ~gold_flag
    keep = np.ones(n, bool)
    status = np.array([r.get("status") or "" for r in results])
    thr_fit, thr_src = fitted_thresholds(ladder_path)

    out = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "coverage": cov,
        "bootstrap": {"reps": reps, "seed": seed, "unit": "claim", "interval": "percentile 95%"},
        "gold": {"n": n, "n_flag_worthy": int(gold_flag.sum()), "n_pass_worthy": int(gold_pass.sum()),
                 "convention": "flag-worthy = gold_true is False (mixed counts as false)",
                 "n_unprovable_veracity3": int((df["subtype"] == "unprovable").sum()),
                 "subtype_counts": {k: int(v) for k, v in df["subtype"].value_counts().items()}},
        "loop_points": {}, "urn": {"fitted_threshold_source": thr_src, "models": {}},
        "anatomy": anatomy(results, gold_flag, run_dir),
        "sign_flips": sign_flips(df, results, gold_flag),
    }

    loop_flags = {}
    for name, passes in POINTS.items():
        f = ~np.isin(status, passes)
        loop_flags[name] = f
        out["loop_points"][name] = {"pass_statuses": list(passes),
                                    **binary_metrics(f, gold_flag, keep, idx)}

    for model, col in URN_MODELS.items():
        s = df[col].to_numpy(float)
        entry = {"score_column": col, "auc": auc_with_ci(s, gold_pass, keep, idx),
                 "roc": roc_curve(s, gold_flag),
                 "fitted": {"threshold": thr_fit[model],
                            **binary_metrics(s <= thr_fit[model], gold_flag, keep, idx)},
                 "matched": {}}
        for name, lf in loop_flags.items():
            target = out["loop_points"][name]["fpr"]["value"]
            t = pick_threshold(s, gold_pass, target)
            uf = s <= t
            entry["matched"][name] = {
                "matched_to_loop_fpr": target, "threshold": t,
                **binary_metrics(uf, gold_flag, keep, idx),
                "recall_diff_loop_minus_urn": diff_with_ci(lf, uf, gold_flag, idx),
                "mcnemar_on_gold_false": mcnemar(lf, uf, gold_flag)}
        out["urn"]["models"][model] = entry
    return out


# =============================================================================
# reporting
# =============================================================================


def _v(d: dict) -> str:
    return f"{d['value']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"


def markdown(c: dict) -> str:
    cov, g, a = c["coverage"], c["gold"], c["anatomy"]
    L = ["# fc-gold: the verify loop against the log-odds urn", "",
         f"The loop run is `{cov['run']}`; the urn scores are `{cov['urn']}`. The population is "
         f"the urn's {cov['n_population']} out-of-fold rows. The loop has "
         f"{cov['n_loop_ok']} usable records ({cov['n_loop_failed']} failed), "
         f"{cov['n_missing']} population rows are missing from it, and everything below is "
         f"computed on the {g['n']} claims both systems cover: {g['n_flag_worthy']} flag-worthy "
         f"(gold_true False, mixed counted as false, including "
         f"{g['n_unprovable_veracity3']} veracity-3 unprovable rows) and {g['n_pass_worthy']} "
         f"pass-worthy. Intervals are 95% percentile bootstraps over claims, "
         f"{c['bootstrap']['reps']} resamples, seed {c['bootstrap']['seed']}, one shared index "
         "matrix so every difference is paired."]
    if cov["missing_review_urls_first10"]:
        L += ["", "Missing from the loop run (first 10):", ""]
        L += [f"- {u}" for u in cov["missing_review_urls_first10"]]

    L += ["", "## Loop verdicts against gold", "",
          "| status | gold false | gold true |", "| --- | --- | --- |"]
    for s, row in a["status_by_gold"].items():
        L.append(f"| {s} | {row['gold_false']} | {row['gold_true']} |")

    L += ["", "## Operating points, loop against the urn at the same false positive rate", "",
          "The urn flags when its score sits at or below the threshold, so matching means taking "
          "the largest threshold whose FPR stays at or below the loop's, then asking what recall "
          "is left. A positive difference means the loop finds more of the false claims than the "
          "urn does at the same cost in false alarms.", "",
          "| point | loop recall | loop FPR | urn 7-flag recall | diff | p | urn 3-voice recall "
          "| diff | p |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, p in c["loop_points"].items():
        cells = [f"{name} (pass = {', '.join(p['pass_statuses'])})", _v(p["flag_recall"]),
                 _v(p["fpr"])]
        for model in URN_MODELS:
            m = c["urn"]["models"][model]["matched"][name]
            cells += [_v(m["flag_recall"]), _v(m["recall_diff_loop_minus_urn"]),
                      f"{m['mcnemar_on_gold_false']['p_exact']:.3f}"]
        L.append("| " + " | ".join(cells) + " |")

    L += ["", "| point | loop accuracy | loop precision | urn 7-flag threshold | urn 7-flag FPR "
              "| urn 3-voice threshold | urn 3-voice FPR |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, p in c["loop_points"].items():
        cells = [name, _v(p["accuracy"]), _v(p["precision"])]
        for model in URN_MODELS:
            m = c["urn"]["models"][model]["matched"][name]
            cells += [f"{m['threshold']:.3f}", f"{m['fpr']['value']:.3f}"]
        L.append("| " + " | ".join(cells) + " |")

    L += ["", "## The urn on its own terms", ""]
    for model, m in c["urn"]["models"].items():
        f = m["fitted"]
        L.append(f"ROC AUC for {model} is {_v(m['auc'])} on n={m['auc']['n']}. At its published "
                 f"fitted threshold {f['threshold']:.4f} it flags with recall "
                 f"{f['flag_recall']['value']:.3f} at FPR {f['fpr']['value']:.3f} "
                 f"(precision {f['precision']['value']:.3f}, accuracy "
                 f"{f['accuracy']['value']:.3f}).")
    L += ["", f"Fitted thresholds come from {c['urn']['fitted_threshold_source']}. The full ROC "
              "curves, one point per distinct score, are in comparison.json under "
              "urn.models.<model>.roc."]

    u, per = a["unsupported_breakdown"], a["per_claim"]
    L += ["", "## Loop anatomy", "",
          f"Of the {sum(u.values())} unsupported verdicts, {u['bar_refusal']} carry a "
          f"close-below-bar guard event (the bar refused a close the model wanted), "
          f"{u['retrieval_miss_zero_evidence']} reached the end with no evidence at all "
          f"(a retrieval miss), and {u['other']} are neither.",
          "", f"The fc-undated drop fired {a['fc_undated_dropped']['events']} times across "
              f"{a['fc_undated_dropped']['claims']} claims.",
          "", f"Per claim the loop makes {per['llm_calls']:.3f} LLM calls, "
              f"{per['serper_calls']:.3f} Serper searches and {per['exa_calls']:.3f} Exa "
              f"searches, and costs ${per['cost_usd']:.5f}. "
              f"{a['share_stopped_exa_final']:.3f} of claims stop at exa-final."]
    for p, d in a["ceiling"].items():
        if d["share_searches_with_post_ceiling_hit"] is None:
            L.append(f"The {d['searches']} {p} searches are not date instrumented.")
        else:
            L.append(f"Of {d['date_instrumented']} instrumented {p} searches, "
                     f"{d['share_searches_with_post_ceiling_hit']:.3f} returned at least one hit "
                     f"published after the claim's date ceiling "
                     f"({d['share_hits_post_ceiling']:.3f} of all hits).")

    sf = c["sign_flips"]
    L += ["", "## Sign flips", "",
          f"{len(sf['gold_true_refuted'])} true claims were refuted and "
          f"{len(sf['gold_false_supported'])} false claims were supported."]
    for key, label in (("gold_true_refuted", "gold true, loop refuted"),
                       ("gold_false_supported", "gold false, loop supported")):
        if sf[key]:
            L += ["", f"{label}:", ""]
            L += [f"- `{r['claim_id']}` {r['review_url']}" for r in sf[key]]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--urn", default=str(DEFAULT_URN))
    ap.add_argument("--ladder", default=str(DEFAULT_LADDER))
    ap.add_argument("--out", default="")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    run = Path(a.run)
    c = compare(run, Path(a.urn), Path(a.ladder), a.reps, a.seed)
    out = Path(a.out) if a.out else run / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(c, indent=1, default=float))
    md = markdown(c)
    (out / "comparison.md").write_text(md)
    print(md)
    print(f"-> {out / 'comparison.json'}\n-> {out / 'comparison.md'}")


if __name__ == "__main__":
    main()
