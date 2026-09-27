"""Where the gap to the paper comes from: the corroboration bar and the date ceiling.

Read-only pass over saved run traces. Three things, per arm:

  (1) anatomy of the unsupported verdicts, split into the code bar refusing a close the
      RESOLVE model proposed, a retrieval miss (no full-read stance evidence at all) and
      everything else;
  (2) counterfactual CF-A "no bar" -- for any claim carrying a close-below-bar or
      close-unbacked guard event, replace the code's final status with the RESOLVE model's
      last proposed non-open status, then rescore four-class (ClaimCheck convention) and
      binary map A over the gold rows;
  (3) the ceiling leak -- among stance-bearing full-read evidence documents, the share
      dated after the claim date and the share served by a fact-checking host.

  uv run python -m claimverify.bar_counterfactual --arms top3,all10,noceil

Writes <run>/comparison/counterfactual.json, which averitec_artifact_stats.py folds into
stats.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from claimverify.compare_averitec import binary_metrics, boot_index, with_ci
from claimverify.config import SRC
from claimverify.harness import load_records
from claimverify.loop import CC_MAP, REFUTE_STANCES, SUPPORT_STANCES
from claimverify.score_averitec import BINARY, DEFAULT_GOLD

RUN = SRC / "eval/data/claimverify_runs/averitec_dev"
DEFAULT_OUT = RUN / "comparison/counterfactual.json"
ARM_DIR = {"noceil": "top3_noceil"}          # page name -> directory name
REFUSAL_GUARDS = {"close-below-bar", "close-unbacked"}
STANCES = SUPPORT_STANCES | REFUTE_STANCES
CLOSING = ("supported", "refuted", "conflicting", "unsupported")

# hosts whose business is publishing verdicts on claims; a page from one of these is
# downstream of the claim by construction, so reading it is a form of ceiling leak
FACT_CHECK_HOSTS = (
    "politifact", "factcheck.org", "snopes", "fullfact", "africacheck", "boomlive",
    "altnews", "leadstories", "healthfeedback", "checkyourfact", "factly", "thequint",
    "vishvasnews", "logically", "poynter", "sciencefeedback", "climatefeedback",
    "verafiles", "misbar", "newschecker", "factcrescendo",
    "apnews.com/article/fact-check", "apnews.com/hub/ap-fact-check",
    "reuters.com/fact-check", "usatoday.com/story/news/factcheck",
    "dpa-factchecking", "factuel.afp.com", "factcheck.afp.com", "afp.com/fact",
    "washingtonpost.com/news/fact-checker", "/fact-check", "/factcheck",
    "/fact-checking", "fact-check/", "factchecker",
)


def is_fact_check_host(url: str | None) -> bool:
    u = (url or "").lower()
    return any(h in u for h in FACT_CHECK_HOSTS)


def load_arm(run_dir: Path) -> dict[str, dict]:
    """claim_id -> loop result, successful records only."""
    return {cid: rec["result"] for cid, rec in load_records(run_dir).items()}


def trace_ledger(run_dir: Path, claim_id: str) -> list[dict]:
    p = Path(run_dir) / "trace" / f"{claim_id}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("ledger") or []


def last_proposed(ledger: list[dict]) -> str | None:
    """The last non-open status the RESOLVE model proposed, across every round."""
    out = None
    for step in ledger:
        for row in step.get("proposed") or []:
            st = (row.get("status") or "").lower()
            if st in CLOSING:
                out = st
    return out


def cf_a_status(run_dir: Path, claim_id: str, result: dict) -> str:
    """The status the loop would have recorded with no corroboration bar."""
    refusals = [g for g in result["guard_events"] if g.get("guard") in REFUSAL_GUARDS]
    if not refusals:
        return result["status"]
    return (last_proposed(trace_ledger(run_dir, claim_id))
            or refusals[-1].get("refused") or result["status"])


def unsupported_anatomy(run_dir: Path, recs: dict[str, dict]) -> dict:
    """Split the unsupported verdicts into bar refusal, retrieval miss and other."""
    counts = {"bar_refusal": 0, "retrieval_miss": 0, "other": 0}
    unsup = [c for c, r in recs.items() if r["status"] == "unsupported"]
    for cid in unsup:
        r = recs[cid]
        if any(g.get("guard") in REFUSAL_GUARDS for g in r["guard_events"]):
            counts["bar_refusal"] += 1
        elif not any(not e.get("snippet_only")
                     and (e.get("stance") or "").lower() in STANCES for e in r["evidence"]):
            counts["retrieval_miss"] += 1
        else:
            counts["other"] += 1
    n = len(unsup)
    return {"n_unsupported": n, "n_claims": len(recs), **counts,
            "shares": {k: (v / n if n else float("nan")) for k, v in counts.items()},
            "definitions": {
                "bar_refusal": "the RESOLVE model proposed a close and the code bar refused it",
                "retrieval_miss": "no full-read supporting or refuting evidence was ever read",
                "other": "read stance evidence, no refusal, the loop still ended open"}}


def _doc_index(result: dict) -> dict[str, dict]:
    docs = {}
    for rd in result["rounds"]:
        for d in rd.get("docs") or []:
            docs[d["id"]] = d
    return docs


def _parse_date(s) -> pd.Timestamp | None:
    d = pd.to_datetime(str(s), errors="coerce", utc=True) if s else None
    if d is None or pd.isna(d):
        return None
    return d.tz_localize(None)


def ceiling_leak(recs: dict[str, dict]) -> dict:
    """Share of stance-bearing read documents dated after the claim, or on a fact-check host."""
    total = after = undated = fact_check = 0
    for r in recs.values():
        claim_date = _parse_date(r.get("date"))
        docs = _doc_index(r)
        seen: dict[str, str | None] = {}
        for e in r["evidence"]:
            if e.get("snippet_only") or (e.get("stance") or "").lower() not in STANCES:
                continue
            d = docs.get(e["src"])
            if d:
                seen[d.get("url")] = d.get("date")
        for url, raw in seen.items():
            total += 1
            dt = _parse_date(raw)
            if dt is None:
                undated += 1
            elif claim_date is not None and dt > claim_date:
                after += 1
            if is_fact_check_host(url):
                fact_check += 1
    den = max(total, 1)
    return {"stance_docs": total, "post_claim": after, "undated": undated,
            "fact_check_host": fact_check,
            "post_claim_share": after / den, "undated_share": undated / den,
            "fact_check_share": fact_check / den,
            "unit": "distinct stance-bearing full-read documents"}


def score(pred: dict[str, str], gold: pd.DataFrame, idx: np.ndarray) -> dict:
    """Four-class ClaimCheck-convention accuracy plus binary map A, over the gold rows."""
    status = gold["claim_id"].map(pred)
    present = status.notna().to_numpy()
    four = status.map(lambda s: CC_MAP[s] if isinstance(s, str) else None)
    passes, _ = BINARY["A"]
    gold_flag = ~gold["gold_label"].isin(passes).to_numpy()
    sys_flag = (status != "supported").to_numpy()
    b = binary_metrics(sys_flag, gold_flag, present, idx)
    return {"n_rows": int(present.sum()), "n_rows_missing": int((~present).sum()),
            "four_class_convention": with_ci(
                (four == gold["gold_label"]).to_numpy() & present, present, idx),
            "binary_A": {k: b[k] for k in ("accuracy", "flag_recall", "fpr", "precision")}}


def analyse_arm(run_dir: Path, gold: pd.DataFrame, idx: np.ndarray) -> dict:
    run_dir = Path(run_dir)
    recs = load_arm(run_dir)
    actual = {c: r["status"] for c, r in recs.items()}
    cf_a = {c: cf_a_status(run_dir, c, r) for c, r in recs.items()}
    flips: dict[str, int] = {}
    for c, s in actual.items():
        if cf_a[c] != s:
            flips[f"{s} -> {cf_a[c]}"] = flips.get(f"{s} -> {cf_a[c]}", 0) + 1
    return {"n_claims": len(recs),
            "n_flipped_by_cf_a": sum(flips.values()),
            "cf_a_flips": dict(sorted(flips.items(), key=lambda kv: -kv[1])),
            "actual": score(actual, gold, idx),
            "cf_a_no_bar": score(cf_a, gold, idx),
            "unsupported_anatomy": unsupported_anatomy(run_dir, recs),
            "leak": ceiling_leak(recs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="top3,all10,noceil")
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    gold = pd.read_parquet(a.gold).copy()
    gold["claim_id"] = gold["claim_id"].astype(str)
    idx = boot_index(len(gold), a.reps, a.seed)

    def rel(p: Path) -> str:
        return str(p.relative_to(SRC)) if p.is_absolute() and p.is_relative_to(SRC) else str(p)

    out: dict = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "gold": rel(Path(a.gold)), "n_gold_rows": len(gold),
                 "bootstrap": {"reps": a.reps, "seed": a.seed, "unit": "gold row",
                               "interval": "percentile 95%"},
                 "counterfactual": "CF-A, no corroboration bar, the model's last proposed "
                                   "non-open status stands",
                 "arms": {}, "arms_missing": []}
    for name in [n.strip() for n in a.arms.split(",") if n.strip()]:
        d = Path(a.run) / ARM_DIR.get(name, name)
        if not d.exists():
            out["arms_missing"].append(name)
            continue
        out["arms"][name] = {"dir": rel(d), **analyse_arm(d, gold, idx)}

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    print("wrote", p)
    for name, e in out["arms"].items():
        an, lk = e["unsupported_anatomy"], e["leak"]
        print(f"{name:8s} n={e['n_claims']:3d} rows={e['actual']['n_rows']:3d} "
              f"4cls {e['actual']['four_class_convention']['value']:.3f} -> "
              f"CF-A {e['cf_a_no_bar']['four_class_convention']['value']:.3f} "
              f"[{e['cf_a_no_bar']['four_class_convention']['ci'][0]:.3f}, "
              f"{e['cf_a_no_bar']['four_class_convention']['ci'][1]:.3f}]  "
              f"recall {e['actual']['binary_A']['flag_recall']['value']:.3f} -> "
              f"{e['cf_a_no_bar']['binary_A']['flag_recall']['value']:.3f}  "
              f"fpr {e['actual']['binary_A']['fpr']['value']:.3f} -> "
              f"{e['cf_a_no_bar']['binary_A']['fpr']['value']:.3f}")
        print(f"{'':8s} unsupported {an['n_unsupported']}: bar {an['bar_refusal']} / "
              f"miss {an['retrieval_miss']} / other {an['other']}; "
              f"leak fact-check {lk['fact_check_share']:.1%} post-claim "
              f"{lk['post_claim_share']:.1%} of {lk['stance_docs']} docs")
    if out["arms_missing"]:
        print("missing arms, skipped:", ", ".join(out["arms_missing"]))


if __name__ == "__main__":
    main()
