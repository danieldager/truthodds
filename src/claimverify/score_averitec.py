"""Score a claimverify run against AVeriTeC dev 500 gold.

  uv run python -m claimverify.score_averitec --run <dir> [--gold <parquet>] [--out <parquet>]

Reads the harness's shard-*.jsonl records, joins on claim_id to EVERY gold row (the 9
duplicate rows share a claim_id and inherit its prediction, as in
eval/scripts/verification_grading/run_averitec_benchmark.py) and computes:
  (a) 4-class accuracy / macro-F1 / confusion under the RAW map (unsupported -> Not Enough
      Evidence) and the CLAIMCHECK CONVENTION map (unsupported -> Refuted);
  (b) three binary pass/flag variants, applied to system and gold alike: A = Supported is
      pass; B = Supported + Conflicting pass; C = gold Conflicting rows dropped, Supported
      pass. unsupported is flag under both maps, so the binary numbers are map-invariant;
  (c) per-claim speed and cost (elapsed quantiles, LLM / Serper / scrape counts, tokens,
      cost with its cost_source breakdown from trace/<claim_id>.json when present).
Rows whose claim has no successful record are listed as missing and excluded from the
metrics (n is reported everywhere). Writes <run>/scored.parquet (gold row order) and
<run>/metrics.json, prints a compact summary.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from claimverify.config import SRC
from claimverify.harness import load_records
from claimverify.loop import CC_MAP, RAW_MAP

DEFAULT_GOLD = SRC / "eval/scripts/verification_grading/data/claims_dev_500_gold.parquet"
CLASSES = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
BINARY = {  # variant -> (labels that count as pass, gold labels dropped)
    "A": ({"Supported"}, set()),
    "B": ({"Supported", "Conflicting Evidence/Cherrypicking"}, set()),
    "C": ({"Supported"}, {"Conflicting Evidence/Cherrypicking"}),
}


def four_class(pred: pd.Series, gold: pd.Series) -> dict:
    n = len(gold)
    f1 = {}
    for c in CLASSES:
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        f1[c] = 2 * tp / max(1, 2 * tp + fp + fn)
    conf = {g: {p: int(((gold == g) & (pred == p)).sum()) for p in CLASSES} for g in CLASSES}
    return {"n": n, "accuracy": float((pred == gold).mean()) if n else 0.0,
            "macro_f1": sum(f1.values()) / len(f1), "f1": f1, "confusion": conf}


def binary(pred: pd.Series, gold: pd.Series, passes: set[str], drop: set[str]) -> dict:
    keep = ~gold.isin(drop)
    sys_flag, gold_flag = ~pred[keep].isin(passes), ~gold[keep].isin(passes)
    n = int(keep.sum())
    tp = int((sys_flag & gold_flag).sum())
    return {"n": n, "n_gold_flag": int(gold_flag.sum()), "n_gold_pass": int((~gold_flag).sum()),
            "accuracy": float((sys_flag == gold_flag).mean()) if n else 0.0,
            "flag_recall": tp / max(1, int(gold_flag.sum())),
            "fpr": int((sys_flag & ~gold_flag).sum()) / max(1, int((~gold_flag).sum())),
            "precision": tp / max(1, int(sys_flag.sum()))}


def speed_cost(results: list[dict], trace_dir: Path) -> dict:
    df = pd.DataFrame(results)
    if df.empty:
        return {"n_claims": 0}
    el = df["elapsed_s"].astype(float)
    tok = pd.DataFrame(list(df["tokens"])).sum() if "tokens" in df else pd.Series(dtype=float)
    by_source: Counter = Counter()
    for r in results:
        p = trace_dir / f"{r['claim_id']}.json"
        if p.exists():
            for c in json.loads(p.read_text()).get("llm_calls", []):
                u = c.get("usage") or {}
                by_source[u.get("cost_source", "unknown")] += float(u.get("cost_usd") or 0.0)
    return {"n_claims": len(df),
            "elapsed_s": {"p50": float(el.quantile(.5)), "p90": float(el.quantile(.9)),
                          "mean": float(el.mean()), "sum": float(el.sum())},
            "llm_calls": int(df["llm_calls"].sum()), "llm_cache_hits": int(df["llm_cache_hits"].sum()),
            "serper_calls": int(df["serper_calls"].sum()), "exa_calls": int(df["exa_calls"].sum()),
            "scrapes": int(df["scrapes"].sum()),
            "tokens": {k: int(v) for k, v in tok.items()},
            "cost_usd": float(df["cost_usd"].sum()),
            "cost_by_source": {k: round(v, 6) for k, v in by_source.items()}}


def score_run(run_dir: Path, gold_path: Path) -> tuple[pd.DataFrame, dict]:
    run_dir = Path(run_dir)
    ok = load_records(run_dir)
    failed = {k: v.get("error") for k, v in load_records(run_dir, ok_only=False).items()
              if not v.get("ok") and k not in ok}
    results = {cid: rec["result"] for cid, rec in ok.items()}

    gold = pd.read_parquet(gold_path)[["claim_id", "gold_label"]].copy()
    gold["claim_id"] = gold["claim_id"].astype(str)
    gold["duplicate"] = gold["claim_id"].duplicated()
    per_claim = ["status", "verdict_raw", "verdict_cc", "elapsed_s", "llm_calls",
                 "serper_calls", "exa_calls", "scrapes", "cost_usd"]
    for col in per_claim:
        gold[col] = gold["claim_id"].map(lambda c: results.get(c, {}).get(col))
    gold["scored"] = gold["status"].notna()
    df = gold
    s = df[df["scored"]]
    df["correct_raw"] = df["verdict_raw"] == df["gold_label"]
    df["correct_cc"] = df["verdict_cc"] == df["gold_label"]
    for v, (passes, drop) in BINARY.items():
        keep = ~df["gold_label"].isin(drop)
        df[f"bin_{v}_gold"] = (~df["gold_label"].isin(passes)).where(keep)
        df[f"bin_{v}_sys"] = (~df["verdict_raw"].isin(passes)).where(keep & df["scored"])

    missing = sorted(set(gold["claim_id"]) - set(results))
    metrics = {
        "run": str(run_dir), "gold": str(gold_path),
        "n_gold_rows": len(df), "n_gold_claims": int(df["claim_id"].nunique()),
        "n_scored_rows": int(df["scored"].sum()), "n_missing_claims": len(missing),
        "missing_claims": missing, "failed_claims": failed,
        "four_class": {"raw": four_class(s["verdict_raw"], s["gold_label"]),
                       "claimcheck_convention": four_class(s["verdict_cc"], s["gold_label"])},
        "binary": {v: binary(s["verdict_raw"], s["gold_label"], p, d) for v, (p, d) in BINARY.items()},
        "speed_cost": speed_cost([results[c] for c in sorted(results)], run_dir / "trace"),
        "maps": {"raw": RAW_MAP, "claimcheck_convention": CC_MAP},
        "status_counts": dict(Counter(r["status"] for r in results.values())),
    }
    return df, metrics


def summary(m: dict) -> str:
    out = [f"{m['n_scored_rows']}/{m['n_gold_rows']} gold rows scored "
           f"({m['n_gold_claims']} claims, {m['n_missing_claims']} missing, "
           f"{len(m['failed_claims'])} failed) | status {m['status_counts']}"]
    for name, fc in m["four_class"].items():
        out.append(f"[{name}] acc {fc['accuracy']:.3f}  macro-F1 {fc['macro_f1']:.3f}  (n={fc['n']})")
        out.append("  gold \\ pred      " + "  ".join(f"{p[:4]:>4}" for p in CLASSES))
        for g in CLASSES:
            out.append(f"    {g[:16]:16} " + "  ".join(f"{fc['confusion'][g][p]:4d}" for p in CLASSES))
    for v, b in m["binary"].items():
        out.append(f"[binary {v}] n={b['n']} acc {b['accuracy']:.3f}  flag-recall {b['flag_recall']:.3f}"
                   f"  FPR {b['fpr']:.3f}  precision {b['precision']:.3f}")
    sc = m["speed_cost"]
    if sc.get("n_claims"):
        e = sc["elapsed_s"]
        out.append(f"[speed] elapsed p50 {e['p50']:.1f}s p90 {e['p90']:.1f}s mean {e['mean']:.1f}s"
                   f" | llm {sc['llm_calls']} (cache {sc['llm_cache_hits']}) | serper {sc['serper_calls']}"
                   f" | exa {sc['exa_calls']} | scrapes {sc['scrapes']}")
        out.append(f"[cost] ${sc['cost_usd']:.4f} | tokens {sc['tokens']} | by source {sc['cost_by_source']}")
    if m["missing_claims"]:
        mc = m["missing_claims"]; out.append(f"missing ({len(mc)}): {mc[:10]}{' ...' if len(mc) > 10 else ''}")
    if m["failed_claims"]:
        out.append(f"failed: {m['failed_claims']}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--out", default="", help="per-row parquet (default <run>/scored.parquet)")
    a = ap.parse_args()
    run_dir = Path(a.run)
    df, m = score_run(run_dir, Path(a.gold))
    out = Path(a.out) if a.out else run_dir / "scored.parquet"
    df.to_parquet(out, index=False)
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=1))
    print(summary(m))
    print(f"-> {out}\n-> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
