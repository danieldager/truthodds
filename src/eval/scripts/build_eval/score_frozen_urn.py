"""Score an urn run with FROZEN ladder weights (no refit).

Applies the 3-voice and 7-flag log-odds weights pinned in
`model_ladder/ladder.json` to every claim in a results jsonl, exactly as
`fit_urn.score` / `graded_urn.score_graded` would with those weights, and
reports how many claims the ladder's 2%-FPR thresholds would flag
(fit_urn.recall_at_fpr: flag = score AT OR BELOW the threshold).

Differences from fit_urn.load, on purpose: EVERY record is scored, including
veracity 3 and records with zero readable documents (they score 0 and carry
n_docs=0), and the gold-side exclusion lists are not applied. No accuracy
against labels is computed here -- that comparison is pre-registered separately.

  uv run python -m eval.scripts.build_eval.score_frozen_urn \\
      --results eval/data/urn_runs/averitec_dev/scores.jsonl \\
      --ladder eval/data/urn_runs/e1_ctx/model_ladder/ladder.json
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval.fit_urn import FLAG_TO_VOICE  # noqa: E402

VOICE_CHANNEL = {"supports": "support", "refutes": "refute", "neutral": "silent",
                 "irrelevant": "silent"}
FLAGS = ("5", "4", "3", "X", "I", "2", "1")


def parse(rec: dict) -> dict:
    """One row per record: flag counts (fit_urn.load's counting rule) + run stats."""
    flags = collections.Counter()
    for d in rec.get("results") or []:
        direction = (d.get("read") or {}).get("direction")
        if FLAG_TO_VOICE.get(direction):
            flags[direction] += 1
    voices = collections.Counter()
    for f, n in flags.items():
        voices[VOICE_CHANNEL[FLAG_TO_VOICE[f]]] += n
    return {"claim_id": rec["review_url"], "averitec_label": rec.get("averitec_label"),
            "veracity": rec.get("veracity"), "excluded": rec.get("excluded"),
            "n_docs": sum(flags.values()),
            **{f"n_{k}": flags.get(k, 0) for k in FLAGS},
            **{f"n_{k}": voices.get(k, 0) for k in ("support", "silent", "refute")},
            "wall_s": rec.get("wall_s"), "cost": rec.get("cost"),
            "llm_calls": rec.get("llm_calls"), "search_calls": rec.get("search_calls")}


def score(row: dict, ladder: dict) -> tuple[float, float]:
    w3 = ladder["models"]["3-voice"]["weights"]
    w7 = ladder["models"]["7-flag"]["weights"]
    s3 = sum(row[f"n_{c}"] * w3[c] for c in ("support", "silent", "refute"))
    s7 = sum(row[f"n_{f}"] * w7[f] for f in FLAGS)
    return s3, s7


def _q(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    q = statistics.quantiles(xs, n=20) if len(xs) > 1 else [xs[0]] * 19
    return (f"min {min(xs):.2f} | p5 {q[0]:.2f} | p25 {q[4]:.2f} | p50 {q[9]:.2f} | "
            f"p75 {q[14]:.2f} | p95 {q[18]:.2f} | max {max(xs):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--ladder", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="per-claim table (.parquet or .jsonl); default <results>.scored.parquet")
    a = ap.parse_args()
    ladder = json.loads(a.ladder.read_text())
    rows = []
    for line in a.results.open():
        row = parse(json.loads(line))
        row["s3"], row["s7"] = score(row, ladder)
        rows.append(row)
    out = a.out or a.results.with_suffix(".scored.parquet")
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_ndjson(out) if out.suffix == ".jsonl" else df.write_parquet(out)

    n0 = sum(1 for r in rows if r["n_docs"] == 0)
    print(f"scored {len(rows)} claims ({n0} with zero readable docs, scored 0) -> {out}")
    print("veracity: " + json.dumps(collections.Counter(r["veracity"] for r in rows)))
    print("docs/claim: " + _q([r["n_docs"] for r in rows]))
    for m, key in (("3-voice", "s3"), ("7-flag", "s7")):
        thr = ladder["models"][m]["threshold_2pct"]
        xs = [r[key] for r in rows]
        flagged = sum(1 for x in xs if x <= thr)
        print(f"{m}: {_q(xs)}\n   threshold_2pct {thr:.3f} -> flagged (score <= thr) "
              f"{flagged}/{len(rows)} = {flagged/len(rows):.1%}")
        for v in sorted({r["veracity"] for r in rows}, key=lambda x: (x is None, x)):
            sub = [r[key] for r in rows if r["veracity"] == v]
            print(f"   veracity {v}: n {len(sub):>5} | median {statistics.median(sub):.2f} | "
                  f"flagged {sum(1 for x in sub if x <= thr)/len(sub):.1%}")


if __name__ == "__main__":
    main()
