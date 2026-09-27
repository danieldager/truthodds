"""Cost / ETA estimator for evidence-URN experiment runs.

Two modes over the empirical unit economics of `evidence_urn_run.py`
(per claim: 1 Serper search + 1 query LLM call + ~10 sequential READ LLM calls;
reads parallel across claims at `workers`; global Serper gate every `S` seconds).

  measure  — parse smoke/run jsonl -> empirical units (cost/claim, cost/read,
             reads/claim, fed-char p50/p90, scrape-vs-snippet split).
  predict  — project a run from a spec: cost with LOW/MID/HIGH band, wall-clock
             from the explicit bottleneck (serper-gate vs read-throughput), and a
             ready-to-paste run_ledger.md row.

Bands follow eval/data/run_ledger.md's error patterns: WALL fails harder than
COST (serial gate + monster-doc tails) so it gets a wider, upward band; COST is
tighter because the fed-char input distribution is measured & cap-bounded.

Usage:
  uv run python -m eval.scripts.build_eval.run_estimator measure eval/data/urn_runs/e1/smoke.jsonl
  uv run python -m eval.scripts.build_eval.run_estimator predict --claims 420
  uv run python -m eval.scripts.build_eval.run_estimator predict --claims 4156 --cap-chars 8000 --workers 4 \
      --name "E1 full 4.2k"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# --- DeepInfra DeepSeek-V4-Flash pricing (verified 2026-07, config/CLAUDE.md) ---
PRICE_IN = 0.088 / 1e6        # $/input token
PRICE_OUT = 0.35 / 1e6        # $/output token
CACHE_MULT = 0.18             # cached input tokens billed at ~18% of full price
CHARS_PER_TOK = 4.0

# --- prompt geometry (evidence_urn_run.py) ---
READ_PREFIX_CHARS = 2735 + 300   # READ_SYS + claim_block; cached across a claim's reads
OUT_TOK_READ = 55                # compact READ JSON
QUERY_IN_TOK = 772 / CHARS_PER_TOK   # QUERY_SYS + claim
OUT_TOK_QUERY = 25

# --- empirical defaults, measured on eval/data/urn_runs/e1/smoke.jsonl (25 claims) ---
MEAS = {
    "reads_per_claim": 9.8,   # 244 reads / 25 claims
    "fed_fill": 0.52,         # mean fed-chars / cap: 4124 / 8000 (avg cap utilisation)
    "searches_per_claim": 1.0,  # 1 Serper query; occasional page-2 backfill -> band
    "read_latency_s": 3.0,    # per READ (LLM + inline scrape), back-solved from smoke wall
    "serper_interval_s": 8.5, # _SERPER_MIN_INTERVAL, global gate
    "cache_hit_rate": 0.66,   # sequential reads prime the per-claim prefix cache
}


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0


def measure(paths):
    recs = [json.loads(l) for r in paths for l in open(r) if l.strip()]
    costs = [r.get("cost", 0.0) for r in recs]
    reads = [len(r.get("results", [])) for r in recs]
    fed, prov = [], {"scrape": 0, "snippet": 0}
    for r in recs:
        for res in r.get("results", []):
            if res.get("n_sents"):
                fed.append(min(res.get("prep", {}).get("doc_chars", 0), 8000))
            prov[res.get("provenance", "scrape")] = prov.get(res.get("provenance", "scrape"), 0) + 1
    n_reads = sum(reads) or 1
    cpr = sum(costs) / n_reads
    cpc = sum(costs) / max(len(recs), 1)
    stats = {
        "claims": len(recs),
        "cost_per_claim_p50": _pct(costs, 0.5),
        "cost_per_claim_mean": cpc,
        "cost_per_read": cpr,
        "reads_per_claim": n_reads / max(len(recs), 1),
        "cost_per_query": cpc - (n_reads / max(len(recs), 1)) * cpr,
        "fed_chars_p50": _pct(fed, 0.5),
        "fed_chars_p90": _pct(fed, 0.9),
        "fed_fill_mean": (sum(fed) / len(fed) / 8000) if fed else 0,
        "scrape_frac": prov["scrape"] / n_reads,
        "snippet_frac": prov["snippet"] / n_reads,
    }
    print(f"# measured on {stats['claims']} claims, {n_reads} reads")
    for k, v in stats.items():
        print(f"  {k:22s} {v:.6f}" if isinstance(v, float) else f"  {k:22s} {v}")
    return stats


def _read_cost(cap_chars, cache_hit):
    prefix_tok = READ_PREFIX_CHARS / CHARS_PER_TOK
    doc_tok = cap_chars * MEAS["fed_fill"] / CHARS_PER_TOK
    prefix_bill = prefix_tok * (1 - cache_hit * (1 - CACHE_MULT))
    return (prefix_bill + doc_tok) * PRICE_IN + OUT_TOK_READ * PRICE_OUT


def predict(claims, reads_per_claim, cap_chars, workers, cache_hit,
            serper_interval, searches_per_claim, read_latency, name, date):
    q_cost = QUERY_IN_TOK * PRICE_IN + OUT_TOK_QUERY * PRICE_OUT
    cost_mid = claims * (q_cost + reads_per_claim * _read_cost(cap_chars, cache_hit))
    # COST band: input dist is measured & cap-bounded -> tight; slight upward skew for
    # cap-hit doc tails and cache-warmth variance (ledger pattern 2).
    cost_lo, cost_hi = cost_mid * 0.80, cost_mid * 1.25

    # WALL: explicit bottleneck = max(global serper gate, parallel read throughput).
    serper_wall = claims * searches_per_claim * serper_interval
    read_wall = claims * reads_per_claim * read_latency / workers
    bottleneck = "serper-gate" if serper_wall >= read_wall else "read-throughput"
    wall_mid = max(serper_wall, read_wall)
    # WALL band wider & upward: serial gate + monster-doc tails + cooldown retries
    # blow past estimates far more than cost (ledger pattern 1).
    wall_lo, wall_hi = wall_mid * 0.90, wall_mid * 1.60

    def hm(s):
        return f"{s/3600:.1f}h" if s >= 3600 else f"{s/60:.0f}m"

    print(f"# {name}: {claims} claims | cap {cap_chars} | {workers} workers | "
          f"cache {cache_hit:.0%} | serper {serper_interval}s")
    print(f"  cost   LOW ${cost_lo:.2f}  MID ${cost_mid:.2f}  HIGH ${cost_hi:.2f}")
    print(f"  wall   LOW {hm(wall_lo)}  MID {hm(wall_mid)}  HIGH {hm(wall_hi)}  "
          f"[bottleneck: {bottleneck}; serper {hm(serper_wall)} vs read {hm(read_wall)}]")
    note = f"cap {cap_chars}, {workers}w, {bottleneck}-bound"
    print("  ledger row (paste into eval/data/run_ledger.md):")
    print(f"| {date} | {name} | {cost_mid:.2f} (${cost_lo:.2f}-${cost_hi:.2f}) | "
          f"~{hm(wall_mid)} ({hm(wall_lo)}-{hm(wall_hi)}) | | | | {note} |")
    return {"cost_mid": cost_mid, "wall_mid": wall_mid, "bottleneck": bottleneck}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    mp = sub.add_parser("measure")
    mp.add_argument("files", nargs="+")
    pp = sub.add_parser("predict")
    pp.add_argument("--claims", type=int, required=True)
    pp.add_argument("--reads-per-claim", type=float, default=MEAS["reads_per_claim"])
    pp.add_argument("--cap-chars", type=int, default=8000)
    pp.add_argument("--workers", type=int, default=4)
    pp.add_argument("--cache-hit-rate", type=float, default=MEAS["cache_hit_rate"])
    pp.add_argument("--serper-interval", type=float, default=MEAS["serper_interval_s"])
    pp.add_argument("--searches-per-claim", type=float, default=MEAS["searches_per_claim"])
    pp.add_argument("--read-latency", type=float, default=MEAS["read_latency_s"])
    pp.add_argument("--name", default="run")
    pp.add_argument("--date", default="--")
    a = ap.parse_args()

    if a.mode == "measure":
        measure([Path(f) for f in a.files])
    else:
        predict(a.claims, a.reads_per_claim, a.cap_chars, a.workers, a.cache_hit_rate,
                a.serper_interval, a.searches_per_claim, a.read_latency, a.name, a.date)


if __name__ == "__main__":
    main()
