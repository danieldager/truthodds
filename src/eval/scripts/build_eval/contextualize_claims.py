"""Contextualize the E1 draw: firewalled {date, claimant, retrieval context} for
every claim entering the run — standing rule 2026-08-03 (no fc-gold eval without
contextualization first).

Replicates the runner's exact draw (sort by review_url, all trues + 2000 falses
seed 505 + 600 mid seed 505) so every row the run touches has context. Live
validation: rows that already carry a native claim_date double as a gold meter
for the extractor (extracted-vs-known agreement, reported at the end and every
progress tick).

  uv run python -m eval.scripts.build_eval.contextualize_claims

Writes eval/data/claim_context_cal.parquet.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.prompt_hash import prompt_hash   # noqa: E402
from eval.scripts.build_eval.extract_claim_dates import EXTRACT_SYS, run_one   # noqa: E402

OUT = Path("eval/data/claim_context_cal.parquet")
# 8 workers is safe BECAUSE the row order is shuffled: the draw sorts by review_url,
# which serializes whole publisher blocks (all snopes back-to-back) — shuffling mixes
# domains so no single site sees more than ~1-2 req/s even at 8 workers.
WORKERS = 8
PROMPT_HASH = prompt_hash(EXTRACT_SYS)


def main() -> None:
    df = pl.read_parquet("eval/data/truthodds_cal.parquet").sort("review_url")
    parts = [df.filter(pl.col("veracity") >= 4),
             df.filter(pl.col("veracity") <= 2).sample(2000, seed=505),
             df.filter(pl.col("veracity") == 3).sample(600, seed=505)]
    draw = pl.concat(parts)
    rows = draw.select(["review_url", "claim_text", "publisher_site", "claim_date",
                        "review_date", "veracity"]).to_dicts()
    for r in rows:
        r["claim_date_source"] = ""
    import random
    random.Random(42).shuffle(rows)     # mix publisher domains — see WORKERS note
    print(f"draw: {len(rows)} rows "
          f"(T {len(parts[0])} / F {len(parts[1])} / M {len(parts[2])})", flush=True)

    t0, done, out = time.time(), 0, []
    gold_n = gold_close = 0
    cost = 0.0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for r in ex.map(run_one, rows):
            out.append(r)
            done += 1
            cost += r.get("cost") or 0
            if r["status"] == "ok" and r.get("x_date") and r.get("claim_date"):
                gold_n += 1
                try:
                    import datetime as dt
                    d1 = dt.date.fromisoformat(str(r["claim_date"])[:10])
                    d2 = dt.date.fromisoformat((r["x_date"] + "-01")[:10]
                                               if len(r["x_date"]) == 7 else r["x_date"][:10])
                    gold_close += abs((d1 - d2).days) <= 7
                except Exception:
                    pass
            if done % 50 == 0:
                el = time.time() - t0
                rate = done / el
                okr = sum(x["status"] == "ok" for x in out)
                print(f"  {done}/{len(rows)}  {rate*60:.0f}/min  eta {(len(rows)-done)/rate/60:.0f}m  "
                      f"ok {okr}  gold-meter {gold_close}/{gold_n} within-7d  ${cost:.2f}",
                      flush=True)

    ok = [r for r in out if r["status"] == "ok"]
    leaks = sum(bool(r.get("leak_flag")) for r in ok)
    dates = sum(bool(r.get("x_date")) for r in ok)
    ctxs = sum(bool(r.get("x_context")) for r in ok)
    fails = {}
    for r in out:
        if r["status"] != "ok":
            fails[r["status"]] = fails.get(r["status"], 0) + 1
    print(f"\ndone {len(out)} in {(time.time()-t0)/60:.0f}m  ${cost:.2f}\n"
          f"ok {len(ok)}  dates {dates}  contexts {ctxs}  leak-flagged {leaks}\n"
          f"gold-meter: {gold_close}/{gold_n} extracted dates within 7d of native\n"
          f"failures: {fails}", flush=True)
    pl.DataFrame([{ "review_url": r["review_url"], "status": r["status"],
                    "x_date": r.get("x_date"), "x_claimant": r.get("x_claimant"),
                    "x_context": r.get("x_context"),
                    "leak_flag": bool(r.get("leak_flag")),
                    "prompt_hash": PROMPT_HASH} for r in out]
                 ).write_parquet(OUT)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
