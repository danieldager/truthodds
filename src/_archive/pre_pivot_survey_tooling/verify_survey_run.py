"""Run verify_text (raw-text loop) over the survey-experiment claim pool, FC-blocked + date-limited,
recording provisional veracities + the full trace. Resumable, flushed progress, checkpoint every row.

raw_context is the synthetic tweet and is verified AS-IS (framing intact, self-contained) — do NOT
de-frame it. No gold/audit exists for these claims, so nothing is joined; veracities are PROVISIONAL.
Mirrors eval/scripts/verification_grading/verify_text_run.py."""
import sys, os, json, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars as pl
sys.path.insert(0, "src")
from pipeline.verify_text import verify_text
from pipeline.verify import VERIFICATION_MODEL

BASE = "src/eval/data/survey_claims"
POOL = f"{BASE}/claim_pool_verify.parquet"
OUT = f"{BASE}/claim_pool_verdicts.parquet"

# grouping columns carried through verbatim from the pool onto each verdict row
GROUP_COLS = ["side", "reliability", "outlet", "domain", "cell",
              "source_url", "headline", "batch", "claim_date"]

def run_one(r):
    tr = {}
    out = verify_text(r["raw_context"], block_factcheck=True, date_ceiling=r.get("claim_date"),
                      pointed_exa=False, model=VERIFICATION_MODEL, max_rounds=10, trace=tr)
    row = {"claim_id": r["claim_id"]}
    row.update({c: r.get(c) for c in GROUP_COLS})
    row.update({"v_veracity": out["veracity"], "v_misinfo": out["misinfo"], "v_type": out["misinfo_type"],
                "main_claim": out["main_claim"], "justification": out["justification"],
                "n_search_rounds": out["n_search_rounds"],
                "provider_sequence": ",".join(out["provider_sequence"]),
                "n_read": out["n_read"], "stopped": out["stopped"], "llm_calls": out["llm_calls"],
                "elapsed_s": out["elapsed_s"], "trace_json": json.dumps(tr, default=str)})
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=OUT)
    ap.add_argument("-i", default=POOL)
    ap.add_argument("-w", type=int, default=3)
    ap.add_argument("-n", type=int, default=0, help="smoke: run only first N pool rows (0 = all)")
    args = ap.parse_args()

    V = pl.read_parquet(args.i)
    if args.n:
        V = V.head(args.n)
    done = set(pl.read_parquet(args.o)["claim_id"].to_list()) if os.path.exists(args.o) else set()
    todo = [r for r in V.to_dicts() if r["claim_id"] not in done]
    print(f">>> survey verify: {len(done)} done, {len(todo)} to go (of {V.height}), "
          f"FC-blocked, model={VERIFICATION_MODEL}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.w) as pool:
        futs = {pool.submit(run_one, r): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  [{i}/{len(todo)}] ERR {r['claim_id'][:12]}: {e}", flush=True); continue
            allrows = (pl.read_parquet(args.o).to_dicts() if os.path.exists(args.o) else []) + [res]
            pl.DataFrame(allrows).write_parquet(args.o)
            el = time.time() - t0; eta = (len(todo) - i) / (i / el) if i else 0
            print(f"  [{i}/{len(todo)}] {res['claim_id'][:12]} side={res['side']}/{res['reliability']} "
                  f"-> v={res['v_veracity']}/{res['v_type'][:4]} misinfo={res['v_misinfo']} | "
                  f"{res['n_search_rounds']}q {res['n_read']}rd [{res['provider_sequence']}] "
                  f"{res['llm_calls']}c {res['elapsed_s']}s (eta {eta:.0f}s)", flush=True)
    print(f">>> DONE {len(todo)} in {time.time() - t0:.0f}s -> {args.o}", flush=True)

if __name__ == "__main__":
    main()
