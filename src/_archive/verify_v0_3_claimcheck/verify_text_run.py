"""Run verify_text (raw-text loop) on the audited with-post dev claims, FC-blocked + date-limited,
recording the full trace. Resumable, flushed progress. Score separately with score_vtext.py."""
import sys, os, json, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars as pl
sys.path.insert(0, "src")
from pipeline.verify_text import verify_text
from pipeline.verify import VERIFICATION_MODEL

DATA = "src/eval/scripts/verification_grading/data"

def build_set():
    C = pl.read_parquet(f"{DATA}/claims_dev_clean.parquet")
    A = pl.read_parquet(f"{DATA}/misinfo_audit_dev.parquet")
    hp = pl.col("raw_context").is_not_null() & (pl.col("raw_context").str.strip_chars().str.len_chars() > 40)
    return C.filter(hp).join(A.select(["claim_id","audit_veracity","audit_type","matches_gold"]), on="claim_id", how="inner")

def run_one(r, pointed_exa=False):
    tr = {}
    out = verify_text(r["raw_context"], block_factcheck=True, date_ceiling=r.get("claim_date"),
                      pointed_exa=pointed_exa, model=VERIFICATION_MODEL, max_rounds=10, trace=tr)
    return {"claim_id": r["claim_id"], "gold_veracity": r["gold_veracity"],
            "audit_veracity": r["audit_veracity"], "audit_type": r["audit_type"], "matches_gold": r["matches_gold"],
            "v_veracity": out["veracity"], "v_misinfo": out["misinfo"], "v_type": out["misinfo_type"],
            "n_search_rounds": out["n_search_rounds"], "provider_sequence": ",".join(out["provider_sequence"]),
            "n_read": out["n_read"], "n_evidence": out["n_evidence"], "stopped": out["stopped"],
            "fc_blocked": out["counters"]["factcheck_blocked"], "llm_calls": out["llm_calls"],
            "elapsed_s": out["elapsed_s"], "main_claim": out["main_claim"],
            "justification": out["justification"], "trace_json": json.dumps(tr, default=str)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=f"{DATA}/vtext_verdicts_checkpoint.parquet")
    ap.add_argument("--ids", default=f"{DATA}/vtext_checkpoint_ids.parquet", help="parquet w/ claim_id col; 'all' for full 81")
    ap.add_argument("-w", type=int, default=3)
    ap.add_argument("--pointed-exa", action="store_true", dest="pointed_exa",
                    help="re-enable the forced pointed-Exa on wind-down (ablation showed 0/9 nudge flips; default off)")
    args = ap.parse_args()
    V = build_set()
    if args.ids != "all":
        keep = pl.read_parquet(args.ids)["claim_id"].to_list()
        V = V.filter(pl.col("claim_id").is_in(keep))
    done = set(pl.read_parquet(args.o)["claim_id"].to_list()) if os.path.exists(args.o) else set()
    todo = [r for r in V.to_dicts() if r["claim_id"] not in done]
    print(f">>> vtext run: {len(done)} done, {len(todo)} to go (of {V.height}), FC-blocked, model={VERIFICATION_MODEL}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.w) as pool:
        futs = {pool.submit(run_one, r, args.pointed_exa): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  [{i}/{len(todo)}] ERR {r['claim_id'][:12]}: {e}", flush=True); continue
            allrows = (pl.read_parquet(args.o).to_dicts() if os.path.exists(args.o) else []) + [res]
            pl.DataFrame(allrows).write_parquet(args.o)
            el = time.time()-t0; eta = (len(todo)-i)/(i/el) if i else 0
            ok = "✓" if bool(res["v_misinfo"]) == (res["audit_veracity"] <= 3) else "✗"
            print(f"  [{i}/{len(todo)}] {ok} audit={res['audit_veracity']}/{res['audit_type'][:4]} "
                  f"-> v={res['v_veracity']}/{res['v_type'][:4]} misinfo={res['v_misinfo']} | "
                  f"{res['n_search_rounds']}q {res['n_read']}rd [{res['provider_sequence']}] fc_blk={res['fc_blocked']} "
                  f"{res['elapsed_s']}s (eta {eta:.0f}s)", flush=True)
    print(f">>> DONE {len(todo)} in {time.time()-t0:.0f}s -> {args.o}", flush=True)

if __name__ == "__main__":
    main()
