"""E3 runner: retrieval + read over the KEY claims (key_claim_extract.py) of the NG<70
outlet posts.

Same instrument as E2 (tweet_urn_run.py: run_claim from evidence_urn_run, query-v3,
Serper top-10 minus origin, date ceiling = post date). `--reader` picks the READ:

  read-v5 (DEFAULT)  evidence_urn_run.read_doc untouched -- the production reader,
                     READ_SYS_MODE on DeepSeek-V4-Flash. This is what the shipped
                     weights and rating cuts (`populations/refit_results.json`) were
                     fitted under, so its scores can be cut with them directly.
  v7qa               reader_lab.read_one under the "v7qa" prompt (flags mapped in
                     code, map_qa) on openai/gpt-oss-120b via --provider.

REVERTED TO read-v5 ON FLASH, Daniel 2026-09-11 (clog/110926.md 10:55). v7qa on
gpt-oss lost on both fc_gold and AVeriTeC, and the paired bootstrap put two thirds of
the loss on the MODEL (dAUC +0.0430 [+0.0118, +0.0746] with the prompt held at read-v5)
rather than the prompt (+0.0214 [-0.0125, +0.0546] with the model held at gpt-oss).
gpt-oss reads in endpoints -- under the identical production prompt it emits 61% fewer
"4"s and 53% fewer "2"s than Flash -- and the urn's discrimination lives in those graded
middle voices. `--reader v7qa` is kept only to reproduce the 2026-09-11 runs.

Query generation runs on --query-model (flash = production
DeepSeek-V4-Flash, gpt-oss = gpt-oss-120b reasoning low), everything else is E2's.
Model calls go through reader_lab.GATE (adaptive concurrency); reads are sequential
inside run_claim, so --workers bounds concurrency and the gate finds the ceiling.

    uv run python -m eval.scripts.build_eval.keyclaim_urn_run --claims eval/data/reader_lab/extract/keyclaims_v2.json --smoke --budget 0.5
    uv run python -m eval.scripts.build_eval.keyclaim_urn_run --claims eval/data/reader_lab/extract/keyclaims_v2.json --budget 15 --workers 16
Output: eval/data/urn_runs/e3_keyclaims/{smoke,results-00}.jsonl (resumable by claim_id)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402

from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval import evidence_urn_run as eur  # noqa: E402
from eval.scripts.build_eval import reader_lab  # noqa: E402
from eval.scripts.build_eval.reader_lab_prompts import MAPPERS, PROMPTS  # noqa: E402
from pipeline.search import newsguard_score_map  # noqa: E402

OUTDIR = Path("eval/data/urn_runs/e3_keyclaims")
READ_PROMPT = "v7qa"
READ_MODEL = "openai/gpt-oss-120b"
# what the manifest and the banner record when --reader is read-v5 (eur's own reader)
READ_V5 = ("v5", eur.VERIFICATION_MODEL)
QUERY_MODELS = {"flash": eur.VERIFICATION_MODEL, "gpt-oss": "openai/gpt-oss-120b"}


def make_query_llm(model):
    """eur.llm's contract (obj, cost, cached, ptok[, lp][, stats]) on `model` at the DeepInfra
    endpoint, through the gate. gpt-oss needs room for its hidden reasoning."""
    def llm(messages, cache_key=None, timeout=60, max_tokens=400, want_logprobs=False, want_stats=False):
        body = {"model": model, "temperature": 0, "max_tokens": 4000 if "gpt-oss" in model else max_tokens,
                "response_format": {"type": "json_object"}, "messages": messages}
        if "gpt-oss" in model:
            body["reasoning_effort"] = "low"
        if cache_key:
            body["prompt_cache_key"] = cache_key
        t0 = time.time()
        r = reader_lab.gated_post(f"{eur.EXTRACTION_BASE_URL}/chat/completions",
                                  {"Authorization": f"Bearer {eur.EXTRACTION_API_KEY}"}, body, max(timeout, 120))
        j = r.json(); usage = j.get("usage", {})
        base = (reader_lab._parse_json(j["choices"][0]["message"]["content"]), usage.get("estimated_cost") or 0.0,
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0, usage.get("prompt_tokens") or 0)
        if want_logprobs:
            base += (None,)
        if want_stats:
            base += ({"latency_s": round(time.time() - t0, 3), "completion_tok": usage.get("completion_tokens") or 0},)
        return base
    return llm


def read_doc_v7qa(claim_block, ids, sents, key, stats=None, doc_date=None):  # doc_date unused: v7qa is frozen
    """read_doc's contract (dict-or-None, cost, cached_tok, prompt_tok) on the lab reader."""
    res, cost = reader_lab.read_one(PROMPTS[READ_PROMPT], claim_block, ids, sents, READ_MODEL, None, MAPPERS[READ_PROMPT])
    st = {"latency_s": res.get("latency_s"), "completion_tok": res.get("completion_tok"),
          "prompt_tok": res.get("prompt_tok"), "cached_tok": 0}
    if stats is not None:
        stats.update(st)
    if res.get("qc_flag") == "read-failed":
        return None, cost, 0, res.get("prompt_tok") or 0
    return ({"direction": res["direction"], "evidence": res["evidence"], "reason": res.get("reason", "")[:80],
             "qc_flag": res.get("qc_flag", ""), "flag_logprob": None, **st}, cost, 0, res.get("prompt_tok") or 0)


def adapt(c, bins):
    return {**c, "review_url": c["claim_id"], "claim_text": c["claim"], "publisher_site": c["domain"],
            "claim_date": (c.get("created_at") or "")[:10], "claim_type": c["type"], "x_context": c.get("post_text") or "",
            "context_ok": True, "resolution_status": "native", "claim_resolved": None,
            "veracity": None, "rating_subtype": None, "review_date": None, "x_date": None, "yr": None, "topic": None,
            "exclude_extra": ["x.com", "twitter.com"],
            "bin": bins.get(str(c["post_id"])), "checkworthy": True, "verify_eligible": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--bins", default="eval/data/reader_lab/inputs/e2_rescored.parquet")
    ap.add_argument("--smoke", action="store_true", help="25 random posts")
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=96, help="claim workers; reads are sequential per claim, so this bounds concurrency")
    ap.add_argument("--provider", default="deepinfra", choices=sorted(reader_lab.PROVIDERS))
    ap.add_argument("--query-model", default="flash", choices=sorted(QUERY_MODELS))
    ap.add_argument("--reader", default="read-v5", choices=["read-v5", "v7qa"],
                    help="read-v5 = the production READ_SYS_MODE reader on DeepSeek-V4-Flash "
                         "(default, and what the shipped weights/cuts were fitted under); "
                         "v7qa = the two-question gpt-oss reader (reverted 2026-09-11)")
    ap.add_argument("--gate-start", type=int, default=48)
    ap.add_argument("--gate-cap", type=int, default=96)
    ap.add_argument("--seed", type=int, default=707)
    ap.add_argument("--out-dir", default=str(OUTDIR), help="one directory per claims file (claim ids repeat across extractions)")
    ap.add_argument("--no-oss-json", action="store_true", help="read gpt-oss without forced JSON mode (DeepInfra json pool busy)")
    ap.add_argument("--fix", action="store_true",
                    help="e4 mid-band fix set (clog/110926 diagnosis): enforce the date "
                         "ceiling on the RESULT (leaked documents are not read), give READ "
                         "each document's publication date, and query the utterance for "
                         "attribution claims (query-v3a). OFF = the pinned e4_midband path.")
    ap.add_argument("--read-asof", action="store_true",
                    help="READ under read_v5_prompts.READ_SYS_ASOF (explicit as-of date rule). "
                         "Separate from --fix because it is a PROMPT change and the three "
                         "previous read-v5 rewordings each inverted a stratum (clog 2026-08-04).")
    ap.add_argument("--only-claims", default=None,
                    help="file of claim_ids (one per line, or a JSON list) — run only these")
    args = ap.parse_args()
    if args.fix:
        eur.FIX.update(drop_leaks=True, doc_dates=True, attr_query=True)
    if args.read_asof:
        eur.FIX["read_asof"] = True
    reader_lab.OSS_JSON_MODE[0] = not args.no_oss_json
    outdir = Path(args.out_dir)
    reader_lab.PROVIDER[0] = args.provider
    reader_lab.GATE.limit = reader_lab.GATE.peak = args.gate_start
    reader_lab.GATE.cap = args.gate_cap
    if args.reader == "v7qa":
        eur.read_doc = read_doc_v7qa   # run_claim looks read_doc up in its module at call time
    read_prompt, read_model = (READ_PROMPT, READ_MODEL) if args.reader == "v7qa" else READ_V5
    eur.llm = make_query_llm(QUERY_MODELS[args.query_model])
    df = pd.read_parquet(args.bins)
    bins = dict(zip(df["post_id"].astype(str), df["bin"].astype(str)))
    claims = json.load(open(args.claims))["claims"]
    if args.only_claims:
        txt = Path(args.only_claims).read_text().strip()
        keep = set(json.loads(txt)) if txt.startswith("[") else {l.strip() for l in txt.splitlines() if l.strip()}
        claims = [c for c in claims if c["claim_id"] in keep]
        assert len(claims) == len(keep), f"{len(keep) - len(claims)} claim_ids not found"
    rows = [adapt(c, bins) for c in claims]
    by_post = {}
    for r in rows:
        by_post.setdefault(r["post_id"], []).append(r)
    order = sorted(by_post)
    random.Random(args.seed).shuffle(order)
    if args.smoke:
        order = order[:25]
    rows = [r for pid in order for r in by_post[pid]]
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / ("smoke.jsonl" if args.smoke else "results-00.jsonl")
    seen = set()
    if out.exists():
        for l in open(out):
            try:
                seen.add(json.loads(l)["review_url"])
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in rows if r["review_url"] not in seen]
    (outdir / "manifest.json").write_text(json.dumps({
        "claims_file": args.claims, "n_claims": len(rows), "n_posts": len(order), "seed": args.seed,
        "prompts": {"query": eur.QUERY_PROMPT_V, "query_hash": prompt_hash(eur.QUERY_SYS),
                    "query_model": QUERY_MODELS[args.query_model],
                    "read": read_prompt + ("-asof" if eur.FIX["read_asof"] else ""),
                    "read_hash": prompt_hash(eur.READ_SYS_ASOF if eur.FIX["read_asof"]
                                             else PROMPTS[read_prompt]),
                    "read_model": read_model, "read_provider": args.provider},
        "gate": {"start": args.gate_start, "cap": args.gate_cap}, "workers": args.workers,
        "fix": dict(eur.FIX)}, indent=1))
    print(f"{len(todo)} claims ({len(order)} posts) to run -> {out.name} | budget ${args.budget} | {args.workers} workers | "
          f"query {eur.QUERY_PROMPT_V} on {QUERY_MODELS[args.query_model]} | read {read_prompt} on {read_model} @ {args.provider} | "
          f"gate {args.gate_start}..{args.gate_cap}", flush=True)
    ng_scores = newsguard_score_map()
    lock = threading.Lock()
    budget = {"spent": 0.0, "done": 0, "cached_tok": 0, "prompt_tok": 0, "stop": False}
    t0 = time.time()
    fh = open(out, "a")

    def work(row):
        if budget["stop"]:
            return
        try:
            rec = eur.run_claim(row, ng_scores, budget, {})
        except Exception as e:  # noqa: BLE001
            with lock:
                budget["done"] += 1
                print(f"  [claim-failed] {row['review_url']}: {type(e).__name__}: {e}", flush=True)
            return
        for k in ("post_id", "ng_score", "bin", "lean", "handle", "why"):
            rec[k] = row.get(k)
        with lock:
            budget["spent"] += rec["cost"]
            budget["done"] += 1
            fh.write(json.dumps(rec) + "\n")
            n = budget["done"]
            if n % 25 == 0 or n == len(todo):
                fh.flush()
                el = (time.time() - t0) / 60
                print(f"  {n}/{len(todo)} | spent ${budget['spent']:.3f} | projected ${budget['spent'] / n * len(todo):.2f} | "
                      f"{el:.0f}m | {n / max(el, .01):.1f} claims/min | ETA {(len(todo) - n) / max(n / max(el, .01), .01):.0f}m | "
                      f"gate {reader_lab.GATE.limit} inflight {reader_lab.GATE.inflight}", flush=True)
            if budget["spent"] >= args.budget:
                budget["stop"] = True
                print("  [budget cap reached, stopping]", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"done: {budget['done']} claims, ${budget['spent']:.3f}, {(time.time() - t0) / 60:.1f} min -> {out}  {reader_lab.GATE.summary()}")


if __name__ == "__main__":
    main()
