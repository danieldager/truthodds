"""UNRESTRAINED production verification instrument over the CN survey claims (Daniel 2026-09-16).

WHAT. The 1,660 kept survey claims (`community_notes/survey_claims_2026-09-16.parquet`) are run
through Arm S of the 2026-09-15 production regime — production query-v3 -> Serper top-10 ->
production scrape/select_regions -> read-v6.1 plain — and graded with the ARM-S six-flag urn
weights fitted that day (eval/data/urn_runs/e1_prodregime/prodregime_metrics.json, conditions.S).

UNRESTRAINED (every constraint the regime normally applies, and its state here):
  date ceiling            OFF  — date_ceiling=None on every Serper call; no post-hoc leak drop.
  fact-check domain drop  OFF  — --keep-fc (default). fc_domain is tagged in the record and READ.
  echo / wire-copy pass   OFF  — --no-echo (default). No echo call is made; nothing is suppressed
                                 for being a duplicate or a fact-check restatement.
  origin-source exclusion ON   — the post itself can never be evidence for its own claim. Two
                                 layers: (1) x.com / twitter.com / t.co are in pipeline.config's
                                 SCRAPE_BLOCKLIST, enforced server-side as -site: operators AND
                                 client-side as a guaranteed drop in search.py; (2) any retrieved
                                 URL containing the post id is dropped as `origin-dropped`
                                 (a syndicated copy / embed of the same post).
  publisher exclusion     n/a  — there is no origin outlet for a tweet claim; nothing else excluded.

The claim text is all the instrument sees, plus the POST DATE (production passes a claim date to
query-v3 and to the read claim block). The Community Notes label, note text and lean never enter
any prompt.

Reuses the prodregime runner's primitives (ApiCounter, per-claim cache, gated read/flash) so the
instrument is byte-identical to Arm S; only the corpus, the leak controls and the query source
differ (queries are generated here because these claims have none).

  # SMOKE (20 claims, 5 per lean x label cell, seed 20260916):
  uv run python -m eval.scripts.build_eval.cn_survey_verify --smoke
  # FULL (after approval):
  uv run python -m eval.scripts.build_eval.cn_survey_verify --full --preflight
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from pipeline import disk_cache  # noqa: E402
from pipeline.config import FACT_CHECK_DOMAINS  # noqa: E402
from pipeline.credibility import TRUSTED_FACTCHECKERS  # noqa: E402
from pipeline.search import SearchError, cache_key, scrape, search  # noqa: E402
from pipeline.verify_tweet_claims import _domain_of  # noqa: E402
from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval import evidence_urn_run as eur  # noqa: E402
from eval.scripts.build_eval import prodregime_run as pr  # noqa: E402
from eval.scripts.build_eval import reader_lab  # noqa: E402
from eval.scripts.build_eval.read_v5_prompts import READ_SYS_MODE, READ_SYS_V6_1  # noqa: E402

# --reader selects the read prompt (see evidence_urn_run.READERS): read-v5 (frozen
# seven-class production instrument, hash 92404a300e14, DEFAULT) or read-v6.1 (the
# six-class 2026-09-14 experiment). The read cache key carries READ_HASH, so v5 reads
# stay distinct from v6.1 reads.
READERS = {"read-v6.1": (READ_SYS_V6_1, "read-v6.1"),
           "read-v5": (READ_SYS_MODE, "read-v5")}
READ_PROMPT_V = "read-v5"
READ_SYS = READ_SYS_MODE        # default reader (read-v5, frozen); main() may reassign per --reader

CLAIMS = SRC / "eval/data/community_notes/survey_claims_2026-09-16.parquet"
GROUPS = SRC / "eval/data/community_notes/dedup_groups_2026-09-15.parquet"
OUT_DIR = SRC / "eval/data/urn_runs/cn_survey"
FC_DOMS = set(FACT_CHECK_DOMAINS) | {d for d in TRUSTED_FACTCHECKERS if "/" not in d}
TOP_K = 10
SMOKE_SEED = 20260916
# Halved for a shared pool (Daniel 2026-09-16): another session is on DeepInfra at the same time.
GATE_TARGET, GATE_CEILING = 80.0, 100.0
RUN_TAG = "cn_survey_2026-09-16"

# per-claim cache lives under this run's dir, not prodregime's (set in run(); importing a
# build_eval module must have no side effects)
CACHE = OUT_DIR / "cache"
_ck, cache_get, cache_put = pr._ck, pr.cache_get, pr.cache_put
ApiCounter = pr.ApiCounter

READ_HASH = prompt_hash(READ_SYS)
QUERY_HASH = prompt_hash(eur.QUERY_SYS)
HASHES = {"read_sys": READ_HASH, "serper_query_prompt": QUERY_HASH}


# -------------------------------------------------------------------- query generation
def gen_query(cid: str, claim: str, claim_date: str, refresh: bool) -> tuple[str, float]:
    """Production query-v3: same system prompt and same user block as evidence_urn_run
    (CLAIM + CLAIM DATE; the neutral-context line does not exist for these claims)."""
    qk = _ck("q3", cid, QUERY_HASH)
    hit = cache_get("q3", qk, refresh)
    if hit is not None:
        return hit["query"], 0.0
    block = f"CLAIM: {claim}"
    if claim_date:
        block += f"\nCLAIM DATE: {claim_date}"
    q, cost = "", 0.0
    for _ in range(3):
        obj, c = pr.flash_json(eur.QUERY_SYS, block, max_tokens=80)
        cost += c
        q = (obj.get("query") or "").strip()[:300]
        if q:
            break
    cache_put("q3", qk, {"query": q})
    return q, cost


# -------------------------------------------------------------------- retrieval
def serper_fetch(cid: str, query: str, serper: ApiCounter, refresh: bool) -> tuple[list[dict], bool]:
    """Production Serper, ceiling OFF, no caller-side domain exclusions (x.com/twitter.com/t.co are
    already excluded by SCRAPE_BLOCKLIST server- and client-side). Cached per claim."""
    ck = _ck("serper", cid, prompt_hash(query))
    hit = cache_get("serper", ck, refresh)
    if hit is not None:
        return hit, True
    dk = cache_key("serper", query, TOP_K, None, [], 0)
    disk = disk_cache.get("serper", dk)
    if disk is not None and not refresh:
        cache_put("serper", ck, disk)
        return disk, True
    serper.take()
    res = search(query, TOP_K, date_ceiling=None, exclude_domains=[],
                 min_results=0, provider="serper")
    cache_put("serper", ck, res)
    return res, False


def prep_doc(h: dict, claim: str, query: str, post_id: str) -> dict:
    """Fetch text (provider content or scrape, snippet fallback), select_regions, tag fc_domain.
    UNRESTRAINED: fact-check domains are NOT dropped. The only drop is the origin post itself."""
    url = h.get("url") or ""
    dom = _domain_of(url)
    fc = any(dom == f or dom.endswith("." + f) for f in FC_DOMS)
    entry = {"url": url, "domain": dom, "fc_domain": fc, "date": h.get("date"),
             "provider": h.get("provider")}
    if post_id and post_id in url:
        entry["read_status"] = "origin-dropped"   # a copy/embed of the post being checked
        return entry
    text = h.get("content") or scrape(url)
    prov = "content" if h.get("content") else "scrape"
    if not text or len(text) < eur.SNIPPET_MIN_TEXT:
        text, prov = h.get("snippet") or "", "snippet"
    entry["provenance"] = prov
    regions, _meta = eur.select_regions(text, claim, query)
    smap = {}
    for rids, sel in regions:
        for i, sn in zip(rids, sel):
            smap[i] = sn
    if not smap:
        entry["read_status"] = "empty-doc"
        entry["sent_ids"], entry["sents"] = [], []
        return entry
    entry["read_status"] = "prepped"
    entry["sent_ids"] = sorted(smap)
    entry["sents"] = [smap[i] for i in sorted(smap)]
    return entry


def prep_docs_parallel(hits, claim: str, query: str, post_id: str) -> list[dict]:
    hits = list(hits or [])[:TOP_K]
    if not hits:
        return []
    with ThreadPoolExecutor(max_workers=min(TOP_K, len(hits)), thread_name_prefix="prep") as ex:
        return list(ex.map(lambda h: prep_doc(h, claim, query, post_id), hits))


# -------------------------------------------------------------------- claims
def load_claims(claims_path: Path = CLAIMS) -> dict[str, dict]:
    c = pl.read_parquet(claims_path)
    if "date" not in c.columns:   # the 2026-09-16 pool carries its dates in the dedup groups file
        g = (pl.read_parquet(GROUPS).select(["post_id", "date"]).unique(subset=["post_id"]))
        c = c.join(g, on="post_id", how="left")
    out = {}
    for r in c.iter_rows(named=True):
        out[r["claim_id"]] = {
            "claim_id": r["claim_id"], "post_id": r["post_id"], "claim": r["claim"],
            "post_date": str(r["date"] or "")[:10], "lean": r.get("lean"), "label": r.get("label"),
            "month": r.get("month"), "subtopic": r.get("subtopic"), "group_id": r.get("group_id")}
    return out


def record_complete(rec: dict) -> bool:
    return not rec.get("error") and not rec.get("serper_error")


# -------------------------------------------------------------------- run
def run(ids: list[str], claims: dict[str, dict], out_tag: str, budget_cap: float,
        serper_cap: int, workers: int, retr_workers: int, preflight: bool,
        refresh: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pr.CACHE = CACHE
    serper = ApiCounter(OUT_DIR / f"serper_credits_{out_tag}.json", serper_cap)
    reader_lab.GATE = reader_lab.RateGate(target=GATE_TARGET, ceiling=GATE_CEILING, floor=32)

    outp = OUT_DIR / f"results_{out_tag}.jsonl"
    done = set()
    if outp.exists() and not refresh:
        for cid, r in _last_wins(outp).items():
            if record_complete(r):
                done.add(cid)
    todo = [c for c in ids if c not in done]
    print(f"loaded {len(ids)} claims | {len(done)} already done | {len(todo)} to run | "
          f"retr_workers={retr_workers} read_workers={workers} | "
          f"gate {GATE_TARGET:.0f}/{GATE_CEILING:.0f} | "
          f"UNRESTRAINED: ceiling OFF, fc-drop OFF, echo OFF, origin-exclusion ON", flush=True)

    spent = [0.0]
    charge_lock, write_lock, prog_lock = (threading.Lock() for _ in range(3))
    prog = {"claims": 0, "reads": 0, "errors": 0}
    t0 = time.time()
    abort = threading.Event()
    abort_reason = [None]

    def charge(c):
        with charge_lock:
            spent[0] += c
            if spent[0] > budget_cap:
                raise SystemExit(f"BUDGET CAP ${budget_cap} breached at ${spent[0]:.3f}")

    def do_read(job, force=False):
        cid, rank, blk, sids, sents = job
        doc = "\n".join(f"[{i}] {s}" for i, s in zip(sids, sents))
        rk = _ck("read", READ_HASH, blk, doc)
        res = None if force else cache_get("read", rk, refresh)
        if res is not None and res.get("qc_flag") == "read-failed" and res.get("err"):
            res = None
        cost = 0.0
        if res is None:
            try:
                res, cost = pr.prod_read(READ_SYS, blk, sids, sents)
            except Exception as e:  # noqa: BLE001
                res = {"direction": "I", "evidence": [], "reason": "",
                       "qc_flag": "read-failed", "err": repr(e)}
            cache_put("read", rk, res)
        charge(cost or 0.0)
        with prog_lock:
            prog["reads"] += 1
        return (cid, rank), res

    read_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="read")

    def process_claim(cid, fh):
        if abort.is_set():
            return
        c = claims[cid]
        rec = {"claim_id": cid, "post_id": c["post_id"], "claim": c["claim"],
               "post_date": c["post_date"], "lean": c["lean"], "label": c["label"],
               "month": c["month"], "subtopic": c["subtopic"], "group_id": c["group_id"],
               "cost": 0.0, "docs": []}
        try:
            q, qcost = gen_query(cid, c["claim"], c["post_date"], refresh)
            rec["cost"] += qcost
            charge(qcost)
            rec["query"] = q
            if not q:
                rec["serper_error"] = "empty query"
            else:
                try:
                    hits, cached = serper_fetch(cid, q, serper, refresh)
                    rec["cached"] = cached
                    rec["docs"] = prep_docs_parallel(hits, c["claim"], q, c["post_id"])
                except SearchError as e:
                    rec["serper_error"] = f"serper: {e}"
            blk = pr.claim_block_plain(c["claim"], c["post_date"])
            rec["claim_block"] = blk
            jobs = [(cid, i, blk, d["sent_ids"], d["sents"])
                    for i, d in enumerate(rec["docs"]) if d.get("read_status") == "prepped"]
            out = {}
            for f in [read_pool.submit(do_read, j) for j in jobs]:
                k, r = f.result()
                out[k] = r
            by_key = {(j[0], j[1]): j for j in jobs}
            for _ in range(2):
                failed = [by_key[k] for k, r in out.items()
                          if r.get("qc_flag") == "read-failed" and k in by_key]
                if not failed:
                    break
                for f in [read_pool.submit(do_read, j, True) for j in failed]:
                    k, r = f.result()
                    out[k] = r
            for i, d in enumerate(rec["docs"]):
                r = out.get((cid, i))
                if r is not None:
                    d["read"] = {"direction": r.get("direction"), "evidence": r.get("evidence"),
                                 "reason": r.get("reason"), "qc_flag": r.get("qc_flag")}
            with write_lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            with prog_lock:
                prog["claims"] += 1
        except SystemExit as se:
            if not abort.is_set():
                abort.set()
                abort_reason[0] = str(se)
                print(f"  [ABORT] {se}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"  [claim-error] {cid}: {ex!r}", flush=True)
            traceback.print_exc()
            rec["error"] = repr(ex)
            with write_lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
            with prog_lock:
                prog["claims"] += 1
                prog["errors"] += 1

    if preflight and todo:
        reader_lab.GATE.target = 48.0
        print("  [preflight] warming up at up to 48 in-flight; 429 check at 60s", flush=True)

        def _pf():
            time.sleep(60)
            n, _n429, rate = reader_lab.GATE.window_stats()
            print(f"  [preflight] 429 rate {rate:.1%} over {n} reqs", flush=True)
            if rate > 0.20:
                print("  [preflight] ABORT: 429 rate over 20%", flush=True)
                abort.set()
            else:
                with reader_lab.GATE.cv:
                    reader_lab.GATE.target = GATE_TARGET
                    reader_lab.GATE.events.clear()
                    reader_lab.GATE.cv.notify_all()
                print(f"  [preflight] OK -> target raised to {GATE_TARGET:.0f} "
                      f"(gate recovers toward ceiling {GATE_CEILING:.0f})", flush=True)
        threading.Thread(target=_pf, daemon=True).start()

    stop_prog = threading.Event()

    def _reporter():
        while not stop_prog.wait(60):
            with prog_lock:
                cd, rd = prog["claims"], prog["reads"]
            el = (time.time() - t0) / 60
            cpm = cd / el if el > 0 else 0.0
            eta = ((len(todo) - cd) / cpm) if cpm > 0 else float("inf")
            print(f"  [progress] {cd}/{len(todo)} claims | {cpm:.1f} claims/min | reads {rd} | "
                  f"serper {serper.n} | ${spent[0]:.3f} | {reader_lab.GATE.summary()} | "
                  f"{el:.1f}m elapsed | ETA {eta:.1f}m", flush=True)
    threading.Thread(target=_reporter, daemon=True).start()

    with outp.open("a") as fh:
        with ThreadPoolExecutor(max_workers=retr_workers, thread_name_prefix="claim") as cex:
            list(cex.map(lambda cid: process_claim(cid, fh), todo))
        if not abort.is_set():
            ts = set(todo)
            retry = sorted(cid for cid, r in _last_wins(outp).items()
                           if cid in ts and not record_complete(r))
            if retry:
                print(f"  [rerun] {len(retry)} errored claims, one pass", flush=True)
                with ThreadPoolExecutor(max_workers=retr_workers, thread_name_prefix="rerun") as cex:
                    list(cex.map(lambda cid: process_claim(cid, fh), retry))
    stop_prog.set()
    read_pool.shutdown(wait=True)

    unique = _dedupe(outp)
    still_bad = [cid for cid, r in _last_wins(outp).items() if not record_complete(r)]
    if abort.is_set():
        print(f"\nABORTED {out_tag}: {abort_reason[0]} | serper {serper.n} | "
              f"${spent[0]:.3f} | {unique} unique", flush=True)
        raise SystemExit(f"ABORTED: {abort_reason[0]}")
    meta = {"tag": out_tag, "run_tag": RUN_TAG, "n_claims": unique,
            "errored_records": len(still_bad), "serper_credits": serper.n,
            "api_cost_usd_this_process": round(spent[0], 4), "hashes": HASHES,
            "prompts": {"query": eur.QUERY_PROMPT_V, "read": "read-v6.1"},
            "model": eur.VERIFICATION_MODEL,
            "regime": {"date_ceiling": None, "fc_domain_drop": False, "echo_pass": False,
                       "exclude_domains_arg": [], "origin_exclusion": True,
                       "gate": {"target": GATE_TARGET, "ceiling": GATE_CEILING, "floor": 32},
                       "origin_exclusion_how": "SCRAPE_BLOCKLIST x.com/twitter.com/t.co "
                                               "(server+client) + post_id-in-URL drop",
                       "top_k": TOP_K},
            "wall_min_this_process": round((time.time() - t0) / 60, 2)}
    (OUT_DIR / f"meta_{out_tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDONE {out_tag}: {unique} claims | {len(still_bad)} errored | ${spent[0]:.3f} | "
          f"serper {serper.n} | {(time.time()-t0)/60:.1f}m -> {outp}", flush=True)


def _last_wins(path: Path) -> dict[str, dict]:
    seen = {}
    if path.exists():
        for l in path.open():
            try:
                r = json.loads(l)
                seen[r["claim_id"]] = r
            except Exception:  # noqa: BLE001
                pass
    return seen


def _dedupe(path: Path) -> int:
    seen = _last_wins(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in seen.values():
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)
    return len(seen)


def smoke_ids(claims: dict[str, dict], per_cell: int = 5) -> list[str]:
    """Stratified: per_cell claims from each lean x label cell, seed 20260916."""
    cells = {}
    for cid, c in claims.items():
        cells.setdefault((c["lean"], c["label"]), []).append(cid)
    rng = random.Random(SMOKE_SEED)
    out = []
    for k in sorted(cells):
        out += sorted(rng.sample(sorted(cells[k]), per_cell))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--serper-cap", type=int, default=40)
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--retr-workers", type=int, default=24)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--reader", choices=list(READERS), default=READ_PROMPT_V,
                    help="read prompt: read-v6.1 (production, DEFAULT) or read-v5 (frozen "
                         "seven-class survey instrument). Prompt text is unchanged.")
    ap.add_argument("--claims", type=Path, default=CLAIMS,
                    help="claims parquet (claim_id, post_id, claim, date); default = the 2026-09-16 pool")
    a = ap.parse_args()

    global READ_SYS, READ_HASH, HASHES
    READ_SYS = READERS[a.reader][0]
    READ_HASH = prompt_hash(READ_SYS)
    HASHES = {"read_sys": READ_HASH, "serper_query_prompt": QUERY_HASH}

    claims = load_claims(a.claims)
    if a.smoke:
        ids = smoke_ids(claims)
        tag = a.tag or "smoke20"
    elif a.full:
        ids = sorted(claims)
        tag = a.tag or "full"
        if a.budget == 1.0:
            a.budget = 12.0
        if a.serper_cap == 40:
            a.serper_cap = 1800
    else:
        ap.error("one of --smoke / --full required")
    run(ids, claims, tag, a.budget, a.serper_cap, a.workers, a.retr_workers,
        a.preflight, a.refresh)


if __name__ == "__main__":
    main()
