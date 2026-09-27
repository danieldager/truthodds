"""Replay synthesis -> Likert over the CACHED evidence pool (no search/scrape).

The full evidence (snippets + summaries + quotes per doc) is already in each claim's trace JSON.
We rebuild the EvidenceDoc pool from rounds[].docs and re-run ONLY the LLM reasoning with the NEW
synthesis + Likert prompts. Tests BOTH prompt changes on FIXED evidence, ~2 LLM calls/claim, zero
paid search. 4-class is dropped — the flag-if-veracity<=3 headline needs only veracity.

Caveat: the new synthesis can't fetch evidence it might now want (e.g. re-searching a media-
authenticity claim), so this UNDER-estimates the gain on proposition-substitution cases needing new
searches. It FULLY captures the overstatement/misleadingness bucket (evidence already present).

Visibility (per Daniel's standing pref): unbuffered, flushed progress + rate + ETA every 10 claims,
incremental checkpoint to replay.parquet every 25, loud per-claim error capture.

  uv run --directory <src> python -u -m eval.scripts.verification_grading.replay_synth
"""
import glob
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import polars as pl

from config import VERIFICATION_MODEL
from pipeline.verify import _synthesise, _evaluate_likert, EvidenceDoc

D = "eval/scripts/verification_grading/data"
T = f"{D}/verdicts_dev_content_cascade_trace"
CKPT = "/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/767f3979-8040-4dcd-a252-87212d87cf8b/scratchpad/replay.parquet"
WORKERS = 6
def cid(u): return hashlib.md5((u or "").encode()).hexdigest()[:16]
def log(m): print(m, flush=True)


def pool_from_trace(t: dict) -> list[EvidenceDoc]:
    by_url: dict[str, EvidenceDoc] = {}
    for rd in t.get("rounds") or []:
        for d in rd.get("docs") or []:
            doc = EvidenceDoc(
                url=d.get("url", ""), provider=d.get("provider", ""),
                snippet=d.get("snippet", "") or "", scraped=bool(d.get("scraped", False)),
                relevant=bool(d.get("relevant", True)), publication_date=d.get("publication_date"),
                summary=d.get("summary"), quotes=list(d.get("quotes") or []))
            prev = by_url.get(doc.url)
            if prev is None or (doc.summary and not prev.summary):
                by_url[doc.url] = doc
    return list(by_url.values())


def replay_one(t: dict) -> dict:
    try:
        claim = t["claim_text"]; img = t.get("image_context") or None
        pool = pool_from_trace(t)
        syn = _synthesise(claim, pool, list(t.get("past_queries") or []), VERIFICATION_MODEL, image_context=img)
        lk = _evaluate_likert(claim, syn["analysis"], VERIFICATION_MODEL, image_context=img)
        return {"claim_id": t["claim_id"], "veracity_new": lk["veracity"],
                "analysis_new": syn["analysis"], "likert_justif_new": lk["justification"],
                "n_docs": len(pool), "err": None}
    except Exception as e:  # noqa: BLE001 — loud capture, keep the pool going
        return {"claim_id": t.get("claim_id"), "veracity_new": None, "analysis_new": "",
                "likert_justif_new": "", "n_docs": 0, "err": f"{type(e).__name__}: {e}"}


# headline set: 319 in-scope content claims, OLD scores for comparison
verd = pl.read_parquet(f"{D}/verdicts_dev_content_cascade.parquet").rename({"veracity": "veracity_old"})
claims = pl.read_parquet(f"{D}/claims_dev_content.parquet").select("claim_id", "gold_veracity", "rating_subtype", "judged_axis")
sf = pl.read_parquet("eval/data/verdict_scope_floor.parquet").with_columns(
    pl.col("review_url").map_elements(cid, return_dtype=pl.Utf8).alias("claim_id")).select("claim_id", "stage3_in_scope")
base = (verd.join(claims, on="claim_id", how="inner").join(sf, on="claim_id", how="left")
        .with_columns(pl.col("stage3_in_scope").fill_null(True))
        .filter(pl.col("error").is_null() & pl.col("stage3_in_scope") & (pl.col("judged_axis") == "content")))
ids = set(base["claim_id"].to_list())
traces = [t for t in (json.load(open(fp)) for fp in glob.glob(f"{T}/*.json")) if t.get("claim_id") in ids]
N = len(traces)
log(f"replaying {N} in-scope claims | synthesis+likert on cached evidence | {VERIFICATION_MODEL} | {WORKERS} workers")

out, errs, t0 = [], 0, time.time()
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(replay_one, t) for t in traces]
    for i, f in enumerate(as_completed(futs), 1):
        r = f.result(); out.append(r)
        if r["err"]:
            errs += 1; log(f"  ! [{i}/{N}] {r['claim_id']}: {r['err']}")
        if i % 10 == 0 or i == N:
            el = time.time() - t0; rate = i / el
            eta = (N - i) / rate if rate else 0
            log(f"  {i}/{N}  {rate*60:.1f} claims/min  elapsed {el/60:.1f}m  ETA {eta/60:.1f}m  errors={errs}")
        if i % 25 == 0 or i == N:
            pl.DataFrame(out).write_parquet(CKPT)

new = pl.DataFrame(out)
new.write_parquet(CKPT)
m = base.join(new.filter(pl.col("veracity_new").is_not_null()), on="claim_id", how="inner")
g = m["gold_veracity"].to_numpy(); po = m["veracity_old"].to_numpy(); pn = m["veracity_new"].to_numpy()

def report(p, tag):
    pf = p <= 3; gf = g <= 3
    tp = int((gf & pf).sum()); fp = int((~gf & pf).sum()); fn = int((gf & ~pf).sum())
    log(f"{tag}: flag-if<=3 ACC={(pf==gf).mean():.3f}  recall={tp/(tp+fn):.3f} precision={tp/(tp+fp):.3f}  "
        f"MAE={np.abs(p-g).mean():.3f}  exact={(p==g).mean():.2f}  | misses(FN)={fn} over-flags(FP)={fp}")

log(f"\n=== RESULTS (n={m.height}, errors={errs}) ===")
report(po, "OLD (cascade baseline)        ")
report(pn, "NEW (replay synth+likert)     ")
of = (po <= 3) != (g <= 3); nf = (pn <= 3) != (g <= 3)
log(f"prior errors fixed: {int((of & ~nf).sum())}/{int(of.sum())}   regressions: {int((~of & nf).sum())}/{int((~of).sum())}")
m.write_parquet(CKPT)
log(f"saved {CKPT}")
sys.exit(0)
