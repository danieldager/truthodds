"""Rebuild READ test-suite cases on the CURRENT instrument.

The failure modes were found on the 25-July pilot, which ran with a 320-char sentence
cap (64% of documents truncated mid-word), no client-side blocklist (8% UGC), and the
pre-27-July Jina gate. Testing prompt changes against those cases would measure the
prompt against artefacts.

This re-derives them: same claims and same document URLs, but re-prepped with the
current splitter (MAX_SENT_CHARS=800) and current region tiling, with every blocklisted
domain dropped, and re-read with the current prompt to get a clean baseline.

No search calls. The URLs are already known, so retrieval is not repeated — which also
means no Serper spend and no risk of a different result set.

scrape() is allowed to use its disk cache on purpose. Forcing a fresh fetch would pull
each page as it stands TODAY, and many have since been updated to carry the fact-check's
own conclusion: that is a ceiling leak. The truncation defect was in sentences(), not in
scrape(), so re-prepping cached raw text is what "untruncated" actually requires.

  uv run python -m eval.scripts.build_eval.build_read_suite \
      -i eval/data/urn_runs/e1/results-00.jsonl -o eval/data/read_suite/cases.json \
      --per-mode 25 --workers 8
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline.config import SCRAPE_BLOCKLIST  # noqa: E402
from pipeline.search import no_content, scrape  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
    MAX_REGIONS, aggregate_reads, read_doc, select_regions,
)

SNIPPET_MIN_TEXT = 400
STALE_HINT = re.compile(r"\b(19[89]\d|20[01]\d)\b")


def norm(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


# ---- failure-mode signatures, evaluated on the FRESH read --------------------
def signatures(c: dict) -> list[str]:
    """Which failure modes this (claim, document) case is a candidate for.

    Heuristics for where a mode tends to live, not assertions. Verdict-vs-direction
    disagreement mixes genuine misreads with faithful reads of wrong documents; the
    label step is what separates them.
    """
    out, d = [], c["model_dir"]
    wrong = bool(c["gold_dir"] and d in ("supports", "refutes") and d != c["gold_dir"])
    right = bool(c["gold_dir"] and d in ("supports", "refutes") and d == c["gold_dir"])
    inst = c["rel"] in ("RELIABLE", "PRIMARY")
    cited = c["model_support"] + c["model_against"] + c["model_context"]
    sm = dict(zip(c["sent_ids"], c["sents"]))
    claim_toks = set(norm(c["claim"]))
    restate = 0.0
    for i in c["model_support"]:
        t = set(norm(sm.get(i, "")))
        if t and claim_toks:
            restate = max(restate, len(t & claim_toks) / len(claim_toks))
    own_empty = d in ("supports", "refutes") and not (
        c["model_support"] if d == "supports" else c["model_against"])
    dup = bool((set(c["model_support"]) & set(c["model_against"]))
               or (set(c["model_support"]) & set(c["model_context"]))
               or (set(c["model_against"]) & set(c["model_context"])))
    stale = bool(STALE_HINT.search(" ".join(sm.get(i, "") for i in cited[:6])))

    if wrong and d == "supports" and inst and not c["fc_domain"] and restate < 0.6:
        out.append("A1_scope_overreach")
    # The 27-July rule coerces these to `context` and stamps a flag, so by the time a
    # read is recorded the empty own-direction list is already gone. Key on the flag.
    if c["qc_flag"] == "empty-directional" or (own_empty and c["model_context"]):
        out.append("A2_context_swallows_refutation")
    if wrong and inst and d == "refutes" and c["gold_dir"] == "supports":
        out.append("A3_opinion_as_evidence")
    if d == "supports" and restate >= 0.6:
        out.append("A4_restates_claim_as_support")
    if wrong and stale and inst:
        out.append("B1_stale_document")
    if wrong and d == "refutes" and any(
            re.search(r"\d", sm.get(i, "")) for i in c["model_against"][:3]):
        out.append("B2_figure_mismatch")
    if c["fc_domain"] and d in ("supports", "refutes"):
        out.append("B5_verdict_relay")
    if dup:
        out.append("B7_polarity_confusion")
    if c["qc_flag"] == "region-conflict":
        out.append("D3_region_conflict")
    if c["prov"] == "snippet" and d in ("supports", "refutes") and len(c["sent_ids"]) <= 1:
        out.append("D4_snippet_single_sentence")
    if c["chars"] > 20000 and d in ("supports", "refutes"):
        out.append("LONG_doc_directional")
    if right and inst and c["prov"] == "scrape" and len(c["model_support"] + c["model_against"]) >= 2:
        out.append("CORRECT_directional")
    if d == "irrelevant" and c["chars"] > 4000 and inst:
        out.append("GENUINE_silence")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=Path)
    ap.add_argument("-o", "--out", required=True, type=Path)
    ap.add_argument("--per-mode", type=int, default=25, help="over-sample target per mode")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # ---- 1. candidate URLs from the old run, oversampled per OLD signature ----
    old, by_mode = [], collections.defaultdict(list)
    for line in args.input.open():
        r = json.loads(line)
        for d in r["results"]:
            rd = d.get("read") or {}
            if not rd:
                continue
            old.append((r, d, rd))
    print(f"old run: {len(old)} (claim, document) pairs", flush=True)

    # Stratify the draw by the OLD run's signature, or the rare modes never appear:
    # stale had 16 candidates in the whole pilot and region-conflict had 8, so an
    # unstratified top-N by hash would miss them entirely.
    cand, old_modes = {}, collections.defaultdict(list)
    for r, d, rd in old:
        cid = hashlib.blake2b((r["review_url"] + "|" + (d.get("url") or "")).encode(),
                              digest_size=8).hexdigest()
        cand[cid] = (r, d, rd)
        v = r.get("veracity")
        probe = {
            "claim": r["claim_text"], "gold_dir": None if v == 3 else ("supports" if v >= 4 else "refutes"),
            "rel": d.get("rel"), "prov": d.get("provenance"), "fc_domain": bool(d.get("fc_domain")),
            "chars": (d.get("prep") or {}).get("doc_chars") or 0,
            "sent_ids": d.get("sent_ids") or [], "sents": d.get("sents") or [],
            "model_dir": rd.get("direction"), "model_support": rd.get("support") or [],
            "model_against": rd.get("against") or [], "model_context": rd.get("neutral") or [],
            "qc_flag": rd.get("qc_flag") or "",
        }
        for m in signatures(probe):
            old_modes[m].append(cid)

    keep, seen = [], set()
    for m, cids in sorted(old_modes.items()):
        # shortest documents first within a mode, then stable hash order
        cids = sorted(set(cids), key=lambda c: (len(cand[c][1].get("sents") or []) > 60, c))
        took = 0
        for cid in cids:
            if cid in seen or took >= args.per_mode:
                continue
            seen.add(cid)
            keep.append(cid)
            took += 1
        print(f"  {m:34s} old pool {len(set(cids)):5d}  drawing {took}", flush=True)
    print(f"candidate documents to re-prep: {len(keep)}", flush=True)

    # ---- 2. re-scrape (cache-backed), re-prep, re-read ------------------------
    t0, done, lock_out = time.time(), [0], []
    total_cost = [0.0]

    def rebuild(cid):
        r, d, _ = cand[cid]
        url = d.get("url") or ""
        claim, query = r["claim_text"], r.get("query") or ""
        dom = (d.get("domain") or "").lower().removeprefix("www.")
        if any(dom == b or dom.endswith("." + b) for b in SCRAPE_BLOCKLIST):
            return {"cid": cid, "_dropped": "blocklisted"}
        text = scrape(url)                      # None if blocklisted, dead, or walled
        prov = "scrape"
        if not text or len(text) < SNIPPET_MIN_TEXT:
            # Snippet-provenance documents are 18% of the corpus and the worst-performing
            # slice (25% of their directional votes disagree with gold). Excluding them
            # for want of a scrape would leave the suite blind to that failure. The old
            # run's stored sentences ARE the prepped snippet, so re-prep those instead.
            old_sents = d.get("sents") or []
            if not old_sents:
                return {"cid": cid, "_dropped": "no-text"}
            text, prov = " ".join(old_sents), "snippet"
        if no_content(text):
            return {"cid": cid, "_dropped": "junk-gate"}
        regions, prep = select_regions(text, claim, query)
        if not regions:
            return {"cid": cid, "_dropped": "no-regions"}
        block = (f"CLAIM: {claim}\n(claimed on {r.get('ceiling') or 'unknown date'}; judge the "
                 f"document's bearing on this exact proposition)")
        reads = []
        for ids, sel in regions[:MAX_REGIONS]:
            out, cost, _, _ = read_doc(block, ids, sel, key=f"suite-{cid}")
            total_cost[0] += cost
            if out:
                reads.append(out)
        if not reads:
            return {"cid": cid, "_dropped": "read-failed"}
        agg = aggregate_reads(reads)
        ids_all, sents_all = [], []
        for ids, sel in regions[:MAX_REGIONS]:
            ids_all += list(ids)
            sents_all += list(sel)
        seen, uid, usent = set(), [], []
        for i, s in zip(ids_all, sents_all):
            if i not in seen:
                seen.add(i)
                uid.append(i)
                usent.append(s)
        v = r.get("veracity")
        rec = {
            "cid": cid, "claim": claim, "gold_veracity": v,
            "gold_dir": None if v == 3 else ("supports" if v >= 4 else "refutes"),
            "review_url": r["review_url"], "publisher": r.get("publisher_site"),
            "doc_url": url, "domain": d.get("domain"), "prov": prov,
            "rel": d.get("rel"), "ng": d.get("ng"), "fc_domain": bool(d.get("fc_domain")),
            "chars": prep.get("doc_chars") or len(text),
            "n_regions": len(reads), "coverage": prep.get("coverage"),
            "sent_ids": uid, "sents": usent,
            "model_dir": agg["direction"], "model_support": agg["support"],
            "model_against": agg["against"], "model_context": agg["neutral"],
            "qc_flag": agg.get("qc_flag") or "",
        }
        rec["_modes"] = signatures(rec)
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(rebuild, keep):
            lock_out.append(rec)
            done[0] += 1
            if done[0] % 25 == 0:
                el = time.time() - t0
                rate = done[0] / el
                print(f"  {done[0]}/{len(keep)}  {rate*60:.0f}/min  "
                      f"${total_cost[0]:.3f}  eta {(len(keep)-done[0])/rate/60:.0f}m", flush=True)

    good = [r for r in lock_out if "_dropped" not in r]
    drops = collections.Counter(r["_dropped"] for r in lock_out if "_dropped" in r)
    print(f"\nrebuilt: {len(good)} | dropped: {dict(drops)} | cost ${total_cost[0]:.3f} "
          f"| {(time.time()-t0)/60:.1f}m", flush=True)

    pools = collections.Counter(m for r in good for m in r["_modes"])
    print("\nfresh mode pools:")
    for m, n in pools.most_common():
        print(f"  {m:34s} {n:5d}")
    unmatched = [r for r in good if not r["_modes"]]
    print(f"  (no signature: {len(unmatched)})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"built": "current-instrument",
                                    "n": len(good), "pools": dict(pools),
                                    "cases": good}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
