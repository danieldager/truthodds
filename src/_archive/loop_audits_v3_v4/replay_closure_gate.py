"""Pure-logic replay: run every supported/refuted close from the saved v3 runs through the
new v4 corroboration gate and compare against the audit's buckets. No API calls.

Expectations: audit-FAILED closes must be refused; audit-MET closes should pass except where
the gate is deliberately stricter (opinion exclusion on assertions); WEAK closes land where
the new rules put them (NG>=90 single, wikipedia-as-reliable, broader primary detection)."""
import glob
import json
from collections import Counter

import pandas as pd

from pipeline.verify_tweet_claims import _meets_bar, _qualifying, _rel_info, _OPINION_URL
from pipeline.search import newsguard_score_map
from scripts.step_corroboration_audit import bucket, SUPPORT_STANCES, REFUTE_STANCES

NG = newsguard_score_map()

RUNS = {
    "lowng_v3": "eval/data/survey_claims/runs/dev50_lowng_v3/results-*.jsonl",
    "main_v3": "eval/data/survey_claims/runs/dev50_main_v3/results-*.jsonl",
    "main_v3_rerun": "eval/data/survey_claims/runs/dev50_main_v3_rerun/results-*.jsonl",
}
PARQUET = "eval/data/survey_claims/dev500_claims.parquet"


def claim_types_by_url():
    df = pd.read_parquet(PARQUET)
    out = {}
    for url, g in df.groupby("url"):
        g = g.assign(_ord=g["claim_id"].str.split(":").str[-1].astype(int)).sort_values("_ord")
        g = g[g["checkworthy"]]
        out[url] = list(zip(g["claim"], g["type"]))
    return out


def main():
    types = claim_types_by_url()
    xtab = Counter()
    divergences = []
    snip_closes = []
    unmatched_types = 0
    total = 0
    for run, pat in RUNS.items():
        for fp in sorted(glob.glob(pat)):
            for line in open(fp):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not (r.get("ok") and r.get("result")):
                    continue
                res = r["result"]
                url_of = {}
                for rd in res.get("rounds") or []:
                    for d in rd.get("docs") or []:
                        url_of[d["id"]] = d.get("url") or ""
                # rebuild claims with types (old records store only the text)
                claims = [dict(c) for c in (res.get("claims") or [])]
                gold = types.get(res.get("id"), [])
                for i, c in enumerate(claims):
                    if i < len(gold) and gold[i][0] == c["c"]:
                        c["t"] = gold[i][1]
                    else:  # order/text mismatch: fall back to exact-text lookup
                        m = [t for txt, t in gold if txt == c["c"]]
                        c["t"] = m[0] if m else None
                        if not m:
                            unmatched_types += 1
                # enrich evidence with the v4 fields the old records lack
                evidence = []
                for e in res.get("evidence") or []:
                    e = dict(e)
                    u = "" if e.get("snippet_only") else url_of.get(e.get("src"), "")
                    e["rel"], e["ng"] = _rel_info(e.get("domain", ""), u, NG)
                    e["opinion"] = bool(_OPINION_URL.search(u)) if u else False
                    evidence.append(e)
                for cid_s, status in (res.get("ledger") or {}).items():
                    if status not in ("supported", "refuted"):
                        continue
                    cid = int(cid_s)
                    total += 1
                    # audit bucket, on the audit's own definitions
                    ents = [e for e in res["evidence"] if e.get("claim_id") == cid]
                    want = SUPPORT_STANCES if status == "supported" else REFUTE_STANCES
                    c_audit = {
                        "full_dir": [e for e in ents if not e.get("snippet_only")
                                     and (e.get("stance") or "").lower() in want],
                        "full_all": [e for e in ents if not e.get("snippet_only")],
                        "snip": [e for e in ents if e.get("snippet_only")],
                    }
                    b, _ = bucket(c_audit)
                    # the v4 gate
                    qual = _qualifying(evidence, cid, status, claims)
                    ok, why, snip_used = _meets_bar(qual)
                    xtab[(b, "PASS" if ok else "REFUSE")] += 1
                    if snip_used:
                        snip_closes.append((run, res.get("id"), cid, status, snip_used))
                    expect_pass = b.startswith("MET")
                    if ok != expect_pass:
                        divergences.append((run, res.get("id"), cid, status,
                                            claims[cid - 1].get("t"), b,
                                            "PASS" if ok else "REFUSE", why))
    print(f"closes replayed: {total}   (claims w/o type match: {unmatched_types})\n")
    print(f"{'audit bucket':22s} {'gate':8s} n")
    for (b, g), n2 in sorted(xtab.items()):
        print(f"{b:22s} {g:8s} {n2}")
    npass = sum(n for (b, g), n in xtab.items() if g == "PASS")
    print(f"\ngate pass rate: {npass}/{total} ({100*npass/total:.0f}%) — "
          f"refusals would have forced another round / resolved unsupported")
    print(f"\n=== SNIPPET-CORROBORATED PASSES (high-rel snippet filled the 2nd slot) — "
          f"{len(snip_closes)} ===")
    for run, pid, cid, st, snips in snip_closes:
        print(f"  [{run}] {pid.split('/')[-1][:24]} claim {cid} = {st}  <- {snips}")
    print(f"\n=== DIVERGENCES from naive expectation (MET<->gate) — {len(divergences)} ===")
    for run, pid, cid, st, t, b, g, why in divergences:
        print(f"\n[{run}] {pid} claim {cid} = {st} (type={t})")
        print(f"  audit={b}  gate={g}  |  {why}")


if __name__ == "__main__":
    main()
