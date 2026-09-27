"""Run-health report for an urn run — the checks fit_urn does NOT do.

fit_urn answers "does the signal separate true from false". This answers "is the
instrument behaving", which is what a staged-run checkpoint needs: the funnel,
non-observations, QC rates, date-leak accounting, and whether the flag mix moves
with gold veracity in the direction it must.

    uv run python -m eval.scripts.build_eval.run_health -i eval/data/urn_runs/e1_ctx/results-00.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

FLAGS = ("5", "4", "3", "2", "1", "X", "I")
SUPPORT, REFUTE = ("5", "4"), ("1", "2")


def pct(a, b):
    return f"{a/b:.1%}" if b else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, type=Path)
    args = ap.parse_args()
    recs = [json.loads(l) for l in args.input.open()]

    excluded = [r for r in recs if r.get("excluded")]
    run = [r for r in recs if not r.get("excluded")]
    print(f"=== FUNNEL ===\nrecords {len(recs)} | excluded {len(excluded)} | ran {len(run)}")
    if excluded:
        print("  exclusions:", dict(collections.Counter(r["excluded"] for r in excluded)))
    nres = [len(r["results"]) for r in run]
    zero = sum(1 for n in nres if n == 0)
    print(f"  results/claim: mean {sum(nres)/max(len(nres),1):.1f} | "
          f"zero-result claims {zero} ({pct(zero, len(run))})")

    docs = [d for r in run for d in r["results"]]
    print(f"\n=== NON-OBSERVATIONS ({len(docs)} docs) ===")
    st = collections.Counter(d["read_status"] for d in docs)
    for k, v in st.most_common():
        print(f"  {k:12s} {v:5d}  {pct(v, len(docs))}")
    prov = collections.Counter(d["provenance"] for d in docs)
    print("  provenance:", dict(prov), "| snippet share", pct(prov["snippet"], len(docs)))

    ok = [d for d in docs if d["read_status"] == "ok"]
    print(f"\n=== QC (over {len(ok)} model reads) ===")
    qc = collections.Counter(d["read"].get("qc_flag") or "clean" for d in ok)
    for k, v in qc.most_common():
        print(f"  {k:22s} {v:5d}  {pct(v, len(ok))}")
    directional = [d for d in ok if d["read"]["direction"] in SUPPORT + REFUTE]
    blanket = [d for d in directional if d["read"].get("qc_flag") == "blanket-citation"]
    print(f"  directional reads {len(directional)} | blanket-cited {len(blanket)} "
          f"({pct(len(blanket), len(directional))})  <- silence-as-refutation proxy")

    print("\n=== DATE INTEGRITY ===")
    leaks = [d for d in docs if d.get("leak_flag")]
    print(f"  results dated after ceiling: {len(leaks)} ({pct(len(leaks), len(docs))})")
    print("  ceiling_src:", dict(collections.Counter(r["ceiling_src"] for r in run)))
    noceil = sum(1 for r in run if not r["ceiling"])
    print(f"  claims with no ceiling: {noceil}")

    print("\n=== FLAG MIX BY GOLD VERACITY (the direction check) ===")
    print(f"  {'veracity':10s} {'claims':>6s} {'docs':>6s} " +
          " ".join(f"{f:>5s}" for f in FLAGS) + "   sup%  ref%")
    for v in (1, 2, 3, 4, 5):
        grp = [r for r in run if r.get("veracity") == v]
        gd = [d for d in grp for d in [d]] and [d for r in grp for d in r["results"]]
        if not gd:
            continue
        c = collections.Counter(d["read"]["direction"] for d in gd)
        s = sum(c[f] for f in SUPPORT)
        rf = sum(c[f] for f in REFUTE)
        print(f"  {v:<10d} {len(grp):6d} {len(gd):6d} " +
              " ".join(f"{c[f]:5d}" for f in FLAGS) +
              f"  {pct(s,len(gd)):>5s} {pct(rf,len(gd)):>5s}")

    print("\n=== STRATA (claims) ===")
    for key in ("screen_verdict", "resolution_status"):
        c = collections.Counter(r.get(key) or "?" for r in run)
        print(f"  {key}: {dict(c.most_common(8))}")
    ctxu = sum(1 for r in run if r.get("context_used"))
    dated = sum(1 for r in run if r.get("claim_date_shown"))
    print(f"  context_used {ctxu} ({pct(ctxu,len(run))}) | "
          f"date shown {dated} ({pct(dated,len(run))})")

    print("\n=== COST ===")
    print(f"  ${sum(r.get('cost', 0) for r in recs):.3f} over {len(recs)} records")


if __name__ == "__main__":
    main()
