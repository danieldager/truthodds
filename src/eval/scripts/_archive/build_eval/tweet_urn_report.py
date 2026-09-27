"""E2 analysis — evidence profile over NG-binned tweets.

Written to run IDENTICALLY on the 10% tranche and on the finished run, so the dress
rehearsal catches problems in the code that will produce the headline (Daniel
2026-08-05). Every number carries its denominator.

Sections:
  0 HEALTH      — instrument sanity; the part that catches problems early
  a FUNNEL      — posts -> claims -> run -> >=1 doc -> >=1 directional read, per bin
  b FLAG MIX    — 5/4/3/2/1/X/I per bin, per CLAIM and per POST. The I rate is a
                  headline, not a nuisance
  c SCORE       — E1's fitted weights applied to tweet claims. NOT re-fitted here:
                  there are no per-claim labels on tweets, so a fit is impossible and
                  the E1 constants are the whole point of the comparison
  d QUALITY     — reliability of the RETRIEVED evidence per bin
  e SWEEP       — nudge rate by cutoff, per bin, per CLAIM and per POST
  f CONTROLS    — the four pre-committed controls (see below)

PRE-COMMITTED before the run (run_ledger.md, ROADMAP.md), so none of this is chosen
after seeing results:
  - PER-POST is the headline unit; per-claim is secondary (Daniel 2026-08-05)
  - post-level aggregation is WORST-CASE: any nudging claim nudges the post
  - primary control is OUTLET-WEIGHTED bin averages (equal posts per outlet is the
    design, so equal outlet weight is what the design implies); claim-weighted shown
    beside it because a prolific outlet can otherwise carry a bin
  - prominence-matched subset as a robustness check
  - incumbent-outlets-only as the roster-swap sensitivity

  uv run python -m eval.scripts.build_eval.tweet_urn_report
  uv run python -m eval.scripts.build_eval.tweet_urn_report --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

RES = Path("eval/data/urn_runs/e2_tweets/results-00.jsonl")
DRAW = Path("eval/data/survey_claims/e2_draw_screen.parquet")   # the full drawn set
PROM = Path("eval/data/survey_claims/e2_claim_prominence.parquet")
ROSTER = Path("eval/data/survey_claims/source_roster.csv")

# E1 fitted weights + operating point. CITED, never re-fitted on tweets — and now
# READ FROM the E1 metrics file rather than pasted, so the two cannot drift apart
# (they did: this block still carried n=3,448 / AUC 0.850 from a superseded fit).
#
# 2026-08-20 refit: media-provenance claims (judged_axis == media_authenticity) are
# excluded from the E1 fit — the fact-checker graded whether a photo or video shows
# what it is presented as showing, which a text-only reader never assesses.
E1_METRICS = Path("eval/data/urn_runs/e1_ctx/headline_metrics.json")


# 2026-09-14: headline_metrics.json is the SIX-FLAG fit (read-v6.1: 5/4/X/I/2/1, no
# "3" class), so the weights arrive keyed by flag and a read-v5 "3" folds into "X";
# the voice collapse this function used to do is gone.
PAD_TO = 10


def _e1_constants() -> tuple[dict, float, dict]:
    m = json.loads(E1_METRICS.read_text())["overall"]
    return dict(m["weights"]), m["threshold"], m


W = E1_THRESHOLD = E1_FIT = None   # loaded by _load_e1() from main(); not at import time


def _load_e1() -> None:
    global W, E1_THRESHOLD, E1_FIT
    W, E1_THRESHOLD, E1_FIT = _e1_constants()


BINS = ["0-30", "30-50", "50-70", "70-90", "90-100"]
NEW_OUTLETS = {"floridadaily.com", "theblaze.com", "heartlandsignal.com",
               "foxbusiness.com", "truthout.org", "abcnews.go.com", "gpb.org",
               "levernews.com", "nytimes.com", "inthesetimes.com"}
FLAGS = ["5", "4", "3", "2", "1", "X", "I"]


def score(rec) -> float:
    """Claim log-odds under the E1 urn.

    Off-claim ("I") reads carry the fitted silent weight, not zero (Daniel
    2026-08-20). The weights being applied here were estimated under fit_urn's
    conventions and those are the only ones they are valid under, so the claim is
    also padded to PAD_TO slots with silent reads (the 2026-09-08 pad rule)."""
    s = 0.0
    n = 0
    for e in rec["results"]:
        d = e["read"]["direction"]
        w = W.get("X" if d == "3" else d)   # six-flag urn (read-v6.1): "3" folds into "X"
        if w is not None:
            s += w
            n += 1
    return s + max(0, PAD_TO - n) * W["I"]


def load() -> tuple[list[dict], list[dict]]:
    recs = [json.loads(l) for l in open(RES)]
    prom = {r["claim_id"]: r for r in pl.read_parquet(PROM).iter_rows(named=True)} \
        if PROM.exists() else {}
    for r in recs:
        p = prom.get(r["review_url"], {})
        r["coverage"] = p.get("coverage", "unknown")
        r["new_outlet"] = r.get("publisher_site") in NEW_OUTLETS
        r["_score"] = score(r) if not r.get("excluded") else None
    run = [r for r in recs if not r.get("excluded")]
    # PER-POST METRICS ARE INVALID ON A PARTIAL RUN. Claims are processed in shuffled
    # order, so a post's claims arrive at different times; "did ANY claim in this post
    # nudge" cannot be answered until all of them are back. Mark posts COMPLETE only when
    # every drawn claim of that post has a record (run or gated). Caught by the 10%
    # dress rehearsal, 2026-08-05 — the tranche showed 1.06 claims/post against a true
    # ~2.2, i.e. most posts were half-finished.
    drawn_per_post = Counter()
    if DRAW.exists():
        for r in pl.read_parquet(DRAW).iter_rows(named=True):
            drawn_per_post[r["post_id"]] += 1
    seen_per_post = Counter(r["post_id"] for r in recs)
    complete = {p for p, n in seen_per_post.items() if n >= drawn_per_post.get(p, 10**9)}
    for r in recs:
        r["post_complete"] = r["post_id"] in complete
    return recs, run


def pct(n, d):
    return f"{n}/{d} = {n/d:.1%}" if d else f"{n}/0 = n/a"


def section_health(recs, run):
    print("=" * 78)
    print("0. HEALTH — does the instrument look right?")
    print("=" * 78)
    gated = [r for r in recs if r.get("excluded")]
    print(f"  records {len(recs)} | run {len(run)} | gated {len(gated)}")
    print(f"  gate reasons: {dict(Counter(r['excluded'] for r in gated))}")
    bad = [r for r in gated if r.get("results")]
    print(f"  gated records carrying results (MUST be 0): {len(bad)}")
    st, docs, snip, leak = Counter(), 0, 0, 0
    for r in run:
        for e in r["results"]:
            st[e.get("read_status")] += 1
            docs += 1
            snip += e.get("provenance") == "snippet"
            leak += bool(e.get("leak_flag"))
    print(f"  docs {docs} ({docs/max(len(run),1):.1f}/claim) | read_status {dict(st)}")
    print(f"    failed reads   {pct(st['failed'], docs)}   (12-worker baseline 0.22%)")
    print(f"    snippet-only   {pct(snip, docs)}   (baseline 12.0%)")
    print(f"    LEAK FLAGS     {pct(leak, docs)}   (ceiling is broken if this is not ~0)")
    print(f"  claims with 0 docs: {sum(1 for r in run if not r['results'])}")
    print(f"  ceiling_src: {dict(Counter(r.get('ceiling_src') for r in run))} "
          f"(expect 100% claim_date)")
    print(f"  context_used: {dict(Counter(bool(r.get('context_used')) for r in run))}")
    print(f"  cost so far ${sum(r.get('cost', 0) for r in recs):.3f} "
          f"| ${sum(r.get('cost',0) for r in recs)/max(len(run),1):.5f}/run-claim "
          f"(E1 measured $0.00134)")
    # tranche balance — the shuffle should keep bins/outlets even as it fills
    print(f"  tranche balance by bin: {dict(Counter(r['bin'] for r in run))}")
    ent = Counter(r["coverage"] for r in run)
    print(f"  prominence mix: {dict(ent)}")


def section_funnel(recs, run):
    print("\n" + "=" * 78)
    print("a. FUNNEL, per bin  (every stage keeps its denominator)")
    print("=" * 78)
    print(f"  {'bin':>7} {'drawn':>6} {'gated':>6} {'run':>6} {'>=1doc':>7} "
          f"{'>=1dir':>7} {'posts':>6} {'post>=1dir':>10}")
    for b in BINS:
        d = [r for r in recs if r["bin"] == b]
        rn = [r for r in run if r["bin"] == b]
        doc = [r for r in rn if r["results"]]
        dr = [r for r in rn if any(e["read"]["direction"] in "54321" for e in r["results"])]
        posts = {r["post_id"] for r in rn}
        pdir = {r["post_id"] for r in dr}
        print(f"  {b:>7} {len(d):>6} {len(d)-len(rn):>6} {len(rn):>6} {len(doc):>7} "
              f"{len(dr):>7} {len(posts):>6} {len(pdir):>10}")


def _flagmix(rows):
    c = Counter()
    for r in rows:
        for e in r["results"]:
            c[e["read"]["direction"]] += 1
    return c


def section_flags(run):
    print("\n" + "=" * 78)
    print("b. FLAG MIX per bin — per CLAIM (docs) and per POST")
    print("=" * 78)
    print(f"  {'bin':>7} {'docs':>6} " + " ".join(f"{f:>6}" for f in FLAGS) + f" {'I rate':>8}")
    for b in BINS:
        rows = [r for r in run if r["bin"] == b]
        c = _flagmix(rows)
        tot = sum(c.values())
        print(f"  {b:>7} {tot:>6} " + " ".join(f"{c[f]:>6}" for f in FLAGS)
              + f" {c['I']/max(tot,1):>7.1%}")
    print("\n  PER POST (headline unit) — COMPLETE POSTS ONLY; partial posts are")
    print("  excluded because worst-case aggregation needs all of a post's claims")
    print(f"  {'bin':>7} {'posts':>6} {'>=1 dir':>8} {'rate':>7} {'claims/post':>12}")
    for b in BINS:
        rows = [r for r in run if r["bin"] == b and r.get("post_complete")]
        byp = defaultdict(list)
        for r in rows:
            byp[r["post_id"]].append(r)
        hit = sum(1 for v in byp.values()
                  if any(e["read"]["direction"] in "54321" for r in v for e in r["results"]))
        cpp = len(rows) / max(len(byp), 1)
        print(f"  {b:>7} {len(byp):>6} {hit:>8} {hit/max(len(byp),1):>6.1%} {cpp:>12.2f}")


def section_score(run):
    print("\n" + "=" * 78)
    print("c. SCORE DISTRIBUTION per bin — E1 six-flag weights ("
          + " ".join(f"{k}{W[k]:+.3f}" for k in FLAGS if k in W) + f"), n={E1_FIT['n']:,},")
    print("   CITED from urn_runs/e1_ctx/headline_metrics.json, NOT re-fitted on tweets")
    print("=" * 78)
    print(f"  {'bin':>7} {'n':>5} {'mean':>8} {'median':>8} {'p10':>8} {'p90':>8}")
    for b in BINS:
        s = sorted(r["_score"] for r in run if r["bin"] == b)
        if not s:
            continue
        q = lambda f: s[min(int(len(s) * f), len(s) - 1)]
        print(f"  {b:>7} {len(s):>5} {statistics.mean(s):>8.2f} "
              f"{statistics.median(s):>8.2f} {q(.10):>8.2f} {q(.90):>8.2f}")


def section_quality(run):
    print("\n" + "=" * 78)
    print("d. EVIDENCE QUALITY per bin — reliability of the RETRIEVED documents")
    print("=" * 78)
    print(f"  {'bin':>7} {'docs':>6} {'mean NG':>8} {'%rated':>7} {'%PRIMARY':>9} "
          f"{'%fc_dom':>8} {'%mirror':>8} {'voices/claim':>13}")
    for b in BINS:
        rows = [r for r in run if r["bin"] == b]
        ngs, tot, prim, fc, mir = [], 0, 0, 0, 0
        vpc = []
        for r in rows:
            vs = set()
            for e in r["results"]:
                tot += 1
                if e.get("ng") is not None:
                    ngs.append(e["ng"])
                prim += e.get("rel") == "PRIMARY"
                fc += bool(e.get("fc_domain"))
                mir += bool(e.get("mirror"))
                vs.add(e.get("voice"))
            vpc.append(len(vs))
        if not tot:
            continue
        print(f"  {b:>7} {tot:>6} {statistics.mean(ngs) if ngs else 0:>8.1f} "
              f"{len(ngs)/tot:>6.1%} {prim/tot:>8.1%} {fc/tot:>7.1%} {mir/tot:>7.1%} "
              f"{statistics.mean(vpc) if vpc else 0:>13.2f}")


def section_sweep(run):
    print("\n" + "=" * 78)
    print("e. THRESHOLD SWEEP — nudge RATE (not nudge quality: tweets have no labels)")
    print("   post-level = WORST CASE, any nudging claim nudges the post")
    print("=" * 78)
    cuts = [-8, -6, E1_THRESHOLD, -3, -2, -1, 0]
    for unit in ("claim", "post"):
        print(f"\n  --- per {unit.upper()} ---")
        print(f"  {'cutoff':>9} {'overall':>8} " + " ".join(f"{b:>8}" for b in BINS)
              + f" {'lo/hi':>7}")
        for c in cuts:
            rates, tot_n, tot_h = [], 0, 0
            for b in BINS:
                rows = [r for r in run if r["bin"] == b]
                if unit == "claim":
                    n = len(rows)
                    h = sum(1 for r in rows if r["_score"] <= c)
                else:
                    byp = defaultdict(list)
                    for r in rows:
                        if r.get("post_complete"):
                            byp[r["post_id"]].append(r)
                    n = len(byp)
                    h = sum(1 for v in byp.values() if any(x["_score"] <= c for x in v))
                rates.append(h / n if n else 0.0)
                tot_n += n
                tot_h += h
            ratio = rates[0] / rates[-1] if rates[-1] else float("nan")
            lab = "E1 op" if abs(c - E1_THRESHOLD) < 1e-6 else f"{c:g}"
            print(f"  {lab:>9} {tot_h/max(tot_n,1):>7.1%} "
                  + " ".join(f"{r:>7.1%}" for r in rates) + f" {ratio:>7.2f}")


def section_controls(run):
    print("\n" + "=" * 78)
    print("f. CONTROLS (all pre-committed before the run)")
    print("=" * 78)

    def post_signal(rows):
        rows = [r for r in rows if r.get("post_complete")]
        byp = defaultdict(list)
        for r in rows:
            byp[r["post_id"]].append(r)
        if not byp:
            return None
        return sum(1 for v in byp.values()
                   if any(e["read"]["direction"] in "54321" for r in v
                          for e in r["results"])) / len(byp)

    print("\n  f1. CLAIM-weighted vs OUTLET-weighted post-signal rate (primary = outlet)")
    print(f"  {'bin':>7} {'claim-wtd':>10} {'outlet-wtd':>11} {'n outlets':>10}")
    for b in BINS:
        rows = [r for r in run if r["bin"] == b]
        per = [post_signal([r for r in rows if r["publisher_site"] == d])
               for d in {r["publisher_site"] for r in rows}]
        per = [x for x in per if x is not None]
        cw = post_signal(rows)
        print(f"  {b:>7} {cw if cw is not None else 0:>9.1%} "
              f"{statistics.mean(per) if per else 0:>10.1%} {len(per):>10}")

    print("\n  f2. STRATIFIED by expected coverage (prominence)")
    print(f"  {'coverage':>9} " + " ".join(f"{b:>8}" for b in BINS) + f" {'n':>7}")
    for cov in ("many", "few", "none"):
        rr = [r for r in run if r["coverage"] == cov]
        cells = []
        for b in BINS:
            v = post_signal([r for r in rr if r["bin"] == b])
            cells.append(f"{v:>7.1%}" if v is not None else f"{'--':>7}")
        print(f"  {cov:>9} " + " ".join(cells) + f" {len(rr):>7}")

    print("\n  f3. INCUMBENT outlets only (roster-swap sensitivity)")
    print(f"  {'bin':>7} {'full':>8} {'incumbent':>10} {'new':>8}")
    for b in BINS:
        rows = [r for r in run if r["bin"] == b]
        f = post_signal(rows)
        i = post_signal([r for r in rows if not r["new_outlet"]])
        n = post_signal([r for r in rows if r["new_outlet"]])
        fmt = lambda v: f"{v:>7.1%}" if v is not None else f"{'--':>7}"
        print(f"  {b:>7} {fmt(f)} {fmt(i):>10} {fmt(n):>8}")

    print("\n  f4. PER-OUTLET scatter (min/median/max post-signal within bin)")
    for b in BINS:
        rows = [r for r in run if r["bin"] == b]
        per = sorted(x for x in (post_signal([r for r in rows if r["publisher_site"] == d])
                                 for d in {r["publisher_site"] for r in rows}) if x is not None)
        if per:
            print(f"  {b:>7}  min {per[0]:.1%}  median {statistics.median(per):.1%}  "
                  f"max {per[-1]:.1%}  (n={len(per)} outlets)")


def main():
    _load_e1()
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    recs, run = load()
    print(f"\nE2 tweet urn — {len(recs)} records, {len(run)} run "
          f"({len(run)/3953:.1%} of the 3,953-claim run set)\n")
    section_health(recs, run)
    section_funnel(recs, run)
    section_flags(run)
    section_score(run)
    section_quality(run)
    section_sweep(run)
    section_controls(run)
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"n_records": len(recs), "n_run": len(run),
             "by_bin": {b: sum(1 for r in run if r["bin"] == b) for b in BINS}}, indent=1))


if __name__ == "__main__":
    main()
