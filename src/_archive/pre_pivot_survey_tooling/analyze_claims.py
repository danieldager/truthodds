"""Analyse the extracted claims against the original tweets.

Joins the check-worthiness CSV (scripts.checkworthy_count output) with the
original Zeeschuimer NDJSON by rest_id, then reports: claim corpus stats,
quantitative-claim share, engagement check-worthy-vs-not, Community Notes
overlap, and top sources. Read-only, no API calls.

    uv run python -m scripts.analyze_claims --csv <results.csv> --ndjson <export.ndjson>
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from collections import Counter


def med(xs: list[float]) -> float:
    return st.median(xs) if xs else 0.0


def load_ndjson(path: str) -> dict[str, dict]:
    by_id = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = d.get("rest_id") or (d.get("legacy") or {}).get("id_str")
            if rid:
                by_id[str(rid)] = d
    return by_id


def engagement(d: dict) -> dict:
    lg = d.get("legacy") or {}
    views = (d.get("views") or {}).get("count")
    return {
        "views": int(views) if views and str(views).isdigit() else 0,
        "likes": lg.get("favorite_count", 0),
        "retweets": lg.get("retweet_count", 0),
        "replies": lg.get("reply_count", 0),
        "quotes": lg.get("quote_count", 0),
        "bookmarks": lg.get("bookmark_count", 0),
    }


def author(d: dict) -> tuple[str, bool]:
    res = ((d.get("core") or {}).get("user_results") or {}).get("result") or {}
    lg = res.get("legacy") or {}
    core = res.get("core") or {}
    sn = lg.get("screen_name") or core.get("screen_name") or "?"
    verified = bool(res.get("is_blue_verified") or lg.get("verified"))
    return sn, verified


NUM_RE = re.compile(r"\b\d|\d%|\b(19|20)\d{2}\b|%|€|\$|£")
SAID_RE = re.compile(r"\b(said|stated|claimed|announced|declared|a déclaré|a dit|a affirmé|selon)\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ndjson", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    by_id = load_ndjson(args.ndjson)

    cw = [r for r in rows if r["is_checkable"] == "True"]
    not_cw = [r for r in rows if r["is_checkable"] == "False"]

    # explode claims
    claims = []
    for r in cw:
        for c in (r["claims"].split(" | ") if r["claims"] else []):
            if c.strip():
                claims.append((r["lang"], c.strip()))

    print("=" * 60)
    print("CLAIM CORPUS")
    print("=" * 60)
    counts = [int(r["n_claims"]) for r in cw]
    print(f"check-worthy posts:   {len(cw)}")
    print(f"total claims:         {len(claims)}")
    print(f"claims / cw-post:     mean {st.mean(counts):.1f}  median {med(counts):.0f}  max {max(counts)}")
    wlens = [len(c.split()) for _, c in claims]
    print(f"claim length (words): mean {st.mean(wlens):.1f}  median {med(wlens):.0f}")

    quant = sum(1 for _, c in claims if NUM_RE.search(c))
    attrib = sum(1 for _, c in claims if SAID_RE.search(c))
    print(f"quantitative claims (contain number/%/date/currency): {quant} ({100*quant/len(claims):.0f}%)")
    print(f"attribution claims (X said/claimed Y):                {attrib} ({100*attrib/len(claims):.0f}%)")

    print("\n" + "=" * 60)
    print("ENGAGEMENT: check-worthy vs not  (median per post)")
    print("=" * 60)

    def eng_rows(rs):
        es = [engagement(by_id[r["id"]]) for r in rs if r["id"] in by_id]
        keys = ["views", "likes", "retweets", "replies", "quotes", "bookmarks"]
        return {k: med([e[k] for e in es]) for k in keys}

    ec, en = eng_rows(cw), eng_rows(not_cw)
    print(f"{'metric':<12}{'check-worthy':>14}{'not':>10}")
    for k in ["views", "likes", "retweets", "replies", "quotes", "bookmarks"]:
        print(f"{k:<12}{ec[k]:>14,.0f}{en[k]:>10,.0f}")

    print("\n" + "=" * 60)
    print("COMMUNITY NOTES (Birdwatch) overlap")
    print("=" * 60)
    # birdwatch_pivot = an actual attached note; has_birdwatch_notes is a present-but-false flag
    bw = [r for r in rows if r["id"] in by_id and by_id[r["id"]].get("birdwatch_pivot")]
    bw_cw = [r for r in bw if r["is_checkable"] == "True"]
    print(f"posts with Community Notes flag: {len(bw)}")
    print(f"  of which check-worthy:         {len(bw_cw)} ({100*len(bw_cw)/max(len(bw),1):.0f}%)")
    print("  (vs 43% base rate — notes should land on claim-bearing posts)")

    print("\n" + "=" * 60)
    print("TOP CHECK-WORTHY POSTS BY VIEWS")
    print("=" * 60)
    ranked = sorted(
        [r for r in cw if r["id"] in by_id],
        key=lambda r: engagement(by_id[r["id"]])["views"], reverse=True)[:5]
    for r in ranked:
        d = by_id[r["id"]]
        sn, ver = author(d)
        v = engagement(d)["views"]
        print(f"  @{sn}{' ✓' if ver else ''}  {v:,} views")
        print(f"    claim: {r['claims'][:150]}")

    print("\n" + "=" * 60)
    print("TOP AUTHORS AMONG CHECK-WORTHY POSTS")
    print("=" * 60)
    auth = Counter()
    verified_cw = 0
    for r in cw:
        if r["id"] in by_id:
            sn, ver = author(by_id[r["id"]])
            auth[sn] += 1
            verified_cw += ver
    for sn, n in auth.most_common(8):
        print(f"  @{sn}: {n}")
    print(f"verified-author share of check-worthy posts: {100*verified_cw/max(len(cw),1):.0f}%")


if __name__ == "__main__":
    main()
