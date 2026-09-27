"""Score the mid-band survey candidates and bucket posts into the survey classes,
with the W=7 variant-A SELECTION floor applied on top.

This is `score_midband.py` plus one selection-side rule and two extra output
fields. It writes the SAME schema, so `build_midband_report.py` reads either.

The floor (clog/110926.md 12:48). A contradicting document that was published more
than W=7 days before the post cannot carry refuting weight: a flag 1 or 2 on such a
document is remapped to **X** (variant A -- X is priced +0.287, I is priced -0.194,
and A beat B everywhere on the mid-band). Support flags are untouched, undateable
documents are untouched, and the WEIGHTS ARE NOT REFITTED -- this is a selection
filter over the pinned instrument, NOT a change to the instrument. The same floor
was measured on fc_gold and REJECTED as an instrument change at every W and both
variants; it survives here only because mid-band selection has a different job
(precision of a hand-pick pool) than the gold fit (recall of the detector).

`--no-floor` reproduces `score_midband.py` exactly, which is the check that the
floor is the only difference: on `e4_midband` it must give back the 11:58 lean x
class table (Left 45/32/456/485, Right 58/46/427/561).

    uv run python -m eval.scripts.build_eval.select_midband \
        --run eval/data/urn_runs/e4_midband_fix/results-00.jsonl
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.scripts.build_eval import fit_urn  # noqa: E402
from eval.scripts.build_eval import graded_urn  # noqa: E402

SC = Path("eval/data/survey_claims")
# PINNED read-v5 cuts, same constants as score_midband.py.
PINNED_CUTS = {"cut_1": -4.079, "cut_2": -2.075, "cut_4": 0.034, "cut_5": 5.560}
ORDER = {"1": 0, "2": 1, "middle": 2, "supported": 3}
FLOOR_W = 7
FLAGS = graded_urn.FLAGS
PAD_TO = fit_urn.PAD_TO


def rate(score: float, cuts: dict) -> str:
    if score <= cuts["cut_1"]:
        return "1"
    if score <= cuts["cut_2"]:
        return "2"
    if score > cuts["cut_5"]:
        return "supported"
    return "middle"


def _doc_date(s):
    """Serper's `date` field, e.g. 'Feb 20, 2026'."""
    try:
        return datetime.datetime.strptime(s, "%b %d, %Y").date()
    except (ValueError, TypeError):
        return None


# Serper also returns RELATIVE dates ("1 day ago", "9 hours ago") that `_doc_date`
# and `evidence_urn_run._is_leak` both fail to parse, so `leak_flag` is False and the
# document is read (clog/110926.md 13:52 item 2). 127 mid-band documents carry one and
# every single one resolves PAST the post's ceiling. Behind `--rel-leak`, default OFF.
_REL = re.compile(r"^(\d+)\s+(hour|day|week|month|year)s?\s+ago$", re.I)
_UNIT = {"hour": 0, "day": 1, "week": 7, "month": 30, "year": 365}


def _rel_date(s, crawl: datetime.date):
    m = _REL.match(str(s or "").strip())
    if not m:
        return None
    return crawl - datetime.timedelta(days=_UNIT[m.group(2).lower()] * int(m.group(1)))


def _claim_date(s):
    try:
        return datetime.date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def counts(rec: dict, floor_w: int | None,
           crawl: datetime.date | None = None) -> tuple[dict, int, int]:
    """load_urn's flag counts (pad-to-PAD_TO silences), with the floor remap.
    `crawl` not None enables the relative-date leak drop.
    Returns (flags, n_silenced, n_rel_leaks)."""
    cd = _claim_date(rec.get("claim_date_shown"))
    fl, silenced, rel = collections.Counter(), 0, 0
    for d in rec.get("results") or []:
        direction = (d.get("read") or {}).get("direction")
        if direction not in FLAGS:
            continue
        dd = _doc_date(d.get("date"))
        if dd is None and crawl is not None:
            dd = _rel_date(d.get("date"), crawl)
            if dd is not None and cd is not None and dd > cd:
                rel += 1          # post-ceiling: not read, so it becomes a pad silence
                continue
        if floor_w is not None and direction in ("1", "2") and cd is not None:
            if dd is not None and (cd - dd).days > floor_w:
                direction = "X"
                silenced += 1
        fl[direction] += 1
    fl["I"] += max(0, PAD_TO - sum(fl.values()))
    return dict(fl), silenced, rel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="eval/data/urn_runs/e4_midband_fix/results-00.jsonl")
    ap.add_argument("--candidates", default=str(SC / "candidate_claims_midband.json"))
    ap.add_argument("--refit", default="eval/data/populations/refit_results.json")
    ap.add_argument("--cuts-from", default=None)
    ap.add_argument("--floor-w", type=int, default=FLOOR_W)
    ap.add_argument("--no-floor", action="store_true",
                    help="reproduce score_midband.py (no selection floor)")
    ap.add_argument("--rel-leak", action="store_true",
                    help="also drop documents whose RELATIVE Serper date ('1 day ago') "
                         "resolves past the post's ceiling (default OFF)")
    ap.add_argument("--crawl-date", default="2026-09-11",
                    help="date the relative dates are resolved against (--rel-leak only)")
    ap.add_argument("--out-scored", default=str(SC / "midband_scored.json"))
    ap.add_argument("--out-selection", default=str(SC / "midband_selection.json"))
    a = ap.parse_args()
    floor_w = None if a.no_floor else a.floor_w
    crawl = datetime.date.fromisoformat(a.crawl_date) if a.rel_leak else None

    refit = json.loads(Path(a.refit).read_text())
    cell = next(c for c in refit["cells"] if c["variant"] == "graded_urn_7flag")
    W = cell["weights"]
    cuts = dict(PINNED_CUTS)
    if a.cuts_from:
        rc = json.loads(Path(a.cuts_from).read_text())["rating_cuts"]
        cuts = {k: rc[k]["cut"] for k in cuts}

    cand = {c["claim_id"]: c for c in json.loads(Path(a.candidates).read_text())["claims"]}
    print(f"refit {a.refit}\n  weights " + " ".join(f"{k}{W[k]:+.3f}" for k in FLAGS))
    print("  cuts " + " ".join(f"{k} {v:+.4f}" for k, v in cuts.items()))
    print(f"  floor " + ("OFF" if floor_w is None else f"W={floor_w} variant A (1/2 -> X)"))
    print(f"  relative-date leak drop " + ("OFF" if crawl is None else f"ON (crawl {a.crawl_date})"))

    claims, missing, n_rows, silenced_tot, rel_tot = [], 0, 0, 0, 0
    for line in open(a.run):
        rec = json.loads(line)
        n_rows += 1
        c = cand.get(rec.get("review_url"))
        if c is None:
            missing += 1
            continue
        fl, silenced, rel = counts(rec, floor_w, crawl)
        silenced_tot += silenced
        rel_tot += rel
        s = graded_urn.score_graded({"flags": fl}, W)
        claims.append({
            "claim_id": rec["review_url"], "post_id": c["post_id"], "claim": c["claim"],
            "handle": c["handle"], "lean": c["lean"], "content_lean": c.get("content_lean"),
            "ng_score": c.get("ng_score"), "created_at": c.get("created_at"),
            "url": c.get("url"), "screen_domain": c.get("screen_domain"),
            "score": round(s, 4), "rating": rate(s, cuts),
            "flags": fl, "n_docs": PAD_TO - fl.get("I", 0),
            "floor_silenced": silenced,
        })
    print(f"run {a.run}: {n_rows} claim rows, {len(cand)} candidates")
    if missing:
        print(f"  WARNING {missing} scored claims are not in the candidate file (skipped)")
    in_run = {c["claim_id"] for c in claims}
    not_run = [k for k in cand if k not in in_run]
    if not_run:
        print(f"  WARNING {len(not_run)} candidate claims have no run row")
    print(f"  floor silenced {silenced_tot} flag-1/2 documents "
          f"over {sum(1 for c in claims if c['floor_silenced'])} claims")
    if crawl is not None:
        print(f"  relative-date leak drop removed {rel_tot} documents "
              f"(crawl date {a.crawl_date})")

    by_post = collections.defaultdict(list)
    for c in claims:
        by_post[c["post_id"]].append(c)
    posts = []
    for pid, cs in by_post.items():
        worst = min(cs, key=lambda c: (ORDER[c["rating"]], c["score"]))
        first = cs[0]
        posts.append({
            "post_id": pid, "handle": first["handle"], "lean": first["lean"],
            "ng_score": first["ng_score"], "created_at": first["created_at"],
            "url": first["url"], "post_text": cand[first["claim_id"]].get("post_text"),
            "n_claims": len(cs),
            "rating": worst["rating"], "min_score": worst["score"],
            "driving_claim_id": worst["claim_id"],
            "content_leans": sorted({c["content_lean"] for c in cs if c["content_lean"]}),
            "claim_ids": [c["claim_id"] for c in cs],
            "floor_silenced": sum(c["floor_silenced"] for c in cs),
            "audit": "",
        })
    posts.sort(key=lambda p: p["min_score"])

    Path(a.out_scored).write_text(json.dumps(
        {"run": a.run, "refit": a.refit, "weights": W, "cuts": cuts,
         "floor": None if floor_w is None else {"w_days": floor_w, "variant": "A",
                                                "silenced": silenced_tot},
         "n_claims": len(claims), "n_posts": len(posts),
         "claims": claims, "posts": posts}, indent=1))
    sel = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in posts:
        sel[p["lean"]][p["rating"]].append(p["post_id"])
    Path(a.out_selection).write_text(json.dumps(
        {"cuts": cuts, "weights": W,
         "floor": None if floor_w is None else {"w_days": floor_w, "variant": "A"},
         "by_lean_class": {k: dict(v) for k, v in sel.items()}}, indent=1))
    print(f"wrote {a.out_scored} ({len(claims)} claims / {len(posts)} posts) and {a.out_selection}")

    print("\nposts by lean x class")
    for lean in sorted(sel):
        line = "  " + f"{lean:<6}"
        for cl in ("1", "2", "middle", "supported"):
            line += f"  {cl}={len(sel[lean].get(cl, [])):>5}"
        print(line)
    # same rule as build_midband_report.summarise: the DRIVING claim's content lean
    driving = {c["claim_id"]: c for c in claims}
    cells = collections.Counter()
    for p in posts:
        d = driving.get(p["driving_claim_id"])
        cells[(p["rating"], (d["content_lean"] if d else None) or "neutral")] += 1
    print("\nposts by CONTENT lean x class")
    for cl in ("pro_dem", "pro_rep", "neutral"):
        print(f"  {cl:<8}" + "".join(f"  {r}={cells[(r, cl)]:>5}"
                                     for r in ("1", "2", "middle", "supported"))
              + f"  | 1+2={cells[('1', cl)] + cells[('2', cl)]}")


if __name__ == "__main__":
    main()
