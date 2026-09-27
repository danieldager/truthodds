"""Score the mid-band survey candidates with the v7qa-refitted urn and bucket
posts into the survey classes.

Nothing here re-implements the model. The flag counts and the pad-to-PAD_TO rule
come from `fit_two_urn.load_urn`, the log-odds sum from `graded_urn.score_graded`,
and the weights from the pinned fit (`populations/refit_results.json`, cell
`graded_urn_7flag`) -- read-v5 on DeepSeek-V4-Flash, the production reader
(the v7qa/gpt-oss reader was tried and REVERTED on 2026-09-11, clog 10:55).

The four rating cuts are PINNED CONSTANTS (clog/090926.md 11:40, scale fixed
2026-09-09 13:20). They are not in refit_results.json -- that file carries the
fit cells only -- so they live here, and `--cuts-from` overrides them with the
`rating_cuts` block of any refit file if a later refit ships some.

Classes (Daniel 2026-09-10 16:40):
  1         score <= cut_1            (2% false-alarm cut)
  2         cut_1 < score <= cut_2    (5% false-alarm cut)
  supported score >  cut_5            (2% miss cut)
  middle    everything else           -- discarded
A post's rating is the MIN over its kept key claims (1 < 2 < middle < supported).

    uv run python -m eval.scripts.build_eval.score_midband
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.scripts.build_eval import fit_two_urn as f2  # noqa: E402
from eval.scripts.build_eval import graded_urn  # noqa: E402

SC = Path("eval/data/survey_claims")
# PINNED read-v5 cuts. cut_1 = 2% false alarm, cut_2 = 5% budget,
# cut_4 = 10% miss, cut_5 = 2% miss.
PINNED_CUTS = {"cut_1": -4.079, "cut_2": -2.075, "cut_4": 0.034, "cut_5": 5.560}
ORDER = {"1": 0, "2": 1, "middle": 2, "supported": 3}
RANK = {v: k for k, v in ORDER.items()}


def rate(score: float, cuts: dict) -> str:
    if score <= cuts["cut_1"]:
        return "1"
    if score <= cuts["cut_2"]:
        return "2"
    if score > cuts["cut_5"]:
        return "supported"
    return "middle"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="eval/data/urn_runs/e4_midband/results-00.jsonl")
    ap.add_argument("--candidates", default=str(SC / "candidate_claims_midband.json"))
    ap.add_argument("--refit", default="eval/data/populations/refit_results.json",
                    help="fit file to take the 7-flag weights from")
    ap.add_argument("--cuts-from", default=None,
                    help="refit file whose rating_cuts block replaces PINNED_CUTS")
    ap.add_argument("--out-scored", default=str(SC / "midband_scored.json"))
    ap.add_argument("--out-selection", default=str(SC / "midband_selection.json"))
    a = ap.parse_args()

    refit = json.loads(Path(a.refit).read_text())
    cell = next(c for c in refit["cells"] if c["variant"] == "graded_urn_7flag")
    W = cell["weights"]
    cuts = dict(PINNED_CUTS)
    if a.cuts_from:
        rc = json.loads(Path(a.cuts_from).read_text())["rating_cuts"]
        cuts = {k: rc[k]["cut"] for k in cuts}

    cand = {c["claim_id"]: c for c in json.loads(Path(a.candidates).read_text())["claims"]}
    rows = f2.load_urn([Path(a.run)])
    print(f"refit {a.refit}\n  weights " + " ".join(f"{k}{W[k]:+.3f}" for k in graded_urn.FLAGS))
    print("  cuts " + " ".join(f"{k} {v:+.4f}" for k, v in cuts.items()))
    print(f"run {a.run}: {len(rows)} claim rows, {len(cand)} candidates")

    claims, missing = [], 0
    for r in rows:
        c = cand.get(r["review_url"])
        if c is None:
            missing += 1
            continue
        s = graded_urn.score_graded(r, W)
        claims.append({
            "claim_id": r["review_url"], "post_id": c["post_id"], "claim": c["claim"],
            "handle": c["handle"], "lean": c["lean"], "content_lean": c.get("content_lean"),
            "ng_score": c.get("ng_score"), "created_at": c.get("created_at"),
            "url": c.get("url"), "screen_domain": c.get("screen_domain"),
            "score": round(s, 4), "rating": rate(s, cuts),
            "flags": r["flags"], "n_docs": 10 - r["flags"].get("I", 0),
        })
    if missing:
        print(f"  WARNING {missing} scored claims are not in the candidate file (skipped)")
    in_run = {r["review_url"] for r in rows}
    not_run = [k for k in cand if k not in in_run]
    if not_run:
        print(f"  WARNING {len(not_run)} candidate claims have no run row")

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
        })
    posts.sort(key=lambda p: p["min_score"])

    Path(a.out_scored).write_text(json.dumps(
        {"run": a.run, "refit": a.refit, "weights": W, "cuts": cuts,
         "n_claims": len(claims), "n_posts": len(posts),
         "claims": claims, "posts": posts}, indent=1))
    sel = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in posts:
        sel[p["lean"]][p["rating"]].append(p["post_id"])
    Path(a.out_selection).write_text(json.dumps(
        {"cuts": cuts, "weights": W,
         "by_lean_class": {k: {kk: vv for kk, vv in v.items()} for k, v in sel.items()}},
        indent=1))
    print(f"wrote {a.out_scored} ({len(claims)} claims / {len(posts)} posts) and {a.out_selection}")

    print("\nposts by lean x class")
    for lean in sorted(sel):
        line = "  " + f"{lean:<6}"
        for cl in ("1", "2", "middle", "supported"):
            line += f"  {cl}={len(sel[lean].get(cl, [])):>5}"
        print(line)


if __name__ == "__main__":
    main()
