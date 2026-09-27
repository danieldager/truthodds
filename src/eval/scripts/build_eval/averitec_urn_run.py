"""LEG 2 — the log-odds urn instrument on AVeriTeC dev (external-benchmark leg).

Runs the SAME instrument as the fc-gold E1 urn and the tweet urns
(`evidence_urn_run.run_claim`: query-v3 / read-v5, Serper top-10 under a date
ceiling, one read per document) over the 491 unique AVeriTeC dev claims, so the
frozen ladder weights (`model_ladder/ladder.json`) can be applied to a population
nobody here labelled. Scoring is `score_frozen_urn.py`; the label comparison is
pre-registered separately and is NOT computed here.

Adapter decisions (mirrors timeline_urn_run.adapt, see that docstring for why the
context switch is load-bearing):
  * review_url = claim_id = sha1(f"{claim}||{original_url or ''}")[:16], the id
    used by claims_dev_500_gold.parquet. Where the gold parquet's id differs
    (whitespace variants in the claim text) the GOLD id wins, so joins hold.
  * ceiling = the AVeriTeC claim date, inclusive (Serper tbs cd_max on that day).
    review_date is None ON PURPOSE: ceiling_for's clamp is review_date-1 and it
    re-labels the source "+clamped", so any review_date at all changes either the
    day or the provenance string. With no review_date the chain yields
    (claim_date, "claim_date") exactly.
  * exclusion = the origin site only, when there is one. web.archive.org wrappers
    are unwrapped to the archived page's host (the origin is the archived site,
    not the archive); other archives (archive.ph, perma.cc) cannot be unwrapped
    and are excluded as themselves. No original_url -> nothing excluded (AVeriTeC
    claims are not tweets; x.com is not an origin here).
  * veracity: Supported 5, Refuted 1, NEE and Conflicting 3 (label kept verbatim
    in averitec_label; rating_subtype "nee" / "conflicting" tells them apart).
  * claim_mode: claim_type="assertion" -> mode_of() gives "assertion" for every
    row; AVeriTeC "Quote Verification" claims therefore read as assertions
    (claim_types is carried on the record so this can be revisited at fit time).
  * x_context=None, context_ok=False (context OFF, as on both tweet urns);
    screen fields neutral (unscreened / no leak); exclusion gate empty.

  uv run python -m eval.scripts.build_eval.averitec_urn_run --dry-run
  uv run python -m eval.scripts.build_eval.averitec_urn_run --smoke 25
  uv run python -m eval.scripts.build_eval.averitec_urn_run --budget 2.0

Output: eval/data/urn_runs/averitec_dev/{scores,smoke}.jsonl + manifest.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from config import VERIFICATION_MODEL  # noqa: E402
from pipeline.search import newsguard_score_map  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
    run_claim, ceiling_for, QUERY_PROMPT_V, READ_PROMPT_V, CLEAN_V, PREP_V)
from claimverify.origin import origin_domain  # noqa: E402

AVERITEC = SRC / "eval/data/averitec/averitec_full.parquet"
GOLD = SRC / "eval/scripts/verification_grading/data/claims_dev_500_gold.parquet"
OUT = SRC / "eval/data/urn_runs/averitec_dev"
SEED = 707
VERACITY = {"Supported": 5, "Refuted": 1, "Not Enough Evidence": 3,
            "Conflicting Evidence/Cherrypicking": 3}
SUBTYPE = {"Not Enough Evidence": "nee", "Conflicting Evidence/Cherrypicking": "conflicting"}
CEILING_POLICY = "claim_date inclusive (D-M-YYYY -> ISO), review_date=None so no clamp"
EXCLUSION_POLICY = ("origin host of original_url only, www. stripped, web.archive.org "
                    "unwrapped to the archived host; none when original_url is null")


def claim_id(claim: str, original_url: str | None) -> str:
    return hashlib.sha1(f"{claim}||{original_url or ''}".encode()).hexdigest()[:16]


def _iso(d: str) -> str:
    """AVeriTeC dates are D-M-YYYY (run_averitec_benchmark._iso)."""
    day, month, year = str(d).split("-")
    return f"{year}-{int(month):02d}-{int(day):02d}"


def adapt(r: dict, cid: str) -> dict:
    """AVeriTeC dev row -> the row shape run_claim needs (see module docstring)."""
    return {
        "review_url": cid, "claim_text": r["claim"], "claim_resolved": r["claim"],
        "publisher_site": origin_domain(r["original_url"]),
        "claim_date": _iso(r["claim_date"]), "review_date": None, "x_date": None,
        "veracity": VERACITY[r["label"]], "rating_subtype": SUBTYPE.get(r["label"]),
        "averitec_label": r["label"], "claim_types": r["claim_types"],
        "speaker": r["speaker"], "reporting_source": r["reporting_source"],
        "fc_url": r["fc_url"], "original_url": r["original_url"],
        "claim_type": "assertion", "topic": None,
        "x_context": None, "context_ok": False,
        "resolution_status": "ok", "screen_verdict": "unscreened", "screen_leak": False,
        "yr": 2020}


CARRY = ("averitec_label", "claim_types", "speaker", "reporting_source", "fc_url",
         "original_url")


def build_rows() -> list[dict]:
    """491 unique dev claims (first row per claim text kept) with gold-matching ids.
    Writes dedup.json (dropped rows) and id_overrides.json next to the outputs."""
    df = pl.read_parquet(AVERITEC).filter(pl.col("split") == "dev")
    gold = pl.read_parquet(GOLD)
    gold_by_norm = {" ".join(t.split()): i for t, i in zip(gold["claim_text"], gold["claim_id"])}
    gold_ids = set(gold["claim_id"])
    seen, rows, dropped, overrides = {}, [], [], []
    for i, r in enumerate(df.iter_rows(named=True)):
        if r["claim"] in seen:
            dropped.append({"row": i, "kept_row": seen[r["claim"]], "claim": r["claim"],
                            "label": r["label"], "original_url": r["original_url"]})
            continue
        seen[r["claim"]] = i
        cid = claim_id(r["claim"], r["original_url"])
        if cid not in gold_ids:
            g = gold_by_norm.get(" ".join(r["claim"].split()))
            overrides.append({"row": i, "hash_id": cid, "gold_id": g, "claim": r["claim"]})
            if g:
                cid = g
        rows.append(adapt(r, cid))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dedup.json").write_text(json.dumps(
        {"n_dev": df.height, "n_unique": len(rows), "dropped": dropped}, indent=1))
    (OUT / "id_overrides.json").write_text(json.dumps(overrides, indent=1))
    unmatched = [o for o in overrides if not o["gold_id"]]
    print(f"dev {df.height} -> {len(rows)} unique claims ({len(dropped)} duplicate rows "
          f"dropped) | gold-id matches {len(rows) - len(unmatched)}/{len(rows)} "
          f"({len(overrides)} hash/gold divergences, {len(unmatched)} unmatched)", flush=True)
    return rows


def manifest(rows: list[dict], workers: int, out: Path, smoke: int | None) -> None:
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True, cwd=SRC).stdout.strip()
    (OUT / ("manifest_open.json" if out.name == "scores_open.jsonl" else "manifest.json")).write_text(json.dumps({
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model": VERIFICATION_MODEL,
        "prompts": {"query": QUERY_PROMPT_V, "read": READ_PROMPT_V, "clean": CLEAN_V, "prep": PREP_V},
        "workers": workers,
        "ceiling_policy": "OFF (ClaimCheck-matched)" if out.name == "scores_open.jsonl" else CEILING_POLICY,
        "exclusion_policy": "none (ClaimCheck-matched)" if out.name == "scores_open.jsonl" else EXCLUSION_POLICY,
        "input": str(AVERITEC.relative_to(SRC)),
        "input_sha256": hashlib.sha256(AVERITEC.read_bytes()).hexdigest(),
        "n_claims": len(rows), "smoke": smoke, "seed": SEED, "output": out.name,
    }, indent=1))


def dry_run(rows: list[dict]) -> None:
    for r in rows[:5]:
        ceil, src = ceiling_for(r)
        xd = [d for d in [r["publisher_site"]] if d]
        print(f"  {r['review_url']} | {r['claim_text'][:80]!r}\n"
              f"      ceiling {ceil} ({src}) | exclude {xd} | veracity {r['veracity']} "
              f"| {r['averitec_label']} | origin {r['original_url']}")
    print("labels: " + json.dumps(Counter(r["averitec_label"] for r in rows)))
    print("veracity: " + json.dumps(Counter(r["veracity"] for r in rows)))
    print("exclusion hosts: " + json.dumps(
        Counter(r["publisher_site"] for r in rows).most_common(8)))
    srcs = Counter(ceiling_for(r)[1] for r in rows)
    assert srcs == {"claim_date": len(rows)}, srcs
    print(f"ceiling src: {dict(srcs)} | dates {min(r['claim_date'] for r in rows)} .. "
          f"{max(r['claim_date'] for r in rows)}")


def reads(rows: list[dict], smoke: int | None, workers: int, cap: float,
          open_: bool = False) -> None:
    if open_:   # ClaimCheck-matched: no date ceiling, origin not excluded (2026-09-08)
        import eval.scripts.build_eval.evidence_urn_run as _eur
        _eur.ceiling_for = lambda row: ("", "off")
        rows = [{**r, "publisher_site": None} for r in rows]
    rows = sorted(rows, key=lambda r: r["review_url"])   # stable order, then seeded shuffle
    random.Random(SEED).shuffle(rows)
    if smoke:
        rows = rows[:smoke]
        out = OUT / "smoke.jsonl"
        out.with_suffix(".sample.json").write_text(json.dumps(
            {"seed": SEED, "n": len(rows), "claim_ids": [r["review_url"] for r in rows]}, indent=1))
    else:
        out = OUT / ("scores_open.jsonl" if open_ else "scores.jsonl")
    manifest(rows, workers, out, smoke)
    seen = {json.loads(l)["review_url"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["review_url"] not in seen]
    print(f"reads: {len(todo)}/{len(rows)} claims -> {out.name} | cap ${cap} | "
          f"{workers} workers | {VERIFICATION_MODEL} | prompts {QUERY_PROMPT_V}/{READ_PROMPT_V}",
          flush=True)

    ng = newsguard_score_map()
    lock = threading.Lock()
    budget = {"spent": 0.0, "done": 0, "failed": 0, "cached_tok": 0, "prompt_tok": 0,
              "searches": 0, "stop": False}
    t0 = time.time()
    fh = open(out, "a")

    def work(row):
        if budget["stop"]:
            return
        try:
            rec = run_claim(row, ng, budget, {})
        except Exception as e:
            with lock:
                budget["done"] += 1
                budget["failed"] += 1
                print(f"  [claim-failed] {row['review_url']}: {type(e).__name__}: {e}", flush=True)
            return
        for k in CARRY:
            rec[k] = row[k]
        with lock:
            budget["spent"] += rec["cost"]
            budget["searches"] += rec.get("search_calls", 0)
            budget["done"] += 1
            fh.write(json.dumps(rec) + "\n")
            n, el = budget["done"], (time.time() - t0) / 60
            if n % 25 == 0 or n == len(todo):
                fh.flush()
                proj = budget["spent"] / n * len(todo)
                cr = budget["cached_tok"] / max(budget["prompt_tok"], 1)
                print(f"  {n}/{len(todo)} | ${budget['spent']:.3f} | proj ${proj:.2f} | "
                      f"cache-hit {cr:.0%} | serper {budget['searches']} | "
                      f"failed {budget['failed']} | {el:.1f}m | {n/max(el,.01):.1f}/min | "
                      f"ETA {el/n*(len(todo)-n):.0f}m", flush=True)
                if proj > cap and n >= 25:
                    print(f"  BUDGET ABORT: projection ${proj:.2f} > cap ${cap}", flush=True)
                    budget["stop"] = True

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"reads done {budget['done']} ({budget['failed']} failed) | ${budget['spent']:.4f} | "
          f"serper {budget['searches']} | {(time.time()-t0)/60:.1f}m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="adapt + print, no network")
    ap.add_argument("--smoke", type=int, default=0, help="first N after the seeded shuffle")
    ap.add_argument("--budget", type=float, default=2.0, help="USD hard cap (LLM meter)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--open", action="store_true",
                    help="ClaimCheck-matched: no date ceiling, origin not excluded -> scores_open.jsonl")
    a = ap.parse_args()
    rows = build_rows()
    if a.dry_run:
        dry_run(rows)
        return
    reads(rows, a.smoke or None, a.workers, 0.25 if a.smoke else a.budget, open_=a.open)


if __name__ == "__main__":
    main()
