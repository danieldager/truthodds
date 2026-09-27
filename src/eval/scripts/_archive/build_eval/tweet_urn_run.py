"""E2 runner — evidence profile over NewsGuard-binned tweets (docs/tweet_urn_plan.md).

Reuses the E1 instrument UNMODIFIED: this module imports `run_claim` from
evidence_urn_run and only supplies rows in the shape that function expects. No
prompt, ceiling, search, or READ code is duplicated here — a divergence between
E1 and E2 would make the two measurements incomparable, which is the whole point
of the exercise (clean-instrument replication of the July tweet urn).

What this file IS: a schema adapter + a balanced draw + the run loop.

Adapter (tweet frame -> E1 column contract):
    claim_id      -> review_url        (key; also the resume key)
    claim         -> claim_text
    domain        -> publisher_site    (origin exclusion drops the outlet's own site)
    created_at[:10] -> claim_date      (post date IS the utterance date, Daniel 2026-08-04)
    type          -> claim_type        (assertion|attribution; ONE claim stream, no mode field)
    post_text     -> x_context, context_ok=True
                  -> resolution_status="native"
Gold-only columns stubbed None: veracity, rating_subtype, review_date, x_date, yr.
NewsGuard scores the PUBLISHER, never the claim, so there are no per-claim labels
and `veracity` is structurally absent, not missing. fit_urn cannot consume this
output; the deliverables are distributional (see the plan, §0).

review_date/x_date stay null so ceiling_for() degrades to ceiling_src="claim_date"
and both date guards become no-ops.

The fc-gold gates are bypassed: gate={} (exclusion_for is keyed on review_url) and
the MODEF join is skipped (its assert fires on tweet rows).

  uv run python -m eval.scripts.build_eval.tweet_urn_run --smoke --budget 0.20
  uv run python -m eval.scripts.build_eval.tweet_urn_run --per-bin 286 --budget 7

Output: eval/data/urn_runs/e2_tweets/{smoke,results-00}.jsonl (resumable by claim_id)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pipeline.search import newsguard_score_map
from eval.scripts.build_eval.evidence_urn_run import (
    QUERY_PROMPT_V, READ_PROMPT_V, run_claim)

# The E2 corpus (extracted 2026-08-04, v4.7 prompts byte-verified against the rev
# that built dev500a/b). The two dev draws are DISJOINT from this and are NOT used:
# every yield estimate came from them, so keeping them out keeps the run set
# out-of-sample with respect to its own sizing.
CLAIMS = [Path("eval/data/survey_claims/e2_verify_input_v2.parquet")]
OUTDIR = Path("eval/data/urn_runs/e2_tweets")
# No-context claim screen over the drawn set (claim + date only, as READ sees it).
# TAG for most classes; EXCLUDE for the three that can attract a directional read on the
# WRONG ENTITY (Daniel 2026-08-05, after the runner smoke): "The company faces strikes"
# queried `company strikes 2025` and collected 3 supporting + 1 refuting flag about some
# other company. A lost observation is harmless; a wrong vote contaminates the score
# distribution, which is deliverable (c).
SCREEN_NC = Path("eval/data/survey_claims/e2_draw_screen.parquet")
SCREEN_EXCLUDE = ("unresolved_referent", "fragment", "artifact_unreproduced")

# Pair collapse (Daniel 2026-08-05). Extraction splits reported speech into an
# attribution ("X said Y") and the bare content assertion ("Y"), sharing a `group`.
# Running both double-counts one utterance and gives the post two chances to fire the
# worst-case post nudge. Collapse keeps the ATTRIBUTION and drops the paired assertion:
#   - it carries the speaker, so it stands alone for READ (which never sees the post);
#   - it is faithful to what the post CLAIMS. The bare assertion states as fact what the
#     post reports as an allegation ("Miller held a gun to her head" vs "Emily Moreno said
#     Miller held a gun to her head") — verifying it would measure a proposition the post
#     never asserted.
# Applied PER PAIR, not per group: 34 groups hold more than one attribution, each pairing
# with its own assertion, so "one claim per group" would be wrong.
COLLAPSE_KEEP = "attribution"

# Sampling bands, claim_sourcing/source_selection.md:9-12 (the SAMPLING scheme —
# three other NG threshold schemes exist in the repo for other purposes; do not
# conflate them with the scrape gate or the read-ordering tiers).
BANDS = ["0-30", "30-50", "50-70", "70-90", "90-100"]


def band_of(s: float) -> str:
    s = float(s)
    return ("0-30" if s < 30 else "30-50" if s < 50 else
            "50-70" if s < 70 else "70-90" if s < 90 else "90-100")


def attach_screen(df: pl.DataFrame) -> pl.DataFrame:
    """Tag each claim with the NO-CONTEXT screen verdict, if the screen has run.

    Daniel 2026-08-04: READ not receiving the context is FINE — query-v3 does, so
    retrieval is correctly targeted and the retrieved document supplies the referent.
    The claim is therefore NOT excluded. But `screen_verdict` is a TAG, never a skip
    (evidence_urn_run.py:586-589), so feeding it costs nothing and lets the analysis
    test whether ill-posed-for-READ claims land in "I" more often than the rest —
    which matters because the I rate is a headline deliverable.
    """
    if not SCREEN_NC.exists():
        return df.with_columns(pl.lit("unscreened").alias("screen_verdict"),
                               pl.lit(False).alias("screen_leak"))
    s = (pl.read_parquet(SCREEN_NC)
         .select(pl.col("review_url").alias("claim_id"),
                 pl.col("verdict").alias("screen_verdict"))
         .unique(subset="claim_id"))
    return (df.join(s, on="claim_id", how="left")
            .with_columns(pl.col("screen_verdict").fill_null("unscreened"),
                          pl.lit(False).alias("screen_leak")))


def load_claims() -> pl.DataFrame:
    """The two dev draws, concatenated. dev500a carries an extra `_fold` column."""
    fs = [pl.read_parquet(p) for p in CLAIMS]
    common = sorted(set.intersection(*[set(f.columns) for f in fs]))
    df = pl.concat([f.select(common) for f in fs], how="vertical_relaxed")
    if "lean" not in df.columns:   # harvest parquet carries no roster columns
        roster = pl.read_csv("eval/data/survey_claims/source_roster.csv")
        keep = [c for c in ("domain", "lean", "register") if c in roster.columns]
        df = df.join(roster.select(keep).unique(subset="domain"), on="domain", how="left")
    assert df["claim_id"].n_unique() == df.height, "claim_id collision across draws"
    assert df["ng_score"].null_count() == 0
    df = df.with_columns(
        pl.col("ng_score").map_elements(band_of, return_dtype=pl.String).alias("bin"))
    return attach_screen(df)


def adapt(r: dict) -> dict:
    """Tweet claim row -> the 10 columns run_claim hard-indexes, + the .get() ones.

    Extras (post_id, ng_score, bin, ...) ride along: run_claim ignores unknown keys
    and builds its own record, so per-bin analysis re-joins on claim_id afterwards.
    """
    return {**r,
            "review_url": r["claim_id"],
            "claim_text": r["claim"],
            "publisher_site": r["domain"],
            "claim_date": (r["created_at"] or "")[:10],
            "claim_type": r["type"],
            "x_context": r["post_text"],
            "context_ok": True,
            "resolution_status": "native",
            "claim_resolved": None,
            # gold-only, structurally absent on tweets (see docstring)
            "veracity": None, "rating_subtype": None,
            "review_date": None, "x_date": None, "yr": None}


def balanced_draw(df: pl.DataFrame, per_bin: int, seed: int, outlet_cap_frac: float):
    """Equal POSTS per bin, capped per outlet, skipping zero-claim posts.

    Every post here already has >=1 verify-eligible claim by construction (the
    frame IS extracted claims), so the "zero-claim skip" of the plan shows up as
    posts PRESENT in the extraction manifest but ABSENT from the claim frame. That
    accounting is done by tweet_urn_report.py off the posts manifests, not here —
    this function only reports what it could not fill.

    Returns (rows, draw_log).
    """
    rng = random.Random(seed)
    log = {}
    picked_posts = []
    for b in BANDS:
        sub = df.filter(pl.col("bin") == b)
        by_outlet = defaultdict(list)
        for pid, dom in zip(sub["post_id"].to_list(), sub["domain"].to_list()):
            by_outlet[dom].append(pid)
        for dom in by_outlet:
            by_outlet[dom] = sorted(set(by_outlet[dom]))
            rng.shuffle(by_outlet[dom])
        n_out = len(by_outlet)
        # Cap so no single account can own a bin. outlet_cap_frac of the quota.
        cap = max(1, int(round(per_bin * outlet_cap_frac)))
        chosen, i = [], 0
        # round-robin across outlets: fills evenly, hits the cap only if a bin is
        # thin on outlets
        while len(chosen) < per_bin and i < cap:
            progressed = False
            for dom in sorted(by_outlet):
                if len(chosen) >= per_bin:
                    break
                if i < len(by_outlet[dom]):
                    chosen.append((by_outlet[dom][i], dom))
                    progressed = True
            if not progressed:
                break
            i += 1
        log[b] = {"requested": per_bin, "drawn": len(chosen), "outlets_available": n_out,
                  "outlet_cap": cap, "pool_posts": sub["post_id"].n_unique(),
                  "short_by": max(0, per_bin - len(chosen))}
        picked_posts += [p for p, _ in chosen]
    rows = df.filter(pl.col("post_id").is_in(picked_posts))
    for b in BANDS:
        log[b]["claims"] = rows.filter(pl.col("bin") == b).height
    return rows, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="25 claims, 5 per NG bin — pipeline shakeout")
    ap.add_argument("--per-bin", type=int, default=0, help="posts per NG bin (balanced draw)")
    ap.add_argument("--budget", type=float, default=5.0, help="hard USD cap")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=707)
    ap.add_argument("--outlet-cap-frac", type=float, default=0.25,
                    help="max share of a bin's post quota one outlet may supply")
    ap.add_argument("--all-claims", action="store_true",
                    help="run claims gated out by topic too (default: verify_eligible only)")
    args = ap.parse_args()

    df = load_claims()
    n_all = df.height
    if not args.all_claims:
        df = df.filter(pl.col("verify_eligible"))
    print(f"claims: {n_all} extracted -> {df.height} run-eligible "
          f"({df['post_id'].n_unique()} posts, {df['domain'].n_unique()} outlets)", flush=True)

    if args.smoke:
        parts = []
        for b in BANDS:
            sub = df.filter(pl.col("bin") == b).sort("claim_id")
            parts.append(sub.sample(min(5, sub.height), seed=args.seed))
        rows_df = pl.concat(parts)
        draw_log = {"mode": "smoke", "per_bin_claims": 5}
        out = OUTDIR / "smoke.jsonl"
    else:
        assert args.per_bin > 0, "--per-bin required for a real draw"
        rows_df, draw_log = balanced_draw(df, args.per_bin, args.seed, args.outlet_cap_frac)
        draw_log["mode"] = "balanced"
        out = OUTDIR / "results-00.jsonl"

    rows = [adapt(r) for r in rows_df.sort("claim_id").iter_rows(named=True)]
    # Shuffle POSTS, not claims, and emit each post's claims consecutively.
    # Why: the headline unit is the POST under worst-case aggregation (any nudging
    # claim nudges the post), which cannot be evaluated until ALL of a post's claims
    # are back. With a claim-level shuffle, a 10% prefix contains ~10% of claims but
    # almost no COMPLETE posts, so the headline is unreadable until the run nearly
    # finishes. Shuffling at post level keeps every prefix a stratified random sample
    # (of posts, which is the sampling unit anyway) AND makes it post-complete.
    # Caught by the 10% dress rehearsal, 2026-08-05.
    by_post = {}
    for r in rows:
        by_post.setdefault(r["post_id"], []).append(r)
    order = sorted(by_post)
    random.Random(args.seed).shuffle(order)
    rows = [r for pid in order for r in by_post[pid]]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():
        for l in open(out):
            try:
                seen.add(json.loads(l)["review_url"])
            except Exception:
                pass
    sample_path = out.with_suffix(".sample.json")
    prev_n = json.loads(sample_path.read_text()).get("n", 0) if sample_path.exists() else 0
    if len(rows) >= prev_n:   # rewrite only when the drawn set grows (staged runs)
        sample_path.write_text(json.dumps(
            {"seed": args.seed, "n": len(rows), "source": [str(p) for p in CLAIMS],
             "draw": draw_log, "claim_ids": [r["review_url"] for r in rows]}, indent=1))

    todo = [r for r in rows if r["review_url"] not in seen]
    print(f"{len(todo)} claims to run -> {out.name} | budget ${args.budget} | "
          f"{args.workers} workers | prompts {QUERY_PROMPT_V}/{READ_PROMPT_V}", flush=True)
    print(f"draw: {json.dumps(draw_log)}", flush=True)

    ng_scores = newsguard_score_map()
    lock = threading.Lock()
    budget = {"spent": 0.0, "done": 0, "cached_tok": 0, "prompt_tok": 0, "stop": False}
    t0 = time.time()
    fh = open(out, "a")
    # The screen's wrong-entity classes are fed through the runner's OWN gate, so each
    # excluded claim is RECORDED with a reason and empty results (repo rule: excluded
    # rows are never silenced) and costs no LLM or Serper call.
    gate = {r["review_url"]: {"decision": "exclude", "kind": r["screen_verdict"]}
            for r in rows if r.get("screen_verdict") in SCREEN_EXCLUDE}
    # collapsed pairs are RECORDED as exclusions too, so the funnel shows them
    paired = {}
    for r in rows:
        if r.get("group") is not None:
            paired.setdefault((r["post_id"], r["group"]), set()).add(r["claim_type"])
    n_col = 0
    for r in rows:
        k = (r["post_id"], r.get("group"))
        if (r.get("group") is not None and len(paired.get(k, ())) > 1
                and r["claim_type"] != COLLAPSE_KEEP
                and r["review_url"] not in gate):
            gate[r["review_url"]] = {"decision": "exclude", "kind": "pair-collapsed"}
            n_col += 1
    print(f"pair-collapsed (recorded, not run): {n_col}", flush=True)
    print(f"screen-excluded (recorded, not run): {len(gate)} of {len(rows)} "
          f"({len(gate)/max(len(rows),1):.2%})", flush=True)

    def work(row):
        if budget["stop"]:
            return
        try:
            rec = run_claim(row, ng_scores, budget, gate)
        except Exception as e:
            with lock:
                budget["done"] += 1
                print(f"  [claim-failed] {row['review_url']}: {type(e).__name__}: {e}", flush=True)
            return
        # carry the bin/post keys the analysis needs; run_claim builds its record
        # from the E1 contract and cannot know about them.
        rec["post_id"] = row["post_id"]
        rec["ng_score"] = row["ng_score"]
        rec["bin"] = row["bin"]
        rec["lean"] = row.get("lean")
        rec["register"] = row.get("register")
        rec["checkworthy"] = row["checkworthy"]
        rec["verify_eligible"] = row["verify_eligible"]
        with lock:
            budget["spent"] += rec["cost"]
            budget["done"] += 1
            fh.write(json.dumps(rec) + "\n")
            n = budget["done"]
            if n % 25 == 0 or n == len(todo):
                fh.flush()
                proj = budget["spent"] / n * len(todo)
                cr = budget["cached_tok"] / max(budget["prompt_tok"], 1)
                print(f"  {n}/{len(todo)} | spent ${budget['spent']:.3f} | "
                      f"projected ${proj:.2f} | cache-hit {cr:.0%} | "
                      f"{(time.time()-t0)/60:.0f}m | {n/max((time.time()-t0)/60,.01):.1f} claims/min",
                      flush=True)
                if proj > args.budget and n >= max(25, len(todo) // 4):
                    print(f"  BUDGET ABORT: projection ${proj:.2f} > cap ${args.budget}", flush=True)
                    budget["stop"] = True

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"\ndone {budget['done']} claims | ${budget['spent']:.4f} | "
          f"cache-hit {budget['cached_tok']/max(budget['prompt_tok'],1):.0%} | "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
