"""Survey funnel steps 1-2: hard screens over the frozen E2 claim pool.

Design and decided forks: docs/survey_claim_selection.md.

    uv run python -m eval.scripts.claim_sourcing.survey_screen_claims --smoke 40
    uv run python -m eval.scripts.claim_sourcing.survey_screen_claims --budget 1

Pool = the 3,953 claims E2 actually ran (`excluded` null in results-00.jsonl).
Claims E2 already dropped (fragment / unresolved referent / artifact / pair-collapsed)
never enter; that is the "reuse the draw screen's exclusions" step and it is free.

Screens, cheapest first, each recorded as its own funnel row rather than folded in:

  D1  no urn score       20 claims returned zero documents, so no band can be
                         assigned. A stimulus whose deployed routing is unknown is
                         useless to a pipeline-dependent design.
  D2  non-English        the survey is English-only.
  D3  near-duplicate     idf-weighted char-trigram Jaccard, blocked on rare tokens,
                         union-find. Threshold 0.58 carried over from cn_cluster.py
                         after inspecting pairs either side of it: at 0.5 and below
                         it starts merging genuinely different claims (two different
                         court cases, a claim and its negation).
  L1  time-stability     LLM. Truth must not drift between selection and field date.
  L2  self-containedness LLM. Judgeable from the POST as the respondent will see it
                         -- text AND attached media (Daniel 2026-08-19) -- so the bar
                         is thread/external-link dependence, not media locus.
  L3  specificity        LLM. Added after the first smoke: the extraction-time
                         checkworthiness gate admits vague commentary ("experts are
                         already warning that X has steamrolled Y"), which a
                         professional fact checker cannot rule on. Free -- it rides
                         in the same call.

L1, L2 and L3 share one call per claim: they are independent judgements of the same
text and splitting them would triple the cost for nothing. L1's boolean is DERIVED
from a forced temporal-type enum rather than asked for directly, because the first
smoke returned rows reading time_stable=true beside the reason "ongoing event".

Plausibility (a stratification score, NOT a screen) runs as a second pass over
survivors only, in `--plausibility` mode, so a screened-out claim is never paid for
twice and the plausibility judge never sees a claim the survey cannot use.

Writes eval/data/survey_claims/e2_survey_screened.parquet (every pool claim, with
its screen verdicts and strata columns -- nothing is deleted, `kept` is a column)
plus a printed funnel with a denominator on every row.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import log
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.scripts.build_eval.cn_cluster import DSU, _STOP, norm, trigrams
from eval.scripts.build_eval.evidence_urn_run import llm
from eval.scripts.build_eval.fit_urn import FLAG_TO_VOICE

DATA = Path("eval/data")
E2 = DATA / "urn_runs/e2_tweets/results-00.jsonl"
VERIFY_IN = DATA / "survey_claims/e2_verify_input_v2.parquet"
PROMINENCE = DATA / "survey_claims/e2_claim_prominence.parquet"
OUT = DATA / "survey_claims/e2_survey_screened.parquet"

# E1 out-of-fold fit (eval/data/urn_runs/e1_ctx/headline_metrics.json, overall).
# Cited, never re-fitted here. SIX-FLAG since 2026-09-14 (read-v6.1: 5/4/X/I/2/1,
# no "3" class -> a read-v5 "3" folds into "X"): the weights are keyed by read flag
# and the claim is padded to PAD_TO silent slots, the conventions the fit was
# estimated under.
_E1M = json.loads((DATA / "urn_runs/e1_ctx/headline_metrics.json").read_text())["overall"]
W = _E1M["weights"]
PAD_TO = 10
T_LOW = _E1M["threshold"]      # deployed FLAG boundary (six-flag v6.1 rep1500 = -3.584)
# OPEN (2026-09-14 PM): the boundaries mix scales and this script is NOT run this
# session (it makes LLM calls). T_LOW now comes from the SIX-FLAG read-v6.1 fit on
# the representative balanced fc_gold_rep1500 (-3.584; superseded the clean-slice
# -3.655). T_HIGH is still the THREE-VOICE PASS boundary agreed 2026-08-19 (2% oof
# miss budget). The rep1500 six-flag fit's own PASS cut (<=2% gold-FALSE above) is
# +4.254 (<=5% is +1.679). Needs Daniel's ruling before this script runs again; the
# pool it already wrote is 3-voice throughout.
T_HIGH = 10.396

SIM_EDGE = 0.58
RARE_DF_MAX = 400
BLOCK_CAP = 60

# Abstract criteria only -- no worked examples and no dataset-derived phrasing, so
# the judge cannot pattern-match the corpus it is screening.
SCREEN_SYS = (
    "You screen claims for use as stimuli in a survey experiment. A respondent will "
    "be shown the SOURCE POST exactly as published -- its text, its date, and any "
    "image or video attached to it -- and then the CLAIM drawn from it.\n\n"
    "Return JSON only:\n"
    '{"temporal_type": "settled|ongoing|pending|recurring_measure|relative_reference", '
    '"time_reason": "<8 words>", "self_contained": true|false, '
    '"contain_reason": "<8 words>", "specific": true|false, '
    '"specific_reason": "<8 words>"}\n\n'
    "temporal_type -- classify what kind of thing the claim asserts about time. "
    "Choose the FIRST that applies, reading top to bottom:\n"
    "- relative_reference: the claim's time anchor is the moment of writing rather "
    "than a fixed date -- currently, now, latest, recent, so far, to date, this "
    "week, still, already, no longer.\n"
    "- pending: the outcome is not yet determined -- scheduled, planned, forecast, "
    "expected, under consideration, will happen, is being decided.\n"
    "- ongoing: the situation is live and developing, so the assertion can stop "
    "being true -- a hearing in progress, an investigation open, a person holding "
    "an office, a policy in force, a dispute unresolved.\n"
    "- recurring_measure: the claim states a quantity that is re-measured and moves "
    "-- counts, totals, prices, rates, polling positions, rankings, records, death "
    "tolls, arrest numbers, anything that can be superseded by a later reading.\n"
    "- settled: none of the above. The event is complete, or the quantity is fixed "
    "for a named past period, or the claim reports what a named person said or did "
    "on a specific occasion, or it states a durable property. Its truth value was "
    "fixed when the post was written and cannot change afterwards.\n"
    "Answer 'settled' only when you are confident. Anything you are unsure about is "
    "one of the other four.\n\n"
    "self_contained = false when a reader with the post in front of them still could "
    "not tell what is being asserted: the claim leans on an earlier post in a thread, "
    "on the contents of a linked article, on a reply, or on a referent that appears "
    "nowhere in the post or its media. That the claim restates wording already in the "
    "post is NOT evidence of self-containedness -- every claim here was drawn from "
    "its post. Judge whether the ASSERTION is interpretable, not whether it overlaps "
    "the post. Do NOT mark false merely because the claim concerns the attached image "
    "or video -- the respondent sees those. Do NOT mark false because the claim is "
    "hard to verify; verifiability is not the question.\n\n"
    "specific = false when the claim is not the kind of statement anyone could rule "
    "true or false: vague characterisation, opinion, prediction, value judgement, "
    "unattributed 'experts warn' framing with no stated fact, or an assertion so "
    "loosely worded that two careful readers would disagree about what would settle "
    "it. true when a determined researcher would know what evidence to look for."
)


PLAUSIBILITY_SYS = (
    "You estimate how a general adult reader, with no research and no fact-checking, "
    "would react to a claim on first reading.\n\n"
    "Return JSON only:\n"
    '{"plausibility": 1|2|3|4|5, "reason": "<10 words>"}\n\n'
    "plausibility is PRIOR CREDIBILITY, not truth. Judge only how ordinary or "
    "surprising the claim sounds to someone who does not know the answer.\n"
    "1 = the reader would immediately doubt it; it contradicts common knowledge or "
    "sounds outlandish\n"
    "2 = the reader would be sceptical\n"
    "3 = the reader would have no strong prior either way\n"
    "4 = the reader would find it credible\n"
    "5 = the reader would accept it without a second thought; it sounds entirely "
    "unremarkable\n\n"
    "Do NOT attempt to determine whether the claim is true. Do not let your own "
    "knowledge of the answer move the score: a well-known falsehood that sounds "
    "plausible scores high, and a true fact that sounds astonishing scores low. "
    "Those two cases are exactly what this score exists to find."
)


def band_of(s: float | None) -> str | None:
    if s is None or pd.isna(s):
        return None
    return "FLAG" if s <= T_LOW else ("PASS" if s >= T_HIGH else "CHECK")


def load_pool() -> pd.DataFrame:
    """The 3,953 E2 ran, with voice counts, urn score, band and every strata column."""
    rows = []
    for line in E2.open():
        r = json.loads(line)
        if r.get("excluded"):
            continue
        c = Counter()
        f = Counter()
        for d in r.get("results") or []:
            direction = (d.get("read") or {}).get("direction")
            v = FLAG_TO_VOICE.get(direction)
            if v:
                c[v] += 1
                f[direction] += 1
        n_docs = sum(c.values())
        n_t, n_f = c["supports"], c["refutes"]
        n_e = max(0, n_docs - n_t - n_f)
        pad = max(0, PAD_TO - n_docs)
        rows.append({
            "claim_id": r["review_url"], "post_id": r.get("post_id"),
            "claim_text": r.get("claim_text"), "publisher_site": r.get("publisher_site"),
            "topic": r.get("topic"), "ng_score": r.get("ng_score"), "bin": r.get("bin"),
            "lean": r.get("lean"), "register": r.get("register"),
            "claim_type": r.get("claim_type"), "claim_date": r.get("claim_date_shown"),
            "screen_verdict": r.get("screen_verdict"),
            "n_docs": n_docs, "n_t": n_t, "n_f": n_f, "n_e": n_e,
            # six-flag urn (read-v6.1, 2026-09-14): a read-v5 "3" folds into "X".
            "urn_score": (sum(W["X" if k == "3" else k] * v for k, v in f.items())
                          + pad * W["I"] if n_docs else None),
        })
    df = pd.DataFrame(rows)
    df["band"] = df.urn_score.map(band_of)

    vi = pd.read_parquet(VERIFY_IN).set_index("claim_id")
    df = df.join(vi[["post_text", "created_at", "lang", "n_images", "n_videos",
                     "is_quote", "url", "handle", "like_count", "view_count"]],
                 on="claim_id")
    pr = pd.read_parquet(PROMINENCE).set_index("claim_id")
    df = df.join(pr[["coverage", "scope", "actor"]], on="claim_id")
    miss = df[["post_text", "coverage"]].isna().sum().to_dict()
    assert not any(miss.values()), f"join lost rows: {miss}"
    return df


def cluster(texts: list[str], sim_edge: float = SIM_EDGE) -> list[int]:
    """idf-weighted char-trigram Jaccard over rare-token blocks -> cluster ids."""
    n = len(texts)
    norms = [norm(t) for t in texts]
    tri = [trigrams(t) for t in norms]
    gdf = Counter()
    for s in tri:
        gdf.update(s)
    idf = {g: log(n / c) for g, c in gdf.items()}
    w = [sum(idf[g] for g in s) for s in tri]

    tok_df, toks = Counter(), []
    for t in norms:
        ts = {x for x in t.split() if len(x) > 3 and x not in _STOP and not x.isdigit()}
        toks.append(ts)
        tok_df.update(ts)
    block = defaultdict(list)
    for i, ts in enumerate(toks):
        for x in ts:
            if tok_df[x] <= RARE_DF_MAX:
                block[x].append(i)

    dsu = DSU(n)
    seen = {}
    for i, t in enumerate(norms):          # exact duplicates are free edges
        if t in seen:
            dsu.union(seen[t], i)
        else:
            seen[t] = i
    pairs = set()
    for members in block.values():
        if 2 <= len(members) <= BLOCK_CAP:
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    pairs.add((members[a], members[b]))
    for i, j in pairs:
        if dsu.find(i) == dsu.find(j):
            continue
        inter = tri[i] & tri[j]
        if not inter:
            continue
        wi = sum(idf[g] for g in inter)
        den = w[i] + w[j] - wi
        if den > 0 and wi / den >= sim_edge:
            dsu.union(i, j)
    return [dsu.find(i) for i in range(n)]


def _media_note(row) -> str:
    bits = []
    if row.n_images:
        bits.append(f"{int(row.n_images)} image(s)")
    if row.n_videos:
        bits.append(f"{int(row.n_videos)} video(s)")
    return ", ".join(bits) if bits else "no attached media"


def screen_one(row, meter) -> dict:
    user = (f"POST (published {str(row.created_at)[:10]}, attachments: {_media_note(row)}):\n"
            f"{row.post_text}\n\nCLAIM:\n{row.claim_text}")
    obj, cost, cached, ptok = llm(
        [{"role": "system", "content": SCREEN_SYS}, {"role": "user", "content": user}],
        cache_key="survey-screen-v2", max_tokens=220)
    meter["cost"] += cost
    meter["cached"] += cached
    meter["ptok"] += ptok
    tt = obj.get("temporal_type")
    return {"claim_id": row.claim_id,
            "temporal_type": tt,
            # The boolean is DERIVED, never asked for: the first smoke produced
            # rows whose reason said "ongoing" beside time_stable=true.
            "time_stable": tt == "settled",
            "time_reason": obj.get("time_reason"),
            "self_contained": bool(obj.get("self_contained")),
            "contain_reason": obj.get("contain_reason"),
            "specific": bool(obj.get("specific")),
            "specific_reason": obj.get("specific_reason")}


def plausibility_one(row, meter) -> dict:
    obj, cost, cached, ptok = llm(
        [{"role": "system", "content": PLAUSIBILITY_SYS},
         {"role": "user", "content": f"CLAIM:\n{row.claim_text}"}],
        cache_key="survey-plaus-v1", max_tokens=80)
    meter["cost"] += cost
    meter["cached"] += cached
    meter["ptok"] += ptok
    p = obj.get("plausibility")
    return {"claim_id": row.claim_id,
            "plausibility": int(p) if isinstance(p, (int, float)) and 1 <= p <= 5 else None,
            "plaus_reason": obj.get("reason")}


def run_pool(rows, fn, workers: int, budget_cap: float, label: str) -> list[dict]:
    """Worker pool with a live progress line, projection abort and a hard cap."""
    meter = {"cost": 0.0, "cached": 0, "ptok": 0, "done": 0, "stop": False}
    out, t0 = [], time.time()
    total = len(rows)
    print(f"{label}: {total} claims | {workers} workers | cap ${budget_cap}", flush=True)

    def work(row):
        if meter["stop"]:
            return
        try:
            out.append(fn(row, meter))
        except Exception as e:                       # one bad row must not kill the pass
            print(f"  [failed] {row.claim_id}: {type(e).__name__}: {e}", flush=True)
        meter["done"] += 1
        n = meter["done"]
        if n % 100 == 0 or n == total:
            el = time.time() - t0
            proj = meter["cost"] / n * total
            print(f"  {n}/{total} | ${meter['cost']:.3f} (proj ${proj:.2f}) | "
                  f"cache {meter['cached'] / max(meter['ptok'], 1):.0%} | "
                  f"{n / max(el, 1e-9) * 60:.0f}/min | {el / 60:.1f}m elapsed | "
                  f"ETA {(total - n) / max(n / max(el, 1e-9), 1e-9) / 60:.1f}m", flush=True)
            if proj > budget_cap and n >= max(25, total // 4):
                print(f"  BUDGET ABORT: projection ${proj:.2f} > cap ${budget_cap}", flush=True)
                meter["stop"] = True

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, rows))
    print(f"{label} done: {len(out)}/{total} | ${meter['cost']:.4f} | "
          f"{(time.time() - t0) / 60:.1f}m\n", flush=True)
    return out


def funnel(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n{'screen':<34}{'dropped':>9}{'remaining':>11}{'% of pool':>11}")
    print(f"{'pool (E2 run set)':<34}{'':>9}{n:>11}{1.0:>10.1%}")
    left = n
    for col, name in [("d1_has_score", "D1 has urn score / band"),
                      ("d2_english", "D2 English"),
                      ("d3_dedup_rep", "D3 near-dup representative"),
                      ("l1_time_stable", "L1 time-stable"),
                      ("l2_self_contained", "L2 self-contained"),
                      ("l3_specific", "L3 specific enough to rule on")]:
        if col not in df:
            continue
        still = df[df.get("kept_upto_" + col, True)] if False else None
        drop = int((~df[col].fillna(False) & df["_alive_" + col]).sum())
        left -= drop
        print(f"{name:<34}{drop:>9}{left:>11}{left / n:>10.1%}")
    kept = df.kept.sum()
    print(f"{'KEPT':<34}{'':>9}{kept:>11}{kept / n:>10.1%}")
    print("\nkept by band:")
    k = df[df.kept]
    for b, sub in k.groupby("band"):
        pool_b = int((df.band == b).sum())
        print(f"  {b:<6} {len(sub):5d} / {pool_b:5d} pool ({len(sub) / pool_b:.1%})  "
              f"on {sub.post_id.nunique()} posts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="screen only N claims, write nothing")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--budget", type=float, default=1.0, help="hard USD cap per pass")
    ap.add_argument("--sim-edge", type=float, default=SIM_EDGE)
    ap.add_argument("--plausibility", action="store_true",
                    help="second pass: score survivors of an existing screened parquet")
    args = ap.parse_args()

    if args.plausibility:
        df = pd.read_parquet(OUT)
        todo = df[df.kept & df.plausibility.isna()] if "plausibility" in df else df[df.kept]
        rows = list(todo.itertuples())
        res = run_pool(rows, plausibility_one, args.workers, args.budget, "plausibility")
        got = pd.DataFrame(res).set_index("claim_id")
        for c in ("plausibility", "plaus_reason"):
            df[c] = df.claim_id.map(got[c]) if c not in df else df.claim_id.map(got[c]).fillna(df[c])
        df.to_parquet(OUT)
        print(df[df.kept].plausibility.value_counts(dropna=False).sort_index().to_string())
        print(f"wrote {OUT}")
        return

    df = load_pool()
    print(f"pool: {len(df)} claims / {df.post_id.nunique()} posts / "
          f"{df.publisher_site.nunique()} outlets", flush=True)

    df["d1_has_score"] = df.band.notna()
    df["_alive_d1_has_score"] = True
    df["d2_english"] = df.lang.eq("en")
    df["_alive_d2_english"] = df.d1_has_score

    alive = df.d1_has_score & df.d2_english
    sub = df[alive].reset_index(drop=True)
    cid = cluster(sub.claim_text.tolist(), args.sim_edge)
    sub["cluster"] = cid
    size = sub.cluster.map(sub.cluster.value_counts())
    sub["cluster_size"] = size
    # Representative = the claim whose band is scarcest (FLAG first), then the
    # longest text (the most fully-stated wording of the same claim), then id.
    order = {"FLAG": 0, "CHECK": 1, "PASS": 2}
    sub["_r"] = list(zip(sub.band.map(order).fillna(9), -sub.claim_text.str.len(), sub.claim_id))
    rep = sub.sort_values("_r").groupby("cluster").head(1).claim_id
    df["cluster"] = df.claim_id.map(sub.set_index("claim_id").cluster)
    df["cluster_size"] = df.claim_id.map(sub.set_index("claim_id").cluster_size)
    df["d3_dedup_rep"] = df.claim_id.isin(set(rep))
    df["_alive_d3_dedup_rep"] = alive
    print(f"dedup @ sim>={args.sim_edge}: {sub.cluster.nunique()} clusters from {len(sub)} "
          f"claims; largest {int(sub.cluster_size.max())}", flush=True)

    alive = alive & df.d3_dedup_rep
    todo = df[alive]
    if args.smoke:
        todo = todo.sample(args.smoke, random_state=20260819)
    rows = list(todo.itertuples())
    res = run_pool(rows, screen_one, args.workers, args.budget, "screen L1+L2")
    got = pd.DataFrame(res).set_index("claim_id")

    for c in ("time_stable", "self_contained", "specific", "temporal_type",
              "time_reason", "contain_reason", "specific_reason"):
        df[c] = df.claim_id.map(got[c]) if len(got) else None
    df["l1_time_stable"] = df.time_stable.fillna(False)
    df["_alive_l1_time_stable"] = alive
    df["l2_self_contained"] = df.self_contained.fillna(False)
    df["_alive_l2_self_contained"] = alive & df.l1_time_stable
    df["l3_specific"] = df.specific.fillna(False)
    df["_alive_l3_specific"] = alive & df.l1_time_stable & df.l2_self_contained
    df["kept"] = alive & df.l1_time_stable & df.l2_self_contained & df.l3_specific

    if args.smoke:
        s = df[df.claim_id.isin(got.index)]
        print(f"\nSMOKE {len(s)} claims")
        print(f"  temporal_type: {s.temporal_type.value_counts().to_dict()}")
        print(f"  time_stable    {s.time_stable.sum():3d} / {len(s)}")
        print(f"  self_contained {s.self_contained.sum():3d} / {len(s)}")
        print(f"  specific       {s.specific.sum():3d} / {len(s)}")
        print(f"  all three      "
              f"{int((s.time_stable & s.self_contained & s.specific).sum()):3d} / {len(s)}")
        for _, r in s.iterrows():
            ok = r.time_stable and r.self_contained and r.specific
            print(f"\n  {'KEEP' if ok else 'DROP'} [{r.band}] {r.claim_text[:120]}")
            print(f"    {r.temporal_type} ({r.time_reason}) | "
                  f"contained={r.self_contained} ({r.contain_reason}) | "
                  f"specific={r.specific} ({r.specific_reason})")
        return

    funnel(df)
    df.drop(columns=[c for c in df if c.startswith("_alive_")] + ["_r"], errors="ignore"
            ).to_parquet(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
