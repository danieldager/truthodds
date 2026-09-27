"""What share of the timeline urn is actually FALSE? A score-blind random audit.

    uv run python -m eval.scripts.build_eval.timeline_eps_audit --sample --n 40 --tag smoke
    uv run python -m eval.scripts.build_eval.timeline_eps_audit --sample --n 400 --tag main
    uv run python -m eval.scripts.build_eval.timeline_eps_audit --ingest --tag smoke
    uv run python -m eval.scripts.build_eval.timeline_eps_audit --corpus

eps is the false share of the timeline urn. Every de-mixed weight, the
contamination-corrected urn-side AUC, and the flag-rate ceiling are denominated
in it, and until now it was ASSUMED at 0.10 and never measured.

WHY THE EXISTING AUDIT CANNOT DO THIS. `band_audit.jsonl` read the bottom 400
claims BY SCORE and found 192 false. Conditioning the draw on the score and then
reporting a false rate measures the score, not the population. That design
estimates precision (conditional on being flagged by construction) and cannot
estimate prevalence. This one draws uniformly at random, seeded, before any score
is consulted, so every claim has the same inclusion probability whether it scores
-12 or +6. That independence is the whole property that makes the rate
extrapolate.

SAMPLE FRAME. The 1,999 claims of `true_timeline/scores.jsonl` that carry at
least one read direction -- exactly the rows `fit_two_urn.load_urn` keeps and
therefore exactly the population eps multiplies in the de-mix. The 78 zero-doc
rows are outside the fit and outside this frame; they are reported as a residual,
not sampled. Sampling the wider checkworthy set would answer a different question
("what fraction of the timeline is false") over a population no downstream number
is denominated in.

BLIND. The auditor sees the claim and its source post and nothing else: no s7, no
flag, no band, no document our retrieval returned, no community note. It does its
own web research. Same protocol as the acquitted-trues audit of 2026-08-27
(clog/270826.md), which returned 44% FALSE against an expected near-zero
precisely because nothing could leak the answer in.

Labels: TRUE / FALSE / UNVERIFIABLE. Post-clustered intervals throughout, because
several claims come from the same post and those are not independent draws.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval.tl_band_audit import load_fit_weights  # noqa: E402

RUN = SRC / "eval/data/urn_runs/true_timeline"          # READ ONLY
SCORES = RUN / "scores.jsonl"
URN = SRC / "eval/data/tweet_corpus/timeline_urn.parquet"
OUT = SRC / "eval/data/eps_audit"                        # new dir, nothing under urn_runs
SEED = 20260828
BATCH = 20            # claims per auditor, same as the 160/8 acquitted audit

LABELS = ("TRUE", "FALSE", "UNVERIFIABLE")
WHY_CODES = ("no_public_record", "media_dependent", "context_dependent",
             "vague_underspecified", "future_or_prediction", "opinion_not_factual",
             "sources_conflict", "record_inaccessible", "other")

AUDIT_BRIEF = """\
You are auditing claims taken from real posts on X. For each claim, decide whether it is
TRUE, FALSE, or UNVERIFIABLE, using your own web research. Work claim by claim.

You are shown the claim and the post it came from, and nothing else. There is no score, no
flag, no prior verdict and no evidence bundle, deliberately. Do not try to guess what any
system thought. Research each claim yourself.

METHOD, per claim
1. Read the claim against its post. The post is context for resolving pronouns, dates and
   referents. The proposition you judge is the CLAIM, not the post's overall thrust.
2. Search the web. Use several queries and different phrasings. Prefer primary sources,
   official records, and reputable outlets over aggregators and social posts. Two
   syndicated copies of one wire story are one source, not two.
3. Mind the date. The post date is given. Judge the claim as it stood when posted, then
   check whether anything since supersedes it. Your training data ends before these posts,
   so nothing about them is in your memory. Search, do not recall.

LABELS
TRUE - the evidence establishes the claim is accurate as stated. Minor imprecision that
  does not change what the claim asserts is still TRUE.
FALSE - the evidence establishes it is inaccurate, fabricated, or misleading as stated.
  This includes a materially wrong number, a wrong attribution, a real fact framed to
  assert something untrue, and a claim whose load-bearing part fails even where an
  adjacent fact checks out. A claim that is half right is FALSE if the wrong half is the
  point.
UNVERIFIABLE - you cannot settle it. Use this honestly and do not stretch to a verdict.
  It is a real category and its size is itself a result we want.

Do NOT let these push a label:
- how plausible, partisan, or unpleasant the claim sounds
- whether the account looks credible
- whether the topic is contested in general
Only what the evidence about THIS proposition says.

CALIBRATION
Most claims on a normal timeline are ordinary true statements. Do not hunt for falsity.
Equally, do not wave through a specific number or attribution you could not confirm -- if
you could not confirm it, that is UNVERIFIABLE, not TRUE.

OUTPUT
Append one JSON object per claim, one per line, to the output file you were given. No
array, no markdown fence, no commentary in the file.

{"aid": "<the aid from the packet, copied exactly>",
 "label": "TRUE" | "FALSE" | "UNVERIFIABLE",
 "confidence": 1-5,
 "why": "<the single reason, <= 35 words>",
 "why_code": "<UNVERIFIABLE only, else null>",
 "sources": ["<url>", ...],
 "checked_part": "<the exact proposition you judged, <= 20 words>"}

why_code, for UNVERIFIABLE only, exactly one of:
  no_public_record      nobody published anything that settles it; too small or too local
  media_dependent       it turns on an image or video you cannot see
  context_dependent     it turns on a linked article, quoted post or thread you cannot see
  vague_underspecified  no fixed referent, quantity or timeframe to check
  future_or_prediction  about the future, or not yet resolved
  opinion_not_factual   on inspection it is not truth-apt at all
  sources_conflict      reputable sources genuinely disagree and none supersedes
  record_inaccessible   a record exists but is paywalled, deleted, or unreachable
  other                 none of the above; say what in `why`

Set `sources` to the URLs that actually decided it, up to 4. TRUE and FALSE need at least
one. Write every line as you finish each claim, so a stall loses nothing.
"""


def brief_hashes() -> dict[str, str]:
    """Hash the auditor briefs as handed out (AUDIT_BRIEF above is their source)."""
    return {f.stem: prompt_hash(f.read_text())
            for f in (OUT / "AUDIT_BRIEF.md", OUT / "AUDIT_BRIEF_DEEP.md") if f.exists()}


# ---------------------------------------------------------------- intervals
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_boot(units: list[tuple[int, int]], reps: int = 20000,
                 seed: int = 909) -> tuple[float, float, float]:
    """Post-clustered bootstrap on eps = sum(F) / sum(T+F).

    `units` is one (n_false, n_decided) pair per POST. Resampling posts with
    replacement propagates the within-post correlation that a claim-level Wilson
    interval ignores; a post whose claims are all false contributes as a block.
    """
    rng = np.random.default_rng(seed)
    f = np.array([u[0] for u in units], float)
    d = np.array([u[1] for u in units], float)
    m = len(units)
    idx = rng.integers(0, m, size=(reps, m))
    num, den = f[idx].sum(1), d[idx].sum(1)
    ok = den > 0
    est = num[ok] / den[ok]
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(f.sum() / max(d.sum(), 1)), float(lo), float(hi)


def design_effect(units: list[tuple[int, int]]) -> float:
    """deff = post-clustered variance / binomial variance, on the same data."""
    n = sum(u[1] for u in units)
    k = sum(u[0] for u in units)
    if not n or not k or k == n:
        return float("nan")
    p = k / n
    _, lo, hi = cluster_boot(units)
    v_cl = ((hi - lo) / (2 * 1.96)) ** 2
    return v_cl / (p * (1 - p) / n)


# ---------------------------------------------------------------- frame
def frame_rows() -> list[dict]:
    """The 1,999 fit rows: >= 1 read direction. s7 is carried for ANALYSIS ONLY
    and never enters a packet."""
    w = load_fit_weights()
    rows = []
    for line in SCORES.open():
        r = json.loads(line)
        flags = [d["read"]["direction"] for d in r["results"]
                 if d.get("read") and d["read"].get("direction")]
        if not flags:
            continue
        rows.append({
            "aid": r["review_url"], "post_id": r["post_id"],
            "claim": r.get("claim_resolved") or r.get("claim_text"),
            "frame": r.get("frame"), "topic": r.get("topic"), "handle": r.get("handle"),
            "s7": sum(w.get(f, 0.0) for f in flags), "n_docs": len(flags),
            "claim_date": r.get("claim_date_shown")})
    rows.sort(key=lambda r: r["aid"])          # deterministic before the shuffle
    return rows


def _iso(s: str | None) -> str | None:
    """'Tue Aug 25 23:50:21 +0000 2026' -> '2026-08-25'. The auditor needs the year."""
    if not s:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").strftime("%Y-%m-%d")
    except ValueError:
        return str(s)[:10]


def post_fields() -> dict[str, dict]:
    u = pl.read_parquet(URN).select(
        ["post_id", "post_text", "handle", "created_at", "is_quote", "quoted_handle",
         "quoted_text", "n_images", "n_videos", "retweeted_by", "is_reply"]).unique(
        subset="post_id", keep="first")
    return {r["post_id"]: r for r in u.iter_rows(named=True)}


def sample(n: int, tag: str, seed: int, exclude: list[str]) -> None:
    rows = frame_rows()
    taken = set()
    for t in exclude:
        p = OUT / t / "sample.parquet"
        if p.exists():
            taken |= set(pl.read_parquet(p)["aid"].to_list())
    pool = [r for r in rows if r["aid"] not in taken]
    print(f"frame {len(rows):,} scored claims / {len({r['post_id'] for r in rows}):,} posts"
          f" | already drawn {len(taken):,} | pool {len(pool):,}", flush=True)
    if n > len(pool):
        raise SystemExit(f"asked {n}, pool {len(pool)}")

    rng = random.Random(seed)
    draw = rng.sample(pool, n)                 # uniform, without replacement, score-blind
    pf = post_fields()
    d = OUT / tag
    (d / "packets").mkdir(parents=True, exist_ok=True)
    (d / "labels").mkdir(parents=True, exist_ok=True)

    pl.DataFrame(draw).write_parquet(d / "sample.parquet")

    n_b = math.ceil(n / BATCH)
    for b in range(n_b):
        chunk = draw[b * BATCH:(b + 1) * BATCH]
        items = []
        for r in chunk:
            p = pf.get(r["post_id"], {})
            items.append({
                "aid": r["aid"],
                "claim": r["claim"],
                "post_text": p.get("post_text"),
                "handle": p.get("handle"),
                "posted_at": _iso(p.get("created_at")),
                "is_quote_post": bool(p.get("is_quote")),
                "quoted_handle": p.get("quoted_handle"),
                "quoted_text": p.get("quoted_text"),
                "post_has_image": int(p.get("n_images") or 0) > 0,
                "post_has_video": int(p.get("n_videos") or 0) > 0,
                "is_reply": bool(p.get("is_reply")),
                "retweeted_by": p.get("retweeted_by")})
        (d / "packets" / f"batch_{b:02d}.json").write_text(json.dumps(items, indent=1))

    # sanity: nothing score-shaped escaped into a packet
    blob = json.dumps([json.loads((d / "packets" / f).read_text())
                       for f in sorted(x.name for x in (d / "packets").glob("*.json"))])
    for leak in ("s7", "n_docs", "direction", "supports", "refutes", "flag", "band",
                 "verdict", "note"):
        assert f'"{leak}"' not in blob, f"packet leak: {leak}"

    print(f"wrote {n_b} packets of <= {BATCH} to {d/'packets'}  (seed {seed})", flush=True)
    print(f"s7 in draw: min {min(r['s7'] for r in draw):.2f} "
          f"max {max(r['s7'] for r in draw):.2f} "
          f"mean {sum(r['s7'] for r in draw)/n:.2f}   (frame mean "
          f"{sum(r['s7'] for r in rows)/len(rows):.2f})", flush=True)


# ---------------------------------------------------------------- ingest
def load_labels(tag: str) -> pl.DataFrame:
    d = OUT / tag
    recs = collections.defaultdict(list)
    for f in sorted((d / "labels").glob("*.jsonl")):
        pas = f.stem.split("__")[-1] if "__" in f.stem else "p1"
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"  ! unparseable line in {f.name}", flush=True)
                continue
            if r.get("label") not in LABELS:
                print(f"  ! bad label {r.get('label')!r} in {f.name}", flush=True)
                continue
            r["_pass"], r["_file"] = pas, f.name
            recs[r["aid"]].append(r)
    return recs


def ingest(tag: str) -> None:
    d = OUT / tag
    smp = pl.read_parquet(d / "sample.parquet")
    recs = load_labels(tag)
    meta = {r["aid"]: r for r in smp.iter_rows(named=True)}

    n_pass = collections.Counter(len(v) for v in recs.values())
    print(f"\n=== {tag} ===")
    print(f"sample {smp.height} | labelled {len(recs)} | passes per claim {dict(n_pass)}")
    missing = [a for a in meta if a not in recs]
    if missing:
        print(f"  ! {len(missing)} unlabelled: {missing[:5]}")

    # ---- inter-auditor agreement, where a claim was labelled more than once
    dbl = {a: v for a, v in recs.items() if len(v) > 1}
    if dbl:
        agree = sum(1 for v in dbl.values() if len({x["label"] for x in v}) == 1)
        print(f"\ndouble-labelled {len(dbl)}  raw agreement {agree}/{len(dbl)} "
              f"({agree/len(dbl)*100:.0f}%)")
        cm = collections.Counter()
        for v in dbl.values():
            cm[tuple(sorted((v[0]["label"], v[1]["label"])))] += 1
        for k, c in cm.most_common():
            print(f"   {k[0]:<13} x {k[1]:<13} {c}")
        # binary T/F agreement, ignoring UNVERIFIABLE
        bin_ = [v for v in dbl.values()
                if all(x["label"] in ("TRUE", "FALSE") for x in v)]
        if bin_:
            ok = sum(1 for v in bin_ if v[0]["label"] == v[1]["label"])
            print(f"   TRUE/FALSE-only pairs {len(bin_)}, agreement {ok}/{len(bin_)}")

    # ---- per-pass eps, so the consensus rule can be seen to matter or not
    if dbl:
        print("\neps by pass (each pass alone, before any consensus rule):")
        for pas in sorted({x["_pass"] for v in recs.values() for x in v}):
            c = collections.Counter(x["label"] for v in recs.values()
                                    for x in v if x["_pass"] == pas)
            dec = c["TRUE"] + c["FALSE"]
            wl, wh = wilson(c["FALSE"], dec) if dec else (0, 1)
            print(f"   {pas}  T {c['TRUE']:>3} F {c['FALSE']:>3} U {c['UNVERIFIABLE']:>3}"
                  f"  eps {c['FALSE']/dec if dec else float('nan'):.4f} [{wl:.4f},{wh:.4f}]")

    # ---- consensus label: unanimous, else the stricter reading (FALSE > UNVER > TRUE)
    order = {"FALSE": 0, "UNVERIFIABLE": 1, "TRUE": 2}
    final = {}
    for a, v in recs.items():
        labs = [x["label"] for x in v]
        final[a] = {"label": sorted(labs, key=lambda l: order[l])[0] if len(set(labs)) > 1
                    else labs[0],
                    "disputed": len(set(labs)) > 1,
                    "recs": v}

    counts = collections.Counter(f["label"] for f in final.values())
    T, F, U = counts["TRUE"], counts["FALSE"], counts["UNVERIFIABLE"]
    n_dec = T + F
    print(f"\nlabels  TRUE {T}  FALSE {F}  UNVERIFIABLE {U}   (n={len(final)})")
    print(f"UNVERIFIABLE share of sample {U/len(final)*100:.1f}%")

    # ---- eps
    print(f"\neps = FALSE/(TRUE+FALSE) = {F}/{n_dec} = {F/n_dec:.4f}")
    lo, hi = wilson(F, n_dec)
    print(f"  naive Wilson 95%       [{lo:.4f}, {hi:.4f}]  (+/- {(hi-lo)/2*100:.1f}pp)")

    per_post = collections.defaultdict(lambda: [0, 0])
    for a, f in final.items():
        if f["label"] == "UNVERIFIABLE":
            continue
        pid = meta[a]["post_id"]
        per_post[pid][1] += 1
        per_post[pid][0] += f["label"] == "FALSE"
    units = [tuple(v) for v in per_post.values()]
    est, clo, chi = cluster_boot(units)
    deff = design_effect(units)
    print(f"  post-clustered 95%     [{clo:.4f}, {chi:.4f}]  (+/- {(chi-clo)/2*100:.1f}pp)"
          f"   posts {len(units)}, claims/post {n_dec/len(units):.2f}, deff {deff:.2f}")

    # eps over the WHOLE sample, treating UNVERIFIABLE as a third outcome, is the
    # bound if every unverifiable claim were secretly true / secretly false
    print(f"  bounds if UNVERIFIABLE all true  {F/len(final):.4f}"
          f" / all false  {(F+U)/len(final):.4f}")

    # ---- UNVERIFIABLE, characterised
    print("\nUNVERIFIABLE by reason:")
    why = collections.Counter()
    for a, f in final.items():
        if f["label"] != "UNVERIFIABLE":
            continue
        codes = [x.get("why_code") for x in f["recs"] if x["label"] == "UNVERIFIABLE"]
        why[(codes[0] or "other") if codes else "other"] += 1
    for k, c in why.most_common():
        print(f"   {k:<22} {c:>4}  {c/max(U,1)*100:>5.1f}% of unver  "
              f"{c/len(final)*100:>5.1f}% of sample")

    # ---- cuts
    def cut(name: str, keyf):
        print(f"\nby {name}:")
        g = collections.defaultdict(lambda: collections.Counter())
        for a, f in final.items():
            g[keyf(meta[a])][f["label"]] += 1
        for k in sorted(g, key=lambda k: -sum(g[k].values())):
            c = g[k]
            dec = c["TRUE"] + c["FALSE"]
            e = c["FALSE"] / dec if dec else float("nan")
            wl, wh = wilson(c["FALSE"], dec)
            print(f"   {str(k):<16} n {sum(c.values()):>4}  T {c['TRUE']:>3} "
                  f"F {c['FALSE']:>3} U {c['UNVERIFIABLE']:>3}  eps {e:.3f} "
                  f"[{wl:.3f},{wh:.3f}]")

    cut("frame", lambda m: m["frame"])
    cut("topic", lambda m: m["topic"])
    bands = [(-99, -6), (-6, -3), (-3, 0), (0, 3), (3, 99)]
    cut("s7 band", lambda m: next(f"[{a},{b})" for a, b in bands if a <= m["s7"] < b))

    payload = {"tag": tag, "prompt_hash": brief_hashes(),
               "n": len(final), "TRUE": T, "FALSE": F, "UNVERIFIABLE": U,
               "eps": F / n_dec, "wilson": [lo, hi],
               "clustered": [clo, chi], "deff": deff, "n_posts": len(units),
               "unver_why": dict(why),
               "labels": {a: {"label": f["label"], "disputed": f["disputed"],
                              "recs": f["recs"]} for a, f in final.items()}}
    (d / "result.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {d/'result.json'}", flush=True)


# ---------------------------------------------------------------- true corpus
def corpus(tags: list[str]) -> None:
    """Confirmed TRUEs, in two loadable forms.

    hard_true_claims.parquet  the urn rows plus the audit label
    hard_true_scores.jsonl    the scores.jsonl subset -- drops straight into
                              fit_urn / fit_two_urn as an eps=0 TRUE urn
    """
    keep = {}
    for t in tags:
        p = OUT / t / "result.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        for aid, l in r["labels"].items():
            # `recs` is the per-tag ingest schema, `passes` the pooled analyze one
            recs = l.get("recs") or [x for v in l.get("passes", {}).values() for x in v]
            labs = {x["label"] for x in recs}
            # A TRUE urn must not contain a claim any reader called false, and it SHOULD
            # contain the ones the deep pass lifted out of UNVERIFIABLE into TRUE, which
            # are the hardest trues in the draw.
            if "FALSE" in labs or "TRUE" not in labs:
                continue
            deep = [x for x in recs if x["_pass"] == P_UNVER3]
            keep[aid] = {"aid": aid, "audit_tag": t,
                         "disputed": l.get("disputed", len(labs) > 1),
                         "n_reads": len(recs), "second_read": len(recs) > 1,
                         "route": ("deep_resolved" if l["label"] == "UNVERIFIABLE"
                                   else "first_pass"),
                         "resolvability": (deep[0].get("resolvability") if deep else None),
                         "confidence": min(x.get("confidence") or 0 for x in recs),
                         "sources": json.dumps(recs[0].get("sources") or []),
                         "why": recs[0].get("why")}
    if not keep:
        raise SystemExit("no confirmed TRUEs yet")

    u = pl.read_parquet(URN).rename({"claim_id": "aid"})
    out = pl.DataFrame(list(keep.values())).join(u, on="aid", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT / "hard_true_claims.parquet")

    with (OUT / "hard_true_scores.jsonl").open("w") as fh:
        n = 0
        for line in SCORES.open():
            r = json.loads(line)
            if r["review_url"] in keep:
                r["audit_label"] = "TRUE"
                r["audit_confidence"] = keep[r["review_url"]]["confidence"]
                fh.write(json.dumps(r) + "\n")
                n += 1
    print(f"hard-true corpus: {out.height} claims / {out['post_id'].n_unique()} posts, "
          f"{n} scored rows -> {OUT}", flush=True)
    print(out.group_by("topic").len().sort("len", descending=True).to_dicts(), flush=True)


def sweep(tag: str) -> None:
    """How far do the existing de-mixed results move at the measured eps?

    Re-runs the two-urn fit and the cross ladder at {lo, point, hi} of the
    post-clustered interval alongside the assumed 0.10, both redirected to
    eval/data/eps_audit/ so nothing pinned under urn_runs is overwritten.
    """
    import os
    import subprocess
    r = json.loads((OUT / tag / "result.json").read_text())
    lo, hi = r["clustered"]
    pts = sorted({round(x, 4) for x in (lo, r["eps"], hi, 0.10)})
    print(f"\nmeasured eps {r['eps']:.4f}, clustered 95% [{lo:.4f}, {hi:.4f}]")
    print(f"sweeping {pts} against the assumed 0.10\n", flush=True)

    env = dict(os.environ, TWO_URN_OUT=str(OUT / tag / "two_urn_fit_measured.json"))
    subprocess.run([sys.executable, "-m", "eval.scripts.build_eval.fit_two_urn",
                    "--eps", *[str(p) for p in pts]], cwd=SRC, env=env, check=True)

    d = OUT / tag / "cross_measured"
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CROSS_OUT=str(d), EPS_HEAD=str(r["eps"]),
               EPS_SWEEP=",".join(str(p) for p in pts), REPS="400")
    subprocess.run([sys.executable, "-m", "eval.scripts.build_eval.cross_ladder"],
                   cwd=SRC, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="tags whose claims are already drawn (extension without overlap)")
    ap.add_argument("--tags", nargs="*", default=["smoke", "main"])
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--followup", action="store_true")
    a = ap.parse_args()
    if a.sample:
        sample(a.n, a.tag, a.seed, a.exclude)
    if a.ingest:
        ingest(a.tag)
    if a.corpus:
        corpus(a.tags)
    if a.followup:
        followup(a.tags)
    if a.analyze:
        analyze(a.tags, a.tag)
    if a.sweep:
        sweep(a.tag)



# ================================================================ scale-up (2026-08-28)
# The n=40 shakedown double-read every claim. At n=400 that is the wrong place to
# spend reads: eps is far more sensitive to numerator error than denominator error,
# so the second reads go where they change the number.
#
#   p1    one blind read on every claim                  (the estimator)
#   p2f   a second blind read on every p1-FALSE          (numerator confirmation)
#   p2t   a second blind read on 40 random p1-TRUEs      (the reverse control)
#   p3u   a deeper blind read on every p1-UNVERIFIABLE   (ambiguity-band narrowing)
#
# p2t is what makes p2f interpretable. Without it a second reader who overturns
# FALSEs cannot be told apart from one who is simply more sceptical: the control
# measures P(read2 = FALSE | read1 = TRUE) on claims where there is nothing to catch.
#
# WEIGHTS. The first pass banded s7 with the two-urn eps=0.05 weights. Layer 0
# reversed on 2026-08-28 (docs/logodds_sprint.md) and fc-gold is now the fit
# corpus, so the bands are re-cut with gold-fit Laplace log-LR weights as well.
# The two weight sets have different dynamic ranges (flag 5 is +2.94 gold against
# +1.29 two-urn), so fixed +/-3, +/-6 cuts are not comparable across them. The
# gold cut is taken at the FRAME PERCENTILES of the two-urn bands, which holds the
# per-band denominators fixed and leaves only the ordering to differ.

P_PRIMARY, P_FALSE2, P_TRUE2, P_UNVER3 = "p1", "p2f", "p2t", "p3u"
BANDS_2URN = [(-99.0, -6.0), (-6.0, -3.0), (-3.0, 0.0), (0.0, 3.0), (3.0, 99.0)]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval. Asked for by name; Wilson is also reported."""
    from scipy import stats
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def gold_fit_weights() -> dict[str, float]:
    """7-flag Laplace log-LR fitted on the fc-gold corpus, both classes.

    Mirrors cross_ladder's `weights(fit="gold", ...)` on the full corpus: the
    timeline is a different corpus, so there is no fold to hold out.
    """
    import numpy as np
    from eval.scripts.build_eval import cross_ladder as CL
    from eval.scripts.build_eval import model_ladder as ML
    gold = [r for r in ML.load_docs() if r["subtype"] != ML.SET_ASIDE]
    y = np.array([r["y"] for r in gold])
    fitrow = ~np.array([r["mid"] for r in gold])
    G = ML.count_matrix(gold, "7-flag")
    w = CL.laplace(G[:, fitrow & (y == 1)].sum(1), G[:, fitrow & (y == 0)].sum(1))
    return {c: float(x) for c, x in zip(ML.channels("7-flag"), w)}


def frame_scores() -> tuple[list[dict], dict[str, float], dict[str, float]]:
    """Every frame claim scored under BOTH weight sets, plus the gold cut points."""
    import numpy as np
    w2, wg = load_fit_weights(), gold_fit_weights()
    rows = []
    for line in SCORES.open():
        r = json.loads(line)
        flags = [d["read"]["direction"] for d in r["results"]
                 if d.get("read") and d["read"].get("direction")]
        if not flags:
            continue
        rows.append({"aid": r["review_url"],
                     "s7": sum(w2.get(f, 0.0) for f in flags),
                     "s7g": sum(wg.get(f, 0.0) for f in flags)})
    s2 = np.array([r["s7"] for r in rows])
    qs = [float((s2 < b).mean()) * 100 for _, b in BANDS_2URN[:-1]]
    cuts = [float(np.percentile([r["s7g"] for r in rows], q)) for q in qs]
    return rows, {r["aid"]: r["s7"] for r in rows}, \
        ({r["aid"]: r["s7g"] for r in rows}, cuts)


def band_of(s: float, cuts: list[float]) -> int:
    import bisect
    return bisect.bisect_right(cuts, s)


def cochran_armitage(k: list[int], n: list[int]) -> tuple[float, float]:
    """Trend in a proportion across ordered bands. Scores are band indices."""
    import numpy as np
    from scipy import stats
    k, n = np.array(k, float), np.array(n, float)
    x = np.arange(len(n), dtype=float)
    N, K = n.sum(), k.sum()
    if N == 0 or K in (0, N):
        return (float("nan"), 1.0)
    p = K / N
    num = (k * x).sum() - p * (n * x).sum()
    var = p * (1 - p) * ((n * x * x).sum() - (n * x).sum() ** 2 / N)
    if var <= 0:
        return (float("nan"), 1.0)
    z = num / math.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def cluster_perm_rank(s: list[float], lab: list[int], pid: list[str],
                      reps: int = 20000, seed: int = 4242) -> tuple[float, float]:
    """Is s lower for FALSE claims than TRUE ones? Rank statistic, post-clustered
    permutation null.

    The statistic is the Mann-Whitney AUC of s separating TRUE from FALSE. The
    null shuffles labels BETWEEN POSTS, not between claims, so a post whose claims
    are all false stays a block and the p-value is not inflated by the fact that
    several claims share a post.
    """
    import numpy as np
    from scipy import stats
    s, lab = np.asarray(s, float), np.asarray(lab, int)
    posts = sorted(set(pid))
    pidx = np.array([posts.index(p) for p in pid])

    def auc(l):
        a, b = s[l == 1], s[l == 0]
        if not len(a) or not len(b):
            return 0.5
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").statistic
                     / (len(a) * len(b)))
    obs = auc(lab)
    # post-level label pool: permute whole posts' label vectors
    by_post = [lab[pidx == i] for i in range(len(posts))]
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(reps):
        order = rng.permutation(len(posts))
        perm = np.empty_like(lab)
        # reassign each post's label BLOCK to another post of the same size where
        # possible; fall back to a within-length-class shuffle of the pooled labels
        pool = np.concatenate([by_post[i] for i in order])
        cur = 0
        for i in range(len(posts)):
            m = pidx == i
            perm[m] = pool[cur:cur + m.sum()]
            cur += m.sum()
        if abs(auc(perm) - 0.5) >= abs(obs - 0.5):
            hits += 1
    return obs, (hits + 1) / (reps + 1)


def _passes(tags: list[str]) -> dict[str, dict[str, list[dict]]]:
    """aid -> pass-role -> records, pooled over tags.

    The smoke ran two FULL passes (p1, p2) over all 40. Its p2 is folded into the
    scale-up's roles by what p1 said: p2 on a p1-FALSE is a numerator confirmation,
    p2 on a p1-TRUE is a reverse control. Same read, same brief, so pooling is
    legitimate and it buys 40 free double-reads.
    """
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in tags:
        for a, recs in load_labels(t).items():
            for r in recs:
                out[a][r["_pass"]].append(r)
    return out


def analyze(tags: list[str], out_tag: str = "pooled") -> None:
    import numpy as np
    from scipy import stats

    _, s2, (sg, gcuts) = frame_scores()
    meta = {}
    for t in tags:
        for r in pl.read_parquet(OUT / t / "sample.parquet").iter_rows(named=True):
            meta[r["aid"]] = r
    P = _passes(tags)
    P = {a: v for a, v in P.items() if a in meta}

    def lab(a, role):
        v = P[a].get(role)
        return v[0]["label"] if v else None

    prim = {a: lab(a, P_PRIMARY) for a in P}
    prim = {a: l for a, l in prim.items() if l}
    print(f"=== eps audit, pooled {tags} ===")
    print(f"frame 1,999 scored claims / 765 posts | drawn {len(meta)} | p1-labelled {len(prim)}")
    miss = [a for a in meta if a not in prim]
    if miss:
        print(f"  ! {len(miss)} drawn but unlabelled")

    c = collections.Counter(prim.values())
    T, F, U = c["TRUE"], c["FALSE"], c["UNVERIFIABLE"]
    n_dec = T + F
    print(f"\np1 labels  TRUE {T}  FALSE {F}  UNVERIFIABLE {U}  (n={len(prim)}, "
          f"U share {U/len(prim)*100:.1f}%)")

    # ------------------------------------------------ 1. eps and intervals
    eps1 = F / n_dec
    cp = clopper_pearson(F, n_dec)
    wl, wh = wilson(F, n_dec)
    per_post = collections.defaultdict(lambda: [0, 0])
    for a, l in prim.items():
        if l == "UNVERIFIABLE":
            continue
        u = per_post[meta[a]["post_id"]]
        u[1] += 1
        u[0] += l == "FALSE"
    units = [tuple(v) for v in per_post.values()]
    _, clo, chi = cluster_boot(units)
    deff = design_effect(units)
    print(f"\neps = FALSE/(TRUE+FALSE) = {F}/{n_dec} = {eps1:.4f}")
    print(f"  Clopper-Pearson 95%    [{cp[0]:.4f}, {cp[1]:.4f}]  (+/-{(cp[1]-cp[0])/2*100:.1f}pp)")
    print(f"  Wilson 95%             [{wl:.4f}, {wh:.4f}]")
    print(f"  post-clustered 95%     [{clo:.4f}, {chi:.4f}]  (+/-{(chi-clo)/2*100:.1f}pp)"
          f"   posts {len(units)}, claims/post {n_dec/len(units):.2f}, deff {deff:.2f}")
    print(f"  UNVERIFIABLE bound: all true {F/len(prim):.4f} / all false {(F+U)/len(prim):.4f}"
          f"  (span {((F+U)/len(prim)-F/len(prim))*100:.1f}pp)")

    # ------------------------------------------------ 2. second reads
    conf = [a for a in prim if prim[a] == "FALSE" and (P[a].get(P_FALSE2) or P[a].get("p2"))]
    ctrl = [a for a in prim if prim[a] == "TRUE" and (P[a].get(P_TRUE2) or P[a].get("p2"))]

    def second(a):
        v = P[a].get(P_FALSE2) or P[a].get(P_TRUE2) or P[a].get("p2")
        return v[0]["label"]
    nc = sum(1 for a in conf if second(a) == "FALSE")
    nf = sum(1 for a in ctrl if second(a) == "FALSE")
    sr = {"n_false_reread": len(conf), "false_confirmed": nc,
          "n_true_reread": len(ctrl), "true_overturned": nf,
          "false_second": dict(collections.Counter(second(a) for a in conf)),
          "true_second": dict(collections.Counter(second(a) for a in ctrl))}
    print(f"\n--- second reads ---")
    if conf:
        lo, hi = clopper_pearson(nc, len(conf))
        print(f"  FALSE confirmed by an independent second read  {nc}/{len(conf)} = "
              f"{nc/len(conf):.3f} [{lo:.3f},{hi:.3f}]")
        print("   " + "  ".join(f"{k}:{v}" for k, v in
                                collections.Counter(second(a) for a in conf).items()))
    if ctrl:
        lo, hi = clopper_pearson(nf, len(ctrl))
        print(f"  TRUE overturned to FALSE by a second read      {nf}/{len(ctrl)} = "
              f"{nf/len(ctrl):.3f} [{lo:.3f},{hi:.3f}]   (reverse control)")
        print("   " + "  ".join(f"{k}:{v}" for k, v in
                                collections.Counter(second(a) for a in ctrl).items()))
    if conf and ctrl:
        odds, pf = stats.fisher_exact([[nc, len(conf) - nc], [nf, len(ctrl) - nf]])
        print(f"  is the second reader catching errors or just sceptical?  "
              f"Fisher exact p={pf:.2e}, OR={odds:.1f}")
        chat, fhat = nc / len(conf), nf / len(ctrl)
        print(f"  eps under AND (both readers FALSE)  {eps1*chat:.4f}")
        print(f"  eps under p1 alone                  {eps1:.4f}   <- the estimator")
        print(f"  eps under OR  (either reader FALSE) {eps1 + (1-eps1)*fhat:.4f}")
        sr |= {"fisher_p": float(pf), "or": float(odds),
               "eps_and": eps1 * chat, "eps_p1": eps1,
               "eps_or": eps1 + (1 - eps1) * fhat}

    # ------------------------------------------------ 3. unverifiable resolution
    unv = [a for a in prim if prim[a] == "UNVERIFIABLE"]
    deep = [a for a in unv if P[a].get(P_UNVER3)]
    du = {"n_unver": len(unv), "n_deep": len(deep)}
    print(f"\n--- UNVERIFIABLE deep pass ---   {len(deep)}/{len(unv)} re-read")
    if deep:
        dl = collections.Counter(P[a][P_UNVER3][0]["label"] for a in deep)
        rt = collections.Counter(P[a][P_UNVER3][0].get("resolvability") for a in deep)
        tx = collections.Counter(P[a][P_UNVER3][0].get("taxon") for a in deep)
        print("  deep label:", dict(dl))
        print("  resolvability:", dict(rt))
        print("  taxonomy:")
        for k, v in tx.most_common():
            print(f"    {str(k):<28} {v:>4}  {v/len(deep)*100:>5.1f}%")
        rT, rF = dl["TRUE"], dl["FALSE"]
        still = dl["UNVERIFIABLE"]
        # scale the resolved fractions to the whole U bucket if only part was re-read
        sc = len(unv) / len(deep)
        nT, nF, nU = rT * sc, rF * sc, still * sc
        num, den = F + nF, n_dec + nT + nF
        print(f"\n  eps with resolved unverifiables folded in = "
              f"({F}+{nF:.0f})/({n_dec}+{nT+nF:.0f}) = {num/den:.4f}")
        blo = num / (den + nU)
        bhi = (num + nU) / (den + nU)
        print(f"  residual ambiguity band  [{blo:.4f}, {bhi:.4f}]  (span "
              f"{(bhi-blo)*100:.1f}pp, was {((F+U)/len(prim)-F/len(prim))*100:.1f}pp)")
        # taxon is recorded for EVERY claim, including the ones the deep pass resolved,
        # so `taxonomy` above is dominated by ordinary_checkable. The interesting cut is
        # the taxonomy of what STAYED unverifiable.
        stuck = [a for a in deep if P[a][P_UNVER3][0]["label"] == "UNVERIFIABLE"]
        tx_stuck = collections.Counter(P[a][P_UNVER3][0].get("taxon") for a in stuck)
        print("  taxonomy of what STAYED unverifiable:")
        for k, v in tx_stuck.most_common():
            print(f"    {str(k):<28} {v:>4}  {v/max(len(stuck),1)*100:>5.1f}%")

        # the folded estimator with a post-clustered interval of its own, so the
        # recommendation of which number to quote is not comparing an exact interval
        # against nothing
        fold = dict(prim)
        for a in deep:
            l2 = P[a][P_UNVER3][0]["label"]
            if l2 in ("TRUE", "FALSE"):
                fold[a] = l2
        pp2 = collections.defaultdict(lambda: [0, 0])
        for a, l in fold.items():
            if l == "UNVERIFIABLE":
                continue
            u = pp2[meta[a]["post_id"]]
            u[1] += 1
            u[0] += l == "FALSE"
        fu = [tuple(v) for v in pp2.values()]
        _, flo, fhi = cluster_boot(fu)
        fF = sum(u[0] for u in fu)
        fD = sum(u[1] for u in fu)
        fcp = clopper_pearson(fF, fD)
        print(f"  folded estimator {fF}/{fD} = {fF/fD:.4f}  Clopper-Pearson "
              f"[{fcp[0]:.4f}, {fcp[1]:.4f}]  post-clustered [{flo:.4f}, {fhi:.4f}]")
        du |= {"deep_label": dict(dl), "resolvability": dict(rt),
               "taxonomy": dict(tx), "taxonomy_stuck": dict(tx_stuck),
               "n_stuck": len(stuck),
               "folded_F": fF, "folded_dec": fD, "folded_eps": fF / fD,
               "folded_cp": list(fcp), "folded_clustered": [flo, fhi],
               "resolved_T": rT, "resolved_F": rF,
               "still_unver": still, "eps_folded": num / den,
               "band": [blo, bhi],
               "band_span_pp": (bhi - blo) * 100,
               "band_span_before_pp": ((F + U) / len(prim) - F / len(prim)) * 100}

    # ------------------------------------------------ 4. band tables
    def band_table(name, score, cuts, edges):
        """Per-band eps with real denominators, plus the urn population per band.

        The urn column is the WHOLE frame, not the sample, so `implied false` is
        the band rate applied to every claim in that band."""
        print(f"\n--- eps by s7 band, {name} ---")
        g = collections.defaultdict(collections.Counter)
        for a, l in prim.items():
            g[band_of(score[a], cuts)][l] += 1
        urn = collections.Counter(band_of(score[a], cuts) for a in score)
        urn_n = sum(urn.values())
        out, ks, ns = [], [], []
        for i in range(len(cuts) + 1):
            cc = g[i]
            d = cc["TRUE"] + cc["FALSE"]
            ks.append(cc["FALSE"])
            ns.append(d)
            lo, hi = clopper_pearson(cc["FALSE"], d)
            e = cc["FALSE"] / d if d else float("nan")
            out.append({"band": edges[i], "audited": sum(cc.values()),
                        "T": cc["TRUE"], "F": cc["FALSE"], "U": cc["UNVERIFIABLE"],
                        "decided": d, "eps": e, "cp": [lo, hi],
                        "urn_n": urn[i], "urn_pct": urn[i] / urn_n * 100,
                        "implied_false": (e * urn[i]) if d else 0.0})
            print(f"  {edges[i]:<18} n {sum(cc.values()):>4}  T {cc['TRUE']:>3} "
                  f"F {cc['FALSE']:>3} U {cc['UNVERIFIABLE']:>3}  dec {d:>3}  "
                  f"eps {e:.3f} [{lo:.3f},{hi:.3f}]  urn {urn[i]:>5,} "
                  f"({urn[i]/urn_n*100:>4.1f}%)  implied F {e*urn[i] if d else 0:>5.0f}")
        z, p = cochran_armitage(ks, ns)
        print(f"  Cochran-Armitage trend across bands: z={z:+.3f}, p={p:.4f}")
        # every two-way collapse, so the strongest split is not cherry-picked silently
        splits = []
        for c in range(1, len(ns)):
            lf, ld = sum(ks[:c]), sum(ns[:c])
            hf, hd = sum(ks[c:]), sum(ns[c:])
            if not ld or not hd:
                continue
            _, pf = stats.fisher_exact([[lf, ld - lf], [hf, hd - hf]],
                                       alternative="greater")
            splits.append({"cut_after": edges[c - 1], "low_F": lf, "low_dec": ld,
                           "hi_F": hf, "hi_dec": hd, "low_eps": lf / ld,
                           "hi_eps": hf / hd, "fisher_p": float(pf)})
            print(f"    split below {edges[c]:<18} {lf}/{ld} = {lf/ld:.3f} vs "
                  f"{hf}/{hd} = {hf/hd:.3f}   Fisher one-sided p={pf:.4f}")
        return {"name": name, "rows": out, "urn_n": urn_n,
                "ca_z": z, "ca_p": p, "splits": splits,
                "monotone": all(out[i]["eps"] >= out[i + 1]["eps"]
                                for i in range(len(out) - 1)
                                if out[i]["decided"] and out[i + 1]["decided"])}

    e2 = [f"[{a:g},{b:g})" for a, b in BANDS_2URN]
    bt2 = band_table("two-urn eps=0.05 weights", s2, [b for _, b in BANDS_2URN[:-1]], e2)
    eg = ([f"(-inf,{gcuts[0]:.2f})"]
          + [f"[{gcuts[i]:.2f},{gcuts[i+1]:.2f})" for i in range(len(gcuts) - 1)]
          + [f"[{gcuts[-1]:.2f},inf)"])
    btg = band_table("GOLD-fit weights, frame-percentile-matched cuts", sg, gcuts, eg)

    # ------------------------------------------------ 5. monotonicity
    print("\n--- does the score separate FALSE from TRUE? ---")
    sep = {}
    dec_a = [a for a in prim if prim[a] in ("TRUE", "FALSE")]
    for nm, sc in (("two-urn", s2), ("gold-fit", sg)):
        s = [sc[a] for a in dec_a]
        yv = [1 if prim[a] == "TRUE" else 0 for a in dec_a]
        pid = [meta[a]["post_id"] for a in dec_a]
        auc, pp = cluster_perm_rank(s, yv, pid)
        mw = stats.mannwhitneyu([sc[a] for a in dec_a if prim[a] == "TRUE"],
                                [sc[a] for a in dec_a if prim[a] == "FALSE"],
                                alternative="greater")
        print(f"  {nm:<9} AUC(TRUE over FALSE) {auc:.4f}   Mann-Whitney one-sided "
              f"p={mw.pvalue:.4f}   post-clustered permutation p={pp:.4f}")
        med = float(np.median([sc[a] for a in dec_a]))
        tab = [[0, 0], [0, 0]]
        for a in dec_a:
            tab[0 if sc[a] < med else 1][0 if prim[a] == "FALSE" else 1] += 1
        odds, pf = stats.fisher_exact(tab, alternative="greater")
        print(f"            median split at {med:+.3f}: below {tab[0]} above {tab[1]}"
              f"  Fisher exact one-sided p={pf:.4f}, OR={odds:.2f}")
        sep[nm] = {"auc": float(auc), "perm_p": float(pp),
                   "mw_p": float(mw.pvalue), "median": med,
                   "median_split": tab, "median_fisher_p": float(pf),
                   "median_or": float(odds)}

    # ------------------------------------------------ 6. other cuts
    for nm, kf in (("frame", lambda m: m["frame"]), ("topic", lambda m: m["topic"])):
        print(f"\nby {nm}:")
        g = collections.defaultdict(collections.Counter)
        for a, l in prim.items():
            g[kf(meta[a])][l] += 1
        for k in sorted(g, key=lambda k: -sum(g[k].values())):
            cc = g[k]
            d = cc["TRUE"] + cc["FALSE"]
            lo, hi = clopper_pearson(cc["FALSE"], d)
            print(f"   {str(k):<18} n {sum(cc.values()):>4}  T {cc['TRUE']:>3} "
                  f"F {cc['FALSE']:>3} U {cc['UNVERIFIABLE']:>3}  "
                  f"eps {cc['FALSE']/d if d else float('nan'):.3f} [{lo:.3f},{hi:.3f}]")

    print("\np1 UNVERIFIABLE by reason:")
    uw = collections.Counter(
        (P[a][P_PRIMARY][0].get("why_code") or "other") for a in unv)
    for k, v in uw.most_common():
        print(f"   {str(k):<22} {v:>4}  {v/max(len(unv),1)*100:>5.1f}% of unver")

    d = OUT / out_tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(
        {"tags": tags, "n": len(prim), "TRUE": T, "FALSE": F, "UNVERIFIABLE": U,
         "eps": eps1, "clopper_pearson": cp, "wilson": [wl, wh],
         "clustered": [clo, chi], "deff": deff, "n_posts": len(units),
         "gold_cuts": gcuts, "frame_n": len(s2),
         "gold_weights": gold_fit_weights(), "two_urn_weights": load_fit_weights(),
         "bound_all_true": F / len(prim), "bound_all_false": (F + U) / len(prim),
         "bands_two_urn": bt2, "bands_gold": btg, "separation": sep,
         "second_reads": sr, "deep_unver": du, "unver_why_p1": dict(uw),
         "labels": {a: {"label": prim[a], "s7": s2[a], "s7_gold": sg[a],
                        "post_id": meta[a]["post_id"], "topic": meta[a]["topic"],
                        "frame": meta[a]["frame"], "claim": meta[a]["claim"],
                        "passes": {k: v for k, v in P[a].items()}}
                    for a in prim}}, indent=1, default=str))
    print(f"\nwrote {d/'result.json'}", flush=True)


def followup(tags: list[str], seed: int = SEED, n_ctrl: int = 40) -> None:
    """Build the p2f / p2t / p3u packets from the p1 labels.

    Packets carry exactly what a p1 packet carries. The second reader is not told
    what p1 said, or that it is a re-read at all — otherwise a confirmation pass
    measures anchoring, not truth. Same reason the control claims are drawn from
    the p1-TRUEs without being marked as such.
    """
    P = _passes(tags)
    meta = {}
    for t in tags:
        for r in pl.read_parquet(OUT / t / "sample.parquet").iter_rows(named=True):
            meta[r["aid"]] = r
    prim = {a: v[P_PRIMARY][0]["label"] for a, v in P.items()
            if a in meta and v.get(P_PRIMARY)}
    done = {r: {a for a, v in P.items() if v.get(r)}
            for r in (P_FALSE2, P_TRUE2, P_UNVER3, "p2")}

    fal = sorted(a for a, l in prim.items() if l == "FALSE"
                 and a not in done[P_FALSE2] and a not in done["p2"])
    unv = sorted(a for a, l in prim.items() if l == "UNVERIFIABLE"
                 and a not in done[P_UNVER3])
    tru_pool = sorted(a for a, l in prim.items() if l == "TRUE"
                      and a not in done[P_TRUE2] and a not in done["p2"])
    rng = random.Random(seed + 1)
    tru = rng.sample(tru_pool, min(n_ctrl, len(tru_pool)))

    pf = post_fields()
    for role, aids in ((P_FALSE2, fal), (P_TRUE2, tru), (P_UNVER3, unv)):
        d = OUT / "main" / f"packets_{role}"
        d.mkdir(parents=True, exist_ok=True)
        for b in range(math.ceil(len(aids) / BATCH)):
            items = []
            for a in aids[b * BATCH:(b + 1) * BATCH]:
                m = meta[a]
                p = pf.get(m["post_id"], {})
                items.append({
                    "aid": a, "claim": m["claim"], "post_text": p.get("post_text"),
                    "handle": p.get("handle"), "posted_at": _iso(p.get("created_at")),
                    "is_quote_post": bool(p.get("is_quote")),
                    "quoted_handle": p.get("quoted_handle"),
                    "quoted_text": p.get("quoted_text"),
                    "post_has_image": int(p.get("n_images") or 0) > 0,
                    "post_has_video": int(p.get("n_videos") or 0) > 0,
                    "is_reply": bool(p.get("is_reply")),
                    "retweeted_by": p.get("retweeted_by")})
            (d / f"batch_{b:02d}.json").write_text(json.dumps(items, indent=1))
        (OUT / "main" / f"labels").mkdir(parents=True, exist_ok=True)
        print(f"{role}: {len(aids)} claims -> {math.ceil(len(aids)/BATCH)} packets in {d}",
              flush=True)
    print(f"(control drawn from {len(tru_pool)} eligible p1-TRUEs, seed {seed+1})", flush=True)

if __name__ == "__main__":
    main()
