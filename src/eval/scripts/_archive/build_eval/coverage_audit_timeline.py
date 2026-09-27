"""Pre-scorer coverage on the TIMELINE pool — what we lose before anything is scored.

Same question as `coverage_audit.py` (which measures the community-noted acquitted pool),
asked of the population we actually deploy on: posts an account's For You feed served.
Denominator is POSTS, not claims.

Two things differ from the noted-post version, and both matter:

  NO NOTES. A noted post carries a human's word that a contestable claim exists. A timeline
  post carries nothing, so the ground truth has to come from the blind auditor instead. That
  makes the auditor load-bearing here in a way it is not on the CN pool, and it is why every
  miss number below is an estimate with the auditor's judgement inside it.

  AN EXTRA GATE. The feed passes a STRICT qualify screen (public affairs AND stakes) that
  the acquitted pool never sees, and it is by far the largest loss. That screen is a SCOPE
  gate, not a claim-existence gate — "I had a sandwich at 3pm" is checkable and correctly
  dropped. So the auditor answers two independent questions, and a drop only counts as a
  miss when a post has a checkable claim AND is in scope.

Strata, all post-level:
  prefilter      dropped before screening (non en/fr, <=15 chars, ad)
  screen_reject  failed the STRICT qualify screen              <- the big one
  no_claims      qualified, extraction emitted nothing
  not_eligible   claims, but none verify-eligible
  reached_verify made it through — audited for the WRONG-claim case (we extracted
                 something, but not what a blind reader saw)

  uv run python -m eval.scripts.build_eval.coverage_audit_timeline --funnel
  uv run python -m eval.scripts.build_eval.coverage_audit_timeline --audit --n 350
  uv run python -m eval.scripts.build_eval.coverage_audit_timeline --report

Reads eval/data/urn_runs/general_pool/screen_strict.jsonl READ-ONLY; writes only to
eval/data/coverage_audit_timeline/. No extraction or gate prompt is touched.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.scripts.build_eval.c2_audit import _chat          # noqa: E402
from eval.scripts.build_eval.coverage_audit import wilson    # noqa: E402

CORP = SRC / "eval/data/tweet_corpus"
POOL = CORP / "general_pool"
RUNS = SRC / "eval/data/urn_runs/general_pool"     # READ ONLY
OUT = SRC / "eval/data/coverage_audit_timeline"
SEED = 20260828
VOICES = ("quote", "user", "outlet")

# Blind: the post and nothing else. Two independent questions, because the screen this is
# auditing is a SCOPE gate — a checkable claim that nobody outside the author's circle has a
# stake in is a CORRECT drop, and conflating the two would invent misses that aren't there.
AUDIT_SYS = """\
You are shown one social-media post. Answer two independent questions about it.

FIRST: does the post contain a factual claim that could in principle be checked against
evidence? A checkable claim asserts something about the world that could be shown true or
false — an event, an action, a number, a ruling, a quantity, a causal effect, or a report
that a named person or organization said something specific. It need not be important,
well-known, or disputed, and it need not be the post's main point.

Not checkable: pure opinion or value judgment; speculation about the future with no factual
anchor; a joke, meme or obvious satire asserting nothing; a question, greeting, reaction or
insult; promotion asserting nothing ("out now", "link in bio"); a bare link or hashtag.

SECOND, and separately: does the post bear on public affairs or on something with real
stakes — health, money, safety, law, elections, institutions, science, public figures acting
in public roles? A personal anecdote, a private matter, a joke, an advertisement, a
reaction, or chatter that matters only to the author and their circle is OUT of scope even
when it contains a perfectly checkable fact. Judge stakes, not topic labels.

Answer the two independently: a post can carry a checkable claim and still be out of scope,
and vice versa.

Judge only what the POST TEXT asserts. If the checkable content lives somewhere you cannot
see — an image, a video, a linked article, a quoted post, an unresolved "this" — record that
in `locus`, and set `has_claim` false unless the text still asserts something checkable on
its own.

Reply with JSON only:
{"has_claim": true|false,
 "claim": "<the single most checkable claim, as a standalone sentence>" or null,
 "in_scope": true|false,
 "locus": "text" | "image_or_video" | "linked_article" | "quoted_post" | "unclear_referent" | "none",
 "post_kind": "factual_report" | "attribution" | "opinion" | "joke_satire" | "promo" | "question_reaction" | "personal_anecdote" | "other",
 "why": "<one short sentence>"}
"""

ADJ_SYS = """\
You compare a reference claim against a list of claims a pipeline extracted from the same
post. Decide whether the pipeline recovered the reference claim.

Recovered means one of the extracted claims asserts the same proposition as the reference.
Wording may differ freely. A claim that is narrower, broader, or about a different aspect of
the post is NOT a recovery. A claim that carries the reference's substance but drops a
qualifier still counts.

Reply with JSON only:
{"recovered": true|false, "idx": <index> or null, "why": "<one short sentence>"}
"""

LABELS = {"prefilter": "dropped before screening (non en/fr, short, ad)",
          "screen_reject": "failed the STRICT qualify screen",
          "no_claims": "qualified, extraction emitted nothing",
          "not_eligible": "claims, none verify-eligible",
          "reached_verify": "reached verify (audited for the wrong-claim case)"}


def load():
    """Raw feed captures, screen verdicts, claims and manifests. All read-only."""
    raw, seen = [], set()
    for fp in sorted(glob.glob(str(POOL / "x_capture_*.ndjson"))):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not d.get("id") or d["id"] in seen:
                continue
            seen.add(d["id"])
            raw.append(d)
    feed = [r for r in raw if r.get("operation") == "HomeTimeline" and r.get("account")]
    scr = {}
    for line in open(RUNS / "screen_strict.jsonl"):
        d = json.loads(line)
        scr[d["id"]] = d
    vi = pl.concat([pl.read_parquet(CORP / f"urn_{v}_verify_input.parquet")
                    for v in VOICES if (CORP / f"urn_{v}_verify_input.parquet").exists()],
                   how="diagonal_relaxed")
    man = pl.concat([pl.read_parquet(CORP / f"urn_{v}_verify_input_posts_manifest.parquet")
                     for v in VOICES
                     if (CORP / f"urn_{v}_verify_input_posts_manifest.parquet").exists()],
                    how="diagonal_relaxed")
    return feed, scr, vi, man


def strata(feed, scr, vi, man):
    """Post-level drop strata. Every feed post lands in exactly one."""
    elig = {r["id"] for r in feed
            if r.get("lang") in ("en", "fr") and len(r.get("full_text") or "") > 15
            and not r.get("promoted")}
    qual = {i for i in elig if scr.get(i, {}).get("qualifies")}
    st = {r["post_id"]: r["status"] for r in man.iter_rows(named=True)}
    out = collections.defaultdict(list)
    for r in feed:
        i = r["id"]
        if i not in elig:
            out["prefilter"].append(r)
        elif i not in qual:
            out["screen_reject"].append(r)
        else:
            s = st.get(i)
            if s == "skip_no_claims":
                out["no_claims"].append(r)
            elif s == "skip_none_checkworthy":
                out["not_eligible"].append(r)
            elif s == "verify":
                out["reached_verify"].append(r)
            else:
                out["unchained"].append(r)
    return out


def funnel(write: bool = True) -> dict:
    feed, scr, vi, man = load()
    S = strata(feed, scr, vi, man)
    N = len(feed)
    elig = N - len(S["prefilter"])
    qual = elig - len(S["screen_reject"])
    # vi spans BOTH frames (feed + search_june); this funnel is feed-only, so every claim
    # count has to be restricted to the feed posts or it silently exceeds its own numerator.
    feed_ids = [r["id"] for r in feed]
    vif = vi.filter(pl.col("post_id").is_in(feed_ids))
    f = {"posts_served": N,
         "passed_prefilter": elig,
         "passed_screen": qual,
         "yielded_claim": qual - len(S["no_claims"]) - len(S["unchained"]),
         "has_checkworthy": vif.filter("checkworthy")["post_id"].n_unique(),
         "reaches_verify": len(S["reached_verify"])}
    f["ci"] = {k: wilson(v, N) for k, v in f.items() if isinstance(v, int)}
    f["strata"] = {k: len(v) for k, v in S.items()}
    f["screen_categories"] = dict(collections.Counter(
        scr[r["id"]]["category"] for r in S["screen_reject"]).most_common())
    cwne = vif.filter(pl.col("checkworthy") & ~pl.col("verify_eligible"))
    f["topic_excluded_claims"] = dict(
        collections.Counter(cwne["topic"].to_list()).most_common())

    print(f"\n{'stage':<42}{'posts':>7}{'share':>9}   95% CI")
    for k in ("posts_served", "passed_prefilter", "passed_screen", "yielded_claim",
              "has_checkworthy", "reaches_verify"):
        lo, hi = f["ci"][k]
        print(f"{k:<42}{f[k]:>7}{f[k]/N*100:>8.1f}%   [{lo*100:.1f}, {hi*100:.1f}]")
    print("\ndrop strata:")
    for k, v in sorted(f["strata"].items(), key=lambda x: -x[1]):
        print(f"  {LABELS.get(k,k):<48}{v:>6}{v/N*100:>7.1f}%")
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "funnel.json").write_text(json.dumps(f, indent=1))
    return f


def sample(n_per: int) -> pl.DataFrame:
    feed, scr, vi, man = load()
    S = strata(feed, scr, vi, man)
    rng = random.Random(SEED)
    rows = []
    for name in ("prefilter", "screen_reject", "no_claims", "not_eligible", "reached_verify"):
        posts = sorted(S[name], key=lambda r: r["id"])
        rng.shuffle(posts)
        for r in posts[:n_per]:
            rows.append({"post_id": r["id"], "stratum": name, "n_stratum": len(S[name]),
                         "post_text": r.get("full_text") or "",
                         "lang": r.get("lang") or "", "handle": r.get("screen_name") or ""})
        print(f"  {name:<16}{len(S[name]):>6} posts -> sampled {min(n_per, len(posts))}")
    df = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT / "sample.parquet")
    return df


def is_miss(b: dict, stratum: str) -> bool:
    """One definition, used by every number in the report.

    A drop is wrong when a blind reader finds a checkable claim that is ALSO in scope, and
    the pipeline did not deliver it. `in_scope` is what keeps the screen honest: it is a
    scope gate, so dropping a checkable but stakes-free post is correct behaviour. For posts
    that DID reach verify, the miss is the wrong-claim case — we extracted, but not this.
    """
    if not (b["blind_has_claim"] and b.get("in_scope")):
        return False
    return not (stratum == "reached_verify" and b.get("recovered"))


def audit(n_per: int, workers: int, cap: float) -> None:
    df = sample(n_per)
    _, _, vi, _ = load()
    extracted = {p: g["claim"].to_list() for (p,), g in vi.group_by("post_id")}
    out = OUT / "blind.jsonl"
    seen = {json.loads(l)["post_id"] for l in out.open()} if out.exists() else set()
    todo = [r for r in df.iter_rows(named=True) if r["post_id"] not in seen]
    print(f"\nblind audit: {len(todo)}/{df.height} posts | cap ${cap} | {workers} workers",
          flush=True)

    lock, state = threading.Lock(), {"cost": 0.0, "done": 0, "stop": False}
    t0 = time.time()
    fh = out.open("a")

    def one(r):
        if state["stop"]:
            return
        o = _chat(AUDIT_SYS, f"POST:\n{r['post_text']}")     # BLIND: post text only
        cost = o.pop("_cost", 0.0)
        rec = {"post_id": r["post_id"], "stratum": r["stratum"], "lang": r["lang"],
               "blind_has_claim": bool(o.get("has_claim")), "blind_claim": o.get("claim"),
               "in_scope": bool(o.get("in_scope")), "locus": o.get("locus"),
               "post_kind": o.get("post_kind"), "why": o.get("why")}
        ex = extracted.get(r["post_id"]) or []
        if rec["blind_has_claim"] and ex:
            numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(ex))
            a = _chat(ADJ_SYS, f"REFERENCE CLAIM:\n{rec['blind_claim']}\n\n"
                               f"EXTRACTED CLAIMS:\n{numbered}")
            cost += a.pop("_cost", 0.0)
            rec["recovered"] = bool(a.get("recovered"))
            rec["recovered_why"] = a.get("why")
        with lock:
            state["cost"] += cost
            state["done"] += 1
            fh.write(json.dumps(rec) + "\n")
            n, el = state["done"], (time.time() - t0) / 60
            if n % 50 == 0 or n == len(todo):
                fh.flush()
                proj = state["cost"] / n * len(todo)
                print(f"  {n}/{len(todo)} | ${state['cost']:.3f} | proj ${proj:.2f} | "
                      f"{el:.1f}m | ETA {el/n*(len(todo)-n):.0f}m", flush=True)
                if proj > cap and n >= 50:
                    print(f"  BUDGET ABORT: proj ${proj:.2f} > cap ${cap}", flush=True)
                    state["stop"] = True

    with ThreadPoolExecutor(max_workers=workers) as ex_:
        list(ex_.map(one, todo))
    fh.close()
    print(f"audit done {state['done']} | ${state['cost']:.4f} | "
          f"{(time.time()-t0)/60:.1f}m", flush=True)
    report()


def decompose(blind: list[dict], sizes: dict) -> dict:
    """Split estimated misses by which stage dropped the post, policy separated from defect.

    Two of these are deliberate scope decisions rather than failures, and lumping them in
    with the rest triples the apparent hole. The prefilter's misses are ~entirely non-en/fr
    posts (94% Spanish in the sample) — that is the language scope, not a bug. The
    eligibility gate's are the topic exclusion (entertainment / sports / lifestyle).
    """
    out = {}
    for s, key in (("prefilter", "language_policy"), ("screen_reject", "scope_gate"),
                   ("no_claims", "extraction_empty"), ("not_eligible", "topic_gate"),
                   ("reached_verify", "extracted_wrong_claim")):
        sub = [b for b in blind if b["stratum"] == s]
        if sub:
            out[key] = sum(1 for b in sub if is_miss(b, s)) / len(sub) * sizes[s]
    return out


def report() -> None:
    f = json.loads((OUT / "funnel.json").read_text())
    blind = [json.loads(l) for l in (OUT / "blind.jsonl").open()]
    N, sizes = f["posts_served"], f["strata"]

    print("\n=== blind audit: is the drop wrong? ===")
    print(f"{'stratum':<16}{'posts':>7}{'aud':>5}{'claim':>8}{'+scope':>8}"
          f"{'  95% CI':>16}{'est misses':>12}")
    tot = 0.0
    for s in ("prefilter", "screen_reject", "no_claims", "not_eligible", "reached_verify"):
        sub = [b for b in blind if b["stratum"] == s]
        if not sub:
            continue
        hc = sum(1 for b in sub if b["blind_has_claim"])
        k = sum(1 for b in sub if is_miss(b, s))
        lo, hi = wilson(k, len(sub))
        est = k / len(sub) * sizes[s]
        tot += est
        print(f"{s:<16}{sizes[s]:>7,}{len(sub):>5}{hc/len(sub)*100:>7.1f}%"
              f"{k/len(sub)*100:>7.1f}%   [{lo*100:.1f}, {hi*100:.1f}]{est:>9,.0f}")
    print(f"\nestimated silent misses: {tot:,.0f} posts = {tot/N*100:.1f}% of "
          f"{N:,} served | reaching verify today: {f['reaches_verify']/N*100:.1f}%")

    d = decompose(blind, sizes)
    print("\n=== which stage dropped it ===")
    order = [("language_policy", "language scope, non-en/fr    (deliberate)"),
             ("scope_gate", "STRICT qualify screen wrong (arguable)"),
             ("topic_gate", "topic exclusion             (deliberate)"),
             ("extracted_wrong_claim", "extracted the wrong claim   (miss)"),
             ("extraction_empty", "extraction emitted nothing  (miss)")]
    for k, lab in order:
        v = d.get(k, 0.0)
        print(f"  {lab:<44}{v:>6,.0f}{v/N*100:>7.1f}%")
    hard = d.get("extraction_empty", 0) + d.get("extracted_wrong_claim", 0)
    nonpol = tot - d.get("language_policy", 0) - d.get("topic_gate", 0)
    print(f"  {'-'*44}")
    print(f"  {'hole excluding deliberate scope policy':<44}{nonpol:>6,.0f}{nonpol/N*100:>7.1f}%")
    print(f"  {'hole from extraction alone':<44}{hard:>6,.0f}{hard/N*100:>7.1f}%")

    for field, title in (("post_kind", "post type"), ("locus", "where the claim lives")):
        w = collections.Counter()
        for s in sizes:
            sub = [b for b in blind if b["stratum"] == s]
            if not sub:
                continue
            per = sizes[s] / len(sub)
            for b in sub:
                if is_miss(b, s):
                    w[b[field]] += per
        t = sum(w.values()) or 1
        print(f"\n=== estimated misses by {title} (population-weighted) ===")
        for k, v in w.most_common():
            print(f"  {str(k):<20}{v:>8,.0f}  {v/t*100:>5.1f}%")

    print("\n=== screen rejects: claim x scope (the gate's own job) ===")
    sub = [b for b in blind if b["stratum"] == "screen_reject"]
    if sub:
        g = collections.Counter((b["blind_has_claim"], b["in_scope"]) for b in sub)
        for (hc, sc), v in sorted(g.items(), key=lambda x: -x[1]):
            print(f"  claim={str(hc):<5} scope={str(sc):<5} {v:>4}  {v/len(sub)*100:>5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funnel", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cap", type=float, default=2.00)
    a = ap.parse_args()
    if a.funnel:
        funnel()
    if a.audit:
        audit(a.n, a.workers, a.cap)
    if a.report:
        report()


if __name__ == "__main__":
    main()
