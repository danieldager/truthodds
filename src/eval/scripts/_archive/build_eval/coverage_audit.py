"""Pre-scorer coverage: what extraction and the scope gate drop before anything is scored.

Every performance number we publish is recall over claims that SURVIVED to scoring, so a
claim we never extracted is a miss the evaluation cannot see. This measures that hole with
a post-level denominator.

Ground truth is community-noted posts: a post with a note contains a contestable claim by
construction, because a human wrote a note about it. The instrument is the run
`acquitted_yield.py` already did over 10,500 noted L3 posts (extract -> normalize ->
build_verify_input -> screen, then MATCH_SYS against the note). This script reads those
outputs — never writes to eval/data/urn_runs — and adds the part that was missing: a BLIND
audit of the posts we dropped.

Blind means the auditor sees the post text and nothing else. Not the note, not our
extracted claims, not which stratum the post came from. That is the only way to separate a
correct drop (the note was about an image, or about the account, or there was no factual
claim) from a silent miss.

Three strata, sampled independently:
  no_claims    extraction produced nothing            (manifest skip_no_claims)
  not_eligible claims, but none verify-eligible       (manifest skip_none_checkworthy)
  no_match     claims, but MATCH found no note target (the interesting failures — we did
               extract something and it was the wrong thing)

  uv run python -m eval.scripts.build_eval.coverage_audit --funnel          # $0
  uv run python -m eval.scripts.build_eval.coverage_audit --audit --n 400   # paid
  uv run python -m eval.scripts.build_eval.coverage_audit --report

Output: eval/data/coverage_audit/{sample.parquet,blind.jsonl,funnel.json}
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.scripts.build_eval.c2_audit import _chat  # noqa: E402

CORP = SRC / "eval/data/tweet_corpus"
YIELD = SRC / "eval/data/urn_runs/acquitted_yield"      # READ ONLY (frozen elsewhere)
OUT = SRC / "eval/data/coverage_audit"
SEED = 20260828

# Blind. The post and nothing else: no note, no extracted claims, no stratum label.
AUDIT_SYS = """\
You are shown one social-media post. Decide whether it contains a factual claim that could
in principle be checked against evidence, and if so, state it.

A checkable claim asserts something about the world that could be shown true or false:
an event, an action, a number, a ruling, a quantity, a causal effect, or a report that a
named person or organization said something specific. It does not have to be important,
well-known, or currently disputed. It does not have to be the post's main point.

These are NOT checkable claims:
- pure opinion, preference, or value judgment ("this policy is evil", "best film of the year")
- prediction or speculation about the future with no factual anchor
- a joke, meme, or obvious satire with no asserted fact
- a question, a greeting, a reaction, an insult with no factual content
- promotion with no factual assertion ("link in bio", "out now")
- a bare link, handle, or hashtag with no proposition

Judge only what the POST TEXT asserts. If the post refers to something you cannot see —
an image, a video, a linked article, a quoted post, "this" with no referent — and the
checkable content lives THERE rather than in the text, say so via `locus`, and set
`has_claim` false unless the text itself still asserts something checkable on its own.

Reply with JSON only:
{"has_claim": true|false,
 "claim": "<the single most checkable claim, as a standalone sentence>" or null,
 "locus": "text" | "image_or_video" | "linked_article" | "quoted_post" | "unclear_referent" | "none",
 "post_kind": "factual_report" | "attribution" | "opinion" | "joke_satire" | "promo" | "question_reaction" | "personal_anecdote" | "other",
 "why": "<one short sentence>"}

`claim` must be readable on its own, with pronouns and demonstratives resolved from the
post where possible. If has_claim is false, claim is null.
"""

# Second pass, NOT blind: did our chain recover what the blind auditor found?
ADJ_SYS = """\
You compare a reference claim against a list of claims a pipeline extracted from the same
post. Decide whether the pipeline recovered the reference claim.

Recovered means one of the extracted claims asserts the same proposition as the reference.
Wording may differ freely. A claim that is narrower, broader, or about a different aspect
of the post is NOT a recovery. A claim that captures the reference's substance but drops a
qualifier still counts.

Reply with JSON only:
{"recovered": true|false, "idx": <index of the matching extracted claim> or null,
 "why": "<one short sentence>"}
"""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _tags() -> list[str]:
    return sorted({p.name.split("_verify_input")[0]
                   for p in CORP.glob("acqy_b*_verify_input.parquet")})


def load() -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    tags = _tags()
    man = pl.concat([pl.read_parquet(CORP / f"{t}_verify_input_posts_manifest.parquet")
                     for t in tags], how="diagonal_relaxed")
    vi = pl.concat([pl.read_parquet(CORP / f"{t}_verify_input.parquet")
                    for t in tags], how="diagonal_relaxed")
    match = {json.loads(l)["post_id"]: json.loads(l)
             for l in (YIELD / "match.jsonl").open()}
    return man, vi, match


def funnel(write: bool = True) -> dict:
    man, vi, match = load()
    n = man.height
    st = collections.Counter(man["status"].to_list())
    cw_ids = set(vi.filter("checkworthy")["claim_id"].to_list())
    ve_ids = set(vi.filter("verify_eligible")["claim_id"].to_list())
    matched = sum(1 for m in match.values() if m.get("target_claim"))
    f = {
        "posts_in": n,
        "posts_with_claim": n - st["skip_no_claims"],
        "posts_with_checkworthy": vi.filter("checkworthy")["post_id"].n_unique(),
        "posts_reaching_verify": st["verify"],
        "target_recovered": matched,
        "target_recovered_checkworthy": sum(
            1 for m in match.values() if m.get("target_claim_id") in cw_ids),
        "target_recovered_eligible": sum(
            1 for m in match.values() if m.get("target_claim_id") in ve_ids),
    }
    f["loss"] = {
        "extraction_produced_nothing": st["skip_no_claims"],
        "claims_but_none_eligible": st["skip_none_checkworthy"],
        "extracted_but_target_not_found": n - st["skip_no_claims"] - matched,
    }
    f["miss_kind"] = dict(collections.Counter(
        m.get("miss_kind") for m in match.values() if not m.get("target_claim")))
    f["no_claim_reason"] = dict(collections.Counter(
        man.filter(pl.col("status") == "skip_no_claims")["no_claim_reason"].to_list()))
    f["ci"] = {k: wilson(v, n) for k, v in f.items() if isinstance(v, int)}

    print(f"\n{'stage':<42}{'posts':>7}{'share':>9}   95% CI")
    for k in ("posts_in", "posts_with_claim", "posts_with_checkworthy",
              "posts_reaching_verify", "target_recovered",
              "target_recovered_checkworthy", "target_recovered_eligible"):
        lo, hi = f["ci"][k]
        print(f"{k:<42}{f[k]:>7}{f[k]/n*100:>8.1f}%   [{lo*100:.1f}, {hi*100:.1f}]")
    print("\nloss:")
    for k, v in f["loss"].items():
        print(f"  {k:<38}{v:>7}{v/n*100:>8.1f}%")
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "funnel.json").write_text(json.dumps(f, indent=1))
    return f


def sample(n_per: int) -> pl.DataFrame:
    """Independent samples from the three drop strata. Seeded, so it extends cleanly."""
    man, vi, match = load()
    posts_with_claims = set(vi["post_id"].to_list())
    matched = {p for p, m in match.items() if m.get("target_claim")}
    strata = {
        "no_claims": man.filter(pl.col("status") == "skip_no_claims")["post_id"].to_list(),
        "not_eligible": man.filter(pl.col("status") == "skip_none_checkworthy")["post_id"].to_list(),
        "no_match": [p for p in posts_with_claims if p not in matched],
    }
    rng = random.Random(SEED)
    rows = []
    for name, ids in strata.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        for p in ids[:n_per]:
            rows.append({"post_id": p, "stratum": name, "n_stratum": len(ids),
                         "post_text": match[p]["tweet"], "note": match[p]["note"],
                         "n_claims": match[p]["n_claims"]})
        print(f"  {name:<14} {len(ids):>6} posts -> sampled {min(n_per, len(ids))}")
    df = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT / "sample.parquet")
    return df


def audit(n_per: int, workers: int, cap: float) -> None:
    df = sample(n_per)
    _, vi, _ = load()
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
        # BLIND: post text only. No note, no stratum, no extracted claims.
        o = _chat(AUDIT_SYS, f"POST:\n{r['post_text']}")
        cost = o.pop("_cost", 0.0)
        rec = {"post_id": r["post_id"], "stratum": r["stratum"],
               "blind_has_claim": bool(o.get("has_claim")), "blind_claim": o.get("claim"),
               "locus": o.get("locus"), "post_kind": o.get("post_kind"),
               "why": o.get("why"), "n_claims": r["n_claims"]}
        # Second pass only where it can say something: we extracted claims AND the blind
        # auditor found one. Did our chain recover what a blind reader saw?
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


def is_miss(b: dict, stratum: str) -> bool:
    """Was this drop wrong? One definition, used by every number in the report.

    A blind reader found a checkable claim in the post, and the pipeline did not deliver
    it. For `no_claims` and `not_eligible` the pipeline delivered nothing scoreable at all
    (extraction empty / gate ruled every claim ineligible), so a blind claim is enough. For
    `no_match` we DID extract claims, so the drop is only wrong if the blind claim is not
    among them.
    """
    if not b["blind_has_claim"]:
        return False
    return not (stratum == "no_match" and b.get("recovered"))


def decompose(blind: list[dict], sizes: dict) -> dict:
    """Split the estimated misses by WHICH stage dropped the post.

    `not_eligible` is two different things wearing one manifest status, and they carry
    opposite implications. verify_eligible = checkworthy AND topic not in
    {entertainment, sports, lifestyle} (build_verify_input.py:268), so a post lands there
    either because every claim was ruled not checkworthy, or because the claims were
    checkworthy but off-topic by deliberate scope policy. The second is not a bug.
    """
    _, vi, _ = load()
    any_cw = {p: bool(g["checkworthy"].any()) for (p,), g in vi.group_by("post_id")}
    out = {}
    for s in ("no_claims", "not_eligible", "no_match"):
        sub = [b for b in blind if b["stratum"] == s]
        if not sub:
            continue
        per = sizes[s] / len(sub)
        if s == "not_eligible":
            m = [b for b in sub if is_miss(b, s)]
            out["topic_policy"] = sum(per for b in m if any_cw.get(b["post_id"]))
            out["checkworthiness_gate"] = sum(per for b in m if not any_cw.get(b["post_id"]))
        else:
            key = {"no_claims": "extraction_empty", "no_match": "extracted_wrong_claim"}[s]
            out[key] = sum(per for b in sub if is_miss(b, s))
    return out


def report() -> None:
    f = json.loads((OUT / "funnel.json").read_text()) if (OUT / "funnel.json").exists() \
        else funnel(write=False)
    blind = [json.loads(l) for l in (OUT / "blind.jsonl").open()]
    n_total = f["posts_in"]
    sizes = {"no_claims": f["loss"]["extraction_produced_nothing"],
             "not_eligible": f["loss"]["claims_but_none_eligible"],
             "no_match": f["loss"]["extracted_but_target_not_found"]}

    print("\n=== blind audit: is the drop correct? ===")
    print(f"{'stratum':<14}{'n':>5}{'has claim':>11}{'95% CI':>16}"
          f"{'silent misses (est)':>22}")
    tot_miss = 0.0
    for s in ("no_claims", "not_eligible", "no_match"):
        sub = [b for b in blind if b["stratum"] == s]
        if not sub:
            continue
        k = sum(1 for b in sub if is_miss(b, s))
        lo, hi = wilson(k, len(sub))
        est = k / len(sub) * sizes[s]
        tot_miss += est
        print(f"{s:<14}{len(sub):>5}{k/len(sub)*100:>10.1f}%"
              f"   [{lo*100:.1f}, {hi*100:.1f}]"
              f"{est:>12,.0f} of {sizes[s]:,}")
    print(f"\nestimated silent misses: {tot_miss:,.0f} posts = "
          f"{tot_miss/n_total*100:.1f}% of the {n_total:,} noted posts")
    print(f"measured end-to-end recovery: "
          f"{f['target_recovered_eligible']/n_total*100:.1f}%")

    d = decompose(blind, sizes)
    print("\n=== which stage dropped it ===")
    order = [("topic_policy", "topic excluded by scope policy  (deliberate)"),
             ("checkworthiness_gate", "checkworthiness gate            (arguable)"),
             ("extraction_empty", "extraction produced nothing     (miss)"),
             ("extracted_wrong_claim", "extracted the wrong claim       (miss)")]
    for k, lab in order:
        v = d.get(k, 0.0)
        print(f"  {lab:<46}{v:>7,.0f}{v/n_total*100:>7.1f}%")
    bug = sum(d.get(k, 0.0) for k in
              ("checkworthiness_gate", "extraction_empty", "extracted_wrong_claim"))
    hard = sum(d.get(k, 0.0) for k in ("extraction_empty", "extracted_wrong_claim"))
    print(f"  {'-'*46}")
    print(f"  {'hole excluding deliberate topic policy':<46}{bug:>7,.0f}{bug/n_total*100:>7.1f}%")
    print(f"  {'hole from extraction alone':<46}{hard:>7,.0f}{hard/n_total*100:>7.1f}%")

    # Breakdowns use the SAME per-stratum miss definition as the estimate above, and are
    # population-weighted: the three strata were sampled equally but are 4,033 / 2,498 /
    # 1,706 posts, so pooling the raw counts would over-represent no_match ~2.4x.
    for field, title in (("post_kind", "post type"), ("locus", "where the claim lives")):
        w = collections.Counter()
        for s in ("no_claims", "not_eligible", "no_match"):
            sub = [b for b in blind if b["stratum"] == s]
            if not sub:
                continue
            per = sizes[s] / len(sub)
            for b in sub:
                if is_miss(b, s):
                    w[b[field]] += per
        tot = sum(w.values()) or 1
        print(f"\n=== estimated misses by {title} (population-weighted) ===")
        for k, v in w.most_common():
            print(f"  {str(k):<20}{v:>8,.0f}  {v/tot*100:>5.1f}%")

    print("\n=== per-stratum composition of the misses ===")
    for s in ("no_claims", "not_eligible", "no_match"):
        sub = [b for b in blind if b["stratum"] == s]
        if not sub:
            continue
        m = [b for b in sub if is_miss(b, s)]
        top = collections.Counter(b["post_kind"] for b in m).most_common(4)
        print(f"  {s:<14} est {len(m)/len(sub)*sizes[s]:>6,.0f} misses  {top}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--funnel", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n", type=int, default=400, help="posts sampled per stratum")
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
