"""Offline paired A/B of a READ polarity rule, on saved sentences. No retrieval.

The bug (Daniel, 2026-08-20): when a claim's predicate is itself a failure,
rejection or negation, the reader matches on VALENCE rather than proposition --
documents reporting the failure read as negative-toward-the-claim, so
"the document reports something bad happening" collapses into "the document
contradicts the claim". The war-powers case scored -12.55 on eight refuting
voices, every one of which confirms the claim.

READ_SYS_V5 has rules for adjacency, circulation, opinion and dates. It has no
polarity rule. The polarity anchoring exists in the verify loop's Likert
("explicit polarity: a strong refutation of a negation/hoax = FALSE", src/CLAUDE.md)
and was never carried into the read step.

    uv run python -m eval.scripts.build_eval.read_polarity_ab --case 2082934283052691946:2
    uv run python -m eval.scripts.build_eval.read_polarity_ab --probe e1 --budget 1.5

MANDATORY protocol (ROADMAP): every past read-prompt edit looked like a fix in
aggregate flag counts -- v5.2's "refutes doubled" WAS the bug. So nothing here
reports an aggregate. Every result is split FOUR ways: gold label x negation vs
affirmative. The fix is good only if refute flags fall on gold-TRUE negation
claims WITHOUT falling on gold-FALSE ones.

Reconstruction is exact: `prep.regions == 1` for 39,368/39,368 E1 docs and
38,910/38,910 E2 docs, so each saved `sent_ids`/`sents` pair IS the single region
the model was shown, and `claim_block` is rebuilt verbatim from evidence_urn_run.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.scripts.build_eval.evidence_urn_run import llm
from eval.scripts.build_eval.read_v5_prompts import READ_SYS_MODE as READ_SYS_BASE

E1 = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
E2 = Path("eval/data/urn_runs/e2_tweets/results-00.jsonl")
HEADLINE = Path("eval/data/urn_runs/e1_ctx/headline_metrics.json")

# Principle only -- no worked example, nothing naming a case from the corpus.
# Placed immediately after the exact-proposition rule it specialises.
POLARITY_RULE = """- Match the claim's polarity, not the document's tone. When the claim
  asserts that something failed, was rejected, did not happen, or is not the case, a
  document reporting that failure or that absence STATES the claim; it does not
  contradict it. Contradiction requires the document to assert the opposite outcome —
  that the thing succeeded, was accepted, or did happen. Before choosing a refuting
  flag, ask what the document would have to say for the claim to be FALSE, and check
  whether it says that."""

ANCHOR = "- Precedence: sentences that explicitly address the claim set the direction."
assert ANCHOR in READ_SYS_BASE, "anchor rule moved; re-point the insertion"
READ_SYS_POL = READ_SYS_BASE.replace(ANCHOR, POLARITY_RULE + "\n" + ANCHOR)

ARMS = {"base": READ_SYS_BASE, "pol": READ_SYS_POL}

# Claim-side normalization (2026-08-21): the war-powers paraphrase test showed the
# inversion is CLAIM-side -- the unchanged reader scores the same documents +1.17
# when the claim reads "the Senate rejected X" and -12.55 when it reads "X failed
# to pass the Senate". So the fix candidate is a rewrite pass at normalization,
# not a reader rule. Abstract principles only -- no dataset-derived examples.
NORM_SYS = (
    "You rewrite factual claims about outcomes so that the decision-maker is the "
    "grammatical subject, without changing what the claim asserts.\n\n"
    "Return JSON only: {\"rewritten\": \"<claim>\", \"changed\": true|false}\n\n"
    "When the claim says something failed, was defeated, did not happen or was "
    "not achieved, identify WHO or WHAT produced that outcome and restate the "
    "claim as that actor performing its decisive act: the body that voted it "
    "down, the official who refused it, the court that struck it down. The "
    "rewritten claim must name the same outcome as a completed action BY the "
    "deciding actor, not as a fate suffered by the subject. Keep every entity, "
    "date, place, number and qualifier exactly as asserted. Never add "
    "information, never drop a hedge, never strengthen or weaken the claim. If "
    "no deciding actor is identifiable, or the predicate is already an action by "
    "its agent, return the claim unchanged with changed=false."
)

_norm_cache: dict[str, str] = {}


def normalize_claim(claim: str, meter: dict) -> str:
    if claim in _norm_cache:
        return _norm_cache[claim]
    obj, cost, cached, ptok = llm(
        [{"role": "system", "content": NORM_SYS},
         {"role": "user", "content": claim}],
        cache_key="polarity-norm-v2", max_tokens=200)
    meter["cost"] += cost
    meter["cached"] += cached
    meter["ptok"] += ptok
    out = obj.get("rewritten") or claim
    _norm_cache[claim] = out
    return out

# Daniel's classes (2026-08-20). FAILURE is the exact shape of the reported case;
# NEG is the broader negation class.
FAILURE = re.compile(r"\b(failed to|fails to|rejected|voted down|struck down|defeated|"
                     r"blocked|halted|overturned|did not pass|was not passed)\b", re.I)
NEG = re.compile(r"\b(not|no|never|failed|fails|rejected|denied|refused|without|"
                 r"nobody|none|cannot|can't|didn't|doesn't|won't|isn't|aren't|"
                 r"struck down|voted down|defeated|blocked|halted|overturned)\b", re.I)


def weights() -> dict:
    return json.loads(HEADLINE.read_text())["overall"]["weights"]


def threshold() -> float:
    return json.loads(HEADLINE.read_text())["overall"]["threshold"]


def claim_block(r: dict) -> str:
    """Verbatim from evidence_urn_run.py:648-650."""
    claim = r.get("claim_resolved") or r.get("claim_text")
    return (f"CLAIM: {claim}\n"
            f"(claimed on {r.get('claim_date_shown') or 'unknown date'}; judge the "
            f"document's bearing on this exact proposition)")


def readable(r: dict):
    """The docs the model actually read (skips code-side placeholders)."""
    for d in r.get("results") or []:
        if d.get("read_status") in ("fc-undated", "empty-doc"):
            continue
        if d.get("sent_ids") and d.get("sents"):
            yield d


def reread(sys_prompt: str, block: str, d: dict, meter: dict) -> str | None:
    doc = "\n".join(f"[{i}] {s}" for i, s in zip(d["sent_ids"], d["sents"]))
    obj, cost, cached, ptok = llm(
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": f"{block}\n\nDOCUMENT:\n{doc}"}],
        cache_key="polarity-ab", max_tokens=400)
    meter["cost"] += cost
    meter["cached"] += cached
    meter["ptok"] += ptok
    v = obj.get("direction")
    return str(v) if v is not None else None


def score(flags, w) -> float:
    n_t = sum(1 for f in flags if f in ("5", "4"))
    n_f = sum(1 for f in flags if f in ("1", "2"))
    n_e = len(flags) - n_t - n_f
    return n_t * w["n_t"] + n_f * w["n_f"] + n_e * w["n_e"]


def load(path: Path, want: set[str] | None = None, pred=None) -> list[dict]:
    out = []
    for line in path.open():
        r = json.loads(line)
        if want is not None and r.get("review_url") not in want:
            continue
        if pred is not None and not pred(r):
            continue
        out.append(r)
    return out


def run_arm(recs, arm, meter, workers):
    """arm "norm" = base reader prompt over an affirmative-normalized claim."""
    sysp = ARMS["base"] if arm == "norm" else ARMS[arm]
    res = {}

    def work(r):
        if arm == "norm":
            claim = r.get("claim_resolved") or r.get("claim_text")
            block = claim_block(r).replace(f"CLAIM: {claim}",
                                           f"CLAIM: {normalize_claim(claim, meter)}")
        else:
            block = claim_block(r)
        flags = []
        for d in readable(r):
            f = None
            for _ in range(2):
                try:
                    f = reread(sysp, block, d, meter)
                    break
                except Exception:
                    time.sleep(2)
            flags.append((d["url"], f or (d["read"] or {}).get("direction")))
        res[r["review_url"]] = flags
        meter["done"] += 1
        if meter["done"] % 50 == 0:
            print(f"    {arm}: {meter['done']} claims | ${meter['cost']:.3f}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, recs))
    return res


def cells(recs, base, pol, w, thr):
    """The four-cell table. Never an aggregate."""
    rows = []
    for r in recs:
        u = r["review_url"]
        bf = [f for _, f in base.get(u, [])]
        pf = [f for _, f in pol.get(u, [])]
        if not bf or not pf:
            continue
        claim = r.get("claim_resolved") or r.get("claim_text") or ""
        v = r.get("veracity")
        rows.append({
            "u": u, "claim": claim,
            "gold": None if v not in (1, 2, 3, 4, 5) else ("TRUE" if v >= 4 else "FALSE"),
            "neg": bool(NEG.search(claim)), "fail": bool(FAILURE.search(claim)),
            "b_ref": sum(1 for f in bf if f in ("1", "2")),
            "p_ref": sum(1 for f in pf if f in ("1", "2")),
            "b_sup": sum(1 for f in bf if f in ("5", "4")),
            "p_sup": sum(1 for f in pf if f in ("5", "4")),
            "b_s": score(bf, w), "p_s": score(pf, w), "n": len(bf)})
    print(f"\n{'cell':<26}{'n':>5}{'refute/doc':>22}{'flagged':>18}{'median score':>22}")
    print(f"{'':<26}{'':>5}{'base -> pol':>22}{'base -> pol':>18}{'base -> pol':>22}")
    for gold in ("TRUE", "FALSE", None):
        for neg in (True, False):
            sub = [x for x in rows if x["gold"] == gold and x["neg"] == neg]
            if not sub:
                continue
            nd = sum(x["n"] for x in sub)
            br, pr = sum(x["b_ref"] for x in sub), sum(x["p_ref"] for x in sub)
            bfl = sum(1 for x in sub if x["b_s"] <= thr)
            pfl = sum(1 for x in sub if x["p_s"] <= thr)
            bm = sorted(x["b_s"] for x in sub)[len(sub) // 2]
            pm = sorted(x["p_s"] for x in sub)[len(sub) // 2]
            lab = f"{gold or 'unlabelled'} / {'negation' if neg else 'affirmative'}"
            print(f"{lab:<26}{len(sub):>5}"
                  f"{f'{br/nd:.1%} -> {pr/nd:.1%}':>22}"
                  f"{f'{bfl} -> {pfl}':>18}"
                  f"{f'{bm:+.2f} -> {pm:+.2f}':>22}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="single claim id, prints every document")
    ap.add_argument("--probe", choices=["e1", "e2"], help="run the split probe set")
    ap.add_argument("--arm-b", default="pol", choices=["pol", "norm"],
                    help="second arm: reader rule (pol) or claim rewrite (norm)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    w, thr = weights(), threshold()

    if args.case:
        recs = load(E2, {args.case}) or load(E1, {args.case})
        if not recs:
            raise SystemExit(f"claim {args.case} not found in E2 or E1")
        r = recs[0]
        print(claim_block(r), "\n")
        meter = {"cost": 0.0, "cached": 0, "ptok": 0, "done": 0}
        if args.arm_b == "norm":
            c0 = r.get("claim_resolved") or r.get("claim_text")
            print(f"normalized -> {normalize_claim(c0, {'cost':0,'cached':0,'ptok':0})}\n")
        base = run_arm(recs, "base", meter, args.workers)
        pol = run_arm(recs, args.arm_b, meter, args.workers)
        bmap = dict(base[r["review_url"]])
        pmap = dict(pol[r["review_url"]])
        print(f"{'domain':<28}{'ran':>5}{'base':>7}{'pol':>6}   cited sentence")
        for d in readable(r):
            u = d["url"]
            cited = (d.get("read") or {}).get("evidence") or []
            first = ""
            if cited:
                m = {i: s for i, s in zip(d["sent_ids"], d["sents"])}
                first = (m.get(cited[0]) or "")[:88]
            print(f"{d['domain'][:27]:<28}{(d.get('read') or {}).get('direction',''):>5}"
                  f"{str(bmap.get(u)):>7}{str(pmap.get(u)):>6}   {first}")
        for lab, mp in (("orig", {d["url"]: (d.get("read") or {}).get("direction")
                                  for d in readable(r)}),
                        ("base", bmap), ("pol", pmap)):
            fl = list(mp.values())
            print(f"  {lab:5s} score {score(fl, w):+7.2f}  "
                  f"{'FLAGGED' if score(fl, w) <= thr else 'passes'}  "
                  f"(thr {thr:.4f})  flags {dict(Counter(fl))}")
        print(f"\ncost ${meter['cost']:.4f}")
        return

    src = E1 if args.probe == "e1" else E2
    if args.probe == "e1":
        pred = lambda r: (r.get("veracity") in (1, 2, 4, 5)
                          and NEG.search(r.get("claim_resolved") or r.get("claim_text") or ""))
    else:
        pred = lambda r: FAILURE.search(r.get("claim_resolved") or r.get("claim_text") or "")
    recs = load(src, None, pred)
    if args.limit:
        recs = recs[:args.limit]
    nd = sum(len(list(readable(r))) for r in recs)
    print(f"probe {args.probe}: {len(recs)} claims / {nd} documents / 2 arms = "
          f"{nd * 2} reads | cap ${args.budget}", flush=True)

    meter = {"cost": 0.0, "cached": 0, "ptok": 0, "done": 0}
    t0 = time.time()
    base = run_arm(recs, "base", meter, args.workers)
    if meter["cost"] * 2 > args.budget:
        raise SystemExit(f"BUDGET ABORT after base arm: ${meter['cost']:.3f} x2 > ${args.budget}")
    meter["done"] = 0
    pol = run_arm(recs, args.arm_b, meter, args.workers)
    rows = cells(recs, base, pol, w, thr)
    print(f"\n${meter['cost']:.4f} | {(time.time()-t0)/60:.1f}m | "
          f"cache {meter['cached']/max(meter['ptok'],1):.0%}")
    if args.out:
        args.out.write_text(json.dumps(
            {"rows": rows, "base": base, "pol": pol,
             "polarity_rule": POLARITY_RULE, "weights": w, "threshold": thr}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
