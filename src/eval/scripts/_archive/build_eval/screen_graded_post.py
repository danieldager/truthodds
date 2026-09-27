"""Screen for the "gold graded the post, not the claim" misalignment stratum (B3).

The fact-checker reviewed a POST; we score one extracted CLAIM. When the verdict
adjudicates the post's framing (spin / embellishment / missing context / old event
shared as new) while the claim's literal proposition is itself accurate, gold and
the scored proposition point at different things — the same eval-misalignment
family as judged_axis's media_authenticity, one layer down. A cross-family judge
(Llama-3.3-70B vs the Qwen reader) tags each claim from the claim + fact-check
title + verbatim rating + subtype (+ post context where leak-screened clean).
Verdict-bearing text is EVAL-ONLY input to this screen; it never feeds the scorer.

  uv run python -m eval.scripts.build_eval.screen_graded_post --smoke 50   # stratified smoke
  uv run python -m eval.scripts.build_eval.screen_graded_post              # full pass (needs go)

Writes eval/data/graded_post_screen[.smoke].parquet. Budget-capped; abort if the
projection exceeds --budget.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))
from eval.scripts.build_eval import fit_urn  # noqa: E402

BASE = "https://api.deepinfra.com/v1/openai"
KEY = next(l.split("=", 1)[1].strip() for l in (SRC / ".env").read_text().splitlines()
           if l.startswith("DEEPINFRA_API_KEY="))
MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # --model overrides
RESULTS = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
OUT = Path("eval/data/graded_post_screen.parquet")
USAGE = {"in": 0, "out": 0, "cost": 0.0, "n": 0}

SYSTEM = """You audit the alignment between a fact-check's verdict and one specific claim scored by an automated system. The fact-checker reviewed a social-media post; our system scored a single CLAIM extracted from it. Classify what the verdict actually adjudicated, into exactly one axis:

- "claim_false": the fact-check finds the CLAIM's own literal proposition inaccurate, unproven, or fabricated as stated. This includes any false element inside the claim (wrong actor or role, wrong place, wrong date stated in the claim, wrong number, a causal or temporal link the fact-check rejects, overstated scope, a lab finding stated as a real-world effect), a claim asserting something the fact-check says did not happen, and a claim stating a statistic whose validity or provenance is what the fact-check disputes.
- "post_framing": the rating is NEGATIVE (false / misleading / mixture / needs-context family), yet the fact-check's own material CONCEDES the claim's literal core — the underlying event happened, the number is right, the decision exists — and the negative rating targets the surrounding post instead: its spin, exaggerated side details, omitted context, misleading juxtaposition, or the resharing of a genuinely old event as current news.
- "attribution_content": the claim has the form "PERSON said/wrote Q", the fact of them saying it is not in dispute, and the verdict rates the truth of Q itself. Always use this axis for such claims, never post_framing.
- "claim_graded": the verdict simply rates the claim's own proposition and agrees or disagrees with it directly, with no framing gap — including every claim whose rating is POSITIVE (true / correct / mostly-true family).
- "cannot_tell": the provided material is not enough to decide.

Discipline, in order:
1. If the publisher rating is positive, the axis is claim_graded. Full stop. Negative and mixed rating families (false, misleading, mixture, half-true, needs-context) decide NOTHING by themselves — for them, only the tests below decide.
2. If the claim is a said-quote whose content the verdict rates, the axis is attribution_content.
3. Test every element written INSIDE the claim's own text — actor, role, place, timing, causal or temporal link, scope, program name, number. If the fact-check corrects ANY of them, the axis is claim_false. post_framing is reserved for faults that live entirely OUTSIDE the claim's text, in the post around it.
4. post_framing additionally requires an explicit CONCESSION in the provided material: wording that itself confirms the claim's core, such as "did happen", "is real, but", "old/outdated news reshared as new", "the number is right, but", "only for certain …" correcting a broader spin while leaving the claim's own words intact. The bare words "misleading", "misleadingly claimed" or "misrepresented" are NOT a concession — fact-checkers routinely use them when the claim itself is wrong. Never infer an unstated implication the fact-checker "really" targeted; if the material reads as a plain refutation of the claim, the axis is claim_false no matter how plausible a framing story sounds.

Return only JSON: {"axis": "claim_false"|"post_framing"|"attribution_content"|"claim_graded"|"cannot_tell", "confidence": "high"|"low", "why": "<one sentence quoting the material that decided it>"}"""


def _obj(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def judge(row: dict, model: str = MODEL) -> dict:
    ctx = f"\nPOST CONTEXT: {row['x_context']}" if row.get("x_context") else ""
    user = (f"CLAIM SCORED BY THE SYSTEM: {row['claim']}\n"
            f"FACT-CHECK TITLE: {row['title']}\n"
            f"PUBLISHER RATING (verbatim): {row['orig']}\n"
            f"RATING SUBTYPE (harmonised): {row['subtype']}{ctx}")
    payload = {"model": model, "temperature": 0, "max_tokens": 250,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user}]}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode())
                u = d.get("usage") or {}
                USAGE["in"] += u.get("prompt_tokens", 0)
                USAGE["out"] += u.get("completion_tokens", 0)
                USAGE["cost"] += u.get("estimated_cost") or 0
                USAGE["n"] += 1
                return _obj(d["choices"][0]["message"]["content"])
        except Exception:
            time.sleep(2 * 2 ** a)
    return {}


def load_rows() -> list[dict]:
    rows = fit_urn.load_headline(RESULTS)
    cal = pl.read_parquet("eval/data/truthodds_cal.parquet").select(
        ["review_url", "rating_subtype", "review_title", "original_rating"])
    meta = {r["review_url"]: r for r in cal.iter_rows(named=True)}
    ctx = pl.read_parquet("eval/data/claim_context_cal.parquet").select(
        ["review_url", "x_context", "context_ok"])
    ctxmap = {r["review_url"]: r["x_context"] for r in ctx.iter_rows(named=True)
              if r["context_ok"] and r["x_context"]}
    for r in rows:
        m = meta.get(r["review_url"], {})
        r["subtype"] = m.get("rating_subtype") or "?"
        r["title"] = m.get("review_title") or ""
        r["orig"] = m.get("original_rating") or ""
        r["x_context"] = (ctxmap.get(r["review_url"]) or "")[:400]
    return rows


def smoke_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Stratified: suspect subtypes and both score tails represented."""
    fr = [r for r in rows if r["y"] == 0]
    tr = [r for r in rows if r["y"] == 1]
    hi = lambda rs: [r for r in rs if r["s7"] > 0]
    lo = lambda rs: [r for r in rs if r["s7"] <= 0]
    mix, cf = [r for r in fr if r["subtype"] == "mixed"], [r for r in fr if r["subtype"] == "clear_false"]
    cells = [(hi(mix), 8), (lo(mix), 7), (hi(cf), 5), (lo(cf), 10),
             ([r for r in fr if r["subtype"] == "mostly_false"], 5),
             ([r for r in fr if r["subtype"] == "unprovable"], 5),
             (sorted(tr, key=lambda r: r["s7"])[:30], 5), (tr, 5)]
    out, seen = [], set()
    for pool, k in cells:
        for r in rng.sample(pool, min(k, len(pool))):
            if r["review_url"] not in seen:
                seen.add(r["review_url"])
                out.append(r)
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="stratified sample size (0 = full pass)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--budget", type=float, default=1.0, help="hard $ cap; abort beyond it")
    args = ap.parse_args()

    rows = load_rows()
    # graded oof score, for stratification + reporting
    from eval.scripts.build_eval.graded_urn import fit_graded, score_graded
    for fold in range(fit_urn.K_FOLDS):
        w = fit_graded([r for r in rows if r["fold"] != fold])
        for r in rows:
            if r["fold"] == fold:
                r["s7"] = score_graded(r, w)

    rng = random.Random(707)
    todo = smoke_sample(rows, args.smoke, rng) if args.smoke else rows
    out_path = OUT.with_suffix(".smoke.parquet") if args.smoke else OUT
    print(f"screening {len(todo)} claims ({args.model}, {args.concurrency} workers) -> {out_path}", flush=True)
    t0 = time.time()

    def run(r: dict) -> dict:
        if USAGE["cost"] > args.budget:
            return {}
        o = judge(r, args.model)
        axis = o.get("axis") or "?"
        return {"review_url": r["review_url"], "claim": r["claim"], "subtype": r["subtype"],
                "title": r["title"], "orig": r["orig"], "y": r["y"], "mid": r["mid"],
                "s7": r["s7"], "axis": axis, "graded_post": axis == "post_framing",
                "confidence": o.get("confidence") or "?", "why": o.get("why") or "",
                "judged": bool(o)}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        recs = [x for x in ex.map(run, todo) if x]
    pl.DataFrame(recs).write_parquet(out_path)
    dt = time.time() - t0
    n_pos = sum(r["graded_post"] for r in recs)
    print(f"done: {len(recs)} judged in {dt/60:.1f}m, ${USAGE['cost']:.4f} "
          f"({USAGE['in']:,} in / {USAGE['out']:,} out) — graded_post {n_pos} ({n_pos/max(len(recs),1):.1%})")
    if USAGE["n"]:
        per = USAGE["cost"] / USAGE["n"]
        print(f"per-claim ${per:.6f} -> full pass ({len(rows)}) projects ${per*len(rows):.2f}")


if __name__ == "__main__":
    main()
