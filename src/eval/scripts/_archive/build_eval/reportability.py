"""Label-blind REPORTABILITY stratifier: would a true version of this claim leave a trace?

    uv run python -m eval.scripts.build_eval.reportability --smoke
    uv run python -m eval.scripts.build_eval.reportability            # full pass (gated)

Why. The silent-evidence weight has been unstable across fits (outlet-fitted
-0.65, gold-fitted near zero). Daniel's hypothesis (2026-08-28): silence is not
one thing. For a claim that would necessarily have been reported if true, silence
is strong evidence of falsehood; for a claim nobody would ever write about,
silence is uninformative. One global w_silent averages the two regimes.

This script assigns the conditioning variable. It is deliberately the DUMBEST
possible instrument: one question, three levels, claim text + claim date in,
one enum out.

LABEL-BLINDNESS is structural, not a convention. The worker function's only
arguments are two strings, `claim` and `date`. The rows handed to the pool are
built by `_todo()` from a three-field tuple, so a veracity label, a Truth Odds
score, a retrieved document or a read flag has no field to travel in. Nothing
downstream of retrieval is loaded by this module at all.

Population: every claim in the three fitted corpora --
  gold  the pinned fc-gold population (media axis out, rating_subtype=mixed out,
        n = 3,274), exactly model_ladder's cut, asserted row-for-row
  cn    the CN-false urn (scores.jsonl + scores_ext.jsonl minus fit_exclusions),
        n = 1,969, asserted against transfer_ladder.load_urn
  tl    the timeline urn, n = 1,999, likewise asserted

Output: eval/data/reportability/labels.jsonl (append, resumable by key).
Nothing under eval/data/urn_runs/ is read for anything but claim text and date,
and nothing there is written.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL
from eval.scripts.build_eval import fit_urn
from eval.scripts.build_eval.graded_urn import FLAGS

E1 = Path("eval/data/urn_runs/e1_ctx/results-00.jsonl")
C2 = Path("eval/data/urn_runs/c2_false")
TL = Path("eval/data/urn_runs/true_timeline")
OUTDIR = Path("eval/data/reportability")
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
PROMPT_V = "reportability-v2"
SET_ASIDE = "mixed"
LEVELS = ("high", "medium", "low")
BUDGET_CAP = 2.00          # hard stop, USD of LLM spend (task ceiling)
# DeepSeek-V4-Flash list price on DeepInfra, used only for the in-run meter.
PRICE_IN, PRICE_OUT = 0.28e-6, 0.42e-6

SYSTEM = """You rate how DISCOVERABLE a factual claim would be IF IT WERE TRUE. You get only the claim and the date it was circulating. You do not know whether it is true, and you must not try to work that out.

One question: supposing the claim were true, would an ordinary web search now turn up contemporaneous reporting of it, an official or institutional record of it, or some other public trace?

Rate reportability, never plausibility. A claim can be plainly false and highly reportable, because what it describes is the kind of thing that would have been covered had it happened. A claim can be true and leave no findable trace. Do not let any sense of whether it is true move the rating.

Raises it: named public actors or institutions; a discrete, datable, consequential event; a domain that is systematically documented (legislation, courts, elections, regulators, markets, published research, official statistics, major sport); specificity such that one identifiable record would settle it; enough time since the date for coverage to be indexed.

Lowers it: private individuals, private exchanges, inner motives, unobserved circumstances; generalisations, interpretations, predictions, evaluations rather than discrete recordable facts; a domain nothing systematically covers; vagueness such that no particular record would establish it; a matter local, obscure or old enough to be undigitised, or so recent relative to the date that coverage may not exist yet.

Three levels, and the middle one is the default. Reserve the ends for clear cases.
- "high": a public record would be near-certain and easy to find. Only when both hold.
- "medium": use this whenever the case is not clear-cut. Some trace would probably exist, but it could be thin, partial, indirect, in specialist or non-English sources, or hard to separate from surrounding material.
- "low": there is little reason to expect any findable public record, so the open web would look much the same whether or not the claim is true.

Respond with JSON only: {"reportability": "high"|"medium"|"low"}"""


# --------------------------------------------------------------------------
# population loaders. Each mirrors the fitting script's own loader exactly and
# is asserted against it in main(); they add only the claim text and the date.
# --------------------------------------------------------------------------
def gold_rows() -> list[dict]:
    """model_ladder.load_docs' population, minus rating_subtype=mixed, keyed."""
    axis = fit_urn.load_judged_axis()
    rows = []
    for line in E1.open():
        r = json.loads(line)
        if r.get("veracity") not in (1, 2, 3, 4, 5):
            continue
        if not any((d.get("read") or {}).get("direction") in FLAGS
                   for d in r.get("results") or []):
            continue
        if (axis.get(r["review_url"]) or "untagged") == fit_urn.MEDIA_AXIS:
            continue
        if (r.get("rating_subtype") or "?") == SET_ASIDE:
            continue
        rows.append({"corpus": "gold", "key": f"gold:{r['review_url']}",
                     "claim": r.get("claim_resolved") or r.get("claim_text") or "",
                     "date": r.get("claim_date_shown") or r.get("ceiling")
                             or r.get("yr") or ""})
    return rows


def urn_rows(corpus: str, paths: list[Path], exclusions: Path | None) -> list[dict]:
    """transfer_ladder.load_urn's population, keyed, with the claim date kept."""
    excl = set()
    if exclusions and exclusions.exists():
        excl = {e["claim"][:80] for e in json.loads(exclusions.read_text())}
    rows = []
    for p in paths:
        for line in p.open():
            r = json.loads(line)
            claim = r.get("claim_resolved") or r.get("claim_text") or ""
            if claim[:80] in excl:
                continue
            if not any((d.get("read") or {}).get("direction") in FLAGS
                       for d in r.get("results") or []):
                continue
            rows.append({"corpus": corpus, "key": f"{corpus}:{r['review_url']}",
                         "claim": claim,
                         "date": r.get("claim_date_shown") or r.get("ceiling") or ""})
    return rows


def load_population() -> list[dict]:
    return (gold_rows()
            + urn_rows("cn", [C2 / "scores.jsonl", C2 / "scores_ext.jsonl"],
                       C2 / "fit_exclusions.json")
            + urn_rows("tl", [TL / "scores.jsonl"], None))


# --------------------------------------------------------------------------
def classify(claim: str, date: str, model: str, timeout: int = 60) -> dict | None:
    """Two strings in, one enum out. This signature is the label-blindness guard."""
    r = requests.post(
        f"{EXTRACTION_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {EXTRACTION_API_KEY}"},
        json={"model": model, "temperature": 0,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user",
                            "content": f"CLAIM DATE: {date or 'unknown'}\nCLAIM: {claim}"}]},
        timeout=timeout)
    r.raise_for_status()
    body = r.json()
    usage = body.get("usage") or {}
    obj = json.loads(body["choices"][0]["message"]["content"] or "{}")
    lvl = obj.get("reportability")
    if lvl not in LEVELS:
        return None
    return {"reportability": lvl, "in_tok": usage.get("prompt_tokens") or 0,
            "out_tok": usage.get("completion_tokens") or 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--smoke", action="store_true",
                    help="90 claims (30 per corpus, deterministic stride)")
    ap.add_argument("--verify-population", action="store_true",
                    help="assert the loaders against model_ladder / transfer_ladder "
                         "and exit without spending anything")
    args = ap.parse_args()

    pop = load_population()
    n = {c: sum(1 for r in pop if r["corpus"] == c) for c in ("gold", "cn", "tl")}
    print(f"population gold {n['gold']} / cn {n['cn']} / tl {n['tl']} = {len(pop)}",
          flush=True)
    assert len({r["key"] for r in pop}) == len(pop), "keys are not unique"

    if args.verify_population:
        from eval.scripts.build_eval import model_ladder as ML
        from eval.scripts.build_eval import transfer_ladder as TLM
        g = [r for r in ML.load_docs() if r["subtype"] != ML.SET_ASIDE]
        cn = TLM.load_urn([TLM.C2 / "scores.jsonl", TLM.C2 / "scores_ext.jsonl"],
                          TLM.C2 / "fit_exclusions.json")
        tl = TLM.load_urn([TLM.TL / "scores.jsonl"])
        assert len(g) == n["gold"] and len(cn) == n["cn"] and len(tl) == n["tl"], \
            (len(g), len(cn), len(tl), n)
        mine = [r for r in pop if r["corpus"] == "cn"]
        assert all(a["claim"] == b["claim"] for a, b in zip(cn, mine))
        mine = [r for r in pop if r["corpus"] == "tl"]
        assert all(a["claim"] == b["claim"] for a, b in zip(tl, mine))
        print("populations match model_ladder / transfer_ladder row-for-row")
        return

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / ("labels.smoke.jsonl" if args.smoke else "labels.jsonl")
    if args.smoke:
        pop = [r for c in ("gold", "cn", "tl")
               for r in [x for x in pop if x["corpus"] == c][::73][:30]]
    seen = set()
    if out.exists():
        for line in out.open():
            try:
                seen.add(json.loads(line)["key"])
            except Exception:
                pass
    # Three fields only. A label has no field to ride in.
    todo = [(r["key"], r["claim"], r["date"]) for r in pop if r["key"] not in seen]
    print(f"{len(todo)} to classify ({len(seen)} already done) with {args.model}, "
          f"{args.workers} workers -> {out.name}  [{PROMPT_V}]", flush=True)
    if not todo:
        return

    lock = threading.Lock()
    st = {"done": 0, "fail": 0, "ti": 0, "to": 0, "stop": False}
    counts = {lvl: 0 for lvl in LEVELS}
    t0 = time.time()
    fh = out.open("a")

    def work(item):
        key, claim, date = item
        if st["stop"]:
            return
        res = None
        for attempt in range(3):
            try:
                res = classify(claim, date, args.model)
                if res:
                    break
            except Exception:
                pass
            time.sleep(1.5 * (attempt + 1))
        with lock:
            if not res:
                st["fail"] += 1
                return
            st["ti"] += res["in_tok"]
            st["to"] += res["out_tok"]
            counts[res["reportability"]] += 1
            fh.write(json.dumps({"key": key, "reportability": res["reportability"],
                                 "prompt_v": PROMPT_V, "model": args.model}) + "\n")
            st["done"] += 1
            cost = st["ti"] * PRICE_IN + st["to"] * PRICE_OUT
            if cost > BUDGET_CAP:
                st["stop"] = True
                print(f"BUDGET CAP ${BUDGET_CAP:.2f} BREACHED at {st['done']} claims "
                      f"-- halting", flush=True)
            if st["done"] % 200 == 0 or st["done"] == len(todo):
                fh.flush()
                el = time.time() - t0
                rate = st["done"] / el
                print(f"  {st['done']}/{len(todo)}  {rate:.1f}/s  "
                      f"elapsed {el/60:.1f}m  ETA {(len(todo)-st['done'])/rate/60:.1f}m  "
                      f"tok {st['ti']}/{st['to']}  ${cost:.3f}  fail {st['fail']}  "
                      f"H/M/L {counts['high']}/{counts['medium']}/{counts['low']}",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    cost = st["ti"] * PRICE_IN + st["to"] * PRICE_OUT
    print(f"done {st['done']} ok / {st['fail']} failed | tok in {st['ti']} out {st['to']} "
          f"| ${cost:.3f} | {(time.time()-t0)/60:.1f}m", flush=True)
    print(f"distribution  high {counts['high']}  medium {counts['medium']}  "
          f"low {counts['low']}", flush=True)


if __name__ == "__main__":
    main()
