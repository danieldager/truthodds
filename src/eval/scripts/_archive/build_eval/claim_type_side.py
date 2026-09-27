"""Label-blind CLAIM-SIDE type: which of the seven axes does the CLAIM assert?

    uv run python -m eval.scripts.build_eval.claim_type_side --smoke
    uv run python -m eval.scripts.build_eval.claim_type_side            # full pass (gated)

Why this exists and why it is not derive_judged_axis.py. `judged_axis` names the
proposition the FACT-CHECKER settled: it is read off the verdict, so it exists
only where a verdict exists, which is never at production time. Any stratifier
built on it is a diagnostic, not a feature. This script assigns the SAME seven
labels from the claim alone, so the Arm-1 result can be re-run with a stratum a
deployed system could actually compute.

LABEL-BLINDNESS is structural, not a convention. `classify`'s only arguments are
two strings, `claim` and `date`. The rows handed to the pool are built by `_todo`
from a three-field tuple, so a veracity label, a Truth Odds score, a retrieved
document, a read flag or the fact-check itself has no field to travel in. The
prompt states principles only -- no example is drawn from any corpus of ours and
none mirrors our data's structure (standing project rule).

Population: the three fitted corpora, keyed identically to reportability.py --
  gold  the pinned fc-gold cut WITH media_authenticity added back (n = 3,568).
        Dropping the media rows reproduces reportability.gold_rows / model_ladder
        row-for-row, asserted under --verify-population.
  cn    the CN-false urn, n = 1,969, asserted against transfer_ladder.load_urn
  tl    the timeline urn, n = 1,999, likewise asserted

Output: eval/data/claim_type/labels.jsonl (append, resumable by key). Nothing
under eval/data/urn_runs/ is written.
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
OUTDIR = Path("eval/data/claim_type")
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
PROMPT_V = "claim-type-side-v1"
SET_ASIDE = "mixed"
TYPES = ("quote_attribution", "media_authenticity", "event_occurrence",
         "statistic_figure", "causal_effect", "policy_law", "attribute_identity")
BUDGET_CAP = 1.80          # hard stop, USD of LLM spend (task ceiling is $2.00)
PRICE_IN, PRICE_OUT = 0.28e-6, 0.42e-6    # DeepInfra list price, in-run meter only

SYSTEM = """You classify what kind of thing a factual claim asserts. You get the claim and the date it was circulating, nothing else. You do not know whether it is true and you must not try to work that out. Classify the SHAPE of the assertion, never its truth, its plausibility, its topic, or how serious it is.

Ask one question: what single fact would have to be established for this claim to be settled? The label names the kind of fact that would settle it.

Pick exactly one label:

- "quote_attribution": the claim is that a named person, outlet or body said, wrote, posted or published a particular statement. What would settle it is whether that utterance was made by that source, not whether the statement's content is correct.
- "media_authenticity": the claim presents an image, video, screenshot or document as showing a particular subject, place or moment, or asserts that such an artefact is genuine, staged, altered or synthetic. What would settle it is what the artefact actually is or depicts.
- "event_occurrence": the claim is that a discrete action or happening took place, or did not take place. What would settle it is a record of the occurrence itself.
- "statistic_figure": the claim's substance is a quantity, count, rate, share, ranking, amount of money, or a trend in one. What would settle it is the number being right or wrong.
- "causal_effect": the claim is that one thing causes, prevents, worsens, protects against or results from another, including any assertion about whether something works, is effective, or is harmful. What would settle it is the causal relationship.
- "policy_law": the claim is that a law, rule, regulation, court order, official decision or institutional policy exists, or that it requires, permits, bans or changes something. What would settle it is the text or status of the instrument itself.
- "attribute_identity": the claim is that a standing property, role, membership, identity, composition, ownership or affiliation holds, with no event, quantity or causal element. What would settle it is the property holding or not.

Deciding between labels:
- Grammar does not decide. A claim worded as a quotation whose real content is a number, a rule or a causal relation is still that, UNLESS the point being asserted is the fact of the saying itself.
- If the claim's whole content is that a source is presenting something visual as authentic, or that a visual record shows a particular thing, choose media_authenticity over the label the depicted content would otherwise take.
- If a quantity is present but incidental to what is being asserted, do not choose statistic_figure; choose it when the number is the substance.
- An announcement, decision or measure that has been taken is policy_law when the claim is about the instrument, and event_occurrence when the claim is about something happening at a time and place.
- Choose attribute_identity only when nothing happened, nothing is counted and nothing causes anything.
- Exactly one label always applies. When two are close, choose the one naming the fact a checker would have to establish FIRST.

Respond with JSON only: {"claim_type": "<one label>"}"""


# --------------------------------------------------------------------------
# population loaders. Each mirrors the fitting script's own loader and is
# asserted against it in main(); they add only the claim text and the date.
# --------------------------------------------------------------------------
def gold_rows(keep_media: bool = True) -> list[dict]:
    """model_ladder's cut (minus rating_subtype=mixed), media optionally kept."""
    axis = fit_urn.load_judged_axis()
    rows = []
    for line in E1.open():
        r = json.loads(line)
        if r.get("veracity") not in (1, 2, 3, 4, 5):
            continue
        if not any((d.get("read") or {}).get("direction") in FLAGS
                   for d in r.get("results") or []):
            continue
        if not keep_media and (axis.get(r["review_url"]) or "") == fit_urn.MEDIA_AXIS:
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
    t = obj.get("claim_type")
    if t not in TYPES:
        return None
    return {"claim_type_side": t, "in_tok": usage.get("prompt_tokens") or 0,
            "out_tok": usage.get("completion_tokens") or 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--smoke", action="store_true",
                    help="90 claims (30 per corpus, deterministic stride)")
    ap.add_argument("--verify-population", action="store_true",
                    help="assert the loaders against model_ladder / transfer_ladder / "
                         "reportability and exit without spending anything")
    args = ap.parse_args()

    pop = load_population()
    n = {c: sum(1 for r in pop if r["corpus"] == c) for c in ("gold", "cn", "tl")}
    print(f"population gold {n['gold']} / cn {n['cn']} / tl {n['tl']} = {len(pop)}",
          flush=True)
    assert len({r["key"] for r in pop}) == len(pop), "keys are not unique"

    if args.verify_population:
        from eval.scripts.build_eval import model_ladder as ML
        from eval.scripts.build_eval import transfer_ladder as TLM
        from eval.scripts.build_eval import reportability as RP
        g = [r for r in ML.load_docs() if r["subtype"] != ML.SET_ASIDE]
        cn = TLM.load_urn([TLM.C2 / "scores.jsonl", TLM.C2 / "scores_ext.jsonl"],
                          TLM.C2 / "fit_exclusions.json")
        tl = TLM.load_urn([TLM.TL / "scores.jsonl"])
        nomedia = gold_rows(keep_media=False)
        assert len(nomedia) == len(g) == 3274, (len(nomedia), len(g))
        rp = RP.gold_rows()
        assert [r["key"] for r in rp] == [r["key"] for r in nomedia]
        assert [r["claim"] for r in rp] == [r["claim"] for r in nomedia]
        assert len(cn) == n["cn"] and len(tl) == n["tl"], (len(cn), len(tl), n)
        assert all(a["claim"] == b["claim"] for a, b in
                   zip(cn, [r for r in pop if r["corpus"] == "cn"]))
        assert all(a["claim"] == b["claim"] for a, b in
                   zip(tl, [r for r in pop if r["corpus"] == "tl"]))
        print("gold minus media == reportability/model_ladder cut (3,274) row-for-row; "
              "urns match transfer_ladder row-for-row")
        print(f"reportability's 7,242 + {n['gold'] - 3274} media rows = {len(pop)}")
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
    counts = {t: 0 for t in TYPES}
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
            counts[res["claim_type_side"]] += 1
            fh.write(json.dumps({"key": key, "claim_type_side": res["claim_type_side"],
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
                      f"tok {st['ti']}/{st['to']}  ${cost:.3f}  fail {st['fail']}",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    cost = st["ti"] * PRICE_IN + st["to"] * PRICE_OUT
    print(f"done {st['done']} ok / {st['fail']} failed | tok in {st['ti']} out {st['to']} "
          f"| ${cost:.3f} | {(time.time()-t0)/60:.1f}m", flush=True)
    print("distribution  " + "  ".join(f"{t} {counts[t]}" for t in TYPES), flush=True)


if __name__ == "__main__":
    main()
