"""Claim screen for CAL/VAL construction: {claim_type, topic, mundane} per claim.

Taxonomy discovered from a 180-claim hand-coded audit (2026-07-24, agent report in
clog); prompt uses ABSTRACT definitions only — no dataset-derived examples (standing
rule). Output is enum-validated JSON; non-conforming responses retry once then land
in a reject file. Prompt is VERSIONED (SCREEN_PROMPT_V); records carry the tag.

Model is selectable (--model); the smoke compares candidates against the hand-coded
reference (eval/data/claim_taxonomy_reference.csv) before any full pass.

  uv run python -m eval.scripts.build_eval.claim_screen --smoke --model Qwen/Qwen3-235B-A22B-Instruct-2507
  uv run python -m eval.scripts.build_eval.claim_screen --smoke --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
  uv run python -m eval.scripts.build_eval.claim_screen --model <winner>   # full pass (GATED)

Output: eval/data/claim_screen/<model-short>.jsonl (append, resumable by claim key)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL  # DeepInfra creds
from eval.prompt_hash import prompt_hash

SRC = Path("eval/data/fc_gold_v2_clustered.parquet")
OUTDIR = Path("eval/data/claim_screen")

SCREEN_PROMPT_V = "screen-v1"

CLAIM_TYPES = ["quote_attribution", "media_authenticity", "event_occurrence",
               "statistic_figure", "causal_effect", "policy_law", "attribute_identity"]
TOPICS = ["politics_government", "health_medicine", "celebrity_entertainment",
          "conflict_international", "economy_finance", "crime_justice",
          "religion_culture_history", "science_tech", "animals_nature",
          "consumer_brands_scams", "sports", "climate_environment", "other"]

SYSTEM = """You classify short factual claims for an academic misinformation-research dataset. For each claim output JSON with three fields.

"claim_type" — the claim's epistemic shape, what verifying it would require. Exactly one of:
- quote_attribution: a specific person/organization/outlet said, wrote, posted, or published a specific statement; verifying means locating the utterance.
- media_authenticity: a photo/video/document genuinely shows what it is claimed to show (right subject, place, time); verifying means checking the artifact's provenance.
- event_occurrence: a specific action or happening took place (a death, arrest, launch, ban enacted, product released); verifying means checking whether it happened.
- statistic_figure: a quantitative magnitude, count, rate, ranking, or trend; verifying means checking the number against data.
- causal_effect: X causes, prevents, or worsens Y (including medical efficacy or harm); verifying means checking evidence for the effect.
- policy_law: a law, rule, or official policy exists, requires, or prohibits something; verifying means checking the legal or regulatory text.
- attribute_identity: a standing property, role, composition, or characteristic of a person, product, organization, or natural kind, with no event or causal element; verifying means looking up the property.
Precedence when types overlap: if the claim asserts that someone SAID something, it is quote_attribution even when the quoted content is a number (unless the claim's point is the number's correctness rather than the saying, then statistic_figure). A claim that a photo/video shows something is always media_authenticity even when what it shows is an event.

"topic" — the subject domain. Exactly one of: politics_government (incl. elections, immigration, government operations), health_medicine, celebrity_entertainment, conflict_international, economy_finance, crime_justice, religion_culture_history, science_tech, animals_nature, consumer_brands_scams, sports, climate_environment, other.

"mundane" — true/false. A claim is mundane when its truth value, even if widely believed wrongly, misleads no one about any decision-relevant or controversy-relevant matter: no bearing on health, safety, money, elections, policy, conflict, institutional trust, or the conduct or reputation of any actor in a live public dispute. Typical shape: brand, entertainment, animal, or history trivia circulating as amazing-fact content rather than persuasion. NOT mundane, however trivial-sounding: anything touching a public figure's statements or conduct, government operations, or an ongoing public controversy. Test: would correcting a false belief in this claim change anyone's stance on a public matter or protect anyone from harm? If no, it is mundane.

Respond with JSON only: {"claim_type": "...", "topic": "...", "mundane": true|false}"""


SCREEN_PROMPT_HASH = prompt_hash(SYSTEM)


def classify(claim: str, model: str, timeout: int = 45) -> dict | None:
    r = requests.post(
        f"{EXTRACTION_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {EXTRACTION_API_KEY}"},
        json={"model": model, "temperature": 0,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": f"CLAIM:\n{claim}"}]},
        timeout=timeout)
    r.raise_for_status()
    usage = r.json().get("usage", {})
    obj = json.loads(r.json()["choices"][0]["message"]["content"] or "{}")
    ct, tp, mu = obj.get("claim_type"), obj.get("topic"), obj.get("mundane")
    if ct not in CLAIM_TYPES or tp not in TOPICS or not isinstance(mu, bool):
        return None
    return {"claim_type": ct, "topic": tp, "mundane": mu,
            "in_tok": usage.get("prompt_tokens"), "out_tok": usage.get("completion_tokens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--smoke", action="store_true", help="200 claims incl. the reference-set rows")
    ap.add_argument("--subset", choices=["calval"], default=None,
                    help="calval = all trues + 8k year-stratified falses + 2.5k contested/NEE")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pl.read_parquet(SRC).filter(
        pl.col("veracity").is_not_null()
        & ~pl.col("rating_subtype").is_in(["altered_media", "satire"])
        & (pl.col("q_flags") == "")
        & ~pl.col("claim_text").str.contains(r"^\s*Says\b")
        & (pl.col("original_rating") != "Scam"))
    # Cluster representative = EARLIEST fact-check of the claim (Daniel 2026-07-25):
    # anchors the ceiling at the first-check moment and makes sibling fact-checks
    # structurally post-ceiling (leak-safe without domain blocking). Nulls last;
    # review_url tiebreak for determinism.
    df = (df.with_columns(pl.coalesce(pl.col("review_date"), pl.col("claim_date"))
                          .fill_null("9999").alias("_rep_date"))
            .sort(["_rep_date", "review_url"])
            .unique(subset=["cluster_id"], keep="first", maintain_order=True)
            .drop("_rep_date"))

    if args.smoke:
        # overlap the hand-coded reference (seeds 201) + fresh rows to 200
        parts = [df.filter(pl.col("veracity") >= 4).sample(60, seed=201),
                 df.filter(pl.col("veracity") <= 2).sample(60, seed=201),
                 df.filter(pl.col("veracity") == 3).sample(60, seed=201),
                 df.sample(20, seed=202)]
        df = pl.concat(parts).unique(subset=["review_url"])
    elif args.subset == "calval":
        # everything CAL+VAL consume (Daniel 2026-07-24): ALL trues; 8k falses
        # year-stratified on coalesced date; 2.5k contested/NEE. Rest deferred.
        df = df.with_columns(pl.coalesce(pl.col("claim_date"), pl.col("review_date"))
                             .str.slice(0, 4).alias("_y"))
        trues = df.filter(pl.col("veracity") >= 4)
        falses = df.filter(pl.col("veracity") <= 2)
        f_parts = []
        for (y,), g in falses.group_by("_y"):
            share = min(len(g), max(1, round(8000 * len(g) / len(falses))))
            f_parts.append(g.sample(share, seed=303))
        mid = df.filter(pl.col("veracity") == 3).sample(2500, seed=303)
        df = pl.concat([trues, *f_parts, mid]).unique(subset=["review_url"]).drop("_y")
        print(f"calval subset: {len(df)} (T {len(trues)}, F {sum(len(p) for p in f_parts)}, mid {len(mid)})")
    if args.limit:
        df = df.head(args.limit)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    short = args.model.split("/")[-1][:40]
    out = OUTDIR / f"{short}{'.smoke' if args.smoke else ''}.jsonl"
    seen = set()
    if out.exists():
        for l in open(out):
            try:
                seen.add(json.loads(l)["review_url"])
            except Exception:
                pass

    todo = [r for r in df.iter_rows(named=True) if r["review_url"] not in seen]
    print(f"{len(todo)} to classify with {args.model} ({args.workers} workers) -> {out.name} "
          f"(prompt {SCREEN_PROMPT_V})", flush=True)

    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    state = {"done": 0, "fail": 0, "ti": 0, "to": 0}
    t0 = time.time()
    f = open(out, "a")

    def work(r):
        res = None
        for _ in range(2):
            try:
                res = classify(r["claim_text"], args.model)
            except Exception:
                time.sleep(2)
            if res:
                break
        with lock:
            if not res:
                state["fail"] += 1
                return
            state["ti"] += res.pop("in_tok") or 0
            state["to"] += res.pop("out_tok") or 0
            f.write(json.dumps({"review_url": r["review_url"],
                                "claim_text": r["claim_text"][:200],
                                "publisher_site": r["publisher_site"],
                                "veracity": r["veracity"], **res,
                                "prompt_v": SCREEN_PROMPT_V,
                                "prompt_hash": SCREEN_PROMPT_HASH,
                                "model": args.model}) + "\n")
            state["done"] += 1
            if state["done"] % 250 == 0:
                f.flush()
                rate = state["done"] / (time.time() - t0)
                print(f"  {state['done']}/{len(todo)} ({rate:.1f}/s, "
                      f"ETA {(len(todo)-state['done'])/rate/60:.0f}m) "
                      f"tokens in/out {state['ti']}/{state['to']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    f.close()
    print(f"done: {state['done']} ok, {state['fail']} failed | tokens in {state['ti']} "
          f"out {state['to']} | {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
