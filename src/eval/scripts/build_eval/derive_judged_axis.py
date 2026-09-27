"""Re-derive judged_axis by LLM, in claim_type's vocabulary (Daniel 2026-08-04).

Replaces the regex in harmonize.py, whose recall is indefensible: it fires only on
explicit wording ("misattributed", "never said", "fake quote"), so 1,603 of 1,951
quote_attribution claims are left null — including "There is no evidence Mr Schwab
has said this" and "Mr Khelif has not said his daughter is transgender".

TWO FIELDS, ONE VOCABULARY. judged_axis stays a separate column from claim_type
because their whole value is the MISMATCH between them:

    claim_type   = the proposition the CLAIM asserts        (from claim text)
    judged_axis  = the proposition the FACT-CHECKER settled (from the rating)

Equal → our verdict and the gold label are about the same thing. Different → we
verify one proposition and get scored against another (claim "Hodkinson says COVID
is a hoax" is quote_attribution and true; BOOM rated the content, false). Collapsing
them to one field would erase exactly the stratum this exists to expose.

judged_axis DRIVES THE READ MODE (Daniel 2026-08-04, superseding the earlier
scoring-only rule). The gold label only applies to the proposition the fact-checker
settled, so READ must assess that proposition or we score its answer against a
verdict that does not apply to it. This is answer-key-derived by construction and
is therefore an EVAL-ALIGNMENT device: in production the mode comes from the claim
itself, since no verdict exists when someone is about to share a post.

  uv run python -m eval.scripts.build_eval.derive_judged_axis --smoke
  uv run python -m eval.scripts.build_eval.derive_judged_axis

Writes eval/data/judged_axis_llm.parquet (review_url, judged_axis_llm, why).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eval.prompt_hash import prompt_hash
from eval.scripts.build_eval.evidence_urn_run import CAL, CTX, llm

SRC = Path("eval/data/fc_gold_v3.parquet")
OUT = Path("eval/data/judged_axis_llm.parquet")   # arm A, title-aware

# claim_type's vocabulary (claim_screen.py) so the two columns are comparable.
AXES = ("quote_attribution", "media_authenticity", "event_occurrence",
        "statistic_figure", "causal_effect", "policy_law", "attribute_identity")

SYS = """You are reading a fact-checker's published verdict to determine WHICH PROPOSITION they actually adjudicated. You get the claim that was checked, the fact-check's headline, the verdict, and sometimes what the fact-checker found. Do not judge whether the claim is true — identify what the fact-checker settled.

This matters because a claim and its verdict can be about different things. "Someone said X" may be rated false because X itself is false, even though the person did say it. Your job is to name the proposition the VERDICT resolves, not the one the claim states.

Reply with JSON only: {"judged_axis": "<one label>", "why": "<one short sentence>"}

Labels — the proposition the verdict settles:
- "quote_attribution": whether a specific person or outlet actually said, wrote, or published the statement. The verdict turns on who said it, not on whether it is true.
- "media_authenticity": whether a photo, video, screenshot, or document genuinely shows what it is presented as showing — right subject, place, or time; or whether it was staged, edited, or synthetic.
- "event_occurrence": whether a specific action or happening took place.
- "statistic_figure": whether a quantity, count, rate, ranking, or trend is accurate.
- "causal_effect": whether X causes, prevents, or worsens Y, including medical efficacy or harm.
- "policy_law": whether a law, rule, or official policy exists, requires, or prohibits something.
- "attribute_identity": whether a standing property, role, composition, or characteristic holds, with no event or causal element.

Guidance: a verdict resting on "the quote is fabricated", "no evidence they said this", "they never said this", or "the post misattributes this" is quote_attribution. A verdict resting on the image or footage being old, from elsewhere, staged, altered, or AI-generated is media_authenticity. A verdict that engages the substance — disputing the number, the mechanism, the legal text, or whether the event happened — takes the corresponding substantive label even when the claim was phrased as a quotation."""


# ARM B — an independent second opinion, deliberately framed differently from the
# 7-label taxonomy above so the two passes do not share a failure mode. Arm A asks
# "which proposition did the verdict settle" over 7 labels; arm B asks the binary
# that actually drives the READ mode, in different words. Agreement between them
# on the BINARY is the guarantee we need before running (Daniel 2026-08-04): the
# fine label can differ (event_occurrence vs statistic_figure) without changing
# what READ is told to assess.
SYS_B = """A fact-checker published a verdict on a claim. Decide what their verdict actually turned on.

Some claims report that a person or outlet said something. For those, a fact-checker may check EITHER whether the person really said it, OR whether what was said is true. These are different questions with different answers — someone can genuinely say something false.

You get the claim, the fact-check's headline, the verdict, and sometimes what they found. The headline usually reveals the answer: a headline about who said something points to the utterance; one that engages the facts points to the substance; one about a clip or image being old, edited or from elsewhere points to the artifact.

Decide which question the fact-check answers.

Reply with JSON only: {"turns_on": "<label>", "why": "<one short sentence>"}

- "utterance": the verdict turns on whether the words were really said, written, or published by that source — it is about authorship, quoting, or misattribution.
- "substance": the verdict turns on whether the state of affairs described is actually so — the event, the number, the effect, the rule, the property.
- "artifact": the verdict turns on whether an image, video, or document genuinely shows or is what it is presented as — staged, edited, old, from elsewhere, or synthetic.

Judge the verdict's actual reasoning, not the claim's grammar. A claim phrased as a quotation whose verdict disputes the facts described is "substance"."""

_B_TO_AXIS = {"utterance": "quote_attribution", "artifact": "media_authenticity"}

PROMPT_HASH = {"a": prompt_hash(SYS), "b": prompt_hash(SYS_B)}


def _user(row) -> str:
    """Claim + every scrap of verdict evidence we hold.

    The rating ALONE is not enough: 84% of ratings are <=15 chars (median 5 —
    "False", "Faux"), which says nothing about WHICH proposition was graded, and
    the two arms diverged 8.7% in that band vs 2.6% where ratings run longer. The
    review TITLE is present for 100% of rows and is highly axis-revealing
    ("...said potholes create good drivers based on satirical article" vs "Fauci
    remark does not negate Covid-19 advice"). fc_rationale exists for 13.9%."""
    parts = [f"CLAIM: {row['claim_text']}",
             f"FACT-CHECK HEADLINE: {row.get('review_title') or '(none)'}",
             f"VERDICT: {row['original_rating']}"]
    if row.get("fc_rationale"):
        parts.append(f"WHAT THE FACT-CHECKER FOUND: {str(row['fc_rationale'])[:600]}")
    return "\n\n".join(parts)


def derive_b(row) -> tuple[dict, float]:
    user = _user(row)
    for _ in range(3):
        try:
            obj, cost, _, _ = llm([{"role": "system", "content": SYS_B},
                                   {"role": "user", "content": user}],
                                  cache_key=f"axisB2-{row['review_url']}", max_tokens=100)
            t = obj.get("turns_on")
            if t in ("utterance", "substance", "artifact"):
                return {"judged_axis_llm": _B_TO_AXIS.get(t, "event_occurrence"),
                        "turns_on": t, "why": (obj.get("why") or "")[:160]}, cost
        except Exception:
            time.sleep(3)
    return {"judged_axis_llm": None, "turns_on": None, "why": ""}, 0.0


def derive(row) -> tuple[dict, float]:
    user = _user(row)
    for _ in range(3):
        try:
            obj, cost, _, _ = llm([{"role": "system", "content": SYS},
                                   {"role": "user", "content": user}],
                                  cache_key=f"axisA2-{row['review_url']}", max_tokens=100)
            a = obj.get("judged_axis")
            if a in AXES:
                return {"judged_axis_llm": a, "why": (obj.get("why") or "")[:160]}, cost
        except Exception:
            time.sleep(3)
    return {"judged_axis_llm": None, "why": ""}, 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="60 rows, cost probe")
    ap.add_argument("--arm", choices=["a", "b"], default="a",
                    help="a = 7-label taxonomy; b = independent binary second opinion")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    fn = derive if args.arm == "a" else derive_b
    out_path = OUT if args.arm == "a" else OUT.with_name("judged_axis_llm_armB.parquet")

    # The E1 draw is what Wednesday's fit needs; VAL can follow on the same pass.
    draw = pl.read_parquet(CTX).select("review_url")
    extra = pl.read_parquet(CAL, columns=["review_url", "review_title", "fc_rationale"])
    g = (pl.read_parquet(SRC, columns=["review_url", "claim_text", "original_rating",
                                       "judged_axis"])
         .join(extra, on="review_url", how="left")
         .join(draw, on="review_url", how="inner")
         .filter(pl.col("original_rating").is_not_null()))
    rows = g.to_dicts()
    if args.smoke:
        rows = rows[::max(1, len(rows) // 60)][:60]
    print(f"deriving judged_axis for {len(rows)} rows | {args.workers} workers", flush=True)

    lock, state, out = threading.Lock(), {"n": 0, "cost": 0.0}, []
    t0 = time.time()

    def work(r):
        v, cost = fn(r)
        with lock:
            state["n"] += 1
            state["cost"] += cost
            out.append({"review_url": r["review_url"], "judged_axis_regex": r["judged_axis"],
                        "prompt_hash": PROMPT_HASH[args.arm], **v})
            if state["n"] % 500 == 0 or state["n"] == len(rows):
                print(f"  {state['n']}/{len(rows)} | ${state['cost']:.3f} | "
                      f"{(time.time()-t0)/60:.1f}m", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, rows))

    df = pl.DataFrame(out)
    print("\njudged_axis_llm:", df["judged_axis_llm"].value_counts().sort("count", descending=True))
    # agreement with the regex WHERE THE REGEX FIRED (its positives are trustworthy;
    # its nulls mean "pattern absent", not "not attribution", so they prove nothing).
    fired = df.filter(pl.col("judged_axis_regex") == "attribution")
    if fired.height:
        agree = fired.filter(pl.col("judged_axis_llm") == "quote_attribution").height
        print(f"\nregex said attribution on {fired.height} rows; LLM agrees on {agree} "
              f"({agree/fired.height:.1%})")
    recovered = df.filter(pl.col("judged_axis_regex").is_null() &
                          (pl.col("judged_axis_llm") == "quote_attribution")).height
    print(f"attribution rows the regex MISSED: {recovered}")
    if not args.smoke:
        df.write_parquet(out_path)
        print(f"\nwrote {out_path}")
    print(f"total ${state['cost']:.3f} | {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
