"""Stage-1 GATE eval — run the check-worthiness gate over a labeled set, report recall/specificity.

  uv run python -m eval.scripts.gate_eval --parquet eval/data/synthetic_negatives.parquet --expect reject
  uv run python -m eval.scripts.gate_eval --parquet <positives> --expect pass

Runs the Stage-1 gate (Qwen3-VL-30B, GATE_SYSTEM below — kept in sync with the spec) over every row,
records {public_interest, misinfo_if_false, claim_locus, check_worthy, reasoning}, and scores against
the expected label. A post is GATED-OUT (rejected) when check_worthy=False OR claim_locus=='video'.
  - expect=reject (negatives): a FALSE POSITIVE = gate keeps it (check_worthy & locus!=video). Prints
    every FP for audit — split mislabels (actually check-worthy → fix the data) from over-broadness.
  - expect=pass (positives): a FALSE NEGATIVE = gate drops a real fact-checked post.
Crash-safe/resumable (pooled_checkpointed). Writes eval/data/gate_eval_<name>.parquet.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

import polars as pl
import requests

from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL
from eval.scripts._pool import pooled_checkpointed

URL = f"{EXTRACTION_BASE_URL}/chat/completions"
HDR = {"Authorization": f"Bearer {EXTRACTION_API_KEY}"}
GATE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MIN_TOKENS = 150
MAX_TOKENS = 600  # concise reasoning keeps output ~100 tok; margin guards an occasional long one

GATE_SYSTEM = (
    "You are the scope-and-check-worthiness gate at the front of a fact-checking pipeline. For each "
    "social-media post decide whether it should pass to verification — judge check-worthiness, NOT whether "
    "the claim is true.\n\n"
    "HOW TO READ THE POST\n"
    "Resolve references ('this','they','the parent'), read text inside images (OCR), and weigh what the "
    "user asserts by posting (including any quoted or parent tweet they amplify or endorse). The checkable "
    "claim often lives in the IMAGE itself — a screenshot, a fabricated or quoted headline, a manipulated "
    "or out-of-context photo — even when the caption is only a reaction, question, or joke; judge the claim "
    "the image makes (a purely decorative, scenic, or personal photo is not a claim).\n\n"
    "DECIDE IN THIS ORDER\n"
    "1. SCOPE (public_interest) — is the post about a matter of SOCIETAL CONSEQUENCE? Popularity, fame, or "
    "money involved does NOT make it in-scope.\n"
    "   IN: politics & elections, government policy/spending, the economy, corruption & bribery, war & "
    "geopolitics, terrorism, immigration, public health & safety, crime, disasters, science & climate.\n"
    "   OUT, unless it carries the consequence noted in parentheses:\n"
    "   - sports results / transfers / player takes (UNLESS doping, match-fixing, corruption, a crime);\n"
    "   - entertainment / celebrity / reality-TV / music / concerts (UNLESS a real-world event: death, "
    "illness, crime, accident);\n"
    "   - personal customer-service or product complaints; ads, giveaways, clickbait/spam, get-rich-quick "
    "or 'free prize' scams (UNLESS a health/safety claim);\n"
    "   - personal updates, jokes, greetings, trivia; opinions, aesthetic judgments, predictions; questions "
    "that assert no fact.\n"
    "   Topic beats specificity: a precise, verifiable fact in an out-of-scope domain is still OUT.\n"
    "2. RISK (misinfo_if_false) — does it state a SPECIFIC, VERIFIABLE fact (a named event, number, action, "
    "or quote) whose falsehood would consequentially mislead people? Judge consequence, not truth — an "
    "obviously-true public claim still qualifies. A sweeping, unfalsifiable value-judgment about "
    "institutions ('the media never tell the truth', 'the system is broken') is rhetoric, not a checkable "
    "claim. If the post asserts no concrete checkable fact, do NOT invent one.\n"
    "3. VERDICT — check_worthy = (in SCOPE) AND (a specific consequential claim).\n\n"
    "EXAMPLES (verdict — which test decides it; the same domain flips on consequence)\n"
    "- 'two A-list stars are reportedly dating' — reject (SCOPE: celebrity).\n"
    "- 'a famous actor died in a car crash' — keep (celebrity, but a real-world death).\n"
    "- 'a star striker is on the transfer list' — reject (SCOPE: sports result).\n"
    "- 'a player failed a doping test' — keep (sports, but a crime/integrity matter).\n"
    "- '@Telco my internet has been down a week, refund me' — reject (SCOPE: personal customer-service).\n"
    "- 'Channel migrants are 24x more likely to be jailed' — keep (immigration statistic, in scope).\n"
    "- 'the media never tell the truth' — reject (RISK: unfalsifiable rhetoric, no checkable fact).\n"
    "- a screenshot of a fabricated tweet from a head of state announcing a policy — keep (politics; the "
    "claim lives in the image).\n\n"
    "WORKED (input → exact output; reasoning walks scope; risk; verdict)\n"
    "POST: 'Pop star A claps back at rapper B in their feud' →\n"
    '{"reasoning":"Scope: celebrity feud, entertainment — out of scope. Risk: specific but no societal consequence. Verdict: not check-worthy.","public_interest":false,"misinfo_if_false":false,"claim_locus":"text","check_worthy":false}\n'
    "POST: '(no caption)' + image = screenshot of a fabricated tweet from a head of state announcing a policy →\n"
    '{"reasoning":"Scope: a head of state announcing policy — politics, in scope. Risk: a specific claim that misleads if fabricated, living in the image. Verdict: check-worthy.","public_interest":true,"misinfo_if_false":true,"claim_locus":"image","check_worthy":true}\n\n'
    "OUTPUT\n"
    "Identify claim_locus — where the checkable claim lives: text | image | both | parent | video | none.\n"
    'Keep "reasoning" to 1-2 short sentences (max 50 words) naming the scope, then the risk, then the '
    "verdict. Output strictly valid JSON and nothing else:\n"
    '{"reasoning":"<scope; risk; verdict>","public_interest":true,"misinfo_if_false":true,"claim_locus":"text|image|both|parent|video|none","check_worthy":true}\n'
    "check_worthy MUST equal (public_interest AND misinfo_if_false)."
)


def _obj(txt: str) -> dict:
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _gate(row: dict) -> dict:
    content = []
    paths = row.get("image_paths") or []
    toks = row.get("image_est_tokens") or [None] * len(paths)
    for i, (p, t) in enumerate(zip(paths, toks)):
        if t is not None and t < MIN_TOKENS:
            continue
        if not Path(p).exists():
            continue
        b = base64.b64encode(Path(p).read_bytes()).decode()
        content.append({"type": "text", "text": f"Image {i+1} (post image):"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b}"}})
    txt = f"POST TEXT: {row.get('text') or row.get('raw_context') or '(none)'}"
    if row.get("is_quote") and row.get("quoted_text"):
        txt += f"\n\nQUOTED TWEET (the user is amplifying/endorsing/replying to this): {row['quoted_text']}"
    if row.get("has_video"):
        txt += "\n\n[NOTE: this post includes a VIDEO; only a poster frame is available — if the main claim would live in the video, set claim_locus='video'.]"
    content.append({"type": "text", "text": txt})
    body = {"model": GATE_MODEL,
            "messages": [{"role": "system", "content": GATE_SYSTEM}, {"role": "user", "content": content}],
            "temperature": 0, "response_format": {"type": "json_object"}, "max_tokens": MAX_TOKENS}
    g: dict = {}
    for _ in range(2):  # one retry: temp=0 output occasionally runs long/unparseable — don't lose the row
        r = requests.post(URL, headers=HDR, json=body, timeout=120)
        r.raise_for_status()
        g = _obj(r.json()["choices"][0]["message"]["content"])
        if g:
            break
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--expect", choices=["pass", "reject"], required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--tag", default="", help="suffix for the output file; namespaces a prompt version so "
                    "the resumable cache doesn't return a prior prompt's results")
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    out = Path(f"eval/data/gate_eval_{Path(args.parquet).stem}{suffix}.parquet")
    rows = pl.read_parquet(args.parquet).to_dicts()
    if args.max:
        rows = rows[: args.max]
    # prefer an explicit uid so image-only rows (empty post text) don't collapse under dedup-on-key
    key = "uid" if "uid" in rows[0] else ("text" if "text" in rows[0] else "raw_context")
    # resume on SUCCESSFUL rows only — errored rows (kept null, e.g. an API 402) must retry, not be skipped
    done: set = set()
    if out.exists():
        prev = pl.read_parquet(out)
        if key in prev.columns:
            done = set(prev.filter(pl.col("kept").is_not_null())[key].to_list()) if "kept" in prev.columns else set(prev[key].to_list())
    todo = [i for i, r in enumerate(rows) if (r.get(key)) not in done]
    print(f"{len(rows)} rows | todo {len(todo)} | expect={args.expect}", flush=True)
    res: dict = {}

    def flush():
        if res:
            new = pl.DataFrame(list(res.values()))
            comb = new if not out.exists() else pl.concat([pl.read_parquet(out), new], how="diagonal_relaxed").unique(key, keep="last")
            comb.write_parquet(out)

    def work(i):
        r = rows[i]
        try:
            g = _gate(r)
            kept = bool(g.get("check_worthy")) and g.get("claim_locus") != "video"
            return i, {key: r.get(key), "category": r.get("category"), "lang": r.get("lang"),
                       "post_text": (r.get("text") or r.get("raw_context") or "")[:200],
                       "public_interest": g.get("public_interest"), "misinfo_if_false": g.get("misinfo_if_false"),
                       "claim_locus": g.get("claim_locus"), "check_worthy": g.get("check_worthy"),
                       "kept": kept, "reasoning": (g.get("reasoning") or "")[:300]}
        except Exception as e:
            return i, {key: r.get(key), "category": r.get("category"), "kept": None, "reasoning": f"ERR {e}"[:200]}

    def apply(i, p):
        res[rows[i].get(key)] = p

    ab = pooled_checkpointed(todo, work, apply, flush, args.workers, "gate", checkpoint_every=50)
    flush()

    df = pl.read_parquet(out)
    n = df.filter(pl.col("kept").is_not_null()).height
    errs = df.filter(pl.col("kept").is_null()).height
    if n == 0:
        print(f"\n0 rows evaluated — all {errs} errored (check API balance/limits; re-run to retry).")
        sys.stdout.flush(); os._exit(0)
    kept = df.filter(pl.col("kept") == True).height  # noqa: E712
    suffix = f"  | {errs} errored (re-run to retry)" if errs else ""
    if args.expect == "reject":
        print(f"\nFALSE POSITIVES (gate kept a negative): {kept}/{n} = {kept/n*100:.1f}%{suffix}")
        print("by category:", df.filter(pl.col("kept") == True).group_by("category").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())  # noqa: E712
    else:
        print(f"\nRECALL (gate kept a positive): {kept}/{n} = {kept/n*100:.1f}%  | FALSE NEGATIVES: {n-kept}{suffix}")
    if ab:
        sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
