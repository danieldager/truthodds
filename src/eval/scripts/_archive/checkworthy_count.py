"""Count how many saved X posts are 'check-worthy' (contain ≥1 verifiable claim).

Reads a Zeeschuimer X/Twitter NDJSON export, builds the reader-visible text for
each post (own text + long-form note-tweet + quoted-tweet text + link-card
title/description), and runs the v0.1 decomposition prompt via Groq to get the
`is_checkable` flag + atomic claims. The questions/queries/embedding steps are
dropped — they don't affect the check-worthiness decision (matches the v0.2
extract() intent).

Staged usage (cautious rollout):
    uv run python -m scripts.checkworthy_count --ndjson <path> --limit 5    # smoke
    uv run python -m scripts.checkworthy_count --ndjson <path> --limit 50   # batch
    uv run python -m scripts.checkworthy_count --ndjson <path>              # full
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

from openai import OpenAI

from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL, EXTRACTION_MODEL

_client = OpenAI(base_url=EXTRACTION_BASE_URL, api_key=EXTRACTION_API_KEY)

# v0.1 _SYSTEM, trimmed to the check-worthiness decision (queries section removed,
# schema reduced to {is_checkable, claims:[{text}]}). Rules kept verbatim.
_SYSTEM = """You are a fact-checking assistant. Given a social media post, determine whether it contains verifiable factual claims.

If it does, decompose the post into atomic claims — each a single, independently checkable statement.

Rules for is_checkable:
- Set is_checkable to TRUE whenever the post contains ANY factual assertion about the real world
- Tone and framing are irrelevant: accusatory, sarcastic, emotional, or rhetorical phrasing does NOT make a claim uncheckable
- The following are ALL checkable: what someone said or did, historical events, statistics, scientific claims, allegations, accusations, conspiracy theories, medical claims, election claims, evidence-based claims
- Set is_checkable to FALSE ONLY when the entire post has ZERO factual content — i.e., it is purely a personal preference, a question with no embedded assertion, a future prediction, or an emotional expression

Checkable examples (is_checkable = true):
- "You are watching the cheaters sending in phony ballots" → asserts voter fraud occurred → checkable
- "The evidence shows masks don't stop aerosol transmission" → claim about evidence and masks → checkable
- "The earth is flat" → obviously false, but still a factual assertion → checkable
- "They've been lying to you about the economy" → checkable allegation

Not checkable examples (is_checkable = false):
- "I love this country!" → pure personal sentiment, no factual content
- "Will Biden ban fracking?" → pure question, no assertion
- "Praying for everyone affected" → emotional expression, no factual claim

Rules for claims:
- Each claim must be a single, standalone factual assertion
- Ignore pure opinions/predictions/questions — but NOT false, absurd, or emotionally framed assertions
- Do NOT pre-judge whether a claim is true or false — that is determined in a later step
- A post may yield 0, 1, or several atomic claims

Respond ONLY with valid JSON matching this schema:
{
  "is_checkable": true,
  "claims": [
    {"text": "<atomic claim>"}
  ]
}

If there are no checkable claims, respond with:
{"is_checkable": false, "claims": []}
"""


def _post_text(d: dict) -> str:
    """Long-form note-tweet text if present, else legacy.full_text."""
    nt = (((d.get("note_tweet") or {}).get("note_tweet_results") or {}).get("result") or {}).get("text")
    if nt:
        return nt
    return (d.get("legacy") or {}).get("full_text", "") or ""


def _card_snippet(d: dict) -> str:
    """Link-card title/description/domain, when the card is hydrated (rare)."""
    card = d.get("card") or {}
    bv = (card.get("legacy") or card).get("binding_values")
    if not isinstance(bv, list):
        return ""
    vals = {}
    for item in bv:
        if isinstance(item, dict) and "key" in item:
            v = item.get("value") or {}
            vals[item["key"]] = v.get("string_value")
    title, desc, dom = vals.get("title"), vals.get("description"), vals.get("domain")
    if not (title or desc):
        return ""
    bits = [b for b in (title, desc) if b]
    snip = " — ".join(bits)
    return f"{snip} ({dom})" if dom else snip


def reader_visible_text(d: dict) -> str:
    """Post text + note-tweet long-form + quoted-tweet text + link-card snippet."""
    parts = [_post_text(d)]
    q = (d.get("quoted_status_result") or {}).get("result") or {}
    if q:
        qt = _post_text(q)
        if qt:
            parts.append(f"[Quoted post] {qt}")
    card = _card_snippet(d)
    if card:
        parts.append(f"[Linked article] {card}")
    return "\n".join(p for p in parts if p).strip()


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def classify(text: str, retries: int = 3) -> dict:
    """Return {is_checkable: bool, claims: [str]}; raises after exhausting retries."""
    last = None
    for attempt in range(retries):
        try:
            resp = _client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"Post:\n{text}\n\n/no_think"},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            data = _parse_json(resp.choices[0].message.content)
            claims = [c.get("text", "") for c in data.get("claims", []) if c.get("text")]
            checkable = bool(data.get("is_checkable", False)) and bool(claims)
            return {"is_checkable": checkable, "claims": claims}
        except Exception as e:  # noqa: BLE001 — record and back off
            last = e
            sleep(2 * (attempt + 1))
    raise last


def load_records(path: str, limit: int | None) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap records (smoke/batch)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="checkworthy_results.csv")
    args = ap.parse_args()

    if not EXTRACTION_API_KEY:
        sys.exit("GROQ_API_KEY is empty — set it in src/.env")

    records = load_records(args.ndjson, args.limit)
    print(f"Loaded {len(records)} posts. Model={EXTRACTION_MODEL}. Workers={args.workers}.")

    def work(idx_d):
        i, d = idx_d
        lg = d.get("legacy") or {}
        text = reader_visible_text(d)
        row = {
            "id": d.get("rest_id") or lg.get("id_str", ""),
            "lang": lg.get("lang", ""),
            "n_chars": len(text),
            "is_checkable": "",
            "n_claims": 0,
            "error": "",
            "text_preview": re.sub(r"\s+", " ", text)[:160],
            "claims": "",
        }
        if not text:
            row["error"] = "empty_text"
            return i, row
        try:
            res = classify(text)
            row["is_checkable"] = res["is_checkable"]
            row["n_claims"] = len(res["claims"])
            row["claims"] = " | ".join(res["claims"])
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)[:200]
        return i, row

    results: list[dict] = [None] * len(records)  # type: ignore
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, (i, d)) for i, d in enumerate(records)]
        for f in as_completed(futs):
            i, row = f.result()
            results[i] = row
            done += 1
            if done % 25 == 0 or done == len(records):
                print(f"  {done}/{len(records)} processed")

    cols = ["id", "lang", "is_checkable", "n_claims", "error", "text_preview", "claims", "n_chars"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(results)

    n = len(results)
    errs = sum(1 for r in results if r["error"])
    ok = [r for r in results if not r["error"]]
    cw = [r for r in ok if r["is_checkable"] is True]
    by_lang = Counter(r["lang"] for r in cw)
    by_lang_total = Counter(r["lang"] for r in ok)
    total_claims = sum(r["n_claims"] for r in ok)

    print("\n==== CHECK-WORTHINESS SUMMARY ====")
    print(f"posts processed:      {n}")
    print(f"errors:               {errs}")
    print(f"check-worthy:         {len(cw)} / {len(ok)}  ({100*len(cw)/max(len(ok),1):.1f}% of clean)")
    print(f"total atomic claims:  {total_claims}")
    print("check-worthy by lang (checkworthy / total):")
    for lang, tot in by_lang_total.most_common():
        print(f"  {lang or '∅':<5} {by_lang.get(lang,0):4d} / {tot:<4d}")
    print(f"\nCSV: {args.out}")


if __name__ == "__main__":
    main()
