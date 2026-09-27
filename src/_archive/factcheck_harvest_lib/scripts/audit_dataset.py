"""Full hand-audit of the Stage-1 gate dataset — every positive and every negative — to find label
contamination (out-of-scope fact-checks parked as positives; check-worthy posts parked as negatives;
junk/unreadable rows). Independent multimodal judge = Qwen3-VL-235B (a DIFFERENT, larger model than the
30B gate, so it's not the gate grading itself; the HUMAN reviews every borderline/junk case after).

  uv run python -m eval.scripts.audit_dataset            ->  eval/data/dataset_audit.parquet

Objective rubric per post (sees text AND image): topic, in_scope, has_claim, readable, recommend
(positive|negative|drop), confidence (high|borderline), note. Resumable / high-visibility (pooled).
All downstream relabels are derived from this parquet + the human's borderline calls, then logged in the
provenance ledger — reproducible.
"""
from __future__ import annotations

import base64
import glob
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
AUDIT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"

AUDIT_SYSTEM = (
    "You audit a dataset for a misinformation fact-checking gate. Assess each social-media post OBJECTIVELY "
    "— do not judge whether its claim is true.\n\n"
    "The gate's SCOPE is matters of SOCIETAL CONSEQUENCE: politics & elections, government policy/spending, "
    "the economy, corruption & bribery, war & geopolitics, terrorism, immigration, public health & safety, "
    "crime, disasters, science & climate. OUT of scope: sports results/transfers, entertainment / celebrity "
    "/ music, consumer products / brands / business gossip, personal life, jokes, trivia — UNLESS the post "
    "carries a societal-consequence angle (a death, crime, corruption, public-health/safety issue, an "
    "election, or public money).\n\n"
    "Read the post TEXT and any IMAGE — the checkable claim often lives in the image (a screenshot, a "
    "fabricated or quoted headline, a miscaptioned or manipulated photo). Then output strict JSON only:\n"
    '{"topic":"<short topic, e.g. politics|election|health|crime|war|sports|celebrity|product|business|personal|other>",'
    '"in_scope":true|false,'              # societal consequence per the scope above
    '"has_claim":true|false,'             # asserts a SPECIFIC, checkable factual claim (in text OR image)
    '"claim_locus":"text|image|both|none",'
    '"readable":true|false,'              # legible / has interpretable content (false = blank, garbled, link-only, no real content)
    '"recommend":"positive|negative|drop",'  # positive = in_scope AND has_claim; negative = readable but out_of_scope OR no checkable claim; drop = unreadable / no real content / not a gateable post
    '"confidence":"high|borderline",'     # borderline if the scope or claim call is genuinely uncertain
    '"note":"<one short sentence>"}'
)


def _obj(txt: str) -> dict:
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S), re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _audit(row: dict) -> dict:
    content = []
    for p in (row.get("image_paths") or [])[:2]:
        if p and Path(p).exists():
            b = base64.b64encode(Path(p).read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b}"}})
    content.append({"type": "text", "text": f"POST TEXT: {row.get('text') or '(none)'}"})
    body = {"model": AUDIT_MODEL, "messages": [{"role": "system", "content": AUDIT_SYSTEM},
            {"role": "user", "content": content}], "temperature": 0,
            "response_format": {"type": "json_object"}, "max_tokens": 400}
    g: dict = {}
    for _ in range(2):
        r = requests.post(URL, headers=HDR, json=body, timeout=120)
        r.raise_for_status()
        g = _obj(r.json()["choices"][0]["message"]["content"])
        if g:
            break
    return g


def build_pool() -> list[dict]:
    rows = []
    # POSITIVES: every fact-checked image post (text-bearing + image-only), non-video
    for f in sorted(glob.glob("eval/data/*_harvest.parquet")):
        d = pl.read_parquet(f)
        if "has_image" not in d.columns:
            continue
        sub = d.filter((pl.col("has_image") == True) & (pl.col("has_video") != True))  # noqa: E712
        src = Path(f).stem.replace("_harvest", "")
        for r in sub.to_dicts():
            rows.append({"text": r.get("raw_context") or "", "image_paths": r.get("image_paths"),
                         "source": f"fc:{src}", "current_label": "positive",
                         "uid": (r.get("image_paths") or [None])[0] or f"pos::{r.get('raw_context')}"})
    # NEGATIVES: synthetic set (text-mostly; a few decorative images)
    for r in pl.read_parquet("eval/data/synthetic_negatives.parquet").to_dicts():
        rows.append({"text": r.get("text") or "", "image_paths": r.get("image_paths"),
                     "source": f"syn:{r.get('source')}", "current_label": "negative",
                     "uid": f"neg::{r.get('text')}"})
    # de-dup on uid
    seen, out = set(), []
    for r in rows:
        if r["uid"] in seen:
            continue
        seen.add(r["uid"]); out.append(r)
    return out


def main() -> None:
    rows = build_pool()
    out = Path("eval/data/dataset_audit.parquet")
    done = set()
    if out.exists():
        prev = pl.read_parquet(out)
        if "recommend" in prev.columns:
            done = set(prev.filter(pl.col("recommend").is_not_null())["uid"].to_list())
    todo = [i for i, r in enumerate(rows) if r["uid"] not in done]
    pos = sum(1 for r in rows if r["current_label"] == "positive")
    print(f"{len(rows)} items ({pos} positives / {len(rows)-pos} negatives) | todo {len(todo)}", flush=True)
    res: dict = {}

    def flush():
        if res:
            new = pl.DataFrame(list(res.values()))
            comb = new if not out.exists() else pl.concat([pl.read_parquet(out), new], how="diagonal_relaxed").unique("uid", keep="last")
            comb.write_parquet(out)

    def work(i):
        r = rows[i]
        try:
            g = _audit(r)
            return i, {"uid": r["uid"], "current_label": r["current_label"], "source": r["source"],
                       "text": (r["text"] or "")[:300], "image_path": (r.get("image_paths") or [None])[0],
                       "topic": g.get("topic"), "in_scope": g.get("in_scope"), "has_claim": g.get("has_claim"),
                       "claim_locus": g.get("claim_locus"), "readable": g.get("readable"),
                       "recommend": g.get("recommend"), "confidence": g.get("confidence"),
                       "note": (g.get("note") or "")[:200]}
        except Exception as e:
            return i, {"uid": r["uid"], "current_label": r["current_label"], "recommend": None, "note": f"ERR {e}"[:150]}

    ab = pooled_checkpointed(todo, work, lambda i, p: res.__setitem__(rows[i]["uid"], p), flush, 8, "audit", checkpoint_every=100)
    flush()

    df = pl.read_parquet(out)
    ev = df.filter(pl.col("recommend").is_not_null())
    print(f"\naudited {ev.height} | errored {df.height - ev.height}")
    # disagreements + flags by current label
    for lab in ["positive", "negative"]:
        s = ev.filter(pl.col("current_label") == lab)
        mism = s.filter(pl.col("recommend") != lab)
        drop = s.filter(pl.col("recommend") == "drop")
        bord = s.filter(pl.col("confidence") == "borderline")
        print(f"{lab}: {s.height} | recommend≠label {mism.height} | drop {drop.height} | borderline {bord.height}")
    if ab:
        sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
