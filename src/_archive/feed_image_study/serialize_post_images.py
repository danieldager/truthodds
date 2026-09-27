"""WS1 — serialize the post images in the verdict dataset for the text-only verifier.

The Tier-3 verifier is text-only; for claims that live in or depend on a post's image, this
turns the image into text the verifier can read (fed later as `image_context`, see
`pipeline/verify.py`). Guardrails (verdict_eval_plan.md WS1):
  - POST IMAGE ONLY (image_paths) — never the fact-check article's graphics (verdict leakage).
  - CONTENT-ONLY, non-judgmental — transcribe in-image text + describe visuals; NO verdict
    language ("false", "altered", "debunked"), NO guessed identities.
Cached per (review_url, model) → resumable, computed once. Output column: `image_serialization`.

  uv run python -m eval.scripts.serialize_post_images --split dev --max 3   # smoke
  uv run python -m eval.scripts.serialize_post_images --split dev
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
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # the locked Stage-2 production VLM
OUT = Path("eval/data/verdict_image_serialization.parquet")
MAX_IMAGES = 4

SERIALIZE_SYS = """You transcribe a social-media post's image(s) into text for a BLIND downstream fact-checker — a text-only model that cannot see the image. Describe ONLY the image's check-worthy content, exactly as it appears.

<rules>
- Transcribe ALL salient in-image text in full and verbatim (full transcription for text-heavy screenshots / statement cards; a line for a simple photo).
- Describe key visuals with as much detail as the checkable content warrants. Describe NON-JUDGMENTALLY.
- Use ONLY what is visible. Do NOT infer absent facts, and do NOT use outside knowledge.
- Identity: name a depicted person/place/logo only when it is confidently identifiable from in-image text, a caption, a watermark, or unambiguous on-image branding; otherwise describe generically (e.g. "an older man in a dark suit"). NEVER guess a name.
- Do NOT judge truth: no "false", "fake", "altered", "AI-generated", "debunked", "misleading" — just describe what is shown. Whether the image is authentic is decided elsewhere.
- If in-image text is not legible or a detail is not visible, say so — do not invent it.
</rules>

Output strictly valid JSON and nothing else: {"image_serialization": "<the description; '' if the image carries no checkable content>"}"""


def _obj(txt: str) -> dict:
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # salvage a single string field if the JSON was truncated mid-object
    m = re.search(r'"image_serialization"\s*:\s*"((?:[^"\\]|\\.)*)"', txt, re.S)
    if m:
        try:
            return {"image_serialization": json.loads('"' + m.group(1) + '"')}
        except json.JSONDecodeError:
            pass
    return {}


def _content(paths: list[str]) -> list:
    parts: list = [{"type": "text", "text":
                    "Serialize the following post image(s) for a blind fact-checker."}]
    for i, p in enumerate(paths):
        b = base64.b64encode(Path(p).read_bytes()).decode()
        parts.append({"type": "text", "text": f"Image {i + 1}:"})
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b}"}})
    return parts


def _serialize(model: str, paths: list[str]) -> dict:
    r = requests.post(URL, headers=HDR, json={
        "model": model,
        "messages": [{"role": "system", "content": SERIALIZE_SYS},
                     {"role": "user", "content": _content(paths)}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }, timeout=180)
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    obj = _obj(ch["message"]["content"])
    return {"image_serialization": (obj.get("image_serialization") or "")[:2000],
            "finish": ch.get("finish_reason"),
            "tokens": (j.get("usage") or {}).get("completion_tokens")}


def _rows(split: str, locus_first: bool) -> list[dict]:
    df = pl.read_parquet("eval/data/verdict_dataset.parquet").filter(pl.col("has_image"))
    if split != "all":
        df = df.filter(pl.col("split") == split)
    # prioritize claim-in-image rows (they NEED the picture) via the Stage-1 image audit
    if locus_first and Path("eval/data/image_audit.parquet").exists():
        aud = pl.read_parquet("eval/data/image_audit.parquet").select("review_url", "claim_location")
        df = df.join(aud, on="review_url", how="left").with_columns(
            pl.col("claim_location").fill_null("text"))
        df = df.with_columns(
            pl.col("claim_location").is_in(["image", "both"]).cast(pl.Int8).alias("_pri")
        ).sort("_pri", descending=True)
    return df.to_dicts()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max", type=int, default=None, help="cap rows (smoke)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--locus-first", action="store_true",
                    help="serialize claim-in-image rows first (image_audit claim_location)")
    args = ap.parse_args()

    rows = _rows(args.split, args.locus_first)
    if args.max:
        rows = rows[: args.max]

    done: set[str] = set()
    if OUT.exists():
        prev = pl.read_parquet(OUT)
        done = set(prev.filter(pl.col("model") == args.model)["review_url"].to_list())
    todo = [i for i, r in enumerate(rows) if r["review_url"] not in done]
    print(f"split={args.split} rows={len(rows)} | cached(model)={len(done)} | todo={len(todo)} | {args.model}",
          flush=True)
    if not todo:
        print("nothing to do.")
        return

    res: dict[str, dict] = {}

    def work(i):
        r = rows[i]
        paths = [p for p in (r.get("image_paths") or []) if p and os.path.exists(p)][:MAX_IMAGES]
        try:
            if not paths:
                return i, {"image_serialization": "", "finish": "no_image", "tokens": None, "error": "no image files"}
            return i, {**_serialize(args.model, paths), "error": None}
        except Exception as e:  # noqa: BLE001
            return i, {"image_serialization": "", "finish": "error", "tokens": None, "error": str(e)[:200]}

    def apply(i, p):
        r = rows[i]
        res[r["review_url"]] = {"review_url": r["review_url"], "model": args.model,
                               "n_images": len([x for x in (r.get("image_paths") or []) if x]),
                               "source": r.get("source"), "split": r.get("split"), **p}

    def flush():
        if not res:
            return
        new = pl.DataFrame(list(res.values()))
        comb = new if not OUT.exists() else pl.concat([pl.read_parquet(OUT), new], how="diagonal_relaxed") \
            .unique(["review_url", "model"], keep="last")
        comb.write_parquet(OUT)

    ab = pooled_checkpointed(todo, work, apply, flush, args.workers,
                             f"serialize:{args.model.split('/')[-1]}", checkpoint_every=25)
    flush()

    final = pl.read_parquet(OUT).filter(
        (pl.col("model") == args.model) & pl.col("review_url").is_in([r["review_url"] for r in rows]))
    errs = final.filter(pl.col("error").is_not_null()).height
    empty = final.filter(pl.col("image_serialization") == "").height
    lens = final.filter(pl.col("image_serialization") != "")["image_serialization"].str.len_chars()
    print(f"\ndone. rows={final.height}  errors={errs}  empty={empty}  -> {OUT}")
    if lens.len():
        print(f"serialization length chars: median={int(lens.median())}  min={lens.min()}  max={lens.max()}")
    if ab:
        sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
