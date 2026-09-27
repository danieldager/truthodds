"""Image-dependence gate + legibility/alignment audit for the image-inclusive eval dataset.

  uv run python -m eval.scripts.image_audit --max 10        # smoke
  uv run python -m eval.scripts.image_audit                 # full

For every harvested row that has an image, a Qwen3-VL (DeepInfra) judge is shown the post TEXT, all
of the post's IMAGES, and the fact-checker's GOLD CLAIM, and decides WHERE the claim's checkable
content lives — the image-dependence gate that defines the image-native eval subset (clog 260626):

  claim_location: text | image | both | neither   (image/both = needs the picture)
  image_role:     carries_claim | adds_detail | decorative | authenticity_subject
  legible:        is in-image text readable at the saved resolution (A4 legibility check)
  image_text:     verbatim text the VLM read off the image (audit trail)

Writes a SEPARATE eval/data/image_audit.parquet keyed by review_url — it does NOT modify the harvest
parquets (keep the harvest immutable; join the gate back when building the deliverable). Per-image
records skip the degenerate tail (est_tokens < MIN_TOKENS — tiny og: thumbnails / avatars / logos).
Crash-safe/resumable via the shared pool driver (same as fetch_sources/fetch_images).
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
from pathlib import Path

import polars as pl
import requests

from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL
from eval.scripts._pool import pooled_checkpointed

MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
MIN_TOKENS = 150  # skip degenerate images (~<384px): logos / avatars / blank thumbnails
OUT = Path("eval/data/image_audit.parquet")

SYS = (
    "You audit an image-native fact-check extraction dataset. Given a social post's TEXT, its "
    "IMAGE(S), and the GOLD CLAIM a fact-checker extracted, decide where the claim's checkable "
    'content lives. Respond JSON only: {"claim_location":"text|image|both|neither",'
    '"image_role":"carries_claim|adds_detail|decorative|authenticity_subject",'
    '"legible":true|false,"image_text":"<verbatim text visible in the image(s), <=160 chars, '
    'empty if none>","reason":"<=20 words"}. '
    "claim_location=image when the post text does NOT state the claim but the image does; "
    "text when the text alone states it; both when each does; neither when the claim is in neither. "
    "image_role=authenticity_subject when the claim is about whether the image/video itself is real."
)


def _gate(row: dict, model: str = MODEL) -> dict:
    imgs = [p for p, t in zip(row["image_paths"], row["image_est_tokens"] or [])
            if t and t >= MIN_TOKENS][:4]
    if not imgs:
        return {"claim_location": None, "image_role": "degenerate", "legible": False,
                "image_text": "", "reason": "all images below the legibility floor"}
    content = [{"type": "text",
                "text": f"POST TEXT: {(row['raw_context'] or '(no post text)')[:500]}\n\nGOLD CLAIM: {row['claim_text']}"}]
    for p in imgs:
        b = base64.b64encode(Path(p).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b}"}})
    r = requests.post(
        f"{EXTRACTION_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {EXTRACTION_API_KEY}"},
        json={"model": model, "messages": [{"role": "system", "content": SYS},
                                           {"role": "user", "content": content}],
              "temperature": 0, "response_format": {"type": "json_object"}, "max_tokens": 300},
        timeout=120)
    r.raise_for_status()
    v = json.loads(r.json()["choices"][0]["message"]["content"] or "{}")
    return {"claim_location": v.get("claim_location"), "image_role": v.get("image_role"),
            "legible": bool(v.get("legible")), "image_text": (v.get("image_text") or "")[:200],
            "reason": (v.get("reason") or "")[:200]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max", type=int, default=None, help="cap rows (smoke test)")
    args = ap.parse_args()

    # gather every has_image row across the per-source harvests
    recs: list[dict] = []
    for f in sorted(glob.glob("eval/data/*_harvest.parquet")):
        src = f.split("/")[-1].split("_harvest")[0]
        df = pl.read_parquet(f)
        if "has_image" not in df.columns:
            continue
        for r in df.filter(pl.col("has_image") == True).select(  # noqa: E712
            "review_url", "claim_text", "raw_context", "raw_related", "judged_axis",
            "image_paths", "image_est_tokens").to_dicts():
            r["publisher_site"] = src
            recs.append(r)

    done = set(pl.read_parquet(OUT)["review_url"].to_list()) if OUT.exists() else set()
    todo_recs = [r for r in recs if r["review_url"] not in done]
    if args.max:
        todo_recs = todo_recs[: args.max]
    print(f"{len(recs)} image rows; {len(done)} already audited; {len(todo_recs)} to do", flush=True)
    if not todo_recs:
        print("nothing to do."); return

    results: dict[str, dict] = {}
    todo = list(range(len(todo_recs)))

    def flush():
        if not results:
            return
        new = pl.DataFrame([{"review_url": k, **v} for k, v in results.items()])
        combined = new if not OUT.exists() else pl.concat([pl.read_parquet(OUT), new], how="diagonal_relaxed").unique("review_url", keep="last")
        combined.write_parquet(OUT)

    def work(i):
        r = todo_recs[i]
        try:
            return i, _gate(r)
        except Exception as e:  # network / parse — record a miss, resumable
            return i, {"claim_location": None, "image_role": "error", "legible": False,
                       "image_text": "", "reason": str(e)[:120]}

    def apply(i, payload):
        r = todo_recs[i]
        results[r["review_url"]] = {**payload, "publisher_site": r["publisher_site"],
                                    "judged_axis": r["judged_axis"], "raw_related": r["raw_related"]}

    abandoned = pooled_checkpointed(todo, work, apply, flush, args.workers, "img-audit", checkpoint_every=50)
    flush()

    df = pl.read_parquet(OUT)
    print(f"\nWrote {OUT}: {df.height} audited rows")
    for col in ("claim_location", "image_role"):
        print(f"{col}:", df.group_by(col).agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())
    print("legible:", df.filter(pl.col("legible") == True).height, "/", df.height)  # noqa: E712
    img_dep = df.filter(pl.col("claim_location").is_in(["image", "both"])).height
    print(f"image-dependent (location in image/both): {img_dep}/{df.height}")
    if abandoned:
        sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
