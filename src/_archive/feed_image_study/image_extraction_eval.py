"""Stage-2 image-native extraction eval — does the VLM recover the fact-checker's gold claim?

  uv run python -m eval.scripts.image_extraction_eval --model Qwen/Qwen3-VL-30B-A3B-Instruct --mode single --split dev --max 10
  uv run python -m eval.scripts.image_extraction_eval --model Qwen/Qwen3-VL-30B-A3B-Instruct --mode multi  --split test

For each post in the CLEAN image-native subset (image-dependent per the gate, non-video, has gold),
a Qwen3-VL **extractor** sees the post text + ALL its images + the quoted tweet, and emits the
checkable claim(s) (Stage-2 of core_pipeline_spec.md). Then an INDEPENDENT **DeepSeek-V4-Flash judge**
scores whether the extraction recovers the fact-checker's gold `claim_text` — **framing-agnostic** (it
ignores fact-checker wrappers: "Image shows…", "Photos/videos show…", "X tweeted…", "authentically
shows") and **combination-aware** (a match may be carried by one claim OR several together).

Two extractor MODES (w/ Daniel 2026-06-26):
  - `single` — emit the ONE primary claim a fact-checker would target (model "thinks like a checker").
  - `multi`  — emit the few atomic claims (primary first); judge matches if any claim or a combination
               of them covers the gold.

DEV/TEST split (frozen, per-axis stratified by md5(review_url); DEV_FRAC=0.4): iterate the prompt on
`--split dev`, report final ladder numbers on the held-out `--split test` so the prompt isn't overfit
to the only dataset we have.

This eval scores **content+attribution only**. The **artifact (authenticity) axis is EXCLUDED** — its
gold is an authenticity verdict ("photo authentically shows X"), not an extractable proposition; it's a
separate **image-authentication task (Phase 1b reverse-image)** with its own eval, so it's neither
processed nor scored here (the artifact rows stay derivable from `image_audit.parquet` as that task's
seed). Extractor ≠ judge (Qwen vs DeepSeek). Crash-safe/resumable; writes
`extraction_eval_<model>_<mode>.parquet`. Ladder = 30B-A3B vs 235B; read the cost/quality knee off the
held-out test split.
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import polars as pl
import requests

from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL, VERIFICATION_MODEL
from eval.scripts._pool import pooled_checkpointed

URL = f"{EXTRACTION_BASE_URL}/chat/completions"
HDR = {"Authorization": f"Bearer {EXTRACTION_API_KEY}"}
MIN_TOKENS = 150
DEV_FRAC = 0.4
HEADLINE_AXES = ["content", "attribution"]
JUDGE_MODEL = VERIFICATION_MODEL  # DeepSeek-V4-Flash, text-only; independent of the Qwen extractor

# Identity policy (w/ Daniel): resolve a depicted person/place to a specific name ONLY when confident
# (in-image text, caption, quoted tweet, or clear recognition); otherwise describe generically and
# record it in "flags" — the honest output when the identity is implicit and not recoverable.
_IDENTITY = (
    "Resolve named entities (no 'this/they/here'). Resolve a depicted person/place/object to a "
    "specific NAME only when confident — from in-image text, the caption, the quoted tweet, or clear "
    "recognition; if you cannot, describe them generically and add a short note to \"flags\" (e.g. "
    "'unresolved identity: man dining with a woman'). Never guess a name."
)

EXTRACT_SYS_SINGLE = (
    "You extract THE single checkable claim a social-media post puts forward — the one proposition a "
    "fact-checker would verify. You see the post text, its image(s), and any quoted tweet. "
    "(1) Briefly transcribe the salient in-image text and NON-judgmentally describe the key visuals "
    "(do NOT assess truth). (2) Identify the ONE primary, most salient/disputed proposition the post "
    "asserts; it may live in the post text, the image (in-image text or visual), or an endorsed quoted "
    "tweet. State it as a single atomic DECONTEXTUALIZED proposition — the bare claim, NOT wrapped as "
    "'the image shows…' or 'a post claims…'. " + _IDENTITY + " "
    'JSON only: {"image_serialization":"...","primary_claim":"...","flags":["..."]}'
)

EXTRACT_SYS_MULTI = (
    "You extract the checkable claims a social-media post puts forward, for fact-checking. You see the "
    "post text, its image(s), and any quoted tweet. (1) Briefly transcribe the salient in-image text "
    "and NON-judgmentally describe the key visuals (do NOT assess truth). (2) Extract the post's "
    "checkable propositions as atomic, DECONTEXTUALIZED, single-proposition claims, ORDERED "
    "most-salient/disputed FIRST. A claim may live in the post text, the image, or an endorsed quoted "
    "tweet. Emit only the few claims a fact-checker would actually check — do NOT fragment one claim "
    "into many trivial pieces (aim for ≤5). State bare propositions, NOT 'the image shows…'. "
    + _IDENTITY + " "
    'JSON only: {"image_serialization":"...","claims":[{"normalized_claim":"...","locus":"text|image|parent"}],"flags":["..."]}'
)

JUDGE_SYS = (
    "You judge whether the claim(s) EXTRACTED from a post recover the GOLD claim a fact-checker wrote, "
    "on the core checkable PROPOSITION. Be FRAMING-AGNOSTIC: ignore fact-checker wrappers on the gold "
    "('Image shows…', 'Photos/videos show…', 'A post claims…', 'X tweeted…', 'authentically/falsely "
    "shows', 'accurately showing') — the extracted claims are the bare content, the gold may wrap it. "
    "Be COMBINATION-AWARE: a match may be carried by a SINGLE extracted claim OR by SEVERAL of them "
    "together. "
    'JSON only: {"verdict":"match|partial|miss","matched_index":<int|null>,"reason":"<=25 words"}. '
    "match = the gold proposition is captured (by one claim or a combination); "
    "partial = related but missing or adding a key element; "
    "miss = no extracted claim(s) capture the gold."
)


def _salvage(txt: str) -> dict:
    """Recover fields from truncated/invalid JSON (the model hit the token cap mid-object)."""
    def _s(pat):
        m = re.search(pat, txt, re.S)
        if not m:
            return None
        try:
            return json.loads('"' + m.group(1) + '"')
        except json.JSONDecodeError:
            return None
    out: dict = {}
    ser = _s(r'"image_serialization"\s*:\s*"((?:[^"\\]|\\.)*)"')
    if ser is not None:
        out["image_serialization"] = ser
    pc = _s(r'"primary_claim"\s*:\s*"((?:[^"\\]|\\.)*)"')
    if pc is not None:
        out["primary_claim"] = pc
    claims = []
    for m in re.finditer(r'"normalized_claim"\s*:\s*"((?:[^"\\]|\\.)*)"', txt, re.S):
        try:
            claims.append({"normalized_claim": json.loads('"' + m.group(1) + '"')})
        except json.JSONDecodeError:
            pass
    if claims:
        out["claims"] = claims
    return out


def _obj(txt: str) -> dict:
    """Outermost JSON object (strips ```/<think>); salvage partial fields on truncation; {} on failure."""
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return _salvage(txt)


def _chat(model: str, messages: list, max_tokens: int = 900, timeout: int = 120) -> dict:
    r = requests.post(URL, headers=HDR, json={"model": model, "messages": messages,
                      "temperature": 0, "response_format": {"type": "json_object"},
                      "max_tokens": max_tokens}, timeout=timeout)
    r.raise_for_status()
    return _obj(r.json()["choices"][0]["message"]["content"])


def _extract(model: str, row: dict, mode: str) -> dict:
    imgs = [p for p, t in zip(row["image_paths"], row["image_est_tokens"] or [])
            if t and t >= MIN_TOKENS][:4]
    content = [{"type": "text", "text":
                f"POST TEXT: {(row['raw_context'] or '(none)')[:600]}"
                + (f"\nQUOTED TWEET: {row['quoted_text'][:400]}" if row.get("quoted_text") else "")}]
    for p in imgs:
        b = base64.b64encode(Path(p).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b}"}})
    sys = EXTRACT_SYS_SINGLE if mode == "single" else EXTRACT_SYS_MULTI
    return _chat(model, [{"role": "system", "content": sys}, {"role": "user", "content": content}],
                 max_tokens=1800)


def _judge(gold: str, claims: list[str]) -> dict:
    if not claims:
        return {"verdict": "miss", "matched_index": None, "reason": "no claims extracted"}
    listing = "\n".join(f"[{i}] {c}" for i, c in enumerate(claims))
    return _chat(JUDGE_MODEL, [{"role": "system", "content": JUDGE_SYS},
                 {"role": "user", "content": f"GOLD CLAIM:\n{gold}\n\nEXTRACTED CLAIMS:\n{listing}"}],
                 max_tokens=400)


def subset() -> pl.DataFrame:
    a = pl.read_parquet("eval/data/image_audit.parquet")
    frames = []
    for f in glob.glob("eval/data/*_harvest.parquet"):
        df = pl.read_parquet(f)
        if "has_video" not in df.columns:
            continue
        frames.append(df.filter(pl.col("has_image") == True).select(  # noqa: E712
            "review_url", "claim_text", "raw_context", "quoted_text", "has_video",
            "image_paths", "image_est_tokens"))
    h = pl.concat(frames, how="diagonal_relaxed").unique("review_url")
    m = a.join(h, on="review_url", how="inner")
    return m.filter(pl.col("claim_location").is_in(["image", "both"]) & (pl.col("has_video") != True)  # noqa: E712
                    & ~pl.col("image_role").is_in(["degenerate", "error"]) & pl.col("claim_text").is_not_null())


def assign_split(rows: list[dict]) -> dict[str, str]:
    """Frozen per-axis stratified dev/test split keyed by md5(review_url) — deterministic, reproducible."""
    def h(u):
        return int(hashlib.md5(u.encode()).hexdigest(), 16) / 2 ** 128
    by = defaultdict(list)
    for r in rows:
        by[r.get("judged_axis")].append(r)
    out: dict[str, str] = {}
    for _, rs in by.items():
        rs = sorted(rs, key=lambda r: h(r["review_url"]))
        k = round(len(rs) * DEV_FRAC)
        for i, r in enumerate(rs):
            out[r["review_url"]] = "dev" if i < k else "test"
    return out


def _report(df: pl.DataFrame) -> None:
    df = df.filter(pl.col("verdict") != "error")
    head = df.filter(pl.col("judged_axis").is_in(HEADLINE_AXES))
    n = head.height
    if n:
        def pct(v):
            return head.filter(pl.col("verdict") == v).height / n * 100
        mp = head.filter(pl.col("verdict").is_in(["match", "partial"])).height / n * 100
        print(f"  HEADLINE content+attribution (n={n}): match+partial {mp:.0f}% "
              f"| match {pct('match'):.0f}% | partial {pct('partial'):.0f}% | miss {pct('miss'):.0f}%")
    for ax in HEADLINE_AXES:
        a = df.filter(pl.col("judged_axis") == ax)
        na = a.height
        if not na:
            continue
        def apct(v):
            return a.filter(pl.col("verdict") == v).height / na * 100
        print(f"    {ax} (n={na}): match {apct('match'):.0f}% | partial {apct('partial'):.0f}% "
              f"| miss {apct('miss'):.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct", help="VLM extractor")
    ap.add_argument("--mode", choices=["single", "multi"], default="single")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    out = Path(f"eval/data/extraction_eval_{args.model.split('/')[-1]}_{args.mode}.parquet")
    rows = subset().to_dicts()
    # Exclude the artifact (authenticity) axis from the extraction eval — it's an image-authentication
    # task (Phase 1b reverse-image), not extraction. Per-axis stratified split is unaffected.
    rows = [r for r in rows if r.get("judged_axis") in HEADLINE_AXES]
    split = assign_split(rows)
    for r in rows:
        r["split"] = split[r["review_url"]]
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    if args.max:
        rows = rows[: args.max]
    done = set(pl.read_parquet(out)["review_url"].to_list()) if out.exists() else set()
    todo = [i for i, r in enumerate(rows) if r["review_url"] not in done]
    print(f"subset {len(rows)} (split={args.split}) | done {len(done)} | todo {len(todo)} "
          f"| extractor={args.model} mode={args.mode}", flush=True)
    if not todo:
        if out.exists():
            _report(pl.read_parquet(out).filter(pl.col("review_url").is_in([r["review_url"] for r in rows])))
        return
    res: dict[str, dict] = {}

    def flush():
        if not res:
            return
        new = pl.DataFrame(list(res.values()))
        comb = new if not out.exists() else pl.concat([pl.read_parquet(out), new], how="diagonal_relaxed").unique("review_url", keep="last")
        comb.write_parquet(out)

    def work(i):
        r = rows[i]
        try:
            ex = _extract(args.model, r, args.mode)
            if args.mode == "single":
                pc = ex.get("primary_claim")
                claims = [pc] if pc else []
            else:
                claims = [c.get("normalized_claim", "") for c in (ex.get("claims") or []) if c.get("normalized_claim")]
            jd = _judge(r["claim_text"], claims)
            return i, {"verdict": jd.get("verdict"), "claims": claims,
                       "flags": ex.get("flags") or [],
                       "image_serialization": (ex.get("image_serialization") or "")[:1000],
                       "judge_reason": jd.get("reason")}
        except Exception as e:
            return i, {"verdict": "error", "claims": [], "flags": [], "image_serialization": "", "judge_reason": str(e)[:120]}

    def apply(i, p):
        r = rows[i]
        res[r["review_url"]] = {"review_url": r["review_url"], "gold": r["claim_text"],
                               "judged_axis": r.get("judged_axis"), "split": r["split"],
                               "mode": args.mode, **p}

    ab = pooled_checkpointed(todo, work, apply, flush, args.workers, f"extract:{args.model.split('/')[-1]}:{args.mode}", checkpoint_every=40)
    flush()

    runset = [r["review_url"] for r in rows]
    _report(pl.read_parquet(out).filter(pl.col("review_url").is_in(runset)))
    if ab:
        import sys; sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
