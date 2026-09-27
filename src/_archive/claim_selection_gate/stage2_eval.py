"""Stage-2 eval — axis-aware claim extraction + serialization fidelity (w/ Daniel 2026-06-27).

ONE combined call per post (`--mode combined`): extract the RELEVANT check-worthy claims AND flag
`needs_attribution` / `needs_authenticity` (the three axes a fact-checker might check — our `judged_axis`
gold is non-exhaustive, so the flags are scored as a RECALL FLOOR vs the checker's chosen axis, precision
by adjudication). `--mode pure` runs claims-only (the A/B control: do the flags degrade extraction?).

Three assessments:
  1. CLAIM COVERAGE — combination-aware DeepSeek judge (match/partial/miss vs the fact-checker gold),
     sliced by image-locus (claims that live IN the image vs text).
  2. SERIALIZATION FIDELITY — an INDEPENDENT SEEING judge (Gemma-3-27B VLM) compares `image_serialization`
     against the actual image: faithful / partial / poor, with missing + hallucinated content. The
     downstream evidence-retrieval model is TEXT-ONLY (blind) → the serialization IS the image to it.
  3. FLAGS — needs_attribution / needs_authenticity recall vs judged_axis (floor).

Qwen JSON-mode hardening: no `max_tokens` (truncates mid-string), `finish_reason`/`completion_tokens`
recorded, partial-JSON salvaged. Output → `stage2_<model>_<mode>.parquet`.

  uv run python -m eval.scripts.stage2_eval --model Qwen/Qwen3-VL-30B-A3B-Instruct --mode combined --max 6
  uv run python -m eval.scripts.stage2_eval --model Qwen/Qwen3-VL-235B-A22B-Instruct --mode pure --split dev --max 100
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path

import polars as pl
import requests

from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL, VERIFICATION_MODEL
from eval.scripts._pool import pooled_checkpointed

URL = f"{EXTRACTION_BASE_URL}/chat/completions"
HDR = {"Authorization": f"Bearer {EXTRACTION_API_KEY}"}
MIN_TOKENS = 150
JUDGE_MODEL = VERIFICATION_MODEL
# Independent SEEING judges (neither is the Qwen extractor). Primary = Llama-4-Maverick (strongest
# independent VLM on DeepInfra); Gemma-3-27B kept as a cross-check → report their agreement.
FIDELITY_JUDGE = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
FIDELITY_JUDGE_X = "google/gemma-3-27b-it"

_CLAIM_TASK = """Extract every RELEVANT check-worthy proposition the post puts forward — the claims that could be misinformation, that a fact-checker would want to verify, that are load-bearing to the post's argument, or that are in the public interest to check. Skip opinions, jokes, questions, rhetoric.
- As many as the post genuinely makes (a long post may carry ten or more; a short one, one or none), most-salient/disputed FIRST. A claim may live in the post text, in-image text, the visuals, or an endorsed quoted tweet.
- Each is a single, atomic, DECONTEXTUALIZED proposition — bare claim, references resolved, NOT "the image shows…". ONE complete proposition each; never fragment one claim into pieces."""

_GROUNDING = """<grounding_rules>
- Transcribe ALL salient in-image text in full; describe key visuals with as much detail as the checkable content warrants (full transcription for text-heavy screenshots, a line for a simple photo). Describe NON-judgmentally.
- Use only what is visible or stated; do not infer absent facts.
- Identity: name a depicted person/place only when confident (in-image text/caption/quoted tweet/clear recognition); else describe generically + add a "flags" note. NEVER guess a name.
- If in-image text isn't legible or a detail isn't visible, say so — don't invent it.
</grounding_rules>"""

COMBINED_SYS = f"""<role>
You read a social-media post and prepare it for fact-checking. You (1) extract the claims worth checking, and decide whether the post also needs (2) its attribution verified or (3) its image/video's authenticity verified. These are independent — a post may need any, all, or none.
</role>

<inputs>
POST TEXT; zero or more images (labeled Image 1, Image 2, …); an optional QUOTED TWEET. Any image may be absent.
</inputs>

<task_1_claims>
{_CLAIM_TASK}
</task_1_claims>

<task_2_attribution>
Decide whether the post's checkable content includes an ATTRIBUTION a fact-checker would verify separately — that a specific named person/outlet SAID, wrote, or did a specific thing. ("Did X really say it?" is checked separately from whether the statement is true.) Set needs_attribution = true when such an attribution is present; else false.
</task_2_attribution>

<task_3_authenticity>
Decide whether the post relies on its IMAGE/VIDEO being GENUINE — the media's own authenticity or context is itself a checkable point (real vs fabricated/edited/AI-generated/staged; truly shows the person/event/place/time claimed; or an old/unrelated image recaptioned). Set needs_authenticity = true when a fact-checker would verify the IMAGE ITSELF, not only the words. Set false when the image merely illustrates/decorates a claim whose truth doesn't depend on it being real. Mere presence of an image is NOT enough. When true, fill purported (who/what/when/where).
</task_3_authenticity>

{_GROUNDING}

<examples>
1 — content, no flags (anti-fragmentation). POST: aluminum-vaccine study.
{{"image_serialization":"","claims":["A 2025 Danish study of more than 1.2 million people found no link between aluminum in childhood vaccines and long-term health problems such as autism, asthma, or autoimmune disease"],"attribution_reason":"a study finding, not a person's statement","needs_attribution":false,"authenticity_reason":"no image","needs_authenticity":false,"purported":null,"flags":[]}}

2 — ATTRIBUTION true. POST: "'We made a huge mistake when we passed the Civil Rights Act' - Charlie Kirk"
{{"image_serialization":"","claims":["Charlie Kirk said, 'We made a huge mistake when we passed the Civil Rights Act in the 1960s'"],"attribution_reason":"the checkable point is whether Kirk made this statement","needs_attribution":true,"authenticity_reason":"no image","needs_authenticity":false,"purported":null,"flags":[]}}

3 — AUTHENTICITY true + identity abstention. POST: leaked Netanyahu/Epstein flight photo. Image 1.
{{"image_serialization":"A photo of two older men seated facing each other in a private-jet cabin; a 'HOUSE_OVERSIGHT_065659' watermark in the corner","claims":["A photograph shows Benjamin Netanyahu and Jeffrey Epstein together on a private flight"],"attribution_reason":"no statement is attributed","needs_attribution":false,"authenticity_reason":"the photo is offered as proof two named people were together — its genuineness is the point","needs_authenticity":true,"purported":"Benjamin Netanyahu and Jeffrey Epstein together on a private flight","flags":["unresolved identity: the two men are not confidently identifiable from the image alone"]}}

4 — BOTH attribution AND authenticity true. POST: Don Lemon attorney statement card. Image 1.
{{"image_serialization":"A statement card branded 'THE DON LEMON SHOW', titled 'STATEMENT FROM ABBE LOWELL, ATTORNEY FOR DON LEMON', transcribed in full: 'Don Lemon was taken into custody by federal agents last night in Los Angeles… Don will fight these charges vigorously and thoroughly in court.' A stylized image of Don Lemon holding a 'DL' microphone below.","claims":["Don Lemon's attorney Abbe Lowell released a statement saying Don Lemon was taken into custody by federal agents in Los Angeles"],"attribution_reason":"the statement is attributed to attorney Abbe Lowell — whether he issued it is checkable","needs_attribution":true,"authenticity_reason":"the image-statement is presented as a genuine release — whether the card is real is the point","needs_authenticity":true,"purported":"an authentic statement from attorney Abbe Lowell confirming Don Lemon's arrest","flags":[]}}

5 — authenticity FALSE near-miss (image only illustrates). POST: Coco Gauff donation collage. Image 1.
{{"image_serialization":"A three-part collage: a flooded river strewn with toppled trees; a man and woman carrying two children across flood debris; a smiling young woman in a neon-yellow tennis kit.","claims":["Tennis player Coco Gauff donated $3 million toward search-and-rescue for the July 2025 Texas floods"],"attribution_reason":"no statement is attributed","needs_attribution":false,"authenticity_reason":"the images illustrate the donation story; whether they are real is not the point","needs_authenticity":false,"purported":null,"flags":[]}}

6 — MULTI-CLAIM, no flags. POST: WhatsApp AI access.
{{"image_serialization":"","claims":["Meta's new WhatsApp AI feature can access all of a user's chats","Enabling WhatsApp's 'advanced privacy' option stops the AI from accessing a user's chats"],"attribution_reason":"no statement is attributed","needs_attribution":false,"authenticity_reason":"no image","needs_authenticity":false,"purported":null,"flags":[]}}
</examples>

<output_format>
Respond with JSON only, keys in THIS order:
{{"image_serialization":"<… ; '' if no image>","claims":["<atomic decontextualized proposition>"],"attribution_reason":"<short>","needs_attribution":<true or false>,"authenticity_reason":"<short>","needs_authenticity":<true or false>,"purported":"<who/what/when/where, or null>","flags":["<≤3 short notes>"]}}
</output_format>

<reminders>
- Extract the RELEVANT checkable claims — distinct and complete, never fragments; skip opinion/rhetoric.
- needs_authenticity = false if the image only illustrates the claim.
- NEVER guess a name; describe + flag instead.
</reminders>"""

PURE_SYS = f"""<role>
You extract the check-worthy claims a fact-checker would verify from a social-media post.
</role>

<inputs>
POST TEXT; zero or more images (labeled Image 1, Image 2, …); an optional QUOTED TWEET. Any image may be absent.
</inputs>

<task>
{_CLAIM_TASK}
</task>

{_GROUNDING}

<examples>
POST: aluminum-vaccine study.
{{"image_serialization":"","claims":["A 2025 Danish study of more than 1.2 million people found no link between aluminum in childhood vaccines and long-term health problems such as autism, asthma, or autoimmune disease"],"flags":[]}}
POST: "As of today, AI is available on WhatsApp and therefore has access to all chats. You can enable the 'advanced privacy' option."
{{"image_serialization":"","claims":["Meta's new WhatsApp AI feature can access all of a user's chats","Enabling WhatsApp's 'advanced privacy' option stops the AI from accessing a user's chats"],"flags":[]}}
POST: "They're really working it with these crisis actors." QUOTED TWEET: hantavirus-cruise news.
{{"image_serialization":"","claims":["A passenger featured in coverage of the cruise-ship hantavirus outbreak is a paid crisis actor"],"flags":[]}}
POST: "Tennis Star, Coco Gauff, has donated $3 million dollars for the search and rescue efforts in Texas." Image 1.
{{"image_serialization":"A three-part collage: a flooded river strewn with toppled trees; a man and woman carrying two children across flood debris; a smiling young woman in a neon-yellow tennis kit.","claims":["Tennis player Coco Gauff donated $3 million toward search-and-rescue for the July 2025 Texas floods"],"flags":[]}}
</examples>

<output_format>
Respond with JSON only, keys in THIS order:
{{"image_serialization":"<… ; '' if no image>","claims":["<atomic decontextualized proposition>"],"flags":["<≤3 short notes>"]}}
</output_format>

<reminders>
- Extract the RELEVANT checkable claims — distinct and complete, never fragments; skip opinion/rhetoric.
- NEVER guess a name; describe + flag instead.
</reminders>"""

JUDGE_SYS = (
    "You judge whether the claim(s) EXTRACTED from a post recover the GOLD claim a fact-checker wrote, "
    "on the core checkable PROPOSITION. Be FRAMING-AGNOSTIC: ignore fact-checker wrappers on the gold "
    "('Image shows…', 'A post claims…', 'X tweeted…', 'authentically/falsely shows'). Be "
    "COMBINATION-AWARE: a match may be carried by a SINGLE extracted claim OR by SEVERAL together. "
    'JSON only: {"verdict":"match|partial|miss","reason":"<=25 words"}. '
    "match = the gold proposition is captured; partial = related but missing/adding a key element; "
    "miss = no extracted claim(s) capture the gold."
)

FIDELITY_SYS = (
    "You verify an image SERIALIZATION written for a BLIND downstream fact-checker (a text-only model "
    "that cannot see the image). Given the actual image(s) and the serialization text, judge whether the "
    "serialization faithfully and completely captures the image's CHECK-WORTHY content: all salient "
    "in-image text transcribed accurately, key visuals described, and NOTHING hallucinated (no text, "
    "details, or identities not actually present). "
    'JSON only: {"verdict":"faithful|partial|poor","missing":"<key checkable content omitted, or none>",'
    '"hallucinated":"<content asserted but not in the image, or none>"}. '
    "faithful = a blind reader gets the checkable content right; partial = some checkable content missing "
    "but nothing invented; poor = hallucinated content OR a major omission that would mislead."
)


def _salvage(txt: str) -> dict:
    def _s(pat):
        m = re.search(pat, txt, re.S)
        try:
            return json.loads('"' + m.group(1) + '"') if m else None
        except json.JSONDecodeError:
            return None
    out: dict = {}
    for key in ("image_serialization", "attribution_reason", "authenticity_reason", "purported"):
        v = _s(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"')
        if v is not None:
            out[key] = v
    mc = re.search(r'"claims"\s*:\s*\[(.*?)(\]|$)', txt, re.S)
    if mc:
        out["claims"] = [json.loads('"' + m.group(1) + '"') for m in
                         re.finditer(r'"((?:[^"\\]|\\.)*)"', mc.group(1))]
    for key in ("needs_attribution", "needs_authenticity"):
        m = re.search(rf'"{key}"\s*:\s*(true|false)', txt)
        if m:
            out[key] = m.group(1) == "true"
    return out


def _obj(txt: str) -> dict:
    txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S)
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return _salvage(txt)
    except Exception:
        return {}


def _chat(model: str, messages: list, timeout: int = 180) -> tuple[dict, str, int]:
    r = requests.post(URL, headers=HDR, json={"model": model, "messages": messages,
                      "temperature": 0, "response_format": {"type": "json_object"}}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    ch = j["choices"][0]
    return _obj(ch["message"]["content"]), ch.get("finish_reason"), (j.get("usage") or {}).get("completion_tokens")


def _imgs(row: dict) -> list[str]:
    return [p for p, t in zip(row["image_paths"] or [], row["image_est_tokens"] or [])
            if t and t >= MIN_TOKENS][:4]


def _content(row: dict) -> list:
    # Feed the REAL post text (raw_claim — the messy original post), not the claim-gated raw_context.
    # raw_claim is validated distinct from the normalized gold (Jaccard ~0.13, 0% exact); raw_context
    # falls to body_quote when raw_claim is gated out, so it's the fallback. (clog 270626)
    post = row.get("raw_claim") or row.get("raw_context") or "(none)"
    parts = [{"type": "text", "text":
              f"POST TEXT: {post[:600]}"
              + (f"\nQUOTED TWEET: {row['quoted_text'][:400]}" if row.get("quoted_text") else "")}]
    for i, p in enumerate(_imgs(row)):
        b = base64.b64encode(Path(p).read_bytes()).decode()
        parts.append({"type": "text", "text": f"Image {i+1}:"})
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b}"}})
    return parts


def _extract(model: str, row: dict, mode: str) -> dict:
    sys = COMBINED_SYS if mode == "combined" else PURE_SYS
    obj, fr, ct = _chat(model, [{"role": "system", "content": sys}, {"role": "user", "content": _content(row)}])
    claims = [c for c in (obj.get("claims") or []) if isinstance(c, str) and c]
    out = {"claims": claims, "flags": obj.get("flags") or [],
           "image_serialization": (obj.get("image_serialization") or "")[:2000],
           "ext_finish": fr, "ext_tokens": ct}
    if mode == "combined":
        out.update({"needs_attribution": obj.get("needs_attribution") if isinstance(obj.get("needs_attribution"), bool) else None,
                    "needs_authenticity": obj.get("needs_authenticity") if isinstance(obj.get("needs_authenticity"), bool) else None,
                    "purported": obj.get("purported"), "attr_reason": obj.get("attribution_reason"),
                    "auth_reason": obj.get("authenticity_reason")})
    return out


def _judge(gold: str, claims: list[str]) -> str:
    if not claims:
        return "miss"
    listing = "\n".join(f"[{i}] {c}" for i, c in enumerate(claims))
    obj, _, _ = _chat(JUDGE_MODEL, [{"role": "system", "content": JUDGE_SYS},
                      {"role": "user", "content": f"GOLD CLAIM:\n{gold}\n\nEXTRACTED CLAIMS:\n{listing}"}])
    return obj.get("verdict") or "miss"


def _fidelity(row: dict, serialization: str) -> dict:
    imgs = _imgs(row)
    if not imgs or not serialization:
        return {"fid_verdict": None, "fid_missing": None, "fid_hallucinated": None, "fid_verdict_x": None}
    content = [{"type": "text", "text": f"SERIALIZATION (written for a blind reader):\n{serialization[:2000]}\n\nThe actual image(s):"}]
    for p in imgs:
        b = base64.b64encode(Path(p).read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b}"}})
    msgs = [{"role": "system", "content": FIDELITY_SYS}, {"role": "user", "content": content}]
    obj, _, _ = _chat(FIDELITY_JUDGE, msgs)
    try:
        xobj, _, _ = _chat(FIDELITY_JUDGE_X, msgs)
        xv = xobj.get("verdict")
    except Exception:
        xv = None
    return {"fid_verdict": obj.get("verdict"), "fid_missing": obj.get("missing"),
            "fid_hallucinated": obj.get("hallucinated"), "fid_verdict_x": xv}


def rows_for(split: str) -> list[dict]:
    # The Stage-2 eval set (`build_stage2_dataset.py`): core 12mo pool filtered to rows that HAVE a post
    # to extract from (drops ~70% empty-input + video) with a frozen dev/test split. Derivation +
    # rationale: docs/claim_extraction_results.md.
    d = pl.read_parquet("eval/data/stage2_dataset.parquet")
    if split != "all":
        d = d.filter(pl.col("split") == split)
    # image-locus from the Stage-1 audit (claims that live IN the image)
    aud = pl.read_parquet("eval/data/image_audit.parquet").select("review_url", "claim_location")
    d = d.join(aud, on="review_url", how="left").with_columns(pl.col("claim_location").fill_null("text"))
    return d.to_dicts()


def _report(df: pl.DataFrame) -> None:
    ok = df.filter(pl.col("ext_verdict").is_not_null())
    def rate(sub):
        n = sub.height
        return f"match {sub.filter(pl.col('ext_verdict')=='match').height/n*100:.0f} | partial {sub.filter(pl.col('ext_verdict')=='partial').height/n*100:.0f} | miss {sub.filter(pl.col('ext_verdict')=='miss').height/n*100:.0f}" if n else "n=0"
    print(f"  EXTRACTION (n={ok.height}): {rate(ok)}")
    img_dep = ok.filter(pl.col("claim_location").is_in(["image", "both"]))
    txt = ok.filter(pl.col("claim_location") == "text")
    print(f"    image-locus claims (n={img_dep.height}): {rate(img_dep)}")
    print(f"    text-locus  claims (n={txt.height}): {rate(txt)}")
    if "fid_verdict" in df.columns:
        f = df.filter(pl.col("fid_verdict").is_not_null())
        if f.height:
            def fp(v):
                return f.filter(pl.col("fid_verdict") == v).height / f.height * 100
            hall = f.filter((pl.col("fid_hallucinated").is_not_null()) & (pl.col("fid_hallucinated").str.to_lowercase() != "none") & (pl.col("fid_hallucinated") != "")).height
            print(f"  SERIALIZATION FIDELITY (Maverick, n={f.height}): faithful {fp('faithful'):.0f} | partial {fp('partial'):.0f} | poor {fp('poor'):.0f}  | hallucinated-flagged {hall}")
            if "fid_verdict_x" in df.columns:
                xa = f.filter(pl.col("fid_verdict_x").is_not_null())
                if xa.height:
                    agree = xa.filter(pl.col("fid_verdict") == pl.col("fid_verdict_x")).height / xa.height * 100
                    print(f"    cross-judge agreement (Maverick vs Gemma, n={xa.height}): {agree:.0f}%")
    if "needs_authenticity" in df.columns:
        im = df.filter(pl.col("needs_authenticity").is_not_null())
        if im.height:
            art = im.filter(pl.col("judged_axis") == "artifact")
            rec = art.filter(pl.col("needs_authenticity")).height / art.height * 100 if art.height else 0
            firerate = im.filter(pl.col("needs_authenticity")).height / im.height * 100
            print(f"  needs_authenticity: recall-vs-artifact {rec:.0f}% (n={art.height}) | fires on {firerate:.0f}% of all rows")
        at = df.filter(pl.col("needs_attribution").is_not_null())
        if at.height:
            ga = at.filter(pl.col("judged_axis") == "attribution")
            rec = ga.filter(pl.col("needs_attribution")).height / ga.height * 100 if ga.height else 0
            print(f"  needs_attribution: recall-vs-attribution {rec:.0f}% (n={ga.height}) | fires on {at.filter(pl.col('needs_attribution')).height/at.height*100:.0f}% of all rows")
    ln = df.filter(pl.col("ext_finish") == "length").height
    print(f"  length-truncated: {ln}/{df.height}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    ap.add_argument("--mode", choices=["combined", "pure"], default="combined")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    out = Path(f"eval/data/stage2_{args.model.split('/')[-1]}_{args.mode}.parquet")
    rows = rows_for(args.split)
    if args.max:
        rows = rows[: args.max]
    done = set(pl.read_parquet(out)["review_url"].to_list()) if out.exists() else set()
    todo = [i for i, r in enumerate(rows) if r["review_url"] not in done]
    print(f"rows {len(rows)} (split={args.split}, mode={args.mode}) | done {len(done)} | todo {len(todo)} | {args.model}", flush=True)
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
            verdict = _judge(r["claim_text"], ex["claims"])
            fid = _fidelity(r, ex["image_serialization"]) if r.get("has_image") is True else {
                "fid_verdict": None, "fid_missing": None, "fid_hallucinated": None}
            return i, {"ext_verdict": verdict, **ex, **fid}
        except Exception as e:
            return i, {"ext_verdict": None, "claims": [], "flags": [], "image_serialization": "",
                       "ext_finish": "error", "ext_tokens": None, "fid_verdict": None,
                       "fid_missing": str(e)[:120], "fid_hallucinated": None, "fid_verdict_x": None}

    def apply(i, p):
        r = rows[i]
        res[r["review_url"]] = {"review_url": r["review_url"], "gold": r["claim_text"],
                               "judged_axis": r.get("judged_axis"), "has_image": r.get("has_image"),
                               "claim_location": r.get("claim_location"), "mode": args.mode, **p}

    ab = pooled_checkpointed(todo, work, apply, flush, args.workers, f"stage2:{args.model.split('/')[-1]}:{args.mode}", checkpoint_every=25)
    flush()
    _report(pl.read_parquet(out).filter(pl.col("review_url").is_in([r["review_url"] for r in rows])))
    if ab:
        import sys; sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
