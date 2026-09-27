"""Test a Gemma-tuned gate prompt (v8g) vs the Qwen-tuned v8, on the clean bake-off set, by axis.
Fixes target Gemma's artifact-axis recall gap: (1) 'personal' = a PRIVATE individual's OWN private life
(public figures' lives + societally-relevant private-person claims are IN); (2) stronger image-as-claim
override (don't dismiss an evidentiary image as opinion/insult because of the caption).

  uv run python -m eval.scripts.gemma_prompt_test
"""
from __future__ import annotations

import glob

import polars as pl

from eval.scripts.gate_eval import GATE_SYSTEM
from eval.scripts.model_bakeoff import build_clean_set, content_for, call, MODELS
from eval.scripts._pool import pooled_checkpointed

# ---- build v8g from v8 via targeted edits (assert each lands) ----
EDITS = [
    ("even when the caption is only a reaction, question, or joke; judge the claim the image makes "
     "(a purely decorative, scenic, or personal photo is not a claim).",
     "even when the caption is only a reaction, question, joke, or insult. If an image is presented as "
     "real evidence (a photo, screenshot, or document of a person, event, or place), the IMAGE is the "
     "claim — evaluate what it asserts and do NOT dismiss the post as personal/opinion/insult because of "
     "its caption. Only a decorative, scenic, or private selfie with no factual claim is not a claim."),
    ("- entertainment / celebrity / reality-TV / music / concerts (UNLESS a real-world event: death, "
     "illness, crime, accident);",
     "- entertainment AS entertainment — a show, song, film, tour, award, celebrity dating/gossip (UNLESS "
     "a real-world event: death, illness, crime, accident — OR a factual claim about a public figure's "
     "conduct, statements, or an image of them, which is IN);"),
    ("- personal updates, jokes, greetings, trivia; opinions, aesthetic judgments, predictions; questions "
     "that assert no fact.",
     "- a PRIVATE individual's OWN private life with no public angle (their day, pet, meal, purchase, mood); "
     "jokes, greetings, trivia; opinions, aesthetic judgments, predictions; questions that assert no fact. "
     "Here 'personal' = a private person's own private life ONLY. A factual claim about a PUBLIC figure "
     "(including their private life, conduct, or an image of them) is IN; a claim about a private person "
     "that carries political/societal implications (e.g. an emotive story spread as political content) is IN."),
]
G = GATE_SYSTEM
for old, new in EDITS:
    assert old in G, f"edit did not match: {old[:50]}..."
    G = G.replace(old, new)
GATE_SYSTEM_V8G = G


def run(slug: str, sys_prompt: str, tag: str, data: list) -> None:
    import requests, time, base64
    from pathlib import Path
    from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL
    from eval.scripts.gate_eval import MAX_TOKENS, _obj
    URL = f"{EXTRACTION_BASE_URL}/chat/completions"; HDR = {"Authorization": f"Bearer {EXTRACTION_API_KEY}"}
    out = "eval/data/bakeoff_results.parquet"
    res: dict = {}

    def flush():
        if res:
            new = pl.DataFrame(list(res.values()))
            comb = new if not pl.read_parquet(out).is_empty() else new
            comb = pl.concat([pl.read_parquet(out), new], how="diagonal_relaxed").unique(["model", "uid"], keep="last")
            comb.write_parquet(out)

    def work(i):
        r = data[i]
        body = {"model": slug, "messages": [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": content_for(r)}], "temperature": 0, "max_tokens": MAX_TOKENS,
                "response_format": {"type": "json_object"}}
        try:
            t0 = time.time(); resp = requests.post(URL, headers=HDR, json=body, timeout=180); lat = time.time()-t0
            resp.raise_for_status(); j = resp.json(); g = _obj(j["choices"][0]["message"]["content"]); u = j.get("usage", {})
            kept = bool(g.get("check_worthy")) and g.get("claim_locus") != "video"
            return i, {"model": tag, "uid": r["uid"], "label": r["label"], "kept": kept, "parse_ok": bool(g),
                       "latency": lat, "ptok": u.get("prompt_tokens"), "ctok": u.get("completion_tokens")}
        except Exception as e:
            return i, {"model": tag, "uid": r["uid"], "label": r["label"], "kept": None, "parse_ok": False, "latency": None}
    pooled_checkpointed(list(range(len(data))), work, lambda i, p: res.__setitem__(p["uid"], p), flush, 4, tag, checkpoint_every=100)
    flush()


def report() -> None:
    ax = {}
    for f in sorted(glob.glob("eval/data/*_harvest.parquet")):
        d = pl.read_parquet(f)
        if "image_paths" in d.columns and "judged_axis" in d.columns:
            for r in d.to_dicts():
                ip = (r.get("image_paths") or [None])[0]
                if ip and ip not in ax:
                    ax[ip] = r.get("judged_axis")
    df = pl.read_parquet("eval/data/bakeoff_results.parquet")
    print(f"\n{'model':14}{'spec':>7}{'content':>9}{'attrib':>8}{'artifact':>10}{'text(c+a)':>11}")
    for key in ["qwen30b", "gemma4", "gemma4_v8g"]:
        m = df.filter((pl.col("model") == key) & pl.col("kept").is_not_null())
        if m.is_empty():
            continue
        neg = m.filter(pl.col("label") == "negative"); pos = m.filter(pl.col("label") == "positive")
        spec = (1 - neg.filter(pl.col("kept") == True).height/neg.height)*100  # noqa: E712
        cell = {}; ca_k = ca_n = 0
        for a in ["content", "attribution", "artifact"]:
            u = [x for x in pos["uid"].to_list() if ax.get(x) == a]
            s = pos.filter(pl.col("uid").is_in(u)); k = s.filter(pl.col("kept") == True).height; n = s.height  # noqa: E712
            cell[a] = f"{k/n*100:.0f}%" if n else "-"
            if a in ("content", "attribution"):
                ca_k += k; ca_n += n
        print(f"{key:14}{spec:6.1f}%{cell['content']:>9}{cell['attribution']:>8}{cell['artifact']:>10}{(str(round(ca_k/ca_n*100))+'%'):>11}")


def main() -> None:
    data = build_clean_set(250)
    print(f"v8g len {len(GATE_SYSTEM_V8G)} chars (v8 {len(GATE_SYSTEM)}); running Gemma-v8g on {len(data)} items", flush=True)
    run(MODELS["gemma4"][0], GATE_SYSTEM_V8G, "gemma4_v8g", data)
    report()


if __name__ == "__main__":
    main()
