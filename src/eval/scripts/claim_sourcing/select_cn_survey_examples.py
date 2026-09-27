"""Select ~20 candidate posts from a Community Notes note-eligible harvest as TEXT-ONLY,
UNVERIFIED examples of the kind of statement the misinformation survey will show (for
a documentation request, 2026-09-14).

Two stages.

Stage 1 - a $0 rule cascade over the harvest, logging the count after each rule:
  english; not a reply; not a repost; claim locus is the text (drop caption-only posts:
  under ~8 words after stripping urls, or a purely deictic opener); impressions at or
  above the HARVEST median; no relative-date leak in the post text (the select_midband
  rel-leak idea, moved from evidence dates to the post's own words); not forecast-form
  ("will" / "set to" / "is going to"); not satire/humour by obvious markers. The
  "primarily about a private individual" screen is NOT reliably mechanisable at $0 and is
  applied in stage 3 from the model's private-person flag, so the cascade records it as a
  pass-through and names where it is really enforced.

Stage 2 - one DeepSeek-V4-Flash pass per survivor (workers <= 8, cached per post),
returning has_checkable_claim, the single decisive checkable proposition, a topic label,
political_lean_of_post and author_lean_from_description (left/right/neutral/unclear),
form_ok, and sensitivity flags (graphic / hate / medical / private person).

Selection - keep has_checkable_claim and form_ok, drop any sensitivity flag, then rank by
impressions within each content lean: top 10 left, top 10 right, top 5 neutral/unclear.

    uv run python -m eval.scripts.claim_sourcing.select_cn_survey_examples \
        --parquet eval/data/community_notes/eligible_posts.parquet \
        --out <scratch>/cn_examples.json --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval import evidence_urn_run as eur  # noqa: E402
from eval.scripts.build_eval.reader_lab import PROVIDER, PROVIDERS, _endpoint, _parse_json  # noqa: E402

# --- stage 1 patterns -------------------------------------------------------
_URL = re.compile(r"https?://\S+")
_DEICTIC_OPEN = re.compile(r"^\s*(this|that|look at (this|that)?|watch|wow|omg|lol|"
                           r"see this|check this|listen|here'?s?)\b", re.I)
# post-text relative-date leak: a claim a survey respondent cannot resolve without knowing
# when the post was made. The select_midband rel-leak parser matched evidence-document
# dates ("N unit ago"); the same idea over the post's own words is date deixis.
_REL_TXT = re.compile(
    r"\b(today|yesterday|tomorrow|tonight|last night|this (morning|afternoon|evening|week|month|year)|"
    r"last (week|month|year)|next (week|month|year)|\d+\s+(hour|day|week|month|year)s?\s+ago|"
    r"just now|moments ago|hours ago|days ago|earlier today|this weekend)\b", re.I)
_FORECAST = re.compile(r"\b(will|set to|is going to|going to)\b", re.I)
_SATIRE = re.compile(r"\b(satire|satiric|parody|parodie|comedy|the onion|babylon bee|not real news|"
                     r"fake news account|shitpost)\b", re.I)


def _wordcount(t: str) -> int:
    return len(re.findall(r"\b\w+\b", _URL.sub("", str(t or ""))))


def cascade(df: pd.DataFrame):
    """Return (survivors_df, funnel list of (rule, remaining))."""
    funnel = [("harvest", len(df))]
    c = df
    c = c[c["lang"] == "en"];                                   funnel.append(("english", len(c)))
    c = c[~c["is_reply"].astype(bool)];                         funnel.append(("not a reply", len(c)))
    c = c[~c["is_repost"].astype(bool)];                        funnel.append(("not a repost", len(c)))
    wc = c["text"].map(_wordcount)
    locus = (wc >= 8) & (~c["text"].fillna("").str.match(_DEICTIC_OPEN))
    c = c[locus];                                               funnel.append(("claim locus in text (>=8 words, not a deictic caption)", len(c)))
    med = df["view_count"].median()
    c = c[c["view_count"] >= med];                              funnel.append((f"impressions >= harvest median ({int(med):,})", len(c)))
    c = c[~c["text"].fillna("").str.contains(_REL_TXT)];        funnel.append(("no relative-date leak in the text", len(c)))
    c = c[~c["text"].fillna("").str.contains(_FORECAST)];       funnel.append(("not forecast-form (will / set to / is going to)", len(c)))
    sat = c["author_description"].fillna("").str.contains(_SATIRE) | c["text"].fillna("").str.contains(_SATIRE)
    c = c[~sat];                                                funnel.append(("not satire/humour by obvious markers", len(c)))
    funnel.append(("primarily about a private individual (enforced in selection via the model private-person flag)", len(c)))
    return c, funnel


# --- stage 2 prompt (abstract principles only, no dataset-derived examples) -------------
SYS = """\
You are screening one social media post for a survey experiment on political
misinformation. The survey will show respondents short factual-looking political
statements and ask them to judge them. Decide whether this post is the kind of statement
the survey uses, extract its single decisive checkable proposition, and describe its
politics. Judge the post's own words; do not use outside knowledge of whether the claim is
true, and do not let the outlet's identity decide the politics.

has_checkable_claim: true when the post asserts at least one concrete state of the world
that could in principle be checked against evidence - an event that happened, an action
taken by a named actor or body, a ruling, a policy, a quantity or a specific attribution.
false for pure opinion, sentiment, exhortation, a bare question, a joke, or a caption that
only points at attached media.

proposition: the one decisive checkable statement the post rests on, rewritten as a single
plain declarative sentence a survey respondent could rate, self-contained and naming its
actor. If nothing is checkable, an empty string.

topic: a short lowercase label of one or two words for the subject.

political_lean_of_post: the lean of the CONTENT if it were taken as true. "left" when a
reader would take it as favourable to the political left or as reflecting badly on the
right; "right" for the reverse; "neutral" when it favours neither or both alike;
"unclear" when the politics cannot be determined from the text. Give a one-clause reason.

author_lean_from_description: the apparent lean of the account from its profile
description alone, on the same four-value scale; "unclear" when the description gives no
political signal.

form_ok: true only when the decisive proposition is a plain declarative factual statement
a respondent can rate - not a question, not a quote-tweet dunk or a reply riffing on
another post, and resting on a single claim rather than several stacked together.

sensitivity: list any of "graphic", "hate", "medical", "private person" that apply.
"private person" when the post is primarily about an identifiable individual who is not a
public figure or office holder acting in a public role.

Reply with JSON only:
{"has_checkable_claim": true|false, "proposition": "<one sentence or empty>",
 "topic": "<short label>",
 "political_lean_of_post": "left"|"right"|"neutral"|"unclear", "lean_reason": "<one clause>",
 "author_lean_from_description": "left"|"right"|"neutral"|"unclear",
 "form_ok": true|false, "sensitivity": ["..."]}
"""
PROMPT_V = "cn-survey-screen-v1"
LEANS = {"left", "right", "neutral", "unclear"}
_SENS = {"graphic", "hate", "medical", "private person"}


def call(model, row, cache):
    key = f"{prompt_hash(SYS)}|{model}|{row['post_id']}"
    ck = cache / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")
    if ck.exists():
        return json.load(open(ck)), 0.0
    user = (f"OUTLET: @{row.get('handle') or ''}\n"
            f"AUTHOR DESCRIPTION: {row.get('author_description') or ''}\n"
            f"POST:\n{row.get('text') or ''}")
    body = {"model": model, "temperature": 0, "max_tokens": 700,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}]}
    base, k = _endpoint()
    for attempt in range(5):
        r = eur._SESSION.post(f"{base}/chat/completions",
                              headers={"Authorization": f"Bearer {k}"}, json=body, timeout=120)
        if r.status_code != 429:
            break
        time.sleep(5 * 2 ** attempt)
    r.raise_for_status()
    j = r.json()
    obj = _parse_json(j["choices"][0]["message"].get("content")) or {}
    usage = j.get("usage") or {}
    cost = usage.get("estimated_cost")
    if cost is None:
        cost = (usage.get("prompt_tokens") or 0) * 0.09e-6 + (usage.get("completion_tokens") or 0) * 0.18e-6
    pl = obj.get("political_lean_of_post")
    al = obj.get("author_lean_from_description")
    sens = [s for s in (obj.get("sensitivity") or []) if s in _SENS]
    res = {"has_checkable_claim": bool(obj.get("has_checkable_claim")),
           "proposition": (obj.get("proposition") or "")[:400],
           "topic": (obj.get("topic") or "")[:40],
           "political_lean_of_post": pl if pl in LEANS else "unclear",
           "lean_reason": (obj.get("lean_reason") or "")[:200],
           "author_lean_from_description": al if al in LEANS else "unclear",
           "form_ok": bool(obj.get("form_ok")),
           "sensitivity": sens}
    json.dump(res, open(ck, "w"))
    return res, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--provider", default="deepinfra", choices=sorted(PROVIDERS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--top-lr", type=int, default=10)
    ap.add_argument("--top-neutral", type=int, default=5)
    a = ap.parse_args()
    assert a.workers <= 8, "keep concurrency <= 8 (a reader run holds the 64 gate)"
    PROVIDER[0] = a.provider

    df = pd.read_parquet(a.parquet)
    surv, funnel = cascade(df)
    print(f"prompt {PROMPT_V} ({prompt_hash(SYS)})  model {a.model} @ {a.provider}", flush=True)
    for name, n in funnel:
        print(f"  {name:70s} {n}", flush=True)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(str(out) + ".cache"); cache.mkdir(exist_ok=True)
    rows = surv.to_dict("records")
    lock = threading.Lock(); done = [0]; spent = [0.0]; t0 = time.time()
    verd = {}

    def work(row):
        r, c, err = None, 0.0, ""
        for attempt in range(3):
            try:
                r, c = call(a.model, row, cache); break
            except Exception as e:  # noqa: BLE001
                err = repr(e); time.sleep(3 * (attempt + 1))
        with lock:
            verd[row["post_id"]] = r or {"has_checkable_claim": False, "form_ok": False,
                                         "sensitivity": [], "political_lean_of_post": "unclear",
                                         "proposition": "", "topic": "", "lean_reason": f"failed {err[:80]}",
                                         "author_lean_from_description": "unclear"}
            done[0] += 1; spent[0] += c
            if done[0] % 25 == 0 or done[0] == len(rows):
                el = time.time() - t0
                print(f"  {done[0]}/{len(rows)}  ${spent[0]:.4f}  {el:.0f}s", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, r) for r in rows]):
            f.result()

    # selection: keep checkable + form_ok + no sensitivity flag
    kept = []
    for row in rows:
        v = verd[row["post_id"]]
        if v["has_checkable_claim"] and v["form_ok"] and not v["sensitivity"]:
            kept.append({"post_id": row["post_id"], "handle": row.get("handle"),
                         "text": row.get("text"), "view_count": int(row.get("view_count") or 0),
                         "author_followers": int(row.get("author_followers") or 0),
                         "created_at": str(row.get("created_at"))[:10], "url": row.get("url"),
                         **v})

    def top(lean, k):
        pool = [x for x in kept if x["political_lean_of_post"] == lean]
        return sorted(pool, key=lambda x: x["view_count"], reverse=True)[:k]

    shortlist = {"left": top("left", a.top_lr), "right": top("right", a.top_lr),
                 "neutral": top("neutral", a.top_neutral), "unclear": top("unclear", a.top_neutral)}

    result = {"prompt": PROMPT_V, "prompt_hash": prompt_hash(SYS), "model": a.model,
              "system_prompt": SYS, "harvest_median_views": int(df["view_count"].median()),
              "funnel": funnel, "spend": round(spent[0], 4),
              "n_survivors": len(rows), "n_kept": len(kept), "shortlist": shortlist,
              "verdicts": verd}
    json.dump(result, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\nspent ${spent[0]:.4f}  survivors {len(rows)}  kept {len(kept)}  "
          f"left {len(shortlist['left'])} right {len(shortlist['right'])} "
          f"neutral {len(shortlist['neutral'])} unclear {len(shortlist['unclear'])}  wrote {out}")


if __name__ == "__main__":
    main()
