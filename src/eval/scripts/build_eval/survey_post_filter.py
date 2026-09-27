"""Post-level pre-extraction filter (Daniel 2026-09-21): a gate that runs BEFORE
key_claim_extract, on the POST itself rather than on already-extracted claims. The 120-post v5
audit found ~6% of posts should never have reached extraction (satire, pure opinion / policy
goal, personal / low-stakes, media-only deception). This gate reads one post and decides
keep/drop with a reason code, plus a broad political yes/no topic flag, so those posts are held
back before any extraction call is spent on them.

This is a sibling of key_claim_filter.py (which is a CLAIM-level filter, kept untouched). Same
DeepInfra call path (eur._SESSION + reader_lab._endpoint), same per-request 429 retry, same
thread pool as the production extract/filter scripts. Input contract is the extract script's
posts JSON.

    uv run python -m eval.scripts.build_eval.survey_post_filter \
        --posts <posts.json> --out <post_filter.json> [--model deepseek-ai/DeepSeek-V4-Flash]

The Community Note text is passed as context (it is the cleanest signal for media-only
deception: a note that says the attached video/image is fake with no text claim). It is context
only, never a source of truth about keep/drop beyond pointing at where the post's content lives.

Output: {"prompt", "prompt_hash", "model", "system_prompt",
         "verdicts": {post_id: {"keep": bool, "reason": str, "political": bool, "why": str}}}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval import evidence_urn_run as eur  # noqa: E402
from eval.scripts.build_eval.reader_lab import PROVIDER, PROVIDERS, _endpoint, _parse_json  # noqa: E402

REASONS = ["keep", "no_checkable_claim", "opinion", "satire", "media_only", "personal", "trivial", "not_political"]

POST_FILTER_SYS = """\
You are screening one social media post before a fact-checking pipeline extracts a claim from it.
Decide whether the post is worth extracting a claim from at all, give the one reason that decides
it, and flag whether the post is about a political matter.

Keep the post when it makes, or points to, at least one concrete, self-contained, consequential
factual assertion that a stranger could in principle rate true or false: something is or was the
case - an event, an action, a quantity, a statement someone made. The assertion may be carried in
the text plainly, or presupposed by it. A post phrased as a question, a reaction, an exclamation,
an insult, a one-sided generalisation, or a link with a comment still counts as keepable when it
asserts or takes for granted a specific event, act or quantity - a rhetorical question that
presupposes a concrete happening is asserting that happening. Do not drop such a post merely
because of its grammar; keep it and let extraction find the claim. Default to keep whenever a
concrete checkable specific about a public matter is present.

Drop the post only when one of these decides it, and give exactly that one reason code:
no_checkable_claim - the text asserts and presupposes nothing a search could settle: it is a bare
reaction, greeting, question or link with no concrete event, act or quantity anywhere in it, or it
only says that something happened, emerged or was revealed without saying what. Do not use this
code when a specific event or act is named or presupposed, even in a question or a reaction.
opinion - the post is opinion, framing, characterisation or a stance whose decisive words are
evaluations no document can settle, with no concrete factual assertion underneath.
satire - the post is satire, parody, sarcasm, a joke or a comedic bit, performed to entertain and
meaning none of it literally. Vivid, extreme, mocking or exaggerated phrasing is not by itself
satire; use this code only when the post is genuinely a performance not meant as fact.
media_only - the checkable content lives ONLY in an attached image or video and the text asserts
nothing checkable on its own. Use this code only when the words alone carry no concrete assertion;
if the text itself names a checkable event, act, quantity or statement, keep the post even when it
also has media, and do not label it media_only.
personal - a personal anecdote or an individual's private experience, about the author or a private
person, that no public consequence turns on.
trivial - low-stakes and of no public consequence: a mundane everyday observation, a minor personal
or consumer matter, small talk, or content nobody would act on and that carries no public import,
even if it is checkable and well formed.
not_political - well formed and checkable, but entirely outside public affairs: sport, celebrity,
entertainment, consumer life, or other private-interest content of no political consequence.

Judge worth and form, never truth. Keep posts whose assertion you believe is false; drop posts
whose assertion you believe is true when a reason above applies. Losing a genuinely marginal post
is cheap, but a checkable, consequential assertion about a public matter must be kept. When more
than one reason could apply, give the one that most decides the post's unsuitability. A Community
Note may be supplied as context; use it only to see where the post's content lives, never as a
reason to keep or drop on whether the post is true or false.

Separately, flag whether the post concerns a political matter, understood broadly: government and
officials, elections and voting, public policy and legislation, immigration and borders, the
economy and cost of living, foreign affairs and conflict, society and culture-war disputes
(identity, religion, race, gender), crime and justice, and health policy. Set political true when
the post's subject falls in any of these, false otherwise. This flag is independent of keep/drop:
a post can be political and dropped, or non-political and kept.

Reply with JSON only:
{"keep": true | false, "reason": "<one reason code>", "political": true | false, "why": "<one clause>"}
"""
PROMPT = "post-filter-v2"


def call(model, post, cache, sys_prompt=POST_FILTER_SYS):
    key = f"{prompt_hash(sys_prompt)}|{model}|{post['post_id']}"
    ck = cache / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")
    if ck.exists():
        return json.load(open(ck)), 0.0
    user = f"OUTLET: @{post.get('handle') or ''}\nPOST DATE: {post.get('created_at') or 'unknown'}\nPOST:\n{post.get('post_text') or ''}"
    if post.get("quoted_text"):
        user += f"\n\nQUOTED POST by @{post.get('quoted_handle') or 'unknown'}:\n{post['quoted_text']}"
    if (post.get("note") or "").strip():
        user += f"\n\nCOMMUNITY NOTE (context, points at where the post's content lives):\n{post['note']}"
    body = {"model": model, "temperature": 0, "max_tokens": 400, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]}
    base, k = _endpoint()
    for attempt in range(5):
        r = eur._SESSION.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {k}"}, json=body, timeout=120)
        if r.status_code != 429:
            break
        time.sleep(5 * 2 ** attempt)
    r.raise_for_status()
    j = r.json()
    obj = _parse_json(j["choices"][0]["message"].get("content"))
    usage = j.get("usage") or {}
    cost = usage.get("estimated_cost")
    if cost is None:
        cost = (usage.get("prompt_tokens") or 0) * 0.15e-6 + (usage.get("completion_tokens") or 0) * 0.75e-6
    reason = obj.get("reason") if obj.get("reason") in REASONS else ("keep" if obj.get("keep") else "no_checkable_claim")
    res = {"keep": bool(obj.get("keep")), "reason": reason,
           "political": bool(obj.get("political")), "why": (obj.get("why") or "")[:160]}
    json.dump(res, open(ck, "w"))
    return res, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--provider", default="deepinfra", choices=sorted(PROVIDERS))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    PROVIDER[0] = a.provider
    posts = json.load(open(a.posts))["posts"]
    if a.limit:
        posts = posts[:a.limit]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(str(out) + ".cache"); cache.mkdir(exist_ok=True)
    lock = threading.Lock(); done = [0]; spent = [0.0]; t0 = time.time()
    verdicts = {}

    def work(p):
        r, c, err = None, 0.0, ""
        for attempt in range(3):
            try:
                r, c = call(a.model, p, cache, POST_FILTER_SYS); break
            except Exception as e:  # noqa: BLE001
                err = repr(e); time.sleep(3 * (attempt + 1))
        with lock:
            verdicts[p["post_id"]] = r or {"keep": True, "reason": "keep", "political": False, "why": f"failed {err[:80]}"}
            done[0] += 1; spent[0] += c
            if done[0] % 100 == 0 or done[0] == len(posts):
                el = time.time() - t0
                print(f"  {done[0]}/{len(posts)}  ${spent[0]:.3f}  {el:.0f}s", flush=True)

    print(f"posts {len(posts)}  prompt {PROMPT} ({prompt_hash(POST_FILTER_SYS)})  model {a.model} @ {a.provider}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, p) for p in posts]):
            f.result()
    json.dump({"prompt": PROMPT, "prompt_hash": prompt_hash(POST_FILTER_SYS), "model": a.model,
               "system_prompt": POST_FILTER_SYS, "verdicts": verdicts}, open(out, "w"), indent=0, ensure_ascii=False)
    kept = sum(1 for v in verdicts.values() if v["keep"])
    pol = sum(1 for v in verdicts.values() if v["political"])
    print(f"spent ${spent[0]:.3f}  posts {len(verdicts)}  kept {kept}  dropped {len(verdicts) - kept}  "
          f"political {pol}  failed {sum(1 for v in verdicts.values() if v['why'].startswith('failed'))}  wrote {out}")


if __name__ == "__main__":
    main()
