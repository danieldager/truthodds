"""Key-claim selector (branch reader-iteration, Daniel 2026-09-09): from the claims
already extracted for a post, pick the few a fact-checker would actually check.

Why: the outlet audit found a post's rating hanging on one peripheral claim in 96 of
153 flagged posts ("Glenn Beck said nobody is talking about the AI singularity"),
because the post takes the rating of its LOWEST claim over 4.3 claims on average.
Selecting the load-bearing claims first is testable at zero retrieval cost: every
extracted claim of the E2 posts is already scored, so post ratings under
"selected claims only" come straight from the existing scores.

    uv run python -m eval.scripts.build_eval.key_claim_select \
        --posts <posts.json from build_outlet_posts / build_posts> --out <selections.json> [--workers 24]

Input posts.json has posts[{post_id, post_text, claims[{claim_id, claim, checkworthy, score, rating}]}].
Output: {post_id: {"selected": [claim_id, ...], "why": str}}. Cached per post in <out>.cache/.
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

SELECT_SYS = """\
You are a fact-checker's editor. You get a social media post and a numbered list of
factual claims that were extracted from it. Choose the claims a professional
fact-checker would actually check: the post's load-bearing assertions, the ones the
post exists to make and whose falsity would make the post misleading.

Select at most three. Select none if the post makes no checkable central assertion.

Do not select: incidental details and side facts that merely set the scene; opinion,
prediction, evaluation, or rhetoric even when phrased as fact; claims that only
report that someone said or believes something, unless who said it is itself the
point of the post; restatements of a claim you already selected; claims about the
post's own framing rather than about the world.

Reply with JSON only:
{"selected": [<claim numbers>], "why": "<one sentence>"}
"""
PROMPT_V = "select-v1"


def llm(post_text, claims, cache_dir, pid):
    ck = cache_dir / (hashlib.sha256(f"{prompt_hash(SELECT_SYS)}|{pid}".encode()).hexdigest()[:24] + ".json")
    if ck.exists():
        return json.load(open(ck)), 0.0
    listing = "\n".join(f"[{i}] {c['claim']}" for i, c in enumerate(claims, 1))
    body = {"model": eur.VERIFICATION_MODEL, "temperature": 0, "max_tokens": 300,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SELECT_SYS},
                         {"role": "user", "content": f"POST:\n{post_text}\n\nCLAIMS:\n{listing}"}]}
    r = eur._SESSION.post(f"{eur.EXTRACTION_BASE_URL}/chat/completions",
                          headers={"Authorization": f"Bearer {eur.EXTRACTION_API_KEY}"}, json=body, timeout=90)
    r.raise_for_status()
    j = r.json()
    obj = json.loads(j["choices"][0]["message"]["content"] or "{}")
    cost = (j.get("usage") or {}).get("estimated_cost") or 0.0
    sel = [i for i in (obj.get("selected") or []) if isinstance(i, int) and 1 <= i <= len(claims)]
    res = {"selected": [claims[i - 1]["claim_id"] for i in sel][:3], "why": (obj.get("why") or "")[:200]}
    json.dump(res, open(ck, "w"))
    return res, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    posts = json.load(open(a.posts))["posts"]
    out = Path(a.out)
    cache = Path(str(out) + ".cache"); cache.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock(); done = [0]; spent = [0.0]; t0 = time.time()
    res = {}

    def work(p):
        r, c = None, 0.0
        for attempt in range(3):
            try:
                r, c = llm(p.get("post_text") or "", p["claims"], cache, p["post_id"]); break
            except Exception:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
        with lock:
            res[p["post_id"]] = r or {"selected": [], "why": "failed"}
            done[0] += 1; spent[0] += c
            if done[0] % 100 == 0 or done[0] == len(posts):
                el = time.time() - t0
                print(f"  {done[0]}/{len(posts)}  ${spent[0]:.3f}  {el:.0f}s  ETA {el / done[0] * (len(posts) - done[0]):.0f}s", flush=True)

    print(f"posts {len(posts)}  prompt {PROMPT_V} ({prompt_hash(SELECT_SYS)})", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, p) for p in posts]):
            f.result()
    json.dump({"prompt": PROMPT_V, "prompt_hash": prompt_hash(SELECT_SYS), "selections": res},
              open(out, "w"), indent=0, ensure_ascii=False)
    n_sel = [len(v["selected"]) for v in res.values()]
    print(f"spent ${spent[0]:.3f}  mean selected {sum(n_sel) / len(n_sel):.2f}  none selected {sum(1 for x in n_sel if x == 0)}  wrote {out}")


if __name__ == "__main__":
    main()
