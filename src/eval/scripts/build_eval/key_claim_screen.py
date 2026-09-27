"""Key-claim screen (branch reader-iteration, Daniel 2026-09-10): a third call after
key_claim_filter.py that decides whether a kept key claim is the KIND of statement the
survey experiment uses — US national politics and public policy, stated as one
self-contained declarative assertion about a named public actor, institution, action or
statistic — and which side the claim's CONTENT favours if taken as true.

Criteria are the PI's statement types from his previous experiment, written here as
principles. Scope and lean are properties of the claim, not of the outlet that posted it.

    uv run python -m eval.scripts.build_eval.key_claim_screen \
        --claims eval/data/reader_lab/extract/keyclaims_midband.json \
        --filter eval/data/reader_lab/extract/filter_midband.json \
        --out eval/data/reader_lab/extract/screen_midband.json

Output: {"prompt", "prompt_hash", "model", "system_prompt",
         "verdicts": {claim_id: {"in_scope": bool, "domain": str, "lean": str, "why": str}}}.
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

SCREEN_SYS = """\
You are selecting statements for a survey experiment on political misinformation. You
get one social media post from a news outlet and the claims already extracted from it.
Judge each claim on its own: is it the kind of statement the experiment uses, and which
side does its content favour.

SUBJECT. In scope: United States national politics and public policy. Elections and
candidates; the actions of federal or state government; courts and their rulings;
immigration and the border; taxation, federal agencies and public spending; the economy
and trade treated as policy; energy; abortion and reproductive health; education and
culture-war policy; public health treated as policy.

Out of scope: sport; entertainment, celebrity and media personalities; the crime blotter
and local incidents with no policy dimension; weather and natural disasters as such;
business, markets and product news; lifestyle, health and consumer advice.

Two subject boundaries are easy to get wrong.

Events abroad are IN scope whenever what the claim asserts is an act, a decision, a
position or a statement of the United States government, its forces, its agencies or its
office holders, wherever in the world it happens, and whenever it is about United States
policy towards another country. Only a claim whose actors and subject are all foreign is
out of scope.

Crime and violence are OUT of scope as ordinary incident reporting — who was arrested,
charged, killed or found, and the procedural steps of a case involving private people.
They are IN scope when the claim is about the conduct of a public actor in a public role,
about a decision taken by a prosecutor, an agency or a court, or about violence directed
at a national political figure because of their politics.

FORM. Apply the form tests first, and apply them to the claim's own decisive words; a
political subject never rescues a claim that fails them.

In scope: one self-contained declarative assertion, a complete sentence in the register of
a wire report, about a named public actor or institution — an office holder, a candidate,
a party, an agency, a court, a legislature, a state government — or about a government
action, a ruling, a policy, or a public statistic. Government action includes a rule, a
guideline, a budget, a nomination, an investigation, an enforcement operation or a military
deployment decided or published by a federal or state body, and the conduct of its officers
while acting in that role.

Reported speech is in scope only when what was said is itself the news: a position taken,
a decision announced, a commitment made, or a specific factual assertion about the world
that could be checked independently of the fact that the words were uttered. Rhetoric,
insults, boasts, slogans, mockery, campaign attacks, warnings, threats and expressions of
sentiment or uncertainty are out of scope, however prominent the speaker and whether the
words flatter or attack, because the only checkable content is the utterance itself. This
applies equally to office holders, candidates, commentators and journalists.

Out of scope on form: opinion, prediction, characterisation, promotion; a claim whose
decisive word is an evaluation of how good, bad, harmful or justified something is; a
claim that would be settled by a judgment of how much, how many, how well or how badly,
with no figure given; a claim whose actor is a generic or unnamed group rather than a
named person or body; and a claim that reports only where an official went, when they
spoke, or what their schedule was.

LEAN. Judge the content of the claim, not the outlet that published it, and not whether
the claim is true. Ask what would follow IF the claim were true. "pro_dem" when a reader
would take it as favourable to Democrats or as reflecting badly on Republicans;
"pro_rep" when the reverse; "neutral" when it favours neither side, or when it is
unfavourable or favourable to both alike. Out-of-scope claims are "neutral".

Give the subject a short lowercase domain label of one or two words, from the subject
areas above or the nearest fitting description when the claim is out of scope.

Reply with JSON only:
{"verdicts": [{"claim": <claim number>, "in_scope": true | false, "domain": "<short label>",
"lean": "pro_dem" | "pro_rep" | "neutral", "why": "<one clause>"}]}
"""
PROMPT_V = "screen-key-v1"
LEANS = {"pro_dem", "pro_rep", "neutral"}


def call(model, post_text, handle, created_at, claims, cache):
    key = f"{prompt_hash(SCREEN_SYS)}|{model}|{claims[0]['post_id']}|{len(claims)}"
    ck = cache / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")
    if ck.exists():
        return json.load(open(ck)), 0.0
    listing = "\n".join(f"[{i}] {c['claim']}" for i, c in enumerate(claims, 1))
    user = f"OUTLET: @{handle or ''}\nPOST DATE: {created_at or 'unknown'}\nPOST:\n{post_text or ''}\n\nCLAIMS:\n{listing}"
    body = {"model": model, "temperature": 0, "max_tokens": 1500, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SCREEN_SYS}, {"role": "user", "content": user}]}
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
    res = {}
    for v in (obj.get("verdicts") or []):
        i = v.get("claim") if isinstance(v, dict) else None
        if isinstance(i, int) and 1 <= i <= len(claims):
            lean = v.get("lean") if v.get("lean") in LEANS else "neutral"
            res[claims[i - 1]["claim_id"]] = {"in_scope": bool(v.get("in_scope")), "domain": (v.get("domain") or "")[:40],
                                              "lean": lean, "why": (v.get("why") or "")[:160]}
    for c in claims:   # a claim the model did not rule on is out of scope
        res.setdefault(c["claim_id"], {"in_scope": False, "domain": "", "lean": "neutral", "why": "no verdict"})
    json.dump(res, open(ck, "w"))
    return res, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--filter", default=None, help="filter-key-v1 json; only kept claims are screened")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--provider", default="deepinfra", choices=sorted(PROVIDERS))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    PROVIDER[0] = a.provider
    claims = json.load(open(a.claims))["claims"]
    if a.filter:
        keep = json.load(open(a.filter))["verdicts"]
        claims = [c for c in claims if keep.get(c["claim_id"], {}).get("keep")]
    by_post = {}
    for c in claims:
        by_post.setdefault(c["post_id"], []).append(c)
    posts = list(by_post.values())
    if a.limit:
        posts = posts[:a.limit]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(str(out) + ".cache"); cache.mkdir(exist_ok=True)
    lock = threading.Lock(); done = [0]; spent = [0.0]; t0 = time.time()
    verdicts = {}

    def work(cs):
        r, c, err = None, 0.0, ""
        for attempt in range(3):
            try:
                r, c = call(a.model, cs[0].get("post_text"), cs[0].get("handle"), cs[0].get("created_at"), cs, cache); break
            except Exception as e:  # noqa: BLE001
                err = repr(e); time.sleep(3 * (attempt + 1))
        with lock:
            verdicts.update(r or {x["claim_id"]: {"in_scope": False, "domain": "", "lean": "neutral", "why": f"failed {err[:80]}"} for x in cs})
            done[0] += 1; spent[0] += c
            if done[0] % 100 == 0 or done[0] == len(posts):
                el = time.time() - t0
                print(f"  {done[0]}/{len(posts)}  ${spent[0]:.3f}  {el:.0f}s  {done[0] / el:.1f}/s  "
                      f"ETA {el / done[0] * (len(posts) - done[0]):.0f}s", flush=True)

    print(f"posts {len(posts)}  claims {sum(len(p) for p in posts)}  prompt {PROMPT_V} ({prompt_hash(SCREEN_SYS)})  "
          f"model {a.model} @ {a.provider}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, p) for p in posts]):
            f.result()
    json.dump({"prompt": PROMPT_V, "prompt_hash": prompt_hash(SCREEN_SYS), "model": a.model, "system_prompt": SCREEN_SYS,
               "verdicts": verdicts}, open(out, "w"), indent=0, ensure_ascii=False)
    ins = sum(1 for v in verdicts.values() if v["in_scope"])
    lean = {}
    for v in verdicts.values():
        if v["in_scope"]:
            lean[v["lean"]] = lean.get(v["lean"], 0) + 1
    print(f"spent ${spent[0]:.3f}  claims {len(verdicts)}  in scope {ins}  out {len(verdicts) - ins}  "
          f"lean {lean}  failed {sum(1 for v in verdicts.values() if v['why'].startswith('failed'))}  wrote {out}")


if __name__ == "__main__":
    main()
