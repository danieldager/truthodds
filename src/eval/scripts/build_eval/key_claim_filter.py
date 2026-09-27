"""Key-claim filter (branch reader-iteration, Daniel 2026-09-10): a second call after
key_claim_extract.py that rejects malformed or pointless key claims before retrieval.

Why: reading the v3 extraction, two residues survive the extractor's own rules. Routine
relayed announcements whose only trace on the web is the outlet itself (they read as ten
silent documents and pull a true post to rating 3), and claims that hedge an unresolved
reference instead of leaving it out. The filter only removes claims, so it can be tested
after a retrieval run at zero cost by dropping rejected claims from the scoring.

    uv run python -m eval.scripts.build_eval.key_claim_filter \
        --claims eval/data/reader_lab/extract/keyclaims_v3.json --out eval/data/reader_lab/extract/filter_v3.json

Output: {"prompt", "prompt_hash", "model", "system_prompt", "verdicts": {claim_id: {"keep": bool, "why": str}}}.
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

FILTER_SYS = """\
You are a fact-checker's editor. You get one social media post from a news outlet and
the candidate claims already extracted from it. Decide for each claim whether it is
worth sending to a fact-checker: worth the search, and well formed enough to search.

Reject a claim when any of these holds.

Not self-contained. It hedges between readings ("or", "possibly", "the period the
report covers", "at some point"), leaves a reference unresolved, or gives a figure or
date the post did not commit to. A fact-checker needs one definite proposition.
A year or date that the extractor filled in from the post's date is not a hedge.

Settled by the post itself. The claim is only that the outlet, or someone it quotes,
said or announced something routine, and the post is the statement. Nothing outside
the post would ever confirm or contradict it, and its falsity would mislead nobody.
A statement by a named official, court or organisation is worth keeping when the
statement itself is news that other reporting would carry.

Not a fact. Opinion, prediction, evaluation, hyperbole, a narrative verdict, or a
characterisation whose decisive word no document can settle. A claim about how much,
how little, how well or how badly something is, with no figure, is a matter of degree
and not a fact, whoever asserts it.

Empty. It says only that something happened, emerged, was revealed or was discussed,
without saying what.

Keep everything else, including claims you believe are true. You are judging form and
worth, not truth. A post may end with no claims kept.

Reply with JSON only:
{"verdicts": [{"claim": <claim number>, "keep": true | false, "why": "<one clause>"}]}
"""
PROMPT_V = "filter-key-v1"

# ---------------------------------------------------------------------------
# filter-survey-v1 (Daniel 2026-09-16): a SURVEY-grade filter, not a retrieval
# readiness filter. filter-key-v1 above asks whether a claim is worth searching;
# this one asks whether a claim could be put in front of a survey respondent as a
# clear, checkable, consequential proposition. It also assigns a subtopic from a
# fixed taxonomy derived from the data. filter-key-v1 is untouched.
# ---------------------------------------------------------------------------
SUBTOPICS = [
    "immigration_and_borders", "elections_and_voting", "crime_and_policing",
    "courts_and_legal_process", "economy_and_cost_of_living", "foreign_policy_and_conflict",
    "health_and_public_health", "government_and_officials", "identity_religion_and_culture",
    "media_and_platforms", "education_and_schools", "welfare_housing_and_labour",
    "environment_energy_and_disasters", "other_or_nonpolitical",
]
REASONS = ["keep", "no_claim", "opinion_or_framing", "trivial_or_low_stakes",
           "attribution_only", "forecast_or_unverifiable", "unclear_referent"]

FILTER_SURVEY_SYS = """\
You are preparing material for a public survey. You get one social media post and the
candidate claims already extracted from it. For each claim decide whether it could be put
in front of a survey respondent, who will see the claim alone and be asked whether it is
true, and then file it under one subject area.

Keep a claim only when all five of these hold.

It is the post's own assertion. The post actually asserts it. A proposition a reader would
infer from the post, supply themselves, or attribute to the post in order to argue with it
is not the post's claim, however obviously the post implies it.

It is a proposition, not a stance. It says that something is or was the case, rather than
approving, blaming, predicting, or grading how much, how well or how badly.

Evidence could settle it. Its decisive words denote observable things - an event, an action,
a quantity, a statement someone made - so a competent person with a search engine could in
principle find what settles it. If the word the claim turns on is an evaluation, no evidence
can settle it.

It stands alone for a stranger. Every person, body, place, period and quantity the claim
needs is named inside the claim itself. Someone who never sees the post must know exactly
what is being asserted.

Something turns on it. Its falsity would change how a reader understands a matter of public
consequence. A detail nobody would act on, or one the post itself settles, does not qualify.

Give exactly one reason code per claim, the one that decides it:
keep - all five hold.
no_claim - the post asserts nothing here; the claim was manufactured out of framing, a link,
a reference to an image, or a bare reaction.
opinion_or_framing - the decisive word is an evaluation, a characterisation or a stance.
trivial_or_low_stakes - well formed and checkable, but nothing of public consequence turns on it.
attribution_only - the only content is that someone said or posted something, where who said
it is not itself news and the post is its own evidence.
forecast_or_unverifiable - about the future, a hypothetical, an intention, or a private or
otherwise unknowable matter.
unclear_referent - a person, body, place, period or quantity the claim needs is missing, so a
stranger cannot tell what is being asserted.

Then file every claim, kept or not, under exactly one subject area:
immigration_and_borders - who may enter, stay, work or claim support in a country; border
enforcement, deportation, asylum, refugees, visas, immigrant status as such.
elections_and_voting - candidates, campaigns, ballots, polling, results, voter eligibility and
registration, party contests and endorsements.
crime_and_policing - a crime, an offender, a victim, an arrest, police conduct or resourcing,
terrorism and political violence, crime rates.
courts_and_legal_process - what a court, prosecutor, jury or investigator did: charges, rulings,
trials, verdicts, filings, subpoenas, and the law itself.
economy_and_cost_of_living - prices, wages, jobs, taxes, debt, budgets, trade, industry, and the
money the state raises or spends.
foreign_policy_and_conflict - wars, militaries, weapons, diplomacy, treaties, sanctions, foreign
aid, disputes between states.
health_and_public_health - disease, vaccines, treatments, health agencies, health coverage,
pandemic policy.
government_and_officials - what a government, agency, legislature or officeholder did or said in
office: legislation, appointments, procedure, and officials' conduct and statements.
identity_religion_and_culture - race, gender, sexuality, religion, language, and national or
historical identity as the subject of the claim.
media_and_platforms - news outlets, journalists, social platforms, moderation, online
manipulation, automated systems, and whether a piece of content is authentic.
education_and_schools - schools, universities, curricula, teachers, students, education policy.
welfare_housing_and_labour - benefits, pensions, social care, housing, homelessness, poverty,
workers' rights.
environment_energy_and_disasters - climate, pollution, energy supply, farming and food, natural
disasters and the response to them.
other_or_nonpolitical - sport, celebrity, entertainment, consumer life, and anything else outside
the political domain.

Judge form, worth and subject, never truth. Keep claims you believe are false and drop claims you
believe are true when the five tests say so. A post may end with no claims kept.

Reply with JSON only:
{"verdicts": [{"claim": <claim number>, "keep": true | false, "reason": "<one reason code>", "subtopic": "<one subject area>", "why": "<one clause>"}]}
"""
PROMPT_SURVEY = "filter-survey-v1"

PROMPTS = {PROMPT_V: FILTER_SYS, PROMPT_SURVEY: FILTER_SURVEY_SYS}


def call(model, post_text, handle, created_at, claims, cache, sys_prompt=FILTER_SYS, quoted_text=None):
    key = f"{prompt_hash(sys_prompt)}|{model}|{claims[0]['post_id']}"
    ck = cache / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")
    if ck.exists():
        return json.load(open(ck)), 0.0
    listing = "\n".join(f"[{i}] {c['claim']}" for i, c in enumerate(claims, 1))
    user = f"OUTLET: @{handle or ''}\nPOST DATE: {created_at or 'unknown'}\nPOST:\n{post_text or ''}"
    if sys_prompt is not FILTER_SYS and quoted_text:
        user += f"\n\nQUOTED POST:\n{quoted_text}"
    user += f"\n\nCLAIMS:\n{listing}"
    body = {"model": model, "temperature": 0, "max_tokens": 1500, "response_format": {"type": "json_object"},
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
    survey = sys_prompt is not FILTER_SYS
    res = {}
    for v in (obj.get("verdicts") or []):
        i = v.get("claim") if isinstance(v, dict) else None
        if isinstance(i, int) and 1 <= i <= len(claims):
            d = {"keep": bool(v.get("keep")), "why": (v.get("why") or "")[:160]}
            if survey:
                r = v.get("reason") if v.get("reason") in REASONS else ("keep" if d["keep"] else "")
                d["reason"] = r
                d["subtopic"] = v.get("subtopic") if v.get("subtopic") in SUBTOPICS else "other_or_nonpolitical"
            res[claims[i - 1]["claim_id"]] = d
    for c in claims:   # a claim the model did not rule on is kept
        d = {"keep": True, "why": "no verdict"}
        if survey:
            d["reason"], d["subtopic"] = "keep", "other_or_nonpolitical"
        res.setdefault(c["claim_id"], d)
    json.dump(res, open(ck, "w"))
    return res, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--provider", default="deepinfra", choices=sorted(PROVIDERS))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt", default=PROMPT_V, choices=sorted(PROMPTS), help="filter prompt version")
    a = ap.parse_args()
    PROVIDER[0] = a.provider
    sys_prompt = PROMPTS[a.prompt]
    claims = json.load(open(a.claims))["claims"]
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
                r, c = call(a.model, cs[0].get("post_text"), cs[0].get("handle"), cs[0].get("created_at"), cs, cache,
                            sys_prompt, cs[0].get("quoted_text")); break
            except Exception as e:  # noqa: BLE001
                err = repr(e); time.sleep(3 * (attempt + 1))
        with lock:
            verdicts.update(r or {x["claim_id"]: {"keep": True, "why": f"failed {err[:80]}"} for x in cs})
            done[0] += 1; spent[0] += c
            if done[0] % 100 == 0 or done[0] == len(posts):
                el = time.time() - t0
                print(f"  {done[0]}/{len(posts)}  ${spent[0]:.3f}  {el:.0f}s  ETA {el / done[0] * (len(posts) - done[0]):.0f}s", flush=True)

    print(f"posts {len(posts)}  claims {sum(len(p) for p in posts)}  prompt {a.prompt} ({prompt_hash(sys_prompt)})  model {a.model} @ {a.provider}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, p) for p in posts]):
            f.result()
    json.dump({"prompt": a.prompt, "prompt_hash": prompt_hash(sys_prompt), "model": a.model, "system_prompt": sys_prompt,
               "verdicts": verdicts}, open(out, "w"), indent=0, ensure_ascii=False)
    kept = sum(1 for v in verdicts.values() if v["keep"])
    print(f"spent ${spent[0]:.3f}  claims {len(verdicts)}  kept {kept}  rejected {len(verdicts) - kept}  "
          f"failed {sum(1 for v in verdicts.values() if v['why'].startswith('failed'))}  wrote {out}")


if __name__ == "__main__":
    main()
