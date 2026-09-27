"""LLM enrichment + QA pass over the harmonised eval set.

For each fact-check row, one LLM call returns:
  - judged_axis: which proposition the fact-checker actually adjudicated —
      content     (is the factual claim true?)
      attribution (did person/entity X really say/do Y?)
      artifact    (is this image/video/screenshot authentic / correctly captioned?)
    The headline eval runs on `content` only; attribution/artifact are sliced out
    because they test a different skill than content-veracity (see clog/230626).
  - is_satire: the claim originated as / is labelled satire (metadata flag, so satire
    can be included or excluded per eval run — not silently folded into a class).
  - harmonization_agrees / suggested_label: a QA check on the rule-based 4-class
    label — the LLM flags rows where it disagrees with the rule parse.

Same provider/pattern as harmonize.py (DeepInfra DeepSeek V4 Flash via EXTRACTION_*).
"""
from __future__ import annotations

import json

import requests

from config import (
    EXTRACTION_API_KEY as LLM_API_KEY,
    EXTRACTION_BASE_URL as LLM_BASE_URL,
    EXTRACTION_MODEL as LLM_MODEL,
    VERDICT_OPTIONS,
)

AXES = ("content", "attribution", "artifact")
_VERDICT_SET = set(VERDICT_OPTIONS)
_VERDICT_LIST = " | ".join(f'"{v}"' for v in VERDICT_OPTIONS)

ENRICH_SYSTEM = (
    "You audit and tag fact-check records for an academic evaluation pipeline. "
    "You are given one fact-checker's record: the claim text they reviewed, the "
    "claimant (may be blank), their textual rating, the publisher, and the 4-class "
    "label a rule table assigned to that rating. Return structured JSON.\n\n"
    "=== 1. judged_axis — which kind of proposition did the fact-checker actually "
    "adjudicate? ===\n"
    "- \"content\": a factual claim about the world; the verdict is about whether that "
    "claim is TRUE (e.g. 'inflation fell last year', 'the new policy bans X', 'this "
    "event happened').\n"
    "- \"attribution\": the proposition is that a person/entity SAID, WROTE or DID "
    "something; the verdict is about whether the attribution is ACCURATE (did they "
    "really say it). Typical ratings: Correct Attribution, Incorrect Attribution.\n"
    "- \"artifact\": the proposition is about a MEDIA ITEM's authenticity or context — "
    "whether an image / video / screenshot / photo / audio is real, unaltered, "
    "AI-generated, or correctly captioned. The verdict judges the media, not a "
    "factual claim about the world. Typical ratings: Miscaptioned, Altered, "
    "AI-generated, Fake.\n"
    "  CRITICAL: artifact claim-texts are very often phrased as 'An image/video "
    "authentically shows X' or 'A screenshot authentically shows X'. This is the "
    "ARTIFACT axis whether the rating affirms OR refutes it — the question being "
    "judged is the media's authenticity, not a worldly fact.\n"
    "  If a record is BOTH (e.g. a fabricated screenshot of a quote): pick the axis "
    "of the PRIMARY thing judged — if the verdict turns on the media being fake/"
    "altered, choose artifact; if it turns purely on whether the person said it "
    "(no media-authenticity question), choose attribution.\n\n"
    "=== 2. is_satire ===\n"
    "true iff the claim originated as, or is labelled, satire/parody. Independent of "
    "axis (a satirical false claim is usually still content). Otherwise false.\n\n"
    "=== 3. harmonization audit ===\n"
    f"The 4 classes are: {_VERDICT_LIST}. They describe the verdict on WHATEVER "
    "proposition was reviewed (content, attribution, or artifact alike): Supported = "
    "the reviewed proposition holds (the claim is true / the attribution is correct / "
    "the media is authentic); Refuted = it does not (false / misattributed / fabricated "
    "/ altered / miscaptioned / satirical / an event that did not happen); Not Enough "
    "Evidence = the fact-checker GENUINELY could not confirm or refute (unproven, "
    "unsubstantiated) — do NOT use it for satire or fabricated content, which is "
    "Refuted because the proposition is demonstrably not real; Conflicting Evidence = "
    "partly true, misleading, missing context, mixture. "
    "Given the rule-assigned label, decide whether it is the best 4-class fit for this "
    "rating+claim. Set harmonization_agrees=true if it is; otherwise false and put the "
    "better label in suggested_label. If no rule label was provided, set "
    "harmonization_agrees=true and suggested_label=\"\".\n\n"
    "Respond with JSON ONLY:\n"
    '{"judged_axis": "content|attribution|artifact", "is_satire": true|false, '
    '"harmonization_agrees": true|false, "suggested_label": "<one of the four exact '
    'strings, or empty>", "note": "<short reason, esp. when disagreeing>"}'
)


def enrich_row(
    claim_text: str | None,
    claimant: str | None,
    original_rating: str | None,
    publisher_site: str | None,
    rule_label: str | None,
    timeout: int = 30,
) -> dict:
    """One LLM call → {judged_axis, is_satire, harmonization_agrees, suggested_label, note}."""
    user = (
        f"publisher: {publisher_site or ''}\n"
        f"claimant: {claimant or ''}\n"
        f"textual_rating: {original_rating or ''}\n"
        f"rule_assigned_label: {rule_label or '(none)'}\n"
        f"claim_text: {claim_text or ''}"
    )
    r = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": ENRICH_SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "reasoning_effort": "none",  # keep JSON in content (DeepSeek reasoning⊗json_object)
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    obj = json.loads(r.json()["choices"][0]["message"]["content"] or "{}")

    axis = str(obj.get("judged_axis", "")).strip().lower()
    if axis not in AXES:
        raise ValueError(f"non-canonical judged_axis: {axis!r}")
    suggested = str(obj.get("suggested_label", "")).strip()
    if suggested and suggested not in _VERDICT_SET:
        suggested = next((v for v in VERDICT_OPTIONS if v.lower() in suggested.lower()), "")
    return {
        "judged_axis": axis,
        "is_satire": bool(obj.get("is_satire", False)),
        "harmonization_agrees": bool(obj.get("harmonization_agrees", True)),
        "suggested_label": suggested,
        "note": str(obj.get("note", "")).strip(),
    }


GATE_SYSTEM = (
    "You screen normalized fact-check claims for an automated VERIFICATION eval — a system will "
    "search the web and judge whether each claim is true. Flag claims unsuitable for that. JSON out.\n\n"
    "=== is_checkable ===\n"
    "true if the claim is a SINGLE, self-contained, verifiable factual proposition a researcher "
    "could judge true/false. false if it is NOT a clean proposition: a sentence fragment or bare "
    "headline with no asserted predicate ('Footage of a protest in Paris', 'vote counting before "
    "polling in Bangladesh'), a QUESTION ('A new road sign banning drivers who wear glasses?'), too "
    "vague/ambiguous to verify, or self-referential about its own truth ('a meme accurately claimed "
    "that...', 'a verified report confirmed that...').\n\n"
    "=== is_attribution ===\n"
    "true if the thing to verify is really WHO SAID / POSTED / WROTE something — whether a named "
    "person or entity actually MADE a statement ('Trump said X', 'Hunter Biden said he would run', "
    "'Leavitt claimed Y') — rather than whether the underlying fact is true. false for ordinary "
    "factual claims about the world (events, policies, statistics, media authenticity). Merely "
    "mentioning a person is NOT attribution unless verification hinges on whether they said it.\n\n"
    'Respond with JSON ONLY: {"is_checkable": true|false, "is_attribution": true|false, '
    '"reason": "<short>"}'
)


def quality_gate_row(claim_text: str | None, model: str | None = None, timeout: int = 30) -> dict:
    """One LLM call screening a normalized claim for the verification eval:
      - is_checkable  : a single self-contained verifiable proposition (not a fragment/question)
      - is_attribution: really a 'did X say Y' claim (catches attribution mislabeled as content)
    -> {is_checkable, is_attribution, gate_reason}. Raises on transport/parse error (caller handles).
    """
    r = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": model or LLM_MODEL,
            "messages": [
                {"role": "system", "content": GATE_SYSTEM},
                {"role": "user", "content": f"claim_text: {claim_text or ''}"},
            ],
            "temperature": 0,
            "reasoning_effort": "none",
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    obj = json.loads(r.json()["choices"][0]["message"]["content"] or "{}")
    return {
        "is_checkable": bool(obj.get("is_checkable", True)),
        "is_attribution": bool(obj.get("is_attribution", False)),
        "gate_reason": str(obj.get("reason", "")).strip(),
    }
