"""The image-aware enrichment + audit calls (canonical; supersedes the text-only enrich.py role).

Two passes over each core fact-check row (see docs/enrichment_audit_spec.md):
  - pass_a — BLIND post audit: labels the post as the gate/extractor sees it (NO gold, NO rating) →
      topic, in_scope, has_claim, claim_locus, readable. Feeds Stage-1 (selection) + Stage-2 (extraction).
  - pass_b — SIGHTED validation audit: sees claim + rating + gold + post + image → judged_axis, is_satire,
      veracity_agrees, suggested_veracity, rating_subtype_ok, claim_matches_post, image_supports_claim.
      Validates the ground truth; feeds Stage-3 (verdict) + dataset cleaning.

Run each pass with the labeller (AUDIT_MODEL = Qwen3-VL-235B) AND the cross-check (CROSS_MODEL = Gemma-3-27B);
field-level disagreement is the borderline signal (derived in the builder, not asked of the model).
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import requests

from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL

AUDIT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
CROSS_MODEL = "google/gemma-3-27b-it"
_URL = f"{EXTRACTION_BASE_URL}/chat/completions"
_HDR = {"Authorization": f"Bearer {EXTRACTION_API_KEY}"}

PASS_A_SYSTEM = (
    "You audit a dataset of social-media posts for the evaluation of an automatic fact-checking system's "
    "FILTER stage. Judge each post OBJECTIVELY and ONLY from what the post itself shows — never decide whether "
    "any claim is true, and you are given no verdict. The checkable claim often lives in the IMAGE "
    "(a screenshot, a fabricated/quoted headline, a miscaptioned or manipulated photo) — read text AND image.\n\n"
    "TARGET: misinformation that drives POLITICAL POLARIZATION, in a GLOBAL sense — any country, and any "
    "political division (left/right, pro/anti-government, immigration, religion or ethnicity AS politics). A "
    "claim has POLITICAL-POLARIZATION IMPLICATIONS if, were it believed, it would SOW DIVISION: make one "
    "political side or group look unreasonably bad, or a favoured side unreasonably good, or shift sentiment "
    "for or against a group or those in power. Decide by IMPLICATIONS, NOT TOPIC:\n"
    "- IN even if the topic is not explicitly political, when the claim's truth carries political implications "
    "(a fabricated migrant-crime story; a politically weaponised health/science claim; a culture-war claim).\n"
    "- OUT even if the topic is political or names political actors, when the claim's veracity carries NO "
    "polarization implication (a politician's mundane biography; a neutral government-process fact; a true, "
    "uncontested event; a sports result; a celebrity's private life; a consumer product).\n\n"
    "A post HAS A CLAIM only if it asserts a SPECIFIC, verifiable factual proposition (in text or image). NOT a "
    "claim: a fragment/bare label with no predicate (\"Footage of a protest in Paris\"), a QUESTION (\"A road "
    "sign banning drivers who wear glasses?\"), an opinion/value judgment, or unfalsifiable rhetoric (\"the "
    "system is broken\").\n\n"
    "OBVIOUS JOKE / SATIRE: set obvious_joke=true if the post reads as an obvious joke, parody, or satire that "
    "no reasonable reader would take as a real claim. Set it false if a satirical or absurd claim is hard to "
    "distinguish from a genuine one (it could be believed as real) — those deceptive posts are exactly what we "
    "must catch.\n\n"
    "Output strict JSON only:\n"
    '{"topic":"politics|election|health|crime|war|sports|celebrity|product|business|personal|other",'
    '"political_implication":true|false,"has_claim":true|false,"claim_locus":"text|image|both|none",'
    '"obvious_joke":true|false,"readable":true|false,"note":"<one short sentence>"}'
)

PASS_B_SYSTEM = (
    "You audit a dataset of fact-checks for the evaluation of an automatic fact-checking system. For each record "
    "you are given the publisher, the claim (the ground-truth claim), the claimant, the publisher's textual "
    "rating, and the rule-assigned gold_veracity (1-5) + rating_subtype — and, when available, the ORIGINAL post "
    "(text and/or image) the claim came from. Validate the record is what we expect; do NOT re-investigate "
    "whether the claim is true.\n\n"
    "1. judged_axis — which KIND of proposition the fact-checker adjudicated: content = a worldly factual claim "
    "(is it true?); attribution = whether a person/entity really SAID/POSTED/WROTE it; artifact = whether a MEDIA "
    "item is authentic / correctly captioned (Miscaptioned/Altered/AI-generated/Fake-photo, and \"an image "
    "authentically shows X\" — artifact whether affirmed OR refuted). If both, pick PRIMARY. Use the image when "
    "present to tell an artifact case from a content case.\n"
    "2. verdict-gold check — is rating->veracity right? veracity: 5 clearly-true, 4 mostly-true, 3 not-clearly-"
    "either, 2 mostly-false, 1 clearly-false. rating_subtype: mixed = contested/half-true/missing-context "
    "(evidence both ways); unprovable = genuinely no evidence either way — NOT satire/fabrication.\n"
    "3. ground-truth check (only when a post is provided) — does the post actually correspond to this claim, and "
    "is the claim a faithful rendering of what the post asserts/shows?\n\n"
    "Output strict JSON only:\n"
    '{"judged_axis":"content|attribution|artifact","is_satire":true|false,"veracity_agrees":true|false,'
    '"suggested_veracity":1-5,"rating_subtype_ok":true|false,"claim_matches_post":true|false|null,'
    '"image_supports_claim":true|false|null,"note":"<one short sentence>"}'
)


def _imgs(image_paths, k: int = 2) -> list[str]:
    out = []
    for p in (image_paths or [])[:k]:
        if p and Path(p).exists():
            out.append(base64.b64encode(Path(p).read_bytes()).decode())
    return out


def _obj(txt: str) -> dict:
    m = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S), re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _chat(model: str, content, timeout: int = 150) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "response_format": {"type": "json_object"}, "max_tokens": 450}
    last = "empty"
    for _ in range(2):
        try:
            r = requests.post(_URL, headers=_HDR, json=body, timeout=timeout)
            r.raise_for_status()
            o = _obj(r.json()["choices"][0]["message"]["content"])
            if o:
                return o
        except Exception as e:  # noqa: BLE001
            last = str(e)[:80]
    return {"_err": last}


def pass_a(row: dict, model: str) -> dict:
    """BLIND post audit — sees only the post text + image (NO gold/rating)."""
    content = [{"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b}"}}
               for b in _imgs(row.get("image_paths"))]
    content.append({"type": "text", "text": f"{PASS_A_SYSTEM}\n\nPOST TEXT: {row.get('raw_context') or '(none)'}"})
    return _chat(model, content)


def pass_b(row: dict, model: str, gold_veracity, rating_subtype: str) -> dict:
    """SIGHTED validation audit — sees claim + rating + gold + post + image."""
    u = (f"publisher: {row.get('publisher_site')}\nclaimant: {row.get('claimant') or ''}\n"
         f"textual_rating: {row.get('original_rating')}\ngold_veracity: {gold_veracity}\n"
         f"rating_subtype: {rating_subtype}\nclaim: {row.get('claim_text')}")
    if (row.get("raw_context") or "").strip():
        u += f"\nPOST TEXT: {row['raw_context']}"
    content = [{"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{b}"}}
               for b in _imgs(row.get("image_paths"))]
    content.append({"type": "text", "text": f"{PASS_B_SYSTEM}\n\n{u}"})
    return _chat(model, content)
