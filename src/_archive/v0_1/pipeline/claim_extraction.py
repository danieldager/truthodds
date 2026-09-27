from __future__ import annotations

import json
import re
from openai import OpenAI

from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL, EXTRACTION_MODEL, QUERIES_PER_CLAIM
from pipeline.models import AtomicClaim, ClaimExtractionResult

_client = OpenAI(base_url=EXTRACTION_BASE_URL, api_key=EXTRACTION_API_KEY)

_QUESTIONS_SYSTEM = """You are a fact-checking assistant. For each given atomic claim, generate 5–8 targeted verifying questions that a fact-checker would ask to establish or refute the claim.

Good questions target:
- Official statements: "What did <Organisation> officially state about <X> on <date>?"
- Scientific or expert consensus: "What is the established scientific consensus on <X>?"
- Contradicting sources: "Has any credible source contradicted this claim?"
- Specific quantitative facts: "What exact figure does <Organisation> report for <metric>?"
- Temporal context: "Was this claim accurate as of <time period>?"

These are NOT search queries — they are analytical questions for reasoning over retrieved evidence.

Respond ONLY with valid JSON:
{
  "claim_questions": [
    {"questions": ["<question 1>", ..., "<question N>"]},
    ...
  ]
}

Output exactly one entry per input claim, in the same order.
"""

_SYSTEM = """You are a fact-checking assistant. Given a social media post, determine whether it contains verifiable factual claims.

If it does, decompose the post into atomic claims — each a single, independently checkable statement — and generate search queries for each.

Rules for is_checkable:
- Set is_checkable to TRUE whenever the post contains ANY factual assertion about the real world
- Tone and framing are irrelevant: accusatory, sarcastic, emotional, or rhetorical phrasing does NOT make a claim uncheckable
- The following are ALL checkable: what someone said or did, historical events, statistics, scientific claims, allegations, accusations, conspiracy theories, medical claims, election claims, evidence-based claims
- Set is_checkable to FALSE ONLY when the entire post has ZERO factual content — i.e., it is purely a personal preference, a question with no embedded assertion, a future prediction, or an emotional expression

Checkable examples (is_checkable = true):
- "You are watching the cheaters sending in phony ballots" → asserts voter fraud occurred → checkable
- "The evidence shows masks don't stop aerosol transmission" → claim about evidence and masks → checkable
- "The earth is flat" → obviously false, but still a factual assertion → checkable
- "They've been lying to you about the economy" → checkable allegation

Not checkable examples (is_checkable = false):
- "I love this country!" → pure personal sentiment, no factual content
- "Will Biden ban fracking?" → pure question, no assertion
- "Praying for everyone affected" → emotional expression, no factual claim

Rules for claims:
- Each claim must be a single, standalone factual assertion
- Ignore pure opinions/predictions/questions — but NOT false, absurd, or emotionally framed assertions
- Do NOT pre-judge whether a claim is true or false — that is determined in a later step
- A post may yield 0, 1, or several atomic claims

Rules for queries:
- Generate exactly {n} queries per claim, each from a different angle:
  1. Direct factual (e.g. "WHO monkeypox pandemic declaration March 2025")
  2. Context/background (e.g. "WHO monkeypox announcement March 2025 details")
  3. Debunking-oriented (e.g. "WHO monkeypox pandemic 2025 false misleading")
- Queries must use concrete named entities from the claim, not pronouns or placeholders

Respond ONLY with valid JSON matching this schema:
{{
  "is_checkable": true,
  "claims": [
    {{
      "text": "<atomic claim>",
      "queries": ["<query 1>", "<query 2>", "<query 3>"]
    }}
  ]
}}

If there are no checkable claims, respond with:
{{
  "is_checkable": false,
  "claims": []
}}
""".format(n=QUERIES_PER_CLAIM)


def extract_claims(post: str) -> ClaimExtractionResult:
    response = _client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Post:\n{post}\n\n/no_think"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    data = _parse_json(raw)
    claims = [AtomicClaim(**c) for c in data.get("claims", [])]

    if not data.get("is_checkable", False) or not claims:
        return ClaimExtractionResult(is_checkable=False, claims=[])

    questions_per_claim = _generate_questions(claims)
    enriched = [
        AtomicClaim(text=c.text, queries=c.queries, questions=q)
        for c, q in zip(claims, questions_per_claim)
    ]
    return ClaimExtractionResult(is_checkable=True, claims=enriched)


def _generate_questions(claims: list) -> list:
    claims_text = "\n".join(f"Claim {i+1}: {c.text}" for i, c in enumerate(claims))

    response = _client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": _QUESTIONS_SYSTEM},
            {"role": "user", "content": f"Claims:\n{claims_text}\n\n/no_think"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    data = _parse_json(raw)
    entries = data.get("claim_questions", [])
    return [
        entries[i].get("questions", []) if i < len(entries) else []
        for i in range(len(claims))
    ]


def _parse_json(text: str) -> dict:
    # Strip markdown fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)
