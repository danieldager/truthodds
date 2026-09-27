from __future__ import annotations

import json
import re
from typing import Optional
from openai import OpenAI

from config import EXTRACTION_API_KEY, EXTRACTION_BASE_URL, EXTRACTION_MODEL
from pipeline.claim_extraction import _generate_questions
from pipeline.models import AtomicClaim, MisleadingnessResult, VerdictResult
from pipeline.retrieval import retrieve_evidence
from pipeline.verification import verify_claim

_client = OpenAI(base_url=EXTRACTION_BASE_URL, api_key=EXTRACTION_API_KEY)

_SCOPE_SYSTEM = """You are a fact-checking assistant. Given a factual claim, strip all qualifiers to produce a broader, unqualified version (Ã).

Qualifiers to remove:
- Hedging language: "a study shows", "researchers suggest", "some scientists claim", "according to one report"
- Time period restrictions: "in 2020", "during the pandemic", "between 2010–2015"
- Sample restrictions: "in mice", "in lab settings", "among 1,000 participants"
- Population restrictions: "in the US", "among women over 60", "in developing countries"
- Count qualifiers: "one study found", "a few experiments showed"

Ã must be the broadest, most general version of the same core assertion — nothing added, only qualifiers removed.

Also generate exactly 3 search queries for Ã:
1. Direct factual query
2. Scientific consensus or authoritative source query
3. Debunking-oriented query

Respond ONLY with valid JSON:
{
  "derived_claim": "<Ã>",
  "queries": ["<query 1>", "<query 2>", "<query 3>"]
}
"""

_IMPLICATION_SYSTEM = """You are a fact-checking assistant. Given a factual claim A, identify the first natural factual implication B that a reasonable, non-expert reader would likely draw from A.

B must be:
- The single most obvious causal or general conclusion a reader would infer
- A standalone factual assertion (not a paraphrase of A)
- Grounded in what A strongly suggests, not a remote or forced inference

Examples:
- "Cities with more immigrants have higher crime rates" → B = "Immigrants cause higher crime rates"
- "A study found coffee drinkers live longer" → B = "Drinking coffee extends lifespan"

Also generate exactly 3 search queries for B:
1. Direct factual query
2. Scientific consensus or authoritative source query
3. Debunking-oriented query

Respond ONLY with valid JSON:
{
  "derived_claim": "<B>",
  "queries": ["<query 1>", "<query 2>", "<query 3>"]
}
"""

_OMISSION_SYSTEM = """You are a fact-checking assistant. Given a factual claim A, augment it with the implicit circumstances a reasonable person would silently assume when reading it, producing A⁺.

Common silent assumptions to make explicit:
- Scope: that the effect is general or applies broadly, not only in a narrow context
- Persistence: that the effect is ongoing, not a one-time or temporary event
- Causality: that a stated correlation holds as a causal relationship
- Prevalence: that the effect applies to most people or cases, not a specific subgroup

Example: "This policy reduces employment in the tech sector" → A⁺ = "This policy reduces employment overall"

A⁺ must be a single, standalone factual assertion that captures what a reasonable reader would silently believe A implies.

Also generate exactly 3 search queries for A⁺:
1. Direct factual query
2. Scientific consensus or authoritative source query
3. Debunking-oriented query

Respond ONLY with valid JSON:
{
  "derived_claim": "<A⁺>",
  "queries": ["<query 1>", "<query 2>", "<query 3>"]
}
"""

_CHECKS = [
    ("scope", _SCOPE_SYSTEM),
    ("implication", _IMPLICATION_SYSTEM),
    ("omission", _OMISSION_SYSTEM),
]


def check_misleadingness(
    claim: AtomicClaim,
    original_verdict: VerdictResult,
    date_cutoff: Optional[str] = None,
) -> MisleadingnessResult:
    if original_verdict.verdict != "Supported":
        return MisleadingnessResult(original_claim=claim.text, is_misleading=False)

    for check_type, system_prompt in _CHECKS:
        print(f"    [misleadingness] Running {check_type} check on: {claim.text[:80]}")
        derived = _generate_derived_claim(claim.text, system_prompt)
        if derived is None:
            continue

        derived_text, queries = derived
        print(f"    [misleadingness] Derived claim ({check_type}): {derived_text[:80]}")

        atomic = AtomicClaim(text=derived_text, queries=queries)
        questions_list = _generate_questions([atomic])
        if questions_list:
            atomic = AtomicClaim(text=derived_text, queries=queries, questions=questions_list[0])

        evidence = retrieve_evidence(atomic, date_cutoff=date_cutoff)
        verdict = verify_claim(evidence)

        print(f"    [misleadingness] Derived verdict: {verdict.verdict}")

        if verdict.verdict == "Refuted":
            return MisleadingnessResult(
                original_claim=claim.text,
                is_misleading=True,
                misleading_type=check_type,
                derived_claim=derived_text,
                derived_verdict=verdict,
            )

    return MisleadingnessResult(original_claim=claim.text, is_misleading=False)


def _generate_derived_claim(
    claim_text: str, system_prompt: str
) -> Optional[tuple[str, list[str]]]:
    try:
        response = _client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Claim: {claim_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = response.choices[0].message.content
        data = _parse_json(raw)
        derived_text = data.get("derived_claim", "").strip()
        queries = data.get("queries", [])
        if not derived_text or not queries:
            return None
        return derived_text, queries
    except Exception as e:
        print(f"    [misleadingness] LLM error: {e}")
        return None


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)
