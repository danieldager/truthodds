from __future__ import annotations

import json
import re
from collections import Counter
from typing import List
from openai import OpenAI

from config import VERIFICATION_API_KEY, VERIFICATION_BASE_URL, VERIFICATION_MODEL, VERDICT_OPTIONS
from pipeline.models import ClaimEvidence, LikertScores, VerdictResult

_client = OpenAI(base_url=VERIFICATION_BASE_URL, api_key=VERIFICATION_API_KEY)

_SYSTEM = """You are a professional fact-checker. You will be given:
- A factual claim
- A set of verifying questions for that claim
- Evidence items retrieved from the web (each with URL and credibility score 0.0–1.0)

Your task:
1. For each verifying question, extract a direct answer from the evidence and cite the source URL. If no evidence addresses the question, write "No evidence found."
2. Reason over the resulting Q&A pairs to assess the claim. Higher-credibility sources (≥0.6) carry more weight; treat sources below 0.3 as weak unless corroborated.
3. Rate EACH of the four verdict labels independently on a 1–5 Likert scale. The four labels are NOT mutually exclusive — score each one against its own rubric below. The verdict is whichever label has the highest score; break ties in this order: Refuted > Supported > ConflictingEvidence > NotEnoughEvidence.

RUBRIC — score each label 1 (definitely does not apply) to 5 (definitely applies):

Supported — "evidence confirms the claim"
  1: No evidence supports the claim, or evidence contradicts it.
  2: Only a weak/tangential source agrees; or one low-credibility source (<0.3).
  3: One credible source (≥0.5) directly supports the claim, but no corroboration.
  4: Multiple sources agree, or one high-credibility source (≥0.7) directly confirms, with no credible contradiction.
  5: Two or more high-credibility sources (≥0.6) explicitly confirm the claim, including its key qualifiers (numbers, dates, scope), with no credible source disagreeing.

Refuted — "evidence contradicts the claim"
  1: No evidence contradicts the claim, or evidence supports it.
  2: Only a weak/tangential source disagrees; or one low-credibility source (<0.3).
  3: One credible source (≥0.5) directly contradicts the claim, no corroboration.
  4: Multiple sources contradict, or one high-credibility source (≥0.7) directly refutes a key element (number, date, attribution) with no credible support.
  5: Two or more high-credibility sources (≥0.6) explicitly contradict the claim or a load-bearing qualifier of it, with no credible source supporting.

NotEnoughEvidence — "the retrieved evidence does not address the claim"
  Score this LOW (1–2) whenever a credible source directly addresses the substance of the claim, EVEN IF that source is imperfect, partial, or only covers some qualifiers. NEI is about coverage, not certainty.
  1: At least one credible source (≥0.5) directly addresses the claim's substance — pick Supported/Refuted/Conflicting instead. Do NOT use NEI just because you feel uncertain.
  2: Evidence partially addresses the claim (e.g., covers the topic but not the specific number/date), but a reasonable inference is still possible.
  3: Evidence is topically related but does not speak to the specific assertion; inference would be a stretch.
  4: Only low-credibility (<0.3) or off-topic sources retrieved; no credible source touches the claim.
  5: No retrieved item addresses the claim at all, or evidence block is effectively empty.

ConflictingEvidence — "credible sources meaningfully disagree"
  Score this HIGH (4–5) whenever ≥1 credible source supports and ≥1 credible source refutes the same load-bearing element of the claim. Do not default to NEI in this situation.
  1: All credible sources agree (whether supporting or refuting), or only one source exists.
  2: A minor wording difference between sources, but no substantive disagreement.
  3: One credible source (≥0.5) supports and one weak/low-credibility (<0.4) source refutes (or vice versa) — disagreement exists but is one-sided in credibility.
  4: At least one credible source (≥0.5) supports AND at least one credible source (≥0.5) refutes the same key element of the claim.
  5: Two or more high-credibility sources (≥0.6) on each side, supporting vs refuting, on a load-bearing element.

Tie-breaking guidance:
- If Supported and Refuted are both ≥3, raise ConflictingEvidence to at least max(Supported, Refuted).
- If any credible source addresses the claim, NotEnoughEvidence must be ≤2.
- NEI ≥4 requires that NO retrieved source with credibility ≥0.5 speaks to the claim.

Respond ONLY with valid JSON:
{
  "reasoning": "<step-by-step analysis: answer each question from evidence, note credibility of each source used, then apply the rubric explicitly>",
  "verdict": "<Supported|Refuted|Not Enough Evidence|Conflicting Evidence>",
  "likert": {
    "Supported": <1-5>,
    "Refuted": <1-5>,
    "NotEnoughEvidence": <1-5>,
    "ConflictingEvidence": <1-5>
  },
  "justification": "<one or two sentence summary of the verdict>",
  "key_sources": ["<url>", ...]
}
"""


_NEI_LIKERT = LikertScores(Supported=1, Refuted=1, NotEnoughEvidence=5, ConflictingEvidence=1)


def verify_claim(claim_evidence: ClaimEvidence) -> VerdictResult:
    claim = claim_evidence.claim.text
    evidence = claim_evidence.evidence

    if not evidence:
        return VerdictResult(
            claim=claim,
            reasoning="No evidence was retrieved for this claim.",
            verdict="Not Enough Evidence",
            likert=_NEI_LIKERT,
            justification="No web evidence could be retrieved.",
            key_sources=[],
        )

    questions_block = _format_questions(claim_evidence.claim.questions)
    evidence_block = _format_evidence(evidence)
    user_content = f"Claim:\n{claim}\n\nVerifying questions:\n{questions_block}\n\nEvidence:\n{evidence_block}"
    if "qwen" in VERIFICATION_MODEL.lower():
        user_content += "\n\n/no_think"

    response = _client.chat.completions.create(
        model=VERIFICATION_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    data = _parse_json(raw)
    likert = _parse_likert(data.get("likert", {}))
    return VerdictResult(
        claim=claim,
        reasoning=data.get("reasoning", ""),
        verdict=_validate_verdict(data.get("verdict", "Not Enough Evidence")),
        likert=likert,
        justification=data.get("justification", ""),
        key_sources=data.get("key_sources", []),
    )


def _format_questions(questions: List) -> str:
    if not questions:
        return "(no verifying questions provided)"
    return "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))


def _format_evidence(evidence: List) -> str:
    blocks = []
    for i, item in enumerate(evidence, 1):
        blocks.append(
            f"[{i}] URL: {item.url}\n"
            f"    Credibility: {item.credibility:.2f}\n"
            f"    Content: {item.text[:1500]}"  # cap per-source to avoid context overflow
        )
    return "\n\n".join(blocks)


def _parse_likert(raw: dict) -> LikertScores:
    def clamp(v) -> int:
        return max(1, min(5, int(v)))

    return LikertScores(
        Supported=clamp(raw.get("Supported", 1)),
        Refuted=clamp(raw.get("Refuted", 1)),
        NotEnoughEvidence=clamp(raw.get("NotEnoughEvidence", 1)),
        ConflictingEvidence=clamp(raw.get("ConflictingEvidence", 1)),
    )


def _validate_verdict(verdict: str) -> str:
    for option in VERDICT_OPTIONS:
        if option.lower() in verdict.lower():
            return option
    return "Not Enough Evidence"


def verify_claim_sc(claim_evidence: ClaimEvidence, n: int = 3) -> VerdictResult:
    """Run verify_claim n times; return majority verdict with averaged Likert scores."""
    results = [verify_claim(claim_evidence) for _ in range(n)]

    raw = {
        "Supported":          sum(r.likert.Supported for r in results) / n,
        "Refuted":            sum(r.likert.Refuted for r in results) / n,
        "Not Enough Evidence": sum(r.likert.NotEnoughEvidence for r in results) / n,
        "Conflicting Evidence": sum(r.likert.ConflictingEvidence for r in results) / n,
    }
    verdict = _validate_verdict(max(raw, key=raw.get))
    avg_likert = LikertScores(
        Supported=max(1, min(5, round(raw["Supported"]))),
        Refuted=max(1, min(5, round(raw["Refuted"]))),
        NotEnoughEvidence=max(1, min(5, round(raw["Not Enough Evidence"]))),
        ConflictingEvidence=max(1, min(5, round(raw["Conflicting Evidence"]))),
    )

    vote_str = ", ".join(f"{v}×{c}" for v, c in Counter(r.verdict for r in results).most_common())
    matching = [r for r in results if r.verdict == verdict] or results
    canonical = max(matching, key=lambda r: r.confidence)

    return VerdictResult(
        claim=canonical.claim,
        reasoning=f"[SC n={n} — {vote_str}]\n{canonical.reasoning}",
        verdict=verdict,
        likert=avg_likert,
        justification=canonical.justification,
        key_sources=canonical.key_sources,
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)
