"""Claimify-style claim extraction prompts, adapted for noisy social posts.

Reconstructs the three-stage pipeline from Claimify (Metropolitansky & Larson,
"Towards Effective Extraction and Evaluation of Factual Claims", ACL 2025;
arXiv:2502.10855) — Selection -> Disambiguation -> Decomposition — rather than
copying its prose-tuned prompts verbatim, because our input is short multilingual
social media posts, not long-form LLM output.

Two deliberate adaptations from the paper:
  1. POST-LEVEL, not sentence-by-sentence. Posts are short; we treat the whole
     post as the unit (the paper splits long passages into sentences first).
  2. HYBRID / CONFIDENCE-FLAGGED disambiguation. The paper either returns a
     disambiguated sentence or the label "Cannot be disambiguated". We keep that
     skip-when-unresolvable behaviour AND attach a confidence level so a
     downstream stage can treat low-confidence resolutions differently.

Each stage: a system prompt, a `build_*` message builder, and a `parse_*` parser
returning a validated dict. Models (gpt-oss-120b, qwen3-32b) may emit
<think>...</think>; `_clean` strips it before JSON decoding.
"""

from __future__ import annotations

import json
import re

Messages = list[dict[str, str]]


# ---------------------------------------------------------------------------
# Shared output cleaner (self-contained, mirrors feed_study.fable._clean).
# ---------------------------------------------------------------------------

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")
_THINK_CLOSED = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TAIL = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _clean(raw: str) -> str:
    text = raw.strip()
    text = _THINK_CLOSED.sub("", text)
    text = _THINK_OPEN_TAIL.sub("", text).strip()
    text = _FENCE_OPEN.sub("", text)
    text = _FENCE_CLOSE.sub("", text)
    return text.strip()


class ClaimifyParseError(ValueError):
    """Raised when a stage response cannot be coerced to the expected schema."""


def _load_json_object(raw: str) -> dict:
    cleaned = _clean(raw)
    if not cleaned:
        raise ClaimifyParseError("empty response after cleaning")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_OBJECT.search(cleaned)
        if not m:
            raise ClaimifyParseError(f"no JSON object found: {cleaned[:200]!r}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise ClaimifyParseError(f"JSON decode failed: {e}; text={cleaned[:200]!r}") from e
    if not isinstance(data, dict):
        raise ClaimifyParseError(f"expected object, got {type(data).__name__}")
    return data


def _req_bool(data: dict, key: str) -> bool:
    if key not in data:
        raise ClaimifyParseError(f"missing key {key!r}; got {sorted(data.keys())}")
    v = data[key]
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "yes", "no"):
        return v.strip().lower() in ("true", "yes")
    raise ClaimifyParseError(f"{key!r} not coercible to bool: {v!r}")


def _opt_str(data: dict, key: str) -> str | None:
    v = data.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


# ===========================================================================
# Stage 1 — SELECTION
# Does the post contain a specific, verifiable proposition? If so, return a
# cleaned version with non-verifiable / evaluative language removed.
# ===========================================================================

SELECTION_SYSTEM = """You are the SELECTION stage of a claim-extraction pipeline for fact-checking social media posts. Your job is to decide whether a post contains at least one SPECIFIC, VERIFIABLE proposition, and if so, to rewrite it keeping only the verifiable factual content.

A VERIFIABLE PROPOSITION is a statement about the world that could in principle be checked against external evidence (records, data, expert consensus, reporting). Judge ONLY verifiability — do NOT judge whether the statement is true, whether it is important, or whether it is ambiguous (later stages handle those).

NOT verifiable: pure opinions, value judgments, feelings, hopes, predictions about the future, questions, requests, greetings, jokes with no factual core, and statements that only say the author lacks information.

Reason in four steps, then decide:
1. What is the post literally saying? (resolve obvious meaning, ignore emojis/hashtags/URLs)
2. Does it contain a specific factual proposition (an event, statistic, attribution, causal or state-of-affairs claim), as opposed to only opinion/expressive content?
3. If yes, what is the minimal factual core? Strip evaluative wrappers ("I think", "outrageously", "sadly"), keep the asserted fact.
4. Decide verifiable true/false.

If the post quotes or reports someone else's factual statement, that quoted proposition COUNTS as verifiable content (keep the attribution in the cleaned text, e.g. "X said that Y").

Output strictly valid JSON, nothing else:
{"reasoning": "<brief 1-2 sentence trace>", "verifiable": <true|false>, "cleaned": "<the factual core, or null if not verifiable>"}

If verifiable is false, cleaned MUST be null. Do not wrap in markdown fences."""


def build_selection_messages(post: str) -> Messages:
    return [
        {"role": "system", "content": SELECTION_SYSTEM},
        {"role": "user", "content": f"Post:\n{post}"},
    ]


def parse_selection(raw: str) -> dict:
    """-> {reasoning: str, verifiable: bool, cleaned: str|None}."""
    data = _load_json_object(raw)
    verifiable = _req_bool(data, "verifiable")
    cleaned = _opt_str(data, "cleaned")
    if not verifiable:
        cleaned = None
    elif cleaned is None:
        raise ClaimifyParseError("verifiable=true but cleaned is null/empty")
    return {
        "reasoning": _opt_str(data, "reasoning") or "",
        "verifiable": verifiable,
        "cleaned": cleaned,
    }


# ===========================================================================
# Stage 2 — DISAMBIGUATION  (hybrid / confidence-flagged)
# Resolve referential + structural ambiguity using the post as context.
# Resolve when confident; otherwise abstain (can_disambiguate=false).
# ===========================================================================

DISAMBIGUATION_SYSTEM = """You are the DISAMBIGUATION stage of a claim-extraction pipeline. You receive a factual sentence extracted from a social media post, plus the original post for context. Your job is to produce a DECONTEXTUALIZED version: a sentence that an informed reader could understand and fact-check on its own, with no reference to the post.

Do two things:
1. COMPLETE partial names, undefined acronyms, and unresolved references (pronouns, "this", "the company", "yesterday") using ONLY information available in the post. Convert relative time ("yesterday", "today") to whatever absolute reference the post provides; if none is available, leave the relative term as-is rather than inventing a date.
2. Detect LINGUISTIC AMBIGUITY that a careful reader could not confidently resolve:
   - REFERENTIAL: it is unclear who/what an entity or pronoun refers to.
   - STRUCTURAL: the sentence has more than one valid syntactic reading.
   (Mere vagueness or generality is NOT ambiguity — do not flag it.)

DECISION (this is the core of the stage):
- If every reference can be completed and any ambiguity has a single interpretation that informed readers would agree on GIVEN THE POST, set can_disambiguate=true and return the fully decontextualized sentence. Do NOT introduce any fact not present in the post.
- If a reference cannot be resolved from the post, or genuine ambiguity remains with no agreed reading, set can_disambiguate=false (abstain). Returning a guessed resolution is worse than abstaining.

Also report your CONFIDENCE in the resolution: "high" (unambiguous given the post), "medium" (resolved but with some interpretive judgment), "low" (resolved only by a weak guess — prefer can_disambiguate=false here).

Output strictly valid JSON, nothing else:
{"reasoning": "<brief trace of references resolved + ambiguity found>", "can_disambiguate": <true|false>, "confidence": "<high|medium|low>", "decontextualized": "<standalone sentence, or null if can_disambiguate is false>"}

If can_disambiguate is false, decontextualized MUST be null. Do not wrap in markdown fences."""


def build_disambiguation_messages(post: str, sentence: str) -> Messages:
    return [
        {"role": "system", "content": DISAMBIGUATION_SYSTEM},
        {"role": "user", "content": f"POST (context):\n{post}\n\nSENTENCE to decontextualize:\n{sentence}"},
    ]


_CONFIDENCE_LEVELS = {"high", "medium", "low"}


def parse_disambiguation(raw: str) -> dict:
    """-> {reasoning, can_disambiguate: bool, confidence: str, decontextualized: str|None}."""
    data = _load_json_object(raw)
    can = _req_bool(data, "can_disambiguate")
    conf = (_opt_str(data, "confidence") or "").lower()
    if conf not in _CONFIDENCE_LEVELS:
        # Don't fail the row on a confidence-label nit; default to the cautious end.
        conf = "low"
    decon = _opt_str(data, "decontextualized")
    if not can:
        decon = None
    elif decon is None:
        raise ClaimifyParseError("can_disambiguate=true but decontextualized is null/empty")
    return {
        "reasoning": _opt_str(data, "reasoning") or "",
        "can_disambiguate": can,
        "confidence": conf,
        "decontextualized": decon,
    }


# ===========================================================================
# Stage 3 — DECOMPOSITION
# Break the decontextualized sentence into self-contained, minimal claims.
# "Molecular" target: standalone but minimal — do NOT over-atomize.
# ===========================================================================

DECOMPOSITION_SYSTEM = """You are the DECOMPOSITION stage of a claim-extraction pipeline. You receive a decontextualized factual sentence. Break it into the set of distinct, self-contained factual claims it asserts, suitable for INDEPENDENT fact-checking.

Rules:
- Each claim must be a single declarative sentence that stands ALONE — fully understandable without the other claims or the original post.
- Aim for MINIMAL but COMPLETE units: split genuinely separate facts, but do NOT over-decompose. Keep a claim whole when splitting it would strip context needed to interpret it. Prefer the smallest unit that is still self-contained and checkable on its own.
- PRESERVE ATTRIBUTION: if the sentence reports that a specific actor said or did something, keep that actor in each claim ("According to [actor], ...", or "[Actor] did X").
- Add only essential clarifications in [square brackets] when a fact-checker would otherwise be unable to interpret the claim. Do NOT add facts beyond what the sentence states.
- If the sentence asserts exactly one fact, return a single-element list.

Output strictly valid JSON, nothing else:
{"claims": ["<claim 1>", "<claim 2>", ...]}

The list must be non-empty. Do not wrap in markdown fences."""


def build_decomposition_messages(sentence: str) -> Messages:
    return [
        {"role": "system", "content": DECOMPOSITION_SYSTEM},
        {"role": "user", "content": f"Decontextualized sentence:\n{sentence}"},
    ]


def parse_decomposition(raw: str) -> dict:
    """-> {claims: list[str]} (non-empty)."""
    data = _load_json_object(raw)
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise ClaimifyParseError(f"claims must be a list, got {type(claims).__name__}")
    claims = [str(c).strip() for c in claims if str(c).strip()]
    if not claims:
        raise ClaimifyParseError("claims list is empty after cleaning")
    return {"claims": claims}
