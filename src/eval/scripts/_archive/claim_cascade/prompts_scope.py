"""Scope-confirmation + FABLE harm prompts for the claim-filtering cascade.

Two prompts, both designed to compose with the validated v4 extraction stack:

- SCOPE_CONFIRM_SYSTEM (Stage 2, openai/gpt-oss-120b): a POST-LEVEL scope gate.
  It reuses prongs 3 + 4 (PUBLIC-CONSEQUENCE DOMAIN, NEWS-SUBSTITUTABLE) of the
  v4 misinformation-candidate criterion VERBATIM — sliced out of
  `extraction_grading.prompts._MISINFO_CRITERION` at import time — so the scope
  gate and the downstream extractor share one definition of "matters for public
  discourse". Disagreement between the cheap gate and the per-claim judge is
  therefore meaningful, not a definitional artefact. Returns
  {in_scope, topic, reason}.

- FABLE_SYSTEM (Stage 5, qwen/qwen3-32b): a PER-CLAIM harm rubric. Five 1-5
  Likert dimensions — Fragmentation, Actionability, Believability,
  spread-Likelihood, Exploitativeness (the F-A-B-L-E acronym) — plus a derived
  `fable_checkworthy` bool (threshold on the summed total). FABLE runs on the
  SAME claims and the SAME model as the per-claim judge, so the headline
  "4-prong misinfo_candidate vs FABLE harm" comparison isolates the *rubric*,
  not the model.

Both models may emit <think>...</think> reasoning blocks; the parsers reuse
`extraction_grading.prompts._clean` to strip them before JSON-decoding.
"""

from __future__ import annotations

import json
import re

from eval.scripts.extraction_grading.prompts import (
    _MISINFO_CRITERION,
    _clean,
)

Messages = list[dict[str, str]]


# ---------------------------------------------------------------------------
# Prongs 3 + 4, extracted VERBATIM from the shared criterion.
# We slice rather than retype so the scope gate can never drift from the
# extractor/judge definition. If the upstream criterion text is restructured,
# the assertion below fails loudly instead of silently shipping stale prongs.
# ---------------------------------------------------------------------------

_PRONGS_34_RE = re.compile(
    r"3\. PUBLIC-CONSEQUENCE DOMAIN.*?(?=\n\nQUOTED CLAIMS:)",
    re.DOTALL,
)
_m = _PRONGS_34_RE.search(_MISINFO_CRITERION)
if _m is None or "NEWS-SUBSTITUTABLE" not in _m.group(0):
    raise RuntimeError(
        "prompts_scope: could not slice prongs 3+4 out of _MISINFO_CRITERION; "
        "the upstream criterion text changed — update _PRONGS_34_RE."
    )
SCOPE_PRONGS = _m.group(0).strip()


# ---------------------------------------------------------------------------
# Topic taxonomy (locked plan decision).
# These label slugs MIRROR anchors.IN_SCOPE / anchors.OUT_SCOPE VERBATIM so
# `topic_llm` is label-comparable to the embedding `top_topic` for a per-label
# embedding<->LLM agreement crosstab (a Stage-6 / analyze_funnel analysis). We
# mirror rather than import `anchors` because anchors.py eagerly loads the
# multilingual embedding model, which the LLM-only scope stage must not pull in.
# KEEP IN SYNC with anchors.py. `topic` is logged as-is — analysis maps any
# off-list label.
# ---------------------------------------------------------------------------

IN_SCOPE_TOPICS = [
    "politics_elections",
    "health_medicine",
    "science_climate",
    "economics",
    "crime",
    "conflict_war",
    "crisis_disaster",
    "public_figure",
    "identity_public",
]
OUT_OF_SCOPE_TOPICS = [
    "sports",
    "entertainment_celebrity",
    "personal_lifestyle",
    "promotional_ads",
    "social_format",
]

_TOPIC_GUIDE = (
    f"  IN-SCOPE topics: {', '.join(IN_SCOPE_TOPICS)}\n"
    f"  OUT-OF-SCOPE topics: {', '.join(OUT_OF_SCOPE_TOPICS)}\n"
    '  Use "empty" for a media-only post with no substantive text, '
    'and "other" only if nothing else fits.'
)


# ---------------------------------------------------------------------------
# Scope-confirmation prompt (gpt-oss-120b)
# ---------------------------------------------------------------------------

SCOPE_CONFIRM_SYSTEM = f"""You are a scope filter for a misinformation fact-checking pipeline. Given a single social media post, decide whether the post is IN SCOPE — i.e. whether it plausibly carries a factual assertion in a PUBLIC-CONSEQUENCE domain that could substitute for news, and is therefore worth passing on to claim extraction.

You are NOT extracting individual claims, you are NOT judging whether anything is true, and you are NOT scoring harm. You only decide whether this post is in the domain where misinformation matters.

Apply these two criteria — the public-consequence and news-substitutable prongs of the project's misinformation-candidate definition:

{SCOPE_PRONGS}

DECISION:
- in_scope = true  if the post plausibly contains at least one factual assertion meeting BOTH criteria above (a public-consequence domain AND news-substitutable). When genuinely borderline, lean TRUE — a later extraction + per-claim judge is the precise gate; this step only removes posts that are clearly out of domain.
- in_scope = false if the post is wholly personal, promotional/advertising, entertainment/sports, or a social-media-specific format (boost/prayer/mutual-aid request, now-playing auto-share, lyric quote) with no public-consequence factual content.

TOPIC: classify the post's dominant topic with exactly ONE label from this list:
{_TOPIC_GUIDE}

Output strictly valid JSON matching this schema, and NOTHING else:
{{"in_scope": <true|false>, "topic": "<label>", "reason": "<one short sentence>"}}

Do not wrap the JSON in markdown fences. Do not add commentary."""


def build_scope_messages(post: str) -> Messages:
    """OpenAI-style messages for one scope-confirmation call."""
    return [
        {"role": "system", "content": SCOPE_CONFIRM_SYSTEM},
        {"role": "user", "content": f"Post:\n{post}"},
    ]


# ---------------------------------------------------------------------------
# FABLE harm rubric prompt (qwen3-32b)
#
# Five 1-5 dimensions. Higher = more harmful if false / more worth checking.
# This is a POTENTIAL-harm rubric: it is INDEPENDENT of whether the claim is
# actually true. The derived bool is computed in code (see fable_total /
# fable_checkworthy), NOT asked of the model, so the threshold can be
# recalibrated on the 50-set without re-running the LLM.
# ---------------------------------------------------------------------------

FABLE_SYSTEM = """You are assessing the potential HARM and CHECK-WORTHINESS of a single factual claim that another system extracted from a social media post. Score the claim on five dimensions from 1 (low) to 5 (high). Higher scores mean the claim would be MORE harmful and MORE worth prioritizing for fact-checking.

These five dimensions are the FABLE misinformation-harm framework (Sehat et al., "Misinformation as a Harm: Structured Approaches for Fact-Checking Prioritization", CSCW 2024): Fragmentation, Actionability, Believability, Likelihood-of-spread, Exploitativeness. Score the potential for HARM, NOT veracity — do not try to verify whether the claim is true.

You are shown the originating POST for context, then the single CLAIM to score. Score the CLAIM; use the POST only to interpret it. Score each dimension INDEPENDENTLY; do not let one high dimension inflate the others.

DIMENSIONS (score each 1-5):

- fragmentation — damage to social cohesion / trust in shared institutions (government, courts, science, journalism, elections) or in a whole community. (Institutional-trust erosion, NOT mere missing context.)
  1 = self-contained claim with no institutional/societal target; believing it changes no one's trust in institutions, science, media, or elections.
  3 = glancingly questions the competence or honesty of one institution, or fits a distrust narrative about a single isolated case ("this one agency botched the response").
  5 = its core message is that a pillar of shared reality is corrupt/rigged/lying, or that a whole community cannot be trusted, slotting into a sweeping "how the world really works" narrative ("the election was systematically stolen", "mainstream science on X is a coordinated hoax").

- actionability — believing or acting on the claim leads to concrete real-world harm, via an explicit call to action, coordination logistics, or identifying information.
  1 = purely descriptive or opinion; a convinced reader has nothing specific to do and no one to act against.
  3 = loosely encourages a harmful behavior or avoidance without specifics ("don't take the medication", "someone should deal with group X").
  5 = explicit call to act PLUS enabling specifics: coordination details (date/time/place), instructions for a dangerous act, or identifying info (names, addresses, workplaces) that points people at specific targets.

- believability — how readily the TARGET audience could accept it as true (surface plausibility plus content credibility cues), judged relative to that audience, not to absolute truth. Score the content only; you cannot see the author's follower count.
  1 = implausible on its face or trivially debunked — obvious satire/fantasy, or instantly refuted by a quick search.
  3 = plausible to the target community but carries at least one checkable tell (internal inconsistency, missing source, easily found partial rebuttal) a careful reader could catch.
  5 = highly convincing to its audience: coherent, specific, fits what they already expect, no easily found rebuttal, and/or carries credibility cues (authoritative tone, fabricated citations, imposter framing mimicking a real outlet).

- spread_likelihood — how far and wide the claim is likely to travel, from its framing (emotional charge, novelty, controversy, shareable format). This is REACH, kept distinct from harm: viral does not mean harmful.
  1 = niche or dull; no emotional hook, awkward to share, unlikely to leave a small audience.
  3 = circulates within one community or interest group on a mild hook (novelty or moderate emotion) but lacks the broad relatability to break out.
  5 = built to spread: strong outrage/fear/awe, broadly relatable or timely, punchy shareable format; primed to go viral across communities.

- exploitativeness — preys on vulnerability: weaponizes fear, identity, or outgroup animus, or targets susceptible audiences (elderly, low-literacy, economically precarious, stigmatized groups); may exploit an active crisis.
  1 = neutral, non-manipulative; no fear/animus, does not single out a vulnerable group.
  3 = some emotional manipulation or group framing, but mild or secondary rather than the engine of the claim.
  5 = engineered to exploit: deliberately inflames fear or hatred of an outgroup, dehumanizes/scapegoats a marginalized group, or preys on an at-risk audience.

Separating overlapping dimensions:
- fragmentation vs exploitativeness: fragmentation attacks the trustworthiness of institutions/communities as an ABSTRACT target (erodes shared reality); exploitativeness PREYS ON or DEHUMANIZES a specific vulnerable audience.
- actionability vs exploitativeness: actionability asks "is there a concrete act or target?"; exploitativeness asks "is a vulnerability or emotion being weaponized?". An incitement claim can legitimately score high on both.

Output strictly valid JSON matching this schema, and NOTHING else:
{"fragmentation": <1-5>, "actionability": <1-5>, "believability": <1-5>, "spread_likelihood": <1-5>, "exploitativeness": <1-5>}

Do not wrap the JSON in markdown fences. Do not add commentary."""


def build_fable_messages(post: str, claim: str) -> Messages:
    """Messages for one FABLE call — post for context, single claim to score."""
    return [
        {"role": "system", "content": FABLE_SYSTEM},
        {"role": "user", "content": f"POST:\n{post}\n\nCLAIM:\n{claim}"},
    ]


# Dimensions in canonical (FABLE acronym) order. The summed total ranges 5-25.
FABLE_DIMENSIONS = (
    "fragmentation",
    "actionability",
    "believability",
    "spread_likelihood",
    "exploitativeness",
)

# Default cut for the DERIVED bool: fable_checkworthy = (fable_total >= thr).
# total runs 5-25; 15 = mean 3/5 ("moderate") across dimensions. NOTE this
# Likert + sum + threshold scoring is a BESPOKE layer: FABLE (Sehat et al.,
# CSCW 2024) defines the five constructs but uses binary diagnostics and no
# composite score. This default leans toward recall; RECALIBRATE on the
# 50-claim set, then re-run stage5 with the chosen --fable-threshold (the bool
# re-derives from stored totals, no LLM calls). KNOWN LIMITATION: a pure sum
# misses concentrated high-harm claims (e.g. actionability=5 doxxing with
# total<15) — at calibration consider an OR-gate (total>=T OR any dim==5 OR
# count(dim>=4)>=2). All five raw dims are persisted, so any such rule is
# computable post-hoc without re-scoring.
FABLE_CHECKWORTHY_THRESHOLD = 15


def fable_total(dims: dict) -> int:
    """Sum the five FABLE dimensions (range 5-25)."""
    return sum(int(dims[k]) for k in FABLE_DIMENSIONS)


def fable_checkworthy(total: int, threshold: int = FABLE_CHECKWORTHY_THRESHOLD) -> bool:
    """Derived check-worthiness bool: total at or above the threshold."""
    return total >= threshold


# ---------------------------------------------------------------------------
# Output parsers (reuse extraction_grading._clean to strip <think> + fences)
# ---------------------------------------------------------------------------

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ScopeParseError(ValueError):
    """Raised when a scope response cannot be coerced to the expected schema."""


class FableParseError(ValueError):
    """Raised when a FABLE response cannot be coerced to the expected schema."""


def _load_json_object(raw: str, err: type[ValueError]) -> dict:
    cleaned = _clean(raw)
    if not cleaned:
        raise err("empty response after cleaning")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_OBJECT.search(cleaned)
        if not m:
            raise err(f"no JSON object found: {cleaned[:200]!r}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise err(f"JSON decode failed: {e}; text={cleaned[:200]!r}") from e
    if not isinstance(data, dict):
        raise err(f"expected object, got {type(data).__name__}")
    return data


def _to_bool(value, err: type[ValueError]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    raise err(f"in_scope not coercible to bool: {value!r}")


def parse_scope(raw: str) -> dict:
    """Parse a scope response into {in_scope: bool, topic: str, reason: str}.

    `topic` is logged verbatim (stripped) — an off-taxonomy label is not an
    error, so a good scope decision is never lost to a label nit.
    """
    data = _load_json_object(raw, ScopeParseError)
    for key in ("in_scope", "topic", "reason"):
        if key not in data:
            raise ScopeParseError(f"missing key {key!r}; got {sorted(data.keys())}")
    return {
        "in_scope": _to_bool(data["in_scope"], ScopeParseError),
        "topic": str(data["topic"]).strip(),
        "reason": str(data["reason"]).strip(),
    }


def parse_fable(raw: str) -> dict:
    """Parse a FABLE response into the five 1-5 integer dimensions."""
    data = _load_json_object(raw, FableParseError)
    out: dict = {}
    for key in FABLE_DIMENSIONS:
        if key not in data:
            raise FableParseError(f"missing key {key!r}; got {sorted(data.keys())}")
        raw_v = data[key]
        # Reject bools (int(True)==1) and non-integral floats (int(4.9)==4) so a
        # malformed score becomes a visible error row, not a silently shifted total.
        if isinstance(raw_v, bool):
            raise FableParseError(f"{key!r} is a bool, expected an integer 1-5: {raw_v!r}")
        if isinstance(raw_v, float) and not raw_v.is_integer():
            raise FableParseError(f"{key!r} not an integer 1-5: {raw_v!r}")
        try:
            v = int(raw_v)
        except (TypeError, ValueError) as e:
            raise FableParseError(f"{key!r} not coercible to int: {raw_v!r}") from e
        if not 1 <= v <= 5:
            raise FableParseError(f"{key!r} out of range 1-5: {v}")
        out[key] = v
    return out
