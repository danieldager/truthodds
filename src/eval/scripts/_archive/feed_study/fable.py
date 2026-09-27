"""FABLE misinformation-harm rubric — self-contained for the feed-study rebuild.

Lifted verbatim from the now-archived `claim_cascade/prompts_scope.py` so the
feed study has no dependency on the abandoned 4-prong "misinfo candidate" stack.
The prompt text is UNCHANGED (Sehat et al., CSCW 2024 — five 1-5 dimensions).

Two scoring modes, IDENTICAL prompt so the two are directly comparable:
  - claim mode  (`build_fable_messages`)      — score a normalized CLAIM, post as context.
  - post  mode  (`build_fable_post_messages`) — score the RAW POST itself, to test
    FABLE as an up-front filter (the post text occupies the CLAIM slot).

`fable_total` = sum of the five dims (5-25). `fable_checkworthy` is DERIVED in
code from a threshold, so recalibrating is a re-run with no LLM calls.
"""

from __future__ import annotations

import json
import re

Messages = list[dict[str, str]]


# ---------------------------------------------------------------------------
# Output cleaner — strips <think>...</think> reasoning blocks (gpt-oss-120b,
# Qwen3) and markdown fences before JSON decoding. Copied from the archived
# extraction_grading.prompts._clean so this module stands alone.
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


# ---------------------------------------------------------------------------
# FABLE harm rubric prompt (UNCHANGED from the archived cascade).
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
    """Claim mode: post for context, single CLAIM to score."""
    return [
        {"role": "system", "content": FABLE_SYSTEM},
        {"role": "user", "content": f"POST:\n{post}\n\nCLAIM:\n{claim}"},
    ]


def build_fable_post_messages(post: str) -> Messages:
    """Post mode (front-filter experiment): score the RAW POST itself.

    The post text fills the CLAIM slot so the prompt is byte-identical to claim
    mode — the only difference is the unit being scored. This tests how FABLE
    would behave as an up-front harm filter on un-normalized posts.
    """
    return build_fable_messages(post, post)


def build_fable_claim_only(claim: str) -> Messages:
    """Claim-only mode: score the CLAIM with NO post context.

    The claim fills both slots (mirrors build_fable_post_messages for posts).
    NOTE: build_fable_messages(post, claim) is the BOTH mode (post as context +
    claim); this is the true claim-only mode.
    """
    return build_fable_messages(claim, claim)


# Dimensions in canonical (FABLE acronym) order. The summed total ranges 5-25.
FABLE_DIMENSIONS = (
    "fragmentation",
    "actionability",
    "believability",
    "spread_likelihood",
    "exploitativeness",
)

def fable_total(dims: dict) -> int:
    """Sum the five FABLE dimensions (range 5-25). Stored for post-hoc analysis."""
    return sum(int(dims[k]) for k in FABLE_DIMENSIONS)


# The three dimensions that encode misinformation HARM proper, and that the two
# model families AGREE on (76-85% exact agreement, vs ~40% for believability and
# spread_likelihood which qwen systematically inflates). The trigger leans on
# these so it is both model-stable and harm-focused.
HARM_DIMS = ("fragmentation", "actionability", "exploitativeness")


def harm3_total(dims: dict) -> int:
    """Sum the three harm-core dimensions (range 3-15)."""
    return sum(int(dims[k]) for k in HARM_DIMS)


# Checkworthy TRIGGER rule. All five raw dims are persisted on every row, so ANY
# rule can be re-derived later with NO LLM calls — re-run run_fable with
# different --total-min / --harm3-min flags.
#
# Rule (user, 2026-06-02): flag if the overall total is high (>=12) OR the three
# harm-core dims are concentrated (>=8). The second clause catches posts that are
# squarely harmful on the agreeing dimensions even when the noisy
# believability/spread dims are low.
TRIGGER_TOTAL_MIN = 12   # fable_total (5-25)
TRIGGER_HARM3_MIN = 8    # sum of HARM_DIMS (3-15)


def fable_flag(dims: dict, total_min: int = TRIGGER_TOTAL_MIN,
               harm3_min: int = TRIGGER_HARM3_MIN) -> bool:
    """Checkworthy trigger: fable_total >= total_min OR harm3_total >= harm3_min."""
    return fable_total(dims) >= total_min or harm3_total(dims) >= harm3_min


class FableParseError(ValueError):
    """Raised when a FABLE response cannot be coerced to the expected schema."""


def _load_json_object(raw: str) -> dict:
    cleaned = _clean(raw)
    if not cleaned:
        raise FableParseError("empty response after cleaning")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_OBJECT.search(cleaned)
        if not m:
            raise FableParseError(f"no JSON object found: {cleaned[:200]!r}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise FableParseError(f"JSON decode failed: {e}; text={cleaned[:200]!r}") from e
    if not isinstance(data, dict):
        raise FableParseError(f"expected object, got {type(data).__name__}")
    return data


def parse_fable(raw: str) -> dict:
    """Parse a FABLE response into the five 1-5 integer dimensions."""
    data = _load_json_object(raw)
    out: dict = {}
    for key in FABLE_DIMENSIONS:
        if key not in data:
            raise FableParseError(f"missing key {key!r}; got {sorted(data.keys())}")
        raw_v = data[key]
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
