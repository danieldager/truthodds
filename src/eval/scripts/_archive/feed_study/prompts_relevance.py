"""Relevance / public-import filter — an INDEPENDENT, composable gate.

This is the "relevance" stage of the proposed cascade (verifiability -> relevance
-> harm). It is deliberately unit-agnostic: the same prompt can be applied to a
raw post (before extraction) or to a normalized claim (after), so we can test it
at different points in the pipeline.

It answers ONE question: does this text concern a matter of PUBLIC CONSEQUENCE
(politics, science, health, public misinformation) — the kind of claim where
being wrong matters publicly — versus personal / interpersonal / promotional /
sports / entertainment content that we do not fact-check.

It does NOT judge verifiability (Selection's job) or harm (FABLE's job).

Returns {in_scope: bool, domain: <label>, reason: <short>}.
"""

from __future__ import annotations

import json
import re

Messages = list[dict[str, str]]


# Reuse the same cleaning approach as the other feed_study prompts.
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


# Domain labels (for analysis; the bool is what gates).
IN_SCOPE_DOMAINS = [
    "politics_government", "elections_officials", "health_medicine",
    "science_climate_environment", "economics_policy", "crime_justice",
    "conflict_war_disaster", "history_public_record", "identity_group_public",
]


RELEVANCE_SYSTEM = """You are a RELEVANCE filter for a misinformation fact-checking pipeline. Given a short piece of social-media text, decide whether it concerns a matter of PUBLIC CONSEQUENCE — the kind of claim where misinformation would actually matter to the public — as opposed to personal, interpersonal, promotional, or pure-entertainment content.

You are NOT judging whether there is a verifiable claim (another stage does that), and NOT judging how harmful it is. Only: is this in the domain we fact-check?

IN SCOPE (in_scope = true) — the text makes or relays a factual claim in a public-consequence domain:
- politics, government, elections, public officials, public policy
- public health, medicine, vaccines, epidemics
- science, climate, environment
- economics, markets, taxation, cost of living
- crime, justice, conflict, war, terrorism, disasters
- history of public record; claims about social or identity groups that shape public discourse
- anything circulating as misinformation/disinformation with public stakes

OUT OF SCOPE (in_scope = false):
- personal or first-person experiences; anything about a PRIVATE individual
- interpersonal or relationship commentary
- personal opinions, tastes, aphorisms, life advice ("this type of comedy doesn't exist anymore", "jealousy turns friends into enemies")
- hyperbole, jokes, rhetorical questions ("the park is a volcano", "why do some children grow more than others?")
- promotional / advertising / product / show content
- SPORTS and ENTERTAINMENT — results, transfers, player performances, celebrity gossip — UNLESS the item carries a genuine political or public-policy implication (a public-safety failure, a discrimination scandal with societal stakes, a state actor's involvement, etc.)
- a description of what an image or video merely shows, when the depicted fact does not connect to a broader, public reality

Borderline rule: if the text plausibly carries a public-consequence factual claim, keep it (in_scope = true). Reserve false for content that is clearly personal, promotional, sports/entertainment, or interpersonal with no public angle — do not drop a genuine public claim just because it is briefly stated.

Output strictly valid JSON, nothing else:
{"in_scope": <true|false>, "domain": "<one in-scope domain label, or 'none'>", "reason": "<one short sentence>"}

Do not wrap in markdown fences. Do not add commentary."""


def build_relevance_messages(text: str) -> Messages:
    """Raw or claim mode: judge a single piece of text (post OR claim)."""
    return [
        {"role": "system", "content": RELEVANCE_SYSTEM},
        {"role": "user", "content": f"Text:\n{text}"},
    ]


def build_relevance_both(post: str, claim: str) -> Messages:
    """Both mode: judge the CLAIM with the POST as context (mirrors FABLE both)."""
    return [
        {"role": "system", "content": RELEVANCE_SYSTEM},
        {"role": "user", "content": f"POST (context):\n{post}\n\nCLAIM to judge:\n{claim}"},
    ]


class RelevanceParseError(ValueError):
    """Raised when a relevance response cannot be coerced to the schema."""


def _to_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "yes", "no"):
        return v.strip().lower() in ("true", "yes")
    return None


def parse_relevance(raw: str) -> dict:
    """-> {in_scope: bool, domain: str, reason: str}."""
    text = _clean(raw)
    if not text:
        raise RelevanceParseError("empty response after cleaning")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT.search(text)
        if not m:
            raise RelevanceParseError(f"no JSON object found: {text[:200]!r}")
        data = json.loads(m.group(0))
    if "in_scope" not in data:
        raise RelevanceParseError(f"missing in_scope; got {sorted(data.keys())}")
    b = _to_bool(data["in_scope"])
    if b is None:
        raise RelevanceParseError(f"in_scope not coercible to bool: {data['in_scope']!r}")
    return {
        "in_scope": b,
        "domain": str(data.get("domain", "none")).strip() or "none",
        "reason": str(data.get("reason", "")).strip(),
    }
