"""v4 extraction + judge prompts — misinformation-candidate filter.

Designed around a synthesis of Guriev/Henry/Marquis/Zhuravskaya (2023),
Vraga & Bode (2020), Wardle & Derakhshan (2017), and Lazer et al. (2018).
See the literature brief notes and clog/200526.md (or v3_vs_v4_comparison.md
in results/) for the operational definition.

Two prompts only:
- EXTRACTION_SYSTEM: detect + extract, returns JSON {has_claim, claims}.
- PER_CLAIM_JUDGE_SYSTEM: per-claim rubric, returns 3 Likert + 1 bool.

The v3 detection judge has been removed. Detection correctness is inferred
from per-claim `misinfo_candidate` (bool) and the extractor's `has_claim`.

Model: gpt-oss-120b (extractor) on Groq; qwen3-32b (judge) on Groq.
Both are reasoning models that may emit <think>...</think> blocks; the
parsers strip those before JSON-decoding.
"""

from __future__ import annotations

import json
import re

Messages = list[dict[str, str]]


# ---------------------------------------------------------------------------
# Shared misinformation-candidate criterion. Embedded verbatim in both prompts
# so extractor and judge use the IDENTICAL definition. Disagreement between
# them is therefore meaningful — not a definitional artefact.
# ---------------------------------------------------------------------------

_MISINFO_CRITERION = """A claim is a POSSIBLE-MISINFORMATION CANDIDATE if and only if ALL FOUR criteria hold:

1. FACTUAL ASSERTION — a specific past or present claim about the world (event, statistic, attribution, causal claim, state of affairs).
   NOT: opinions, value judgments, predictions, hypotheticals, questions, wishes, hopes, feelings, instructions, requests.

2. VERIFIABLE IN PRINCIPLE — can be checked against external evidence (expert consensus, public records, scientific findings, news archives).
   NOT: vague generalizations that can't be tested ("rights are being taken away", "workplace is changing", "mosquitoes are wild"), pure subjective generalizations ("X is the best"), abstract sociological musings ("positive externalities are vanishing").

3. PUBLIC-CONSEQUENCE DOMAIN — politics / elections, public officials, health / medicine, vaccines, science / climate, economics / markets, crime, conflict / war, crisis events (disasters, attacks), identity-group claims that affect public discourse, history of public record.
   NOT: personal experiences, interpersonal anecdotes, promotional content, event logistics, podcast / show plugs, social-media-specific formats (boost requests, prayer requests, mutual aid, now-playing auto-shares).

4. NEWS-SUBSTITUTABLE — could plausibly be shared as informational content that mimics news, not just as expressive / social content. The post does not have to be news-styled, but the claim itself must be the kind that could appear in a news article or fact-check piece.
   NOT: first-person emotional framings even when they contain a kernel of political content (e.g. "I'm distressed her rights are being taken away" — the kernel is too vague and the framing is expressive).

QUOTED CLAIMS: if the post quotes someone else (op-ed, news clip, public figure's statement, retweet) and the quoted content is itself a verifiable claim in a public-consequence domain, TREAT IT AS A MISINFORMATION CANDIDATE. Amplifying false claims by quoting is a misinformation vector — the speaker is effectively asserting by sharing.

EMBEDDED CLAIMS: if a post about a private person contains an embedded claim that itself meets criteria 1-4, extract the embedded claim only ("My coworker said vaccines cause autism" — extract "Vaccines cause autism").

Reference examples — POSITIVE (misinformation candidates):
- "Mexican President criticized Trump in a televised address." — politics, public figure, verifiable, news-form.
- "Vaccines cause autism." — health / science, canonical misinformation.
- "Sweetwater Wetlands are burning." — local crisis event, verifiable.
- "The Earth is flat." — general factual claim in science, no current news hook needed.
- '"Agricultural trade provides benefits for US farmers and consumers" — Cato op-ed quoted in a post': the speaker is amplifying a quoted economic-policy claim in a public-consequence domain. Misinfo candidate.
- "US secured the release of three Americans in a prisoner swap with China." — current events, politics.

Reference examples — NEGATIVE (skip):
- "I am now listening to 朝 by Verandah." — first-person current activity.
- "I tracked 6 songs in 4 days at the Albuquerque studio." — first-person private experience with specifics; still personal.
- "Cats always win #vore." — subjective generalization, not verifiable.
- "The mosquitoes are WILD this summer!" — hyperbolic personal observation.
- "We even have a manifestation of Noel Edmonds this week!" — podcast plug, promotional.
- "Preview show is this Sunday at 7 pm Eastern." — event logistics.
- "Today my granddaughter will arrive in the world... her rights are being taken away." — first-person emotional framing; political kernel too vague to test.
- "Don't stop believin' / Hold on to that feelin'." — lyric quote, social-media-specific format.
- "BREAKING: Florida man arrested for trying to teach gator to drive." — satire-intended-as-satire, not news-substitutable.
- "Since we found out the horrors of turkey factories, any ideas for substitutes?" — the "horrors of X" is a rhetorical premise setting up a question; not load-bearing as an assertion."""


# ---------------------------------------------------------------------------
# Extraction prompt (gpt-oss-120b)
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = f"""You are a fact-checking assistant. Given a social media post, extract only atomic claims that are POSSIBLE-MISINFORMATION CANDIDATES — the kind of claims that beg to be fact-checked because misinformation is a documented danger in their domain.

{_MISINFO_CRITERION}

For posts that DO contain at least one misinformation-candidate claim:
- Decompose into the smallest factual statements that assert it. One sentence per claim.
- Resolve pronouns and references using only information in the post. Do not introduce facts not in the post.
- Sarcastic or emotional framing does NOT disqualify a claim — what matters is whether the underlying assertion meets all four criteria.
- Do NOT pre-judge whether a claim is true or false; that is determined later.

If the post contains no misinformation-candidate claim, return has_claim=false with claims=[].

Test before extracting: would a professional fact-checker, paid per claim, take this one? If unsure, lean toward has_claim=false. The cost of a false positive (extracting non-misinfo content) is higher than the cost of a false negative (missing one).

Output strictly valid JSON matching this schema, and NOTHING else:
{{"has_claim": <true|false>, "claims": [<string>, ...]}}

Constraints:
- If has_claim is false, claims MUST be [].
- If has_claim is true, claims MUST be a non-empty list, each entry a single clear sentence.
- Do not wrap the JSON in markdown fences. Do not add commentary."""


def build_messages(post: str) -> Messages:
    """OpenAI-style messages for one extraction call."""
    return [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": f"Post:\n{post}"},
    ]


# ---------------------------------------------------------------------------
# Per-claim judge prompt (qwen3-32b)
# ---------------------------------------------------------------------------

PER_CLAIM_JUDGE_SYSTEM = f"""You are an expert fact-checker evaluating a single normalized factual claim that another system extracted from a social media post. Score the claim on three Likert dimensions (1-5) plus one boolean.

DIMENSIONS:

- fidelity (1-5): does the claim faithfully preserve the post's factual content without adding or omitting load-bearing detail?
  - 5: claim matches what the post asserts exactly
  - 1: claim misrepresents or fabricates content not in the post

- decontextualized (1-5): can the claim be checked standalone — no pronouns, no unresolved references, named entities and dates resolved using only what the post provides?
  - 5: fully standalone, all entities named
  - 1: depends entirely on the post's context to be intelligible

- verifiability (1-5): is the claim a CHECKABLE factual assertion in principle? This is INDEPENDENT of subject matter — purely about whether the text is verifiable text or not. A claim about personal experience can score verifiability=5 if it is concrete and falsifiable (even though it would not be a misinformation candidate).
  - 5: concrete falsifiable assertion with specific named entities, dates, or quantities.
       Example: "The Mexican President addressed the UN on October 12, 2024."
  - 4: specific factual assertion with some vagueness.
       Example: "Most economists agree tariffs raise consumer prices."
  - 3: factually framed but vague, contested, or generalised.
       Example: "The economy is recovering."
  - 2: largely subjective generalization with a factual surface.
       Examples: "Cats always win", "Mosquitoes are wild this summer".
  - 1: not a checkable claim at all — pure opinion, metaphor, prediction, feeling.
       Examples: "X is the best", "Drake is transforming into a corn cob", "I love this", "Things will get worse".

MISINFO_CANDIDATE (boolean):

misinfo_candidate is TRUE if and only if the claim is a possible-misinformation candidate per the criterion below. FALSE otherwise.

{_MISINFO_CRITERION}

Reasoning hints:
- verifiability and misinfo_candidate are RELATED but DISTINCT. A claim can be verifiability=5 (concrete, falsifiable) AND misinfo_candidate=FALSE (e.g. a personal experience with specific numbers like "I scored 8 goals last Sunday" — concrete but not in a public-consequence domain).
- A claim CANNOT be verifiability=1 AND misinfo_candidate=TRUE (pure opinion / metaphor can't be a misinformation candidate, since it's not a factual assertion).
- For misinfo_candidate=TRUE you typically expect verifiability>=3.

You see exactly ONE claim. Do NOT consider whether the post contains other claims; judge this one only.

Return strict JSON only, with this shape and nothing else:
{{"fidelity": <1-5>, "decontextualized": <1-5>, "verifiability": <1-5>, "misinfo_candidate": <true|false>}}
No prose outside the JSON."""


def build_per_claim_messages(post: str, claim: str) -> Messages:
    user = f"POST:\n{post}\n\nCLAIM:\n{claim}"
    return [
        {"role": "system", "content": PER_CLAIM_JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")
_THINK_CLOSED = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_TAIL = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _clean(raw: str) -> str:
    text = raw.strip()
    # Strip closed reasoning blocks (gpt-oss-120b, Qwen3, DeepSeek-R1).
    text = _THINK_CLOSED.sub("", text)
    # Drop an unclosed <think> tail if max_tokens cut it off mid-reasoning.
    text = _THINK_OPEN_TAIL.sub("", text).strip()
    # Strip markdown code fences if the model wrapped its JSON despite instructions.
    text = _FENCE_OPEN.sub("", text)
    text = _FENCE_CLOSE.sub("", text)
    return text.strip()


class ExtractionParseError(ValueError):
    """Raised when the LLM output cannot be coerced to the expected schema."""


def parse_extraction(raw: str) -> dict:
    """Parse a raw LLM response into `{"has_claim": bool, "claims": list[str]}`."""
    cleaned = _clean(raw)
    if not cleaned:
        raise ExtractionParseError("empty response after cleaning")

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_OBJECT.search(cleaned)
        if not m:
            raise ExtractionParseError(f"no JSON object found in response: {cleaned[:200]!r}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise ExtractionParseError(f"JSON decode failed: {e}; text={cleaned[:200]!r}") from e

    if not isinstance(data, dict):
        raise ExtractionParseError(f"expected object, got {type(data).__name__}")
    if "has_claim" not in data or "claims" not in data:
        raise ExtractionParseError(f"missing keys; got {sorted(data.keys())}")

    has_claim = data["has_claim"]
    claims = data["claims"]

    if not isinstance(has_claim, bool):
        raise ExtractionParseError(f"has_claim must be bool, got {type(has_claim).__name__}")
    if not isinstance(claims, list):
        raise ExtractionParseError(f"claims must be list, got {type(claims).__name__}")
    if not all(isinstance(c, str) for c in claims):
        raise ExtractionParseError("all claims must be strings")

    claims = [c.strip() for c in claims if c.strip()]

    if has_claim and not claims:
        raise ExtractionParseError("has_claim=true but claims is empty")
    if not has_claim and claims:
        raise ExtractionParseError("has_claim=false but claims is non-empty")

    return {"has_claim": has_claim, "claims": claims}


class JudgeParseError(ValueError):
    """Raised when a judge response cannot be coerced to the expected schema."""


_PER_CLAIM_INT_KEYS = ("fidelity", "decontextualized", "verifiability")
_PER_CLAIM_BOOL_KEYS = ("misinfo_candidate",)


def _load_json_object(raw: str) -> dict:
    cleaned = _clean(raw)
    if not cleaned:
        raise JudgeParseError("empty response after cleaning")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_OBJECT.search(cleaned)
        if not m:
            raise JudgeParseError(f"no JSON object found: {cleaned[:200]!r}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise JudgeParseError(f"JSON decode failed: {e}; text={cleaned[:200]!r}") from e
    if not isinstance(data, dict):
        raise JudgeParseError(f"expected object, got {type(data).__name__}")
    return data


def _coerce_bool(value, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    raise JudgeParseError(f"{key!r} not coercible to bool: {value!r}")


def parse_per_claim(raw: str) -> dict:
    """Parse a per-claim judge response into 3 ints (1-5) + 1 bool."""
    data = _load_json_object(raw)
    out: dict = {}
    for key in _PER_CLAIM_INT_KEYS:
        if key not in data:
            raise JudgeParseError(f"missing key {key!r}; got {sorted(data.keys())}")
        try:
            v = int(data[key])
        except (TypeError, ValueError) as e:
            raise JudgeParseError(f"{key!r} not coercible to int: {data[key]!r}") from e
        if not 1 <= v <= 5:
            raise JudgeParseError(f"{key!r} out of range 1-5: {v}")
        out[key] = v
    for key in _PER_CLAIM_BOOL_KEYS:
        if key not in data:
            raise JudgeParseError(f"missing key {key!r}; got {sorted(data.keys())}")
        out[key] = _coerce_bool(data[key], key)
    return out
