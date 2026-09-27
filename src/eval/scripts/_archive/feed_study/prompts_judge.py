"""3rd-model judge prompts (llama-3.3-70b-versatile): claim-set comparison + quality.

Two independent jobs, both structured JSON, run by a DIFFERENT model family than
the extractors (gpt-oss, qwen) to avoid self-evaluation bias:

1. CLAIM_COMPARE — given a post + the two models' claim sets, how much do they
   agree (alignment, uniques, overall agreement, which is more complete)?
2. EXTRACTION_QUALITY — given a post + one model's claims, score each claim on
   faithfulness / decontextualization / atomicity, the set on coverage, and an
   overall good/borderline/bad flag (we then eyeball the flagged ones).

Quality dims ground in Claimify (Entailment/Decontextualization/Coverage,
arXiv:2502.10855) + molecular-facts atomicity (arXiv:2406.20079).
"""

from __future__ import annotations

import json
import re

Messages = list[dict[str, str]]

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _clean(raw: str) -> str:
    t = raw.strip()
    t = _FENCE_OPEN.sub("", t)
    t = _FENCE_CLOSE.sub("", t)
    return t.strip()


class JudgeParseError(ValueError):
    """Raised when a judge response cannot be coerced to the expected schema."""


def _load_obj(raw: str) -> dict:
    t = _clean(raw)
    if not t:
        raise JudgeParseError("empty response")
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = _JSON_OBJECT.search(t)
        if not m:
            raise JudgeParseError(f"no JSON object: {t[:200]!r}")
        return json.loads(m.group(0))


def _claims_block(claims: list[str]) -> str:
    return "\n".join(f"{i}. {c}" for i, c in enumerate(claims))


# ===========================================================================
# 1. CLAIM-SET COMPARISON
# ===========================================================================

CLAIM_COMPARE_SYSTEM = """You compare two sets of factual claims that two different systems extracted from the SAME social-media post. Judge how much they agree on what the post factually asserts.

You receive the POST, then claim list A and claim list B (each numbered from 0).

Do:
- ALIGN: find pairs (one claim from A, one from B) that assert the SAME fact — match on meaning, ignore wording.
- UNIQUES: list A-indices with no semantic match in B, and B-indices with no match in A.
- AGREEMENT: "high" = the two sets capture essentially the same facts; "medium" = substantial overlap but each misses some; "low" = largely different.
- MORE_COMPLETE: which set better covers the post's checkable content — "A", "B", or "equal".

Output strictly valid JSON, nothing else:
{"aligned_pairs": [[a_index, b_index], ...], "only_a": [int, ...], "only_b": [int, ...], "agreement": "high|medium|low", "more_complete": "A|B|equal", "notes": "<one short sentence>"}

Do not wrap in markdown fences."""


def build_compare_messages(post: str, claims_a: list[str], claims_b: list[str]) -> Messages:
    user = (f"POST:\n{post}\n\nCLAIM LIST A:\n{_claims_block(claims_a)}\n\n"
            f"CLAIM LIST B:\n{_claims_block(claims_b)}")
    return [
        {"role": "system", "content": CLAIM_COMPARE_SYSTEM},
        {"role": "user", "content": user},
    ]


_AGREE = {"high", "medium", "low"}
_MORE = {"A", "B", "equal"}


def parse_compare(raw: str) -> dict:
    d = _load_obj(raw)
    agreement = str(d.get("agreement", "")).strip().lower()
    if agreement not in _AGREE:
        agreement = "low"
    more = str(d.get("more_complete", "")).strip()
    more = more if more in _MORE else "equal"

    def _int_list(v):
        if not isinstance(v, list):
            return []
        out = []
        for x in v:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                pass
        return out

    pairs = []
    for p in (d.get("aligned_pairs") or []):
        if isinstance(p, (list, tuple)) and len(p) == 2:
            try:
                pairs.append([int(p[0]), int(p[1])])
            except (TypeError, ValueError):
                pass
    return {
        "aligned_pairs": pairs,
        "only_a": _int_list(d.get("only_a")),
        "only_b": _int_list(d.get("only_b")),
        "agreement": agreement,
        "more_complete": more,
        "notes": str(d.get("notes", "")).strip(),
    }


# ===========================================================================
# 2. EXTRACTION QUALITY
# ===========================================================================

EXTRACTION_QUALITY_SYSTEM = """You audit the QUALITY of factual claims a system extracted and normalized from a social-media post. You receive the POST and a numbered list of CLAIMS (from 0).

Score EACH claim 1-5:
- faithful: faithful to the post — entailed by it, with no added, removed, or distorted facts. 5 = fully entailed; 1 = fabricates or distorts.
- decontextualized: understandable and checkable on its own — pronouns/entities/dates resolved, no dependence on the post. 5 = fully standalone; 1 = unintelligible without the post.
- atomicity: a single, appropriately-sized verifiable claim — neither several facts crammed together nor pointlessly fragmented. 5 = ideal single checkable unit; 1 = badly over- or under-split.

Then for the WHOLE SET:
- coverage (1-5): do the claims capture the post's verifiable content without omitting important checkable facts? 5 = complete; 1 = major omissions.
- flag: "good" | "borderline" | "bad" — overall extraction quality.
- reason: one short sentence (required if borderline/bad).

Output strictly valid JSON, nothing else:
{"claims": [{"index": 0, "faithful": n, "decontextualized": n, "atomicity": n}, ...], "coverage": n, "flag": "good|borderline|bad", "reason": "<short>"}

Score every claim by its index. Do not wrap in markdown fences."""


def build_quality_messages(post: str, claims: list[str]) -> Messages:
    user = f"POST:\n{post}\n\nCLAIMS:\n{_claims_block(claims)}"
    return [
        {"role": "system", "content": EXTRACTION_QUALITY_SYSTEM},
        {"role": "user", "content": user},
    ]


_FLAG = {"good", "borderline", "bad"}


def _clamp15(v):
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    return i if 1 <= i <= 5 else None


def parse_quality(raw: str) -> dict:
    """-> {claim_scores: [{index,faithful,decontextualized,atomicity}], coverage, flag, reason}."""
    d = _load_obj(raw)
    scores = []
    for c in (d.get("claims") or []):
        if not isinstance(c, dict):
            continue
        try:
            idx = int(c.get("index"))
        except (TypeError, ValueError):
            continue
        scores.append({
            "index": idx,
            "faithful": _clamp15(c.get("faithful")),
            "decontextualized": _clamp15(c.get("decontextualized")),
            "atomicity": _clamp15(c.get("atomicity")),
        })
    flag = str(d.get("flag", "")).strip().lower()
    if flag not in _FLAG:
        flag = "borderline"
    return {
        "claim_scores": scores,
        "coverage": _clamp15(d.get("coverage")),
        "flag": flag,
        "reason": str(d.get("reason", "")).strip(),
    }
