"""Map publisher-specific rating strings to the project's 4-class scheme.

Targets `config.VERDICT_OPTIONS`:
  Supported · Refuted · Not Enough Evidence · Conflicting Evidence

Rule-based maps cover 7 of the 8 curated publishers. Full Fact uses
free-text ratings and is handled by an LLM call against Groq.
"""
from __future__ import annotations

import json
import re

import requests

from config import (
    EXTRACTION_API_KEY as DEEPINFRA_API_KEY,
    EXTRACTION_BASE_URL as DEEPINFRA_BASE_URL,
    VERDICT_OPTIONS,
)

SUPPORTED, REFUTED, NEI, CE = VERDICT_OPTIONS  # for readability

# Free-text-rating LLM fallback model (llm_map / veracity_llm). MUST differ from the verifier
# (config.VERIFICATION_MODEL = DeepSeek-V4-Flash) — the gold must not be assigned by the system under
# test (non-circularity, verdict_eval_plan.md §2). All on DeepInfra (api.deepinfra.com).
VERACITY_LLM_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

# Per-publisher mappings. Keys are lowercased stripped rating strings.
# Anything not in the map falls through to None → handled by caller.
RULES: dict[str, dict[str, str]] = {
    "politifact.com": {
        "true": SUPPORTED,
        "mostly true": SUPPORTED,
        "half true": CE,
        "mostly false": REFUTED,
        "false": REFUTED,
        "pants on fire": REFUTED,
        "pants on fire!": REFUTED,
    },
    "snopes.com": {
        "true": SUPPORTED,
        "mostly true": SUPPORTED,
        "mixture": CE,
        "mostly false": REFUTED,
        "false": REFUTED,
        "unproven": NEI,
        "outdated": NEI,
        "fake": REFUTED,
        "correct attribution": SUPPORTED,
        "incorrect attribution": REFUTED,
        "miscaptioned": REFUTED,
        "originated as satire": REFUTED,
        "labeled satire": REFUTED,
        "scam": REFUTED,
        "legit": SUPPORTED,
    },
    "factcheck.afp.com": {
        "true": SUPPORTED,
        "false": REFUTED,
        "partly false": CE,
        "misleading": CE,
        "missing context": CE,
        "unsubstantiated": NEI,
        "satire": REFUTED,
        "ai-generated": REFUTED,
        "altered picture": REFUTED,
        "altered video": REFUTED,
    },
    "factuel.afp.com": {  # AFP's French vertical — same enum, French strings.
        "vrai": SUPPORTED,
        "faux": REFUTED,
        "partiellement faux": CE,
        "trompeur": CE,
        "manque de contexte": CE,
        "infondé": NEI,
        "satire": REFUTED,
        "image générée par ia": REFUTED,
        "vidéo manipulée": REFUTED,
        "photo manipulée": REFUTED,
    },
    "newschecker.in": {
        "false": REFUTED,
        "altered photo/video": REFUTED,
        "altered media": REFUTED,
        "misleading": CE,
        "missing context": CE,
        "partly false": CE,
        "satire": REFUTED,
        "true": SUPPORTED,
    },
    "verafiles.org": {
        "fake": REFUTED,
        "false": REFUTED,
        "misleading": CE,
        "needs context": CE,
        "satire": REFUTED,
    },
    "rumorscanner.com": {
        "false": REFUTED,
        "misleading": CE,
        "ai-generated": REFUTED,
        "altered": REFUTED,
    },
    "factcheck.org": {
        # Looser scheme — most labels indicate falsity or misleadingness.
        "false": REFUTED,
        "misleading": CE,
        "unsupported": NEI,
        "no evidence": NEI,
        "exaggerated": CE,
        "exaggerates": CE,
        "distorts the facts": CE,
        "not the whole story": CE,
        "disputed": NEI,
        "outdated": NEI,
        "true": SUPPORTED,
    },
}


def rule_map(publisher_site: str, rating: str | None) -> str | None:
    """Return the harmonised label via lookup, or None if no rule applies."""
    if not rating:
        return None
    key = rating.strip().lower().rstrip(".")
    return RULES.get(publisher_site, {}).get(key)


def numeric_map(rating_value, best=5, worst=1) -> str | None:
    """Harmonise a numeric ClaimReview `reviewRating` to the 4-class scheme.

    Some publishers (Lead Stories, 20 Minutes, Science Feedback) carry the verdict in
    `ratingValue` on a worst..best scale while `alternateName` is a free-text/topical tag.
    Maps by normalised position so inverted scales (worst>best) work too. None if unparseable.
    """
    try:
        v, hi, lo = float(rating_value), float(best), float(worst)
    except (TypeError, ValueError):
        return None
    if hi == lo:
        return None
    frac = (v - lo) / (hi - lo)  # 0 = worst (false) … 1 = best (true)
    if frac < 0.4:
        return REFUTED
    if frac <= 0.6:
        return CE
    return SUPPORTED


_NEI_PAT = re.compile(
    r"\bno (?:evidence|proof|record|reports?|sign|trace|basis)\b"
    r"|unverified|unproven|unsubstantiated|baseless|no confirmation", re.I)
_CE_PAT = re.compile(
    r"nuanced|close,? but|partly|partially|mixture|misleading"
    r"|missing context|out of context|needs context|half[- ]?true", re.I)


def refine_numeric(textual, rating_value, best=5, worst=1) -> str | None:
    """Numeric ClaimReview verdict, refined by the textual tag where it carries verdict nuance.
    Lead Stories' `ratingValue=1` conflates *false* with *no-evidence* / *nuanced*; its
    `alternateName` ("No Evidence", "Unverified", "Nuanced", …) disambiguates → NEI / CE."""
    if textual:
        if _NEI_PAT.search(textual):
            return NEI
        if _CE_PAT.search(textual):
            return CE
    return numeric_map(rating_value, best, worst)


# --- LLM-based harmonisation (Full Fact + any fall-through) -------------

_VERDICT_SET = set(VERDICT_OPTIONS)
_VERDICT_LIST = " | ".join(f'"{v}"' for v in VERDICT_OPTIONS)

LLM_SYSTEM = (
    "You are classifying fact-check verdicts into a 4-class scheme used by an "
    "academic fact-checking evaluation pipeline. The scheme follows AVeriTeC: "
    "the class depends on the EVIDENCE state, not on the fact-checker's "
    "rhetorical framing.\n\n"
    f"The four classes are: {_VERDICT_LIST}.\n\n"
    "Definitions and decision rules:\n\n"
    "- Supported: there is evidence that the claim is true. "
    "Examples: 'True.', 'Correct.', 'This did happen.'\n\n"
    "- Refuted: there is evidence that the claim is false. The fact-checker "
    "found and cited something that directly contradicts the claim, OR the "
    "claim is about an artifact (image / video / quote / document) that is "
    "demonstrably fabricated, altered, AI-generated, miscaptioned, or "
    "misattributed. "
    "Examples: 'False, the policy was actually X.', 'The video is doctored.', "
    "'The quote was never said.', 'This is AI-generated.'\n\n"
    "- Not Enough Evidence: the fact-checker searched and could NOT find "
    "evidence either supporting or refuting the claim. This is the correct "
    "label whenever the verdict's main finding is the ABSENCE of evidence — "
    "even if the fact-checker phrases it assertively. "
    "Examples: 'There is no evidence X happened.', 'We could find no record "
    "of this.', 'Unproven.', 'Unsubstantiated.', 'It is not possible to "
    "verify this.', 'No credible source confirms this.'\n"
    "  Key distinction: if the fact-checker found POSITIVE counter-evidence "
    "(an official statement, a record, a forensic analysis), it is Refuted. "
    "If the fact-checker only reports failing to find supporting evidence, "
    "it is Not Enough Evidence.\n\n"
    "- Conflicting Evidence: the claim is partly true / partly false, "
    "misleading by exaggeration, missing context, cherry-picking, or a "
    "mixture of true and false elements. The fact-checker reaches a "
    "qualified verdict, not a clean true/false. "
    "Examples: 'Mostly true but missing context.', 'Half true.', "
    "'The number is right but the cause is wrong.', 'Mixture.'\n\n"
    "You will be given a fact-checker's free-text verdict. Map it to exactly "
    "one of the four classes. Respond with JSON: "
    '{"label": "<one of the four exact strings>"}'
)


def llm_map(model: str, rating: str, timeout: int = 30) -> str:
    """LLM classification via DeepInfra REST. Returns one of VERDICT_OPTIONS."""
    r = requests.post(
        f"{DEEPINFRA_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": f"Fact-checker verdict:\n{rating}"},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"] or ""
    obj = json.loads(text)
    label = obj.get("label", "").strip()
    if label not in _VERDICT_SET:
        for v in VERDICT_OPTIONS:
            if v.lower() in label.lower():
                return v
        raise ValueError(f"LLM returned non-canonical label: {label!r} for rating {rating!r}")
    return label


# --- Veracity 1–5 + rating_subtype (WS0 of verdict_eval_plan.md) -----------
# A FINER gold than the 4-class: keeps the 5-vs-4 gradation and splits the "3" into
# `mixed` (contested, evidence both ways) vs `unprovable` (no evidence either way) — the gold-3
# split the confidence-dim construct-validity test (WS4 H1/H2) rests on. Re-harmonised from the
# RAW `original_rating` (+ numeric `rating_value` for LeadStories/20 Minutes), NOT `harmonised_label`
# (which already collapsed True+Mostly-True). veracity=None ⇒ exclude from the veracity gold
# (satire by default / genuinely unrated). `is_satire` is set separately (enrich/Pass B).
RATING_SUBTYPES = ("clear_true", "mostly_true", "mixed", "unprovable",
                   "mostly_false", "clear_false", "altered_media", "satire", "unrated")

_SAT = re.compile(r"\b(satire|parody)\b", re.I)
_ALT = re.compile(r"ai[ -]?gener|artificial intelligence|miscaption|\baltered\b|doctored|manipul"
                  r"|deepfake|g[ée]n[ée]r[ée]e par ia|fabricat|is\W*n.?t real|not real and was made", re.I)
_MT = re.compile(r"\bmostly true\b|plut[oô]t vrai", re.I)
_MF = re.compile(r"\bmostly false\b|plut[oô]t faux", re.I)
_UNRATED = re.compile(r"research in progress|what we know|viet spam|^developing|^investigating", re.I)
_NEI_V = re.compile(r"no (?:evidence|proof|record|reports?|sign|trace|basis|confirmation)|unproven"
                    r"|unsubstantiated|unsupported|unverified|unfounded|baseless|\bdisputed\b"
                    r"|\boutdated\b|infond", re.I)
_CE_V = re.compile(r"misleading|partly|partial|half[ -]?true|mixture|missing context|needs? context"
                   r"|out of context|close,? but|manque de contexte|exaggerat|distort|not the whole story"
                   r"|nuanced|trompeur|partiellement|en partie|bait\s*&\s*switch", re.I)
_CT = re.compile(r"^(?:true|vrai|correct|legit|real|accurate)\b|\bcorrect attribution", re.I)
# Qualified-true: a leading true-family word followed by a hedge ("Accurate, but needs
# caution", "ACCURATE WITH CONSIDERATION", "Accurate; however ...") is NOT a clean 5 —
# route to the claim-aware LLM instead of guessing (fc-gold v2 audit, 2026-07-23).
_QUAL_TRUE = re.compile(r"^(?:true|vrai|correct|legit|real|accurate)\b.{0,80}?"
                        r"\b(?:but|however|although|though|caution|consideration|caveat)\b"
                        r"|^accurate with", re.I)
_CF = re.compile(r"\bfalse\b|\bfaux\b|\bfake\b|scam|pants on fire|did\W*n.?t happen|made[ -]?up"
                 r"|no such|not genuine|hoax|incorrect attribution|misattribut|debunked|^legend$", re.I)


def _veracity_numeric(rating_value, best=5, worst=1) -> tuple[int | None, str]:
    try:
        v, hi, lo = float(rating_value), float(best), float(worst)
    except (TypeError, ValueError):
        return (None, "")
    if hi == lo:
        return (None, "")
    frac = (v - lo) / (hi - lo)  # 0 = worst (false) … 1 = best (true)
    if frac < 0.2:
        return (1, "clear_false")
    if frac < 0.4:
        return (2, "mostly_false")
    if frac <= 0.6:
        return (3, "mixed")
    if frac < 0.8:
        return (4, "mostly_true")
    return (5, "clear_true")


def harmonise_veracity(rating, rating_value=None, best=5, worst=1) -> tuple[int | None, str]:
    """Raw rating (+ numeric value) → (veracity 1–5 | None, rating_subtype). Ordered so the
    finer/earlier tests win: unrated → satire → altered media → mostly-X → no-evidence → mixed →
    clear. Text first (the `alternateName` disambiguates LeadStories' all-1 `ratingValue`), numeric
    fallback, else (None, "") → caller's LLM fallback. veracity None ⇒ exclude from gold."""
    t = (rating or "").strip()
    has_num = rating_value is not None
    if t:
        if _UNRATED.search(t):                    return (None, "unrated")
        if _QUAL_TRUE.search(t):                  return (None, "")            # hedged true → LLM
        if _SAT.search(t):                        return (1, "satire")          # false content, flagged satire
        if _ALT.search(t):                        return (1, "altered_media")
        if not has_num and re.match(r"(?i)^(?:false|faux)\b", t): return (1, "clear_false")  # explicit leading
        if not has_num and re.match(r"(?i)^(?:true|vrai)\b", t):  return (5, "clear_true")   # verdict wins (text pubs)
        if _MT.search(t):                         return (4, "mostly_true")
        if _MF.search(t):                         return (2, "mostly_false")
        if _NEI_V.search(t):                      return (3, "unprovable")
        if _CE_V.search(t):                       return (3, "mixed")
        if not has_num:   # numeric publishers (LeadStories/20Min): a loose text tag must NOT override ratingValue
            if _CT.search(t):                     return (5, "clear_true")
            if _CF.search(t):                     return (1, "clear_false")
    if has_num:
        return _veracity_numeric(rating_value, best, worst)
    return (None, "")


_SUBTYPE_V = {"clear_true": 5, "mostly_true": 4, "mixed": 3, "unprovable": 3,
              "mostly_false": 2, "clear_false": 1, "altered_media": 1, "satire": 1}

VERACITY_LLM_SYSTEM = (
    "You translate a fact-checker's free-text verdict into a 1–5 veracity scale + a subtype FOR THE "
    "CLAIM, for an academic eval. You are given the CLAIM and the fact-checker's VERDICT prose. Rate how "
    "true the CLAIM is, judging from the EVIDENCE STATE the verdict describes, NOT rhetorical framing. Do "
    "not re-investigate.\n\n"
    "CRITICAL — polarity: the verdict prose usually states the CORRECT facts, which are often the OPPOSITE "
    "of the claim (e.g. claim 'Vaccines are poison' + verdict 'Vaccines are rigorously tested for safety' "
    "⇒ the claim is clearly FALSE, veracity 1). A verdict that affirms facts CONTRADICTING the claim means "
    "the claim is false; only score high when the verdict confirms the CLAIM ITSELF is true. Words like "
    "'actually', \"isn't right\", 'inaccurate', 'no evidence' signal the claim is wrong.\n\n"
    "veracity: 5 clearly-true · 4 mostly-true · 3 not-clearly-either · 2 mostly-false · 1 clearly-false.\n"
    "subtype (pick one): clear_true | mostly_true | mixed | unprovable | mostly_false | clear_false | "
    "altered_media | satire | not_a_verdict.\n"
    "- mixed (veracity 3): contested / half-true / missing-context / exaggerated — evidence points BOTH ways.\n"
    "- unprovable (veracity 3): the verdict's finding is the ABSENCE of evidence (unproven, no record). NOT "
    "for fabrication.\n"
    "- altered_media (veracity 1): the verdict is that an image/video/photo is AI-generated, altered, "
    "doctored, or miscaptioned.\n"
    "- satire (veracity 1): originated as / labelled satire or parody.\n"
    "- not_a_verdict (veracity null): the text delivers NO rating on the claim — it is a topic label, "
    "a status note, a question, a bare headline, or otherwise says nothing about whether the claim "
    "holds. Use null (JSON null) for veracity in this case, and ONLY this case.\n"
    'Respond JSON only: {"veracity": 1-5 or null, "subtype": "<one of the above>"}'
)


# --- judged_axis rule-derivation (Daniel 280626) -------------------------
# judged_axis = the axis the FACT-CHECKER chose to adjudicate; it's a property of the rating (+ claim),
# NOT the post. The RATING is the tell: explicit attribution/media ratings map deterministically; generic
# ratings (True/False/Mixture) → None → leave to Pass B's sighted LLM (rating + claim + image), which
# disambiguates content-vs-artifact. Removes the content↔attribution confusion the smoke surfaced.
_AX_ATTRIB = re.compile(r"\battribut|misattribut|misquot|never said|did\W*n.?t say|fake quote"
                        r"|quote[^.]{0,24}(?:fake|fabricat|never)", re.I)
_AX_ARTIFACT = re.compile(r"miscaption|\baltered\b|ai[ -]?gener|deepfake|doctored|digitally"
                          r"|g[ée]n[ée]r[ée]e par ia|manipul|fabricat(?:ed)?\s*(?:photo|image|video)"
                          r"|(?:photo|image|video|footage|clip)[^.]{0,18}(?:is(?:n.?t)? real|fake|altered)", re.I)


def judged_axis_from_rating(rating: str | None) -> str | None:
    """Rating → {attribution, artifact} for the EXPLICIT cases; None (→ Pass B LLM) otherwise.
    None means "generic verdict, axis is content unless the sighted model + image says artifact"."""
    t = rating or ""
    if _AX_ATTRIB.search(t):
        return "attribution"
    if _AX_ARTIFACT.search(t):
        return "artifact"
    return None


def veracity_llm(rating: str, claim: str | None = None, model: str = VERACITY_LLM_MODEL,
                 timeout: int = 30) -> tuple[int, str]:
    """LLM fallback for free-text ratings the rules miss (e.g. Full Fact) → (veracity, subtype).

    The verdict prose alone is polarity-ambiguous (it states the TRUE facts, often the claim's
    negation), so the CLAIM is passed and the model rates the CLAIM given the verdict. `model`
    defaults to an independent model (≠ the verifier) for non-circularity.
    """
    user = (f"CLAIM:\n{claim}\n\nFact-checker VERDICT:\n{rating}" if claim
            else f"Fact-checker verdict:\n{rating}")
    r = requests.post(
        f"{DEEPINFRA_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {DEEPINFRA_API_KEY}"},
        json={"model": model, "messages": [
            {"role": "system", "content": VERACITY_LLM_SYSTEM},
            {"role": "user", "content": user}],
            "temperature": 0,
            "response_format": {"type": "json_object"}},
        timeout=timeout,
    )
    r.raise_for_status()
    obj = json.loads(r.json()["choices"][0]["message"]["content"] or "{}")
    sub = str(obj.get("subtype", "")).strip().lower()
    if sub not in _SUBTYPE_V:
        sub = "clear_false"
    try:
        ver = int(obj.get("veracity") or _SUBTYPE_V[sub])
    except (TypeError, ValueError):
        ver = _SUBTYPE_V[sub]
    return (max(1, min(5, ver)), sub)
