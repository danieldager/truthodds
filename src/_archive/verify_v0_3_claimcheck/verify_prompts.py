"""Prompts + message builders + parsers for the Tier-3 verifier.

ClaimCheck-faithful (idirlab/claimcheck): the planning / summarise / synthesise prompt
bodies follow ClaimCheck's `plan` / `summarize` / `develop` closely, deviating only to
emit structured JSON (instead of regex-parsed free-text "action lines") and to drop the
multimodal actions — we are text-only with one tool (web search). The 4-class evaluator is
ClaimCheck's judge prompt verbatim; the Likert evaluator is ours.

Each LLM step has a `build_*_messages` (-> OpenAI chat messages) and a `parse_*` (-> dict
or str). Parsers strip reasoning `<think>` blocks and code fences before JSON-decoding.
"""
from __future__ import annotations

import json
import re

FOURCLASS_LABELS = (
    "Supported",
    "Refuted",
    "Conflicting Evidence/Cherrypicking",
    "Not Enough Evidence",
)


class VerifyParseError(Exception):
    """Raised when an LLM step's output can't be parsed into the expected shape."""


# =============================================================================
# Shared parsing helpers
# =============================================================================

def _strip(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _load_obj(text: str) -> dict:
    """Parse the outermost JSON object from a (possibly prose-wrapped) response."""
    cleaned = _strip(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise VerifyParseError(f"no JSON object found in: {text[:200]!r}")
        return json.loads(m.group(0))


def _clamp_1_5(v) -> int:
    return max(1, min(5, int(v)))


def _img_block(image_context: str | None) -> str:
    """A labeled block for a post-image serialization, fed as CLAIM CONTEXT (not evidence).

    The verifier is text-only; for claims that live in or depend on an image, this is how it
    'sees' the picture. Flagged as context (not a retrieved source) so the model does not treat
    the description itself as corroborating evidence.
    """
    if not image_context or not image_context.strip():
        return ""
    return ("\n\nImage content (transcribed from the post's attached image — part of the claim's "
            "context, NOT retrieved evidence):\n" + image_context.strip())


# =============================================================================
# 1 — Planning  ->  {"query": str}
# =============================================================================

PLAN_SYSTEM = """You are a fact-checker. The available knowledge is insufficient to assess the Claim, so you must retrieve evidence from the web. Produce ONE web search query that would best surface evidence that could confirm or refute the Claim.

- Use the specific named entities, dates, numbers, and exact phrases from the Claim.
- Prefer the terms a journalist or institution would use, not informal phrasing.
- Plain text only — no search operators (site:, quotes, AND/OR).

Output strictly valid JSON and nothing else: {"query": "<the search query>"}"""


def build_plan_messages(claim_text: str, image_context: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": (
            f"Claim: {claim_text}{_img_block(image_context)}\n\nProduce the opening search query."
        )},
    ]


def parse_plan(raw: str, fallback: str) -> str:
    """Return the query string; fall back to the claim text if parsing fails."""
    try:
        q = str(_load_obj(raw).get("query", "")).strip()
    except VerifyParseError:
        q = ""
    return q or fallback


# =============================================================================
# 2 — Summarise (per scraped doc)  ->  {relevant, publication_date, summary, quotes}
# =============================================================================

SUMMARISE_SYSTEM = """You just ran a web search to find evidence for a fact-check. Summarize the Search Result concisely, including ONLY information relevant to the Claim.

Include:
- Information useful for fact-checking the Claim.
- If available, the publication date, and the author or publisher.
- Direct quotes from the Search Result that bear on the Claim.

Do NOT include advertisements or anything unrelated to the Claim. Do NOT add information not present in the Search Result; do not use outside knowledge. Do not take a position on whether the Claim is true — only state what the Search Result says.

If the Search Result contains no information relevant to the Claim, set relevant=false.

Output strictly valid JSON and nothing else:
{"relevant": <true|false>, "publication_date": "YYYY-MM-DD" or null, "summary": "<at most 5 sentences, Claim-relevant content only; empty string if relevant=false>", "quotes": ["<verbatim quote bearing on the Claim>", ...]}"""


def build_summarise_messages(claim_text: str, url: str, content: str, truncate: int) -> list[dict]:
    return [
        {"role": "system", "content": SUMMARISE_SYSTEM},
        {"role": "user", "content": (
            f"Claim: {claim_text}\n\n"
            f"Search Result:\n{url}\n{content[:truncate]}"
        )},
    ]


def parse_summarise(raw: str) -> dict:
    data = _load_obj(raw)
    quotes = data.get("quotes") or []
    if not isinstance(quotes, list):
        quotes = []
    return {
        "relevant": bool(data.get("relevant", False)),
        "publication_date": (data.get("publication_date") or None),
        "summary": str(data.get("summary", "")).strip(),
        "quotes": [str(q).strip() for q in quotes if str(q).strip()],
    }


# =============================================================================
# 3 — Synthesise  ->  {"analysis": str, "next_query": str | None}
# =============================================================================

SYNTHESISE_SYSTEM = """You have gathered Evidence about a Claim through one or more rounds of web search. Analyze the Claim's veracity using ONLY the recorded Evidence.

- Focus on new insights; do not restate the Claim or repeat the Evidence verbatim.
- Pin down the Claim's EXACT load-bearing proposition before analyzing — does it hinge on a specific image/video/quote being authentic, a causal link, a precise quantity, a scope word ("all", "first", "every"), or a timeframe? Assess THAT proposition, and state explicitly when the evidence only confirms an adjacent or weaker fact (e.g. the event happened) rather than the claim itself (e.g. that this footage depicts it).
- Flag misleading framing: if the evidence supports the literal core but the Claim overstates scope, omits decisive context, uses a loaded characterization, or presents a stale/narrow fact as broad or current, say so.
- One to three paragraphs — the fewer, the better.
- Note agreements, contradictions, gaps, and the reliability of the sources.
- If there is insufficient information to verify the Claim, explicitly state what specific information is missing.
- You may use commonsense knowledge, but do not introduce facts not implied by the Evidence.

Then decide whether to conclude or search again. You retrieve in TWO PHASES: a keyword engine (Serper) first, then a neural/semantic engine (Exa) that surfaces harder-to-find primary and specialist sources the keyword engine misses — so if you lack good sources, requesting another search is worthwhile because it can escalate to Exa.

CORROBORATION RULE: never conclude (next_query=null) from a SINGLE source — always request another search. With only two sources, still prefer one more search UNLESS both are authoritative and directly settle the claim. Conclude only when multiple credible sources corroborate, or when further targeted search is genuinely futile.

Output strictly valid JSON and nothing else:
{"analysis": "<your 1-3 paragraph analysis>", "next_query": "<a NEW search query targeting the missing information, or null if the evidence is sufficient>"}

The next_query, if any, must target a specific gap and must NOT repeat or paraphrase a previous query."""


_NUDGE = ("\n\nNOTE: your previous proposed query was too similar to one already tried — it "
          "would return the same results. Propose a SUBSTANTIVELY different angle (different "
          "entities, a different aspect of the claim, a different timeframe, or a different "
          "source type), OR set next_query to null if you have genuinely exhausted useful angles.")

# Appended when the next search will run on Exa (neural search) after the keyword engine
# (Serper) was exhausted. Relaxes the "do not repeat" rule: a prior query can be productive
# on a different engine, so reuse/combination is explicitly allowed alongside a fresh query.
_ESCALATE_EXA = ("\n\nNOTE: your next search will run on a DIFFERENT engine — Exa, which searches "
                 "by MEANING (neural/semantic), not keywords, and rewards a long, descriptive, "
                 "content-rich query rather than a short keyword phrase. This is your FINAL search, so "
                 "make next_query the STRONGEST POSSIBLE verification query: in ONE self-contained query, "
                 "(1) restate the claim's core proposition with its key entities, dates, numbers and any "
                 "quotes; (2) fold in the distinct angles you already tried in the previous queries; and "
                 "(3) add any new sub-questions still needed to confirm or refute the claim. The "
                 "'do not repeat' rule is SUSPENDED — compose and combine the previous queries freely. Do "
                 "NOT write a short keyword query; write the richest, most complete query you can that "
                 "would surface decisive confirming or refuting evidence.")


def build_synthesise_messages(
    claim_text: str, evidence_block: str, past_queries: list[str], nudge: bool = False,
    escalate_to: str | None = None, image_context: str | None = None,
) -> list[dict]:
    queries = "\n".join(f"{i}. {q}" for i, q in enumerate(past_queries, 1)) or "(none)"
    extra = (_NUDGE if nudge else "") + (_ESCALATE_EXA if escalate_to == "exa" else "")
    return [
        {"role": "system", "content": SYNTHESISE_SYSTEM},
        {"role": "user", "content": (
            f"Claim: {claim_text}{_img_block(image_context)}\n\n"
            f"Evidence:\n{evidence_block or '(none)'}\n\n"
            f"Previous queries (do not repeat):\n{queries}\n\n"
            f"Analyze the Claim and decide whether to search again."
            + extra
        )},
    ]


def parse_synthesise(raw: str) -> dict:
    data = _load_obj(raw)
    nq = data.get("next_query")
    nq = str(nq).strip() if nq else ""
    return {"analysis": str(data.get("analysis", "")).strip(), "next_query": nq or None}


# =============================================================================
# 4 — Likert evaluator (ours)  ->  {4 ints, justification}
# =============================================================================

LIKERT_SYSTEM = """You are assessing a CLAIM against a synthesized analysis of evidence — the analysis is your ONLY context; add no outside knowledge. Output one veracity score (is the claim, exactly as stated, true?) and three confidence scores (how much to trust that read). Score each 1-5 independently against its own definition.

**Use the full scale, including 2 and 4 deliberately** — partial, leaning, or qualified cases belong at 2 or 4; do not collapse every judgement to 1, 3, or 5.

First fix the claim's EXACT load-bearing proposition and score THAT — not a weaker or adjacent version. Veracity is the claim's truth, never merely how strong the evidence is:
- POLARITY: if the claim asserts something did NOT happen, or is fake / staged / manipulated / a hoax, and the evidence shows the opposite, the CLAIM is FALSE (1-2). Never score a strong refutation as "true".
- PROPOSITION FIDELITY: confirming a related fact does NOT confirm the claim. If the claim is that a specific image/video/quote authentically shows X, the underlying event being real does not make the media genuine. If the claim asserts causation ("X because of Y"), confirming X and Y separately does not confirm the link. If it states a precise figure, a scope word ("all", "first", "every"), or a timeframe, a looser version being true does not make the claim true.
- MISLEADINGNESS: a claim whose literal core is supported is still only HALF TRUE (3) if it materially overstates scope, uses a loaded or exaggerated characterization, omits decisive context, or presents a stale or narrow fact as broad or current — i.e. a reader is left with a false impression.

veracity — is the claim, exactly as stated, true?
  5: True — the claim as stated is confirmed, including its load-bearing specifics, with no material caveat and no misleading framing.
  4: Mostly true — the main assertion is confirmed and not misleading; a minor specific may be approximate, slightly off, contradicted (e.g. a date off by a day, a rounded figure), or merely unconfirmed — a wrong, off, or missing MINOR detail on an otherwise-accurate claim is 4, not 2.
  3: Half-true, mixture, or indeterminate — EITHER the literal core is supported but the claim materially overstates, decontextualizes, misattributes, or misframes it; OR it asserts a specific proposition (media authenticity, causation, an exact figure) the evidence does not actually establish though it confirms something adjacent; OR the evidence genuinely neither confirms nor contradicts the main assertion.
  2: Mostly false — the main assertion is contradicted, though a detail may be unclear.
  1: Clearly false — the evidence contradicts the claim as stated (a false negation, fabricated media, or invented causation included).

evidence_sufficiency — how directly and fully does the evidence address the claim?
  5: The evidence speaks directly to the main assertion AND its load-bearing specifics.
  4: The evidence directly addresses the main assertion, but leaves a specific (a number, date, qualifier) untouched.
  3: Partial — it touches the topic but addresses the specific assertion only loosely, or through a single piece of evidence.
  2: Only topically adjacent — it does not speak to the specific assertion.
  1: Nothing in the analysis bears on the claim's substance.

evidence_agreement — do the pieces of evidence point the same way?
  5: All pieces point the same way (toward true or toward false); no contradictions.
  4: Broad convergence; only minor differences of scope or wording.
  3: Some tension between pieces — or only one piece exists, so agreement cannot be judged.
  2: Notable conflict — at least one piece points each way.
  1: The evidence flatly contradicts itself on a load-bearing point.

source_reliability — how trustworthy are the cited sources, regardless of what they say?
  5: Primary or authoritative — official records, peer-reviewed work, named domain experts, court/legislative documents.
  4: Established mainstream outlets or institutional reporting.
  3: Mixed — some reputable, but leaning on secondary or lower-tier reporting.
  2: Mostly blogs, opinion, or low-traffic sites.
  1: Only fringe, anonymous, or unidentifiable sources — or none usable.

Output strict JSON, no prose outside the JSON:
{
  "veracity": <1-5>,
  "evidence_sufficiency": <1-5>,
  "evidence_agreement": <1-5>,
  "source_reliability": <1-5>,
  "justification": "<one or two sentences citing what in the analysis drives the scores>"
}"""


def build_likert_messages(claim_text: str, analysis: str, image_context: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": LIKERT_SYSTEM},
        {"role": "user", "content": (
            f"Claim: {claim_text}{_img_block(image_context)}\n\n"
            f"Synthesized analysis:\n{analysis}\n\nAssign the four scores."
        )},
    ]


def parse_likert(raw: str) -> dict:
    data = _load_obj(raw)
    return {
        "veracity": _clamp_1_5(data["veracity"]),
        "evidence_sufficiency": _clamp_1_5(data["evidence_sufficiency"]),
        "evidence_agreement": _clamp_1_5(data["evidence_agreement"]),
        "source_reliability": _clamp_1_5(data["source_reliability"]),
        "justification": str(data.get("justification", "")).strip(),
    }


# =============================================================================
# 5 — 4-class evaluator (ClaimCheck verbatim)  ->  {verdict, justification}
# =============================================================================

FOURCLASS_SYSTEM = """Determine the Claim's veracity from the synthesized analysis, following these steps:

1. Briefly summarize the key insights from the fact-check in at most one paragraph.
2. Write one paragraph about which of the Decision Options applies best, and emit the chosen option at the end.

Decision Options:
Supported | Refuted | Conflicting Evidence/Cherrypicking | Not Enough Evidence

Rules:

Supported - The claim is directly and clearly backed by strong, credible evidence. Minor uncertainty or lack of detail does not disqualify a claim from being Supported if the main point is well-evidenced.
- Use Supported if the overall weight of evidence points to the claim being true, even if there are minor caveats or not every detail is confirmed.

Refuted - The claim is contradicted by strong, credible evidence — the search surfaced something that directly disproves the main point — or the claim is a fabricated, altered, deceptive, or misattributed artifact.
- Use Refuted when POSITIVE counter-evidence disproves the central elements, even if some minor details are unclear.

Conflicting Evidence/Cherrypicking - Only use this if there are reputable sources that directly and irreconcilably contradict each other about the main point of the claim, and no clear resolution is possible after careful analysis.
- Do NOT use this for minor disagreements, incomplete evidence, or if most evidence points one way but a few sources disagree.

Not Enough Evidence - The search found no evidence either supporting or refuting the claim: the main finding is the ABSENCE of supporting evidence. Use this when you cannot corroborate the claim and also did not surface anything that directly disproves it.
- Distinguish from Refuted: Refuted requires positive counter-evidence; Not Enough Evidence is the failure to find support. A claim with no credible support but no direct disproof is Not Enough Evidence — NOT Refuted.

Output JSON, no prose outside the JSON:
{
  "verdict": "<one of: Supported | Refuted | Conflicting Evidence/Cherrypicking | Not Enough Evidence>",
  "justification": "<the one-paragraph decision rationale from step 2>"
}"""


def build_fourclass_messages(claim_text: str, analysis: str, image_context: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": FOURCLASS_SYSTEM},
        {"role": "user", "content": (
            f"Claim: {claim_text}{_img_block(image_context)}\n\n"
            f"Synthesized analysis:\n{analysis}\n\nAssign one of the four decision options."
        )},
    ]


def parse_fourclass(raw: str) -> dict:
    data = _load_obj(raw)
    verdict = str(data.get("verdict", "")).strip()
    if verdict not in FOURCLASS_LABELS:
        low = verdict.lower()
        if "support" in low:
            verdict = "Supported"
        elif "refut" in low or "false" in low:
            verdict = "Refuted"
        elif "conflict" in low or "cherry" in low:
            verdict = "Conflicting Evidence/Cherrypicking"
        else:
            verdict = "Not Enough Evidence"
    return {"verdict": verdict, "justification": str(data.get("justification", "")).strip()}


# =============================================================================
# 6 — Post-level FLAG (sees the RAW POST alongside the atomic claim + evidence)
# =============================================================================
# Separate from the atomic claim verdict (which stays context-free + cacheable). This step
# judges the POST: a claim can be literally true yet used misleadingly. See design §3b.

FLAG_SYSTEM = """You decide whether a social-media POST should be flagged as potential misinformation, for a tool that warns users before they share misleading content.

You are given the POST, an atomic CLAIM drawn from it, and an evidence-based ANALYSIS of that claim's veracity. Decide whether the POST should be flagged.

Flag the post (flag=true) if a reader would be misled by it: the claim is false or unsupported, OR the claim is literally true but the POST frames it misleadingly — implying causation from mere sequence, omitting essential context, presenting a stale event as recent, or attributing it falsely. Pass the post (flag=false) only if it is accurate AND its framing is not misleading.

Judge the POST as written, using the evidence — do not rely on the claim's literal wording alone, and do not flag merely because a minor detail is uncertain. Output strict JSON: {"flag": <true|false>, "reason": "<one sentence grounded in the evidence and the post's framing>"}"""


def build_flag_messages(post: str, claim_text: str, analysis: str,
                        image_context: str | None = None) -> list[dict]:
    return [
        {"role": "system", "content": FLAG_SYSTEM},
        {"role": "user", "content": (
            f"POST:\n{post}{_img_block(image_context)}\n\n"
            f"ATOMIC CLAIM (extracted from the post):\n{claim_text}\n\n"
            f"EVIDENCE ANALYSIS of the claim:\n{analysis}\n\nShould this POST be flagged?"
        )},
    ]


def parse_flag(raw: str) -> dict:
    data = _load_obj(raw)
    return {"flag": bool(data.get("flag", False)), "reason": str(data.get("reason", "")).strip()}


# =============================================================================
# 7 — Misinformation evaluator (CONTEXT-AWARE, misinfo-native)  ->  {veracity, misinfo, type}
# =============================================================================
# For claims EXTRACTED FROM ARTICLES/POSTS: the primary question is misinformation.
# NOTE the anchor change: the validated Likert (0.915) anchors veracity to "is the claim, exactly
# as stated, TRUE?" (misleadingness only DEMOTES a true claim to 3). This prompt REFRAMES the anchor
# to "would a reader come away MISLED?" — it reuses the rubric scaffolding (polarity / proposition-
# fidelity / misleadingness / the 1-5 levels) but changes the lead question, so its number is NOT
# the validated 0.915 veracity and must be re-validated. The two anchors agree on clear true/false
# but can DIVERGE on not-enough-evidence and — the danger case — a TRUE ATTRIBUTION OF A FALSE CLAIM
# (accurate reporting a misinfo-anchor may wrongly flag; see CLAUDE.md nudge-safety open question).
# Emits ONE flag (the veracity<=3 cut) + a type. See docs/verdict_nudge_design.md.

MISINFO_EVAL_SYSTEM = """You assess a CLAIM for MISINFORMATION — the single question a tool must answer before warning someone against sharing false or misleading content. You are given the CLAIM, the CONTEXT it appears in (the originating post or article), and an evidence-based ANALYSIS. Decide whether a reader who took the claim at face value, as presented in its context, would come away misled.

Misinformation is not only outright falsehood. A claim is also misinformation when its literal words are supportable but its presentation deceives: it omits essential context, implies a causal link the evidence does not establish, inflates a narrow or stale fact into a broad or current one, or asserts a specific the evidence never confirms (an exact figure, a scope word like "all"/"first", or the authenticity of a specific image/video).

Fix the claim's EXACT load-bearing proposition, read in its context, and judge THAT — not a weaker or adjacent version:
- POLARITY: if the claim says something did NOT happen, or is fake / staged / a hoax, and the evidence shows the opposite, the claim is FALSE.
- PROPOSITION FIDELITY: confirming a related fact does not confirm the claim (a causal link, a scope word, an exact figure, or media authenticity must each be established in its own right).
- MISLEADING PRESENTATION: a literally-supportable core presented so as to leave a false impression is still misinformation.

Score veracity 1-5 — the misinformation scale, from dangerous falsehood to true-and-not-misleading:
  5 — True and not misleading: confirmed as stated, load-bearing specifics included, no deceptive framing.
  4 — Essentially accurate: the main assertion is confirmed and not misleading; at most a minor specific is approximate, slightly off, or merely unconfirmed.
  3 — Misleading or unverifiable: EITHER the literal core is supportable but the presentation materially deceives (missing context, implied-but-unproven link, inflated or decontextualised scope), OR the evidence genuinely neither confirms nor refutes the main assertion.
  2 — Mostly false: the main assertion is contradicted, though a detail may be unclear.
  1 — False: the evidence contradicts the claim as stated (a false negation, fabricated event, or invented causation included).

Then emit the misinformation verdict:
  misinfo = true when veracity <= 3 (a reader would be misled), false when veracity >= 4.
  misinfo_type — why it is (or is not) misinformation:
    FALSE — factually untrue (veracity 1-2).
    MISLEADING — literally supportable but deceptively presented (veracity 3, core true).
    UNSUPPORTED — the evidence cannot establish it either way (veracity 3, no support).
    NONE — not misinformation (veracity 4-5).

Use ONLY the analysis and the context; add no outside knowledge. Output strict JSON, no prose outside it:
{"veracity": <1-5>, "misinfo": <true|false>, "misinfo_type": "FALSE|MISLEADING|UNSUPPORTED|NONE", "justification": "<one or two sentences citing what in the analysis and the context drives this>"}"""


def build_misinfo_messages(claim_text: str, context: str | None, analysis: str,
                           image_context: str | None = None) -> list[dict]:
    ctx = (context or "").strip() or "(no surrounding context supplied — judge the claim as stated)"
    return [
        {"role": "system", "content": MISINFO_EVAL_SYSTEM},
        {"role": "user", "content": (
            f"CLAIM:\n{claim_text}{_img_block(image_context)}\n\n"
            f"CONTEXT (the post/article the claim appears in):\n{ctx}\n\n"
            f"EVIDENCE ANALYSIS:\n{analysis}\n\nAssess the claim for misinformation."
        )},
    ]


_MISINFO_TYPES = {"FALSE", "MISLEADING", "UNSUPPORTED", "NONE"}


def parse_misinfo(raw: str) -> dict:
    data = _load_obj(raw)
    v = _clamp_1_5(data["veracity"])
    misinfo = v <= 3  # the flag IS the validated nudge cut on the misinfo scale — not an independent judgement
    t = str(data.get("misinfo_type", "")).strip().upper()
    if v >= 4:
        t = "NONE"
    elif t not in _MISINFO_TYPES or t == "NONE":
        t = "FALSE" if v <= 2 else "MISLEADING"  # coerce to a misinfo type when veracity<=3
    return {"veracity": v, "misinfo": misinfo, "misinfo_type": t,
            "justification": str(data.get("justification", "")).strip()}


# =============================================================================
# 8 — Raw-text verification loop (EXPERIMENTAL): plan -> controller -> verdict
# =============================================================================
# Verifies RAW TEXT (headline/tweet/post, spin intact); claim decomposition is IMPLICIT — the model
# chooses what to check to decide misinformation. Attribution and media-authenticity are native
# considerations. Model-driven Serper/Exa (Serper forced first). See docs/verdict_nudge_design.md.

PLAN_TEXT_SYSTEM = """You are fact-checking a piece of RAW TEXT (a headline, tweet, or post — with whatever spin or framing it carries) to decide whether it is misinformation. Identify the ONE claim in it that most determines whether a reader would be misled, and write the single best web-search query to test that claim.

Choose what to check by what would make the text misinformation:
- If the load-bearing point is who said something (an attribution or a quote), test the attribution and the quote's real context.
- If it hinges on whether an image/video authentically shows what is claimed, target that.
- Otherwise target the central factual assertion.

Output strict JSON: {"main_claim": "<the claim you are testing, one sentence>", "query": "<search query>"}"""


CONTROLLER_SYSTEM = """You are running an evidence-gathering loop to decide whether a piece of RAW TEXT is misinformation. You see the full state so far: the raw text, every query and provider already used, all evidence gathered (search-result snippets, and full-read summaries for any you chose to read), and your own running analysis and open questions. Decide the single next action.

Two search providers are available:
- serper: broad Google index — fast; best for mainstream news, official records, well-covered events.
- exa: neural/semantic search — best when the evidence is niche, phrased unusually, or unlikely to surface via Google-style keyword search.

Choose ONE action:
- "VERDICT": you can already decide — EITHER the snippets/summaries settle it, OR no credible source exists for a central claim (the ABSENCE of any credible evidence for a load-bearing claim is itself grounds to flag the text as misinformation).
- "READ": some listed results look decisive but you must confirm before deciding — give their indices to fetch and read in full.
- "SEARCH": issue another query — a different angle for better evidence, or escalate to exa because what you need is unlikely to be in Google results. Give the query AND the provider.

Judge MISINFORMATION, not just literal truth:
- ATTRIBUTION: when the text quotes or attributes something, the question is whether the attribution is accurate AND whether the real surrounding context changes the quote's meaning — a genuine quote stripped of meaning-changing context is misleading. Accurate reporting that "X said Y" (Y false) is NOT itself misinformation unless the text presents Y as true.
- MEDIA: when truth hinges on whether an image/video is genuine or authentically shows the claimed thing, that is the claim to resolve.

Keep your analysis and open questions cumulative — carry forward what earlier steps established. VERDICT as soon as the evidence is decisive; keep searching only while a load-bearing question is genuinely open.

Discipline (avoid spinning):
- Prefer READ over re-searching: if a result already in hand could answer an open question, READ it before issuing another query.
- NEVER repeat a query you have already tried (they are listed under QUERIES ALREADY TRIED). The redundancy check is WORD-BASED: a follow-up that reuses most of the same words as a prior query — even reordered, or with just one word added — is treated as a duplicate, SKIPPED, and wasted. So if you genuinely need to search again, change the actual TERMS: use synonyms, a different entity or date, or a different sub-question — not a reworded version of the same query.
- If the load-bearing evidence simply isn't appearing after a couple of distinct queries, STOP — return VERDICT with type UNSUPPORTED. The absence of credible evidence for a central claim is itself a finding (grounds to flag), not a reason to keep searching.

Output strict JSON:
{"analysis": "<cumulative reading of the evidence so far>", "open_questions": ["<what still needs resolving>"], "action": "VERDICT|READ|SEARCH", "read_indices": [<result indices, only for READ>], "query": "<only for SEARCH>", "provider": "serper|exa (only for SEARCH)"}"""


VERDICT_TEXT_SYSTEM = """You are issuing a final misinformation verdict on a piece of RAW TEXT, using ONLY the gathered evidence and analysis — add no outside knowledge. Judge whether a reader who took the text at face value, in its context, would be misled.

CRITICAL — distinguish MISINFORMATION from mere bias, slant, opinion, or clumsy wording: only the first lowers the score. A claim whose factual core the evidence supports is TRUE (4-5) even when the text is editorialized, one-sided, uses loaded language, draws a stronger-than-proven inference, or is awkwardly/oddly worded. Bias belongs to the source (a separate axis), NOT to the claim's veracity — do not lower veracity for slant.

Downgrade to MISLEADING (3) ONLY when the framing makes a reasonable reader form a FALSE factual belief — i.e. the presentation asserts or implies something the evidence CONTRADICTS, or that NO reasonable reading of the evidence supports:
- CAUSATION: a causal / quid-pro-quo link the evidence CONTRADICTS or fabricates — NOT one the evidence reports as a real, documented connection (a stronger-than-proven but evidence-consistent inference, e.g. "in exchange for" on a documented investment-then-benefit sequence, is opinion, not misinformation).
- SCOPE: a scope the evidence CONTRADICTS ("every state" when it is one; "all channels" when only one closed; "across the state" when it was one county) — NOT mere emphasis, a loaded adjective, or a defensible characterization.
- CONTEXT: an omission that REVERSES the factual meaning, or presents a decisively-refuted claim as open — NOT context that merely adds nuance or that a partisan reader would weigh differently.
- COMPARISON: a comparison that is factually false or fabricates the takeaway.

If the factual takeaway a reasonable reader forms is TRUE — even from biased, opinionated, or oddly-worded text — the verdict is NONE. Reserve MISLEADING for framing that yields a FALSE factual impression the evidence does not support.

Classify:
- FALSE: a central claim is contradicted by the evidence (a fabricated event, a false attribution, or an image/video that does not authentically show what is claimed).
- MISLEADING: the literal words are supportable but the presentation deceives — essential context omitted, a quote stripped of meaning-changing context, an implied-but-unproven link, or inflated/decontextualised scope.
- UNSUPPORTED: no credible evidence establishes a central claim either way — grounds to flag.
- NONE: accurate and not misleading (including accurate reporting that "X said Y" where the text does not itself assert the false Y as true).

Score veracity 1-5 — the misinformation scale: 5 true and not misleading, 4 essentially accurate, 3 misleading or unsupported, 2 mostly false, 1 false. misinfo = veracity <= 3.

Output strict JSON: {"veracity": <1-5>, "misinfo": <true|false>, "misinfo_type": "FALSE|MISLEADING|UNSUPPORTED|NONE", "justification": "<one or two sentences citing the evidence and the text's framing>"}"""


def build_plan_text_messages(raw_text: str) -> list[dict]:
    return [{"role": "system", "content": PLAN_TEXT_SYSTEM},
            {"role": "user", "content": f"RAW TEXT:\n{raw_text}\n\nIdentify the load-bearing claim and the best query."}]


def parse_plan_text(raw: str) -> dict:
    d = _load_obj(raw)
    return {"main_claim": str(d.get("main_claim", "")).strip(), "query": str(d.get("query", "")).strip()}


def build_controller_messages(state: str) -> list[dict]:
    return [{"role": "system", "content": CONTROLLER_SYSTEM},
            {"role": "user", "content": state + "\n\nChoose the next action."}]


def parse_controller(raw: str) -> dict:
    d = _load_obj(raw)
    act = str(d.get("action", "")).strip().upper()
    if act not in {"VERDICT", "READ", "SEARCH"}:
        act = "VERDICT"
    idx = [int(i) for i in (d.get("read_indices") or []) if str(i).lstrip("-").isdigit()]
    prov = str(d.get("provider", "serper")).strip().lower()
    return {"analysis": str(d.get("analysis", "")).strip(),
            "open_questions": [str(x) for x in (d.get("open_questions") or [])],
            "action": act, "read_indices": idx,
            "query": str(d.get("query", "")).strip(),
            "provider": prov if prov in {"serper", "exa"} else "serper"}


def build_verdict_text_messages(raw_text: str, analysis: str, evidence: str) -> list[dict]:
    return [{"role": "system", "content": VERDICT_TEXT_SYSTEM},
            {"role": "user", "content": (
                f"RAW TEXT:\n{raw_text}\n\nRUNNING ANALYSIS:\n{analysis}\n\n"
                f"EVIDENCE:\n{evidence}\n\nIssue the misinformation verdict."
            )}]


REPAIR_JSON_SYSTEM = """You fix malformed JSON. You are given one JSON object that has a syntax error — most often an unescaped double-quote or newline inside a string value, a missing comma, or a trailing comma. Return ONLY the corrected JSON object, with nothing before or after it.

Rules:
- Preserve EVERY field and ALL text content exactly — do not add, drop, reword, summarize, translate, or shorten anything.
- Fix ONLY the syntax: escape stray quotes/newlines inside string values, insert missing commas, remove trailing commas.
- The output must be valid, parseable JSON and nothing else."""


def build_repair_messages(broken: str) -> list[dict]:
    return [{"role": "system", "content": REPAIR_JSON_SYSTEM},
            {"role": "user", "content": f"Fix this JSON:\n{broken}"}]
