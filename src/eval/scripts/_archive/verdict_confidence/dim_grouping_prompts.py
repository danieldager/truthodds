"""Likert scoring prompts for the dimension-grouping experiment (5 dims).

Goal: A/B whether scoring the 5 dims in ONE call vs 2 calls (2+3) vs 5 calls changes
(a) cross-dim independence, (b) scale usage (the 1/3/5 clustering, i.e. how often 2 & 4 are
used), (c) how well the scores predict correctness — and at what cost.

Design guard against confounds: every grouping draws its per-dim rubric text from the SAME
`DIM_RUBRICS` strings and the SAME `HEADER`. The only thing that varies between groupings is
which dims share a call (and therefore the JSON key set in the output) — exactly the variable
under test. `FEWSHOT_BLOCK` is a single shared string (empty for the zero-shot baseline), so
adding few-shot later stays byte-identical across groupings too.

The 4 dims are verbatim from `verify_prompts.py::LIKERT_SYSTEM`; `contextual_integrity` is the
new 5th truth dim (misleadingness/framing) from `docs/likert_dimensions.md` — anchors 5/3/1 are
the doc's; 2 and 4 are authored here to complete the scale (2/4 usage is itself under test).
"""
from __future__ import annotations

from pipeline.verify_prompts import VerifyParseError, _clamp_1_5, _load_obj, _strip  # noqa: F401

# Canonical order. Truth dims: veracity, contextual_integrity. Confidence dims: the three evidence_*.
DIMS: tuple[str, ...] = (
    "veracity",
    "evidence_sufficiency",
    "evidence_agreement",
    "source_reliability",
    "contextual_integrity",
)

DIM_RUBRICS: dict[str, str] = {
    "veracity": (
        "veracity — which way, and how strongly, does the evidence point on the claim's MAIN assertion?\n"
        "  5: Clearly true — the evidence confirms the main assertion and its load-bearing specifics (numbers, dates, named entities).\n"
        "  4: Mostly true — the main assertion holds, but a load-bearing specific may be approximate or slightly off.\n"
        "  3: Mixed or indeterminate — the evidence neither clearly confirms nor clearly contradicts the main assertion.\n"
        "  2: Mostly false — the main assertion is contradicted, though a detail may be unclear.\n"
        "  1: Clearly false — the evidence contradicts the main assertion."
    ),
    "evidence_sufficiency": (
        "evidence_sufficiency — how directly and fully does the evidence address the claim?\n"
        "  5: The evidence speaks directly to the main assertion AND its load-bearing specifics.\n"
        "  4: The evidence directly addresses the main assertion, but leaves a specific (a number, date, qualifier) untouched.\n"
        "  3: Partial — it touches the topic but addresses the specific assertion only loosely, or through a single piece of evidence.\n"
        "  2: Only topically adjacent — it does not speak to the specific assertion.\n"
        "  1: Nothing in the analysis bears on the claim's substance."
    ),
    "evidence_agreement": (
        "evidence_agreement — do the pieces of evidence point the same way?\n"
        "  5: All pieces point the same way (toward true or toward false); no contradictions.\n"
        "  4: Broad convergence; only minor differences of scope or wording.\n"
        "  3: Some tension between pieces — or only one piece exists, so agreement cannot be judged.\n"
        "  2: Notable conflict — at least one piece points each way.\n"
        "  1: The evidence flatly contradicts itself on a load-bearing point."
    ),
    "source_reliability": (
        "source_reliability — how trustworthy are the cited sources, regardless of what they say?\n"
        "  5: Primary or authoritative — official records, peer-reviewed work, named domain experts, court/legislative documents.\n"
        "  4: Established mainstream outlets or institutional reporting.\n"
        "  3: Mixed — some reputable, but leaning on secondary or lower-tier reporting.\n"
        "  2: Mostly blogs, opinion, or low-traffic sites.\n"
        "  1: Only fringe, anonymous, or unidentifiable sources — or none usable."
    ),
    "contextual_integrity": (
        "contextual_integrity — beyond the literal assertion, does the impression the claim creates survive the full evidence?\n"
        "  5: Literal content AND its evident implication are supported; no material context omitted that would change a reader's takeaway.\n"
        "  4: The gist holds, but a minor qualification or piece of context is missing that only slightly colors the takeaway.\n"
        "  3: Literally accurate but missing context or qualification a reasonable reader needs — a partially unsupported impression.\n"
        "  2: The framing materially misleads — important context is omitted or the emphasis implies more than the evidence supports, even though the literal words are defensible.\n"
        "  1: True-ish details arranged to imply a conclusion the evidence contradicts — cherry-picking, implied causation, or a stale event shown as current."
    ),
}

HEADER = (
    "You are assessing a claim from a synthesized analysis of evidence — the analysis is your ONLY "
    "context; add no outside knowledge. Score each dimension below 1-5 against its own definition.\n\n"
    "Use the full scale, including 2 and 4 deliberately — partial, leaning, or qualified cases "
    "belong at 2 or 4; do not collapse every judgement to 1, 3, or 5."
)

# Shared across ALL groupings (byte-identical). Empty = zero-shot baseline. Populate with the
# SAME exemplar block for the few-shot pass; the output portion of each exemplar will naturally
# list only the in-scope dims, which is inherent to grouping, not a confound.
FEWSHOT_BLOCK = ""

# Grouping configs: each is a list of dim-subsets; one LLM call per subset. Every config scores
# all 5 dims — they differ only in how the dims are partitioned across calls.
GROUPINGS: dict[str, list[list[str]]] = {
    "all5": [list(DIMS)],
    "split_2_3": [["veracity", "contextual_integrity"],
                  ["evidence_sufficiency", "evidence_agreement", "source_reliability"]],
    "per_dim": [[d] for d in DIMS],
}


def build_messages(claim_text: str, analysis: str, dims: list[str]) -> list[dict]:
    """One scoring call over `dims`. Justification precedes the scores so the model reasons
    in-band before committing to numbers (matters most for the non-reasoning arm)."""
    rubric = "\n\n".join(DIM_RUBRICS[d] for d in dims)
    keys = ",\n  ".join(f'"{d}": <1-5>' for d in dims)
    schema = ('{\n  "justification": "<one or two sentences citing what in the analysis drives '
              'the score(s)>",\n  ' + keys + "\n}")
    fewshot = f"{FEWSHOT_BLOCK}\n\n" if FEWSHOT_BLOCK else ""
    system = f"{HEADER}\n\n{rubric}\n\n{fewshot}Output strict JSON, no prose outside the JSON:\n{schema}"
    user = f"Claim: {claim_text}\n\nSynthesized analysis:\n{analysis}\n\nAssign the score(s)."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse(raw: str, dims: list[str]) -> dict:
    """-> {justification, <dim>: int|None, ...}. A missing/unparseable dim is None (not silently
    defaulted) so the scorer can exclude it rather than bias the distribution."""
    obj = _load_obj(_strip(raw))
    out: dict = {"justification": str(obj.get("justification", "")).strip()}
    for d in dims:
        v = obj.get(d)
        out[d] = _clamp_1_5(v) if v is not None else None
    return out
