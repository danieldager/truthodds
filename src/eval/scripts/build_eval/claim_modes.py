"""Claim MODE — the one distinction QUERY and READ must both respect.

Daniel 2026-08-04. The 2026-08-04 tranche-1 audit found the reader granting
directional flags to documents that only RELAY an assertion (a report of DOGE's
USAID claim flagged 5; a conditional-mood rumour, "seraient interdits", flagged 4
next to an uncited sentence saying the info was not yet official). The v5 rubric
already forbids this in one line; the reader does not honour it. Naming the mode
explicitly, in both prompts, is the minimal intervention.

SOURCE OF THE MODE (Daniel 2026-08-04, settled): the mode fed to QUERY and READ is
derived from **judged_axis** — the axis the FACT-CHECKER adjudicated
(derive_judged_axis.py) — NOT from claim_type. The gold label only applies to the
proposition the fact-checker actually settled. If READ assesses the other
proposition, we score its answer against a verdict that does not apply to it: the
claim "Hodkinson says COVID-19 is a hoax" is TRUE as an utterance and rated FALSE
as content, so reading the utterance and comparing to that label compares two
different claims. Both fields share this vocabulary precisely so either can drive
the mapping.

(Deployment note, not an eval concern: in production there is no verdict, so the
mode there comes from the claim itself and the extraction step verifies the
attribution and the content separately. That is the attribution-aware-nudging
design, and it does not change what this measurement must do.)

Definitions live here once and are imported by every prompt that needs them, so
QUERY and READ cannot drift apart. A third mode, "authentication" (does this
image/video genuinely show what it is presented as showing), is deliberately
deferred — claim_type == media_authenticity is already excluded at CAL/VAL
construction, and those claims are gated out of E1 by hand-read. When it lands,
add it in ONE place: here.
"""
from __future__ import annotations

MODE_ATTRIBUTION = "attribution"
MODE_ASSERTION = "assertion"

# claim_type (claim_screen.py) -> mode. media_authenticity is absent by design:
# such claims do not enter E1. Anything unrecognised falls back to assertion,
# which is the conservative default (it applies the stricter no-circulation rule).
_MODE_OF = {
    "quote_attribution": MODE_ATTRIBUTION,
    "event_occurrence": MODE_ASSERTION,
    "statistic_figure": MODE_ASSERTION,
    "causal_effect": MODE_ASSERTION,
    "policy_law": MODE_ASSERTION,
    "attribute_identity": MODE_ASSERTION,
}


def mode_of(claim_type: str | None) -> str:
    return _MODE_OF.get((claim_type or "").strip(), MODE_ASSERTION)


# Shared definitional text. Identical wording in QUERY and READ by construction.
MODE_DEFS = """A claim's MODE decides what settles it:
- "attribution": the claim is that a specific person or outlet said, wrote, or published a specific statement. What settles it is whether the utterance happened — NOT whether the statement's content is true. Someone can accurately be reported saying something false, and a true statement can be falsely attributed.
- "assertion": the claim states something about the world — an event, a quantity, an effect, a rule, a property. What settles it is the substance itself, not who repeated it."""

# Operative line per prompt. Same distinction, applied to each step's job.
# v5.3 wording (2026-08-04). TWO measured failures preceded it, in opposite
# directions, both from one clause each:
#   v5.1  "...addresses a different proposition — that is 'X', never a direction"
#         collided with the base rubric's definition of "I" -> X collapsed 31->0,
#         refutes on gold-FALSE attribution claims 16->6, stratum AUC .848->.800.
#   v5.2  "...said nothing of the kind, or that no such statement exists REFUTES
#         it — among the strongest available" licensed silence-as-refutation:
#         refutes on gold-TRUE attribution claims 2->12 while gold-FALSE refutes
#         FELL 16->10, stratum AUC .878->.692.
# v5.3 therefore states only what the mode CHANGES (what the direction is about,
# and that content-only docs are X rather than I) and explicitly closes the
# absence-is-refutation door. Refutation is left to the base rubric's own "1 =
# contradicts or disproves". VALIDATE ANY FUTURE EDIT SPLIT BY GOLD LABEL: the
# v5.2 aggregate "refutes doubled" looked like a fix and was the bug.
#
# Superseded note — the v5.1 wording read "...addresses a
# different proposition — that is 'X', never a direction", which collides head-on
# with the base rubric's definition of "I" ("irrelevant: a different proposition,
# even if the topic is the same"). Measured effect on 46 paired attribution
# claims: X collapsed 31 -> 0, I rose 70.6% -> 79.9%, and refuting voices on
# gold-FALSE claims fell 16 -> 6, taking the stratum's AUC from 0.848 to 0.800.
# A document whose cited sentence read "Bernice King and Martin Luther King III
# did not mention Trump in their statement" — the most on-point utterance
# evidence available — was flagged "I". So the rule is now stated POSITIVELY
# (what IS directional), it says outright that absence-of-utterance evidence is
# a REFUTING direction, and it blocks the slide to "I" explicitly.
MODE_RULES_READ = """Apply the mode when judging this document:
- attribution: the direction is set by evidence about the SAYING, not by whether the statement is true. A document that reproduces, quotes, or reports the speaker making the statement bears on the claim directly. A document that only argues whether the statement's content is right or wrong settles nothing about who said it: flag that "X", not "I" — it concerns this claim's subject without reaching the question of authorship. A document that simply does not mention the statement has not shown the statement was never made; absence of the quote is not evidence against it.
- assertion: a document that merely reports someone making the claim, or reports that the claim is circulating, asserts nothing itself — that is "X". Only a document that speaks in its own voice about the substance sets a direction."""

MODE_RULES_QUERY = """Apply the mode when writing the query:
- attribution: query for the utterance — the speaker plus the distinctive wording. Do NOT query whether the content is true; that is a different proposition.
- assertion: query for the substance, not for who repeated it."""
