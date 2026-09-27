"""READ v5 prompts — APPROVED by Daniel 2026-08-03 (suite-runner wiring next).

Schema settled 2026-08-03: one evidence bucket + one single-token flag, direction FIRST
(consistent with the 2026-07-30 four-arm ordering measurement). Flags 5/4/3/2/1/X/I.
The flag token's logprob is read as a soft per-document vote (verify DeepInfra returns
logprobs for DeepSeek-V4-Flash before relying on it).

READ is SINGLE-CLAIM (Daniel 2026-08-03, final): one call judges one document against
one claim. In production a document shared by several claims of a post is read once per
claim — we accept the redundancy to keep reads unconditioned and cacheable per claim.

Two prompts, one schema:
  READ_SYS_V5   — the production read step (DeepSeek-V4-Flash).
  LABELLER_SYS  — the silver-labeller variant (Sonnet at most, Haiku if the bake-off
                  says it suffices — Daniel 2026-08-03; no Opus/Fable): same task,
                  same schema, principles-only (no worked examples), plus a notes
                  field. Daniel's 3-bucket gold converts losslessly for scoring:
                  evidence = union of his three lists; direction maps 1:1
                  (supports→5, partially-supports→4, partially-refutes→2, refutes→1,
                  context→X, irrelevant→I; 3 had no UI button in round 1).
"""

READ_SYS_V5 = """\
You are reading ONE web document to judge its bearing on ONE claim.

You get the claim, its MODE, and a numbered list of sentences from the document.
Judge only what THIS document says about THIS exact claim — not what you know or
believe about the claim from anywhere else.

__MODE_DEFS__

__MODE_RULES_READ__

Reply with JSON only, direction first:
{"direction": "<flag>", "evidence": [<sentence numbers>], "reason": "<only when direction is I>"}

direction — the document's bearing on the exact claim. ONE flag:
  "5"  the document states or establishes the claim itself
  "4"  the document points toward the claim but does not establish it
  "3"  contested: the document carries real evidence both for and against the
       claim itself, or directional evidence on both sides that stays adjacent,
       so the claim cannot honestly be called supported or refuted
  "2"  the document points against the claim but does not disprove it
  "1"  the document contradicts or disproves the claim itself
  "X"  on-claim context: information about the claim's subject that permits
       neither support nor refutation — partial figures, background, or
       confirmation of side details but not the decisive part
  "I"  irrelevant: a different proposition, even if the topic is the same

evidence — the numbers of ALL sentences that bear on the claim. Select contiguous
runs: when relevant sentences are separated by one or two connecting sentences,
include the connecting sentences too. Empty only with "I".

Rules:
- Judge the exact proposition. A document establishing an adjacent fact — a
  related event, a similar figure, the same subject at another time or place —
  is "X" or "I", never "4".
- Precedence: sentences that explicitly address the claim set the direction.
  Credible conflicting sentences downgrade ("5"→"4", "1"→"2"). Only evidence
  that overturns the anchoring sentences flips the direction. Real balance on
  the claim itself is "3".
- Reporting that people make the claim, that it circulates, or that it was
  posted somewhere is not support. Judge what the document itself asserts.
- Opinion, prediction, and advocacy are not evidence in either direction.
- Mind the dates when given: a document from well before the claim's date
  describes earlier events, not the claim's — at most "X".
- If the decisive part of the claim — its quote, figure, or key event — is not
  addressed, the document is at most "X".

reason — only when direction is "I": "off-claim" (different subject) or
"adjacent-only" (same subject, different proposition).
"""

# Mode-aware READ (Daniel 2026-08-04): the mode blocks are substituted from
# claim_modes.py so QUERY, READ and the enrichment cannot drift apart. The prompt
# VERSION is tracked by READ_PROMPT_V in evidence_urn_run (v5 = placeholders
# stripped; v5.1 = first mode wording; v5.2 = attribution rule restated after the
# X-collapse it caused).
from eval.scripts.build_eval.claim_modes import (  # noqa: E402
    MODE_DEFS, MODE_RULES_READ)

# REVERTED to plain v5 (Daniel 2026-08-04): three successive attribution-rule
# wordings each inverted the stratum in a different way (v5.1 X->I collapse;
# v5.2 absence-as-refutation, false refutes on TRUE claims; v5.3 "reports the
# speaker making the statement" readmitted circulation as support, false support
# on FALSE claims). The mode blocks stay importable for post-Thursday work.
READ_SYS_MODE = (READ_SYS_V5.replace("__MODE_DEFS__\n\n__MODE_RULES_READ__\n\n", "")
                 .replace("You get the claim, its MODE, and a numbered list",
                          "You get the claim and a numbered list"))

# READ_SYS_ASOF — NEW CONSTANT, NEW NAME (Daniel 2026-09-11, clog/110926.md).
# READ_SYS_V5 / READ_SYS_MODE are untouched: every pinned run keeps its prompt hash.
# The mid-band diagnosis (12 worst class-1 posts) found HALF the errors were
# time-window blindness. Supplying the document's publication date alone does not
# fix it -- the base rubric's one date line says "well before", and three weeks of
# pre-war build-up coverage was still read as contradicting a war three weeks later.
# This constant is the same prompt with that one line replaced by an explicit
# as-of rule. Used only under keyclaim_urn_run --fix --read-asof.
READ_SYS_ASOF = READ_SYS_MODE.replace(
    """- Mind the dates when given: a document from well before the claim's date
  describes earlier events, not the claim's — at most "X".""",
    """- The claim is asserted AS OF its date, and each document carries its own
  publication date. When the claim is about a specific event, a document
  published BEFORE that event describes an EARLIER state of affairs: it can give
  context ("X") but it cannot contradict the claim, and its silence about the
  event is not evidence against it. A document published AFTER the claim's date
  is judged on whether the claim was accurate WHEN MADE, not on how the matter
  later turned out. A standing claim with no event date is judged on the evidence
  whenever it was published.""")
assert READ_SYS_ASOF != READ_SYS_MODE, "as-of substitution missed its anchor"


# READ_SYS_V6 — SIX-CLASS reader (Daniel 2026-09-14, clog/140926.md). The 7-class
# "3" (contested / mixed) fires on only ~2.5% of documents, and merging it into X at
# fit time costs no measurable AUC (dAUC +0.0011 [-0.0001, +0.0025] on the frozen
# 3,280). Daniel's ruling: a document that argues both ways is X -- on-claim context
# with no net direction -- so the reader no longer has a "3" button. Labels 5/4/X/I/2/1
# keep their meaning and their tokens, so every downstream parser is unchanged. This is
# READ_SYS_MODE with the "3" flag removed and the precedence rule's both-ways case
# routed to "X". READ_SYS_MODE / READ_SYS_V5 are untouched so every pinned run keeps
# its prompt hash.
READ_SYS_V6 = (READ_SYS_MODE
    .replace('''  "3"  contested: the document carries real evidence both for and against the
       claim itself, or directional evidence on both sides that stays adjacent,
       so the claim cannot honestly be called supported or refuted
''', "")
    .replace('''Only evidence
  that overturns the anchoring sentences flips the direction. Real balance on
  the claim itself is "3".''',
             '''Only evidence
  that overturns the anchoring sentences flips the direction. A document that
  both supports and undercuts the claim itself, leaving no net direction, is
  on-claim context ("X"), not a directional flag.'''))
assert '"3"' not in READ_SYS_V6, "v6 still defines a 3 flag"
assert READ_SYS_V6 != READ_SYS_MODE, "v6 substitution missed its anchor"


# READ_SYS_V6_1 — SIX-CLASS reader, MINIMAL edit (Daniel 2026-09-14, clog/140926.md).
# read-v6 was REJECTED: its extra sentence "A document that both supports and undercuts
# the claim ... is on-claim context (X)" over-recruited X (+575 X reads on the slice, flag 2
# fell 440->244) and cost AUC (0.8828 vs v5-cached 0.8891, gap -0.0063, outside the noise
# floor). v6.1 removes the "3" (contested/mixed) button and NOTHING ELSE: it deletes the "3"
# definition and drops the trailing "Real balance on the claim itself is '3'." clause from the
# precedence rule, WITHOUT adding any both-ways guidance. Labels 5/4/X/I/2/1 keep their meaning
# and tokens. READ_SYS_MODE / READ_SYS_V5 / READ_SYS_V6 are untouched (pinned hashes preserved).
READ_SYS_V6_1 = (READ_SYS_MODE
    .replace('''  "3"  contested: the document carries real evidence both for and against the
       claim itself, or directional evidence on both sides that stays adjacent,
       so the claim cannot honestly be called supported or refuted
''', "")
    .replace('''Only evidence
  that overturns the anchoring sentences flips the direction. Real balance on
  the claim itself is "3".''',
             '''Only evidence
  that overturns the anchoring sentences flips the direction.'''))
assert '"3"' not in READ_SYS_V6_1, "v6.1 still defines a 3 flag"
assert READ_SYS_V6_1 != READ_SYS_MODE, "v6.1 substitution missed its anchor"
assert "both supports and undercuts" not in READ_SYS_V6_1, "v6.1 must not carry v6's both-ways sentence"


# READ_SYS_V5B — v5 with the "decisive part" rule REPLACED (Daniel 2026-09-18,
# clog/180926.md). read-v5's "If the decisive part of the claim ... is not addressed,
# the document is at most X" made strict readers (Kimi K2.6, DeepSeek-V4-Pro) obey it
# literally and refuse to refute a false claim from a document that states the contrary
# facts without naming the claim's figure or source. This deletes ONLY that rule (the
# adjacent-fact rule is kept) and replaces it with two rules that separate
# contradiction-by-consequence from genuinely compatible facts. READ_SYS_MODE /
# READ_SYS_V5 / READ_SYS_V6 / READ_SYS_V6_1 are untouched (pinned hashes preserved).
READ_SYS_V5B = READ_SYS_MODE.replace(
    '''- If the decisive part of the claim — its quote, figure, or key event — is not
  addressed, the document is at most "X".''',
    '''- A document refutes the claim when the facts it establishes cannot be true if
  the claim is true, even if it never mentions the claim's wording, figure or
  source: a state of affairs contrary to the claim, an outcome the claim rules
  out, the real figure where the claim gives a different one. Grade that "1" or
  "2" by how directly it bears. The same holds for support.
- A document is at most "X" only when its facts and the claim can both be true:
  a different instance, object, time, place or measure, or a side detail of the
  same matter. A claim's exact quote or figure is not the decisive part when the
  document settles the underlying matter.''')
assert READ_SYS_V5B != READ_SYS_MODE, "v5b substitution missed its anchor"
assert "its quote, figure, or key event" not in READ_SYS_V5B, "v5b still carries the decisive-part rule"


# READ_SYS_V5C — SMALLEST change to v5 (Daniel 2026-09-18, clog/180926.md). v5b rewrote two
# rules and made Flash over-abstain; v5c keeps EVERY other line of v5 identical and only appends a
# refutation-by-consequence clause onto the existing "decisive part" rule. READ_SYS_MODE / V5 / V5B
# / V6 / V6_1 untouched (pinned hashes preserved).
READ_SYS_V5C = READ_SYS_MODE.replace(
    '''- If the decisive part of the claim — its quote, figure, or key event — is not
  addressed, the document is at most "X".''',
    '''- If the decisive part of the claim — its quote, figure, or key event — is not
  addressed, the document is at most "X", unless the facts it establishes cannot
  be true if the claim is true; then it refutes the claim ("1" or "2" by how
  directly it bears) even without naming the claim's figure or wording. A
  different instance, object, time, place or measure is still at most "X".''')
assert READ_SYS_V5C != READ_SYS_MODE, "v5c substitution missed its anchor"
assert "unless the facts it establishes cannot" in READ_SYS_V5C, "v5c clause missing"


# READ_SYS_V5X — the ABLATION (main thread 2026-09-18): v5 with the "decisive part" rule DELETED
# and nothing added. Isolates the rule itself from v5b's replacement wording: if V4-Pro on v5x
# recovers the votes it dropped under v5, the rule alone is the mechanism.
READ_SYS_V5X = READ_SYS_MODE.replace(
    '''- If the decisive part of the claim — its quote, figure, or key event — is not
  addressed, the document is at most "X".
''', "")
assert READ_SYS_V5X != READ_SYS_MODE, "v5x substitution missed its anchor"
assert "its quote, figure, or key event" not in READ_SYS_V5X, "v5x still carries the decisive-part rule"


LABELLER_SYS = """\
You are creating gold labels for an evidence-reading benchmark. For each case
you get one claim and the numbered sentences of one web document. Your labels
will be treated as ground truth, so accuracy beats speed: read every sentence
before deciding.

Judge only what THIS document says about THIS exact claim. Never use your own
knowledge of whether the claim is true — a document can support a false claim
and refute a true one, and the label must reflect the document.

Reply with JSON only, direction first:
{"direction": "<flag>", "evidence": [<sentence numbers>], "reason": "<only when direction is I>", "notes": "<one sentence when a case is borderline, else empty>"}

direction — the document's bearing on the exact claim. ONE flag:
  "5"  the document states or establishes the claim itself
  "4"  the document points toward the claim but does not establish it
  "3"  contested: real evidence both for and against the claim itself, or
       directional evidence on both sides that stays adjacent, so the claim
       cannot honestly be called supported or refuted
  "2"  the document points against the claim but does not disprove it
  "1"  the document contradicts or disproves the claim itself
  "X"  on-claim context: information about the claim's subject that permits
       neither support nor refutation — partial figures, background, or
       confirmation of side details but not the decisive part
  "I"  irrelevant: a different proposition, even if the topic is the same

evidence — ALL sentences that bear on the claim, as a highlighter would mark
them: contiguous runs, including one- or two-sentence connectors between
relevant sentences. On-claim context is welcome in the bucket; leave out only
sentences that neither bear on the claim nor give context to ones that do.
Empty only with "I".

Principles, in order of weight:
1. Exact proposition. Identify the decisive part of the claim — its quote,
   figure, or key event. A document that confirms surrounding details but not
   the decisive part is "X". A document about the same subject but a different
   proposition is "I".
2. Precedence. Sentences explicitly addressing the claim set the direction;
   credible conflicting sentences downgrade "5"→"4" or "1"→"2"; evidence that
   overturns the anchoring sentences flips it; genuine balance is "3".
3. Assertion, not circulation. That a claim is quoted, reported as circulating,
   or attributed to someone is not support for the claim. Opinion, prediction,
   and advocacy count in neither direction. Mind the dates when given: a
   document from well before the claim's date describes earlier events, not
   the claim's — at most "X".
4. When torn between a directional flag and "X", ask: could a careful reader
   cite this document as evidence in a dispute about the claim? If not, "X".

reason — only when direction is "I": "off-claim" (different subject) or
"adjacent-only" (same subject, different proposition).
"""

# READ_SYS_ATTRIB — TH1a (Daniel 2026-08-25): a SEPARATE prompt for attribution-
# mode claims, not a rule injected into v5 (the v5.1/v5.2/v5.3 route; all three
# inverted a stratum). Assertion claims keep plain v5 untouched. Daniel's rulings:
# a content fact-check that quotes the statement CONFIRMS the utterance ("5");
# reports of the utterance ARE support; silence is handled by the stratum's own
# fitted threshold, never by a prompt rule (the v5.2 burn). GATED: wired into the
# runner only after the offline re-read beats the stratum baseline four-cell
# (gold x mode); until then the runner routes every mode to READ_SYS_MODE.
READ_SYS_ATTRIB = """\
You are reading ONE web document to judge its bearing on ONE claim.

The claim is a REPORTED-SPEECH claim: it attributes a statement, quote, or post
to a named source. The question is whether the ATTRIBUTION is accurate — did
the source actually say, write, or post this — NOT whether the quoted content
is true. A document can confirm the source said something false; that confirms
this claim.

You get the claim and a numbered list of sentences from the document. Judge
only what THIS document shows about THIS exact attribution — not what you know
or believe from anywhere else.

Reply with JSON only, direction first:
{"direction": "<flag>", "evidence": [<sentence numbers>], "reason": "<only when direction is I>"}

direction — the document's bearing on the attribution. ONE flag:
  "5"  the document establishes that the source made the statement: it quotes
       or reports the statement as something the source actually said —
       including a document that disputes the CONTENT while confirming the
       source said it
  "4"  the document points toward the attribution without establishing it: a
       close paraphrase, a partial quote, or the source making the same point
       in other words on the same occasion
  "3"  contested: real evidence both ways on whether the source said it
  "2"  the document points against the attribution but does not disprove it —
       it attributes the statement to a different source, or shows the source's
       actual words on that occasion were materially different
  "1"  the document disproves the attribution: its own finding is that the
       quote is fabricated, doctored, misattributed, or never said
  "X"  on-claim context: about the source or the statement's subject, but shows
       nothing about whether the source made this statement
  "I"  irrelevant: a different statement, a different source, or a different
       subject

evidence — the numbers of ALL sentences that bear on the attribution. Select
contiguous runs: when relevant sentences are separated by one or two connecting
sentences, include the connecting sentences too. Empty only with "I".

Rules:
- The unit is the utterance. Truth, falsity, or fact-check verdicts about the
  quoted content set no direction by themselves; they matter only through what
  the document shows about whether the source said it.
- Reporting can be evidence here — but only when the DOCUMENT ITSELF vouches
  for the utterance: its own reporting, its own interview or transcript, or a
  quotation made in its own journalistic voice. Words the document attributes
  to a viral post, to critics or opponents of the source, or to the very claim
  it is examining are the claim circulating, not confirmation — at most "X".
- When the document examines whether the source really said it, its FINDING
  sets the direction and outranks every other signal in the document: it
  concludes the source said it, "5"; it concludes the statement is fabricated,
  doctored, misattributed, from a satire site, or that no record of the source
  saying it exists, "1"; it leaves the question open, "X". Such a document
  always restates the quote; the restatement is never itself support, and a
  document whose finding is "fabricated" is "1" no matter how fully it quotes
  the statement.
- Mind the claim's own polarity. When the claim is that the source did NOT say
  something, evidence the source DID say it points AGAINST the claim ("2" or
  "1"), and a finding that they never said it supports it ("5"). Judge the
  claim as stated, not its inner quotation.
- Exactness matters. "4" requires the same statement in nearly the same words
  on the claimed occasion. The same source saying something thematically
  similar — another occasion, another venue, a materially different meaning —
  is "X" or "I", never "4".
- Absence is not refutation: a document that simply does not mention the
  statement is "X" or "I", never "2" or "1". Only affirmative evidence of
  fabrication or misattribution refutes.
- Mind the dates when given: a document from well before the claim's date
  cannot report this statement — at most "X".
- Opinion about the source or about the content is not evidence either way.

reason — only when direction is "I": "off-claim" (different subject) or
"adjacent-only" (same source or subject, different statement).
"""
