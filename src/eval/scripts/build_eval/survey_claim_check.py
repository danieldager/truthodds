"""Claim-level check (Daniel 2026-09-21): the third and final gate of the survey selection
pipeline, run AFTER key_claim_extract. The post-filter (survey_post_filter.py) screens posts before
extraction; this screens the extracted claims themselves, so that only clean, self-contained,
contested, in-scope propositions survive to the selection page. Daniel's rule: what matters is that
the claims that SURVIVE are clean; losing a good post is cheap.

Per claim it decides keep/drop and records the flags the decision rests on: whether the claim is
actually asserted by the post (not imported from the note or elsewhere), whether it is a contested
key claim, whether it is satire/sarcasm, trivial, or opinion/characterisation, whether it stands
alone for a stranger, and whether it keeps the post's contested wording. It also tags one political
subtopic and a broad political yes/no.

The Community Note is passed as context, to help judge whether the claim is really the post's own
assertion and whether it is contested. It must NEVER drive keep/drop on veracity grounds: a claim
being false is not a reason to drop it, and the note never writes the claim text.

    uv run python -m eval.scripts.build_eval.survey_claim_check \
        --posts <posts.json> --claims <extract_out.json> --out <claim_check.json> \
        [--model deepseek-ai/DeepSeek-V4-Flash | deepseek-ai/DeepSeek-V4-Pro]

All DeepInfra calls go through the shared deepinfra_runner (RateGate, per-item cache, retries,
sweep, machine-wide slot pool); this script only builds request bodies and parses replies.

Output: {"prompt","prompt_hash","model","system_prompt",
         "posts": {post_id: {"claims": [{claim, ...flags..., keep, drop_reason, subtopic, political}]}},
         "survivors": [flat list of kept claims with post_id]}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.prompt_hash import prompt_hash  # noqa: E402
from deepinfra_runner import run  # noqa: E402
from eval.scripts.build_eval.reader_lab import PROVIDER, PROVIDERS, _parse_json  # noqa: E402

SUBTOPICS = ["immigration", "economy", "foreign_affairs", "society_and_culture",
             "elections_and_government", "crime_and_justice", "health_and_science", "other_political"]

CLAIM_CHECK_SYS = """\
You are the final reviewer of a fact-checking pipeline. Earlier steps took one social media post and
extracted one or two candidate claims from it. Your job is to decide, for each candidate claim,
whether it is clean enough to put in front of a fact-checker, and to tag it. What matters is that the
claims that survive are clean; dropping a marginal claim is cheap.

You are given the post's text (and any quoted post), a Community Note as context, and the candidate
claims. The note is context only: it helps you see whether the claim is really the post's own
assertion and whether readers dispute it. The note NEVER decides keep or drop on grounds of truth. A
claim being false, or contradicted by the note, is not a reason to drop it - a false claim that the
post genuinely makes is exactly what we want to check. Judge each claim on its own text against the
post, not on whether it is true.

For each candidate claim, set these flags:
- asserted_by_post: true when the post itself states or plainly presupposes this claim's content. Set
  it false when the claim's substance comes from the note or from outside knowledge rather than the
  post - for instance a person's role, team, location, date or a corrected fact that appears only in
  the note and not in the post. The claim must be the post's assertion, not the note's.
- is_contested_key_claim: true when the claim is a consequential, disputable proposition that a reader
  could reasonably contest and that is central to what the post puts across; false when it is minor,
  background, or something nobody would dispute.
- is_satire_or_sarcasm: true when the claim comes from a post performed to entertain - satire, parody,
  a joke, a sarcastic bit - and is not meant literally. Vivid, extreme or mocking phrasing is not by
  itself satire.
- is_trivial: true when the claim is low-stakes and of no public consequence - a personal or private
  matter, a mundane everyday observation, small talk - something nobody would act on. A claim about a
  public figure or public matter is not trivial.
- is_opinion_or_characterisation: true when the claim's decisive word is an evaluation, judgment,
  prediction, characterisation or stated goal that no document can settle, rather than a concrete
  event, act or quantity.
- stands_alone: true when a stranger with a search engine and no access to the post could find and
  rate this claim true or false as written - every person, place, event and figure is named and no
  reference is left dangling.
- keeps_contested_wording: true when the claim keeps the post's own contested specifics - the
  nationality, identity, number, cause or loaded descriptor the claim turns on - rather than softening
  them into a neutral paraphrase.

Then decide overall keep. Keep the claim only when ALL of these hold: asserted_by_post is true,
stands_alone is true, is_satire_or_sarcasm is false, is_trivial is false, and
is_opinion_or_characterisation is false. Drop it otherwise. A claim may be kept even if it is not the
most contested one and even if it slightly loosened the wording; those are quality flags, not drop
conditions. When you drop, give the single reason that most decides it, as a short code:
not_asserted_by_post, satire, trivial, opinion, or does_not_stand_alone.

Also tag each kept-or-dropped claim:
- subtopic: exactly one of immigration; economy; foreign_affairs; society_and_culture (identity,
  religion, race, gender, culture-war, social issues); elections_and_government (elections, voting,
  officials, government, legislation); crime_and_justice; health_and_science; other_political (any
  other political matter, or anything not political). Choose the single best fit.
- political: true when the claim concerns a political matter broadly understood (government,
  elections, policy, immigration, economy, foreign affairs, culture-war/identity disputes, crime and
  justice, health policy), false otherwise.

Reply with JSON only, one object per candidate claim in the same order given:
{"claims": [{"asserted_by_post": bool, "is_contested_key_claim": bool, "is_satire_or_sarcasm": bool,
  "is_trivial": bool, "is_opinion_or_characterisation": bool, "stands_alone": bool,
  "keeps_contested_wording": bool, "keep": bool, "drop_reason": "<code or empty>",
  "subtopic": "<one subtopic>", "political": bool}]}
"""
PROMPT_V1 = "claim-check-v1"

# claim-check-v2 (Daniel 2026-09-21): one principled iteration on v1 after the 146-post hand-judge. v1
# erred in both directions: it dropped legitimate claims (asserted_by_post fired on claims carried in a
# quoted/embedded post, and on neutral rephrasings of what the post states; satire fired on genuine but
# absurd-sounding claims about real named actors) while keeping claims it should have dropped (a vote or
# statement re-described with a loaded evaluative label, a stated policy goal, broad editorializing, and
# trivially-true background that is not the point). v2 fixes both on principle (no dataset examples):
# (a) a claim carried in a quoted/embedded post, or a faithful neutral rephrasing of what the post
# states, IS asserted by the post; only content coming from the note or outside knowledge is not;
# (b) satire only when the post is genuinely performed to entertain - a surprising, extreme or false-
# sounding claim asserted literally about a real named person or institution is not satire; (c) sharpen
# opinion/characterisation to catch an action re-described with an evaluative label, a stated goal or
# intention with no concrete act, and broad editorialising; (d) require is_contested_key_claim for keep,
# so trivially-true background that is not what the post is about drops out.
CLAIM_CHECK_SYS_V2 = """\
You are the final reviewer of a fact-checking pipeline. Earlier steps took one social media post and
extracted one or two candidate claims from it. Your job is to decide, for each candidate claim, whether
it is clean enough to put in front of a fact-checker, and to tag it. What matters is that the claims
that survive are clean; dropping a marginal claim is cheap.

You are given the post's text (and any quoted or embedded post), a Community Note as context, and the
candidate claims. The note is context only: it helps you see whether the claim is really the post's own
assertion and whether readers dispute it. The note NEVER decides keep or drop on grounds of truth. A
claim being false, or contradicted by the note, is not a reason to drop it - a false claim that the
post genuinely makes is exactly what we want to check. Judge each claim on its own text against the
post, not on whether it is true.

For each candidate claim, set these flags:
- asserted_by_post: true when the post itself states or plainly presupposes this claim's content. A
  claim carried in a quoted or embedded post shown with the post counts as asserted by the post, because
  the post puts it across. A faithful, neutral rephrasing or summary of what the post says also counts -
  do not require the claim to use the post's exact words. Set it false only when the claim's substance
  comes from the Community Note or from outside knowledge rather than from the post - for instance a
  person's role, team, location, date or a corrected fact that appears only in the note and not in the
  post, or in the quoted post.
- is_contested_key_claim: true when the claim is a consequential, disputable proposition that is central
  to what the post puts across - the assertion a reader would dispute or a correction would target. Set
  it false when the claim is minor, background, scene-setting, or a trivially-true fact that the post
  states only in passing and is not what the post is really asserting.
- is_satire_or_sarcasm: true only when the post is genuinely performed to entertain - satire, parody, a
  comedic bit - and asserts nothing it means literally. A surprising, extreme, absurd-sounding or
  false-sounding claim that the post asserts literally about a real named person, institution or event
  is NOT satire; neither is vivid, mocking or exaggerated phrasing on its own.
- is_trivial: true when the claim is low-stakes and of no public consequence - a personal or private
  matter, a mundane everyday observation, small talk, celebrity or consumer trivia - something nobody
  would act on. A claim about a public figure acting in public life, or about a public matter, is not
  trivial.
- is_opinion_or_characterisation: true when the claim is not a plain factual proposition but a judgment,
  evaluation, prediction, stated goal or characterisation that no document can settle. This includes an
  action re-described with a loaded evaluative label rather than stated plainly (describing how someone
  voted, spoke or acted in terms of the cause or atrocity it is said to serve, or calling an ambiguous
  act "advocating" or "endorsing" a position), a stated goal, aim or intention with no concrete act
  attached, and broad editorialising about what a group is free to do or what an institution allows.
- stands_alone: true when a stranger with a search engine and no access to the post could find and rate
  this claim true or false as written - the central person, place and event are named. A missing minor
  descriptor does not fail this; a dangling core subject or event ("a man", "the event described", "a
  crime" with no specifics) does.
- keeps_contested_wording: true when the claim keeps the post's own contested specifics - the
  nationality, identity, number, cause or loaded descriptor the claim turns on - rather than softening
  them into a neutral paraphrase.

Then decide overall keep. Keep the claim only when ALL of these hold: asserted_by_post is true,
is_contested_key_claim is true, stands_alone is true, is_satire_or_sarcasm is false, is_trivial is
false, and is_opinion_or_characterisation is false. Drop it otherwise. A claim may be kept even if it
slightly loosened the wording; that is a quality flag, not a drop condition. When you drop, give the
single reason that most decides it, as a short code: not_asserted_by_post, not_contested, satire,
trivial, opinion, or does_not_stand_alone.

Also tag each claim:
- subtopic: exactly one of immigration; economy; foreign_affairs; society_and_culture (identity,
  religion, race, gender, culture-war, social issues); elections_and_government (elections, voting,
  officials, government, legislation); crime_and_justice; health_and_science; other_political (any other
  political matter, or anything not political). Choose the single best fit.
- political: true when the claim concerns a political matter broadly understood (government, elections,
  policy, immigration, economy, foreign affairs, culture-war/identity disputes, crime and justice,
  health policy), false otherwise.

Reply with JSON only, one object per candidate claim in the same order given:
{"claims": [{"asserted_by_post": bool, "is_contested_key_claim": bool, "is_satire_or_sarcasm": bool,
  "is_trivial": bool, "is_opinion_or_characterisation": bool, "stands_alone": bool,
  "keeps_contested_wording": bool, "keep": bool, "drop_reason": "<code or empty>",
  "subtopic": "<one subtopic>", "political": bool}]}
"""
PROMPT_V2 = "claim-check-v2"

# claim-check-v3 (Daniel 2026-09-21): iteration on two failure TYPES seen on the dev-146 set (each hurting
# several posts, not a per-post patch), left otherwise identical to v2. Type 1: stands_alone fired on claims
# whose actor is given by role/nationality/category with a named place and concrete event (over-strict).
# Type 2: asserted_by_post fired on a fact carried in a quoted/embedded post or a faithful rephrasing of a
# figure the post states (over-strict). v3 loosens exactly those two flag definitions; keep rule and every
# other flag are byte-identical to v2. Validated out-of-sample on the pilot, not on the dev set.
_V3_STANDS = """- stands_alone: true when a stranger with a search engine and no access to the post could find and rate
  this claim true or false as written - the central person, place and event are named. A claim whose
  actor is given by role, nationality or category (not a proper name) together with a named place and a
  concrete event DOES stand alone; a specific but unnamed individual is fine when the event and place are
  concrete. It fails only when the core subject or event is a dangling placeholder with no specifics ("a
  man" with no context, "the event described", "a crime" with nothing else)."""
_V3_ASSERT = """- asserted_by_post: true when the post itself states or plainly presupposes this claim's content. A
  claim carried in a quoted or embedded post shown with the post counts as asserted by the post, because
  the post puts it across. A faithful, neutral rephrasing or summary of what the post says also counts,
  including restating a figure the post gives in its own framing (a price, a count, a share) as a general
  or average figure - do not require the claim to use the post's exact words. Set it false only when the
  claim's substance comes from the Community Note or from outside knowledge rather than from the post - a
  person's role, team, location, date or a corrected fact that appears only in the note and not in the
  post or its quoted post."""
_A2 = """- asserted_by_post: true when the post itself states or plainly presupposes this claim's content. A
  claim carried in a quoted or embedded post shown with the post counts as asserted by the post, because
  the post puts it across. A faithful, neutral rephrasing or summary of what the post says also counts -
  do not require the claim to use the post's exact words. Set it false only when the claim's substance
  comes from the Community Note or from outside knowledge rather than from the post - for instance a
  person's role, team, location, date or a corrected fact that appears only in the note and not in the
  post, or in the quoted post."""
_S2 = """- stands_alone: true when a stranger with a search engine and no access to the post could find and rate
  this claim true or false as written - the central person, place and event are named. A missing minor
  descriptor does not fail this; a dangling core subject or event ("a man", "the event described", "a
  crime" with no specifics) does."""
assert CLAIM_CHECK_SYS_V2.count(_A2) == 1 and CLAIM_CHECK_SYS_V2.count(_S2) == 1
CLAIM_CHECK_SYS_V3 = CLAIM_CHECK_SYS_V2.replace(_A2, _V3_ASSERT).replace(_S2, _V3_STANDS)
PROMPT_V3 = "claim-check-v3"

# claim-check-v4 (Daniel 2026-09-21): principled iteration on v3 after the held-out 1,000-post pilot,
# where v3 survivors were only 79% clean out-of-sample. The residual not-clean sat in recurring failure
# TYPES, not one-off posts: (1) a loaded characterisation or accusatory label presented as fact (an act
# re-described in terms of the cause/atrocity/motive it is said to serve, "advocating"/"believes in"/
# "endorses" a position, a person called a criminal/bigot/traitor without a stated concrete act); (2)
# hyperbole and exaggeration used for emphasis; (3) claims about popularity, virality, mood, momentum or
# how widely a thing is believed/discussed/shared; (4) sweeping rhetoric about what a whole group is,
# wants or is doing; and (5) a large non-political fraction. v4 changes exactly two things vs v3, both
# tied to those types (no dataset examples): (a) broaden is_opinion_or_characterisation to name these
# rhetoric/characterisation/mood/group types and re-state the keep target (a concrete, specific,
# consequential thing a named actor did/said/decided, or a specific number/event/state of the world,
# stated plainly); (b) add a HARD political-only keep - a claim is kept only when political is true - and
# a not_political drop code. Every other flag definition is byte-identical to v3.
_V3_OPINION = """- is_opinion_or_characterisation: true when the claim is not a plain factual proposition but a judgment,
  evaluation, prediction, stated goal or characterisation that no document can settle. This includes an
  action re-described with a loaded evaluative label rather than stated plainly (describing how someone
  voted, spoke or acted in terms of the cause or atrocity it is said to serve, or calling an ambiguous
  act "advocating" or "endorsing" a position), a stated goal, aim or intention with no concrete act
  attached, and broad editorialising about what a group is free to do or what an institution allows.
"""
_V4_OPINION = """- is_opinion_or_characterisation: true when the claim is not a plain factual proposition but a judgment,
  evaluation, prediction, stated goal or characterisation that no document can settle. Read this broadly:
  it fires on an action, vote or statement re-described with a loaded evaluative or accusatory label
  rather than stated plainly (framing what someone did in terms of the cause, atrocity or motive it is
  said to serve, or saying a person is "advocating", "endorsing", "supporting" or "believes in" a
  position rather than naming the concrete thing they did); on an accusatory label, slur or insult
  fixed to a person (that someone is a criminal, a bigot, a pedophile, a traitor, corrupt, and the like)
  where the label rather than a stated concrete act is the substance; on hyperbole or exaggeration used
  for emphasis rather than as a literal quantity; on a claim about popularity, virality, mood, momentum,
  or how widely a thing is believed, discussed, shared or has "gone viral"; on sweeping rhetoric about
  what a whole group is, wants, believes or is doing; and on a stated goal, aim or intention with no
  concrete act attached. The claim clears this flag only when its substance is a concrete, specific,
  consequential thing that a named person or body did, said or decided, or a specific number, event or
  state of the world, stated plainly rather than as a verdict wrapped around it.
"""
_V3_KEEP = """Then decide overall keep. Keep the claim only when ALL of these hold: asserted_by_post is true,
is_contested_key_claim is true, stands_alone is true, is_satire_or_sarcasm is false, is_trivial is
false, and is_opinion_or_characterisation is false. Drop it otherwise. A claim may be kept even if it
slightly loosened the wording; that is a quality flag, not a drop condition. When you drop, give the
single reason that most decides it, as a short code: not_asserted_by_post, not_contested, satire,
trivial, opinion, or does_not_stand_alone."""
_V4_KEEP = """Then decide overall keep. Keep the claim only when ALL of these hold: asserted_by_post is true,
is_contested_key_claim is true, stands_alone is true, political is true, is_satire_or_sarcasm is false,
is_trivial is false, and is_opinion_or_characterisation is false. The political requirement is hard: a
claim that is not about a political matter is dropped however clean and checkable it is. Drop it
otherwise. A claim may be kept even if it slightly loosened the wording; that is a quality flag, not a
drop condition. When you drop, give the single reason that most decides it, as a short code:
not_asserted_by_post, not_contested, not_political, satire, trivial, opinion, or does_not_stand_alone."""
assert CLAIM_CHECK_SYS_V3.count(_V3_OPINION) == 1 and CLAIM_CHECK_SYS_V3.count(_V3_KEEP) == 1
CLAIM_CHECK_SYS_V4 = CLAIM_CHECK_SYS_V3.replace(_V3_OPINION, _V4_OPINION).replace(_V3_KEEP, _V4_KEEP)
PROMPT_V4 = "claim-check-v4"

# claim-check-v5 (Daniel 2026-09-21): one further principled iteration on v4 after the held-out pilot
# comparison. v4 broadened the opinion/characterisation gate in prose but the model kept firing it False
# on the very TYPES the prose named - a sweeping generalisation about what a group is doing ("X are taking
# over the country"), a causal generality ("a president's policies control gas prices"), a causal-blame
# characterisation ("Y is getting people killed"), and an attribution of a belief or stance to a person
# ("Z believes in ..."). These are general/characterising propositions, not discrete checkable events, and
# a negative flag the model under-applies is the wrong lever. v5 adds a POSITIVE requirement that names the
# keep target Daniel stated: a claim is kept only when it asserts a specific, discrete, datable-or-locatable
# act, event, decision, statement or quantity attributable to a named actor - not a general condition, a
# trend, a causal generality, a group generalisation, or a belief/stance/motive attributed to a person.
# Everything else is byte-identical to v4. This is a general principle tied to the observed failure TYPE,
# not a per-post patch, and it leaves specific inflammatory-but-concrete claims (a named event at a named
# place, a specific vote, a specific quoted statement, a specific figure) untouched.
_V4_OPIN_ANCHOR = """- stands_alone: true when a stranger with a search engine and no access to the post could find and rate
  this claim true or false as written - the central person, place and event are named. A claim whose"""
_V5_SPECIFIC = """- is_specific_event_or_fact: true when the claim asserts a specific, discrete, datable-or-locatable act,
  event, decision, statement, or quantity attributable to a named actor - a thing that happened, was done,
  was said, was decided, or a number/state at a specific time or place. Set it false when the claim is
  instead a general condition or trend, a causal generality about what controls or causes an outcome, a
  sweeping generalisation about what a whole group is, wants or is doing, or an attribution of a belief,
  stance, motive or intent to a person rather than a concrete act they took. A specific event stated in
  inflammatory or partisan words is still specific; a general characterisation dressed as a declarative
  sentence is not.
- stands_alone: true when a stranger with a search engine and no access to the post could find and rate
  this claim true or false as written - the central person, place and event are named. A claim whose"""
_V4_KEEP_ANCHOR = "is_contested_key_claim is true, stands_alone is true, political is true, is_satire_or_sarcasm is false,\nis_trivial is false, and is_opinion_or_characterisation is false."
_V5_KEEP = "is_contested_key_claim is true, stands_alone is true, political is true, is_specific_event_or_fact is\ntrue, is_satire_or_sarcasm is false, is_trivial is false, and is_opinion_or_characterisation is false."
_V4_CODES = "not_asserted_by_post, not_contested, not_political, satire, trivial, opinion, or does_not_stand_alone."
_V5_CODES = "not_asserted_by_post, not_contested, not_political, not_specific, satire, trivial, opinion, or\ndoes_not_stand_alone."
_V4_SCHEMA = '  "is_trivial": bool, "is_opinion_or_characterisation": bool, "stands_alone": bool,'
_V5_SCHEMA = '  "is_trivial": bool, "is_opinion_or_characterisation": bool, "is_specific_event_or_fact": bool,\n  "stands_alone": bool,'
for _s in (_V4_OPIN_ANCHOR, _V4_KEEP_ANCHOR, _V4_CODES, _V4_SCHEMA):
    assert CLAIM_CHECK_SYS_V4.count(_s) == 1, _s[:40]
CLAIM_CHECK_SYS_V5 = (CLAIM_CHECK_SYS_V4
                      .replace(_V4_OPIN_ANCHOR, _V5_SPECIFIC)
                      .replace(_V4_KEEP_ANCHOR, _V5_KEEP)
                      .replace(_V4_CODES, _V5_CODES)
                      .replace(_V4_SCHEMA, _V5_SCHEMA))
PROMPT_V5 = "claim-check-v5"

PROMPTS = {PROMPT_V1: CLAIM_CHECK_SYS, PROMPT_V2: CLAIM_CHECK_SYS_V2, PROMPT_V3: CLAIM_CHECK_SYS_V3,
           PROMPT_V4: CLAIM_CHECK_SYS_V4, PROMPT_V5: CLAIM_CHECK_SYS_V5}
PROMPT = PROMPT_V5

# political-gate-v1 (Daniel 2026-09-22): a dedicated STRICT political scope gate, run as its own mode
# (--mode political-gate) AFTER claim-check-v5. The honest read of the v5-selected pool showed the
# residual not-clean was dominated by NON-POLITICAL posts (crypto, celebrity, sport, tech hype) that
# v5's broad political flag let through - a scope-recall problem, not a cleanliness one. v5's political
# flag is one field among many keep conditions and the model over-tags it. This gate asks ONE thing per
# claim - is the claim's SUBJECT politics/public affairs, and how squarely - reading claim + post +
# author bio, and returns a 3-level strength so only CORE political claims are kept. Abstract principles
# only, no dataset examples; it judges subject matter, never truth, tone, or claim quality (those are
# claim-check's job). Output schema differs from claim-check: {political, subtopic, strength}.
POLGATE_SYS = """\
You are a strict scope gate for a survey of POLITICAL claims. An earlier step extracted a candidate
factual claim from a social media post. Your only job is to decide whether that claim belongs to
politics and public affairs, and how squarely it sits there. You are NOT judging whether the claim is
true, well-formed, contested, or interesting - only what its subject matter is.

You are given the post's text (and any quoted post), the author's profile bio, a Community Note as
context, and the candidate claim. Use the post and the bio to see what the claim is really about. The
note is context only; ignore its verdict entirely.

A claim is POLITICAL when its substance concerns the conduct of public life: government and its
branches and agencies, public policy and legislation, elections, campaigns and political parties,
public officials or candidates acting in their official or public capacity, the courts, law
enforcement, the military or intelligence services treated as a matter of public policy or state
conduct, international relations, diplomacy, war and armed conflict, or a contested public-policy
issue such as immigration, economic and fiscal policy, health policy, or culture-war disputes over
legislation and public institutions.

A claim is NOT POLITICAL when its substance is celebrity, entertainment or the arts, sport, personal
lives and relationships, crypto, markets, stocks or the movement of asset prices, consumer or
technology products and company launches, ordinary business and corporate news, an ordinary crime
with no policy or official-conduct dimension, a human-interest or personal story, weather or natural
events, science or health with no policy dimension, or viral internet chatter. A claim does not
become political merely because a famous person, a large sum of money, a public institution's name,
or a strong public reaction appears in it, nor because it could in principle be discussed in
political terms. It is political only when the action of a government, of an officeholder in their
public role, or of a public policy is central to what the claim actually asserts.

Judge the SUBJECT of the claim, not its tone. An inflammatory or partisan-sounding claim about a
non-political subject is still not political; a plainly-worded claim about a government act is
political.

For the claim, return three fields:
- political: true when the claim is political by the standard above, false otherwise.
- subtopic: the single best-fitting label from - immigration; economy; foreign_affairs;
  society_and_culture (identity, religion, race, gender, culture-war, social issues);
  elections_and_government (elections, voting, officials, government, legislation); crime_and_justice;
  health_and_science; other_political (any other political matter). If the claim is not political,
  still choose the label closest to its surface topic.
- strength: exactly one of - "core" when politics or public policy is the centre of the claim and the
  action of a government, an official in their public role, or a public policy is what the claim
  asserts; "marginal" when there is a genuine but secondary or incidental political angle, or the
  political dimension depends on context the claim itself does not carry; "not" when the claim is not
  about politics or public affairs at all.

Reply with JSON only, one object per candidate claim in the order given:
{"claims": [{"political": bool, "subtopic": "<one label>", "strength": "core|marginal|not"}]}
"""
POLGATE_V1 = "political-gate-v1"
GATE_PROMPTS = {POLGATE_V1: POLGATE_SYS}

# survey-labels-v1 (Daniel 2026-09-22): one labelling pass over the selected survey pool, run as
# --mode survey-labels AFTER the gate. Per claim: which political side the claim serves if true
# (claim lean), the author's lean from bio + own posts, the country whose politics the claim
# concerns, and whether the post recycles an old real event as current news (the note usually
# says so). The 2x2 (left/right x tool-true/tool-false) is built on the CLAIM lean, the author
# lean is a check column. Abstract principles only; the note's verdict is never a label input
# except for the recycled-news question, which is exactly what notes report.
LABELS_SYS = """\
You label claims for a survey on political misinformation. An earlier step extracted one or two
factual claims from a social media post. You are given the post text (and any quoted post), the
author's profile bio, up to three other recent posts by the same author, a Community Note as
context, and the candidate claims. Return labels only; do not judge whether the claims are true.

For EACH claim return:
- claim_lean: "left", "right" or "neither". Ask which political side would want this claim to be
  true, or whose narrative it serves when believed. Judge the claim's content in the post's
  framing, not the author's identity. A claim that flatters or attacks a party, leader, policy or
  movement leans toward the side it helps. Use "neither" when the claim serves no side, when both
  sides could equally embrace it, or when the country's politics does not map onto a left/right
  axis. Use the conventions of the country the claim concerns (in the United States, Democrats and
  progressives are left, Republicans and MAGA are right; in the United Kingdom, Labour and the
  Greens are left, Conservatives and Reform are right; and so on).
- claim_lean_confidence: "high", "medium" or "low".
- country: the country whose politics or public affairs the claim primarily concerns, as a short
  common English name ("United States", "United Kingdom", "India", "Israel"), or "international"
  when it concerns relations between several countries with no primary one.
- recycled_news: true when the post presents as current or breaking an event that actually
  happened well before the post date (months or years earlier), as the note or the post's own
  dates indicate; false otherwise or when unknown.

Return ONCE for the post:
- author_lean: "left", "right" or "neutral". Infer from the bio and the author's own posts, not from
  the single claim; "neutral" when the author is a news organisation, an institution, or gives no
  usable signal.
- author_lean_confidence: "high", "medium" or "low".

Reply with JSON only:
{"author_lean": "left|right|neutral", "author_lean_confidence": "high|medium|low",
 "claims": [{"claim_lean": "left|right|neither", "claim_lean_confidence": "high|medium|low",
             "country": "<name>", "recycled_news": bool}]}
"""
LABELS_V1 = "survey-labels-v1"
LABEL_PROMPTS = {LABELS_V1: LABELS_SYS}
_LEANS = ("left", "right", "neither")
_CONF = ("high", "medium", "low")

_FLAGS = ["asserted_by_post", "is_contested_key_claim", "is_satire_or_sarcasm", "is_trivial",
          "is_opinion_or_characterisation", "is_specific_event_or_fact", "stands_alone",
          "keeps_contested_wording"]


def _norm(v, claim_text):
    keep = bool(v.get("keep"))
    out = {"claim": claim_text}
    for f in _FLAGS:
        out[f] = bool(v.get(f))
    out["keep"] = keep
    out["drop_reason"] = "" if keep else (v.get("drop_reason") or "")[:40]
    out["subtopic"] = v.get("subtopic") if v.get("subtopic") in SUBTOPICS else "other_political"
    out["political"] = bool(v.get("political"))
    return out


def _norm_gate(v, claim_text):
    strength = v.get("strength") if v.get("strength") in ("core", "marginal", "not") else "not"
    return {"claim": claim_text, "political": bool(v.get("political")),
            "subtopic": v.get("subtopic") if v.get("subtopic") in SUBTOPICS else "other_political",
            "strength": strength, "keep": strength == "core",
            "drop_reason": "" if strength == "core" else f"not_core_{strength}"}


def _norm_labels(v, claim_text, head):
    return {"claim": claim_text,
            "claim_lean": v.get("claim_lean") if v.get("claim_lean") in _LEANS else "neither",
            "claim_lean_confidence": v.get("claim_lean_confidence") if v.get("claim_lean_confidence") in _CONF else "low",
            "country": str(v.get("country") or "unknown")[:40],
            "recycled_news": bool(v.get("recycled_news")),
            "author_lean": head.get("author_lean") if head.get("author_lean") in ("left", "right", "neutral") else "neutral",
            "author_lean_confidence": head.get("author_lean_confidence") if head.get("author_lean_confidence") in _CONF else "low",
            "keep": True, "drop_reason": ""}


def _txt(v) -> str:
    """Post fields may arrive as None or a pandas NaN; both mean no text."""
    return "" if v is None or isinstance(v, float) else str(v)


def build_request(post, claims, sys_prompt, gate=False, labels=False, temperature=0.0):
    """One chat/completions body per post; the shared deepinfra_runner sends it."""
    user = f"OUTLET: @{_txt(post.get('handle'))}\nPOST DATE: {_txt(post.get('created_at')) or 'unknown'}"
    if (gate or labels) and _txt(post.get("author_description")).strip():
        user += f"\nAUTHOR BIO: {_txt(post['author_description'])}"
    user += f"\nPOST:\n{_txt(post.get('post_text'))}"
    if _txt(post.get("quoted_text")):
        user += f"\n\nQUOTED POST by @{_txt(post.get('quoted_handle')) or 'unknown'}:\n{_txt(post['quoted_text'])}"
    if labels and post.get("author_other_posts"):
        user += "\n\nOTHER RECENT POSTS BY THE SAME AUTHOR:\n" + "\n".join(f"- {t}" for t in post["author_other_posts"])
    if _txt(post.get("note")).strip():
        user += f"\n\nCOMMUNITY NOTE (context only, never a truth verdict for keep/drop):\n{_txt(post['note'])}"
    numbered = "\n".join(f"{i}. {c['claim']}" for i, c in enumerate(claims, 1))
    user += f"\n\nCANDIDATE CLAIMS:\n{numbered}"
    return {"temperature": temperature, "max_tokens": 900, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]}


def parse_response(j, claims, gate=False, labels=False):
    obj = _parse_json(j["choices"][0]["message"].get("content"))
    raw = obj.get("claims") if isinstance(obj.get("claims"), list) else []
    res = []
    for i, c in enumerate(claims):
        v = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
        res.append(_norm_labels(v, c["claim"], obj) if labels else
                   _norm_gate(v, c["claim"]) if gate else _norm(v, c["claim"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True, help="posts json (text/note/quoted), same contract as extract")
    ap.add_argument("--claims", required=True, help="key_claim_extract output json (posts dict with claims)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--provider", default="deepinfra", choices=sorted(PROVIDERS))
    ap.add_argument("--workers", type=int, default=None, help="None = the runner's per-model default")
    ap.add_argument("--max-spend", type=float, default=None, help="hard $ cap for this run")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = deterministic; >0 for extra consensus votes (the cache key covers the body, so each temperature is a fresh run)")
    ap.add_argument("--vote", type=int, default=0, help="tag for a consensus vote; only changes the cache key so repeated votes at one temperature stay distinct")
    ap.add_argument("--prompt", default=PROMPT, choices=sorted(PROMPTS))
    ap.add_argument("--mode", default="claim-check", choices=["claim-check", "political-gate", "survey-labels"],
                    help="claim-check runs a --prompt; political-gate runs the strict scope gate "
                         "(--gate-prompt), keeping only strength=core claims; survey-labels labels "
                         "claim lean, author lean, country and recycled news (--labels-prompt).")
    ap.add_argument("--gate-prompt", default=POLGATE_V1, choices=sorted(GATE_PROMPTS))
    ap.add_argument("--labels-prompt", default=LABELS_V1, choices=sorted(LABEL_PROMPTS))
    a = ap.parse_args()
    PROVIDER[0] = a.provider
    gate = a.mode == "political-gate"
    labels = a.mode == "survey-labels"
    sys_prompt = LABEL_PROMPTS[a.labels_prompt] if labels else GATE_PROMPTS[a.gate_prompt] if gate else PROMPTS[a.prompt]
    tag = a.labels_prompt if labels else a.gate_prompt if gate else a.prompt
    posts = {str(p["post_id"]): p for p in json.load(open(a.posts))["posts"]}
    extract = json.load(open(a.claims))["posts"]
    jobs = []
    for pid, rec in extract.items():
        claims = rec.get("claims") or []
        if claims and str(pid) in posts:
            jobs.append((str(pid), claims))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"posts-with-claims {len(jobs)}  mode {a.mode}  prompt {tag} ({prompt_hash(sys_prompt)})  model {a.model} @ {a.provider}", flush=True)
    jobs_by_pid = {pid: claims for pid, claims in jobs}
    rr = run(jobs, lambda job: build_request(posts[job[0]], job[1], sys_prompt, gate=gate, labels=labels, temperature=a.temperature),
             model=a.model, workers=a.workers, cache_path=str(out) + ".cache.jsonl",
             item_id=lambda job: f"{prompt_hash(sys_prompt)}|v{a.vote}|{job[0]}|{hashlib.sha256(json.dumps([c['claim'] for c in job[1]], ensure_ascii=False).encode()).hexdigest()[:16]}",
             parse=lambda j, job: parse_response(j, job[1], gate=gate, labels=labels),
             max_spend=a.max_spend)
    print(rr.summary(), flush=True)
    spent = [rr.spent]
    results = {}
    for job in jobs:
        pid, claims = job
        key = f"{prompt_hash(sys_prompt)}|v{a.vote}|{pid}|{hashlib.sha256(json.dumps([c['claim'] for c in claims], ensure_ascii=False).encode()).hexdigest()[:16]}"
        r = rr.results.get(key)
        if r is None:
            err = str(rr.errors.get(key, "no result"))
            r = ([{**_norm_labels({}, cl["claim"], {}), "keep": False, "drop_reason": f"failed {err[:60]}"} for cl in claims]
                 if labels else
                 [{"claim": cl["claim"], "keep": False, "drop_reason": f"failed {err[:60]}",
                   "political": False, "subtopic": "other_political", "strength": "not"} for cl in claims]
                 if gate else
                 [{"claim": cl["claim"], "keep": False, "drop_reason": f"failed {err[:60]}",
                   "asserted_by_post": False, "is_contested_key_claim": False, "is_satire_or_sarcasm": False,
                   "is_trivial": False, "is_opinion_or_characterisation": False, "stands_alone": False,
                   "keeps_contested_wording": False, "subtopic": "other_political", "political": False} for cl in claims])
        results[pid] = {"claims": r}
    survivors = []
    for pid, rec in results.items():
        for cc in rec["claims"]:
            if cc["keep"]:
                survivors.append({"post_id": pid, **cc})
    json.dump({"prompt": tag, "mode": a.mode, "prompt_hash": prompt_hash(sys_prompt), "model": a.model,
               "system_prompt": sys_prompt, "posts": results, "survivors": survivors},
              open(out, "w"), indent=0, ensure_ascii=False)
    ncl = sum(len(v["claims"]) for v in results.values())
    print(f"spent ${spent[0]:.3f}  posts {len(results)}  claims-in {ncl}  survivors {len(survivors)}  "
          f"dropped {ncl - len(survivors)}  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
