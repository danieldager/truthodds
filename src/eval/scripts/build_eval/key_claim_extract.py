"""Key-claim extraction (branch reader-iteration, Daniel 2026-09-10): ONE pass from post
to the few self-contained claims a fact-checker would check, replacing for this
experiment the extract -> normalize -> checkworthy chain that produced 4.3 claims per
post, most of them peripheral fragments with unresolved referents ("the cold case
involves a father-of-three", "the latest child victim managed to survive").

    uv run python -m eval.scripts.build_eval.key_claim_extract \
        --posts eval/data/reader_lab/inputs/outlet_posts_all.json \
        --dates eval/data/reader_lab/inputs/e2_rescored.parquet \
        --out eval/data/reader_lab/extract/keyclaims_v1.json [--model openai/gpt-oss-120b] [--provider groq]

Production prompt is extract-key-v5 (the --prompt default); --model deepseek-ai/DeepSeek-V4-Flash
@ deepinfra, temperature 0, non-thinking, text-only.

Output: {"prompt": ..., "posts": {post_id: {"claims": [{"claim", "type", "why"}], "none_why"}},
         "claims": [{claim_id, post_id, claim, type, domain, handle, ng_score, lean, created_at, post_text}]}
The flat "claims" list is the input contract for the re-query run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.prompt_hash import prompt_hash  # noqa: E402
from eval.scripts.build_eval import evidence_urn_run as eur  # noqa: E402
from eval.scripts.build_eval.reader_lab import PROVIDER, PROVIDERS, _endpoint, _parse_json  # noqa: E402

EXTRACT_SYS = """\
You are a fact-checker's editor reading one social media post from a news outlet.
Write the claims a professional fact-checker would check: the post's load-bearing
assertions, the ones the post exists to make and whose falsity would make the post
misleading. At most three. Often one. None if the post makes no checkable central
assertion.

Each claim must stand on its own. A stranger with a search engine and no access to
the post must know exactly which event, person, place, ruling, figure or statement
is meant. Resolve every reference from the post: name the people and organisations,
name the place, give the date or period when the post gives one. Never leave "the
case", "the victim", "the strike", "the bill", "he", "they" unresolved. If the post
does not contain what is needed to resolve a reference, the claim is not checkable
from this post. Leave it out.

Write each claim as one plain literal proposition in neutral words. Keep the post's
own specifics (numbers, names, quotes). Drop the post's evaluations, sarcasm,
adjectives and framing. Say what happened or what is the case, not how the post
feels about it.

Do not write: scene setting and side facts; opinion, prediction, characterisation,
or a headline's thesis whose load-bearing words are judgments; the same proposition
twice; claims about the post's own wording; promotion of the outlet. Test each
candidate: if its decisive word is an evaluation (betrayed, notorious, disturbing,
suppression, demagogue, ready to help, failed the public) no document can settle
it. Either rewrite it to the observable fact the post states, or leave it out.
Readiness, intentions and warnings are not claims unless the post states a
concrete action.

Rhetoric is not a claim. Hyperbole, sweeping quantifiers used for emphasis (nobody,
everyone, never, all, no one), superlatives, verdicts on a story or a narrative,
counterfactuals about what would have happened, and assessments of mood, morale,
popularity, momentum, or of how much something is discussed, noticed, covered or
believed, are not claims even when phrased as fact. Test each candidate: if settling
it would mean deciding a matter of degree, emphasis or taste rather than whether an
event, figure or statement is real, it is rhetoric. If such a sentence also carries a
checkable specific (a number, a named event, a named action), write only that
specific. Otherwise leave it out, however central it is to the post. Also leave out
any claim that only says something happened, emerged, was revealed or was discussed
without saying what.

Reported speech. When the post's point is what someone did, said or claimed, write
the content as the claim and mark it "attribution" only when who said it is itself
the news (a named official, a court, an organisation making a statement). When the
post merely relays a named speaker's factual assertion about the world (an event, a
figure, an action) as its own point, write the assertion as the claim, marked
"assertion", and note the speaker in "why". A speaker's judgment, assessment,
prediction or rhetoric is never lifted out as an assertion: keep it with the speaker
as an "attribution" when who said it is the news, otherwise leave it out. An
"attribution" claim names the speaker inside the claim text itself.

Reply with JSON only:
{"claims": [{"claim": "<self-contained proposition>", "type": "assertion" | "attribution", "why": "<why a fact-checker would check this, one clause>"}], "none_why": "<only when claims is empty: one clause>"}
"""
PROMPT_V = "extract-key-v3"

EXTRACT_SYS_V4 = """\
You are a fact-checker's editor reading one social media post from a news outlet.
Write the claims a professional fact-checker would check: the post's load-bearing
assertions, the ones the post exists to make and whose falsity would make the post
misleading. At most three. Often one. None if the post makes no checkable central
assertion.

Each claim must stand on its own. A stranger with a search engine and no access to
the post must know exactly which event, person, place, ruling, figure or statement
is meant. Resolve every reference from the post: name the people and organisations,
name the place, give the date or period when the post gives one. Never leave "the
case", "the victim", "the strike", "the bill", "he", "they" unresolved. If the post
does not contain what is needed to resolve a reference, the claim is not checkable
from this post. Leave it out.

Write each claim as one plain literal proposition in neutral words. Keep the post's
own specifics (numbers, names, quotes). Drop the post's evaluations, sarcasm,
adjectives and framing. Say what happened or what is the case, not how the post
feels about it.

Take nothing from outside the post. Every name, office, title, body, place, contest,
date, period and figure in a claim must be given by the post, or follow from the post
date. Spell a name exactly as the post spells it. Do not supply the office a person
holds, the race they are in, the chamber or agency that acted, or the day something
happened when the post does not say it, and do not shift a relative time the post
gives into a different one. A claim that needs a specific the post never states is
not checkable from this post: leave it out. It is better to write a shorter claim
than a fuller one you had to complete yourself.

Read the post literally, and check that a literal reading is available. A post often
credits an action to a figure who stands for a side, a party or a campaign rather
than to the person who performed it, and writes a contest, a defeat or a victory in
the same borrowed way. Before writing such a sentence as a claim, ask whether the
named actor could have done the thing as written, in that role, on that occasion. If
not, the wording is figurative: write the literal fact the post states instead, or
leave it out. Never write a claim whose falsity would turn on the post's phrasing
rather than on the world.

One claim per assertion. When the parts of a sentence belong to a single assertion —
a figure and the comparison that grades it, an action and the purpose given for it, a
quoted passage and the conditions it sets out, an event and the circumstances it is
reported with — write them as ONE claim. Splitting one assertion into several leaves
fragments that no document addresses on its own and that no longer carry the post's
point. Separate claims are for separate assertions the post makes.

Prominence. Extract only what the post is for. A sentence is not a claim merely
because it is checkable. Background from an earlier period offered for contrast, a
figure restated or recomputed from something already claimed, an actor's schedule,
travel or itinerary, the identity of whoever relayed, first reported or pointed out a
fact, and the outlet's own sourcing of its story are all setting. If the post would
still make its point with the sentence deleted, leave it out. Prefer the assertion
the post leads with over anything it mentions in passing.

Do not write: scene setting and side facts; opinion, prediction, characterisation,
or a headline's thesis whose load-bearing words are judgments; the same proposition
twice; claims about the post's own wording; promotion of the outlet. Test each
candidate: if its decisive word is an evaluation (betrayed, notorious, disturbing,
suppression, demagogue, ready to help, failed the public) no document can settle
it. Either rewrite it to the observable fact the post states, or leave it out.
Readiness, intentions and warnings are not claims unless the post states a
concrete action.

Rhetoric is not a claim. Hyperbole, sweeping quantifiers used for emphasis (nobody,
everyone, never, all, no one), superlatives, verdicts on a story or a narrative,
counterfactuals about what would have happened, and assessments of mood, morale,
popularity, momentum, or of how much something is discussed, noticed, covered or
believed, are not claims even when phrased as fact. Test each candidate: if settling
it would mean deciding a matter of degree, emphasis or taste rather than whether an
event, figure or statement is real, it is rhetoric. If such a sentence also carries a
checkable specific (a number, a named event, a named action), write only that
specific. Otherwise leave it out, however central it is to the post. Also leave out
any claim that only says something happened, emerged, was revealed or was discussed
without saying what. A quantifier or a measure of scale that the post applies to a
named action, rule, figure or programme is part of that claim and is kept with it:
what this paragraph excludes is the sentence that has nothing else in it.

Reported speech. When the post's point is what someone did, said or claimed, write
the content as the claim and mark it "attribution" only when who said it is itself
the news (a named official, a court, an organisation making a statement). When the
post merely relays a named speaker's factual assertion about the world (an event, a
figure, an action) as its own point, write the assertion as the claim, marked
"assertion", and note the speaker in "why". A speaker's judgment, assessment,
prediction or rhetoric is never lifted out as an assertion: keep it with the speaker
as an "attribution" when who said it is the news, otherwise leave it out. An
"attribution" claim names the speaker inside the claim text itself.

Who is speaking decides what may be lifted. Only a person or body acting in an
official or professional role, speaking about their own sphere, may have a factual
assertion lifted out of their words and written as a plain fact about the world. When
the post's factual content comes from a guest, a commentator, an anonymous or
personal account, a caller or a bystander the post is relaying, the claim is that
this person said it — write it with the speaker, or leave it out — because nothing
outside the post makes the bare content theirs to assert. Words performed to
entertain — a joke, a bit, a routine, a sketch by a comedian or entertainer — are not
claims at all, however factual their grammar, and neither is the content inside them.

Reply with JSON only:
{"claims": [{"claim": "<self-contained proposition>", "type": "assertion" | "attribution", "why": "<why a fact-checker would check this, one clause>"}], "none_why": "<only when claims is empty: one clause>"}
"""
PROMPT_V4 = "extract-key-v4"

# extract-key-v5 (Daniel 2026-09-14): v3 verbatim plus exactly two abstract principles
# that the 53-post audit found worked without regression — (i) prominence + fold, and
# (ii) never promote the relayer over the fact. Built from EXTRACT_SYS by insertion so
# the v3 base stays byte-identical; nothing else changes.
_V5_PRINCIPLES = """\
The decisive claim is the most consequential checkable proposition in the post: the
one the post exists to assert and whose falsity would most mislead. A background,
temporal, arithmetic or framing sub-clause that the post treats as already given is not
itself the claim, however checkable it looks on its own. When such a clause sits inside
a larger assertion, fold it into that substantive claim rather than promoting it to a
claim of its own; do not lead with a detail the post mentions only in passing.

Never promote the clause that says who relayed, reported or first said a fact over the
fact itself. When the post credits a fact to whoever passed it along, the fact is the
claim and the relaying is not. This does not touch attribution as content: when who
said something is itself the news and what they said is the point, keep the speaker in
the claim as before.

"""
_V5_ANCHOR = "Do not write: scene setting and side facts;"
assert EXTRACT_SYS.count(_V5_ANCHOR) == 1
EXTRACT_SYS_V5 = EXTRACT_SYS.replace(_V5_ANCHOR, _V5_PRINCIPLES + _V5_ANCHOR)
PROMPT_V5 = "extract-key-v5"

# extract-key-v6 (Daniel 2026-09-21): rebuilt on principle from the 120-post v5 audit. v5
# emitted up to three claims and often led with the blandest checkable one; v6 names the SINGLE
# most-contested, most-consequential proposition (at most one fallback candidate), keeps the
# loaded/contested word, writes a figure's world-fact plainly and reserves attribution for when
# the saying is the news, and routes satire / pure opinion / prediction / anecdote / media-only
# deception to a no-claim outcome. No dataset examples; abstract principles only. v3/v4/v5 stay
# byte-identical.
EXTRACT_SYS_V6 = """\
You are a fact-checker's editor reading one social media post from a news outlet. Name the
single claim in the post most worth checking: the one contested, consequential proposition the
post exists to put across, the one whose falsity would most mislead a reader. Output that one
claim. You may add at most one alternative candidate, and only when a second, clearly distinct
assertion has a genuine competing case to be the one worth checking; never pad to reach two.
Output none when the post makes no such claim.

Choosing the one claim. Among everything the post asserts, the claim is the most contested and
most consequential: the assertion a reader would dispute, the reason the post would draw a
correction, the thing that is at stake. It is not the blandest checkable fact, not the
background a reader already grants, not a date, figure or side detail the post treats as settled
scaffolding for its real point. When a background, temporal, arithmetic or framing detail sits
inside a larger assertion, fold it into that assertion rather than lifting it out as the claim.
Do not lead with a detail the post mentions only in passing while the disputed proposition goes
unwritten. Prefer the proposition the post is built to make over anything that is merely also
true.

Keep the contested word. The word a claim turns on is often the loaded one: a nationality, an
identity, a religion, a cause, a motive, a quantity, a decisive factual descriptor. That word is
the point of the claim and the thing a reader would check. Keep it exactly as the post commits
to it. Do not soften a loaded specific into a neutral paraphrase, do not generalise a named
group or person into a vaguer one, and do not drop the number. Blanding the contested word away
leaves a claim nobody would dispute and misses what the post asserts. This does not license
keeping evaluation or framing; it means never discarding the substantive specific the claim is
about.

Each claim stands on its own. A stranger with a search engine and no access to the post must
know exactly which event, person, place, ruling, figure or statement is meant. Resolve every
reference from the post: name the people and organisations, name the place, give the date or
period when the post gives one. Never leave "the case", "the victim", "the bill", "he", "they"
unresolved. Take nothing from outside the post: every name, office, place, date and figure must
be given by the post or follow from the post's date. Spell names as the post spells them. If the
post does not contain what is needed to resolve a reference, the claim is not checkable from this
post: leave it out.

Write it as a plain proposition. One literal sentence in neutral words. Keep the post's own
specifics; drop its evaluations, sarcasm, adjectives of judgment and framing. Say what happened
or what is the case, not how the post feels about it. Read the post literally and check that a
literal reading is available: when the post credits an action, a defeat or a victory to a figure
who stands for a side or a campaign rather than to the person who performed it, write the literal
fact the post states instead of the figurative wording.

Assertion versus attribution. When a person or body speaking in an official or professional role
asserts a fact about the world, write that fact plainly as the claim, marked "assertion" (note
the speaker in "why") - the world-fact, not that they said it - because the post's point is the
fact. Reserve an "attribution" claim, which names the speaker inside the claim text, for when the
saying itself is the news: a named official, court or organisation whose statement is the event,
or a quote whose authenticity or wording is what is contested. Never lift a bare factual
assertion out of a guest, commentator, anonymous account, caller or bystander the post is only
relaying: there the claim is that this person said it, or there is no claim. A speaker's judgment,
assessment or prediction is never lifted out as a fact.

What is never the claim. Do not write these, and return a no-claim outcome when the post has
nothing else:
- an interpretation of what an attached image or video shows, or any claim that rests on the
  media rather than on something the text itself asserts;
- a statement of intent, goal, plan, readiness, threat or warning, with no concrete action
  stated;
- a characterisation, opinion, prediction, or a headline thesis whose decisive word is a
  judgment, which no document can settle;
- rhetoric: hyperbole, sweeping quantifiers used for emphasis, superlatives, verdicts on a
  narrative, counterfactuals about what would have happened, and assessments of mood, momentum,
  popularity, or of how much a thing is discussed, noticed or believed;
- a sentence that only says something happened, emerged, was revealed or was discussed without
  saying what.
If settling a candidate would mean deciding a matter of degree, emphasis or taste rather than
whether an event, figure or statement is real, it is not a claim.

When to output none. Return no claim when the post is satire, a joke or a comedic bit (however
factual its grammar), pure opinion or framing, a bare prediction, a personal anecdote or
reaction, or when its deception lives only in an attached image or video and the text asserts
nothing checkable on its own. It is better to output none than to manufacture a claim the post
does not make.

Reply with JSON only:
{"claims": [{"claim": "<self-contained proposition>", "type": "assertion" | "attribution", "why": "<why a fact-checker would check this, one clause>"}], "none_why": "<only when claims is empty: one clause>"}
"""
PROMPT_V6 = "extract-key-v6"

# extract-key-v6-note (Daniel 2026-09-21): v6 plus a Community Note supplied to the model as a
# POINTER to which of the post's own assertions is contested. Guardrails are abstract principles,
# not example-driven: the claim stays the post's own assertion in the post's words, the note's
# verdict/correction/figures never leak into the wording, the note only guides selection (it can
# be wrong or target a side issue), and a note that only impugns the media does not create a text
# claim. Built by insertion so the v6 body stays byte-identical.
_V6_NOTE_BLOCK = """\
A Community Note is provided below as context. Use it only as a pointer to which of the post's
own assertions is the contested one; the note flags what readers dispute. These rules bind you:
- The claim must be something the POST asserts, worded from the post's side, as if the note did
  not exist. Never write the note's correction, its verdict or its figures. Do not import words
  such as "falsely", "misleadingly" or "no evidence", and do not replace the post's own number,
  name, nationality or cause with the note's corrected one.
- The note only guides selection among the post's assertions. A note can be wrong, or can address
  a side issue rather than the post's main point; when it points at something the post does not
  actually assert, ignore it and choose from what the post does assert.
- If the note only says the attached image or video is fake, edited or out of context, and the
  post's text asserts nothing checkable on its own, output none. A note about the media does not
  create a text claim.

"""
_V6_ANCHOR = "Reply with JSON only:"
assert EXTRACT_SYS_V6.count(_V6_ANCHOR) == 1
EXTRACT_SYS_V6_NOTE = EXTRACT_SYS_V6.replace(_V6_ANCHOR, _V6_NOTE_BLOCK + _V6_ANCHOR)
PROMPT_V6_NOTE = "extract-key-v6-note"

# extract-key-v7 (Daniel 2026-09-21): one principled iteration on v6 after the 120+picks hand-judge.
# v6 was MORE polarised than v5 (more clean captures, fewer partials, but MORE misses): it over-abstained
# on inflammatory-but-checkable posts, still led with the first/blander of two parallel claims, and
# sometimes lifted a stance. v7 keeps v6's frame and changes four things on principle (no examples):
# (a) abstain only when the text carries no checkable assertion at all; (b) among several claims lead
# with the most disputed/specific/surprising; (c) extract the concrete fact underneath a stance;
# (d) treat a quoted post's assertion as the claim when the host text only reacts.
EXTRACT_SYS_V7 = """\
You are a fact-checker's editor reading one social media post from a news outlet. Name the
single claim in the post most worth checking: the one contested, consequential proposition the
post exists to put across, the one whose falsity would most mislead a reader. Output that one
claim. You may add at most one alternative candidate, and only when a second, clearly distinct
assertion has a genuine competing case to be the one worth checking; never pad to reach two.
Output none only when the post makes no checkable assertion at all.

Choosing the one claim. Among everything the post asserts, the claim is the most contested and
most consequential: the assertion a reader would dispute, the reason the post would draw a
correction, the thing that is at stake. It is not the blandest checkable fact, not the
background a reader already grants, not a date, figure or side detail the post treats as settled
scaffolding for its real point. When the post makes two or more checkable assertions, lead with
the one whose falsity would most mislead: usually the most specific, the most surprising, or the
one a correction would target - not the first one stated, not the more familiar or expected of
them, not the vaguer one. When a background, temporal, arithmetic or framing detail sits inside a
larger assertion, fold it into that assertion rather than lifting it out as the claim. Do not lead
with a detail the post mentions only in passing while the disputed proposition goes unwritten.

Keep the contested word. The word a claim turns on is often the loaded one: a nationality, an
identity, a religion, a cause, a motive, a quantity, a decisive factual descriptor. That word is
the point of the claim and the thing a reader would check. Keep it exactly as the post commits
to it. Do not soften a loaded specific into a neutral paraphrase, do not generalise a named
group or person into a vaguer one, and do not drop the number. Blanding the contested word away
leaves a claim nobody would dispute and misses what the post asserts. This does not license
keeping evaluation or framing; it means never discarding the substantive specific the claim is
about.

Each claim stands on its own. A stranger with a search engine and no access to the post must
know exactly which event, person, place, ruling, figure or statement is meant. Resolve every
reference from the post: name the people and organisations, name the place, give the date or
period when the post gives one. Never leave "the case", "the victim", "the bill", "he", "they"
unresolved. Take nothing from outside the post: every name, office, place, date and figure must
be given by the post or follow from the post's date. Spell names as the post spells them. If the
post does not contain what is needed to resolve a reference, the claim is not checkable from this
post: leave it out.

The quoted post counts. When a quoted or embedded post is shown to you, it is part of what this
post puts across. If the host text only reacts, comments or asks a question and the checkable
assertion lives in the quoted post, take that assertion as the claim.

Write it as a plain proposition. One literal sentence in neutral words. Keep the post's own
specifics; drop its evaluations, sarcasm, adjectives of judgment and framing. Say what happened
or what is the case, not how the post feels about it. Read the post literally and check that a
literal reading is available: when the post credits an action, a defeat or a victory to a figure
who stands for a side or a campaign rather than to the person who performed it, write the literal
fact the post states instead of the figurative wording.

Stance versus the fact under it. A verdict on a person's guilt, competence, motive or record - that
someone was unjustly treated, is a failure or a success, betrayed or saved - is a stance, and no
document settles it. But a stance usually rests on a concrete fact the post also states: who caused
a death, how a vote fell, what an official did. Extract that concrete fact as the claim, not the
verdict wrapped around it. If there is no such fact, leave it out.

Assertion versus attribution. When a person or body speaking in an official or professional role
asserts a fact about the world, write that fact plainly as the claim, marked "assertion" (note
the speaker in "why") - the world-fact, not that they said it - because the post's point is the
fact. Reserve an "attribution" claim, which names the speaker inside the claim text, for when the
saying itself is the news: a named official, court or organisation whose statement is the event,
or a quote whose authenticity or wording is what is contested. Never lift a bare factual
assertion out of a guest, commentator, anonymous account, caller or bystander the post is only
relaying: there the claim is that this person said it, or there is no claim. A speaker's judgment,
assessment or prediction is never lifted out as a fact.

What is never the claim. Do not write these, and return a no-claim outcome only when the post has
nothing checkable besides:
- an interpretation of what an attached image or video shows, or any claim that rests on the
  media rather than on something the text itself asserts;
- a statement of intent, goal, plan, readiness, threat or warning, with no concrete action
  stated;
- a characterisation, opinion, prediction, or a headline thesis whose decisive word is a
  judgment, which no document can settle;
- rhetoric with nothing else in it: hyperbole, sweeping quantifiers used only for emphasis,
  superlatives, verdicts on a narrative, counterfactuals about what would have happened, and
  assessments of mood, momentum, popularity, or of how much a thing is discussed or believed;
- a sentence that only says something happened, emerged, was revealed or was discussed without
  saying what.

When to output none. Abstain only when nothing in the text can be checked. A post may be a
reaction, a rhetorical question, an insult, or a sweeping and one-sided generalisation and still
assert a checkable event, act or quantity: when it names a specific one, that specific assertion
is the claim, however inflammatory or partisan the wrapping around it, and a rhetorical question
that presupposes a specific event asserts that event. Treat a post as satire or a joke, and output
none, only when it is performed to entertain and asserts nothing it means literally; vivid,
extreme or mocking phrasing is not by itself satire. Output none as well for a post that is pure
opinion, framing or a stated goal with no concrete fact under it, a bare prediction, a personal
anecdote or reaction with nothing of public consequence, or one whose deception lives only in an
attached image or video while the text asserts nothing checkable. It is better to output none
than to manufacture a claim the post does not make - but it is a failure to output none when the
text does assert a checkable specific.

Reply with JSON only:
{"claims": [{"claim": "<self-contained proposition>", "type": "assertion" | "attribution", "why": "<why a fact-checker would check this, one clause>"}], "none_why": "<only when claims is empty: one clause>"}
"""
PROMPT_V7 = "extract-key-v7"

# extract-key-v7-note (Daniel 2026-09-21): v7 plus the Community Note as a pointer, with the v6-note
# guardrails PLUS one added on principle after round 1 leaked note-only facts (a subject's team, a
# location, a date, the note's correction) into the claim: name nothing that appears only in the note.
_V7_NOTE_BLOCK = """\
A Community Note is provided below as context. Use it only as a pointer to which of the post's
own assertions is the contested one; the note flags what readers dispute. These rules bind you:
- The claim must be something the POST asserts, worded from the post's side, as if the note did
  not exist. Never write the note's correction, its verdict or its figures. Do not import words
  such as "falsely", "misleadingly" or "no evidence", and do not replace the post's own number,
  name, nationality or cause with the note's corrected one.
- Name nothing that appears only in the note. If the note mentions a fact the post does not
  state - a person's role or team, a place, a date, a background detail, a correction - do not put
  it in the claim. The note points at which assertion is contested; it never supplies content.
- The note only guides selection among the post's assertions. A note can be wrong, or can address
  a side issue rather than the post's main point; when it points at something the post does not
  actually assert, ignore it and choose from what the post does assert.
- If the note only says the attached image or video is fake, edited or out of context, and the
  post's text asserts nothing checkable on its own, output none. A note about the media does not
  create a text claim.

"""
assert EXTRACT_SYS_V7.count(_V6_ANCHOR) == 1
EXTRACT_SYS_V7_NOTE = EXTRACT_SYS_V7.replace(_V6_ANCHOR, _V7_NOTE_BLOCK + _V6_ANCHOR)
PROMPT_V7_NOTE = "extract-key-v7-note"

# extract-key-v8 (Daniel 2026-09-21, POST-ONLY): one principled iteration on v7 after the 146-post
# hand-judge, for a pipeline where a post-filter runs before extraction and a claim-check runs after.
# v7's residual picks-misses were all posts making TWO separate equally-contested claims where v7 output
# only one (Daniel's proposition was the other); v7 also read some true satire / low-stakes posts
# literally and stripped one contested identity descriptor. v8 changes three things on principle
# (no dataset examples): (a) emit BOTH claims when the post makes two separate, equally contested,
# consequential assertions - one otherwise, per Daniel's two-parallel-claim rule; (b) restore a firm
# satire and triviality no-claim outcome while keeping v7's rule that inflammatory-but-checkable text
# is a claim; (c) keep a nationality / identity / religion / role / affiliation descriptor attached to
# a named person, since it is a factual point and often the contested one. v3-v7 stay byte-identical.
EXTRACT_SYS_V8 = """\
You are a fact-checker's editor reading one social media post from a news outlet. Name the claim, or
the two claims, in the post most worth checking: the contested, consequential proposition the post
exists to put across, the one whose falsity would most mislead a reader. Usually the post has one such
claim - output that one. Output two claims only when the post makes two separate, equally contested,
consequential assertions that a reader would dispute independently and a fact-checker would check on
their own; when it does, output both and rank neither above the other. Never split one assertion into
two, and never pad to reach two. Output none when the post makes no checkable, consequential assertion
of its own.

Choosing the claim. Among everything the post asserts, the claim is the most contested and most
consequential: the assertion a reader would dispute, the reason the post would draw a correction, the
thing that is at stake. It is not the blandest checkable fact, not the background a reader already
grants, not a date, figure or side detail the post treats as settled scaffolding for its real point.
When the post makes several checkable assertions but one is plainly the point, lead with the one whose
falsity would most mislead: usually the most specific, the most surprising, or the one a correction
would target - not the first one stated, not the more familiar or expected of them, not the vaguer one.
Emit a second claim only when a second assertion is genuinely co-equal in dispute and consequence, not
merely also checkable. When a background, temporal, arithmetic or framing detail sits inside a larger
assertion, fold it into that assertion rather than lifting it out as the claim.

Keep the contested word. The word a claim turns on is often the loaded one: a nationality, an identity,
a religion, a cause, a motive, a quantity, a decisive factual descriptor. That word is the point of the
claim and the thing a reader would check. Keep it exactly as the post commits to it. Do not soften a
loaded specific into a neutral paraphrase, do not generalise a named group or person into a vaguer one,
and do not drop the number. A nationality, identity, religion, role or affiliation the post attaches to
a named person is a factual descriptor, not framing, and is frequently the contested point: keep it in
the claim even when it reads like a label. This does not license keeping evaluation or framing; it means
never discarding the substantive specific the claim is about.

Each claim stands on its own. A stranger with a search engine and no access to the post must know
exactly which event, person, place, ruling, figure or statement is meant. Resolve every reference from
the post: name the people and organisations, name the place, give the date or period when the post gives
one. Never leave "the case", "the victim", "the bill", "he", "they" unresolved. Take nothing from
outside the post: every name, office, place, date and figure must be given by the post or follow from
the post's date. Spell names as the post spells them. If the post does not contain what is needed to
resolve a reference, the claim is not checkable from this post: leave it out.

The quoted post counts. When a quoted or embedded post is shown to you, it is part of what this post
puts across. If the host text only reacts, comments or asks a question and the checkable assertion lives
in the quoted post, take that assertion as the claim.

Write it as a plain proposition. One literal sentence in neutral words. Keep the post's own specifics;
drop its evaluations, sarcasm, adjectives of judgment and framing. Say what happened or what is the
case, not how the post feels about it. Read the post literally and check that a literal reading is
available: when the post credits an action, a defeat or a victory to a figure who stands for a side or a
campaign rather than to the person who performed it, write the literal fact the post states instead of
the figurative wording.

Stance versus the fact under it. A verdict on a person's guilt, competence, motive or record - that
someone was unjustly treated, is a failure or a success, betrayed or saved - is a stance, and no
document settles it. But a stance usually rests on a concrete fact the post also states: who caused a
death, how a vote fell, what an official did. Extract that concrete fact as the claim, not the verdict
wrapped around it. If there is no such fact, leave it out.

Assertion versus attribution. When a person or body speaking in an official or professional role asserts
a fact about the world, write that fact plainly as the claim, marked "assertion" (note the speaker in
"why") - the world-fact, not that they said it - because the post's point is the fact. Reserve an
"attribution" claim, which names the speaker inside the claim text, for when the saying itself is the
news: a named official, court or organisation whose statement is the event, or a quote whose
authenticity or wording is what is contested. Never lift a bare factual assertion out of a guest,
commentator, anonymous account, caller or bystander the post is only relaying: there the claim is that
this person said it, or there is no claim. A speaker's judgment, assessment or prediction is never
lifted out as a fact.

What is never the claim. Do not write these:
- an interpretation of what an attached image or video shows, or any claim that rests on the media
  rather than on something the text itself asserts;
- a statement of intent, goal, plan, readiness, threat or warning, with no concrete action stated;
- a characterisation, opinion, prediction, or a headline thesis whose decisive word is a judgment,
  which no document can settle;
- rhetoric with nothing else in it: hyperbole, sweeping quantifiers used only for emphasis,
  superlatives, verdicts on a narrative, counterfactuals about what would have happened, and
  assessments of mood, momentum, popularity, or of how much a thing is discussed or believed;
- a sentence that only says something happened, emerged, was revealed or was discussed without saying
  what.

When to output none. Two different posts get a no-claim outcome, and it is important to tell them apart.
First, abstain when nothing in the text can be checked: a post may be a reaction, a rhetorical question,
an insult, or a sweeping one-sided generalisation and still assert a checkable event, act or quantity -
when it names a specific one, that specific assertion is the claim, however inflammatory or partisan the
wrapping, and a rhetorical question that presupposes a specific event asserts that event; output none
only when even that is absent. Second, abstain when the post is out of scope for checking at all: it is
satire, a joke, a parody or a comedic bit, performed to entertain and asserting nothing it means
literally; or it is trivial - a personal anecdote, an individual's private and low-stakes matter, or a
mundane everyday observation that no public consequence turns on. Vivid, extreme or mocking phrasing is
not by itself satire, and a claim about a public figure or public matter is not trivial; but when a post
really is entertainment or a private low-stakes matter, output none even if its grammar looks factual.
It is better to output none than to manufacture a claim the post does not make, and better to output
none on satire or triviality than to hand on a claim that should never be checked - but it is a failure
to output none when the text does assert a checkable, consequential specific.

Reply with JSON only:
{"claims": [{"claim": "<self-contained proposition>", "type": "assertion" | "attribution", "why": "<why a fact-checker would check this, one clause>"}], "none_why": "<only when claims is empty: one clause>"}
"""
PROMPT_V8 = "extract-key-v8"

PROMPTS = {PROMPT_V: EXTRACT_SYS, PROMPT_V4: EXTRACT_SYS_V4, PROMPT_V5: EXTRACT_SYS_V5,
           PROMPT_V6: EXTRACT_SYS_V6, PROMPT_V6_NOTE: EXTRACT_SYS_V6_NOTE,
           PROMPT_V7: EXTRACT_SYS_V7, PROMPT_V7_NOTE: EXTRACT_SYS_V7_NOTE,
           PROMPT_V8: EXTRACT_SYS_V8}

# PRODUCTION (Daniel 2026-09-16): extract-key-v5. Adopted on the 2026-09-14 five-draw test
# (claim-side FPs removed per draw 4.4 -> 7.6, no extraction-side true-false regression) and
# confirmed on the CN survey pool 2026-09-16. v3 and v4 stay byte-identical and selectable
# with --prompt; v4 was tested and rejected.
PROMPT_PRODUCTION = PROMPT_V5


def call(model, post, cache, reasoning="low", sys_prompt=EXTRACT_SYS_V5, include_note=False):
    ck = cache / (hashlib.sha256(f"{prompt_hash(sys_prompt)}|{model}|{reasoning}|{post['post_id']}".encode()).hexdigest()[:24] + ".json")
    if ck.exists():
        return json.load(open(ck)), 0.0
    user = f"OUTLET: @{post.get('handle') or ''}\nPOST DATE: {post.get('created_at') or 'unknown'}\nPOST:\n{post.get('post_text') or ''}"
    if post.get("quoted_text"):
        user += f"\n\nQUOTED POST by @{post.get('quoted_handle') or 'unknown'}:\n{post['quoted_text']}"
    if include_note and (post.get("note") or "").strip():
        user += f"\n\nCOMMUNITY NOTE (context only, a pointer to the contested assertion):\n{post['note']}"
    body = {"model": model, "temperature": 0, "max_tokens": 4000, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = reasoning
    base, key = _endpoint()
    for attempt in range(5):
        r = eur._SESSION.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
        if r.status_code != 429:
            break
        time.sleep(5 * 2 ** attempt)
    r.raise_for_status()
    j = r.json()
    obj = _parse_json(j["choices"][0]["message"].get("content"))
    usage = j.get("usage") or {}
    cost = usage.get("estimated_cost")
    if cost is None:
        cost = (usage.get("prompt_tokens") or 0) * 0.15e-6 + (usage.get("completion_tokens") or 0) * 0.75e-6
    claims = []
    for c in (obj.get("claims") or [])[:3]:
        if isinstance(c, dict) and (c.get("claim") or "").strip():
            claims.append({"claim": c["claim"].strip(), "type": "attribution" if c.get("type") == "attribution" else "assertion",
                           "why": (c.get("why") or "")[:160]})
    res = {"claims": claims, "none_why": (obj.get("none_why") or "")[:160] if not claims else ""}
    json.dump(res, open(ck, "w"))
    return res, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--dates", default=None, help="parquet with post_id + date (post date per claim)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--provider", default="deepinfra", choices=sorted(PROVIDERS))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reasoning", default="low", help="gpt-oss reasoning_effort")
    ap.add_argument("--prompt", default=PROMPT_PRODUCTION, choices=sorted(PROMPTS), help="extraction prompt version")
    a = ap.parse_args()
    PROVIDER[0] = a.provider
    sys_prompt = PROMPTS[a.prompt]
    include_note = a.prompt in (PROMPT_V6_NOTE, PROMPT_V7_NOTE)
    posts = json.load(open(a.posts))["posts"]
    if a.limit:
        posts = posts[:a.limit]
    if a.dates:
        import pandas as pd
        df = pd.read_parquet(a.dates)
        dates = dict(zip(df["post_id"].astype(str), df["date"].astype(str)))
        for p in posts:
            p["created_at"] = p.get("created_at") or dates.get(str(p["post_id"]))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(str(out) + ".cache"); cache.mkdir(exist_ok=True)
    lock = threading.Lock(); done = [0]; spent = [0.0]; t0 = time.time()
    res = {}

    def work(p):
        r, c = None, 0.0
        for attempt in range(3):
            try:
                r, c = call(a.model, p, cache, a.reasoning, sys_prompt, include_note); break
            except Exception as e:  # noqa: BLE001
                err = repr(e); time.sleep(3 * (attempt + 1))
        with lock:
            res[p["post_id"]] = r or {"claims": [], "none_why": f"failed {err[:80]}"}
            done[0] += 1; spent[0] += c
            if done[0] % 100 == 0 or done[0] == len(posts):
                el = time.time() - t0
                print(f"  {done[0]}/{len(posts)}  ${spent[0]:.3f}  {el:.0f}s  ETA {el / done[0] * (len(posts) - done[0]):.0f}s", flush=True)

    print(f"posts {len(posts)}  prompt {a.prompt} ({prompt_hash(sys_prompt)})  model {a.model} @ {a.provider}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for f in as_completed([ex.submit(work, p) for p in posts]):
            f.result()
    flat = []
    for p in posts:
        for i, c in enumerate(res[p["post_id"]]["claims"], 1):
            flat.append({"claim_id": f"{p['post_id']}:k{i}", "post_id": p["post_id"], "claim": c["claim"], "type": c["type"],
                         "why": c["why"], "domain": p.get("domain"), "handle": p.get("handle"), "ng_score": p.get("ng_score"),
                         "lean": p.get("lean"), "created_at": p.get("created_at"), "post_text": p.get("post_text")})
    json.dump({"prompt": a.prompt, "prompt_hash": prompt_hash(sys_prompt), "model": a.model, "reasoning": a.reasoning,
               "system_prompt": sys_prompt, "posts": res, "claims": flat},
              open(out, "w"), indent=0, ensure_ascii=False)
    n = [len(v["claims"]) for v in res.values()]
    print(f"spent ${spent[0]:.3f}  posts {len(n)}  claims {len(flat)}  mean/post {sum(n) / len(n):.2f}  "
          f"none {sum(1 for x in n if x == 0)}  failed {sum(1 for v in res.values() if v['none_why'].startswith('failed'))}  wrote {out}")


if __name__ == "__main__":
    main()
