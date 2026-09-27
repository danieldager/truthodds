"""Read prompt variants for the reader lab (branch reader-iteration).

"v6.1" is the PRODUCTION prompt since 2026-09-14 (READ_SYS_V6_1, six-class: the "3"
contested/mixed button removed, no other guidance; evidence_urn_run imports it as
READ_SYS). "v5" (READ_SYS_MODE) is the previous production prompt, still importable
for pinned runs. Every other entry is a candidate; nothing else here is wired into a
runner. Add a variant, run reader_lab.py on the audited claims, keep or drop it on the
transition table.
"""
from eval.scripts.build_eval.read_v5_prompts import READ_SYS_MODE, READ_SYS_V6, READ_SYS_V6_1, READ_SYS_V5B, READ_SYS_V5C, READ_SYS_V5X

# Diagnostic only: identical to v5 except the reason is required for EVERY flag, so
# the transcript shows why a confirming sentence is read as a contradiction.
V5_REASON = (READ_SYS_MODE
             .replace('"reason": "<only when direction is I>"', '"reason": "<one sentence: which sentence decided it and why>"')
             .replace('reason — only when direction is "I": "off-claim" (different subject) or\n"adjacent-only" (same subject, different proposition).',
                      'reason — one sentence, always: the deciding sentence number and what it says about the claim.'))
assert V5_REASON != READ_SYS_MODE

# v6a: v5 + reason always + an explicit paraphrase rule. The lab transcripts
# (2026-09-09) show "1" on documents that state the claim in other words: "I'll not
# comment further" read as contradicting "declined to comment further", "abstained,
# decided not to use their veto" as contradicting "failed to veto". Direction stays first.
PARAPHRASE_RULE = """\
- Paraphrase is not contradiction. A document that reports the same fact, event or
  statement as the claim in other words, with a different emphasis, tone, framing,
  or assignment of blame, supports the claim ("5" or "4"). "1" or "2" require a
  sentence that asserts the decisive part of the claim is false, did not happen,
  or happened in a materially different way. Before choosing "1" or "2", name that
  sentence in the reason; if you cannot, the direction is not "1" or "2".
"""
V6A = (V5_REASON.replace("Rules:\n", "Rules:\n" + PARAPHRASE_RULE))
assert V6A != V5_REASON

# v6b: same rule, reason BEFORE direction (the reader commits to the deciding
# sentence before it commits to a flag). Tests whether direction-first still holds
# once the reason is mandatory.
V6B = (V6A.replace('{"direction": "<flag>", "evidence": [<sentence numbers>], "reason": "<one sentence: which sentence decided it and why>"}',
                   '{"reason": "<one sentence: which sentence decided it and why>", "direction": "<flag>", "evidence": [<sentence numbers>]}')
          .replace("Reply with JSON only, direction first:", "Reply with JSON only, reason first:"))
assert V6B != V6A

PROMPTS = {
    "v5": READ_SYS_MODE,
    "v5b": READ_SYS_V5B,  # v5 with the "decisive part" rule replaced; Daniel 2026-09-18
    "v5c": READ_SYS_V5C,  # v5 with only a consequence clause appended to the decisive-part rule; 2026-09-18
    "v5x": READ_SYS_V5X,  # ablation: v5 with the decisive-part rule deleted, nothing added; 2026-09-18
    "v6": READ_SYS_V6,   # six-class reader (no "3"); Daniel 2026-09-14
    "v6.1": READ_SYS_V6_1,  # six-class, minimal edit (no both-ways sentence); Daniel 2026-09-14
    "v5_reason": V5_REASON,
    "v6a": V6A,
    "v6b": V6B,
}

# v7q "copy before judging" (Daniel 2026-09-09): the reader must copy the deciding
# sentence verbatim BEFORE it picks a flag. reader_lab checks the copy against the
# document; a directional flag whose quote is not in the document becomes "X"
# (qc_flag "quote-missing"), so the flag is anchored to a sentence that exists.
QUOTE_RULE = """\
- Copy before judging. First copy, word for word, the single sentence that decides
  the direction. Then choose the flag that sentence supports when read literally.
  A sentence that reports the claim's fact, event or statement in other words,
  with any emphasis, tone, framing or blame, supports the claim ("5" or "4").
  Only a sentence that says the claim's decisive part is false, did not happen,
  or happened in a materially different way points against it ("1" or "2"). If
  no sentence can be copied, the direction is "X" or "I".
"""
V7Q = (READ_SYS_MODE
       .replace('Reply with JSON only, direction first:\n{"direction": "<flag>", "evidence": [<sentence numbers>], "reason": "<only when direction is I>"}',
                'Reply with JSON only, quote first:\n{"quote": "<the one sentence that decides the direction, copied verbatim; empty only for I>", "direction": "<flag>", "evidence": [<sentence numbers>]}')
       .replace("Rules:\n", "Rules:\n" + QUOTE_RULE)
       .replace('reason — only when direction is "I": "off-claim" (different subject) or\n"adjacent-only" (same subject, different proposition).',
                'quote — one sentence copied verbatim from the document, the one that decided\nthe direction. Empty only when direction is "I".'))
assert V7Q.count("quote") >= 4

# v7qa "two questions" (2026-09-09): no seven-way flag from the model. It answers
# whether the document asserts the claim's content (A) and whether it asserts the
# opposite (B), each direct / partial / no, plus whether it is on the claim's subject.
# reader_lab maps the answers to the flags (map_qa), so the urn sees the same flags.
V7QA = """\
You are reading ONE web document to judge its bearing on ONE claim.

You get the claim and a numbered list of sentences from the document. Judge only
what THIS document says about THIS exact claim, not what you know or believe about
the claim from anywhere else.

Answer two questions about the document's own assertions.

A. Does the document assert that what the claim describes is the case: the event
   happened, the figure is as stated, the statement was made, the situation holds?
   "direct"  a sentence says so outright, in the document's own voice, even in
             other words or with a different emphasis, framing or blame
   "partial" the document points toward it without establishing it
   "no"      it does not

B. Does the document assert the opposite: that it is not the case, did not happen,
   or happened in a materially different way (different actor, number, time, or
   outcome on the decisive part)?
   "direct"  a sentence says so outright
   "partial" the document points against it without establishing the opposite
   "no"      it does not

Rules:
- Judge the exact proposition. An adjacent fact, a related event, a similar
  figure, the same subject at another time or place, is "no" on both questions.
- Silence is not denial. A document that does not mention the decisive part of
  the claim is "no" on B, never "direct" or "partial".
- Reporting that people make the claim, that it circulates, or that it was posted
  somewhere is "no" on A. Judge what the document itself asserts.
- Opinion, prediction and advocacy count for neither question.
- Mind the dates when given: a document from well before the claim's date
  describes earlier events, "no" on both.

on_subject: true when the document is about the claim's subject at all, false
when it is about something else.

evidence: the numbers of ALL sentences that bear on the claim, contiguous runs,
empty only when on_subject is false.

Reply with JSON only:
{"a": "<direct|partial|no>", "b": "<direct|partial|no>", "on_subject": <true|false>, "evidence": [<sentence numbers>], "deciding": "<the sentence number that decided A or B, or null>"}
"""


def _norm(s):
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in s).split())


def map_quote(obj, ids, sents):
    q = _norm(str(obj.get("quote") or ""))
    d = str(obj.get("direction", ""))
    if d in ("5", "4", "3", "2", "1"):
        ok = False
        if len(q) >= 12:
            qt = set(q.split())
            for s in sents:
                ns = _norm(s)
                if q in ns or ns in q or len(qt & set(ns.split())) >= 0.8 * len(qt):
                    ok = True
                    break
        if not ok:
            return {"direction": "X", "evidence": obj.get("evidence") or [], "reason": f"quote-missing: {q[:80]}", "qc_flag": "quote-missing"}
    return {"direction": d, "evidence": obj.get("evidence") or [], "reason": (obj.get("quote") or "")[:120]}


def map_qa(obj, ids, sents):
    a = str(obj.get("a", "no")).lower(); b = str(obj.get("b", "no")).lower()
    on = bool(obj.get("on_subject"))
    if a not in ("direct", "partial", "no") or b not in ("direct", "partial", "no"):
        return {"direction": "bad", "evidence": []}
    if a != "no" and b != "no":
        d = "3"
    elif a == "direct":
        d = "5"
    elif a == "partial":
        d = "4"
    elif b == "direct":
        d = "1"
    elif b == "partial":
        d = "2"
    else:
        d = "X" if on else "I"
    ev = obj.get("evidence") or []
    if d == "I":
        ev = []
    return {"direction": d, "evidence": ev, "reason": f"a={a} b={b} deciding={obj.get('deciding')}"}


PROMPTS["v7q"] = V7Q
PROMPTS["v7qa"] = V7QA
MAPPERS = {"v7q": map_quote, "v7qa": map_qa}

# v7qqa: quote-first + two questions (Daniel 2026-09-09 19:45). The reader copies the
# deciding sentence, then answers A and B; map_qa gives the flag and the quote check
# anchors it (directional flag without a quote in the document -> X).
V7QQA = (V7QA
         .replace("Answer two questions about the document's own assertions.",
                  "First copy, word for word, the single sentence that decides your answers. Then\n"
                  "answer two questions about the document's own assertions, reading that sentence\n"
                  "literally.")
         .replace('{"a": "<direct|partial|no>",',
                  '{"quote": "<the deciding sentence copied verbatim; empty when neither question is direct or partial>", "a": "<direct|partial|no>",'))
assert V7QQA != V7QA


def map_qqa(obj, ids, sents):
    r = map_qa(obj, ids, sents)
    if r["direction"] in ("5", "4", "3", "2", "1"):
        chk = map_quote({"quote": obj.get("quote"), "direction": r["direction"], "evidence": r["evidence"]}, ids, sents)
        if chk.get("qc_flag"):
            return {**chk, "reason": f"{r['reason']} {chk['reason']}"[:120]}
        r["reason"] = f"{r['reason']} | {str(obj.get('quote') or '')[:70]}"[:120]
    return r


PROMPTS["v7qqa"] = V7QQA
MAPPERS["v7qqa"] = map_qqa

# Canonicalisation pre-step (Daniel's proposal 1, tested at the reader before it moves
# into normalize_tweet_claims.py): rewrite the claim as one plain literal proposition,
# no framing verbs, negatives as "did not". reader_lab --canon feeds the rewrite to the
# reader instead of the stored claim_resolved.
CANON_SYS = """\
Rewrite a claim as one plain, literal proposition that a reader can check against a
document sentence by sentence.

Keep every named entity, number, date and place exactly. Add nothing, drop nothing.
Replace framing and evaluative verbs with what literally happened or did not happen:
"failed to X", "refused to X", "declined to X", "abstained from X", "neglected to X"
become "did not X"; "was forced to X", "caved and X", "finally X", "admitted X",
"slammed", "blasted" become the plain action or statement. State negatives with
"did not" or "is not". Keep reported speech as reported speech: "X said Y" stays
"X said Y" with Y rewritten the same way. If the claim is already plain, return it
unchanged.

Reply with JSON only: {"claim": "<the rewritten claim>"}
"""
