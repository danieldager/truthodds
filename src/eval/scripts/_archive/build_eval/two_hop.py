"""TWO-HOP retrieval, prototyped on quote_attribution (clog/280826, oracle probe).

The oracle probe established that the reviewer's cited evidence IS indexed (76.6% of
ceiling-legal targets) and that ZERO of 82 reviewer-cited sources were documents about
the claim in the claim's own terms. The reviewer reasons in two hops: claim -> the
underlying fact the claim distorts -> a document about that fact. Our retrieval does
only the first hop, and for a false claim the web contains no document that states it.
Six retrieval interventions have failed because each was a better way to execute a hop
that cannot succeed.

quote_attribution is the test type: it is the weakest kept type (within-type AUC 0.8317,
recall@2%FPR 28.2%, 41.9% all-irrelevant, roughly double every other type), the oracle
probe independently scored it 0/15 on query reach, and its second hop is unusually well
defined -- the claim asserts someone said something, so the implied record is the actual
utterance.

Three changes, each isolated so the report can attribute movement:

  HOP 1  referent resolution. From claim text + date ONLY (never the label, never a
         document) emit a structured description of the RECORD that would settle the
         attribution: speaker, the proposition attributed, the occasion, its date, the
         canonical name of the record, and a query for THAT record. Principles only in
         the prompt -- no dataset-derived examples (standing project rule).
  HOP 2  retrieval against the record, not against the claim. Production Serper path,
         k=10 (num above 10 is silently ignored on this plan), date ceiling ON so the
         headline arm stays production-legal.
  READ   a substitution-aware rubric. The production reader asks whether a document
         supports or refutes the claim; a transcript that simply shows what was said
         never mentions the claim, so that rubric cannot use it. The variant asks the
         substitution question and maps onto the SAME 7 flags so scores stay comparable.

ORIGIN SOURCE, as an explicit sub-arm (Daniel's question). Two searches per claim:
  hop2_x  ORIGIN EXCLUDED: production `search()` -- UGC/platform blocklist on, plus
          [publisher_site] and hop-1's `source_domains` (the speaker's own channel).
  hop2_o  ORIGIN INCLUDED: platform blocklist OFF and the speaker's own channel
          admitted; only [publisher_site] stays excluded, because that is the
          fact-check answer key, not an origin source, and lifting it would be a leak.

POPULATION, both classes, matched on the condition the whole line of work targets:
  qa_false  quote_attribution, gold veracity 1-2, ALL-IRRELEVANT (every retrieved doc
            read I at baseline) -- `retrieval_recovery_arms.load_corpora` + `_allirr`.
  qa_true   quote_attribution, gold veracity 4-5, ALL-IRRELEVANT, from the pinned
            fc-gold cut. Both classes therefore start at exactly zero directional
            documents, so any gain is attributable to the new query.
Claim-type labels are the claim-side (label-blind) ones from eval/data/claim_type/.

METRIC is SELECTIVITY. An earlier arm was killed for producing more refutations on true
claims than on false ones. Headline = P(gains >=1 refuting doc | FALSE) vs the same on
TRUE, Wilson intervals, and the ratio.

    uv run python -m eval.scripts.build_eval.two_hop --sample
    uv run python -m eval.scripts.build_eval.two_hop --hop1 --workers 16
    uv run python -m eval.scripts.build_eval.two_hop --show-hop1 8
    uv run python -m eval.scripts.build_eval.two_hop --run --smoke 8 --workers 6
    uv run python -m eval.scripts.build_eval.two_hop --run --workers 12
    uv run python -m eval.scripts.build_eval.two_hop --report

Nothing under eval/data/urn_runs/ is read-modified or written. Output: eval/data/two_hop/.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

from pipeline.config import FACT_CHECK_DOMAINS, SCRAPE_BLOCKLIST  # noqa: E402
from pipeline.credibility import TRUSTED_FACTCHECKERS  # noqa: E402
from pipeline.search import SearchError, scrape, search  # noqa: E402
from pipeline.verify_tweet_claims import _domain_of  # noqa: E402
from eval.scripts.build_eval import fit_urn  # noqa: E402
from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
    DIRECTIONS, READ_SYS, _serper_gate, aggregate_reads, llm, select_regions)
from eval.scripts.build_eval.graded_urn import FLAGS  # noqa: E402
from eval.scripts.build_eval.refute_verify import w7  # noqa: E402
from eval.scripts.build_eval.retrieval_recovery_arms import (  # noqa: E402
    _allirr, search_unblocked)

OUT = SRC / "eval/data/two_hop"
SAMPLE = OUT / "sample.jsonl"
HOP1 = OUT / "hop1.jsonl"
RESULTS = OUT / "results.jsonl"
E1 = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
LABELS = SRC / "eval/data/claim_type/labels.jsonl"

SEED = 20260828
TYPE = "quote_attribution"
N_PER_CLASS = 70
MAX_DOCS = 20            # unique URLs read per claim across both sub-arms
FLAG_EDGE = -4.05
FC_DOMS = set(FACT_CHECK_DOMAINS) | {d for d in TRUSTED_FACTCHECKERS if "/" not in d}

# Hard stops. The in-code LLM meter is DeepInfra's `estimated_cost`, which the 280826
# reportability calibration puts ~6-7x above real spend, so $4.00 metered is ~$0.6 real;
# Serper is 1 credit per REQUEST (verified 280826) at ~$0.001. Task ceiling $3.00 real.
LLM_CAP = 2.40
SERPER_CAP = 400

HOP1_PROMPT_V = "hop1-referent-v2"
SUBST_PROMPT_V = "read-subst-v2"

# --------------------------------------------------------------------------- prompts

HOP1_SYS = """\
You are given one factual claim and the date it was circulating. The claim asserts that a
named source said, wrote, posted or published a particular statement.

Do NOT judge the claim. Do not say whether it is true, likely, or suspicious. Your only
job is to name the RECORD that would settle it.

Reason like this. A claim of this kind implies an underlying event: a specific person or
body producing specific words on a specific occasion. Somewhere there either is or is not
a record of that occasion -- a transcript, a recording, a published text, an official
release, a contemporaneous report of it. Name that occasion and that record as precisely
as the claim allows, in the vocabulary the record itself would use rather than the
vocabulary the claim uses.

Respond with JSON only:
{"speaker": "...", "proposition": "...", "occasion": "...", "occasion_date": "...",
 "record_name": "...", "source_domains": ["..."], "record_query": "..."}

speaker -- the person, office, outlet, institution or body alleged to have produced the
  statement, by its full canonical name as an index would carry it. Expand nicknames,
  titles and abbreviations. Empty string only if the claim names no source at all.

proposition -- the substance attributed to the source, in one plain sentence, stripped of
  the claim's framing and of any wording about who is reporting it.

occasion -- the specific event, venue, hearing, broadcast, programme, publication or
  platform on which the statement is alleged to have been made. If the claim names none,
  infer the most likely one from the speaker's role, their activity and the date, and
  write that. Empty string only when nothing at all can be inferred.

occasion_date -- the approximate date or period of the alleged utterance, as YYYY-MM-DD
  or YYYY-MM or YYYY. This is the date of the OCCASION, which is often earlier than the
  date the claim was circulating. Empty string if it cannot be placed.

record_name -- the canonical name of the document or recording that would carry the
  actual words, if such a thing has a name. Empty string if it has no name.

source_domains -- up to two website domains, bare, no scheme and no path, where the
  source itself would publish or host its own record of this. The speaker's own outlet,
  official site, or the institution whose proceedings these are. Empty list if the
  speaker has no such channel.

record_query -- one web search query that would retrieve the record. Compose it, in this
  order, from:
    (a) the speaker's canonical name, or their account name when the occasion is a post;
    (b) the occasion, named the way its own record names it -- the committee, the
        programme, the summit, the venue, the publication, the platform;
    (c) one or two words that a RECORD carries and that commentary about it does not:
        transcript, full remarks, full text, readout, press briefing, statement,
        proceedings, floor speech, official release, archive;
    (d) the month and year of the occasion, when it is known.

  The attributed proposition's own distinctive wording is DELIBERATELY LEFT OUT. This is
  the point of the task. A query built from the attributed wording retrieves documents
  about the attribution; a query built from the occasion retrieves the record, and the
  record is the only thing that can show what was actually said.

  Include the general subject of the occasion ONLY when it is needed to tell this
  occasion apart from other occasions of the same speaker, and when you do, use the
  subject's neutral, externally recognisable name rather than the disputed formulation.
  Never quote the attributed statement. Never ask whether anything is true, and never use
  words like fact check, false, fake, hoax, debunk, misleading or claim.

  When no occasion can be placed at all, build the query from the speaker, their role,
  the period, and a record word, and nothing else.

  Keep it a query, not a sentence.

Never assume the statement was actually made. The record is what would show it either way.
"""

READ_SYS_SUBST = """\
You are reading ONE web document. The task is a SUBSTITUTION check, not a support check.

A claim attributes a statement to a named source on a particular occasion. You are shown
that attribution and a numbered list of sentences from a document that was retrieved as a
possible RECORD of what the source actually said on that occasion.

The document may never mention the claim, and may never mention that anyone attributed
anything to anyone. That is expected. It is NOT a reason to call the document irrelevant.
Judge the document as a record.

Ask two questions, in this order:
  1. Does this document carry the source's actual words on the occasion in question --
     its own transcript, recording, full text, official release, or its own first-hand
     report of the occasion?
  2. If it does, how do the words the source actually used compare with the proposition
     attributed to them?

Judge only what THIS document shows. Never use your own knowledge of whether the
attribution is accurate.

Reply with JSON only, direction first:
{"direction": "<flag>", "evidence": [<sentence numbers>], "occasion_match": "<same|different|unclear>", "basis": "<match|contradiction|different_speaker|absence|none>", "reason": "<only when direction is I>"}

direction -- ONE flag:
  "5"  the document carries the source's words on that occasion and they state the
       attributed proposition
  "4"  the document carries the source's words, or a first-hand account of them, and they
       convey the attributed proposition in substance though not in that form
  "3"  the document carries the actual words and they are genuinely ambiguous between
       matching and contradicting the attribution, or the record contains both
  "2"  the document carries the source's words on that occasion and they differ
       materially from the attributed proposition without excluding it: a narrower or
       qualified version, a different object, an omitted condition, a different sense
  "1"  the document establishes what the source actually said and it contradicts the
       attributed proposition; or it establishes that the words belong to a different
       speaker; or it is a COMPLETE record of the named occasion -- a full transcript,
       the entire published text, the whole broadcast -- in which the attributed
       statement does not appear
  "X"  the document concerns the source, or the occasion, or the subject matter, but does
       not carry what the source said on that occasion
  "I"  a different source, a different occasion, or a different subject

evidence -- the numbers of ALL sentences that carry or bear on the source's actual words.
Select contiguous runs: when relevant sentences are separated by one or two connecting
sentences, include the connecting sentences too. Empty only with "I".

occasion_match -- whether the occasion THIS DOCUMENT records is the occasion named above.
  "same" only when the document's own date, venue, programme or event matches the named
  occasion. A record of the same speaker on a different date, at a different venue, or in
  a different programme is "different", however similar its subject. "unclear" when the
  document does not say which occasion it records.

basis -- what the direction rests on:
  "match"             the document carries words that match the attributed proposition
  "contradiction"     the document carries words that affirmatively contradict it
  "different_speaker" the document shows the words belong to someone else
  "absence"           the document is a complete record of the occasion and the
                      attributed statement is simply not in it
  "none"              the direction is "X" or "I"

Rules:
- The unit is the utterance. Whether the attributed statement is TRUE is irrelevant here.
  A document that disputes the content while showing the source said it is "5".
- Only the record counts. Words the document attributes to a viral post, to social media
  users, to critics, or to the claim it is examining are the attribution circulating, not
  the record -- "X", never "5" or "4".
- Absence is not contradiction, with one exception, and the exception is narrow. ALL of
  the following must hold, or the flag is "X": the document must present itself as the
  COMPLETE record of an occasion; that occasion must be the one named above, so
  occasion_match must be "same"; and the attributed statement must be missing from it. A
  complete transcript of a DIFFERENT speech, hearing or broadcast by the same speaker
  shows nothing at all about the named occasion and is "X" or "I", never "1". A partial
  report, an excerpt, a summary, a highlights piece, or an article about the occasion that
  does not purport to reproduce it is "X" when the statement is missing, never "1".
- Exactness matters for "5" and "4": the same statement, on the claimed occasion. The same
  source saying something thematically similar on another occasion or in another venue is
  "X" or "I".
- Mind the claim's own polarity. When the attribution is that the source did NOT say
  something, a record showing they DID say it points against it ("2" or "1").
- Mind the dates. A document from well before the occasion cannot record it -- at most "X".
- Opinion about the source or about the content is evidence in neither direction.

reason -- only when direction is "I": "off-claim" (different subject) or "adjacent-only"
(same source or subject, different occasion or statement).
"""


# --------------------------------------------------------------------------- meters
_lock = threading.Lock()
_llm_cost = [0.0]
_credits = [0]


def spend_llm(c: float) -> None:
    with _lock:
        _llm_cost[0] += c
        if _llm_cost[0] > LLM_CAP:
            raise SystemExit(f"LLM CAP ${LLM_CAP} breached (${_llm_cost[0]:.3f})")


def spend_serper(n: int = 1) -> None:
    with _lock:
        _credits[0] += n
        if _credits[0] > SERPER_CAP:
            raise SystemExit(f"SERPER CAP {SERPER_CAP} breached")


def dom(u: str) -> str:
    return urlsplit(u).netloc.lower().removeprefix("www.").removeprefix("m.")


def bare_domain(s: str) -> str:
    """A model-emitted domain, which may arrive bare ("senate.gov"), with a scheme, or
    with a path. urlsplit puts a bare domain in `path`, not `netloc`, hence this."""
    s = (s or "").strip().lower()
    s = s.split("://", 1)[-1].split("/", 1)[0].split("?", 1)[0]
    return s.removeprefix("www.").removeprefix("m.")


def blocked(d: str) -> bool:
    return d in SCRAPE_BLOCKLIST or any(d.endswith("." + b) for b in SCRAPE_BLOCKLIST)


def is_fc(d: str) -> bool:
    return any(d == f or d.endswith("." + f) for f in FC_DOMS)


# --------------------------------------------------------------------------- sample
def load_labels() -> dict:
    return {json.loads(l)["key"]: json.loads(l)["claim_type_side"] for l in LABELS.open()}


def gold_pinned() -> dict:
    """model_ladder's pinned cut, keyed by review_url. Mirrors
    retrieval_recovery_arms.load_corpora's gold filter, with the veracity window
    opened to 4-5 so the TRUE control is drawable from the same population."""
    axis = fit_urn.load_judged_axis()
    out = {}
    for line in E1.open():
        r = json.loads(line)
        if r.get("veracity") not in (1, 2, 3, 4, 5):
            continue
        if (r.get("rating_subtype") or "?") == "mixed":
            continue
        if (axis.get(r["review_url"]) or "untagged") == fit_urn.MEDIA_AXIS:
            continue
        if not [d for d in r["results"] if d["read"]["direction"] in FLAGS]:
            continue
        out[r["review_url"]] = r
    return out


def build_sample() -> None:
    w = w7()
    lab = load_labels()
    gold = gold_pinned()
    assert len(gold) == 3274, f"pinned cut is {len(gold)}, expected 3274"
    pools = {"qa_false": [], "qa_true": []}
    for k, r in gold.items():
        if lab.get("gold:" + k) != TYPE:
            continue
        if not _allirr(r):
            continue
        v = r["veracity"]
        pop = "qa_false" if v in (1.0, 2.0) else ("qa_true" if v in (4.0, 5.0) else None)
        if pop is None:            # veracity 3 (mid) is set aside, not a class
            continue
        docs = [d for d in r["results"] if d["read"]["direction"] in FLAGS]
        flags = [d["read"]["direction"] for d in docs]
        pools[pop].append({
            "pop": pop, "key": k, "veracity": v,
            "claim": r.get("claim_resolved") or r.get("claim_text") or "",
            "claim_date": r.get("claim_date_shown") or "", "ceiling": r.get("ceiling") or "",
            "publisher_site": r.get("publisher_site") or "",
            "query_prod": r.get("query") or "",
            "s7_base": round(sum(w.get(f, 0.0) for f in flags), 4),
            "base_flags": flags, "base_urls": [d.get("url") for d in docs],
            "n_refute_base": sum(1 for f in flags if f in ("1", "2")),
        })
    rng = random.Random(SEED)
    rows = []
    for pop, pool in pools.items():
        pool.sort(key=lambda r: r["key"])
        rng.shuffle(pool)
        print(f"  {pop}: pool {len(pool)} -> draw {min(N_PER_CLASS, len(pool))}")
        rows += pool[:N_PER_CLASS]
    OUT.mkdir(parents=True, exist_ok=True)
    with SAMPLE.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for pop in pools:
        sub = [r for r in rows if r["pop"] == pop]
        assert all(r["n_refute_base"] == 0 for r in sub)
        print(f"  {pop:9s} n={len(sub)}  mean s7 base {sum(r['s7_base'] for r in sub)/len(sub):+.2f}  "
              f"mean base docs {sum(len(r['base_flags']) for r in sub)/len(sub):.1f}")
    print(f"  TOTAL {len(rows)} -> {SAMPLE}")


# --------------------------------------------------------------------------- hop 1
def hop1_one(row: dict) -> dict:
    blk = [f"CLAIM: {row['claim']}"]
    if row.get("claim_date"):
        blk.append(f"CLAIM CIRCULATING ON: {row['claim_date']}")
    for _ in range(3):
        try:
            j, c, _, _ = llm([{"role": "system", "content": HOP1_SYS},
                              {"role": "user", "content": "\n".join(blk)}], max_tokens=420)
            spend_llm(c)
            q = (j.get("record_query") or "").strip()[:300]
            if q:
                sd = [b for b in (bare_domain(d) for d in (j.get("source_domains") or [])[:2]
                                  if isinstance(d, str)) if "." in b]
                return {"key": row["key"], "pop": row["pop"], "claim": row["claim"],
                        "claim_date": row.get("claim_date"), "query_prod": row["query_prod"],
                        "speaker": (j.get("speaker") or "")[:200],
                        "proposition": (j.get("proposition") or "")[:400],
                        "occasion": (j.get("occasion") or "")[:300],
                        "occasion_date": (j.get("occasion_date") or "")[:12],
                        "record_name": (j.get("record_name") or "")[:200],
                        "source_domains": sd, "record_query": q,
                        "prompt_v": HOP1_PROMPT_V, "cost": c}
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001
            time.sleep(2)
    raise RuntimeError(f"hop1 failed: {row['key']}")


def run_hop1(workers: int, smoke: int) -> None:
    rows = [json.loads(l) for l in SAMPLE.open()]
    done = {json.loads(l)["key"] for l in HOP1.open()} if HOP1.exists() else set()
    todo = [r for r in rows if r["key"] not in done]
    if smoke:
        todo = todo[:smoke]
    if not todo:
        print("hop1: nothing to do")
        return
    print(f"hop1: {len(todo)} claims | {workers} workers", flush=True)
    t0, n = time.time(), [0]
    with HOP1.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(hop1_one, r) for r in todo]):
            try:
                o = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {type(e).__name__}: {e}", flush=True)
                continue
            with _lock:
                f.write(json.dumps(o) + "\n")
                f.flush()
            n[0] += 1
            if n[0] % 20 == 0 or n[0] == len(todo):
                el = time.time() - t0
                print(f"  {n[0]}/{len(todo)}  ${_llm_cost[0]:.4f}  "
                      f"{n[0]/max(el,1e-3)*60:.0f}/min  "
                      f"ETA {(len(todo)-n[0])/max(n[0]/max(el,1e-3),1e-6)/60:.1f}m", flush=True)
    print(f"hop1 done: {n[0]} claims, ${_llm_cost[0]:.4f}, {(time.time()-t0)/60:.1f}m", flush=True)


def show_hop1(k: int) -> None:
    rows = [json.loads(l) for l in HOP1.open()]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    per = collections.defaultdict(list)
    for r in rows:
        per[r["pop"]].append(r)
    for pop in sorted(per):
        for r in per[pop][:k]:
            print("=" * 100)
            print(f"[{pop}] {r['claim'][:220]}")
            print(f"  date        {r.get('claim_date') or '-'}")
            print(f"  Q_prod      {r['query_prod']}")
            print(f"  speaker     {r['speaker']}")
            print(f"  proposition {r['proposition'][:200]}")
            print(f"  occasion    {r['occasion'][:160]}  [{r['occasion_date'] or '-'}]")
            print(f"  record      {r['record_name'][:160]}")
            print(f"  domains     {r['source_domains']}")
            print(f"  Q_hop2      {r['record_query']}")


# --------------------------------------------------------------------------- reads
def _read(sys_prompt: str, block: str, ids: list[int], sents: list[str], key: str,
          subst: bool = False):
    """read_doc's validation, with the system prompt as a parameter. evidence_urn_run
    hardcodes READ_SYS and must not be edited, so the 15 lines are restated here.

    The substitution rubric carries ONE extra, code-enforced guard. Its narrow
    absence-as-refutation exception ("a complete record of the occasion that does not
    contain the statement") misfired on the first smoke: complete transcripts of a
    DIFFERENT speech by the same speaker were read as "1". That is the read-v5.2 burn
    (absence-as-refutation inverted the stratum) reappearing. Rather than plead in the
    prompt, the rubric now emits `occasion_match` and `basis`, and a refutation resting on
    absence is coerced to "X" unless the document records the occasion actually named."""
    doc = "\n".join(f"[{i}] {s}" for i, s in zip(ids, sents))
    obj, cost, _, _ = llm([{"role": "system", "content": sys_prompt},
                           {"role": "user", "content": f"{block}\n\nDOCUMENT:\n{doc}"}],
                          cache_key=key)
    spend_llm(cost)
    d = obj.get("direction")
    if isinstance(d, int):
        d = str(d)
    ev = obj.get("evidence") or []
    idset = set(ids)
    if d not in DIRECTIONS or not all(isinstance(i, int) and i in idset for i in ev):
        return None
    qc = ""
    if d != "I" and not ev:                       # uncited directional vote is not evidence
        d, qc = "X", "empty-directional"
    om, basis = (obj.get("occasion_match") or ""), (obj.get("basis") or "")
    if subst and d in ("1", "2") and basis == "absence" and om != "same":
        d, qc = "X", "absence-off-occasion"
    return {"direction": d, "evidence": sorted(set(ev)),
            "reason": (obj.get("reason") or "")[:80], "qc_flag": qc,
            "occasion_match": om[:12], "basis": basis[:20]}


def read_both(row: dict, h1: dict, query: str, url: str, text: str, prov: str) -> dict:
    regions, _ = select_regions(text, row["claim"], query)
    if not regions:
        return {"prod": None, "subst": None, "status": "empty-doc", "sents": ""}
    prod_block = (f"CLAIM: {row['claim']}\n"
                  f"(claimed on {row.get('claim_date') or 'unknown date'}; judge the "
                  f"document's bearing on this exact proposition)")
    subst_block = (
        f"CLAIM: {row['claim']}\n"
        f"(claimed on {row.get('claim_date') or 'unknown date'})\n\n"
        f"THE ATTRIBUTION UNDER TEST\n"
        f"  SOURCE: {h1['speaker'] or '(unnamed)'}\n"
        f"  ATTRIBUTED STATEMENT: {h1['proposition']}\n"
        f"  OCCASION: {h1['occasion'] or '(not named in the claim)'}\n"
        f"  OCCASION DATE: {h1['occasion_date'] or '(unknown)'}")
    out = {"status": "ok", "prov": prov,
           "sents": " ".join(s for _, sel in regions for s in sel)[:700]}
    for name, sysp, blk in (("prod", READ_SYS, prod_block),
                            ("subst", READ_SYS_SUBST, subst_block)):
        rr = []
        for ids, sel in regions:
            got = None
            for _ in range(2):
                try:
                    got = _read(sysp, blk, ids, sel, row["key"] + "|" + name,
                                subst=(name == "subst"))
                except SystemExit:
                    raise
                except Exception:  # noqa: BLE001
                    time.sleep(2)
                    continue
                if got:
                    break
            rr.append(got or {"direction": "I", "evidence": [], "reason": "",
                              "qc_flag": "read-failed"})
        agg = aggregate_reads(rr)
        out[name] = {"direction": agg["direction"], "reason": agg.get("reason") or "",
                     "occasion_match": rr[0].get("occasion_match", ""),
                     "basis": rr[0].get("basis", ""), "qc": rr[0].get("qc_flag", "")}
    return out


# --------------------------------------------------------------------------- run
def run_one(row: dict, h1: dict) -> dict:
    ceil = row.get("ceiling") or None
    pub = row["publisher_site"]
    query = h1["record_query"]
    # ORIGIN EXCLUDED: production path. Blocklist on; the fact-check publisher and the
    # speaker's own channel both dropped.
    xd_x = [pub] + [d for d in h1["source_domains"] if d and d != pub]
    hits = {}
    for arm, fn in (("hop2_x", lambda: search(query, 10, date_ceiling=ceil,
                                              exclude_domains=xd_x, min_results=0,
                                              provider="serper")),
                    # ORIGIN INCLUDED: platform blocklist off, speaker's channel admitted.
                    # [publisher_site] stays excluded: that is the answer key, not an origin.
                    ("hop2_o", lambda: search_unblocked(query, 10, ceil, [pub], None))):
        got = None
        for attempt in range(2):
            _serper_gate()
            spend_serper()
            try:
                got = fn()
                break
            except SearchError:
                if attempt == 0:
                    time.sleep(60)
                else:
                    raise
        hits[arm] = got or []

    base = set(row["base_urls"])
    docs, seen = [], set()
    for arm in ("hop2_x", "hop2_o"):
        for h in hits[arm][:10]:
            u = h.get("url") or ""
            if not u:
                continue
            if u in seen:
                docs[[d["url"] for d in docs].index(u)]["arms"].append(arm)
                continue
            if len(seen) >= MAX_DOCS:
                continue
            seen.add(u)
            docs.append({"url": u, "domain": _domain_of(u), "arms": [arm],
                         "date": h.get("date"), "snippet": h.get("snippet") or "",
                         "content": h.get("content") or "", "repeat_of_base": u in base})
    for d in docs:
        d["ugc"] = blocked(dom(d["url"]))
        d["fc_domain"] = is_fc(d["domain"])
        d["origin_domain"] = d["domain"] in set(h1["source_domains"])
        if d["fc_domain"] and not (d.get("date") or "").strip():
            d.update(status="fc-undated", prod=None, subst=None)   # undated FC = leak risk
            continue
        text, prov = d.pop("content"), "scrape"
        if not text:
            text = scrape(d["url"]) or ""
        if not text or len(text) < 200:
            text, prov = d["snippet"], "snippet"
        if not text:
            d.update(status="no-text", prod=None, subst=None)
            continue
        d.update(read_both(row, h1, query, d["url"], text, prov))
    for d in docs:
        d.pop("snippet", None)
        d.pop("content", None)
    return {"key": row["key"], "pop": row["pop"], "claim": row["claim"],
            "veracity": row["veracity"], "query_prod": row["query_prod"],
            "query_hop2": query, "hop1": {k: h1[k] for k in
                                          ("speaker", "proposition", "occasion",
                                           "occasion_date", "record_name", "source_domains")},
            "s7_base": row["s7_base"], "base_flags": row["base_flags"],
            "n_hits_x": len(hits["hop2_x"]), "n_hits_o": len(hits["hop2_o"]),
            "docs": docs}


def run(workers: int, smoke: int) -> None:
    rows = {json.loads(l)["key"]: json.loads(l) for l in SAMPLE.open()}
    h1 = {json.loads(l)["key"]: json.loads(l) for l in HOP1.open()}
    done = {json.loads(l)["key"] for l in RESULTS.open()} if RESULTS.exists() else set()
    todo = [k for k in rows if k in h1 and k not in done]
    if smoke:                                    # equal draw per class, seeded
        rng = random.Random(SEED)
        by = collections.defaultdict(list)
        for k in sorted(todo):
            by[rows[k]["pop"]].append(k)
        todo = []
        for pop in sorted(by):
            rng.shuffle(by[pop])
            todo += by[pop][:smoke]
    if not todo:
        print("run: nothing to do")
        return
    print(f"run: {len(todo)} claims | {workers} workers | "
          f"{dict(collections.Counter(rows[k]['pop'] for k in todo))}", flush=True)
    t0, n = time.time(), [0]
    with RESULTS.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, rows[k], h1[k]) for k in todo]
        for fut in as_completed(futs):
            try:
                o = fut.result()
            except SystemExit as e:
                print(f"  STOP {e}", flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"  ERR {type(e).__name__}: {e}", flush=True)
                continue
            with _lock:
                f.write(json.dumps(o) + "\n")
                f.flush()
            n[0] += 1
            el = time.time() - t0
            rate = n[0] / max(el, 1e-3)
            if n[0] % 5 == 0 or n[0] == len(todo):
                print(f"  {n[0]}/{len(todo)}  LLM ${_llm_cost[0]:.3f}  "
                      f"serper {_credits[0]}  {rate*60:.1f}/min  "
                      f"ETA {(len(todo)-n[0])/max(rate,1e-6)/60:.1f}m", flush=True)
    print(f"run done: {n[0]} claims, LLM ${_llm_cost[0]:.3f}, {_credits[0]} serper credits, "
          f"{(time.time()-t0)/60:.1f}m", flush=True)


# --------------------------------------------------------------------------- report
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def auc(scores: list[float], y: list[int]) -> float:
    """P(score(true) > score(false)) with ties at 0.5. y=1 is a TRUE claim."""
    pos = [s for s, t in zip(scores, y) if t == 1]
    neg = [s for s, t in zip(scores, y) if t == 0]
    if not pos or not neg:
        return float("nan")
    tot = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return tot / (len(pos) * len(neg))


def arm_docs(r: dict, arm: str, rubric: str) -> list[str]:
    out = []
    for d in r["docs"]:
        if arm not in d["arms"]:
            continue
        f = (d.get(rubric) or {}).get("direction") if d.get(rubric) else None
        if f in FLAGS:
            out.append(f)
    return out


ARMS = [("baseline", None, None),
        ("hop2_x + prod read", "hop2_x", "prod"),
        ("hop2_x + subst read", "hop2_x", "subst"),
        ("hop2_o + prod read", "hop2_o", "prod"),
        ("hop2_o + subst read", "hop2_o", "subst")]


def report() -> None:
    w = w7()
    res = [json.loads(l) for l in RESULTS.open()]
    per = collections.defaultdict(list)
    for r in res:
        per[r["pop"]].append(r)
    print(f"\nn = {len(res)}  " + "  ".join(f"{p} {len(v)}" for p, v in sorted(per.items())))

    # ---- retrieval shape
    print("\n### RETRIEVAL")
    print(f"{'pop':10s} {'hits_x':>7s} {'hits_o':>7s} {'uniq':>6s} {'read':>6s} "
          f"{'ugc':>6s} {'fc':>5s} {'origin':>7s} {'repeat_base':>12s}")
    for pop in sorted(per):
        rs = per[pop]
        n = len(rs)
        alld = [d for r in rs for d in r["docs"]]
        rd = [d for d in alld if d.get("status") == "ok"]
        print(f"{pop:10s} {sum(r['n_hits_x'] for r in rs)/n:7.2f} "
              f"{sum(r['n_hits_o'] for r in rs)/n:7.2f} {len(alld)/n:6.2f} {len(rd)/n:6.2f} "
              f"{sum(1 for d in alld if d['ugc'])/n:6.2f} "
              f"{sum(1 for d in alld if d['fc_domain'])/n:5.2f} "
              f"{sum(1 for d in alld if d['origin_domain'])/n:7.2f} "
              f"{sum(1 for d in alld if d['repeat_of_base'])/n:12.2f}")

    # ---- per-arm
    print("\n### FLAG DISTRIBUTION (new documents only)")
    for name, arm, rub in ARMS:
        if arm is None:
            for pop in sorted(per):
                c = collections.Counter(f for r in per[pop] for f in r["base_flags"])
                print(f"{name:22s} {pop:9s} " + " ".join(f"{k}:{c[k]}" for k in "54321XI" if c[k]))
            continue
        for pop in sorted(per):
            c = collections.Counter(f for r in per[pop] for f in arm_docs(r, arm, rub))
            tot = sum(c.values())
            print(f"{name:22s} {pop:9s} n_docs={tot:4d} " +
                  " ".join(f"{k}:{c[k]}" for k in "54321XI" if c[k]))

    print("\n### SUBSTITUTION READ diagnostics (all read docs)")
    for pop in sorted(per):
        sd = [d["subst"] for r in per[pop] for d in r["docs"]
              if d.get("status") == "ok" and d.get("subst")]
        om = collections.Counter(x.get("occasion_match") or "-" for x in sd)
        bs = collections.Counter(x.get("basis") or "-" for x in sd)
        qc = collections.Counter(x.get("qc") for x in sd if x.get("qc"))
        print(f"  {pop:9s} occasion_match {dict(om)}")
        print(f"  {pop:9s} basis          {dict(bs)}")
        print(f"  {pop:9s} qc coercions   {dict(qc)}")

    print("\n### SELECTIVITY  P(gains >=1 refuting doc, flag 1 or 2)")
    print(f"{'arm':22s} {'FALSE':>22s} {'TRUE':>22s} {'ratio':>7s}")
    rows_out = []
    for name, arm, rub in ARMS:
        if arm is None:
            continue
        cell = {}
        for pop in ("qa_false", "qa_true"):
            rs = per.get(pop, [])
            k = sum(1 for r in rs if any(f in ("1", "2") for f in arm_docs(r, arm, rub)))
            cell[pop] = (k, len(rs)) + wilson(k, len(rs))[1:]
            cell[pop + "_p"] = k / max(len(rs), 1)
        pf, pt = cell["qa_false_p"], cell["qa_true_p"]
        ratio = pf / pt if pt > 0 else float("inf") if pf > 0 else float("nan")
        kf, nf, lf, hf = cell["qa_false"]
        kt, nt, lt, ht = cell["qa_true"]
        print(f"{name:22s} {kf:3d}/{nf:3d}={pf:5.1%} [{lf:.2f},{hf:.2f}]  "
              f"{kt:3d}/{nt:3d}={pt:5.1%} [{lt:.2f},{ht:.2f}]  {ratio:7.2f}")
        rows_out.append((name, pf, pt, ratio))

    print("\n### SUPPORT SIDE  P(gains >=1 supporting doc, flag 4 or 5)")
    for name, arm, rub in ARMS:
        if arm is None:
            continue
        line = f"{name:22s}"
        for pop in ("qa_false", "qa_true"):
            rs = per.get(pop, [])
            k = sum(1 for r in rs if any(f in ("4", "5") for f in arm_docs(r, arm, rub)))
            p, lo, hi = wilson(k, len(rs))
            line += f"  {pop} {k:3d}/{len(rs):3d}={p:5.1%} [{lo:.2f},{hi:.2f}]"
        print(line)

    print("\n### s7 (7-flag pinned weights, eps 0.10) -- REPLACE semantics")
    print(f"{'arm':22s} {'FALSE base':>11s} {'FALSE new':>10s} {'TRUE base':>10s} "
          f"{'TRUE new':>9s} {'AUC':>7s}")
    base_auc = auc([r["s7_base"] for r in res], [1 if r["pop"] == "qa_true" else 0 for r in res])
    print(f"{'baseline':22s} "
          f"{sum(r['s7_base'] for r in per['qa_false'])/max(len(per['qa_false']),1):11.2f} "
          f"{'-':>10s} "
          f"{sum(r['s7_base'] for r in per['qa_true'])/max(len(per['qa_true']),1):10.2f} "
          f"{'-':>9s} {base_auc:7.4f}")
    for name, arm, rub in ARMS:
        if arm is None:
            continue
        sc, y, mean = [], [], {}
        for pop in ("qa_false", "qa_true"):
            vals = [sum(w.get(f, 0.0) for f in arm_docs(r, arm, rub)) for r in per.get(pop, [])]
            mean[pop] = sum(vals) / max(len(vals), 1)
        for r in res:
            sc.append(sum(w.get(f, 0.0) for f in arm_docs(r, arm, rub)))
            y.append(1 if r["pop"] == "qa_true" else 0)
        print(f"{name:22s} "
              f"{sum(r['s7_base'] for r in per['qa_false'])/max(len(per['qa_false']),1):11.2f} "
              f"{mean['qa_false']:10.2f} "
              f"{sum(r['s7_base'] for r in per['qa_true'])/max(len(per['qa_true']),1):10.2f} "
              f"{mean['qa_true']:9.2f} {auc(sc, y):7.4f}")

    print("\n### ORIGIN SOURCE: documents on the speaker's own channel / a blocked platform")
    for rub in ("prod", "subst"):
        for pop in ("qa_false", "qa_true"):
            rs = per.get(pop, [])
            o = [d for r in rs for d in r["docs"]
                 if "hop2_o" in d["arms"] and "hop2_x" not in d["arms"]
                 and (d.get(rub) or {}).get("direction") in FLAGS]
            ref = [d for d in o if d[rub]["direction"] in ("1", "2")]
            sup = [d for d in o if d[rub]["direction"] in ("4", "5")]
            print(f"  {rub:5s} {pop:9s} origin-only docs {len(o):4d} "
                  f"({len(o)/max(len(rs),1):.2f}/claim)  refuting {len(ref):3d}  "
                  f"supporting {len(sup):3d}")
    print("\n  (origin-only = returned by hop2_o and NOT by hop2_x: the speaker's own "
          "channel or a platform the production blocklist drops)")


def show(k: int, pop: str | None, only_directional: bool) -> None:
    res = [json.loads(l) for l in RESULTS.open()]
    rng = random.Random(SEED)
    rng.shuffle(res)
    shown = 0
    for r in res:
        if pop and r["pop"] != pop:
            continue
        dd = [d for d in r["docs"] if d.get("status") == "ok"]
        if only_directional:
            dd = [d for d in dd if (d.get("subst") or {}).get("direction") in
                  ("5", "4", "2", "1") or (d.get("prod") or {}).get("direction") in
                  ("5", "4", "2", "1")]
            if not dd:
                continue
        print("=" * 100)
        print(f"[{r['pop']} v={r['veracity']}] {r['claim'][:250]}")
        h = r["hop1"]
        print(f"  HOP1 speaker     : {h['speaker']}")
        print(f"  HOP1 proposition : {h['proposition'][:200]}")
        print(f"  HOP1 occasion    : {h['occasion'][:160]}  [{h['occasion_date'] or '-'}]")
        print(f"  HOP1 record      : {h['record_name'][:160]}   domains {h['source_domains']}")
        print(f"  Q_prod : {r['query_prod']}")
        print(f"  Q_hop2 : {r['query_hop2']}")
        for d in dd[:5]:
            print(f"    [prod {(d.get('prod') or {}).get('direction')}]"
                  f"[subst {(d.get('subst') or {}).get('direction')}] "
                  f"{'UGC ' if d['ugc'] else ''}{'ORIGIN ' if d['origin_domain'] else ''}"
                  f"{d['domain']:30s} {d['arms']}")
            print(f"      {(d.get('sents') or '')[:300]}")
        shown += 1
        if shown >= k:
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--hop1", action="store_true")
    ap.add_argument("--show-hop1", type=int, default=0)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--pop", default=None)
    ap.add_argument("--directional", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.sample:
        build_sample()
    if a.hop1:
        run_hop1(a.workers, a.smoke)
    if a.show_hop1:
        show_hop1(a.show_hop1)
    if a.run:
        run(a.workers, a.smoke)
    if a.report:
        report()
    if a.show:
        show(a.show, a.pop, a.directional)
