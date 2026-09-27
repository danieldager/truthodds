"""Is "refutation by substitution" worth building? Measure the addressable share first.

THE FINDING IT ANSWERS TO (retrieval_forensics, clog/280826.md 11:40). Of the
all-irrelevant FALSE claims — every one of ten retrieved documents read I — the
dominant query-failure shape is `nonexistent_event`: 41.0% (CN) / 48.0% (fc-gold).
The claim describes an event, ruling, quote, appointment or person that does not
exist, so a positive query has nothing to match and Google returns term-frequency
junk. This is NOT a coverage problem: 76.7% of gold bucket-A claims cite a source
on a well-indexed domain. Two retrieval fixes were already tested on exactly these
claims and BOTH FAILED (+90d ceiling surfaced the reviewer's source 1/25;
fact-check-targeted query 0/25).

DANIEL'S HYPOTHESIS (2026-08-28). Human reviewers refute these by consulting an
AUTHORITATIVE INDEX — a parliament member list, senate.gov, a product label, a
transfer database — and the forensics saw exactly that in the bucket-C residue.
So: do not query the claim, query the SLOT and check its occupant. A claim that
person P was appointed to office O is refuted by finding who actually holds O.

THIS SCRIPT DOES NOT BUILD THE MECHANISM. It measures whether it is worth
building, in two parts:

  --classify     re-label the `nonexistent_event` claims into refutation
                 STRATEGIES (slot_substitution / index_lookup / absence_only /
                 other). Three judges, three families, majority. The strategy is
                 also COMPOSED IN CODE from two independently elicited fields
                 (unique_slot, bounded_register) so the split is auditable rather
                 than one opaque label, and the two are reported side by side.

  --feasibility  actually try it on a sample of slot_substitution claims. The
                 slot query is MODEL-COMPOSED from the claim alone (exactly what
                 an automated version could produce — a hand-written query would
                 flatter the mechanism). It is issued through the production
                 search path (pipeline.search, Serper) at the claim's own date
                 ceiling, and every retrieved document is put through BOTH:
                   - the CURRENT read prompt (read_v5_prompts.READ_SYS_MODE, the
                     exact prompt and claim_block evidence_urn_run.run_claim uses)
                   - an independent two-judge adjudicator asking whether the
                     document names the slot's true occupant and whether a reader
                     could conclude the claim is false from it.
                 THE CRUX is the cross-tab: on documents the adjudicator says are
                 refuting, what does the CURRENT reader flag? If it flags I, the
                 fix needs a new READ flag as well as a new query — two changes,
                 not one.

    uv run python -m eval.scripts.build_eval.slot_substitution_probe --pool
    uv run python -m eval.scripts.build_eval.slot_substitution_probe --classify [--smoke]
    uv run python -m eval.scripts.build_eval.slot_substitution_probe --report
    uv run python -m eval.scripts.build_eval.slot_substitution_probe --feasibility [--n 36]
    uv run python -m eval.scripts.build_eval.slot_substitution_probe --feas-report

Outputs live in eval/data/slot_substitution/ (NEW dir). NOTHING under
eval/data/urn_runs/ is read for anything but input, and nothing there is written.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC))

FORENSICS = SRC / "eval/data/retrieval_forensics/taxonomy.jsonl"
C2 = SRC / "eval/data/urn_runs/c2_false"
E1 = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"
OUT = SRC / "eval/data/slot_substitution"

BASE = "https://api.deepinfra.com/v1/openai"
KEY = next(l.split("=", 1)[1].strip() for l in (SRC / ".env").read_text().splitlines()
           if l.startswith("DEEPINFRA_API_KEY="))

SEED = 20260828
BUDGET_CAP = 1.00

# Same three families as cn_note_target_audit / media_provenance_purge, so this
# measurement is comparable with the other audits and no single model's blind spot
# decides a strategy label.
JUDGES = ["openai/gpt-oss-120b",
          "Qwen/Qwen3-235B-A22B-Instruct-2507",
          "deepseek-ai/DeepSeek-V4-Flash"]
# The adjudicator must NOT be the read model, or the crux cross-tab compares a model
# with itself. DeepSeek-V4-Flash is the reader; it is excluded here.
ADJUDICATORS = ["openai/gpt-oss-120b", "Qwen/Qwen3-235B-A22B-Instruct-2507"]
STRATEGIES = ["slot_substitution", "index_lookup", "absence_only", "other"]

# Denominators, reproduced by retrieval_forensics.py and pinned in clog/280826.md.
POP = {"cn": {"allirr": 335, "false": 1969, "sampled": 200},
       "fg": {"allirr": 494, "false": 1772, "sampled": 200}}


# ---------------------------------------------------------------- the classifier
# No dataset-derived examples. Abstract principles only.

CLASSIFY_SYS = """You are triaging FALSE claims by the kind of evidence that could refute them.

Each claim you see has already been established as false, and has already been diagnosed as describing something that does not exist — an event that never happened, a ruling never issued, a quote never said, an appointment never made, a person or product that is not real. A search engine asked a positive question about such a claim retrieves nothing, because there is no document reporting a non-event.

A human reviewer refutes these anyway. Your job is to decide WHICH refutation route is available for THIS claim. Answer three questions independently.

1. unique_slot — does the claim assert that some named entity occupies a position that, at the relevant time, exactly ONE entity occupies? Such a position is a role or office, a title, a job, a record, a rank in an ordering, the winner of a contest, the signatory of a document, the manufacturer or owner of a thing, the club a player plays for, the cause of an event, the perpetrator or victim of a specific incident. The test is exclusivity: if the claim is true, then the entity named is the one and only occupant, and if anyone else occupies it the claim is false. A claim that someone merely attended, said, supported, visited or was involved in something asserts no such exclusive position.
   Set unique_slot true only when the exclusivity holds. Give slot_description as the position stated in words that do not name the claimed entity, and claimed_occupant as the entity the claim puts in it. Leave both empty strings when false.

2. bounded_register — is there some authoritative, publicly checkable register that would NECESSARILY list this fact if it were true, and whose completeness is the whole point of its existence? Rosters and membership lists, official schedules and fixture lists, statute books and legal codes, court dockets and case databases, company and property filings, patent and trademark registers, licence and approval databases, product labels and ingredient declarations, official results tables, transfer and transaction databases.
   The bar is completeness, not authority. A news archive is NOT a bounded register: "no outlet reported it" is silence, not absence from a list. Neither is a general encyclopedia, nor a search engine, nor social media.
   Give register_name when true, empty string when false.

3. strategy — the single best refutation route for this claim.
   "slot_substitution": query the position, find who or what ACTUALLY occupies it, and the mismatch refutes the claim. Available only when a unique slot exists AND its true occupant is the kind of fact the public record states.
   "index_lookup": query a bounded register and refute by ABSENCE FROM IT — the entry that would have to be there is not. There need be no competing occupant.
   "absence_only": neither route exists. Nothing occupies a slot the claim fills, and no bounded list would have to record it. The only refutation is that no credible source reports it — a fabricated quote with no transcript for the occasion, a fabricated incident, a fabricated statistic, an invented anecdote.
   "other": genuinely neither. Say what in other_kind.

Discipline, all of it load-bearing:

- Question 3 asks what WOULD WORK, not what a search engine happens to return today. Do not reason about search difficulty; reason about whether the fact has a determinate public answer.

- Prefer slot_substitution over index_lookup when both are available: a competing occupant is affirmative evidence, whereas absence from a list requires trusting the list is complete.

- Prefer index_lookup over absence_only only when you can name the register. If the register you would name is "news coverage", "the internet", "official statements" or anything else unbounded, the answer is absence_only.

- Do not treat the existence of a person or organisation as a slot. "Person P does not exist" is checked by absence, not by finding someone else who is P.

- A claim about a quantity — a death toll, a price, a poll number, a temperature — has a unique slot when an authority states the true figure for that exact measurement, and the true figure contradicts the claimed one. If the measurement itself is not one an authority publishes, it is absence_only.

- Judge only the claim as written. Do not use your own knowledge of what the truth is, only whether a determinate public answer exists for the position or the list.

- If the claim contains several assertions, judge the one that makes it false.

Return only JSON:
{"unique_slot": true|false, "slot_description": "<...>", "claimed_occupant": "<...>", "bounded_register": true|false, "register_name": "<...>", "strategy": "<one of slot_substitution|index_lookup|absence_only|other>", "other_kind": "<only when strategy is other>", "reason": "<one sentence>"}"""


# ------------------------------------------------------- the slot-query composer
# Sees the CLAIM and its date and nothing else — the same information
# evidence_urn_run's QUERY_SYS gets. A hand-written query would use knowledge the
# production system does not have, so this is the honest ceiling.

SLOTQ_SYS = """You write ONE web search query for a fact-checking system.

The claim you are given asserts that a named entity occupies a position that exactly one entity can occupy at a time — an office, a role, a title, a record, a rank, a winner, a signatory, an owner, a cause.

Do NOT search for the claim. Searching for it returns nothing when it is false.

Instead, write the query that finds who or what ACTUALLY occupies that position at the claim's date. Name the position, the organisation or context it belongs to, and the date or year. DO NOT put the claimed entity's name in the query — including it is what makes the search fail.

The query must be one a general web search engine can answer from ordinary public sources. At most 12 words. No quotes unless a distinctive official title requires them.

Return only JSON:
{"slot": "<the position, in words, without the claimed entity's name>", "query": "<the search query>"}"""


# ------------------------------------------------------------- the adjudicator
# Independent of the read prompt. Asks the substitution question directly, so the
# cross-tab against the current reader's flag is a comparison of two different
# rubrics on the same document, not a model agreeing with itself.

ADJ_SYS = """You are judging whether one web document could be used to refute one false claim by SUBSTITUTION.

The claim asserts that a named entity occupies a position that exactly one entity can occupy at a time. The refutation route being tested is: find who or what ACTUALLY occupies that position, and show it is someone or something else.

You get the claim, the position being tested, and numbered sentences from one document. Judge only what THIS document says. Do not use your own knowledge of who holds the position.

Answer three questions.

1. names_occupant — does the document state who or what actually occupies the position, at or around the claim's date? True only if a specific occupant is named or identified; false if the document is merely about the same topic, discusses the position without naming its holder, or names a holder for a clearly different period.

2. contradicts_claim — is the occupant the document names DIFFERENT from the entity the claim puts in the position? True only when the document names an occupant AND that occupant is not the claim's entity. False when the document names no occupant, or names the same entity.

3. reader_could_conclude — could a careful reader, holding this document and the claim side by side, reasonably conclude the claim is FALSE? Require that the document settle the position for the claim's own time, and that the reader need add nothing but the observation that the two names differ. If the reader would still need another source, or would have to assume the document is complete or current, answer false.

Discipline:
- The document will almost certainly never mention the claim or the entity it names. That is expected and is not a reason to answer false — the whole point of substitution is that the true occupant's page has no reason to mention a person who was never appointed.
- Do not credit a document that only shows the position exists, or lists past holders ending before the claim's date, or describes the organisation in general.
- Mind the dates. A document naming an occupant for a period that does not contain the claim's date does not settle the claim.
- Absence of the claim's entity from the document is not by itself a refutation unless the document is stating the position's actual occupant.

Return only JSON:
{"names_occupant": true|false, "named_occupant": "<the occupant the document names, else empty>", "contradicts_claim": true|false, "reader_could_conclude": true|false, "reason": "<one sentence>"}"""


# ------------------------------------------------------------------- plumbing

def _chat(model: str, sys_prompt: str, user: str, retries: int = 4,
          max_tokens: int = 700) -> dict:
    payload = {"model": model, "temperature": 0, "max_tokens": max_tokens,
               "response_format": {"type": "json_object"},
               "messages": [{"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user}]}
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    last = None
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode())
            m = re.search(r"\{.*\}", d["choices"][0]["message"]["content"] or "", re.S)
            out = json.loads(m.group(0)) if m else {}
            out["_cost"] = (d.get("usage") or {}).get("estimated_cost") or 0
            return out
        except Exception as e:
            last = e
            time.sleep(3 * 2 ** a)
    raise RuntimeError(f"{type(last).__name__}: {last}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def _runner(jobs, out_path: Path, tag: str, workers: int, work_one):
    """Shared worker pool. Progress + throughput + ETA, flushed; hard budget cap."""
    print(f"{tag}: {len(jobs)} calls | cap ${BUDGET_CAP} | {workers} workers", flush=True)
    lock, st, t0 = threading.Lock(), {"cost": 0.0, "n": 0, "fail": 0}, time.time()
    fh = open(out_path, "a")

    def work(job):
        try:
            rec, cost = work_one(job)
        except Exception as e:
            with lock:
                st["n"] += 1
                st["fail"] += 1
                print(f"  [failed] {job!s:.90s}: {type(e).__name__} {e}", flush=True)
            return
        with lock:
            st["cost"] += cost
            st["n"] += 1
            if rec is not None:
                fh.write(json.dumps(rec) + "\n")
            else:
                st["fail"] += 1
            n, el = st["n"], (time.time() - t0) / 60
            if n % 25 == 0 or n == len(jobs):
                fh.flush()
                print(f"  {n}/{len(jobs)} | ${st['cost']:.4f} | {el:.1f}m | "
                      f"{n / max(el, .01):.0f}/min | ETA {el / n * (len(jobs) - n):.1f}m | "
                      f"fail {st['fail']}", flush=True)
            if st["cost"] > BUDGET_CAP:
                raise SystemExit(f"BUDGET CAP ${BUDGET_CAP} HIT at {n} calls")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, jobs))
    fh.close()
    print(f"{tag} done {st['n']} | ${st['cost']:.4f} | {(time.time() - t0) / 60:.1f}m | "
          f"failed {st['fail']}", flush=True)
    return st


# ------------------------------------------------------------------- the pool

def build_pool() -> list[dict]:
    """The `nonexistent_event` claims from the forensics taxonomy, joined back to
    the run records so the feasibility arm has the date ceiling and exclusions the
    original run used."""
    tax = [json.loads(l) for l in FORENSICS.open()]
    ne = [t for t in tax if t["code"] == "nonexistent_event"]

    cn = {}
    for p in ("scores.jsonl", "scores_ext.jsonl"):
        for line in (C2 / p).open():
            r = json.loads(line)
            cn[r["post_id"]] = r
    fg = {}
    for line in E1.open():
        r = json.loads(line)
        fg[r["review_url"]] = r

    rows, miss = [], 0
    for t in ne:
        arm = t["arm"]
        corpus = "cn" if arm.startswith("cn") else "fg"
        key = t["id"].split(":", 1)[1]
        src = (cn if corpus == "cn" else fg).get(key)
        if src is None:
            miss += 1
            continue
        rows.append({
            "id": t["id"], "arm": arm, "corpus": corpus,
            "allirr": arm.endswith("allirr"),
            "claim": src.get("claim_resolved") or src["claim_text"],
            "claim_date": src.get("claim_date_shown") or "",
            "ceiling": src.get("ceiling") or "",
            "publisher_site": src.get("publisher_site") or "",
            "orig_query": t["query"], "orig_got": t["got"], "orig_why": t["why"],
            "topic": src.get("topic"), "claim_type": src.get("claim_type"),
        })
    rows.sort(key=lambda r: r["id"])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pool.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    c = Counter(r["arm"] for r in rows)
    print(f"pool: {len(rows)} nonexistent_event claims ({miss} unjoinable) {dict(c)}",
          flush=True)
    return rows


def load_pool() -> list[dict]:
    return [json.loads(l) for l in (OUT / "pool.jsonl").open()]


# ---------------------------------------------------------------- --classify

def _norm_cls(o: dict) -> dict | None:
    s = str(o.get("strategy", "")).strip().lower()
    if s not in STRATEGIES:
        return None
    return {"unique_slot": bool(o.get("unique_slot")),
            "slot_description": str(o.get("slot_description") or "")[:200],
            "claimed_occupant": str(o.get("claimed_occupant") or "")[:120],
            "bounded_register": bool(o.get("bounded_register")),
            "register_name": str(o.get("register_name") or "")[:120],
            "strategy": s, "other_kind": str(o.get("other_kind") or "")[:120],
            "reason": str(o.get("reason") or "")[:200]}


def cls_payload(r: dict) -> str:
    d = f"\nCLAIM DATE: {r['claim_date']}" if r["claim_date"] else ""
    return f"CLAIM: {r['claim']}{d}"


def classify(smoke: bool, workers: int) -> None:
    rows = load_pool()
    if smoke:
        rows = random.Random(SEED).sample(rows, 20)
        print(f"SMOKE: {len(rows)} claims", flush=True)
    path = OUT / ("judged_smoke.jsonl" if smoke else "judged.jsonl")
    done = {(j["id"], j["model"]) for j in (json.loads(l) for l in path.open())} \
        if path.exists() else set()
    jobs = [(r, m) for r in rows for m in JUDGES if (r["id"], m) not in done]

    def one(job):
        r, model = job
        # 1500, not the 700 default: gpt-oss-120b is a reasoning model and ate a
        # smaller budget before emitting JSON (8/20 empty at 700, the same failure
        # cn_note_target_audit hit at 400). Never counted as done — they resume.
        o = _chat(model, CLASSIFY_SYS, cls_payload(r), max_tokens=1500)
        cost = o.pop("_cost", 0)
        v = _norm_cls(o)
        return (None if v is None else {"id": r["id"], "arm": r["arm"],
                                        "corpus": r["corpus"], "model": model, **v}), cost

    _runner(jobs, path, "classify" + (" SMOKE" if smoke else ""), workers, one)


def code_strategy(v: dict) -> str:
    """Strategy composed IN CODE from the two independently elicited fields, so the
    split is auditable and not one opaque label."""
    if v["unique_slot"]:
        return "slot_substitution"
    if v["bounded_register"]:
        return "index_lookup"
    return "absence_only"


def majority(labels: list[str]) -> str | None:
    c = Counter(labels).most_common()
    return c[0][0] if c and c[0][1] >= 2 else None


def load_labels(smoke: bool = False) -> dict[str, dict]:
    path = OUT / ("judged_smoke.jsonl" if smoke else "judged.jsonl")
    by = defaultdict(dict)
    for l in path.open():
        j = json.loads(l)
        by[j["id"]][j["model"]] = j
    return by


def report(smoke: bool) -> None:
    pool = {r["id"]: r for r in load_pool()}
    by = load_labels(smoke)
    full = {k: v for k, v in by.items() if len(v) == len(JUDGES)}
    print(f"\n=== strategy split | n={len(full)} of {len(by)} with all {len(JUDGES)} "
          f"judges ===", flush=True)

    lab, codelab = {}, {}
    for k, v in full.items():
        m = majority([v[j]["strategy"] for j in JUDGES])
        cm = majority([code_strategy(v[j]) for j in JUDGES])
        if m:
            lab[k] = m
        if cm:
            codelab[k] = cm

    for arm_name, pred in (("CN all-irrelevant", lambda r: r["arm"] == "cn_allirr"),
                           ("fc-gold all-irrelevant", lambda r: r["arm"] == "fg_allirr"),
                           ("CN control", lambda r: r["arm"] == "cn_ctrl"),
                           ("fc-gold control", lambda r: r["arm"] == "fg_ctrl")):
        ids = [k for k in lab if pred(pool[k])]
        if not ids:
            continue
        n = len(ids)
        print(f"\n  {arm_name}  n={n}")
        c = Counter(lab[k] for k in ids)
        cc = Counter(codelab[k] for k in ids if k in codelab)
        for s in STRATEGIES:
            p, lo, hi = wilson(c[s], n)
            print(f"    {s:18s} {c[s]:4d}  {p * 100:5.1f}%  [{lo * 100:4.1f}, {hi * 100:4.1f}]"
                  f"   code-composed {cc[s] / max(n, 1) * 100:5.1f}%")
        agree = sum(1 for k in ids if lab.get(k) == codelab.get(k))
        print(f"    direct-vs-code agreement {agree}/{n} = {agree / n * 100:.1f}%")
        unan = sum(1 for k in ids if len({full[k][j]["strategy"] for j in JUDGES}) == 1)
        print(f"    unanimous {unan}/{n} = {unan / n * 100:.1f}%")
        per = {j: Counter(full[k][j]["strategy"] for k in ids) for j in JUDGES}
        for j in JUDGES:
            print(f"      {j:38s} " + "  ".join(
                f"{s[:4]} {per[j][s] / n * 100:4.1f}%" for s in STRATEGIES))

    # ------- size of the prize.
    # The interval is taken DIRECTLY on the forensics' 200-claim sample of each
    # all-irrelevant arm (verified a spread random draw of the 335 / 494, not a
    # head-of-file slice), so it propagates BOTH the nonexistent_event rate and the
    # strategy rate. Multiplying two point estimates would understate the width.
    print("\n=== SIZE OF THE PRIZE (share of the whole all-irrelevant block, and of "
          "each corpus's FALSE claims) ===")
    for corp, name in (("cn", "CN-false urn"), ("fg", "fc-gold")):
        ids = [k for k in lab if pool[k]["corpus"] == corp and pool[k]["allirr"]]
        c = Counter(lab[k] for k in ids)
        blk, fal, samp = POP[corp]["allirr"], POP[corp]["false"], POP[corp]["sampled"]
        ne = sum(1 for k in pool if pool[k]["corpus"] == corp and pool[k]["allirr"])
        print(f"\n  {name}: all-irrelevant {blk} of {fal} FALSE = {blk / fal * 100:.1f}%; "
              f"nonexistent_event {ne} of the {samp}-claim sample of the block "
              f"= {ne / samp * 100:.1f}% ({round(blk * ne / samp)} claims); "
              f"{ne - len(ids)} of them split 3 ways and carry no majority strategy, "
              f"so every row below is a LOWER bound by up to {(ne - len(ids)) / samp * 100:.1f}pp")
        for s in STRATEGIES:
            p, lo, hi = wilson(c[s], samp)     # denominator = the WHOLE all-irr sample
            print(f"    {s:18s} {c[s]:3d}/{samp} of the all-irrelevant block = "
                  f"{p * 100:5.1f}% [{lo * 100:4.1f}, {hi * 100:4.1f}]"
                  f"  ({round(blk * p):4d} claims)"
                  f"  ->  {p * blk / fal * 100:5.2f}% of FALSE "
                  f"[{lo * blk / fal * 100:.2f}, {hi * blk / fal * 100:.2f}]")

    # ------- what "other" actually is, and the slots/registers named, read as examples
    oth = [full[k][j].get("other_kind") for k in lab if lab[k] == "other" for j in JUDGES
           if full[k][j]["strategy"] == "other" and full[k][j].get("other_kind")]
    if oth:
        print("\n  other_kind, verbatim:", Counter(oth).most_common(12))
    regs = [full[k][j]["register_name"] for k in lab if lab[k] == "index_lookup"
            for j in JUDGES if full[k][j]["register_name"]]
    print("\n  registers named (index_lookup):", Counter(regs).most_common(15))


# ------------------------------------------------------------- --feasibility

def pick_feasibility(n: int) -> list[dict]:
    pool = {r["id"]: r for r in load_pool()}
    by = load_labels(False)
    full = {k: v for k, v in by.items() if len(v) == len(JUDGES)}
    # slot_substitution turned out RARE: only 21 all-irrelevant claims carry a
    # majority label (5 CN + 16 gold), short of the 30-40 the feasibility arm wants.
    # So the sample is the MAJORITY set plus a top-up from claims where at least one
    # judge said slot_substitution — the most generous population the labels support,
    # which is the right way to be wrong about a mechanism you are trying to kill.
    # `votes` is carried through so the two strata are reported SEPARATELY, never pooled.
    def nvote(k):
        return sum(full[k][j]["strategy"] == "slot_substitution" for j in JUDGES)

    maj = sorted(k for k, v in full.items() if pool[k]["allirr"] and nvote(k) >= 2)
    one = sorted(k for k, v in full.items() if pool[k]["allirr"] and nvote(k) == 1)
    rnd = random.Random(SEED)
    out = list(maj) + rnd.sample(one, min(max(n - len(maj), 0), len(one)))
    rows = []
    for k in sorted(out):
        r = dict(pool[k])
        v = full[k]
        r["slot_votes"] = [v[j]["slot_description"] for j in JUDGES if v[j]["slot_description"]]
        r["occupant_votes"] = [v[j]["claimed_occupant"] for j in JUDGES if v[j]["claimed_occupant"]]
        r["n_slot_votes"] = nvote(k)
        r["stratum"] = "majority" if nvote(k) >= 2 else "single_vote"
        rows.append(r)
    return rows


def feasibility(n: int, workers: int) -> None:
    from pipeline.search import search, scrape   # noqa: E402
    from eval.scripts.build_eval.evidence_urn_run import (  # noqa: E402
        select_regions, read_doc, aggregate_reads, SNIPPET_MIN_TEXT, _domain_of)

    rows = pick_feasibility(n)
    with open(OUT / "feas_pool.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"feasibility sample: {len(rows)} slot_substitution claims "
          f"({sum(r['corpus'] == 'cn' for r in rows)} CN / "
          f"{sum(r['corpus'] == 'fg' for r in rows)} fc-gold)", flush=True)

    path = OUT / "feas.jsonl"
    done = {json.loads(l)["id"] for l in path.open()} if path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]

    # Serper pacing: reuse the runner's own gate so this job obeys the same window.
    from eval.scripts.build_eval.evidence_urn_run import _serper_gate  # noqa: E402

    def one(r):
        cost = 0.0
        d = f"\nCLAIM DATE: {r['claim_date']}" if r["claim_date"] else ""
        q = _chat(JUDGES[2], SLOTQ_SYS, f"CLAIM: {r['claim']}{d}", max_tokens=200)
        cost += q.pop("_cost", 0)
        query = str(q.get("query") or "").strip()[:300]
        slot = str(q.get("slot") or "").strip()[:200]
        if not query:
            raise RuntimeError("slot-query composition returned nothing")

        _serper_gate()
        xd = [r["publisher_site"]] if r["publisher_site"] else []
        hits = search(query, 10, date_ceiling=r["ceiling"] or None,
                      exclude_domains=xd, min_results=0, provider="serper")

        # The EXACT claim block evidence_urn_run.run_claim builds, so the read is
        # the production read and not a variant of it.
        claim_block = (f"CLAIM: {r['claim']}\n"
                       f"(claimed on {r['claim_date'] or 'unknown date'}; judge the "
                       f"document's bearing on this exact proposition)")
        adj_head = (f"CLAIM: {r['claim']}\n"
                    f"CLAIM DATE: {r['claim_date'] or 'unknown'}\n"
                    f"POSITION BEING TESTED: {slot}\n")

        docs = []
        for rank, h in enumerate(hits[:10], 1):
            url = h.get("url") or ""
            text = h.get("content") or scrape(url)
            prov = "scrape"
            if not text or len(text) < SNIPPET_MIN_TEXT:
                text, prov = h.get("snippet") or "", "snippet"
            regions, prep = select_regions(text, r["claim"], query)
            smap = {}
            for ids, sel in regions:
                for i, sn in zip(ids, sel):
                    smap[i] = sn
            entry = {"rank": rank, "url": url, "domain": _domain_of(url),
                     "date": h.get("date"), "provenance": prov, "n_sents": len(smap)}
            if not smap:
                entry["read"] = {"direction": "I", "evidence": [], "qc_flag": "empty-doc-code"}
                entry["adj"] = {}
                docs.append(entry)
                continue
            reads = []
            for ids, sel in regions:
                out = None
                for _ in range(2):
                    try:
                        out, c, _, _ = read_doc(claim_block, ids, sel, r["id"])
                        cost += c
                    except Exception:
                        time.sleep(2)
                        continue
                    if out:
                        break
                reads.append(out or {"direction": "I", "evidence": [], "qc_flag": "read-failed"})
            entry["read"] = aggregate_reads(reads)
            doc_txt = "\n".join(f"[{i}] {smap[i]}" for i in sorted(smap))
            adj = {}
            for m in ADJUDICATORS:
                try:
                    o = _chat(m, ADJ_SYS, f"{adj_head}\nDOCUMENT:\n{doc_txt}")
                    cost += o.pop("_cost", 0)
                    adj[m] = {"names_occupant": bool(o.get("names_occupant")),
                              "named_occupant": str(o.get("named_occupant") or "")[:120],
                              "contradicts_claim": bool(o.get("contradicts_claim")),
                              "reader_could_conclude": bool(o.get("reader_could_conclude")),
                              "reason": str(o.get("reason") or "")[:200]}
                except Exception as e:
                    adj[m] = {"error": f"{type(e).__name__}"}
            entry["adj"] = adj
            docs.append(entry)
        return {"id": r["id"], "corpus": r["corpus"], "stratum": r["stratum"],
                "n_slot_votes": r["n_slot_votes"], "claim": r["claim"],
                "claim_date": r["claim_date"], "ceiling": r["ceiling"],
                "orig_query": r["orig_query"], "slot": slot, "slot_query": query,
                "n_hits": len(hits), "docs": docs}, cost

    _runner(todo, path, "feasibility", workers, one)


def feas_report() -> None:
    rows = [json.loads(l) for l in (OUT / "feas.jsonl").open()]
    print(f"\n=== FEASIBILITY | {len(rows)} slot_substitution claims, "
          f"model-composed slot query, production Serper path ===")
    n = len(rows)
    docs_all = [(r, d) for r in rows for d in r["docs"]]
    print(f"  documents retrieved: {len(docs_all)} ({len(docs_all) / n:.1f}/claim)")

    def av(d, field, rule="or"):
        vs = [a.get(field) for a in d["adj"].values() if "error" not in a]
        if not vs:
            return None
        return (any(vs) if rule == "or" else all(vs))

    for rule in ("or", "and"):
        occ = [r for r in rows if any(av(d, "names_occupant", rule) for d in r["docs"])]
        con = [r for r in rows if any(av(d, "contradicts_claim", rule) for d in r["docs"])]
        con3 = [r for r in rows if any(av(d, "contradicts_claim", rule)
                                       for d in r["docs"] if d["rank"] <= 3)]
        rc = [r for r in rows if any(av(d, "reader_could_conclude", rule) for d in r["docs"])]
        tag = "either adjudicator" if rule == "or" else "BOTH adjudicators"
        print(f"\n  -- {tag} --")
        for lbl, s in (("true occupant named anywhere in top-10", occ),
                       ("occupant named AND differs from the claim", con),
                       ("  ...and it is in the top-3", con3),
                       ("reader could conclude the claim is FALSE", rc)):
            p, lo, hi = wilson(len(s), n)
            print(f"    {lbl:44s} {len(s):3d}/{n}  {p * 100:5.1f}%  "
                  f"[{lo * 100:4.1f}, {hi * 100:4.1f}]")

    # ---- THE CRUX: what does the CURRENT read prompt flag those documents?
    print("\n  === THE CRUX: the CURRENT read prompt (read-v5) on the refuting documents ===")
    for lbl, rule in (("adjudicators AGREE the doc refutes", "and"),
                      ("EITHER adjudicator says the doc refutes", "or")):
        sel = [d for _, d in docs_all if av(d, "reader_could_conclude", rule)]
        if not sel:
            print(f"    {lbl}: none")
            continue
        c = Counter(d["read"]["direction"] for d in sel)
        print(f"    {lbl}  n={len(sel)} docs")
        for f in "54321XI":
            if c[f]:
                p, lo, hi = wilson(c[f], len(sel))
                print(f"      flag {f}: {c[f]:4d}  {p * 100:5.1f}%  "
                      f"[{lo * 100:4.1f}, {hi * 100:4.1f}]")
        di = sum(c[f] for f in "12")
        p, lo, hi = wilson(di, len(sel))
        print(f"      refuting flag (1 or 2): {di}/{len(sel)} = {p * 100:.1f}% "
              f"[{lo * 100:.1f}, {hi * 100:.1f}]")
    allc = Counter(d["read"]["direction"] for _, d in docs_all)
    print(f"    ALL retrieved docs, for contrast: " +
          "  ".join(f"{f} {allc[f] / len(docs_all) * 100:.1f}%" for f in "54321XI"))

    # claim-level: would the new dossier flip the claim off all-irrelevant?
    flip = [r for r in rows if any(d["read"]["direction"] in "12" for d in r["docs"])]
    p, lo, hi = wilson(len(flip), n)
    print(f"\n    claims where the CURRENT reader emits ANY refuting flag on the new "
          f"dossier: {len(flip)}/{n} = {p * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]")
    nonI = [r for r in rows if any(d["read"]["direction"] != "I" for d in r["docs"])]
    p, lo, hi = wilson(len(nonI), n)
    print(f"    claims no longer ALL-IRRELEVANT under the current reader: "
          f"{len(nonI)}/{n} = {p * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]")

    print("\n  -- examples (claim | slot query | best doc | adjudicator | read flag) --")
    for r in rows[:12]:
        best = None
        for d in r["docs"]:
            if av(d, "reader_could_conclude", "or"):
                best = d
                break
        print(f"\n   * {r['claim'][:110]}")
        print(f"     orig query : {r['orig_query'][:100]}")
        print(f"     slot query : {r['slot_query'][:100]}")
        if best is None:
            print(f"     no refuting doc; flags "
                  f"{Counter(d['read']['direction'] for d in r['docs']).most_common()}")
            continue
        a = next(iter(best["adj"].values()))
        print(f"     doc #{best['rank']} {best['domain']} -> occupant "
              f"{a.get('named_occupant', '')[:60]!r} | read flag "
              f"{best['read']['direction']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--feasibility", action="store_true")
    ap.add_argument("--feas-report", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.pool:
        build_pool()
    if a.classify:
        classify(a.smoke, a.workers)
    if a.report:
        report(a.smoke)
    if a.feasibility:
        feasibility(a.n, a.workers)
    if a.feas_report:
        feas_report()


if __name__ == "__main__":
    main()
