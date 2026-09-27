"""Paired offline A/B of READ prompt arms on saved pilot reads.

Re-reads the EXACT sentences each region showed the model in the E1 pilot
(reconstructed from prep.read_regions spans + sent_ids/sents), so arms differ
only in the prompt (and optionally the model). No search, no scraping.

Arms
  A0   read-v4, as run in the pilot (baseline + temp-0 noise floor)
  A2   read-v5-scoped: scope and direction as SEPARATE fields, principled rules only
  A2L  A2 without the diagnostic slot_diff field

A0->A2 measures the new scheme; --model on either arm isolates reader capacity.

  uv run python -m eval.scripts.build_eval.read_ab --arms A0,A2 --limit 30
"""

# ============================================================================
# DO NOT RUN — the arms this harness tests were ABANDONED (2026-07-25/27).
#   A1  cut by Daniel: it tuned the prompt to dataset errors.
#   A2 / A2L  their win was RETRACTED. The agent-generated read labels used as the
#             yardstick over-demote (93% demotion rate vs Daniel's 63% on the same
#             reads); A2 shares that bias (90%), so its measured gain was largely
#             rewarded confound, not improvement. Two LLM panels agreeing with each
#             other while both diverge from the human anchor is correlated bias,
#             not evidence.
# The READ step is being rebuilt as a TWO-PASS scheme (pass 1 = cheap relevance
# selection, pass 2 = stance extraction over the union). Kept only because the
# region-reconstruction logic (regions_for) is reusable and because the file is the
# record of what was tried. Its recent mtime is misleading — nothing here is live.
# ============================================================================

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import eval.scripts.build_eval.evidence_urn_run as R

PILOT = os.path.join(os.path.dirname(R.__file__), "..", "..", "data", "urn_runs", "e1",
                     "results-00.jsonl")
PILOT = os.path.abspath(PILOT)

# ---------------------------------------------------------------- prompt arms

# Principles of evidence reading only. Nothing here names a failure observed in a prior
# run: each rule is one a careful reader would be given before seeing any data
# (evidence independence, verdict-is-not-evidence, judge-only-what-is-shown).
_CONTENT_RULES = """- A document that only repeats the claim is not independent evidence for it. Where its sole claim-bearing text relays the assertion without contributing an observation, record, measurement or named source of its own, it cannot support the claim; cite it as neutral. A first-party document — the named actor's own words, or an institution's own account of what it did — IS contributing evidence.
- Someone else's verdict on the claim is not evidence about the claim. Where the document adjudicates, rates or characterises the claim, judge only the underlying facts it reports."""

_SLOTS = """First fix the claim's SLOTS: who (actor, speaker), what (act, object, scheme, product, exact wording), where (jurisdiction, place), when (date, period), how much (quantity, population, measure), and each separate thing it asserts. A slot MATCHES even when the document names it differently — synonym, common/brand/scientific name, acronym, later name, or a definite description of the same person or event. A slot MISMATCHES when the document gives an explicitly DIFFERENT value, or asserts a weaker, narrower, partial or merely intended version of what the claim asserts as done. A slot the document simply does not mention is NOT a mismatch."""

_A2_TEMPLATE = """You judge ONE web document against ONE factual claim for an academic misinformation-research measurement. You see the claim, then the document as numbered sentences. Judge STRICTLY from what the document text asserts — never from your own knowledge of the claim.

%%SLOTS%%

Answer TWO separate questions, in this order.

FIRST "scope" — what is this document about, relative to the claim? Exactly one of:
- "exact": it discusses the claim's own proposition, with every slot the claim names either matching or unmentioned.
- "adjacent": it discusses the same subject but at least one slot the claim names has an explicitly DIFFERENT value, or it covers only part of a multi-part claim.
- "off_claim": readable content in which no slot of the claim appears — a different subject, however thematically adjacent.
- "unreadable": no readable propositional content — cookie/consent/paywall/bot-check walls, navigation shells, link farms, index pages, table dumps, error pages.

%%SLOTFIELD%%THEN "direction" — only if scope is "exact" or "adjacent". Which way does what the document asserts point, for the proposition it is discussing?
- "supports": it asserts, reports or evidences that proposition is true. Minor caveats still count.
- "refutes": it asserts a fact incompatible with it. Not merely making it implausible.
- "none": it takes no position — background, unresolved dispute, attributed positions without endorsement, or a relay of the claim.
If scope is "off_claim" or "unreadable", direction is "none".

"support" / "neutral" / "against" — lists of sentence NUMBERS, AT MOST 8 per list, most probative first. Cite a sentence if it asserts something bearing on a slot of the claim. "support"/"against" carry the sentences asserting or denying the proposition; every other on-claim sentence goes in "neutral". Background, biography, definitions and related events belong in NO list; most sentences of most documents belong in no list. If direction is "supports" or "refutes", the matching list must be non-empty. If scope is "exact" or "adjacent" and direction is "none", "neutral" must be non-empty — if you cannot cite one sentence, scope is "off_claim". If scope is "off_claim" or "unreadable", all three lists are empty.

%%RULES%%

You may be shown ONE CONTIGUOUS REGION of a longer document; sentence numbering shows its position, and a leading context block may be prepended (numbering will jump) — its sentences are citable. Judge only WHAT YOU ARE SHOWN. A search-result snippet is judged the same way, but asserts only what it literally says.

Respond JSON only: {"scope": "...", %%SLOTJSON%%"direction": "...", "support": [..], "neutral": [..], "against": [..]}"""

_SLOT_TEXT = """"slot_diff" — if scope is "adjacent", which slot has the differing value, using exactly one of these names: "who" | "what" | "where" | "when" | "how_much" | "population" | "modality" (the act is intended or permitted rather than done) | "conjunct" (only part of a multi-part claim). Otherwise "".

"""

_A2_BASE = _A2_TEMPLATE.replace("%%SLOTS%%", _SLOTS).replace("%%RULES%%", _CONTENT_RULES)
A2_SYS = _A2_BASE.replace("%%SLOTFIELD%%", _SLOT_TEXT).replace("%%SLOTJSON%%", '"slot_diff": "...", ')
# A2L drops the diagnostic-only slot_diff field: isolates its cost in judgment quality.
A2L_SYS = _A2_BASE.replace("%%SLOTFIELD%%", "").replace("%%SLOTJSON%%", "")

ARMS = {"A0": ("read-v4", R.READ_SYS), "A2": ("read-v5-scoped", A2_SYS),
        "A2L": ("read-v5-scoped-noslot", A2L_SYS)}

SCOPES = {"exact", "adjacent", "off_claim", "unreadable"}


def collapse_a2(obj: dict) -> str:
    """Map the two-field scheme onto the v4 alphabet (conservative collapse:
    adjacent-directional -> neutral). The graded reading keeps them distinct."""
    sc, d = obj.get("scope"), obj.get("direction")
    if sc == "off_claim":
        return "irrelevant"
    if sc == "unreadable":
        return "junk"
    if sc == "exact" and d in ("supports", "refutes"):
        return d
    return "neutral"


# --------------------------------------------------------------- input rebuild

def regions_for(res: dict):
    """Rebuild the exact (ids, sents) each region showed the model."""
    ids, sents = res["sent_ids"], res["sents"]
    m = dict(zip(ids, sents))
    prep = res.get("prep") or {}
    rr = prep.get("read_regions")
    if not prep.get("windowed") or not rr:
        return [(list(ids), list(sents))]
    out = []
    for r in rr:
        a, b = r["span"]
        lede = [i for i in ids if i < a and i <= 3] if a > 3 else []
        rids = lede + [i for i in range(a, b + 1) if i in m]
        out.append((rids, [m[i] for i in rids]))
    return out


def check_reconstruction(recs) -> tuple[int, int]:
    ok = bad = 0
    for rec in recs:
        for res in rec["results"]:
            if not res.get("sents"):
                continue
            union = sorted({i for ids, _ in regions_for(res) for i in ids})
            (ok, bad) = (ok + 1, bad) if union == sorted(res["sent_ids"]) else (ok, bad + 1)
    return ok, bad


# ------------------------------------------------------------------- read call

_LOCK = threading.Lock()
_STATE = {"cost": 0.0, "done": 0}


def read_arm(arm: str, claim_block: str, ids, sents, key: str, model: str | None):
    sys_prompt = ARMS[arm][1]
    doc = "\n".join(f"[{i}] {s}" for i, s in zip(ids, sents))
    body_model = model or R.VERIFICATION_MODEL
    obj, cost, cached, ptok = _llm(sys_prompt, f"{claim_block}\n\nDOCUMENT:\n{doc}", key, body_model)
    idset = set(ids)
    ok_ptr = all(isinstance(i, int) and i in idset
                 for f in ("support", "neutral", "against") for i in (obj.get(f) or []))
    counts = {f: sorted(set(obj.get(f) or [])) for f in ("support", "neutral", "against")}
    if arm.startswith("A2"):
        if obj.get("scope") not in SCOPES or not ok_ptr:
            return None, cost
        return {"scope": obj["scope"], "slot_diff": obj.get("slot_diff") or "",
                "direction_raw": obj.get("direction") or "none",
                "direction": collapse_a2(obj), **counts}, cost
    if obj.get("direction") not in R.DIRECTIONS or not ok_ptr:
        return None, cost
    return {"direction": obj["direction"], **counts}, cost


def _llm(sys_prompt, user, key, model):
    body = {"model": model, "temperature": 0, "max_tokens": 400,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": user}]}
    if key:
        body["prompt_cache_key"] = key
    r = R._SESSION.post(f"{R.EXTRACTION_BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {R.EXTRACTION_API_KEY}"},
                        json=body, timeout=90)
    r.raise_for_status()
    j = r.json()
    usage = j.get("usage", {})
    return (json.loads(j["choices"][0]["message"]["content"] or "{}"),
            usage.get("estimated_cost") or 0.0, 0, usage.get("prompt_tokens") or 0)


def job(task, model):
    arm, rec, res, pid = task
    claim_block = (f"CLAIM: {rec['claim_text']}\n"
                   f"(claimed on {rec.get('claim_date') or 'unknown date'}; judge the "
                   f"document's bearing on this exact proposition)")
    reads, cost_tot = [], 0.0
    for ids, sents in regions_for(res):
        out, cost = read_arm(arm, claim_block, ids, sents, f"{arm}:{rec['review_url']}", model)
        cost_tot += cost
        reads.append(out)
    agg = R.aggregate_reads([r for r in reads if r]) if any(reads) else None
    with _LOCK:
        _STATE["cost"] += cost_tot
        _STATE["done"] += 1
        if _STATE["done"] % 25 == 0:
            print(f"  {_STATE['done']} reads | ${_STATE['cost']:.3f}", flush=True)
    return {"pid": pid, "arm": arm, "v4_direction": res["read"]["direction"],
            "gold_veracity": rec["veracity"], "rating_subtype": rec.get("rating_subtype"),
            "domain": res["domain"], "n_regions": len(reads),
            "region_reads": reads, "agg": agg, "cost": cost_tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A0,A1,A2")
    ap.add_argument("--pids", help="file with one pid per line (default: all diagnosed)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None, help="override VERIFICATION_MODEL (capacity arm)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(PILOT), "read_ab.jsonl"))
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(PILOT)]
    ok, bad = check_reconstruction(recs)
    print(f"input reconstruction: {ok} exact, {bad} mismatched")
    if bad:
        print("  (mismatched results are skipped — paired design requires exact inputs)")

    want = None
    if a.pids and os.path.exists(a.pids):
        want = {l.strip() for l in open(a.pids) if l.strip()}

    import hashlib
    tasks = []
    for rec in recs:
        for res in rec["results"]:
            if not res.get("sents"):
                continue
            union = sorted({i for ids, _ in regions_for(res) for i in ids})
            if union != sorted(res["sent_ids"]):
                continue
            hh = hashlib.blake2b(f"{rec['review_url']}|{res['rank']}".encode(),
                                 digest_size=8).hexdigest()
            if want is not None and hh not in want:
                continue
            tasks.append((rec, res, hh))
    tasks.sort(key=lambda t: t[2])
    if a.limit:
        tasks = tasks[:a.limit]

    seen = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                j = json.loads(l)
                seen.add((j["pid"], j["arm"]))
            except Exception:
                pass
    arms = [x.strip() for x in a.arms.split(",")]
    jobs = [(arm, rec, res, pid) for arm in arms for rec, res, pid in tasks
            if (pid, arm) not in seen]
    print(f"{len(tasks)} reads x {len(arms)} arms = {len(jobs)} calls "
          f"({len(seen)} cached) | model={a.model or R.VERIFICATION_MODEL}")

    with open(a.out, "a") as f, ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, t, a.model) for t in jobs]
        for fu in as_completed(futs):
            try:
                f.write(json.dumps(fu.result()) + "\n")
                f.flush()
            except Exception as e:
                print(f"  job failed: {e}", flush=True)
    print(f"== {_STATE['done']} reads | ${_STATE['cost']:.3f} -> {a.out}")


if __name__ == "__main__":
    main()
