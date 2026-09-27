"""TH1c SYNTHESIS stage (logodds_sprint Workstream TH, Daniel 2026-08-25).

One LLM call per claim over its ~10 read dossiers (scores.jsonl schema) →
{verdict: true|false|unsure, confidence, reason, key_evidence: [ranks]}.
Score-blind by design: the s7 score/band never enters the prompt — synthesis is
an independent instrument over the same evidence, used to label banded claims
for the TH5 precision audit. Joint weighing (directions x source quality x
dates x independence), principles only, no dataset-derived examples.

Escalation (escalate_unsure.py) re-runs this over combined evidence; the
--extra flag lets it append new dossiers to a claim before synthesis.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, ".")
import eval.scripts.build_eval.evidence_urn_run as eur  # noqa: E402  (llm client)

SYNTH_V = "synth-v1"

SYNTH_SYS = """\
You are the final verdict step of a fact-checking pipeline. You get ONE claim,
its MODE, and the evidence dossiers the pipeline collected: for each retrieved
document, its source domain, a source-reliability rating where one exists, its
publication date, the reading step's direction flag, and the sentences that
reading cited. You saw none of the documents in full; judge from the dossiers.

Direction flags, per document: 5 = the document establishes the claim itself,
4 = points toward it, 3 = contested within the document, 2 = points against,
1 = contradicts or disproves it, X = on-claim context only, I = irrelevant.

MODE decides what the verdict is about:
- "attribution": the claim is that a named source said, wrote, or posted a
  specific statement. The verdict concerns the utterance — whether the source
  really said it — not whether the statement's content is true.
- "assertion": the claim states something about the world; the verdict
  concerns that substance.

Reply with JSON only:
{"verdict": "true" | "false" | "unsure", "confidence": <0.0-1.0>, "reason": "<= 2 sentences>", "key_evidence": [<ranks of the documents that decided it>]}

Weigh the dossiers jointly — never merely count flags:
- Source quality: a rated, reputable outlet outweighs an unrated site; several
  unrated sites do not add up to one reliable one.
- Independence: documents repeating the same underlying report or wire story
  are ONE piece of evidence, not several. Judge independence from the cited
  sentences and domains.
- Specificity beats volume: one document that addresses the decisive part of
  the claim outweighs many that circle it. Context-only (X) and irrelevant (I)
  documents establish nothing by themselves.
- Dates: a document predating the claimed events cannot confirm them; prefer
  evidence positioned to know.
- Direction conflicts: when credible documents genuinely point both ways,
  or support rests only on unrated sources, or nothing addresses the decisive
  part, the verdict is "unsure" — never force a call the dossiers do not carry.

verdict — "true": the evidence establishes the claim is accurate. "false": the
evidence establishes it is false, fabricated, or a distortion of what happened.
"unsure": insufficient, conflicting, or off-claim evidence.

confidence — your probability that the verdict is correct: 0.5 = a coin flip,
0.9+ = decisive evidence from reliable, independent sources.

reason — the decisive consideration, stated concretely (which sources, what
they establish). Not a restatement of the flag counts.
"""

MAX_SENTS_PER_DOC = 10


def dossier(doc: dict) -> str | None:
    rd = doc.get("read") or {}
    d = rd.get("direction")
    if not d or doc.get("read_status") not in ("ok", None):
        return None
    ng = f" (NewsGuard {doc['ng']})" if doc.get("ng") is not None else ""
    head = (f"[doc {doc['rank']}] {doc['domain']} | rating: {doc.get('rel') or 'UNRATED'}{ng}"
            f" | date: {doc.get('date') or 'undated'} | direction: {d}")
    if d == "I":
        return head
    ids = doc.get("sent_ids") or []
    sents = doc.get("sents") or []
    smap = dict(zip(ids, sents))
    cited = [i for i in (rd.get("evidence") or []) if i in smap][:MAX_SENTS_PER_DOC]
    body = "\n".join(f"  - {smap[i]}" for i in cited)
    extra = len(rd.get("evidence") or []) - len(cited)
    if extra > 0:
        body += f"\n  (+{extra} more cited sentences)"
    return head + ("\n" + body if body else "")


def synth_block(rec: dict, mode: str, extra_docs: list[dict] | None = None) -> str:
    claim = rec.get("claim_resolved") or rec["claim_text"]
    parts = [f"CLAIM: {claim}",
             f"MODE: {mode}",
             f"CLAIM DATE: {rec.get('claim_date_shown') or 'unknown'}",
             "", "DOSSIERS:"]
    docs = list(rec.get("results") or []) + list(extra_docs or [])
    rendered = [x for x in (dossier(d) for d in docs) if x]
    if not rendered:
        parts.append("(no readable documents were retrieved)")
    parts += rendered
    return "\n".join(parts)


def synthesize(rec: dict, mode: str, extra_docs: list[dict] | None = None):
    """One synthesis call. Returns (row, cost); row is None on failure."""
    cost_all = 0.0
    for att in range(3):
        try:
            obj, cost, _, _ = eur.llm(
                [{"role": "system", "content": SYNTH_SYS},
                 {"role": "user", "content": synth_block(rec, mode, extra_docs)}],
                max_tokens=300, timeout=90)
            cost_all += cost
            v = obj.get("verdict")
            if v in ("true", "false", "unsure"):
                ranks = {d["rank"] for d in (rec.get("results") or [])} | \
                        {d["rank"] for d in (extra_docs or [])}
                return ({"review_url": rec["review_url"], "claim_id": rec.get("claim_id"),
                         "verdict": v,
                         "confidence": float(obj.get("confidence") or 0.0),
                         "reason": (obj.get("reason") or "")[:400],
                         "key_evidence": sorted(r for r in (obj.get("key_evidence") or [])
                                                if r in ranks),
                         "prompt": SYNTH_V}, cost_all)
        except Exception:
            time.sleep(2 * (att + 1))
    return None, cost_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="scores.jsonl-schema file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", help="file of claim_ids to synthesize (default: all)")
    ap.add_argument("--modes", help="mode_labels.jsonl (labeller routing); "
                                    "default falls back to the record's claim_type")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--cap", type=float, default=2.0)
    a = ap.parse_args()

    modes = {}
    if a.modes:
        for line in open(a.modes):
            d = json.loads(line)
            if d.get("mode_llm"):
                modes[d["claim_id"]] = d["mode_llm"]
    want = None
    if a.ids:
        want = {line.strip() for line in open(a.ids) if line.strip()}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = {json.loads(l)["claim_id"] for l in open(out)} if out.exists() else set()

    jobs = []
    for line in open(a.input):
        rec = json.loads(line)
        cid = rec.get("claim_id") or rec["review_url"]
        rec.setdefault("claim_id", cid)
        if cid in seen or (want is not None and cid not in want):
            continue
        mode = modes.get(cid) or ("attribution" if rec.get("claim_type") == "attribution"
                                  else "assertion")
        jobs.append((rec, mode))
        if a.limit and len(jobs) >= a.limit:
            break
    print(f"{len(jobs)} claims to synthesize ({len(seen)} done) | "
          f"workers {a.workers} | cap ${a.cap}", flush=True)

    lock = threading.Lock()
    state = {"spent": 0.0, "done": 0, "fails": 0, "t0": time.time()}
    with open(out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(synthesize, rec, mode): (rec, mode) for rec, mode in jobs}
        for fu in as_completed(futs):
            row, cost = fu.result()
            rec, mode = futs[fu]
            with lock:
                state["spent"] += cost
                state["done"] += 1
                if row is None:
                    state["fails"] += 1
                else:
                    row["mode"] = mode
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                d, el = state["done"], time.time() - state["t0"]
                if d % 100 == 0 or d == len(jobs):
                    print(f"  {d}/{len(jobs)} | ${state['spent']:.3f} | "
                          f"{state['fails']} fails | {d/el*60:.0f}/min | "
                          f"ETA {(len(jobs)-d)/max(d/el, 1e-9)/60:.1f}m", flush=True)
                if state["spent"] > a.cap:
                    print("CAP HIT — aborting", flush=True)
                    sys.exit(1)
    print(f"DONE {state['done']} claims ${state['spent']:.3f} "
          f"{state['fails']} fails", flush=True)


if __name__ == "__main__":
    main()
