"""Read-only audit: does STEP close supported/refuted claims on sufficient corroboration?

Bar: a claim may be resolved supported/refuted only on (i) ONE definitive PRIMARY source, or
(ii) >=2 INDEPENDENT RELIABLE SECONDARY sources. Snippets never suffice; NG<60 never a basis.
"""
from __future__ import annotations

import glob
import json
import re
from collections import Counter

from pipeline.search import newsguard_score_map

NG = newsguard_score_map()

RUNS = {
    "lowng_v3": "eval/data/survey_claims/runs/dev50_lowng_v3/results-*.jsonl",
    "main_v3": "eval/data/survey_claims/runs/dev50_main_v3/results-*.jsonl",
    "main_v3_rerun": "eval/data/survey_claims/runs/dev50_main_v3_rerun/results-*.jsonl",
}


def domain_of(url: str) -> str:
    m = re.findall(r"https?://([^/]+)", url or "")
    return re.sub(r"^www\.", "", m[0]).lower() if m else ""


def ng_score(dom: str):
    s = NG.get(dom)
    if s is None:
        parts = dom.split(".")
        if len(parts) > 2:
            s = NG.get(".".join(parts[-2:]))
    return s


def tier(dom: str, url: str = "") -> int:
    if dom.endswith((".gov", ".mil", ".edu")) or ".gov/" in (url or ""):
        return 0
    s = ng_score(dom)
    if s is None:
        return 3
    return 1 if s >= 75 else 2 if s >= 60 else 4


TIER_LABEL = {0: "PRIMARY", 1: "RELIABLE(NG>=75)", 2: "RELIABLE(NG60-75)",
              3: "UNRATED", 4: "UNRELIABLE(NG<60)"}


def classify(dom: str, url: str = ""):
    t = tier(dom, url)
    return t, TIER_LABEL[t], ng_score(dom)


# stance from READ that counts toward supporting/refuting a status
SUPPORT_STANCES = {"supports", "partially-supports"}
REFUTE_STANCES = {"refutes", "partially-refutes"}


def load_records(pattern):
    recs = []
    for fp in sorted(glob.glob(pattern)):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("ok") and r.get("result"):
                recs.append((fp, r["result"]))
    return recs


def analyze():
    all_closes = []
    for run, pat in RUNS.items():
        for fp, res in load_records(pat):
            ledger = res.get("ledger") or {}
            claims = res.get("claims") or []
            evidence = res.get("evidence") or []
            for cid_s, status in ledger.items():
                if status not in ("supported", "refuted"):
                    continue
                cid = int(cid_s)
                ctext = claims[cid - 1]["c"] if cid - 1 < len(claims) else "?"
                # evidence entries bearing on this claim
                ents = [e for e in evidence if e.get("claim_id") == cid]
                want = SUPPORT_STANCES if status == "supported" else REFUTE_STANCES
                # full-read entries whose stance points the same way as the close
                full_dir = [e for e in ents if not e.get("snippet_only")
                            and (e.get("stance") or "").lower() in want]
                # also count neutral/other full reads as present-but-not-directional
                full_all = [e for e in ents if not e.get("snippet_only")]
                snip = [e for e in ents if e.get("snippet_only")]
                all_closes.append({
                    "run": run, "file": fp.split("/")[-1], "post_id": res.get("id"),
                    "claim_id": cid, "claim": ctext, "status": status,
                    "full_dir": full_dir, "full_all": full_all, "snip": snip,
                    "verdict": res.get("verdict"),
                })
    return all_closes


def independent_reliable(entries):
    """Given directional full-read entries, return list of independent reliable-secondary
    voices (tier 1-2), collapsing republication-flagged and near-identical-domain-group ones.
    Returns (primary_list, reliable_secondary_domains, unrated_domains, unreliable_domains)."""
    primary, sec, unrated, unrel = [], [], [], []
    for e in entries:
        dom = e.get("domain", "")
        if e.get("republication"):
            continue  # not independent corroboration
        t, lbl, s = classify(dom, "")
        if t == 0:
            primary.append(dom)
        elif t in (1, 2):
            sec.append(dom)
        elif t == 3:
            unrated.append(dom)
        else:
            unrel.append(dom)
    return primary, sec, unrated, unrel


def bucket(c):
    """Return (bucket, detail) for a close."""
    fd = c["full_dir"]
    if not fd:
        # no directional full-read evidence at all
        if c["snip"]:
            return "FAILED", "closed with only snippet-level (or non-directional) evidence"
        if c["full_all"]:
            return "FAILED", "full reads present but none directional (neutral only)"
        return "FAILED", "no evidence entries at all for this claim"
    prim, sec, unrated, unrel = independent_reliable(fd)
    sec_u = sorted(set(sec))
    if prim:
        return "MET_PRIMARY", f"primary: {sorted(set(prim))}"
    if len(sec_u) >= 2:
        return "MET_2SEC", f"reliable secondary x{len(sec_u)}: {sec_u}"
    if len(sec_u) == 1:
        return "WEAK_SINGLE_SEC", f"single reliable secondary: {sec_u}"
    # no reliable secondary and no primary -> rested on unrated/unreliable
    if unrel:
        return "WEAK_UNRELIABLE", f"unreliable(NG<60): {sorted(set(unrel))}; unrated: {sorted(set(unrated))}"
    if unrated:
        return "WEAK_UNRATED", f"unrated only: {sorted(set(unrated))}"
    return "FAILED", "directional reads all republication-flagged (non-independent)"


def main():
    closes = analyze()
    print(f"TOTAL supported/refuted closes across runs: {len(closes)}\n")
    counts = Counter()
    rows = []
    for c in closes:
        b, detail = bucket(c)
        counts[b] += 1
        rows.append((c, b, detail))
    print("=== BUCKET COUNTS ===")
    for b, n in counts.most_common():
        print(f"  {b:20s} {n}")
    print()
    # per-run
    print("=== BY RUN ===")
    byrun = {}
    for c, b, d in rows:
        byrun.setdefault(c["run"], Counter())[b] += 1
    for run, ctr in byrun.items():
        print(f"  {run}: {dict(ctr)}  (total {sum(ctr.values())})")
    print()
    # WEAK/FAILED table
    print("=== WEAK / FAILED CLOSES ===")
    for c, b, d in rows:
        if b.startswith("MET"):
            continue
        srcs = []
        for e in c["full_dir"]:
            dom = e.get("domain", "")
            t, lbl, s = classify(dom)
            rep = " REPUB" if e.get("republication") else ""
            srcs.append(f"{dom}[{lbl}{'/NG'+str(s) if s is not None else ''}{rep}|{e.get('stance')}]")
        nsnip = len(c["snip"])
        print(f"\n[{c['run']}/{c['file']}] claim {c['claim_id']} = {c['status']} | BUCKET={b}")
        print(f"  claim: {c['claim'][:150]}")
        print(f"  reason: {d}")
        print(f"  directional full-read srcs: {srcs if srcs else '(none)'}")
        print(f"  snippet entries: {nsnip}")
        v = c["verdict"] or {}
        print(f"  justification: {(v.get('justification') or '')[:400]}")
    # dump justifications for a MET sample too (capability question)
    print("\n\n=== SAMPLE MET CLOSE JUSTIFICATIONS (capability probe) ===")
    shown = 0
    for c, b, d in rows:
        if not b.startswith("MET"):
            continue
        v = c["verdict"] or {}
        j = v.get("justification") or ""
        print(f"\n[{c['run']}/{c['file']}] claim {c['claim_id']} = {c['status']} | {b} | {d}")
        print(f"  justification: {j[:450]}")
        shown += 1
        if shown >= 12:
            break


if __name__ == "__main__":
    main()
