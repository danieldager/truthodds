"""Syndication collapse — post hoc, on the documents an urn run already stored.

Origin exclusion is domain-literal (`evidence_urn_run.run_claim`: publisher_site +
`exclude_extra`), and outlet_aliases only widens it to the outlet's OWN domains. It
cannot touch the wire problem: blocking apnews.com does not block the same AP item
republished verbatim by wbur, ktsm, npr and thehill, each of which is then counted as
an independent RELIABLE voice supporting the outlet's own claim.

Two rules, both conservative, both applied to the stored `results[]` of a run:

  (a) ORIGIN BY BYLINE. When the CLAIM's outlet is a wire service (AP / Reuters, incl.
      their outlet_aliases twins), a document whose OPENING carries that wire's credit
      is that outlet's own copy under someone else's domain -> `origin_copy=True`.
      The anchor is position: the credit must sit in the first ~3 sentences / ~400
      chars, where datelines and bylines live. "Reuters reported X" in paragraph nine
      is a citation, not a byline, and is left alone.

  (b) NEAR-DUPLICATE COLLAPSE. Among a claim's remaining documents, sentence-set
      overlap above a threshold means one document seen twice. Union-find, keep the
      highest-ranked member of each cluster, mark the rest `syndicated_of=<kept rank>`.
      Containment |A∩B|/min(|A|,|B|) is the clustering metric (Jaccard is reported
      alongside): a wire item republished with a local intro or truncated by the
      region selector is a subset, not a symmetric twin, and Jaccard punishes that.
      THRESHOLD 0.7, calibrated 2026-09-10 on true_outlet/scores.jsonl (11,098 docs):
      the per-doc max-containment distribution bottoms out at 0.6-0.8 (35 and 36 docs
      per bin) and rebounds to 50 and 110 in the top two bins, and the hand check of
      the sampled pairs matches that shape — above 0.7 the intersecting sentences are
      verbatim same-copy (nbcnews/yahoo 1.00, cbsaustin/fox23 1.00, calmatters/marinij
      0.93), while 0.3-0.7 is mostly distinct articles on one event sharing a couple of
      quoted paragraphs, which are the independent voices this must NOT suppress.

Nothing is deleted: both marks are added to the doc dicts, so a fit filters on them
and every rule stays auditable and re-runnable at $0. Dropped slots pad silent under
the pad-to-ten rule, so a collapse can only move a claim toward silence, never toward
support.

    uv run python -m eval.scripts.build_eval.syndication --calibrate eval/data/urn_runs/true_outlet/scores.jsonl
    uv run python -m eval.scripts.build_eval.syndication --apply <run.jsonl> --thr 0.7
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from eval.scripts.build_eval import outlet_aliases  # noqa: E402

PAIRS_OUT = Path("eval/data/urn_runs/wire_true/calibration_pairs.json")
HEAD_SENTS, HEAD_CHARS = 3, 400
MIN_SENT_CHARS = 25   # boilerplate ("Advertisement", "Sign in") must not create overlap

WIRES = {
    "ap": (("apnews.com",) + outlet_aliases.ALIASES["apnews.com"],
           (r"\(AP\)", r"\bAP Photo\b", r"\b(?:The )?Associated Press\b")),
    "reuters": (("reuters.com",) + outlet_aliases.ALIASES["reuters.com"],
                (r"\(Reuters\)", r"\bReuters\b")),
}
WIRE_DOMAINS = {d for doms, _ in WIRES.values() for d in doms}
_PATS = {w: re.compile("|".join(p), re.I) for w, (_, p) in WIRES.items()}
_PUNCT = re.compile(r"[^\w\s]+")
_WS = re.compile(r"\s+")


def wire_of(domain: str | None) -> str | None:
    d = (domain or "").lower().removeprefix("www.")
    for w, (doms, _) in WIRES.items():
        if d in doms:
            return w
    return None


def head_of(sents: list[str]) -> str:
    """The dateline/byline zone: first ~3 sentences, or ~400 chars if those are short."""
    h = " ".join(sents[:HEAD_SENTS])
    return h if len(h) >= HEAD_CHARS else " ".join(sents)[:HEAD_CHARS]


def norm_sents(sents: list[str]) -> set[str]:
    out = set()
    for s in sents or ():
        n = _WS.sub(" ", _PUNCT.sub(" ", (s or "").lower())).strip()
        if len(n) >= MIN_SENT_CHARS:
            out.add(n)
    return out


def overlap(a: set[str], b: set[str]) -> tuple[float, float]:
    """(containment, jaccard); (0, 0) when either side has no usable sentence."""
    if not a or not b:
        return 0.0, 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b)), inter / len(a | b)


def collapse(record: dict, thr: float) -> dict:
    """Mark (a) wire origin copies and (b) near-duplicates on a run record, in place."""
    docs = record.get("results") or []
    wire = wire_of(record.get("domain") or record.get("publisher_site"))
    sets = []
    for d in docs:
        d["origin_copy"] = bool(wire and _PATS[wire].search(head_of(d.get("sents") or [])))
        d["syndicated_of"] = None
        sets.append(norm_sents(d.get("sents")))
    live = [i for i, d in enumerate(docs) if not d["origin_copy"] and sets[i]]
    parent = {i: i for i in live}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for x, i in enumerate(live):
        for j in live[x + 1:]:
            if overlap(sets[i], sets[j])[0] >= thr:
                parent[find(i)] = find(j)
    clusters: dict[int, list[int]] = {}
    for i in live:
        clusters.setdefault(find(i), []).append(i)
    for members in clusters.values():
        keep = min(members, key=lambda i: docs[i].get("rank") or 99)
        for i in members:
            if i != keep:
                docs[i]["syndicated_of"] = docs[keep].get("rank")
    return record


# --------------------------------------------------------------------------- report

def _flags(docs) -> Counter:
    return Counter(d["read"]["direction"] for d in docs
                   if d.get("read_status") == "ok" and d.get("read"))


def _mix(c: Counter) -> str:
    t = sum(c.values()) or 1
    sup, ref = (c["5"] + c["4"]) / t, (c["1"] + c["2"]) / t
    return f"n={t:<6} support {sup:6.1%} refute {ref:6.1%} silent {1 - sup - ref:6.1%}"


def _bins(vals: list[float]) -> str:
    b = Counter(min(int(v * 10), 9) for v in vals)
    n = len(vals) or 1
    return "\n".join(f"    {i / 10:.1f}-{(i + 1) / 10:.1f}  {b[i]:>6}  {b[i] / n:6.1%}" for i in range(10))


def load(path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def calibrate(path: Path) -> None:
    recs = load(path)
    docs_n = sum(len(r.get("results") or []) for r in recs)
    print(f"{path}: {len(recs)} claims, {docs_n} docs", flush=True)
    max_c, max_j, band, high, bylines = [], [], [], [], Counter()
    rng = random.Random(707)
    for r in recs:
        docs = r.get("results") or []
        wire = wire_of(r.get("domain") or r.get("publisher_site"))
        sets = [norm_sents(d.get("sents")) for d in docs]
        for d in docs:
            if wire and _PATS[wire].search(head_of(d.get("sents") or [])):
                bylines[f"{wire} <- {d.get('domain')}"] += 1
        for i, d in enumerate(docs):
            if not sets[i]:
                continue
            best_c = best_j = 0.0
            for j, e in enumerate(docs):
                if i == j or not sets[j]:
                    continue
                c, jc = overlap(sets[i], sets[j])
                best_c, best_j = max(best_c, c), max(best_j, jc)
                if i < j and c >= 0.3:
                    pair = {"claim": r.get("claim_text"), "claim_domain": r.get("domain"),
                            "containment": round(c, 3), "jaccard": round(jc, 3),
                            "a": {"url": d.get("url"), "domain": d.get("domain"), "rank": d.get("rank"),
                                  "sents": (d.get("sents") or [])[:2]},
                            "b": {"url": e.get("url"), "domain": e.get("domain"), "rank": e.get("rank"),
                                  "sents": (e.get("sents") or [])[:2]}}
                    (high if c > 0.7 else band).append(pair)
            max_c.append(best_c)
            max_j.append(best_j)
    print(f"\nmax pairwise CONTAINMENT per doc (n={len(max_c)}, docs with usable sentences)")
    print(_bins(max_c))
    print(f"\nmax pairwise JACCARD per doc (n={len(max_j)})")
    print(_bins(max_j))
    print(f"\npairs: {len(band)} in [0.3,0.7], {len(high)} above 0.7 (containment)")
    print(f"wire bylines matched on {sum(bylines.values())} docs; top sources:")
    for k, v in bylines.most_common(15):
        print(f"    {k:<50} {v}")
    PAIRS_OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"band_0.3_0.7": rng.sample(band, min(20, len(band))),
               "above_0.7": rng.sample(high, min(10, len(high)))},
              open(PAIRS_OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {PAIRS_OUT}")


def apply(path: Path, thr: float) -> None:
    recs = load(path)
    for r in recs:
        collapse(r, thr)
    out = path.with_name(path.stem + "_collapsed" + path.suffix)
    with open(out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")

    def split(name, sub):
        docs = [d for r in sub for d in (r.get("results") or [])]
        if not docs:
            return
        oc = sum(1 for d in docs if d["origin_copy"])
        sy = sum(1 for d in docs if d["syndicated_of"] is not None)
        kept = [d for d in docs if not d["origin_copy"] and d["syndicated_of"] is None]
        print(f"  {name:<12} {len(sub):>5} claims | docs {len(docs):>6} -> {len(kept):>6} "
              f"({len(docs) / max(len(sub), 1):.2f} -> {len(kept) / max(len(sub), 1):.2f} per claim) | "
              f"origin_copy {oc / len(docs):6.1%} | syndicated {sy / len(docs):6.1%}")
        print(f"      before  {_mix(_flags(docs))}")
        print(f"      after   {_mix(_flags(kept))}")

    print(f"\n== collapse {path.name} at containment >= {thr} -> {out.name} ==")
    split("ALL", recs)
    for tier in sorted({r.get("ng_tier") for r in recs if r.get("ng_tier")}):
        split(tier, [r for r in recs if r.get("ng_tier") == tier])
    split("wire", [r for r in recs if (r.get("domain") or "") in WIRE_DOMAINS])
    split("non-wire", [r for r in recs if (r.get("domain") or "") not in WIRE_DOMAINS])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate")
    ap.add_argument("--apply")
    ap.add_argument("--thr", type=float, default=0.7)
    a = ap.parse_args()
    if a.calibrate:
        calibrate(Path(a.calibrate))
    if a.apply:
        apply(Path(a.apply), a.thr)


if __name__ == "__main__":
    main()
