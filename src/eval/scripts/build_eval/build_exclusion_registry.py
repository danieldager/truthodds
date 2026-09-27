"""One exclusion registry for every claim any screen ever ruled on.

dataset_improvements.md block F, first item: rulings live in `fit_exclusions.json`
(prefix-matched free text), `e1_gate_decisions.json`, two media-provenance purge
JSONs, a mixed-framing screen marked tag-only, two claim-screen parquets and a
scatter of clog entries. This flattens all of them into one schema so "is this
claim excluded, by whom, on what date, and does anything actually apply it" is a
single lookup.

    uv run python -m eval.scripts.build_eval.build_exclusion_registry

Output: eval/data/exclusions/registry.parquet. One row per (claim_id, rule).
`applied` is whether the CURRENT default loaders drop the claim, not whether the
ruling was sound — the tag-only screens ride along with applied=false so nothing
is invisible.

`fit_exclusions.json` carries no claim ids (it was hand-written against claim
text), so ids are recovered by the same claim[:80] prefix match `fit_two_urn`
uses; prefixes that resolve to nothing are recorded with a null claim_id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

E1 = Path("eval/data/urn_runs/e1_ctx")
C2 = Path("eval/data/urn_runs/c2_false")
TL = Path("eval/data/urn_runs/true_timeline")
OUT = Path("eval/data/exclusions/registry.parquet")

COLS = ["claim_id", "corpus", "rule", "source_file", "decided_on", "decided_by",
        "applied", "note"]


def row(claim_id, corpus, rule, src, decided_on, decided_by, applied, note):
    return {"claim_id": claim_id, "corpus": corpus, "rule": rule,
            "source_file": str(src), "decided_on": decided_on,
            "decided_by": decided_by, "applied": applied, "note": note or ""}


def gate_decisions() -> list[dict]:
    """93 hand rulings on media-locus and demonstrative claims (Daniel, 2026-08-04)."""
    src = Path("eval/data/e1_gate_decisions.json")
    d = json.loads(src.read_text())
    return [row(e["review_url"], "fc_gold", f"gate_{e['decision']}_{e['kind'].lower()}",
                src, "2026-08-04", "human", e["decision"] == "exclude",
                e.get("reason")) for e in d["decisions"]]


def purge(src: Path, corpus: str, decided_on: str) -> list[dict]:
    """Three-judge media-provenance purge. Non-excluded verdicts ride along."""
    out = []
    for e in json.loads(src.read_text()):
        keep = e.get("excluded", True)
        out.append(row(e.get("claim_id"), corpus,
                       "media_provenance" if keep else "media_provenance_cleared",
                       src, decided_on, "llm-majority", keep,
                       f"rule={e.get('rule')} n_votes={e.get('n_votes')}"))
    return out


def cn_prefix_exclusions() -> list[dict]:
    """fit_exclusions.json has claim_id=null throughout: resolve by claim[:80]."""
    src = C2 / "fit_exclusions.json"
    by_prefix: dict[str, list[str]] = {}
    for p in (C2 / "scores.jsonl", C2 / "scores_ext.jsonl"):
        for line in p.open():
            r = json.loads(line)
            claim = (r.get("claim_resolved") or r.get("claim_text") or "")[:80]
            by_prefix.setdefault(claim, []).append(r["review_url"])
    out, unresolved = [], 0
    for e in json.loads(src.read_text()):
        ids = by_prefix.get(e["claim"][:80])
        if not ids:
            unresolved += 1
            out.append(row(None, "cn_false", e["reason"], src, "2026-08-26",
                           "human", True, f"UNRESOLVED prefix: {e['claim'][:80]}"))
            continue
        out += [row(i, "cn_false", e["reason"], src, "2026-08-26", "human", True,
                    e["claim"][:80]) for i in ids]
    if unresolved:
        print(f"  fit_exclusions.json: {unresolved} of 34 prefixes resolve to no claim")
    return out


def mixed_framing() -> list[dict]:
    """241 notes that tick missing-context only. Screened, never applied."""
    src = C2 / "mixed_framing_exclusions.json"
    return [row(e["claim_id"], "cn_false", "mixed_framing_missing_context_only", src,
                "2026-08-28", "code", False, e.get("basis"))
            for e in json.loads(src.read_text())]


def claim_screen_e1() -> list[dict]:
    """Verdict-blind LLM claim screen over the E1 population. Tag only."""
    src = Path("eval/data/claim_screen_e1.parquet")
    d = pl.read_parquet(src).filter(pl.col("verdict") != "ok")
    return [row(r["review_url"], "fc_gold", f"claim_screen_{r['verdict']}", src,
                "2026-08-04", "llm", False, r.get("why"))
            for r in d.iter_rows(named=True)]


def tl_screen() -> list[dict]:
    """Timeline residue screen. Applied since Daniel's 2026-09-08 decision."""
    src = TL / "tl_screen_nocontext.parquet"
    d = pl.read_parquet(src).filter(pl.col("verdict") != "ok")
    return [row(r["claim_id"], "timeline", f"residue_screen_{r['verdict']}", src,
                "2026-08-27", "llm", True, r.get("why"))
            for r in d.iter_rows(named=True)]


def main():
    rows = (gate_decisions()
            + purge(E1 / "media_provenance_exclusions.json", "fc_gold", "2026-08-28")
            + cn_prefix_exclusions()
            + purge(C2 / "media_provenance_exclusions.json", "cn_false", "2026-08-28")
            + mixed_framing()
            + claim_screen_e1()
            + tl_screen())
    df = pl.DataFrame(rows).select(COLS).sort(["corpus", "rule", "claim_id"],
                                              nulls_last=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)
    print(f"{df.height} rows -> {OUT}")
    with pl.Config(tbl_rows=-1, fmt_str_lengths=40):
        print(df.group_by(["corpus", "rule", "applied"]).len()
                .sort(["corpus", "rule"]))


if __name__ == "__main__":
    main()
