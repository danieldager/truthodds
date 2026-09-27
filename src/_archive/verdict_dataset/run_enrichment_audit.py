"""Run the 2-pass enrichment + audit over the core fact-checks (docs/enrichment_audit_spec.md).

  uv run python -m eval.scripts.run_enrichment_audit --smoke           # ~30 rows, print, no write
  uv run python -m eval.scripts.run_enrichment_audit --full            # all rows → audit_results.parquet (resumable)

Per row: Pass A (BLIND, has-input rows) + Pass B (SIGHTED, all rated rows), EACH with the labeller
(Qwen3-VL-235B) AND the cross-check (Gemma-3-27B). Field-level 235↔Gemma disagreement is the borderline
signal (derived later in the builders, not asked of the model). Resumable / checkpointed / high-visibility
(eval.scripts._pool). gold_veracity+rating_subtype come from eval.harmonize (rule + cached LLM fallback).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

import polars as pl

from eval.audit import AUDIT_MODEL, CROSS_MODEL, PASS_A_SYSTEM, PASS_B_SYSTEM, pass_a, pass_b
from eval.harmonize import harmonise_veracity
from eval.scripts._pool import pooled_checkpointed

OUT = Path("eval/data/audit_results.parquet")
# Stamp the audit with a hash of the prompts + models. Resuming (--full on top of --dev) is only valid if
# the enrichment step is UNCHANGED; if the prompts/models change, a resume would mix stale rows → we refuse
# unless --reset (Daniel 280626).
AUDIT_VERSION = hashlib.md5((PASS_A_SYSTEM + PASS_B_SYSTEM + AUDIT_MODEL + CROSS_MODEL).encode()).hexdigest()[:10]


def _u01(key: str) -> float:
    return (int(hashlib.md5((key or "").encode()).hexdigest(), 16) % 10_000) / 10_000
CACHE = Path("eval/data/veracity_llm_cache.parquet")
CORE = ["review_url", "claim_text", "original_rating", "rating_value", "publisher_site", "claimant",
        "raw_context", "has_image", "image_paths"]
A_FIELDS = ["topic", "political_implication", "has_claim", "claim_locus", "obvious_joke", "readable", "note"]
B_FIELDS = ["judged_axis", "is_satire", "veracity_agrees", "suggested_veracity", "rating_subtype_ok",
            "claim_matches_post", "image_supports_claim", "note"]


def _key(r: dict) -> str:
    return r.get("review_url") or "h:" + hashlib.md5((r.get("claim_text") or "").encode()).hexdigest()


def _has_input(r: dict) -> bool:
    if (r.get("raw_context") or "").strip():
        return True
    return any(p and Path(p).exists() for p in (r.get("image_paths") or []))


def load_core() -> list[dict]:
    cache = {}
    if CACHE.exists():
        for c in pl.read_parquet(CACHE).to_dicts():
            cache[c["rating"]] = (c["veracity"], c["subtype"])
    rows = []
    for f in sorted(glob.glob("eval/data/*_harvest.parquet")):
        src = Path(f).stem.replace("_harvest", "")
        for r in pl.read_parquet(f).to_dicts():
            row = {c: r.get(c) for c in CORE} | {"source": src}
            v, s = harmonise_veracity(r.get("original_rating"), r.get("rating_value"))
            if (v, s) == (None, ""):
                v, s = cache.get(r.get("original_rating") or "", (None, "unrated"))
            row["gold_veracity"], row["rating_subtype"] = v, s
            row["has_input"] = _has_input(row)
            rows.append(row)
    return rows


def _flat(prefix: str, d: dict | None, fields: list[str]) -> dict:
    d = d or {}
    return {f"{prefix}_{f}": d.get(f) for f in fields}


def audit_row(r: dict, models: tuple[str, str]) -> dict:
    lab, cross = models
    out = {"key": _key(r), "source": r["source"], "claim_text": (r.get("claim_text") or "")[:160],
           "original_rating": r.get("original_rating"), "gold_veracity": r.get("gold_veracity"),
           "rating_subtype": r.get("rating_subtype"), "has_input": r["has_input"],
           "_audit_version": AUDIT_VERSION}
    if r["has_input"]:
        out |= _flat("A_lab", pass_a(r, lab), A_FIELDS) | _flat("A_x", pass_a(r, cross), A_FIELDS)
    if r.get("gold_veracity") is not None:  # rated rows (incl. satire) get the sighted verdict audit
        gv, sub = r["gold_veracity"], r["rating_subtype"]
        out |= _flat("B_lab", pass_b(r, lab, gv, sub), B_FIELDS) | _flat("B_x", pass_b(r, cross, gv, sub), B_FIELDS)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="~30 stratified rows, print + agreement, no write")
    g.add_argument("--frac", type=float, help="audit a hash-stable subset (u01<FRAC) → write, resumable (dev-first)")
    g.add_argument("--full", action="store_true", help="all rows (== --frac 1.0), resumable on top of any --frac run")
    g.add_argument("--split", choices=["dev", "test", "train"], help="audit exactly the in_gold rows of this "
                   "verdict_dataset split (targeted top-up; the --frac hash band coincided with test, so dev was 0%)")
    ap.add_argument("--reset", action="store_true", help="wipe audit_results.parquet first (use if the audit step changed)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    frac = 1.0 if args.full else args.frac
    models = (AUDIT_MODEL, CROSS_MODEL)
    rows = load_core()
    elig = [r for r in rows if r["has_input"] or r.get("gold_veracity") is not None]
    print(f"core {len(rows)} | eligible {len(elig)} (has-input {sum(r['has_input'] for r in rows)}, "
          f"rated {sum(r.get('gold_veracity') is not None for r in rows)}) | audit_version {AUDIT_VERSION}", flush=True)

    if args.smoke:
        # stratified ~30: image / text / empty-input, deterministic by md5(key)
        buckets = {"image": [], "text": [], "empty": []}
        for r in sorted(elig, key=lambda r: hashlib.md5(_key(r).encode()).hexdigest()):
            b = ("image" if (r.get("has_image") and any(p and Path(p).exists() for p in (r.get("image_paths") or [])))
                 else "text" if (r.get("raw_context") or "").strip() else "empty")
            if len(buckets[b]) < 10:
                buckets[b].append(r)
        sample = [r for bs in buckets.values() for r in bs]
        print(f"smoke {len(sample)} rows", flush=True)
        out = [audit_row(r, models) for r in sample]
        errs = sum(1 for o in out for k, v in o.items() if isinstance(v, dict) and "_err" in v)
        for fam, flds in (("A", A_FIELDS), ("B", B_FIELDS)):
            print(f"\n{fam} cross-model agreement:")
            for f in flds:
                if f == "note":
                    continue
                bo = sa = 0
                for o in out:
                    x, y = o.get(f"{fam}_lab_{f}"), o.get(f"{fam}_x_{f}")
                    if x is None or y is None:
                        continue
                    bo += 1; sa += int(x == y)
                print(f"   {f:20} {sa}/{bo}  {100*sa/bo if bo else 0:.0f}%")
        print(f"\nerrored sub-calls: {errs}")
        return

    # --frac / --full: write to audit_results.parquet, resumable + version-guarded
    if args.reset and OUT.exists():
        OUT.unlink(); print("--reset: wiped audit_results.parquet", flush=True)
    done = set()
    if OUT.exists():
        prev = pl.read_parquet(OUT)
        pv = prev["_audit_version"][0] if "_audit_version" in prev.columns and prev.height else None
        if pv and pv != AUDIT_VERSION:
            print(f"REFUSING: existing audit_results.parquet was built with audit_version {pv} != current "
                  f"{AUDIT_VERSION} — the enrichment prompts/models changed, so a resume would MIX stale rows. "
                  f"Re-run with --reset to rebuild from scratch.", flush=True)
            sys.exit(1)
        done = set(prev["key"].to_list())
    if args.split:  # targeted: exactly the in_gold review_urls of this split (canonical membership from verdict_dataset)
        vd = pl.read_parquet("eval/data/verdict_dataset.parquet")
        keys = set(vd.filter(pl.col("split") == args.split)["review_url"].to_list())
        sel = [r for r in elig if _key(r) in keys]
        seldesc = f"--split {args.split}"
    else:
        sel = [r for r in elig if _u01(_key(r)) < frac]
        seldesc = f"frac {frac}"
    todo = [i for i, r in enumerate(sel) if _key(r) not in done]
    print(f"{seldesc}: selected {len(sel)}/{len(elig)} eligible "
          f"(has-input {sum(r['has_input'] for r in sel)}) | todo {len(todo)} ({len(done)} already done)", flush=True)
    res: dict[str, dict] = {}

    def flush():
        if not res:
            return
        new = pl.DataFrame(list(res.values()), infer_schema_length=None)
        comb = new if not OUT.exists() else pl.concat([pl.read_parquet(OUT), new], how="diagonal_relaxed").unique("key", keep="last")
        comb.write_parquet(OUT)

    def work(i):
        return _key(sel[i]), audit_row(sel[i], models)

    ab = pooled_checkpointed(todo, work, lambda k, p: res.__setitem__(k, p), flush, args.workers,
                             "audit", checkpoint_every=100)
    flush()
    df = pl.read_parquet(OUT)
    pcol = "A_lab_political_implication"
    aud_a = df.filter(pl.col(pcol).is_not_null()) if pcol in df.columns else df.head(0)
    print(f"\nwrote {OUT}: {df.height} rows total (Pass-A/has-input audited {aud_a.height})")
    if aud_a.height:
        print("  Pass A political_implication (235):", dict(Counter(aud_a[pcol].to_list())))
        print("  Pass A has_claim (235):", dict(Counter(aud_a["A_lab_has_claim"].to_list())))
        print("  Pass A obvious_joke (235):", dict(Counter(aud_a["A_lab_obvious_joke"].to_list())))
    if "B_lab_is_satire" in df.columns:
        print("  satire flagged (B 235):", df.filter(pl.col("B_lab_is_satire") == True).height)  # noqa: E712
    if ab:
        sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
