"""Build the Stage-3 VERDICT eval dataset from the per-source harvests (WS0 of verdict_eval_plan.md).

  uv run python -m eval.scripts.build_verdict_dataset

Re-harmonises the RAW rating → gold_veracity (1–5) + rating_subtype (eval.harmonize; rule + cached LLM
fallback for free-text), tags judged_axis from the rating, carries the leakage fields (review_url/date),
drops un-rated rows, excludes satire from the default gold (kept as a flag), and assigns a hash-stable
stratified split with the false pole down-sampled for balanced dev/test (a `natural` slice is kept too).
→ eval/data/verdict_dataset.parquet. Reproducible: same harvest + cache ⇒ identical output.
"""
from __future__ import annotations

import glob
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from eval.harmonize import harmonise_veracity, judged_axis_from_rating, veracity_llm

OUT = Path("eval/data/verdict_dataset.parquet")
CACHE = Path("eval/data/veracity_llm_cache.parquet")
COLS = ["claim_text", "original_rating", "rating_value", "publisher_site", "publisher_name", "claimant",
        "review_url", "review_date", "claim_date", "language_code", "sources", "has_image", "image_count",
        "image_paths", "has_video", "raw_context", "raw_tier"]
TEST_FRAC, DEV_FRAC = 0.18, 0.30   # hash thresholds: test < .18, dev < .30, else train


def _u01(key) -> float:
    return (int(hashlib.md5((key or "").encode()).hexdigest(), 16) % 10_000) / 10_000


def main() -> None:
    rows = []
    for f in sorted(glob.glob("eval/data/*_harvest.parquet")):
        src = Path(f).stem.replace("_harvest", "")
        for r in pl.read_parquet(f).to_dicts():
            rows.append({c: r.get(c) for c in COLS} | {"source": src})
    print(f"loaded {len(rows)} core rows", flush=True)

    # ---- gold_veracity + subtype (rule, then cached LLM fallback for free-text misses) ----
    # The fallback is keyed on (claim, rating): the verdict prose alone is polarity-ambiguous (it states
    # the TRUE facts, often the claim's negation), so the claim is needed to rate the claim's veracity.
    cache: dict[tuple, tuple] = {}
    if CACHE.exists():
        cdf = pl.read_parquet(CACHE)
        if "claim" in cdf.columns:  # new (claim, rating) format; ignore the stale rating-only cache
            for r in cdf.to_dicts():
                cache[(r["claim"] or "", r["rating"] or "")] = (r["veracity"], r["subtype"])
    def _key(r):
        return ((r.get("claim_text") or ""), (r.get("original_rating") or ""))
    need = sorted({_key(r) for r in rows
                   if harmonise_veracity(r.get("original_rating"), r.get("rating_value")) == (None, "")
                   and _key(r) not in cache})
    if need:
        print(f"LLM fallback for {len(need)} uncached free-text (claim, rating) pairs ...", flush=True)
        def _fb(pair):
            claim, rat = pair
            try: return pair, veracity_llm(rat, claim)
            except Exception: return pair, (None, "unrated")
        with ThreadPoolExecutor(max_workers=8) as ex:
            for pair, vs in ex.map(_fb, need):
                cache[pair] = vs
        pl.DataFrame([{"claim": c, "rating": k, "veracity": v, "subtype": s}
                      for (c, k), (v, s) in cache.items()]).write_parquet(CACHE)
        print(f"  cache now {len(cache)} pairs → {CACHE}", flush=True)

    for r in rows:
        v, s = harmonise_veracity(r.get("original_rating"), r.get("rating_value"))
        if (v, s) == (None, ""):
            v, s = cache.get(_key(r), (None, "unrated"))
        r["gold_veracity"], r["rating_subtype"] = v, s
        r["is_satire"] = (s == "satire")
        r["judged_axis"] = judged_axis_from_rating(r.get("original_rating")) or "content"
        r["in_gold"] = (v is not None) and (s != "satire")  # satire excluded by default; unrated (v=None) excluded

    # ---- split (hash-stable) + false-pole down-sample for balanced dev/test ----
    gold = [r for r in rows if r["in_gold"]]
    for r in gold:
        r["u"] = _u01(r.get("review_url") or r.get("claim_text"))
        r["split"] = "test" if r["u"] < TEST_FRAC else ("dev" if r["u"] < DEV_FRAC else "train")
        r["balanced"] = True
    for sp in ("dev", "test"):
        sub = [r for r in gold if r["split"] == sp]
        from collections import Counter
        vc = Counter(r["gold_veracity"] for r in sub)
        cap = max((vc[v] for v in (2, 3, 4, 5) if v in vc), default=0)  # largest non-false level
        ones = sorted((r for r in sub if r["gold_veracity"] == 1), key=lambda r: r["u"])
        for r in ones[cap:]:
            r["balanced"] = False  # down-sampled out of the balanced eval (still in `natural`)

    for r in gold:  # normalise the one mixed-type column (LeadStories rating_value is a str)
        r["rating_value"] = None if r["rating_value"] is None else str(r["rating_value"])
    df = pl.DataFrame(gold, infer_schema_length=None)
    df.write_parquet(OUT)

    # ---- report ----
    from collections import Counter
    print(f"\nin_gold {len(gold)} / {len(rows)} core "
          f"(excluded: satire {sum(r['is_satire'] for r in rows)}, "
          f"unrated {sum(r['rating_subtype'] == 'unrated' for r in rows)})")
    print("veracity:", dict(sorted(Counter(r['gold_veracity'] for r in gold).items())))
    print("subtype :", Counter(r['rating_subtype'] for r in gold).most_common())
    print("axis    :", Counter(r['judged_axis'] for r in gold).most_common())
    print("by source:", Counter(r['source'] for r in gold).most_common())
    print("split   :", Counter(r['split'] for r in gold).most_common())
    for sp in ("dev", "test"):
        nat = Counter(r['gold_veracity'] for r in gold if r['split'] == sp)
        bal = Counter(r['gold_veracity'] for r in gold if r['split'] == sp and r['balanced'])
        print(f"  {sp}: natural {dict(sorted(nat.items()))} (n={sum(nat.values())})  "
              f"balanced {dict(sorted(bal.items()))} (n={sum(bal.values())})")
    print("language:", Counter(r['language_code'] for r in gold).most_common(6))
    print("has_image:", Counter(bool(r['has_image']) for r in gold).most_common())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
