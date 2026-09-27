"""Draw the 200-claim dev slice for the end-to-end DeepSeek eval.

Source: eval_v1 content-judged rows (judged_axis=='content'). Lightly stratified:
  - label: oversample the rare 'pass' class to ~35% (from the 12% population rate) for
    calibration power on the high-stakes confident-TRUE bucket; importance-weight back to
    population for any prevalence-sensitive headline.
  - publisher: within each label, allocate proportionally across publishers (largest-remainder)
    so no single publisher's style dominates.
satire rows are kept (is_satire is a metadata flag, sliced at analysis time, not excluded here).
Reproducible (seed=0). Writes data/dev200.parquet.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import polars as pl

SRC = Path("eval/data/eval_v1.parquet")
OUT = Path("eval/scripts/verdict_confidence/data/dev200.parquet")
TARGET = 200
PASS_FRAC = 0.35
SEED = 0


def alloc(groups: dict[str, list], n: int) -> list:
    """Largest-remainder proportional allocation across publisher groups, then sample."""
    rng = random.Random(SEED)
    total = sum(len(v) for v in groups.values())
    exact = {p: n * len(v) / total for p, v in groups.items()}
    base = {p: int(e) for p, e in exact.items()}
    rem = n - sum(base.values())
    for p, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:rem]:
        base[p] += 1
    picked = []
    for p, k in base.items():
        rows = groups[p][:]
        rng.shuffle(rows)
        picked.extend(rows[:min(k, len(rows))])
    return picked


def stratified(sub: pl.DataFrame, n: int) -> list:
    groups: dict[str, list] = defaultdict(list)
    for r in sub.to_dicts():
        groups[r["publisher_site"]].append(r)
    return alloc(groups, n)


def main() -> None:
    df = pl.read_parquet(SRC).filter(
        (pl.col("judged_axis") == "content") & (pl.col("binary_label").is_not_null())
    )
    n_pass = round(TARGET * PASS_FRAC)
    n_flag = TARGET - n_pass
    picked = (stratified(df.filter(pl.col("binary_label") == "pass"), n_pass)
              + stratified(df.filter(pl.col("binary_label") == "flag"), n_flag))
    out = pl.DataFrame(picked)
    out.write_parquet(OUT)

    print(f"wrote {out.height} -> {OUT}")
    print("label:", out["binary_label"].value_counts().sort("count", descending=True).to_dicts())
    print("publisher:", out["publisher_site"].value_counts().sort("count", descending=True).to_dicts())
    print("is_satire:", out.filter(pl.col("is_satire") == True).height)
    print("harmonised:", out["harmonised_label"].value_counts().sort("count", descending=True).to_dicts())


if __name__ == "__main__":
    main()
