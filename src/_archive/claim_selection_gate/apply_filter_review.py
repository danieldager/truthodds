"""Apply borderline-review decisions (an `Fxxx:label` block) → the provenance ledger.

  uv run python -m eval.scripts.apply_filter_review <decisions.txt>

Parses `F001:positive F002:negative …` → joins `filter_review_items.parquet` (id→key) →
appends to `eval/data/filter_provenance.parquet` (key, recommend, source='human', reason). Human
decisions accumulate (last wins); `build_filter_dev` applies them as overrides → reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ITEMS = Path("eval/data/filter_review_items.parquet")
LEDGER = Path("eval/data/filter_provenance.parquet")


def main() -> None:
    dec = dict(t.split(":") for t in Path(sys.argv[1]).read_text().split())
    items = {r["id"]: r["key"] for r in pl.read_parquet(ITEMS).to_dicts()}
    rows = [{"key": items[fid], "recommend": lab, "source": "human", "reason": "borderline_review"}
            for fid, lab in dec.items() if fid in items and lab in ("positive", "negative", "drop")]
    df = pl.DataFrame(rows)
    if LEDGER.exists():
        df = pl.concat([pl.read_parquet(LEDGER), df], how="diagonal_relaxed").unique("key", keep="last")
    df.write_parquet(LEDGER)
    print(f"wrote {LEDGER}: {df.height} human decisions ({len(rows)} from this block)")


if __name__ == "__main__":
    main()
