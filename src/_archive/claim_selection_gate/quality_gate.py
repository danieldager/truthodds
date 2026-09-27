"""Claim quality gate — screen normalized claims for verification-eval suitability.

  uv run python -m eval.scripts.quality_gate --parquet eval/data/snopes_harvest.parquet

Adds (additive, resumable, crash-safe) for each CONTENT-axis row:
  - is_checkable    : a single self-contained verifiable proposition (not a fragment/question)
  - is_attribution  : really a 'did X say Y' claim (catches attribution mislabeled as content)
  - gate_reason     : short justification
The verification eval samples content & is_checkable & not is_attribution (w/ Daniel, clog 250626).
Pipeline step between enrich_eval and combine_dataset; does NOT touch judged_axis/is_satire, so it
won't churn labels from the V3.1→V4 model switch. Only content rows are gated (the eval pool).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

from eval.enrich import quality_gate_row
from eval.scripts._pool import pooled_checkpointed

GC = ["is_checkable", "is_attribution", "gate_reason"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    p = Path(args.parquet)
    df = pl.read_parquet(p)
    n = df.height
    cols = {c: (df[c].to_list() if c in df.columns else [None] * n) for c in GC}
    rows = df.to_dicts()
    has_axis = "judged_axis" in df.columns
    todo = [i for i in range(n)
            if cols["is_checkable"][i] is None and (not has_axis or rows[i].get("judged_axis") == "content")]
    print(f"{len(todo)} content rows to gate (of {n})")

    def build() -> pl.DataFrame:
        return df.with_columns(
            is_checkable=pl.Series(cols["is_checkable"], dtype=pl.Boolean),
            is_attribution=pl.Series(cols["is_attribution"], dtype=pl.Boolean),
            gate_reason=pl.Series(cols["gate_reason"], dtype=pl.Utf8),
        )

    abandoned = False
    if todo:
        def work(i):
            try:
                return i, quality_gate_row(rows[i].get("claim_text"))
            except Exception as e:  # noqa: BLE001
                return i, {"_error": str(e)}

        def apply(i, res):
            if res and "_error" not in res:
                cols["is_checkable"][i] = res["is_checkable"]
                cols["is_attribution"][i] = res["is_attribution"]
                cols["gate_reason"][i] = res["gate_reason"]

        abandoned = pooled_checkpointed(todo, work, apply, lambda: build().write_parquet(p),
                                        args.workers, "gate")
    df = build()
    df.write_parquet(p)

    if has_axis:
        c = df.filter(pl.col("judged_axis") == "content")
        unck = c.filter(~pl.col("is_checkable").fill_null(True)).height
        attr = c.filter(pl.col("is_attribution").fill_null(False)).height
        print(f"\nWrote {p}: content={c.height}  not-checkable={unck}  attribution-leak={attr}  "
              f"(both dropped from the eval)")
    if abandoned:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
