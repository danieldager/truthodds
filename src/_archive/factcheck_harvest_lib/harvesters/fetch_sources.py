"""Fill raw_claim from claim_source_url for a harvest parquet — the bonus extraction-pair pass.

  uv run python -m eval.scripts.harvesters.fetch_sources --parquet eval/data/leadstories_harvest.parquet

Best-effort, idempotent, and CRASH-SAFE. Results are checkpointed to the parquet every
CHECKPOINT_EVERY completed rows and applied out-of-order, so a kill — or a single hung straggler
request — never loses finished work: a re-run skips rows already attempted (source_method set) and
finishes the rest. A row whose request hangs (e.g. a slow-DNS host that ignores the socket timeout)
is abandoned after IDLE_TIMEOUT of no progress instead of stalling the whole pass. One hung tweet
once cost us 97% of a completed run because the old code only wrote at the very end (clog 250626).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

from eval.scripts._pool import pooled_checkpointed
from eval.source_fetch import _related, assemble_raw, fetch_raw_claim, post_states_claim


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    p = Path(args.parquet)
    df = pl.read_parquet(p)
    rows = df.to_dicts()
    n = df.height
    raw = df["raw_claim"].to_list() if "raw_claim" in df.columns else [None] * n
    meth = df["source_method"].to_list() if "source_method" in df.columns else [None] * n
    # claim_date = the original-claim date. Keep any existing (e.g. PolitiFact's "stated on" date);
    # a resolved post's created_at is the precise original-post date and refines it.
    cdate = df["claim_date"].to_list() if "claim_date" in df.columns else [None] * n
    related = df["raw_related"].to_list() if "raw_related" in df.columns else [None] * n
    abandoned = False

    # --- fetch pass: resolve claim_source_url -> raw_claim. Resumable: skip rows already attempted
    #     (source_method set), so checkpointed work is never redone. ---
    todo = [i for i, r in enumerate(rows) if r.get("claim_source_url") and not meth[i]]
    print(f"{len(todo)} rows to fetch (of {n})")

    def fetch_df() -> pl.DataFrame:
        return df.with_columns(
            raw_claim=pl.Series(raw, dtype=pl.Utf8),
            source_method=pl.Series(meth, dtype=pl.Utf8),
            claim_date=pl.Series(cdate, dtype=pl.Utf8),
        )

    if todo:
        def work(i):
            return i, fetch_raw_claim(rows[i]["claim_source_url"])

        def apply(i, res):
            raw[i], meth[i] = res["raw_claim"], res["source_method"]
            if res.get("claim_date"):
                cdate[i] = res["claim_date"]

        abandoned |= pooled_checkpointed(todo, work, apply, lambda: fetch_df().write_parquet(p),
                                         args.workers, "sources")
    df = fetch_df()

    # --- relevance gate: confirm each resolved post actually MAKES the claim (LLM; token-overlap
    #     backstop on error) — fact-check pages embed the claim's post AND its evidence/amplifiers.
    #     Resumable: only gates rows whose raw_related isn't already set. ---
    ct = df["claim_text"].to_list()
    gate_todo = [i for i in range(n) if raw[i] and related[i] is None]

    def gate_df() -> pl.DataFrame:
        return df.with_columns(raw_related=pl.Series(related, dtype=pl.Boolean))

    if gate_todo:
        def gwork(i):
            v = post_states_claim(raw[i], ct[i] or "")
            return i, (v if v is not None else _related(raw[i], ct[i] or ""))

        def gapply(i, v):
            related[i] = v

        abandoned |= pooled_checkpointed(gate_todo, gwork, gapply, lambda: gate_df().write_parquet(p),
                                         args.workers, "gate")
    df = gate_df()

    # --- raw-side assembly: pair a genuine raw with the normalized claim — the resolved original
    #     post (raw_claim, if it passed the gate) or a fuller verbatim quote the fact-check
    #     reproduces (body_quote). Always runs over every row, so raw_context/raw_tier are
    #     consistent on any completed run. ---
    rc, rt = zip(*(assemble_raw(r) for r in df.to_dicts())) if n else ([], [])
    df = df.with_columns(
        raw_context=pl.Series(rc, dtype=pl.Utf8),
        raw_tier=pl.Series(rt, dtype=pl.Utf8),
    )
    df.write_parquet(p)

    got = sum(1 for x in raw if x)
    ctx = df.filter(pl.col("raw_context").is_not_null()).height
    cd = df.filter(pl.col("claim_date").is_not_null()).height
    print(f"\nWrote {p}")
    print(f"raw_claim filled: {got}/{n} ({got/n*100:.0f}%)  |  "
          f"raw_context (any tier): {ctx}/{n} ({ctx/n*100:.0f}%)  |  "
          f"claim_date: {cd}/{n} ({cd/n*100:.0f}%)")
    if "source_method" in df.columns:
        print("by method:", df.group_by("source_method").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())
    print("by raw_tier:", df.group_by("raw_tier").agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())

    if abandoned:
        # A hung worker thread is non-daemon and would keep the process alive forever; all data is
        # already flushed, so force-exit cleanly. Re-run to retry the abandoned rows.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
