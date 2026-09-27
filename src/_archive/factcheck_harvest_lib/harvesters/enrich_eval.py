"""Apply the LLM enrichment + QA pass (eval.enrich) to the harmonised eval set.

  uv run python -m eval.scripts.harvesters.enrich_eval --sample      # validate on tricky ratings, print only
  uv run python -m eval.scripts.harvesters.enrich_eval --full        # enrich every row, write columns back

Adds columns: judged_axis, is_satire, harmonization_agrees, suggested_label, enrich_note.
`--sample` runs a small stratified set covering the axis-ambiguous ratings and prints a
comparison table WITHOUT writing — eyeball it before committing to a full pass.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

from eval.enrich import enrich_row
from eval.scripts._pool import pooled_checkpointed

PARQUET = Path("eval/data/eval_v1.parquet")

# Exact ratings worth eyeballing — axis-ambiguous or boundary cases (clog/230626).
SAMPLE_RATINGS = [
    "True", "False", "Fake", "Correct Attribution", "Incorrect Attribution",
    "Miscaptioned", "Originated as Satire", "Labeled Satire", "Mostly True",
    "Legit", "Mixture", "Half True", "Pants on Fire!", "Misleading",
    "AI-generated", "Altered Photo/Video",
]
PER_RATING = 2          # rows per sampled rating
N_FULLFACT = 4          # extra free-text rows from Full Fact


def _select_sample(df: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for rating in SAMPLE_RATINGS:
        parts.append(df.filter(pl.col("original_rating") == rating).head(PER_RATING))
    parts.append(df.filter(pl.col("publisher_site") == "fullfact.org").head(N_FULLFACT))
    return pl.concat(parts).unique(subset=["review_url"], keep="first")


def _enrich_df(df: pl.DataFrame, workers: int) -> list[dict | None]:
    rows = df.to_dicts()

    def call(row):
        try:
            return enrich_row(
                row.get("claim_text"), row.get("claimant"), row.get("original_rating"),
                row.get("publisher_site"), row.get("harmonised_label"),
            )
        except Exception as e:  # noqa: BLE001
            return {"_error": str(e)}

    if workers <= 1:
        return [call(r) for r in tqdm(rows, desc="enrich")]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(tqdm(ex.map(call, rows), total=len(rows), desc="enrich"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true", help="stratified validation set; print only")
    g.add_argument("--full", action="store_true", help="enrich all rows and write columns back")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--parquet", default=str(PARQUET), help="target parquet (default eval_v1)")
    args = ap.parse_args()

    parquet = Path(args.parquet)
    df = pl.read_parquet(parquet)
    if args.sample:
        sub = _select_sample(df)
        print(f"Enriching {len(sub)} rows (sample)")
        results = _enrich_df(sub, workers=1)
        errs = 0
        for row, res in zip(sub.to_dicts(), results):
            if res is None or "_error" in (res or {}):
                errs += 1
                print(f"\n[ERR] {row['publisher_site']} {row['original_rating']!r}: "
                      f"{(res or {}).get('_error')}")
                continue
            flag = "" if res["harmonization_agrees"] else f"  ⚠ →{res['suggested_label']!r}"
            print(
                f"\n[{row['publisher_site']:<17} | {str(row['original_rating'])[:34]:<34}] "
                f"harmonised={row['harmonised_label']}"
            )
            print(f"   claim: {str(row['claim_text'])[:120]}")
            print(f"   → axis={res['judged_axis']:<11} satire={str(res['is_satire']):<5} "
                  f"agree={res['harmonization_agrees']}{flag}")
            if res["note"]:
                print(f"     note: {res['note'][:140]}")
        print(f"\n{len(sub) - errs}/{len(sub)} ok, {errs} errors. (sample — nothing written)")
        return

    # --full, RESUMABLE: only (re)enrich rows still missing judged_axis, so a connection drop
    # mid-pass costs nothing — re-run picks up where it left off.
    EC = ["judged_axis", "is_satire", "harmonization_agrees", "suggested_label", "enrich_note"]
    cols = {c: (df[c].to_list() if c in df.columns else [None] * df.height) for c in EC}
    rows = df.to_dicts()
    todo = [i for i in range(df.height) if cols["judged_axis"][i] is None]
    print(f"Enriching {len(todo)} of {df.height} rows (resumable; {df.height - len(todo)} already done)")

    def _call(i):
        r = rows[i]
        try:
            return i, enrich_row(r.get("claim_text"), r.get("claimant"), r.get("original_rating"),
                                 r.get("publisher_site"), r.get("harmonised_label"))
        except Exception as e:  # noqa: BLE001
            return i, {"_error": str(e)}

    def enrich_df() -> pl.DataFrame:
        return df.with_columns(
            judged_axis=pl.Series(cols["judged_axis"], dtype=pl.Utf8),
            is_satire=pl.Series(cols["is_satire"], dtype=pl.Boolean),
            harmonization_agrees=pl.Series(cols["harmonization_agrees"], dtype=pl.Boolean),
            suggested_label=pl.Series(cols["suggested_label"], dtype=pl.Utf8),
            enrich_note=pl.Series(cols["enrich_note"], dtype=pl.Utf8),
        )

    abandoned = False
    if todo:
        def _apply(i, res):
            if res and "_error" not in res:
                cols["judged_axis"][i] = res["judged_axis"]
                cols["is_satire"][i] = res["is_satire"]
                cols["harmonization_agrees"][i] = res["harmonization_agrees"]
                cols["suggested_label"][i] = res["suggested_label"]
                cols["enrich_note"][i] = res["note"]

        abandoned = pooled_checkpointed(todo, _call, _apply,
                                        lambda: enrich_df().write_parquet(parquet),
                                        args.workers, "enrich")
    df = enrich_df()
    df.write_parquet(parquet)
    n_err = sum(1 for v in cols["judged_axis"] if v is None)
    print(f"Wrote {parquet}, {n_err} still null (re-run to fill)")
    print("\n=== judged_axis ===")
    print(df.group_by("judged_axis").agg(pl.len().alias("n")).sort("n", descending=True))
    dis = df.filter(pl.col("harmonization_agrees") == False)  # noqa: E712
    n_sat = dis.filter(pl.col("is_satire") == True).height  # noqa: E712
    print(f"=== harmonization disagreements: {dis.height} total "
          f"({n_sat} on satire rows — expected boundary, not bugs) ===")
    print("--- non-satire disagreements (real bugs to review) ---")
    print(dis.filter(pl.col("is_satire") != True)  # noqa: E712
            .group_by(["publisher_site", "original_rating", "harmonised_label", "suggested_label"])
            .agg(pl.len().alias("n")).sort("n", descending=True))

    if abandoned:
        # A hung LLM call leaves a non-daemon worker thread that would keep the process alive; all
        # enriched rows are already flushed, so force-exit. Re-run to retry the abandoned rows.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
