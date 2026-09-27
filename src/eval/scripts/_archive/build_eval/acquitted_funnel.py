"""Acquitted TRUE-stratum funnel (L0-L4) from a raw Community Notes dump.

Reproduces the ladder in docs/tweet_fit_corpus.md ("Sub-stratum (a)") from the
raw notes + noteStatusHistory zips, per-tweet:

  L0  >=1 misleading-classification note CRNH (currentStatus), zero CRH of any
      kind on the tweet (neither currentStatus nor lockedStatus CRH)
  L1  + zero pending misleading notes (currentStatus NEEDS_MORE_RATINGS)
  L2  + not all-rejected-media-only / satire-only ("only" = that checkbox is the
      note's sole misleading reason),
      all rejected notes locked CRNH, latest note >= 30 d before dump date
  L3  + >=1 NOT_MISLEADING defender note (any status)
  L4  + >=2 independent rejected accusations

  uv run python -m eval.scripts.build_eval.acquitted_funnel --dump 20260723
      [--out pool.parquet --rung L3]

Output schema matches eval/data/tweet_corpus/acquitted_l3_pool.parquet.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import io
import sys
import zipfile
from pathlib import Path

import polars as pl

RAW = Path("eval/data/community_notes/raw")
CRH = "CURRENTLY_RATED_HELPFUL"
CRNH = "CURRENTLY_RATED_NOT_HELPFUL"
NMR = "NEEDS_MORE_RATINGS"
MIS = "MISINFORMED_OR_POTENTIALLY_MISLEADING"


def load_notes(dump: str) -> pl.DataFrame:
    rows = []
    for zp in sorted(RAW.glob(f"{dump}-notes-*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    r = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
                    for row in r:
                        reasons = {k: row[k] == "1" for k in (
                            "misleadingOther", "misleadingFactualError",
                            "misleadingManipulatedMedia", "misleadingOutdatedInformation",
                            "misleadingMissingImportantContext",
                            "misleadingUnverifiedClaimAsFact", "misleadingSatire")}
                        n_reasons = sum(reasons.values())
                        rows.append((row["noteId"], row["tweetId"], row["classification"],
                                     reasons["misleadingManipulatedMedia"],
                                     reasons["misleadingManipulatedMedia"] and n_reasons == 1,
                                     reasons["misleadingSatire"] and n_reasons == 1,
                                     int(row["createdAtMillis"]) if row["createdAtMillis"] else 0))
        print(f"  {zp.name}: cumulative {len(rows):,} notes", flush=True)
    return pl.DataFrame(rows, schema=["noteId", "tweetId", "classification", "media",
                                      "media_only", "satire_only", "createdAtMillis"], orient="row")


def load_nsh(dump: str) -> pl.DataFrame:
    rows = []
    for zp in sorted(RAW.glob(f"{dump}-noteStatusHistory-*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    r = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
                    for row in r:
                        rows.append((row["noteId"], row["currentStatus"], row["lockedStatus"]))
        print(f"  {zp.name}: cumulative {len(rows):,} statuses", flush=True)
    return pl.DataFrame(rows, schema=["noteId", "currentStatus", "lockedStatus"], orient="row")


def funnel(dump: str, out: Path | None, rung: str) -> None:
    dump_date = datetime.date(int(dump[:4]), int(dump[4:6]), int(dump[6:8]))
    print(f"loading notes ({dump})...", flush=True)
    notes = load_notes(dump)
    print("loading noteStatusHistory...", flush=True)
    nsh = load_nsh(dump)
    df = notes.join(nsh, on="noteId", how="left").with_columns(
        pl.col("currentStatus").fill_null(""), pl.col("lockedStatus").fill_null(""),
        pl.from_epoch(pl.col("createdAtMillis") // 1000).dt.date().alias("note_date"),
    ).with_columns(
        (pl.col("classification") == MIS).alias("is_mis"),
        (pl.col("classification") == "NOT_MISLEADING").alias("is_notmis"),
    ).with_columns(
        ((pl.col("is_mis")) & (pl.col("currentStatus") == CRNH)).alias("mis_crnh"),
        ((pl.col("is_mis")) & (pl.col("currentStatus") == NMR)).alias("mis_pending"),
        ((pl.col("currentStatus") == CRH) | (pl.col("lockedStatus") == CRH)).alias("any_crh"),
    )

    per = df.group_by("tweetId").agg(
        pl.col("mis_crnh").sum().alias("n_mis_crnh"),
        pl.col("mis_pending").sum().alias("n_mis_pending"),
        pl.col("any_crh").sum().alias("n_crh"),
        pl.col("is_notmis").sum().alias("n_notmis_any"),
        pl.len().alias("n_notes"),
        pl.col("note_date").max().alias("latest_note"),
        # over rejected (mis_crnh) notes only:
        (pl.col("media_only").filter(pl.col("mis_crnh")).all()).alias("all_media_only"),
        (pl.col("satire_only").filter(pl.col("mis_crnh")).all()).alias("all_satire_only"),
        (pl.col("media").any()).alias("any_media_note"),
        ((pl.col("lockedStatus").filter(pl.col("mis_crnh")) == CRNH).all()).alias("all_locked"),
    )

    l0 = per.filter((pl.col("n_mis_crnh") >= 1) & (pl.col("n_crh") == 0))
    l1 = l0.filter(pl.col("n_mis_pending") == 0)
    cutoff = dump_date - datetime.timedelta(days=30)
    l2 = l1.filter(~pl.col("all_media_only") & ~pl.col("all_satire_only")
                   & pl.col("all_locked") & (pl.col("latest_note") <= cutoff))
    l3 = l2.filter(pl.col("n_notmis_any") >= 1)
    l4 = l3.filter(pl.col("n_mis_crnh") >= 2)
    for name, d in [("L0", l0), ("L1", l1), ("L2", l2), ("L3", l3), ("L4", l4)]:
        print(f"{name}: {d.height:,}")

    if out:
        sel = {"L2": l2, "L3": l3, "L4": l4}[rung]
        cols = ["tweetId", "all_media_only", "all_satire_only", "any_media_note",
                "all_locked", "latest_note", "n_mis_crnh", "n_notmis_any", "n_notes"]
        sel.select(cols).write_parquet(out)
        print(f"wrote {out} ({sel.height:,} rows, rung {rung})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rung", default="L3")
    a = ap.parse_args()
    funnel(a.dump, a.out, a.rung)
