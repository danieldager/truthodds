"""Community Notes harvest → cn_gold.parquet (SEPARATE dataset from fc-gold).

One day's public snapshot is the FULL history (cumulative since 2021; ~7-day
retention, so always pull everything from the same day's directory). We download
notes (3 shards, ~471MB) + noteStatusHistory (1 shard, ~174MB) and SKIP the ratings
family (11.3GB): ratings are the scorer's raw INPUTS; noteStatusHistory carries its
OUTPUTS, which is what we consume. Fetch pattern adapted from
llm_bench/aggregator/fetch_community_notes.py — but where llm_bench wants fresh
candidates (no status filter), we want GOLD, so we keep only notes at:

  tier "gold"        — lockedStatus == CURRENTLY_RATED_HELPFUL (survived the ~2-week
                       stabilization; cross-ideological by construction)  [Daniel 2026-07-23]
  tier "provisional" — currentStatus == CURRENTLY_RATED_HELPFUL, not (yet) locked

Kept separate from fc-gold on purpose: different label process (crowd consensus vs
professional fact-checker) and different unit (post-tied vs claim).

  uv run python -m eval.scripts.build_eval.harvest_community_notes [--snapshot-day 20260723]

The snapshot day is PINNED (--snapshot-day, default 20260723 = the day every Truth
Odds corpus was built from); pass `latest` to walk back from today instead.

Outputs:
  eval/data/community_notes/raw/            downloaded zips (kept for reproducibility)
  eval/data/community_notes/cn_gold.parquet joined + tiered CRH notes
"""
from __future__ import annotations

import argparse
import csv
import datetime
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

BASE = "https://ton.twimg.com/birdwatch-public-data"
UA = {"User-Agent": "Mozilla/5.0"}
RAW = Path("eval/data/community_notes/raw")
OUT = Path("eval/data/community_notes/cn_gold.parquet")

CRH = "CURRENTLY_RATED_HELPFUL"

# note fields we keep (by header name; the TSV has many rater-checkbox columns)
NOTE_FIELDS = ["noteId", "tweetId", "classification", "summary", "createdAtMillis",
               "misleadingFactualError", "misleadingManipulatedMedia",
               "misleadingOutdatedInformation", "misleadingMissingImportantContext",
               "misleadingUnverifiedClaimAsFact", "misleadingSatire",
               "notMisleadingFactuallyCorrect", "trustworthySources"]
NSH_FIELDS = ["noteId", "currentStatus", "lockedStatus", "timestampMillisOfStatusLock",
              "firstNonNMRStatus", "mostRecentNonNMRStatus"]


def _get(url, timeout=1800):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


SNAPSHOT_DAY = "20260723"   # the pinned dump day; --snapshot-day latest re-discovers


def latest_day():
    day = datetime.date.today()
    for _ in range(6):
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                f"{BASE}/{day:%Y/%m/%d}/notes/notes-00000.zip", headers=UA, method="HEAD"), timeout=20)
            if r.status == 200:
                return day
        except Exception:
            pass
        day -= datetime.timedelta(days=1)
    raise SystemExit("no recent snapshot found")


def download(day, family, prefix, cap=30):
    """Download all shards of one family for `day`; returns local paths. Skips files
    already fully downloaded (size match via HEAD)."""
    paths = []
    for i in range(cap):
        url = f"{BASE}/{day:%Y/%m/%d}/{family}/{prefix}-{i:05d}.zip"
        dest = RAW / f"{day:%Y%m%d}-{prefix}-{i:05d}.zip"
        try:
            head = urllib.request.urlopen(urllib.request.Request(url, headers=UA, method="HEAD"), timeout=30)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                break
            raise
        size = int(head.headers.get("Content-Length") or 0)
        if dest.exists() and dest.stat().st_size == size:
            print(f"  [cached] {dest.name} ({size/1e6:.0f}MB)", flush=True)
            paths.append(dest)
            continue
        t0 = time.time()
        with _get(url) as r, open(dest, "wb") as f:
            done = 0
            while chunk := r.read(1 << 22):
                f.write(chunk)
                done += len(chunk)
                if done % (1 << 26) < (1 << 22):
                    print(f"  {dest.name}: {done/1e6:.0f}/{size/1e6:.0f}MB "
                          f"({done/max(time.time()-t0,1)/1e6:.0f}MB/s)", flush=True)
        print(f"  [done] {dest.name} ({size/1e6:.0f}MB, {time.time()-t0:.0f}s)", flush=True)
        paths.append(dest)
    return paths


def parse_tsv(path, fields):
    """Stream one zipped TSV, keeping `fields` (by header name)."""
    rows = []
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            rdr = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"), delimiter="\t")
            for r in rdr:
                rows.append({k: r.get(k) for k in fields})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot-day", default=SNAPSHOT_DAY,
                    help="dump day YYYYMMDD (default the pinned %(default)s), "
                         "or `latest` to discover the most recent one")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    day = (latest_day() if args.snapshot_day == "latest"
           else datetime.datetime.strptime(args.snapshot_day, "%Y%m%d").date())
    print(f"snapshot day: {day} (cumulative full history)", flush=True)

    print("downloading noteStatusHistory...", flush=True)
    nsh_paths = download(day, "noteStatusHistory", "noteStatusHistory")
    print("downloading notes...", flush=True)
    note_paths = download(day, "notes", "notes")

    print("parsing noteStatusHistory...", flush=True)
    nsh = []
    for p in nsh_paths:
        nsh.extend(parse_tsv(p, NSH_FIELDS))
    nsh_df = pl.DataFrame(nsh)
    print(f"  {len(nsh_df)} status rows; currentStatus counts:", flush=True)
    print(nsh_df.group_by("currentStatus").len().sort("len", descending=True))

    # keep only CRH (current or locked) BEFORE the big join — everything else is out of scope
    keep_ids = nsh_df.filter((pl.col("currentStatus") == CRH) | (pl.col("lockedStatus") == CRH))
    print(f"  CRH (current or locked): {len(keep_ids)}", flush=True)

    print("parsing notes shards...", flush=True)
    notes = []
    for p in note_paths:
        notes.extend(parse_tsv(p, NOTE_FIELDS))
    notes_df = pl.DataFrame(notes)
    print(f"  {len(notes_df)} notes total", flush=True)

    df = keep_ids.join(notes_df, on="noteId", how="inner")
    df = df.with_columns(
        pl.when(pl.col("lockedStatus") == CRH).then(pl.lit("gold"))
          .otherwise(pl.lit("provisional")).alias("tier"),
        pl.from_epoch(pl.col("createdAtMillis").cast(pl.Int64), time_unit="ms")
          .dt.date().alias("note_date"),
        pl.lit(f"{day:%Y-%m-%d}").alias("snapshot_day"),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)
    print(f"\nwrote {OUT}: {len(df)} rows", flush=True)
    print(df.group_by("tier", "classification").len().sort("len", descending=True), flush=True)
    print("by year:", dict(df.with_columns(pl.col("note_date").dt.year().alias("y"))
          .group_by("y").len().sort("y").iter_rows()), flush=True)


if __name__ == "__main__":
    main()
