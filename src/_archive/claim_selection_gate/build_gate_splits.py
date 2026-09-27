"""Build reproducible DEV/TEST splits for Stage-1 gate prompt iteration.

Rationale: the synthetic negatives are our ONLY (and partly-synthetic) negative set. If we tune the
gate prompt against all of them and report specificity on the same rows, we overfit. So we split:
prompt iteration looks at DEV only; TEST is measured once, as the honest generalization estimate.

Outputs (eval/data/):
  gate_mislabels.parquet      — 7 FP-audit MISLABELS pulled out of the negatives (actually check-worthy:
                                corruption / terrorism-financing / public policy). Run with --expect pass.
  negatives_dev/test.parquet  — cleaned negatives (547 − 7 mislabels), stratified 50/50 by category.
  positives_dev/test.parquet  — ALL real fact-checked image posts (has_image, no video), split by source.
  video_sample.parquet        — real video posts, for the claim_locus=='video' drop-rate check.

dev/test partition the ENTIRE eval set (every positive + every negative we gate against), so recall is
measured over the full positive set — we catch ANY positive the prompt starts dropping. Both pos and neg
are split 50/50 so each fold is a representative pos+neg mix. Split is deterministic (md5 of the text
key) so it's identical on every re-run / future session.
"""

# NOT REPRODUCIBLE AS WRITTEN (found 2026-07-28): FP_JSON below points at a Claude
# Code session scratchpad from a dead session; that file no longer exists anywhere in
# the repo or on disk. The gate_eval_positives/negatives_* splits it produced ARE
# still present and are cited by gate_dataset_provenance.md, so the outputs are usable
# — but this script cannot be re-run to regenerate or extend them. Fixing it means
# recovering fp_list.json (gone) or re-deriving the false-positive list from scratch.

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import polars as pl

DATA = Path("eval/data")
FP_JSON = Path(
    "/private/tmp/claude-503/-Users-daniel-dager-dev-disinform-factchecking-with-LLMs/"
    "e2f7fda8-1f86-4031-bd09-0873c637b6a7/scratchpad/fp_list.json"
)
# FP-audit verdict: indices in fp_list.json that are genuinely check-worthy (MISLABELS, not over-broadness).
MISLABEL_IDX = [6, 30, 41, 43, 49, 66, 87]
VIDEO_N = 80

# Out-of-scope fact-checks in the positive set (entertainment/sports/celebrity/product trivia) are
# reclassified as NEGATIVES (clog 260626) — candidates from triage_positives_scope.py, REVIEWED by hand.
# KEEP_OVERRIDE = classifier false-flags kept as positives: image-borne claims the text-only scan couldn't
# see, and genuine political/culture-war misinfo it mislabelled. (substring match on raw_context)
OOS_KEEP_OVERRIDE = [
    "Brigitte Macron", "Coca-Cola CEO", "Chimps Spotted Wearing Masks", "Truth Social has a new AI",
    "Ancient City Discovered Beneath Wyoming", "height of patriotism", "this is a Costco in Texas",
    "Christian nation for over 1,400", "Grindr in Phoenix", "Holy shit it", "PENTAGON RELEASED DOCUMENT",
    "reconstructed a 3D model", "Mookie Betts", "Iran player Mohammed Mohebi", "Folarin Balogun",
]


def confirmed_oos() -> set[str]:
    """Reviewed out-of-scope raw_contexts = triage flags minus the hand-kept false-flags."""
    tri = DATA / "positives_scope_triage.parquet"
    if not tri.exists():
        return set()
    flagged = pl.read_parquet(tri).filter(pl.col("out_of_scope") == True)["raw_context"].to_list()  # noqa: E712
    return {t for t in flagged if not any(k in (t or "") for k in OOS_KEEP_OVERRIDE)}


def h2(s: str) -> int:
    return int(hashlib.md5((s or "").encode("utf-8")).hexdigest()[:15], 16)  # 60-bit, fits UInt64


def split_5050(df: pl.DataFrame, key: str, strat: str | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stable 50/50 split: a row's fold = parity of md5(key), so it is FIXED regardless of how many other
    rows are in the pool. (Earlier rank-parity-within-stratum reshuffled folds on every rebuild — that
    scrambled the held-out test across dataset versions; clog 270626.) `strat` is unused, kept for call
    compatibility; md5 is uniform so per-source/category balance stays ~50/50."""
    df = df.with_columns((pl.col(key).map_elements(h2, return_dtype=pl.UInt64) % 2).alias("_f"))
    return df.filter(pl.col("_f") == 0).drop("_f"), df.filter(pl.col("_f") == 1).drop("_f")


def main() -> None:
    # ---- 1. mislabels + cleaned negatives ----------------------------------------------------
    fp = json.loads(FP_JSON.read_text())
    mislabel_texts = {fp[i]["text"] for i in MISLABEL_IDX}
    neg = pl.read_parquet(DATA / "synthetic_negatives.parquet")
    mis = neg.filter(pl.col("text").is_in(mislabel_texts))
    clean = neg.filter(~pl.col("text").is_in(mislabel_texts))
    mis.write_parquet(DATA / "gate_mislabels.parquet")
    assert mis.height == len(MISLABEL_IDX), f"matched {mis.height} mislabels, expected {len(MISLABEL_IDX)}"
    # video negatives belong to the video-drop metric (measured on real video_sample), not check-worthiness
    clean = clean.filter(pl.col("category") != "video_dependent")

    # ---- 2. positives: real fact-checked image posts (no video) ------------------------------
    cols = ["raw_context", "image_paths", "image_est_tokens", "has_video", "quoted_text", "language_code"]
    pos_parts = []
    for f in sorted(glob.glob(str(DATA / "*_harvest.parquet"))):
        d = pl.read_parquet(f)
        if "has_image" not in d.columns:
            continue
        sub = d.filter((pl.col("has_image") == True) & (pl.col("has_video") != True))  # noqa: E712
        sub = sub.filter(pl.col("raw_context").is_not_null() & (pl.col("raw_context").str.len_chars() > 0))
        sub = sub.select([c for c in cols if c in sub.columns])
        src = Path(f).stem.replace("_harvest", "")
        pos_parts.append(sub.with_columns(pl.lit(src).alias("source")))
    pos = pl.concat(pos_parts, how="diagonal_relaxed").unique("raw_context", keep="first")
    pos = pos.rename({"language_code": "lang"}) if "lang" not in pos.columns and "language_code" in pos.columns else pos

    # reclassify reviewed out-of-scope fact-checks: drop from positives, fold into negatives as image-bearing
    oos = confirmed_oos()
    oos_rows = pos.filter(pl.col("raw_context").is_in(oos))
    pos = pos.filter(~pl.col("raw_context").is_in(oos)).with_columns(pl.lit("positive").alias("category"))
    oos_neg = oos_rows.rename({"raw_context": "text"}).with_columns(pl.lit("oos_factcheck").alias("category"))
    clean = pl.concat([clean, oos_neg], how="diagonal_relaxed")

    ndev, ntest = split_5050(clean, "text", "category")
    ndev.write_parquet(DATA / "negatives_dev.parquet")
    ntest.write_parquet(DATA / "negatives_test.parquet")
    pdev, ptest = split_5050(pos, "raw_context", "source")
    pdev.write_parquet(DATA / "positives_dev.parquet")
    ptest.write_parquet(DATA / "positives_test.parquet")
    print(f"reclassified out-of-scope positives -> negatives: {oos_neg.height}")

    # ---- 3. video sample (drop-rate check) ---------------------------------------------------
    vid_parts = []
    for f in sorted(glob.glob(str(DATA / "*_harvest.parquet"))):
        d = pl.read_parquet(f)
        if "has_video" not in d.columns:
            continue
        sub = d.filter(pl.col("has_video") == True).select([c for c in cols if c in d.columns])  # noqa: E712
        sub = sub.with_columns(pl.lit(Path(f).stem.replace("_harvest", "")).alias("source"))
        vid_parts.append(sub)
    vid = pl.concat(vid_parts, how="diagonal_relaxed")
    vid = vid.with_columns(pl.col("raw_context").fill_null("").map_elements(h2, return_dtype=pl.UInt64).alias("_h")).sort("_h").head(VIDEO_N).drop("_h")
    vid = vid.with_columns(pl.lit("video").alias("category"))
    vid.write_parquet(DATA / "video_sample.parquet")

    # ---- 4. image-only positives (empty post text — claim lives purely in the image) ---------
    # Excluded from positives_dev/test because the harness keyed on text and they'd collapse; gate them
    # via a uid key (first image path). Tests the prompt's image-claim clause on pure-visual claims.
    io_parts = []
    for f in sorted(glob.glob(str(DATA / "*_harvest.parquet"))):
        d = pl.read_parquet(f)
        if "has_image" not in d.columns:
            continue
        sub = d.filter((pl.col("has_image") == True) & (pl.col("has_video") != True))  # noqa: E712
        sub = sub.filter(pl.col("raw_context").is_null() | (pl.col("raw_context").str.len_chars() == 0))
        sub = sub.select([c for c in cols if c in sub.columns])
        io_parts.append(sub.with_columns(pl.lit(Path(f).stem.replace("_harvest", "")).alias("source")))
    io = pl.concat(io_parts, how="diagonal_relaxed").with_columns(pl.col("image_paths").list.first().alias("uid"))
    io = io.filter(pl.col("uid").is_not_null()).unique("uid", keep="first")
    io = io.rename({"language_code": "lang"}) if "language_code" in io.columns else io
    io = io.with_columns(pl.lit("image_only").alias("category"))
    iodev, iotest = split_5050(io, "uid", "source")
    iodev.write_parquet(DATA / "image_only_dev.parquet")
    iotest.write_parquet(DATA / "image_only_test.parquet")

    # ---- report ------------------------------------------------------------------------------
    print(f"mislabels (recovered positives): {mis.height}")
    print(f"cleaned negatives: {clean.height}  -> dev {ndev.height} / test {ntest.height}")
    print("  by category dev:", ndev.group_by("category").agg(pl.len()).sort("category").to_dicts())
    print("  by category test:", ntest.group_by("category").agg(pl.len()).sort("category").to_dicts())
    print(f"positives: {pos.height}  -> dev {pdev.height} / test {ptest.height}")
    print("  by source dev:", pdev.group_by("source").agg(pl.len()).sort("source").to_dicts())
    print(f"video sample: {vid.height}")
    print(f"image-only positives: {io.height}  -> dev {iodev.height} / test {iotest.height}")


if __name__ == "__main__":
    main()
