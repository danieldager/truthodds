"""WS4 — build the verifier-input claims file from the Stage-3 verdict gold.

Maps `verdict_dataset.parquet` rows into the schema `verification_grading/verify_run.py` consumes,
applying the locked leakage controls (verdict_eval_plan.md, revised w/ Daniel 2026-06-28):
  - date_ceiling = claim_date (when the claim circulated — the live-fact-checking cutoff), and
    where claim_date is missing (~40%) fall back to review_date; `ceiling_source` tags which, so
    the scorer can report the lenient-fallback rows separately.
  - exclude the row's own review_url (fact_checking_article). NO fact-check-domain block — the
    date ceiling does the rest ("we don't care if it finds fact-checks, just live performance").
  - image_serialization (WS1) is carried as `image_context` for the text-only verifier.

  uv run python -m eval.scripts.load_claims_verdict --split dev --balanced
  uv run python -m eval.scripts.load_claims_verdict --split dev --balanced --max 20 -o .../claims_smoke.parquet
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import polars as pl

GOLD = "eval/data/verdict_dataset.parquet"
SERIAL = "eval/data/verdict_image_serialization.parquet"
DEFAULT_DIR = Path("eval/scripts/verification_grading/data")


def _claim_id(url: str) -> str:
    return hashlib.md5((url or "").encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test", "train", "all"], default="dev")
    ap.add_argument("--balanced", action="store_true", help="use the down-sampled balanced slice")
    ap.add_argument("--content-only", action="store_true",
                    help="keep only judged_axis=='content' (headline veracity population)")
    ap.add_argument("--max", type=int, default=None, help="cap rows (smoke); stratified by gold_veracity")
    ap.add_argument("--serial-model", default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                    help="which image-serialization model's output to attach")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    df = pl.read_parquet(GOLD)
    if args.split != "all":
        df = df.filter(pl.col("split") == args.split)
    if args.balanced:
        df = df.filter(pl.col("balanced"))
    if args.content_only:
        df = df.filter(pl.col("judged_axis") == "content")

    # Resolve the eval date ceiling: claim_date when present, else review_date (tagged).
    has_cd = pl.col("claim_date").is_not_null() & (pl.col("claim_date").cast(pl.Utf8).str.len_chars() > 0)
    df = df.with_columns(
        pl.when(has_cd).then(pl.col("claim_date")).otherwise(pl.col("review_date")).alias("ceiling_date"),
        pl.when(has_cd).then(pl.lit("claim_date")).otherwise(pl.lit("review_date")).alias("ceiling_source"),
    )

    # Attach the WS1 image serialization (post image -> text), if computed for this model.
    if Path(SERIAL).exists():
        ser = (pl.read_parquet(SERIAL)
               .filter(pl.col("model") == args.serial_model)
               .select("review_url", pl.col("image_serialization").alias("_serial")))
        df = df.join(ser, on="review_url", how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("_serial"))

    out = df.select(
        pl.col("review_url").map_elements(_claim_id, return_dtype=pl.Utf8).alias("claim_id"),
        pl.col("claim_text"),
        pl.col("ceiling_date").alias("claim_date"),   # verify_run reads `claim_date` as the ceiling
        pl.col("ceiling_source"),
        pl.col("review_url").alias("fact_checking_article"),  # -> exclude_urls
        pl.col("raw_context"),                                 # -> post_context (nudge step)
        pl.col("_serial").alias("image_serialization"),        # -> image_context
        pl.col("gold_veracity"),
        pl.col("gold_veracity").cast(pl.Utf8).alias("gold_label"),  # trace compat
        pl.col("rating_subtype"),
        pl.col("judged_axis"),
        pl.col("has_image"),
        pl.col("publisher_site"),
        pl.col("language_code"),
        pl.col("source"),
        pl.col("split"),
        pl.col("balanced"),
    )

    if args.max and out.height > args.max:
        # stratified by gold_veracity so a smoke still spans the scale
        per = max(1, args.max // out["gold_veracity"].n_unique())
        out = out.group_by("gold_veracity").head(per).head(args.max)

    if args.output is None:
        tag = f"verdict_{args.split}{'_bal' if args.balanced else ''}{'_content' if args.content_only else ''}"
        if args.max:
            tag += f"_n{args.max}"
        args.output = DEFAULT_DIR / f"claims_{tag}.parquet"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.output)

    n_img = out.filter(pl.col("image_serialization").is_not_null() &
                       (pl.col("image_serialization").str.len_chars() > 0)).height
    n_img_rows = out.filter(pl.col("has_image")).height
    print(f"wrote {out.height} claims -> {args.output}")
    print(f"  veracity dist: {dict(sorted(out['gold_veracity'].value_counts().iter_rows()))}")
    print(f"  ceiling_source: {dict(out['ceiling_source'].value_counts().iter_rows())}")
    print(f"  judged_axis: {dict(out['judged_axis'].value_counts().iter_rows())}")
    print(f"  has_image rows: {n_img_rows}  | with serialization attached: {n_img}")
    if n_img_rows and n_img < n_img_rows:
        print(f"  WARNING: {n_img_rows - n_img} image rows have NO serialization — run "
              f"serialize_post_images.py --split {args.split} first (they'll verify text-only).")


if __name__ == "__main__":
    main()
