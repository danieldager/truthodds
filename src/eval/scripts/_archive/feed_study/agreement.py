"""Pairwise detection-agreement report across Claimify runs, FABLE runs, detectors.

THIS PHASE measures AGREEMENT, not accuracy — we have no gold set yet (manual
annotation is a deferred TODO; see README). So the headline is inter-rater
agreement on a per-post BINARY decision, plus the confusion matrix.

Each input parquet is reduced to one row per post with a boolean `has_claim`:
  - Claimify run (`outcome` col)        -> has_claim = (outcome == 'claims')
  - FABLE run    (`fable_checkworthy`)  -> has_claim = fable_checkworthy
                                           (threshold-derived; see caveat below)
  - generic detector (`has_claim` col)  -> used directly

IMPORTANT — what is comparable:
  - Claimify(gpt-oss) vs Claimify(qwen)  : same construct (verifiability), diff model. Apples-to-apples.
  - FABLE(gpt-oss)    vs FABLE(qwen)     : same construct (harm>=thr), diff model. Apples-to-apples.
  - Claimify          vs FABLE           : DIFFERENT constructs (verifiable vs harmful). A low
                                           number here is expected, not a failure — read it as
                                           "how much do verifiability and harm overlap on this feed".
  - FABLE `has_claim` is THRESHOLD-DEPENDENT (fable_total >= threshold, default 15, UNCALIBRATED).
    The confusion matrix moves with the threshold; recalibration needs a gold set.

--prefilter <parquet>: also emit a section restricting every pairwise comparison
to the posts that input marks has_claim=true (e.g. "among FABLE-harmful posts,
do the two Claimify models agree better?").

  uv run python -m eval.scripts.feed_study.agreement              # globs claimify_*.parquet
  uv run python -m eval.scripts.feed_study.agreement -i a.parquet b.parquet c.parquet
  uv run python -m eval.scripts.feed_study.agreement -i claimify_gpt-oss-120b.parquet \\
      claimify_qwen3-32b.parquet --prefilter fable_post_qwen3-32b.parquet
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import polars as pl

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def _reduce(path: Path) -> tuple[str, pl.DataFrame]:
    """Reduce an input parquet to (label, df[post_id, has_claim])."""
    df = pl.read_parquet(path)
    modeltag = path.stem
    if "model" in df.columns:
        models = [m for m in df["model"].unique().to_list() if m]
        if len(models) == 1:
            modeltag = models[0].split("/")[-1]

    if "outcome" in df.columns:  # Claimify run
        out = df.select("post_id", (pl.col("outcome") == "claims").alias("has_claim"))
        label = f"claimify:{modeltag}"
    elif "fable_checkworthy" in df.columns:  # FABLE run
        mode = "post"
        if "mode" in df.columns:
            modes = [m for m in df["mode"].unique().to_list() if m]
            if len(modes) == 1:
                mode = modes[0]
        out = df.select("post_id", pl.col("fable_checkworthy").cast(pl.Boolean).alias("has_claim"))
        label = f"fable-{mode}:{modeltag}"
    elif "has_claim" in df.columns:  # generic detector
        out = df.select("post_id", pl.col("has_claim").cast(pl.Boolean))
        label = modeltag
    else:
        raise SystemExit(f"{path} has none of: outcome / fable_checkworthy / has_claim")

    out = out.unique(subset=["post_id"], keep="first")
    return label, out


def _cohen_kappa(a: list, b: list) -> float:
    """Cohen's kappa for two equal-length sequences of hashable labels."""
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b), key=str)
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    conf = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        conf[idx[x]][idx[y]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    row = [sum(conf[i]) for i in range(k)]
    col = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row[i] / n) * (col[i] / n) for i in range(k))
    if pe == 1.0:
        return 1.0  # degenerate single-category; avoid 0/0
    return (po - pe) / (1 - pe)


def _interpret_kappa(k: float) -> str:
    if k != k:  # nan
        return "n/a"
    if k < 0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def _pairwise_lines(label_a, da, label_b, db, restrict: set | None) -> list[str]:
    j = da.join(db, on="post_id", how="inner", suffix="_b")
    if restrict is not None:
        j = j.filter(pl.col("post_id").is_in(list(restrict)))
    n = j.height
    if n == 0:
        return [f"### {label_a} vs {label_b}\n", "- (no shared posts after filter)\n"]
    a = j["has_claim"].to_list()
    b = j["has_claim_b"].to_list()
    agree = sum(1 for x, y in zip(a, b) if x == y)
    both = sum(1 for x, y in zip(a, b) if x and y)
    neither = sum(1 for x, y in zip(a, b) if (not x) and (not y))
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if (not x) and y)
    kappa = _cohen_kappa(a, b)
    return [
        f"### {label_a} vs {label_b}  (n={n} shared posts)\n",
        f"- **% agreement:** {100.0 * agree / n:.1f}%",
        f"- **Cohen's kappa:** {kappa:.3f} ({_interpret_kappa(kappa)})",
        "",
        f"| | {label_b}=yes | {label_b}=no |",
        "|---|--:|--:|",
        f"| **{label_a}=yes** | {both} | {a_only} |",
        f"| **{label_a}=no** | {b_only} | {neither} |",
        "",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--inputs", type=Path, nargs="*", default=None,
                    help="parquet inputs (default: data/claimify_*.parquet)")
    ap.add_argument("--prefilter", type=Path, default=None,
                    help="restrict every comparison to posts this input marks has_claim=true")
    ap.add_argument("-o", "--out", type=Path,
                    default=RESULTS_DIR / "agreement_detection.md")
    args = ap.parse_args()

    inputs = args.inputs or sorted(DEFAULT_DATA_DIR.glob("claimify_*.parquet"))
    inputs = [DEFAULT_DATA_DIR / p if not p.exists() and (DEFAULT_DATA_DIR / p).exists() else p
              for p in inputs]
    if len(inputs) < 2:
        raise SystemExit(
            f"need >=2 inputs to compare; found {len(inputs)}: {[str(p) for p in inputs]}"
        )

    reduced = [_reduce(p) for p in inputs]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["# Detection / filter agreement (no gold set — agreement only)\n"]
    lines.append("## Per-input positive rate\n")
    lines.append("| input | posts | yes | rate |")
    lines.append("|---|--:|--:|--:|")
    for label, df in reduced:
        n = df.height
        hc = int(df.select(pl.col("has_claim").sum()).item() or 0)
        lines.append(f"| {label} | {n} | {hc} | {100.0 * hc / n:.1f}% |")
    lines.append("")

    lines.append("## Pairwise agreement (all posts)\n")
    for (la, da), (lb, db) in combinations(reduced, 2):
        lines += _pairwise_lines(la, da, lb, db, restrict=None)

    if args.prefilter is not None:
        pf_path = args.prefilter
        if not pf_path.exists() and (DEFAULT_DATA_DIR / pf_path).exists():
            pf_path = DEFAULT_DATA_DIR / pf_path
        pf_label, pf_df = _reduce(pf_path)
        keep = set(pf_df.filter(pl.col("has_claim"))["post_id"].to_list())
        lines.append(f"## Pairwise agreement — PREFILTERED to {pf_label}=yes ({len(keep)} posts)\n")
        lines.append(
            f"_Only posts where `{pf_label}` is positive. Tests whether models agree "
            f"better once we condition on this filter._\n"
        )
        for (la, da), (lb, db) in combinations(reduced, 2):
            lines += _pairwise_lines(la, da, lb, db, restrict=keep)

    report = "\n".join(lines)
    args.out.write_text(report)
    print(report)
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
