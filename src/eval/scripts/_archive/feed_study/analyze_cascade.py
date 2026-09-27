"""Test the proposed pre-extraction cascade on EXISTING data (read-only, no LLM).

Cascade:  SELECTION (verifiable?) -> FABLE on the RAW POST (harmful?) -> decompose
Only posts that pass BOTH gates get decomposed/extracted for verification. FABLE
runs on the raw post (not the stripped claim) so the inflammatory/misleading
language is still present when harm is judged.

Inputs (all per-post booleans we already computed):
  claimify_<model>.parquet   -> verifiable   (Selection)
  fable_post_<model>.parquet -> fable_checkworthy (harm on raw post, current rule)

Reports, for each model and cross-model:
  - the funnel: posts -> verifiable -> (+harmful) = to-decompose
  - cross-model agreement (kappa) at each gate: selection, harm-on-post, cascade
  - harm agreement CONDITIONED on both-verifiable (the gate that matters)
  - a sample of the SELECTION-disagreement posts, to study the boundary

  uv run python -m eval.scripts.feed_study.analyze_cascade
  uv run python -m eval.scripts.feed_study.analyze_cascade --show 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

D = Path(__file__).parent / "data"


def _kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    both = sum(1 for x, y in zip(a, b) if x and y)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)
    po = (both + neither) / n
    ry = sum(a) / n
    rqy = sum(b) / n
    pe = ry * rqy + (1 - ry) * (1 - rqy)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def _lbl(k: float) -> str:
    return ("n/a" if k != k else "worse-than-chance" if k < 0 else "slight" if k < 0.2
            else "fair" if k < 0.4 else "moderate" if k < 0.6
            else "substantial" if k < 0.8 else "almost-perfect")


def _agree(a: list[bool], b: list[bool], na: str, nb: str) -> None:
    n = len(a)
    agr = 100 * sum(1 for x, y in zip(a, b) if x == y) / n
    both = sum(1 for x, y in zip(a, b) if x and y)
    ao = sum(1 for x, y in zip(a, b) if x and not y)
    bo = sum(1 for x, y in zip(a, b) if not x and y)
    neither = n - both - ao - bo
    print(f"    n={n}  %agree={agr:.1f}  kappa={_kappa(a,b):.3f} ({_lbl(_kappa(a,b))})")
    print(f"    both={both}  {na}-only={ao}  {nb}-only={bo}  neither={neither}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ga", type=Path, default=D / "claimify_gpt-oss-120b.parquet")
    ap.add_argument("--qa", type=Path, default=D / "claimify_qwen3-32b.parquet")
    ap.add_argument("--gf", type=Path, default=D / "fable_post_gpt-oss-120b.parquet")
    ap.add_argument("--qf", type=Path, default=D / "fable_post_qwen3-32b.parquet")
    ap.add_argument("--show", type=int, default=15, help="how many selection-disagreement posts to print")
    args = ap.parse_args()

    g_sel = pl.read_parquet(args.ga).filter(pl.col("error").is_null()).select(["post_id", "verifiable"])
    q_sel = pl.read_parquet(args.qa).filter(pl.col("error").is_null()).select(["post_id", "verifiable"])
    g_h = pl.read_parquet(args.gf).filter(pl.col("error").is_null()).select(
        ["post_id", "fable_checkworthy", "unit_text"])
    q_h = pl.read_parquet(args.qf).filter(pl.col("error").is_null()).select(
        ["post_id", "fable_checkworthy"])

    j = (g_sel.rename({"verifiable": "vg"})
         .join(q_sel.rename({"verifiable": "vq"}), on="post_id")
         .join(g_h.rename({"fable_checkworthy": "hg"}), on="post_id")
         .join(q_h.rename({"fable_checkworthy": "hq"}), on="post_id"))
    j = j.with_columns((pl.col("vg") & pl.col("hg")).alias("cg"),
                       (pl.col("vq") & pl.col("hq")).alias("cq"))
    n = j.height
    print(f"paired posts: {n}\n")

    print("=" * 64)
    print("FUNNEL: posts -> verifiable -> (+harmful on raw post) = to-decompose")
    for name, v, c in (("gpt-oss-120b", "vg", "cg"), ("qwen3-32b", "vq", "cq")):
        nv = int(j.select(pl.col(v).sum()).item())
        nc = int(j.select(pl.col(c).sum()).item())
        print(f"  {name:14s}: {n} -> verifiable {nv} ({100*nv/n:.0f}%) "
              f"-> +harmful {nc} ({100*nc/n:.0f}%)")

    print("\n" + "=" * 64)
    print("CROSS-MODEL AGREEMENT at each gate")
    print("  GATE 1 — selection (verifiable), all posts:")
    _agree(j["vg"].to_list(), j["vq"].to_list(), "gpt", "qwen")
    print("  GATE 2 — harm on raw post (FABLE checkworthy), all posts:")
    _agree(j["hg"].to_list(), j["hq"].to_list(), "gpt", "qwen")
    print("  GATE 2 — harm, CONDITIONED on both-verifiable (the real cascade subset):")
    sub = j.filter(pl.col("vg") & pl.col("vq"))
    _agree(sub["hg"].to_list(), sub["hq"].to_list(), "gpt", "qwen")
    print("  CASCADE — verifiable AND harmful (the to-decompose set):")
    _agree(j["cg"].to_list(), j["cq"].to_list(), "gpt", "qwen")

    print("\n" + "=" * 64)
    print(f"SELECTION-DISAGREEMENT posts (verifiable differs) — sample of {args.show}")
    dis = j.filter(pl.col("vg") != pl.col("vq")).select(["post_id", "vg", "vq", "unit_text"])
    print(f"  total selection disagreements: {dis.height}")
    take = min(args.show, dis.height)
    idxs = [round(i * (dis.height - 1) / max(take - 1, 1)) for i in range(take)] if dis.height else []
    for i in idxs:
        r = dis.row(i, named=True)
        who = "gpt=Y qwen=N" if r["vg"] else "gpt=N qwen=Y"
        print(f"  [{who}] {(r['unit_text'] or '')[:130].replace(chr(10),' ')}")


if __name__ == "__main__":
    main()
