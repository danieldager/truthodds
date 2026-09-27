"""Score the flag-step eval (flag_eval.parquet): does the context-aware post-FLAG beat the bare
veracity>=4 binary? The tension — catch more misleading-but-true posts (flag-recall up) WITHOUT
over-flagging true posts (pass-recall down). Reported per gold class, per soft_flag, per split,
plus a flip analysis (fixed vs broke) and a leak-suspect audit.

Run (from src/): uv run python -m eval.scripts.verdict_confidence.flag_eval_score
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

IN = Path("eval/scripts/verdict_confidence/data/flag_eval.parquet")


def _recalls(df: pl.DataFrame, pred_col: str) -> dict:
    flag = df.filter(pl.col("gold") == "flag")
    passx = df.filter(pl.col("gold") == "pass")
    fr = float((flag[pred_col] == "flag").mean()) if flag.height else float("nan")  # flag recall
    pr = float((passx[pred_col] == "pass").mean()) if passx.height else float("nan")  # pass recall (control)
    bal = (fr + pr) / 2
    return {"flag_recall": fr, "pass_recall": pr, "balanced_acc": bal}


def _line(name: str, n: int, bare: dict, post: dict) -> None:
    print(f"  {name:<26} n={n:<5} "
          f"flag-recall {bare['flag_recall']:.3f}→{post['flag_recall']:.3f}  "
          f"pass-recall {bare['pass_recall']:.3f}→{post['pass_recall']:.3f}  "
          f"bal-acc {bare['balanced_acc']:.3f}→{post['balanced_acc']:.3f}")


def main() -> None:
    df = pl.read_parquet(IN).filter(pl.col("error").is_null())
    n = df.height
    print(f"loaded {df.height} scored text claims  (leak-suspect: {int(df['leak_suspect'].sum())})\n")
    print("bare = veracity>=4 ; post = context-aware §3b flag step.  recalls: bare→post")

    slices = [
        ("ALL text", df),
        ("  soft_flag (misleading)", df.filter(pl.col("soft_flag"))),
        ("  hard-flag + clear", df.filter(~pl.col("soft_flag"))),
        ("dev", df.filter(pl.col("split") == "dev")),
        ("test", df.filter(pl.col("split") == "test")),
    ]
    print("\n### recalls by slice")
    for name, s in slices:
        if s.height:
            _line(name, s.height, _recalls(s, "bare_pred"), _recalls(s, "post_pred"))

    # Flip analysis: where does post change the bare decision, and is it right?
    print("\n### flip analysis (post vs bare, whole text set)")
    fixed = df.filter((pl.col("bare_correct") == 0) & (pl.col("post_correct") == 1)).height
    broke = df.filter((pl.col("bare_correct") == 1) & (pl.col("post_correct") == 0)).height
    print(f"  FIXED (bare wrong → post right): {fixed}")
    print(f"  BROKE (bare right → post wrong): {broke}")
    print(f"  net: {fixed - broke:+d}   (overall correct: bare {int(df['bare_correct'].sum())} "
          f"→ post {int(df['post_correct'].sum())} / {n})")

    # Over-flag cost: gold=pass posts the flag step wrongly flips to flag
    over = df.filter((pl.col("gold") == "pass") & (pl.col("bare_pred") == "pass") & (pl.col("post_pred") == "flag"))
    print(f"\n### over-flag cost: {over.height} clear-PASS posts flipped pass→flag by the flag step")
    for r in over.head(5).iter_rows(named=True):
        print(f"  {r['claim_id']}: {r['claim_text'][:60]}  | {r['post_flag_reason'][:60]}")

    # Leak-suspect audit
    leak = df.filter(pl.col("leak_suspect"))
    if leak.height:
        print(f"\n### leak-suspect reconstructions ({leak.height}) — review for verdict leakage")
        for r in leak.head(8).iter_rows(named=True):
            print(f"  {r['claim_id']}: {r['contextualized_post'][:80]}")


if __name__ == "__main__":
    main()
