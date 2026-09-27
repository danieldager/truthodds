"""Analyse the cascade — funnel, embedding↔LLM agreement, 4-prong vs FABLE.

Reads ``graded_posts_x862.parquet`` and ``graded_claims_x862.parquet`` (from
``stage6_grade.py``) and emits a markdown report plus stdout:

1. **Funnel** — the cascade as a monotone subset chain:
       862 posts → in_scope_llm → has_claim posts
       ‖ (unit switches to claims)
       total claims → check-worthy (misinfo_candidate OR fable_checkworthy)
       → FCT-matched.
   `in_scope_embed` is reported alongside but is a *parallel* measurement
   (Stage 2 runs the LLM on all 862), not a gate — so it is excluded from the
   monotonicity check.
2. **Embedding ↔ LLM agreement** — 2×2 confusion of in_scope_embed vs
   in_scope_llm over all 862 posts.
3. **4-prong vs FABLE** — 2×2 confusion of misinfo_candidate vs
   fable_checkworthy over all claims, with example claims from each
   disagreement cell.
4. **Over-decomposition** — posts with n_claims > 8 and the worst case.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

DEFAULT_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_RESULTS_DIR = Path(__file__).parent / "results"
OVERDECOMP_THRESHOLD = 8


def _b(df: pl.DataFrame, col: str) -> pl.Expr:
    """Boolean column as a fill-null(False) expression (missing → False)."""
    if col not in df.columns:
        return pl.lit(False)
    return pl.col(col).fill_null(False)


def _count_true(df: pl.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df.filter(_b(df, col)).height)


def confusion(df: pl.DataFrame, a: str, b: str) -> dict[str, int]:
    # with_columns (not select) so a missing column's pl.lit(False) broadcasts
    # across df.height rows — otherwise both-missing would collapse to 1 row.
    g = df.with_columns(_b(df, a).alias("_a"), _b(df, b).alias("_b"))
    return {
        "tt": g.filter(pl.col("_a") & pl.col("_b")).height,
        "tf": g.filter(pl.col("_a") & ~pl.col("_b")).height,
        "ft": g.filter(~pl.col("_a") & pl.col("_b")).height,
        "ff": g.filter(~pl.col("_a") & ~pl.col("_b")).height,
    }


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "—"


def build_report(posts: pl.DataFrame, claims: pl.DataFrame) -> str:
    L: list[str] = []
    w = L.append

    # --- 1. Funnel -------------------------------------------------------
    n_posts = posts.height
    n_embed = _count_true(posts, "in_scope_embed")
    n_llm = _count_true(posts, "in_scope_llm")
    n_hasclaim = _count_true(posts, "extractor_has_claim")
    n_claims = claims.height
    checkworthy_expr = _b(claims, "misinfo_candidate") | _b(claims, "fable_checkworthy")
    n_checkworthy = int(claims.filter(checkworthy_expr).height)
    n_fct = _count_true(claims, "fct_match")

    w("# Claim cascade — funnel & rubric comparison\n")
    w("## 1. Funnel\n")
    w("| Stage | Unit | Count | % of unit | % of prev |")
    w("|---|---|---:|---:|---:|")
    # (label, unit, count, unit_total, prev): % of unit divides by the row's
    # own population (posts ÷ posts, claims ÷ claims) — never claims ÷ posts.
    rows = [
        ("Captured posts", "posts", n_posts, n_posts, n_posts),
        ("In scope (LLM, Stage 2)", "posts", n_llm, n_posts, n_posts),
        ("Has ≥1 claim (Stage 3)", "posts", n_hasclaim, n_posts, n_llm),
        ("Extracted claims", "claims", n_claims, n_claims, n_hasclaim),
        ("Check-worthy (4-prong OR FABLE)", "claims", n_checkworthy, n_claims, n_claims),
    ]
    for label, unit, count, unit_total, prev in rows:
        w(f"| {label} | {unit} | {count} | {_pct(count, unit_total)} | {_pct(count, prev)} |")
    w(f"\n*Parallel measurement (not a gate):* in_scope_embed = **{n_embed}** "
      f"({_pct(n_embed, n_posts)} of posts) — Stage 2 runs the LLM on all "
      f"{n_posts} posts, so embedding scope is logged, not gated on.\n")
    w(f"**Tier-2 short-circuit (FCT):** of the {n_checkworthy} check-worthy "
      f"claims, **{n_fct}** ({_pct(n_fct, n_checkworthy)}) already have a "
      f"published fact-check; the other **{n_checkworthy - n_fct}** need novel "
      f"verification. FCT runs on the whole check-worthy union — this is a "
      f"short-circuit *rate*, not a narrowing gate.\n")

    # Monotonicity over the genuine gate chain (split at the posts→claims
    # boundary). FCT is excluded: matched claims EXIT the pipeline rather than
    # narrowing toward more work, so it is not a forward gate.
    post_chain = [n_posts, n_llm, n_hasclaim]
    claim_chain = [n_claims, n_checkworthy]
    mono = all(a >= b for a, b in zip(post_chain, post_chain[1:])) and \
        all(a >= b for a, b in zip(claim_chain, claim_chain[1:]))
    w(f"**Monotonic gate chain:** {'PASS' if mono else 'FAIL'} "
      f"(posts {post_chain}, claims {claim_chain})\n")

    # --- 2. Embedding ↔ LLM agreement -----------------------------------
    w("## 2. Embedding ↔ LLM scope agreement (n = posts)\n")
    c = confusion(posts, "in_scope_embed", "in_scope_llm")
    n = n_posts
    agree = c["tt"] + c["ff"]
    w("| | LLM in-scope | LLM out |")
    w("|---|---:|---:|")
    w(f"| **Embed in-scope** | {c['tt']} | {c['tf']} |")
    w(f"| **Embed out** | {c['ft']} | {c['ff']} |")
    w(f"\nAgreement: **{_pct(agree, n)}** ({agree}/{n}). "
      f"Embed-only (embed yes / LLM no): {c['tf']}; "
      f"LLM-only (LLM yes / embed no): {c['ft']}.\n")

    # --- 3. 4-prong vs FABLE --------------------------------------------
    w("## 3. Check-worthiness rubrics — 4-prong vs FABLE (n = claims)\n")
    cc = confusion(claims, "misinfo_candidate", "fable_checkworthy")
    nc = n_claims
    agree_c = cc["tt"] + cc["ff"]
    w("| | FABLE checkworthy | FABLE not |")
    w("|---|---:|---:|")
    w(f"| **4-prong candidate** | {cc['tt']} | {cc['tf']} |")
    w(f"| **4-prong not** | {cc['ft']} | {cc['ff']} |")
    w(f"\nAgreement: **{_pct(agree_c, nc)}** ({agree_c}/{nc}). "
      f"Both: {cc['tt']}; 4-prong only: {cc['tf']}; FABLE only: {cc['ft']}; "
      f"neither: {cc['ff']}.\n")

    def _examples(expr: pl.Expr, k: int = 5) -> list[str]:
        if "claim_text" not in claims.columns:
            return []
        sub = claims.filter(expr).select("claim_text").head(k)
        return [t for t in sub["claim_text"].to_list() if t]

    w("**4-prong says check-worthy but FABLE-low (verifiable-but-low-harm):**\n")
    for t in _examples(_b(claims, "misinfo_candidate") & ~_b(claims, "fable_checkworthy")):
        w(f"- {t}")
    w("\n**FABLE says check-worthy but 4-prong says no (high-harm, not 4-prong):**\n")
    for t in _examples(~_b(claims, "misinfo_candidate") & _b(claims, "fable_checkworthy")):
        w(f"- {t}")
    w("")

    # --- 4. Over-decomposition ------------------------------------------
    w("## 4. Over-decomposition\n")
    if "n_claims" in posts.columns:
        over = posts.filter(pl.col("n_claims") > OVERDECOMP_THRESHOLD)
        w(f"Posts with > {OVERDECOMP_THRESHOLD} claims: **{over.height}**.")
        if over.height:
            worst = over.sort("n_claims", descending=True).head(1)
            pid = worst["post_id"][0]
            nmax = worst["n_claims"][0]
            snippet = ""
            if "text" in worst.columns and worst["text"][0]:
                snippet = worst["text"][0][:160].replace("\n", " ")
            w(f"Worst case: `{pid}` with **{nmax}** claims. {snippet}")
    else:
        w("(n_claims column absent — skipped.)")
    w("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = ap.parse_args()

    posts_path = args.data_dir / "graded_posts_x862.parquet"
    claims_path = args.data_dir / "graded_claims_x862.parquet"
    for p in (posts_path, claims_path):
        if not p.exists():
            raise SystemExit(f"missing {p} — run stage6_grade.py first")

    posts = pl.read_parquet(posts_path)
    claims = pl.read_parquet(claims_path)
    report = build_report(posts, claims)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / "funnel_report.md"
    out.write_text(report)
    print(report)
    print(f"\n(report written to {out})")


if __name__ == "__main__":
    main()
