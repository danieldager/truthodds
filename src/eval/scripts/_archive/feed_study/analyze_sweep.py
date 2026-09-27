"""Post-mortem analysis of the filter sweep. Read-only; writes markdown to results/.

Reads master_post.parquet + master_claim.parquet (from build_masters.py) and reports:
  1. Cross-model agreement per filter x mode (RAW post-level + claim/both claim-level).
  2. Extraction agreement: n_claims stats + llama judge_compare distribution.
  3. Normalization/decomposition quality (llama): score dists, flag rate, flagged dump.
  4. Raw-vs-claim stripping (same model scores post and its claims).
  5. Offline filter-order simulator (V/R/H, RAW post-level): funnel per ordering +
     cross-model agreement on the surviving set. Zero LLM.
  6. Inter-filter overlap.

  uv run python -m eval.scripts.feed_study.analyze_sweep --dir eval/scripts/feed_study/data/sample1000
"""

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import polars as pl

RESULTS = Path(__file__).parent / "results"


def _kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    both = sum(1 for x, y in zip(a, b) if x and y)
    nei = sum(1 for x, y in zip(a, b) if not x and not y)
    po = (both + nei) / n
    ry, rqy = sum(a) / n, sum(b) / n
    pe = ry * rqy + (1 - ry) * (1 - rqy)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def _lbl(k: float) -> str:
    return ("n/a" if k != k else "worse" if k < 0 else "slight" if k < 0.2 else "fair"
            if k < 0.4 else "moderate" if k < 0.6 else "substantial" if k < 0.8 else "almost-perfect")


def _agree_pair(df: pl.DataFrame, ca: str, cb: str) -> str:
    if ca not in df.columns or cb not in df.columns:
        return f"- `{ca}` vs `{cb}`: (missing column)\n"
    sub = df.select(ca, cb).drop_nulls()
    n = sub.height
    if n == 0:
        return f"- `{ca}` vs `{cb}`: (no paired rows)\n"
    a, b = sub[ca].to_list(), sub[cb].to_list()
    agree = 100 * sum(1 for x, y in zip(a, b) if x == y) / n
    k = _kappa(a, b)
    ra, rb = 100 * sum(a) / n, 100 * sum(b) / n
    return (f"- `{ca}`({ra:.0f}%) vs `{cb}`({rb:.0f}%): n={n}, "
            f"agree={agree:.1f}%, kappa={k:.3f} ({_lbl(k)})\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, default=RESULTS / "sweep_report.md")
    args = ap.parse_args()

    mp = pl.read_parquet(args.dir / "master_post.parquet")
    mc = pl.read_parquet(args.dir / "master_claim.parquet")
    L: list[str] = ["# Filter sweep — post-mortem (agreement only, no gold)\n"]
    L.append(f"posts: {mp.height} | claims: {mc.height} "
             f"(by extractor: {dict(mc.group_by('extractor_model').len().iter_rows())})\n")

    # 1. cross-model / cross-scorer agreement
    L.append("## 1. Cross-model agreement per filter × mode\n")
    L.append("### RAW (post-level, both models score the post)\n")
    for ca, cb in [("verif_raw_g", "verif_raw_q"), ("relev_raw_g", "relev_raw_q"),
                   ("harm_raw_g", "harm_raw_q")]:
        L.append(_agree_pair(mp, ca, cb))
    L.append("\n### CLAIM / BOTH (claim-level, both scorers score the same claim)\n")
    for ca, cb in [("relev_claim_g", "relev_claim_q"), ("relev_both_g", "relev_both_q"),
                   ("harm_claim_g", "harm_claim_q"), ("harm_both_g", "harm_both_q")]:
        L.append(_agree_pair(mc, ca, cb))

    # 2. extraction agreement
    L.append("\n## 2. Extraction agreement between models\n")
    if {"n_claims_g", "n_claims_q"} <= set(mp.columns):
        s = mp.select("n_claims_g", "n_claims_q").fill_null(0)
        g, q = s["n_claims_g"].to_list(), s["n_claims_q"].to_list()
        both = sum(1 for x, y in zip(g, q) if x > 0 and y > 0)
        gonly = sum(1 for x, y in zip(g, q) if x > 0 and y == 0)
        qonly = sum(1 for x, y in zip(g, q) if x == 0 and y > 0)
        exact = 100 * sum(1 for x, y in zip(g, q) if x == y) / len(g)
        w1 = 100 * sum(1 for x, y in zip(g, q) if abs(x - y) <= 1) / len(g)
        L.append(f"- n_claims: exact match={exact:.1f}%, |diff|<=1={w1:.1f}%\n")
        L.append(f"- both-extracted={both}, gpt-only={gonly}, qwen-only={qonly}, "
                 f"neither={len(g)-both-gonly-qonly}\n")
    if "cmp_agreement" in mp.columns:
        dist = dict(mp.select("cmp_agreement").drop_nulls()
                    .group_by("cmp_agreement").len().iter_rows())
        more = dict(mp.select("cmp_more_complete").drop_nulls()
                    .group_by("cmp_more_complete").len().iter_rows())
        L.append(f"- llama claim-set comparison (both-extracted posts): agreement {dist}\n")
        L.append(f"  more-complete: {more} (A=gpt, B=qwen)\n")

    # 3. normalization/decomposition quality
    L.append("\n## 3. Normalization/decomposition quality (llama judge)\n")
    if {"faithful", "decontextualized", "atomicity", "extractor_model"} <= set(mc.columns):
        for ext, sub in mc.group_by("extractor_model"):
            ext = ext[0] if isinstance(ext, tuple) else ext
            means = {d: round(sub[d].drop_nulls().mean() or 0, 2)
                     for d in ("faithful", "decontextualized", "atomicity")}
            L.append(f"- **{ext}** per-claim means: {means}\n")
        # per-post coverage + flag (dedupe to post level)
        if "flag" in mc.columns:
            pf = mc.select("extractor_model", "post_id", "coverage", "flag").unique(
                subset=["extractor_model", "post_id"])
            for ext, sub in pf.group_by("extractor_model"):
                ext = ext[0] if isinstance(ext, tuple) else ext
                fl = dict(sub.group_by("flag").len().iter_rows())
                cov = round(sub["coverage"].drop_nulls().mean() or 0, 2)
                L.append(f"  {ext}: coverage mean={cov}, flag dist={fl}\n")
        # dump flagged claims for eyeball
        flagged = mc.filter(pl.col("flag").is_in(["borderline", "bad"])) if "flag" in mc.columns else mc.head(0)
        if flagged.height:
            fl_lines = ["# Quality-flagged claims (borderline/bad) — eyeball\n"]
            for r in flagged.sort("flag").iter_rows(named=True):
                fl_lines.append(
                    f"- [{r['flag']}] ext={r['extractor_model']} "
                    f"f={r.get('faithful')} d={r.get('decontextualized')} a={r.get('atomicity')} "
                    f"| {(r.get('claim_text') or '')[:110]}\n")
            (RESULTS / "quality_flagged.md").write_text("".join(fl_lines))
            L.append(f"- flagged (borderline/bad) claims dumped: {flagged.height} "
                     f"-> results/quality_flagged.md\n")

    # 4. raw-vs-claim stripping (same model)
    L.append("\n## 4. Raw-vs-claim stripping (same model scores post and its claims)\n")
    for t, name in (("g", "gpt-oss"), ("q", "qwen")):
        hr, hc = f"harm_raw_{t}", f"harm_claim_{t}"
        if hr in mp.columns and hc in mc.columns:
            j = (mc.select("post_id", hc)
                   .join(mp.select("post_id", hr), on="post_id", how="inner").drop_nulls())
            if j.height:
                raw_pos = j.filter(pl.col(hr)).height
                claim_pos = j.filter(pl.col(hc)).height
                flip_down = j.filter(pl.col(hr) & ~pl.col(hc)).height  # post harmful, claim not
                L.append(f"- **{name}** harm: post-harmful claims={raw_pos}, claim-harmful={claim_pos}, "
                         f"post→claim drop (harmful post, claim not)={flip_down} of {j.height}\n")

    # 5. offline filter-order simulator (RAW post-level)
    L.append("\n## 5. Filter-order simulator (RAW post-level, AND of V/R/H)\n")
    L.append("_Final survivor set is order-independent (AND); the funnel shows where volume dies._\n")
    filt = {"V": "verif_raw", "R": "relev_raw", "H": "harm_raw"}
    for t, name in (("g", "gpt-oss"), ("q", "qwen")):
        cols = {k: f"{v}_{t}" for k, v in filt.items()}
        if not all(c in mp.columns for c in cols.values()):
            L.append(f"- {name}: (missing filter columns)\n")
            continue
        base = mp.select(["post_id"] + list(cols.values())).with_columns(
            [pl.col(c).fill_null(False) for c in cols.values()])
        n0 = base.height
        L.append(f"\n**{name}** (start {n0} posts):\n")
        for order in permutations("VRH"):
            cur = base
            funnel = []
            for f in order:
                cur = cur.filter(pl.col(cols[f]))
                funnel.append(f"{f}={cur.height}")
            L.append(f"- {'→'.join(order)}: {' '.join(funnel)}\n")
    # cross-model agreement on final survivor
    if all(f"{v}_{t}" in mp.columns for v in filt.values() for t in ("g", "q")):
        surv = mp.with_columns(
            (pl.col("verif_raw_g").fill_null(False) & pl.col("relev_raw_g").fill_null(False)
             & pl.col("harm_raw_g").fill_null(False)).alias("surv_g"),
            (pl.col("verif_raw_q").fill_null(False) & pl.col("relev_raw_q").fill_null(False)
             & pl.col("harm_raw_q").fill_null(False)).alias("surv_q"),
        )
        L.append("\n**Cross-model agreement on final survivor set (V∧R∧H, raw):**\n")
        L.append(_agree_pair(surv, "surv_g", "surv_q"))

    # 6. inter-filter overlap (one model, raw)
    L.append("\n## 6. Inter-filter overlap (gpt-oss, raw)\n")
    for ca, cb in [("verif_raw_g", "relev_raw_g"), ("verif_raw_g", "harm_raw_g"),
                   ("relev_raw_g", "harm_raw_g")]:
        L.append(_agree_pair(mp, ca, cb))

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = "".join(L)
    args.out.write_text(report)
    print(report)
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
