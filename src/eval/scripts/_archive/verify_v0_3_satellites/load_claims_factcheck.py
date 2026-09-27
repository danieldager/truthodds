"""Build Tier-3 verification claims from our harvested dataset (factcheck_textonly.parquet).

  uv run python -m eval.scripts.verification_grading.load_claims_factcheck -n 200

Emits the SAME schema load_claims.py produces for AVeriTeC, so verify_run.py consumes either
interchangeably. We eval CONTENT-axis claims only (the system verifies the content, not "did X say
Y" — attribution is excluded from the headline). Stratified on the 4-class gold label with a floor
on the rare classes (NEE/Conflicting are thin in the population) so macro-F1 isn't dominated by
Refuted; within each class, allocate across publishers (largest-remainder) so no one publisher's
style dominates. Reproducible (seed). Importance-weight back to population for any prevalence headline.

claim_date falls back to review_date when missing — the date_ceiling leakage guard MUST have a bound,
else verify() would search a year of post-claim evidence (incl. the fact-check itself).
"""
from __future__ import annotations

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import polars as pl

SRC = Path("eval/data/factcheck_textonly.parquet")

# Per-class dev-200 targets: boost the rare classes above their population share so every class has
# enough for macro-F1, while Refuted stays the plurality (realistic). Sums to 200.
TARGETS = {"Refuted": 80, "Supported": 50, "Conflicting Evidence": 40, "Not Enough Evidence": 30}


def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _alloc(groups: dict[str, list], n: int, seed: int) -> list:
    """Largest-remainder proportional allocation across publisher groups, then sample."""
    rng = random.Random(seed)
    total = sum(len(v) for v in groups.values()) or 1
    exact = {p: n * len(v) / total for p, v in groups.items()}
    base = {p: int(e) for p, e in exact.items()}
    for p in sorted(exact, key=lambda k: exact[k] - int(exact[k]), reverse=True)[: n - sum(base.values())]:
        base[p] += 1
    picked: list = []
    for p, k in base.items():
        rows = groups[p][:]
        rng.shuffle(rows)
        picked.extend(rows[: min(k, len(rows))])
    return picked


def _stratified(df: pl.DataFrame, n: int, seed: int) -> list:
    groups: dict[str, list] = defaultdict(list)
    for r in df.to_dicts():
        groups[r["publisher_site"]].append(r)
    return _alloc(groups, n, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n", type=int, default=200, help="total claims (split across classes by TARGETS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()
    out = args.output or Path(__file__).parent / "data" / f"claims_factcheck_n{args.n}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Eval pool: content-axis, non-satire, well-formed, non-attribution (w/ Daniel, clog 250626).
    #   - satire pushes the verifier toward an ambiguous "no real reports" call;
    #   - is_checkable=False drops fragments/questions that aren't verifiable propositions;
    #   - is_attribution=True drops "did X say Y" claims (a different skill than content veracity).
    # All three stay metadata flags on the full set; just not sampled into the eval here.
    df = pl.read_parquet(SRC).filter(
        (pl.col("judged_axis") == "content")
        & (~pl.col("is_satire").fill_null(False))
        & pl.col("is_checkable").fill_null(True)
        & (~pl.col("is_attribution").fill_null(False))
    )
    scale = args.n / sum(TARGETS.values())
    picked: list = []
    for label, base in TARGETS.items():
        want = round(base * scale)
        sub = df.filter(pl.col("harmonised_label") == label)
        got = _stratified(sub, min(want, sub.height), args.seed)
        picked.extend(got)
        print(f"  {label:<22} want {want:>3}, have {sub.height:>4}, took {len(got):>3}")

    rows = [{
        "claim_id": stable_id(f"{r.get('claim_text', '')}||{r.get('review_url', '')}"),
        "claim_text": r.get("claim_text") or "",
        "gold_label": r.get("harmonised_label"),
        "claim_date": r.get("claim_date") or r.get("review_date") or None,  # fallback: leakage bound
        "speaker": r.get("claimant") or None,
        "fact_checking_article": r.get("review_url") or None,              # excluded from retrieval
        "gold_justification": "",
        "original_claim_url": r.get("claim_source_url") or None,
        # the raw original post (when resolved) — fed to the separate nudge/flag call, NOT the
        # verdict (verdict stays on the normalized claim). None for ~3/4 of rows (w/ Daniel).
        "raw_context": r.get("raw_context"),
        # extras for stratified scoring (verify_run ignores unknown keys)
        "publisher_site": r.get("publisher_site"),
        "language_code": r.get("language_code"),
        "is_satire": r.get("is_satire"),
    } for r in picked]
    out_df = pl.DataFrame(rows).unique(subset="claim_id", keep="first", maintain_order=True)
    out_df.write_parquet(out)

    print(f"\nwrote {out_df.height} rows -> {out}")
    print("gold_label:", out_df["gold_label"].value_counts(sort=True).to_dicts())
    print("language:", out_df["language_code"].value_counts(sort=True).to_dicts())
    print("publishers:", out_df["publisher_site"].n_unique(),
          "| satire:", out_df.filter(pl.col("is_satire") == True).height)  # noqa: E712


if __name__ == "__main__":
    main()
