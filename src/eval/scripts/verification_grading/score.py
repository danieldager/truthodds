"""Score Tier-3 4-class verdicts against AVeriTeC gold labels.

Joins claims_n{N}.parquet (gold_label) with a verdicts parquet (verdict_4class) on
claim_id, and reports accuracy, macro-F1, per-class precision/recall/F1, the confusion
matrix, and the majority-class baseline. This is the direct comparison to ClaimCheck's
76.4% AVeriTeC-dev label accuracy.

    uv run python -m eval.scripts.verification_grading.score \
        -c eval/scripts/verification_grading/data/claims_n100.parquet \
        -v eval/scripts/verification_grading/data/verdicts_n100_v3.parquet
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import polars as pl

LABELS = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
SHORT = {"Supported": "SUP", "Refuted": "REF",
         "Not Enough Evidence": "NEI", "Conflicting Evidence/Cherrypicking": "CE"}


# Our harmonised gold says "Conflicting Evidence"; verify's verdict_4class (and AVeriTeC) says
# "Conflicting Evidence/Cherrypicking". Canonicalise to the verify/LABELS form so they compare.
_ALIAS = {"Conflicting Evidence": "Conflicting Evidence/Cherrypicking"}


def _norm(s: str | None) -> str:
    s = (s or "").strip()
    return _ALIAS.get(s, s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--claims", type=Path, required=True)
    ap.add_argument("-v", "--verdicts", type=Path, required=True)
    args = ap.parse_args()

    claims = pl.read_parquet(args.claims).select(["claim_id", "claim_text", "gold_label"])
    vcols = pl.read_parquet(args.verdicts)
    keep = [c for c in ("claim_id", "verdict_4class", "error", "rounds_used",
                        "stopped_reason", "llm_calls", "elapsed_seconds",
                        "n_urls_seen", "n_search_errors",
                        "resolving_provider", "providers_used") if c in vcols.columns]
    df = claims.join(vcols.select(keep), on="claim_id", how="inner")

    n_joined = df.height
    if n_joined != claims.height:
        print(f"WARNING: join produced {n_joined} rows from {claims.height} claims — "
              f"non-unique claim_id (claims unique={claims['claim_id'].n_unique()}). "
              f"Scores below are unreliable; regenerate claims with unique ids.")

    # Retrieval-health guard: if many claims got zero/errored retrieval, the score is
    # measuring "what the model guesses with no evidence", not verdict quality.
    if "n_urls_seen" in df.columns:
        zero_ret = df.filter(pl.col("n_urls_seen") == 0).height
        err = df.filter(pl.col("n_search_errors") > 0).height if "n_search_errors" in df.columns else 0
        if zero_ret or err:
            print(f"RETRIEVAL HEALTH: {zero_ret}/{n_joined} claims got ZERO results"
                  f"{f' ({err} hit search API errors)' if err else ''}. "
                  f"If high, the score is contaminated by missing evidence, not verdict quality.")
    ok = df.filter((pl.col("error").is_null()) & (pl.col("verdict_4class") != ""))
    n_err = n_joined - ok.height

    gold = [_norm(x) for x in ok["gold_label"].to_list()]
    pred = [_norm(x) for x in ok["verdict_4class"].to_list()]
    n = len(gold)
    if n == 0:
        print("no scorable rows.")
        return

    correct = sum(g == p for g, p in zip(gold, pred))
    acc = correct / n
    maj_label, maj_count = Counter(gold).most_common(1)[0]

    print(f"\nclaims: {args.claims.name}   verdicts: {args.verdicts.name}")
    print(f"joined={n_joined}  scored={n}  errors={n_err}")
    print(f"\n{'='*54}\n4-CLASS ACCURACY: {acc:.3f}  ({correct}/{n})")
    print(f"majority baseline ({SHORT[maj_label]}): {maj_count / n:.3f}")
    print("=" * 54)

    print(f"\n{'class':<8}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
    f1s = []
    for lab in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        support = sum(1 for g in gold if g == lab)
        print(f"{SHORT[lab]:<8}{prec:>7.3f}{rec:>7.3f}{f1:>7.3f}{support:>9}")
    print(f"\nmacro-F1: {sum(f1s) / len(f1s):.3f}")

    # Confusion matrix: rows = gold, cols = pred.
    idx = {lab: i for i, lab in enumerate(LABELS)}
    cm = [[0] * len(LABELS) for _ in LABELS]
    other = 0
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            cm[idx[g]][idx[p]] += 1
        else:
            other += 1
    print(f"\nconfusion (gold \\ pred):")
    print("        " + "".join(f"{SHORT[l]:>6}" for l in LABELS))
    for lab in LABELS:
        print(f"{SHORT[lab]:<6}" + "  " + "".join(f"{cm[idx[lab]][idx[c]]:>6}" for c in LABELS))
    if other:
        print(f"(off-label gold/pred rows: {other})")

    # Loop diagnostics if present.
    if "stopped_reason" in ok.columns:
        reasons = Counter(_norm(r) for r in ok["stopped_reason"].to_list())
        avg_rounds = sum(ok["rounds_used"].to_list()) / n
        avg_llm = sum(ok["llm_calls"].to_list()) / n
        avg_s = sum(ok["elapsed_seconds"].to_list()) / n
        print(f"\nloop: avg rounds={avg_rounds:.2f}  avg llm_calls={avg_llm:.1f}  "
              f"avg elapsed={avg_s:.1f}s")
        print(f"stops: " + "  ".join(f"{k}={v}" for k, v in reasons.most_common()))

    # Cascade figures: which provider resolved each claim + per-provider accuracy.
    if "resolving_provider" in ok.columns and any(ok["resolving_provider"].to_list()):
        rp = ok["resolving_provider"].to_list()
        used = ok["providers_used"].to_list()
        corr = [g == p for g, p in zip(gold, pred)]
        from collections import Counter as _C
        dist = _C(rp)
        escalated = sum(1 for u in used if u is not None and len(u) > 1)
        print(f"\nCASCADE: escalated past first provider on {escalated}/{n} claims "
              f"({escalated / n:.0%})")
        print("resolving provider (who got it confident) + accuracy when they resolved:")
        for prov, cnt in dist.most_common():
            pc = sum(1 for r, c in zip(rp, corr) if r == prov and c)
            print(f"  {prov:<8}: resolved {cnt:>3} ({cnt/n:.0%})   acc {pc}/{cnt}={pc/cnt:.2f}")

    # Misclassified examples (gold != pred), for eyeballing where it differs.
    if "claim_text" in ok.columns:
        miss = [(g, p, t) for g, p, t in
                zip(gold, pred, ok["claim_text"].to_list()) if g != p]
        if miss:
            print(f"\nmisclassified ({len(miss)}), first {min(12, len(miss))}:")
            for g, p, t in miss[:12]:
                print(f"  gold {SHORT.get(g, g[:3]):<4} -> pred {SHORT.get(p, p[:3]):<4}  {t[:90]}")


if __name__ == "__main__":
    main()
