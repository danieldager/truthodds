"""Build the Stage-1 FILTER dev set from the audit results (docs/enrichment_audit_spec.md §3).

  uv run python -m eval.scripts.build_filter_dev

Joins `audit_results.parquet` (235B blind Pass-A gold) back to the harvest posts, derives the gold filter
label, applies the human provenance ledger (borderline-review decisions), flags borderlines
(235B↔Gemma), and writes `eval/data/filter_dev.parquet`.

Inclusion (Daniel 280628) — a row is in the filter dataset only if we can present the post:
  (1) raw post recoverable  → raw_tier ∈ {post, quote}        (real recovered text carrying the claim)
  (2) image-extractable claim → claim_locus ∈ {image, both}
  else → drop (synthetic option (3) deferred). The SIGHTED claim_matches_post does NOT gate the filter.

Label rule (gold = the 235B labeller; human ledger overrides on reviewed rows):
  drop      ← not readable, OR not included by (1)/(2)
  positive  ← readable ∧ included ∧ political_implication ∧ has_claim ∧ ¬obvious_joke
  negative  ← readable ∧ included ∧ (¬political_implication ∨ ¬has_claim ∨ obvious_joke)
              [obvious_joke = blatant satire → don't nudge; a DECEPTIVE satire that reads genuine → positive]
"""
from __future__ import annotations

import glob
from collections import Counter
from pathlib import Path

import polars as pl

AUDIT = Path("eval/data/audit_results.parquet")
LEDGER = Path("eval/data/filter_provenance.parquet")
OUT = Path("eval/data/filter_dev.parquet")
POST = ["raw_context", "image_paths", "has_image", "language_code", "raw_tier"]


def _included(raw_tier, claim_locus) -> bool:
    return raw_tier in ("post", "quote") or claim_locus in ("image", "both")


def _recommend(readable, included, political, has_claim, obvious_joke) -> tuple[str, str]:
    if not readable:
        return "drop", "unreadable"
    if not included:
        return "drop", "no_post"               # no recoverable post text (1) and no claim in image (2)
    if political and has_claim and not obvious_joke:
        return "positive", ""
    if obvious_joke:
        return "negative", "obvious_joke"      # blatant satire/parody — don't nudge (deceptive satire → positive)
    if not political:
        return "negative", "not_political"     # no political-polarization implication
    if not has_claim:
        return "negative", "no_claim"
    return "negative", "other"


def main() -> None:
    aud = pl.read_parquet(AUDIT).filter(pl.col("A_lab_political_implication").is_not_null())  # has-input rows
    look = {}
    for f in sorted(glob.glob("eval/data/*_harvest.parquet")):
        for r in pl.read_parquet(f).to_dicts():
            if r.get("review_url"):
                look[r["review_url"]] = {k: r.get(k) for k in POST}
    human = {}
    if LEDGER.exists():
        for r in pl.read_parquet(LEDGER).to_dicts():
            human[r["key"]] = r["recommend"]

    rows = []
    for a in aud.to_dicts():
        post = look.get(a["key"], {})
        rt = post.get("raw_tier")
        oj = bool(a.get("A_lab_obvious_joke"))
        inc = _included(rt, a.get("A_lab_claim_locus"))
        rec, reason = _recommend(a.get("A_lab_readable"), inc, a.get("A_lab_political_implication"),
                                 a.get("A_lab_has_claim"), oj)
        x_oj = bool(a.get("A_x_obvious_joke"))
        x_inc = _included(rt, a.get("A_x_claim_locus"))
        x_rec, _ = _recommend(a.get("A_x_readable"), x_inc, a.get("A_x_political_implication"),
                              a.get("A_x_has_claim"), x_oj)
        borderline = (a.get("A_lab_political_implication") != a.get("A_x_political_implication")
                      or a.get("A_lab_has_claim") != a.get("A_x_has_claim")
                      or oj != x_oj or rec != x_rec)
        src_label = "human" if a["key"] in human else "derived"
        if a["key"] in human:
            rec, reason = human[a["key"]], "human_review"
        rows.append({
            "key": a["key"], "source": a["source"], "claim_text": a.get("claim_text"),
            "raw_context": post.get("raw_context"), "image_paths": post.get("image_paths"),
            "has_image": post.get("has_image"), "language_code": post.get("language_code"),
            "raw_tier": rt, "topic": a.get("A_lab_topic"), "original_rating": a.get("original_rating"),
            "political_implication": a.get("A_lab_political_implication"), "has_claim": a.get("A_lab_has_claim"),
            "claim_locus": a.get("A_lab_claim_locus"), "readable": a.get("A_lab_readable"),
            "obvious_joke": oj, "is_satire": bool(a.get("B_lab_is_satire")), "included": inc,
            "recommend": rec, "neg_reason": reason, "borderline": borderline, "label_source": src_label,
        })
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_parquet(OUT)

    print(f"wrote {OUT}: {df.height} has-input rows")
    print("recommend  :", dict(Counter(r["recommend"] for r in rows)))
    print("neg reasons:", dict(Counter(r["neg_reason"] for r in rows if r["recommend"] == "negative")))
    print("drop reasons:", dict(Counter(r["neg_reason"] for r in rows if r["recommend"] == "drop")))
    print(f"human-labelled {sum(r['label_source']=='human' for r in rows)} | obvious_joke "
          f"{sum(r['obvious_joke'] for r in rows)} | gold-satire {sum(r['is_satire'] for r in rows)} "
          f"| borderline {sum(r['borderline'] for r in rows)} | filter-usable (pos+neg) "
          f"{sum(r['recommend'] in ('positive','negative') for r in rows)}")
    print("topic      :", Counter(r["topic"] for r in rows).most_common(8))


if __name__ == "__main__":
    main()
