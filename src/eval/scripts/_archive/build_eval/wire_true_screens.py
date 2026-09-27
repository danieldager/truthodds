"""Wire TRUE-stratum screens + register-blind check (audit stage, no verification).

Input: wire_true_verify_input.parquet (chain output over AP+Reuters text-only
posts). Applies the FALSE-side screen set as tags: attribution-form (from the
chain's own claim type + regex), media-locus, claim-level near-dup, date decode.
Then the register-blind check: bag-of-words logistic classifier on claim text,
wire-TRUE vs the 99 gated-FALSE CN claims from the A/B run. Outputs
eval/data/urn_runs/wire_audit/{screens.parquet, summary.json}.
"""
import json, re, sys
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
TC = SRC / "eval/data/tweet_corpus"
OUT = SRC / "eval/data/urn_runs/wire_audit"

MEDIA_RE = re.compile(
    r"\b(video|photo|image|footage|clip|picture|audio|recording|screenshot)s?\b.{0,40}"
    r"\b(show|shows|showing|depict|depicts|depicting|captur|of|from|is real|is fake|"
    r"is authentic|AI-generated|doctored|manipulated|edited)\b", re.I)
ATTR_RE = re.compile(
    r"\b(said|says|say|stated|states|claimed|claims|announced|announces|according to|"
    r"told|tells|warned|warns|denied|denies|reported that|reports that)\b", re.I)


def _tokens(s):
    return set(re.findall(r"[a-z0-9]{3,}", (s or "").lower()))


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    df = pl.read_parquet(TC / "wire_true_verify_input.parquet")
    claim_col = "claim_resolved" if "claim_resolved" in df.columns else (
        "claim_normalized" if "claim_normalized" in df.columns else "claim")
    rows = df.to_dicts()
    print(f"{len(rows)} claims from {df['post_id'].n_unique()} posts; cols={df.columns}",
          flush=True)

    # screens as tags
    toksets, kept = [], []
    n_attr = n_media = n_dup = 0
    for r in rows:
        c = r[claim_col] or ""
        r["tag_attr"] = bool(ATTR_RE.search(c)) or (str(r.get("type", "")) == "attribution")
        r["tag_media"] = bool(MEDIA_RE.search(c))
        t = _tokens(c)
        r["tag_dup"] = any(_jaccard(t, u) >= 0.8 for u in toksets)
        toksets.append(t)
        n_attr += r["tag_attr"]; n_media += r["tag_media"]; n_dup += r["tag_dup"]
        r["fit_eligible"] = not (r["tag_attr"] or r["tag_media"] or r["tag_dup"])
        kept.append(r)
    elig = [r for r in kept if r["fit_eligible"]]
    print(f"attribution {n_attr} | media {n_media} | dup {n_dup} | fit-eligible {len(elig)}",
          flush=True)

    # register-blind: BoW logistic, wire claims vs gated-FALSE CN claims
    cn = [json.loads(l) for l in open(SRC / "eval/data/urn_runs/c2_audit/ab_gate_new.jsonl")]
    cn_false = [r["claim"] for r in cn if r["band"] == "false"]
    wire = [r[claim_col] for r in elig]
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
        import numpy as np
        X_txt = wire + cn_false
        y = np.array([1] * len(wire) + [0] * len(cn_false))
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        X = vec.fit_transform(X_txt)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, p)
        print(f"register-blind AUC (wire vs CN-false, 5-fold oof): {auc:.3f}", flush=True)
    except ImportError:
        auc = None
        print("sklearn unavailable — register check skipped", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{k: r.get(k) for k in
                   ["post_id", "handle", claim_col, "type", "tag_attr", "tag_media",
                    "tag_dup", "fit_eligible"]} for r in kept]
                 ).write_parquet(OUT / "screens.parquet")
    json.dump({"n_claims": len(rows), "n_posts": df["post_id"].n_unique(),
               "attr": n_attr, "media": n_media, "dup": n_dup,
               "fit_eligible": len(elig), "register_auc": auc,
               "cn_false_n": len(cn_false), "claim_col": claim_col},
              open(OUT / "summary.json", "w"), indent=1)

    # 30-claim audit draw
    import random
    random.Random(20260824).shuffle(elig)
    with open(OUT / "audit_30.jsonl", "w") as f:
        for r in elig[:30]:
            f.write(json.dumps({"handle": r.get("handle"), "post_id": r["post_id"],
                                "claim": r[claim_col]}, ensure_ascii=False) + "\n")
    print("audit_30.jsonl written", flush=True)


if __name__ == "__main__":
    main()
