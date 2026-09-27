"""Embedding dedup of the cascade-surviving claims → novel-claim count (≈ Tier-1).

Takes the claims from posts that passed V∧R∧H (verifiable ∧ relevant ∧ harmful)
for a given extractor, embeds them with the project's planned Tier-1 model
(paraphrase-multilingual-MiniLM-L12-v2), and merges near-duplicates by cosine
similarity (union-find). The number of clusters = novel claims we'd actually
need to check. This also seeds the Tier-1 vector cache.

Reports cluster counts across thresholds (sensitivity) and dumps example merges
to eyeball, then writes the novel-claim representatives for the Tier-2 FCT step.

  uv run python -m eval.scripts.feed_study.dedup_claims \
      --dir eval/scripts/feed_study/data/sample1000 --extractor gpt-oss-120b --threshold 0.85
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

EMB_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _vrh_claims(d: Path, posts_path: Path, ext: str) -> pl.DataFrame:
    """Claims from V∧R∧H posts for the given extractor -> (post_id, claim_index, claim_text, lang)."""
    mp = pl.read_parquet(d / "master_post.parquet")
    t = "g" if "gpt" in ext else "q"
    keep = mp.filter(
        pl.col(f"verif_raw_{t}").fill_null(False)
        & pl.col(f"relev_raw_{t}").fill_null(False)
        & pl.col(f"harm_raw_{t}").fill_null(False)
    ).select("post_id", "lang")
    cl = pl.read_parquet(d / f"claimify_{ext}.parquet").select("post_id", "claims")
    ex = (cl.with_columns(pl.int_ranges(pl.col("claims").list.len()).alias("claim_index"))
            .explode(["claims", "claim_index"]).rename({"claims": "claim_text"})
            .filter(pl.col("claim_text").is_not_null() & (pl.col("claim_text").str.strip_chars() != "")))
    return ex.join(keep, on="post_id", how="inner")


def _components(sim: np.ndarray, thr: float) -> list[list[int]]:
    """Union-find connected components over pairs with cosine >= thr."""
    n = sim.shape[0]
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    iu = np.argwhere(np.triu(sim >= thr, k=1))
    for a, b in iu:
        union(int(a), int(b))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--posts", type=Path, default=None)
    ap.add_argument("--extractor", type=str, default="gpt-oss-120b")
    ap.add_argument("--threshold", type=float, default=0.85, help="headline cosine merge threshold")
    ap.add_argument("--show", type=int, default=12, help="example merged clusters to print")
    args = ap.parse_args()

    claims = _vrh_claims(args.dir, args.posts, args.extractor)
    texts = claims["claim_text"].to_list()
    n = len(texts)
    print(f"V∧R∧H claims for {args.extractor}: {n}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    sim = emb @ emb.T  # cosine (normalized)

    print("\nNovel-claim count by merge threshold (clusters = novel claims):")
    for thr in (0.75, 0.80, 0.85, 0.90, 0.95):
        comps = _components(sim, thr)
        merged = sum(1 for c in comps if len(c) > 1)
        print(f"  cosine>={thr:.2f}: {len(comps):4d} clusters  ({n}->{len(comps)}, "
              f"{n-len(comps)} merged away, {merged} multi-claim clusters)")

    comps = _components(sim, args.threshold)
    comps.sort(key=len, reverse=True)
    print(f"\nExample merges at cosine>={args.threshold} (largest clusters):")
    for c in comps[:args.show]:
        if len(c) > 1:
            print(f"  -- cluster of {len(c)}:")
            for i in c[:4]:
                print(f"       {texts[i][:90]}")

    # write novel representatives (MEDOID = member closest to the cluster centroid)
    # + the full member list, for the Tier-2 FCT step and LLM synthesis.
    rows = []
    rec = claims.to_dicts()
    for cid, c in enumerate(comps):
        if len(c) == 1:
            medoid = c[0]
        else:
            sub = sim[np.ix_(c, c)]
            medoid = c[int(sub.sum(axis=1).argmax())]  # max total similarity = most central
        rows.append({
            "cluster_id": cid,
            "representative": texts[medoid],  # medoid
            "lang": rec[medoid]["lang"],
            "n_members": len(c),
            "members": [texts[i] for i in c],
            "member_post_ids": [rec[i]["post_id"] for i in c],
        })
    out = pl.DataFrame(rows)
    out.write_parquet(args.dir / f"novel_claims_{args.extractor}.parquet")
    print(f"\nwrote {out.height} novel-claim MEDOID representatives (thr={args.threshold}) -> "
          f"{args.dir / f'novel_claims_{args.extractor}.parquet'}")


if __name__ == "__main__":
    main()
