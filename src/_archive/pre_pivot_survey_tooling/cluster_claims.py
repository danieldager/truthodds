"""Theme-cluster the extracted claims with the project's MiniLM model (CPU, local).

Embeds every atomic claim with EMBEDDING_MODEL, runs a small spherical k-means
(cosine), and labels each cluster by its most central claims + dominant language.
No external API, no new dependency (numpy ships with sentence-transformers).

    uv run python -m scripts.cluster_claims --csv <results.csv> --k 10
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


def kmeans_cosine(X: np.ndarray, k: int, iters: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(len(X), k, replace=False)].copy()
    assign = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        sims = X @ cent.T                  # (N,k) cosine, X & cent unit-norm
        new = sims.argmax(1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            members = X[assign == j]
            if len(members):
                c = members.mean(0)
                n = np.linalg.norm(c)
                cent[j] = c / n if n else cent[j]
    return assign, cent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    claims, langs = [], []
    for r in rows:
        if r["is_checkable"] == "True" and r["claims"]:
            for c in r["claims"].split(" | "):
                c = c.strip()
                if c:
                    claims.append(c)
                    langs.append(r["lang"])
    print(f"Embedding {len(claims)} claims with {EMBEDDING_MODEL} (CPU)...")

    model = SentenceTransformer(EMBEDDING_MODEL)
    X = model.encode(claims, normalize_embeddings=True, show_progress_bar=False,
                     batch_size=64).astype(np.float32)

    assign, cent = kmeans_cosine(X, args.k)
    print(f"\n{args.k} theme clusters (by size):\n")
    order = [j for j, _ in Counter(assign.tolist()).most_common()]
    for j in order:
        idx = np.where(assign == j)[0]
        sims = X[idx] @ cent[j]
        central = idx[np.argsort(-sims)][:3]
        lang_mix = Counter(langs[i] for i in idx).most_common(3)
        lang_str = " ".join(f"{l}:{n}" for l, n in lang_mix)
        print(f"── cluster {j}  ({len(idx)} claims | {lang_str})")
        for i in central:
            print(f"     • {claims[i][:130]}")
        print()


if __name__ == "__main__":
    main()
