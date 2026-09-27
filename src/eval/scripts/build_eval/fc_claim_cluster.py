"""Cross-publisher claim clustering for fc-gold v2 admitted (Truth Odds program §1).

The same viral claim is checked by several publishers; CAL/VAL splits must be
disjoint at the CLAIM level or the validation leaks. Same machinery as cn_cluster:
rare-token blocking → idf-weighted char-trigram Jaccard → union-find (claims carry no
citation URLs, so text blocking only, plus exact-normalized dup edges).

Also emits code-level quality flags per row (deixis / fragment / question /
non-latin / exact-dup) and the balance tables the split design needs.

  uv run python -m eval.scripts.build_eval.fc_claim_cluster
Output: eval/data/fc_gold_v2_clustered.parquet (+ cluster_id, cluster_size,
cluster_pubs, q_flags)
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter, defaultdict
from math import log
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SRC = Path("eval/data/fc_gold_v2_admitted.parquet")
OUT = Path("eval/data/fc_gold_v2_clustered.parquet")

SIM_EDGE = 0.60
RARE_DF_MAX = 300
ULTRA_DF_MAX = 15
BLOCK_CAP = 40

_STOP = set("""the and this that with from have not was were are is of to in it on for
a an as at by be or if so but about into over after during before shows show said says
claim claims video image photo post viral people new old man woman covid year years""".split())

_DEIXIS = re.compile(r"^(?:a |an |this |the )?(?:video|image|photo(?:graph)?|picture|clip|footage|"
                     r"screenshot|audio|recording|meme|post|tweet)\b.{0,20}\b(?:shows?|showing|of|"
                     r"claims?|depicts?|captures?|features?)", re.I)
_QUESTION = re.compile(r"\?\s*$")
_FRAGMENT = re.compile(r"(?:\.\.\.|…)\s*$|^[a-z]")  # trailing ellipsis or lowercase start


def norm(s: str) -> str:
    s = re.sub(r"https?://\S+", "", (s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", s)).strip()[:400]


def trigrams(t: str) -> set:
    t = re.sub(r"\s+", " ", t)
    return {t[i:i + 3] for i in range(len(t) - 2)} if len(t) >= 3 else {t}


def latin_ratio(s: str) -> float:
    letters = [c for c in (s or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isascii() or c in "àâçéèêëîïôùûüœæÀÂÇÉÈÊËÎÏÔÙÛÜ") / len(letters)


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main():
    t0 = time.time()
    df = pl.read_parquet(SRC)
    n = len(df)
    texts = df["claim_text"].to_list()
    norms = [norm(t) for t in texts]

    # quality flags (code-detectable pathologies; agents audit the judgment-level ones)
    flags = []
    for t in texts:
        t = t or ""
        f = []
        if _DEIXIS.search(t):
            f.append("deixis")
        if _QUESTION.search(t):
            f.append("question")
        if _FRAGMENT.search(t):
            f.append("fragment")
        if latin_ratio(t) < 0.9:
            f.append("non-latin")
        if len(t) < 25:
            f.append("short")
        flags.append(",".join(f))

    tri = [trigrams(t) for t in norms]
    tri_df = Counter()
    for s in tri:
        tri_df.update(s)
    idf = {g: log(n / c) for g, c in tri_df.items()}
    w = [sum(idf[g] for g in s) for s in tri]

    def sim(i, j):
        inter = tri[i] & tri[j]
        if not inter:
            return 0.0
        wi = sum(idf[g] for g in inter)
        d = w[i] + w[j] - wi
        return wi / d if d > 0 else 0.0

    tok_df = Counter()
    toks = []
    for t in norms:
        ts = {x for x in t.split() if len(x) > 3 and x not in _STOP and not x.isdigit()}
        toks.append(ts)
        tok_df.update(ts)
    rare_block = defaultdict(list)
    for i, ts in enumerate(toks):
        for x in ts:
            if tok_df[x] <= RARE_DF_MAX:
                rare_block[x].append(i)

    pair_hits = Counter()
    ultra_pairs = set()
    for x, members in rare_block.items():
        if len(members) < 2 or len(members) > BLOCK_CAP:
            continue
        ultra = tok_df[x] <= ULTRA_DF_MAX
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                key = (min(members[a], members[b]) << 32) | max(members[a], members[b])
                if ultra:
                    ultra_pairs.add(key)
                else:
                    pair_hits[key] += 1
    cands = ultra_pairs | {k for k, c in pair_hits.items() if c >= 2}
    print(f"{n} rows; candidate pairs {len(cands)} ({time.time()-t0:.0f}s)", flush=True)

    dsu = DSU(n)
    seen_txt = {}
    for i, t in enumerate(norms):
        if t in seen_txt:
            dsu.union(seen_txt[t], i)
        else:
            seen_txt[t] = i
    edges = 0
    for key in cands:
        i, j = key >> 32, key & 0xFFFFFFFF
        if dsu.find(i) != dsu.find(j) and sim(i, j) >= SIM_EDGE:
            dsu.union(i, j)
            edges += 1
    print(f"edges {edges} ({time.time()-t0:.0f}s)", flush=True)

    roots = [dsu.find(i) for i in range(n)]
    csize = Counter(roots)
    pubs = df["publisher_site"].to_list()
    cpubs = defaultdict(set)
    for i, r in enumerate(roots):
        cpubs[r].add(pubs[i])

    out = df.with_columns(
        pl.Series("cluster_id", roots),
        pl.Series("cluster_size", [csize[r] for r in roots]),
        pl.Series("cluster_pubs", [len(cpubs[r]) for r in roots]),
        pl.Series("q_flags", flags),
    )
    out.write_parquet(OUT)

    multi = sum(1 for c in csize.values() if c > 1)
    xpub = sum(1 for r, ps in cpubs.items() if len(ps) > 1)
    print(f"\nwrote {OUT}")
    print(f"{n} rows -> {len(csize)} claim clusters | multi-row {multi} | "
          f"CROSS-PUBLISHER clusters {xpub} (the split-leak risk) | largest {csize.most_common(3)}")
    fl = Counter()
    for f in flags:
        for x in (f.split(",") if f else []):
            fl[x] += 1
    print("quality flags:", dict(fl))
    # balance tables for split design
    g = out.filter(pl.col("veracity").is_not_null()).with_columns(
        pl.coalesce(pl.col("claim_date"), pl.col("review_date")).str.slice(0, 4).alias("yr"))
    print("\nclass x year (rows):")
    print(g.pivot(values="review_url", index="yr", on="veracity",
                  aggregate_function="len").sort("yr"))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
