"""CN layer 4 — claim-level clustering across tweets (design agreed 2026-07-23).

One cluster = one underlying claim. Blocking (shared normalized citations + rare-token
keys) -> idf-weighted char-trigram Jaccard on candidate pairs only -> union-find.
No LLM anywhere. Representative per cluster drives hydration order.

  uv run python -m eval.scripts.build_eval.cn_cluster

Output: eval/data/community_notes/cn_gold_clusters.parquet (= l3 + cluster_id,
cluster_size, is_rep) + printed stats. Mega-clusters (>200) flagged for inspection.
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

SRC = Path("eval/data/community_notes/cn_gold_l3.parquet")
OUT = Path("eval/data/community_notes/cn_gold_clusters.parquet")

SIM_EDGE = 0.58          # idf-weighted trigram Jaccard for a same-claim edge
SIM_EDGE_CITED = 0.25    # relaxed bar when the pair shares >=2 citations + a rare token
RARE_DF_MAX = 500        # token df ceiling to count as "rare" (improbable keyword)
ULTRA_DF_MAX = 20        # a single shared ultra-rare token suffices as candidate key
BLOCK_CAP = 40           # all-pairs only within blocks up to this size
BIG_BLOCK_LINKS = 8      # in larger blocks, each member links to hub + k spread members

_STOP = set("""the and this that with from have not was are for is of to in it on no there
has been they their his her him she he you your who what when where which would could
should does did doing were will can may might been being a an as at by be or if so but
about into over after before more most other some such only own same than too very just
video image photo post tweet claim claims false true misleading context none sources
source says said show shows shown account user people https http com www x t co""".split())


def norm(s: str) -> str:
    s = re.sub(r"https?://\S+", "", (s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'àâçéèêëîïôùûüœæ-]", " ", s)).strip()[:500]


def norm_url(u: str) -> str | None:
    u = u.rstrip('.,);]"’').lower()
    m = re.match(r"https?://(?:www\.)?([^\s]+)", u)
    if not m:
        return None
    u = m.group(1)
    # keep identity-bearing query ids (youtube), else strip query/fragment
    if "youtube.com/watch" in u:
        vid = re.search(r"[?&]v=([\w-]+)", u)
        return f"youtube.com/watch?v={vid.group(1)}" if vid else None
    u = u.split("?")[0].split("#")[0].rstrip("/")
    return u if len(u) > 12 else None    # drop bare domains — not claim identifiers


def trigrams(t: str) -> set:
    t = re.sub(r"\s+", " ", t)
    return {t[i:i + 3] for i in range(len(t) - 2)} if len(t) >= 3 else {t}


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
    summaries = df["summary"].to_list()
    norms = [norm(s) for s in summaries]
    print(f"{n} notes; normalizing + trigram/idf pass...", flush=True)

    tri = [trigrams(t) for t in norms]
    tri_df = Counter()
    for s in tri:
        tri_df.update(s)
    idf = {g: log(n / c) for g, c in tri_df.items()}
    w = [sum(idf[g] for g in s) for s in tri]   # per-note total trigram weight

    def sim(i, j):
        inter = tri[i] & tri[j]
        if not inter:
            return 0.0
        wi = sum(idf[g] for g in inter)
        return wi / (w[i] + w[j] - wi) if (w[i] + w[j] - wi) > 0 else 0.0

    # ---- blocking keys ----
    urls = [set(filter(None, (norm_url(u) for u in re.findall(r"https?://\S+", s or ""))))
            for s in summaries]
    url_block = defaultdict(list)
    for i, us in enumerate(urls):
        for u in us:
            url_block[u].append(i)

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

    # ---- candidate pairs ----
    def block_pairs(members):
        if len(members) <= BLOCK_CAP:
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    yield members[a], members[b]
        else:  # star + spread: hub gets everyone; consecutive chain adds local links
            hub = members[0]
            for m in members[1:]:
                yield hub, m
            step = max(1, len(members) // (BIG_BLOCK_LINKS * len(members) // len(members)))
            for k in range(0, len(members) - 1, 1):
                if k % max(1, len(members) // (BIG_BLOCK_LINKS * 10)) == 0:
                    yield members[k], members[k + 1]

    pair_src = defaultdict(int)   # packed pair -> flags (1=url, 2=rare, 4=ultra)
    for u, members in url_block.items():
        if len(members) < 2:
            continue
        for a, b in block_pairs(members):
            pair_src[(min(a, b) << 32) | max(a, b)] |= 1
    rare_hits = Counter()
    for x, members in rare_block.items():
        if len(members) < 2 or len(members) > BLOCK_CAP:
            continue
        ultra = tok_df[x] <= ULTRA_DF_MAX
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                key = (min(members[a], members[b]) << 32) | max(members[a], members[b])
                if ultra:
                    pair_src[key] |= 4
                else:
                    rare_hits[key] += 1
    for key, c in rare_hits.items():
        if c >= 2:
            pair_src[key] |= 2
    cands = [k for k, f in pair_src.items() if f]
    print(f"candidate pairs: {len(cands)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- scoring + union ----
    dsu = DSU(n)
    # exact-dup edges are free
    seen_txt = {}
    for i, t in enumerate(norms):
        if t in seen_txt:
            dsu.union(seen_txt[t], i)
        else:
            seen_txt[t] = i
    edges = 0
    shared2 = lambda i, j: len(urls[i] & urls[j]) >= 2
    for key in cands:
        i, j = key >> 32, key & 0xFFFFFFFF
        if dsu.find(i) == dsu.find(j):
            continue
        s = sim(i, j)
        f = pair_src[key]
        if s >= SIM_EDGE or (s >= SIM_EDGE_CITED and (f & 1) and (f & 6) and shared2(i, j)):
            dsu.union(i, j)
            edges += 1
    print(f"accepted edges: {edges} ({time.time()-t0:.0f}s)", flush=True)

    # ---- clusters + representatives ----
    roots = [dsu.find(i) for i in range(n)]
    csize = Counter(roots)
    tiers = df["tier"].to_list()
    ncite = [len(u) for u in urls]
    slen = [len(s or "") for s in summaries]
    best = {}
    for i, r in enumerate(roots):
        k = (tiers[i] == "gold", ncite[i], slen[i])
        if r not in best or k > best[r][0]:
            best[r] = (k, i)
    is_rep = [best[r][1] == i for i, r in enumerate(roots)]

    # Generic-correction clusters ("The video is AI-generated" + explainer link) share
    # wording but NOT a claim — the identity lives in the media, which text can't see.
    # Mark them non-collapsible so hydration keeps every member (mega-cluster
    # inspection, 2026-07-23).
    stripped = [re.sub(r"\s+", " ", norms[i]).strip() for i in range(n)]
    generic_root = {r: len(stripped[best[r][1]]) < 70 for r in csize}
    collapsible = [not generic_root[r] for r in roots]

    out = df.with_columns(
        pl.Series("cluster_id", roots),
        pl.Series("cluster_size", [csize[r] for r in roots]),
        pl.Series("is_rep", is_rep),
        pl.Series("collapsible", collapsible),
    )
    out.write_parquet(OUT)
    n_cl = len(csize)
    multi = sum(1 for c in csize.values() if c > 1)
    mega = sorted((c for c in csize.values() if c > 200), reverse=True)
    n_hydrate = sum(1 for i in range(n) if is_rep[i] or not collapsible[i])
    print(f"\nwrote {OUT}")
    print(f"{n} notes -> {n_cl} clusters ({multi} multi-note; mass reduction "
          f"{1 - n_cl/n:.1%}); largest: {csize.most_common(5)}")
    print(f"mega-clusters >200 (INSPECT): {mega}")
    print(f"non-collapsible (generic-correction) cluster members kept individually: "
          f"{sum(1 for c in collapsible if not c)}")
    print(f"hydration list: {n_hydrate} (reps + non-collapsible members; vs {n} raw) "
          f"| total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
