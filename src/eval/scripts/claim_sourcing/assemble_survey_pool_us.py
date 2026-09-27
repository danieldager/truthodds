"""Assemble the US-only survey pool (Daniel 2026-09-24; left-false refill branch 2026-09-25).

Default behaviour keeps the US posts of the 2026-09-21 core pool (656 posts, gate + Pro labels)
and refills from the 2026-09-24 Flash consensus chain (372 US core posts / 396 claims, 2-of-3
votes at claim check and gate) with posts whose claims ALL pass the frozen instrument, ranked by
recency x log likes. Writes one survivors parquet (same columns as
survey_pool_survivors_2026-09-21.parquet, plus `source` = pool656 | refill and `gate_votes`), a
merged labels json, a merged dossier jsonl, so build_survey_selection_page.py runs unchanged.

  uv run python -m eval.scripts.claim_sourcing.assemble_survey_pool_us --scratch <dir> --refill-scored <parquet>

LEFT-FALSE branch (Daniel 2026-09-25): when --base-pool / --extra-scored are given, ADD to an
existing assembled pool the posts of a targeted left-leaning slice whose claims the frozen
instrument FLAGS false (score <= boundary). Keeps every post with at least one flagged claim (and
all that post's surviving claims), tags them `--source-tag` (default pool_leftfalse), merges their
labels / second opinions / dossiers, re-ranks the whole pool, and writes dated 2026-09-25 files.

  uv run python -m eval.scripts.claim_sourcing.assemble_survey_pool_us \\
      --base-pool <2026-09-24 pool> --extra-scored A.parquet B.parquet --extra-posts Ap.parquet Bp.parquet \\
      --extra-labels A.json B.json --extra-second As.json Bs.json --extra-doss A.jsonl B.jsonl \\
      --out-parquet <..2026-09-25.parquet> --out-labels <..> --out-second <..> --out-doss <..>
$0.
"""
from __future__ import annotations
import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[3]
DATA = SRC / "eval/data/community_notes"
OLD = DATA / "survey_pool_survivors_2026-09-21.parquet"
OLD_LABELS = DATA / "survey_pool656_labels_2026-09-22.json"
REFILL_CLAIMS = DATA / "survey_refill_claims_2026-09-24.parquet"
DOSS_OLD = SRC / "eval/data/urn_runs/cn_survey/results_pool500.jsonl"
DOSS_NEW = SRC / "eval/data/urn_runs/cn_survey/results_pool_refill.jsonl"
OUT = DATA / "survey_pool_us_2026-09-24.parquet"
OUT_LABELS = DATA / "survey_pool_us_labels_2026-09-24.json"
OUT_DOSS = SRC / "eval/data/urn_runs/cn_survey/results_pool_us.jsonl"


def _rerank(pool: pd.DataFrame) -> pd.DataFrame:
    """Re-rank the whole pool by recency x log likes, one rank per post."""
    pp = pool.drop_duplicates("post_id").set_index("post_id")
    ep = pp.created_at.astype("int64"); ll = pp.likes.fillna(0).map(math.log1p)
    score = 0.5 * (ep - ep.min()) / (ep.max() - ep.min() + 1) + 0.5 * (ll - ll.min()) / (ll.max() - ll.min() + 1e-9)
    rank = {pid: i + 1 for i, pid in enumerate(score.sort_values(ascending=False).index)}
    pool["post_rank"] = pool.post_id.map(rank); pool["claim_rank"] = pool.post_rank
    pool["score"] = pool.post_id.map(score); pool["selected_top500"] = True
    return pool.sort_values(["post_rank", "claim_idx_in_post"])


def leftfalse(a) -> None:
    """Add flagged left-leaning posts to an existing assembled pool; write dated 2026-09-25 files."""
    base = pd.read_parquet(a.base_pool); base["post_id"] = base.post_id.astype(str)
    cols = list(base.columns)
    base_posts = set(base.post_id)

    new_rows = []
    for scored_p, posts_p in zip(a.extra_scored, a.extra_posts):
        sc = pd.read_parquet(scored_p); sc["post_id"] = sc.post_id.astype(str)
        flagged_posts = set(sc.groupby("post_id").flag.agg(lambda s: s.fillna(False).any()).loc[lambda x: x].index)
        # keep every surviving claim of a flagged post; never re-add a post already in the base pool
        sc = sc[sc.post_id.isin(flagged_posts) & ~sc.post_id.isin(base_posts)]
        posts = pd.read_parquet(posts_p); posts["post_id"] = posts.post_id.astype(str)
        posts = posts.drop_duplicates("post_id").set_index("post_id")
        meta_cols = [c for c in cols if c in posts.columns and c != "post_id"]
        for r in sc.itertuples():
            row = {c: None for c in cols}
            row["post_id"] = r.post_id
            row["claim"] = r.claim
            row["claim_idx_in_post"] = int(r.claim_idx_in_post)
            row["subtopic"] = r.subtopic
            row["gate_votes"] = int(r.gate_votes) if pd.notna(r.gate_votes) else None
            row["source"] = a.source_tag
            if r.post_id in posts.index:
                for c in meta_cols:
                    row[c] = posts.loc[r.post_id, c]
            new_rows.append(row)

    new = pd.DataFrame(new_rows, columns=cols) if new_rows else pd.DataFrame(columns=cols)
    pool = pd.concat([base, new], ignore_index=True)
    pool = pool.drop_duplicates(["post_id", "claim"])
    pool = _rerank(pool)
    a.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(a.out_parquet)

    # labels: base + extra, filtered to the pool's (post_id, claim) pairs
    keep_pairs = set(zip(pool.post_id, pool.claim))
    merged = [s for s in json.load(open(a.base_labels))["survivors"]
              if (str(s["post_id"]), s["claim"]) in keep_pairs]
    seen = {(str(s["post_id"]), s["claim"]) for s in merged}
    for lp in a.extra_labels:
        for s in json.load(open(lp))["survivors"]:
            k = (str(s["post_id"]), s["claim"])
            if k in keep_pairs and k not in seen:
                merged.append(s); seen.add(k)
    json.dump({"survivors": merged}, open(a.out_labels, "w"), ensure_ascii=False)

    # second opinion: base claims dict + extra claims dicts
    so = json.load(open(a.base_second))
    for sp in a.extra_second:
        so["claims"].update(json.load(open(sp))["claims"])
    json.dump(so, open(a.out_second, "w"), ensure_ascii=False)

    # dossiers: back up the current results_pool_us, then concat base + extras. When base and out
    # are the same file, read the base from the just-made backup so the "w" truncation cannot eat it.
    same = a.base_doss.resolve() == a.out_doss.resolve()
    if a.out_doss.exists() and a.backup_doss:
        shutil.copy(a.out_doss, a.backup_doss)
    base_src = a.backup_doss if (same and a.backup_doss) else a.base_doss
    with a.out_doss.open("w") as f:
        for line in base_src.open():
            f.write(line)
        for dp in a.extra_doss:
            for line in Path(dp).open():
                f.write(line)

    n_new_posts = new.post_id.nunique() if len(new) else 0
    print(f"base pool {base.post_id.nunique()} posts / {len(base)} claims | added {n_new_posts} left-false posts / {len(new)} claims "
          f"(source {a.source_tag}) | pool {pool.post_id.nunique()} posts / {len(pool)} claims | labels {len(merged)} | second {len(so['claims'])}")
    print(f"wrote {a.out_parquet}\n      {a.out_labels}\n      {a.out_second}\n      {a.out_doss}"
          + (f"\n      backup {a.backup_doss}" if a.backup_doss else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, help="dir holding refill_labels_flash.json and refill_core_posts.parquet")
    ap.add_argument("--refill-scored", type=Path, help="scored parquet of the refill claims (flag column)")
    ap.add_argument("--max-claims", type=int, default=720)
    # left-false additive branch
    ap.add_argument("--base-pool", type=Path, help="existing assembled pool parquet to add to (triggers left-false branch)")
    ap.add_argument("--base-labels", type=Path, default=OUT_LABELS)
    ap.add_argument("--base-second", type=Path, default=DATA / "survey_pool_us_second_opinion_2026-09-24.json")
    ap.add_argument("--base-doss", type=Path, default=OUT_DOSS)
    ap.add_argument("--extra-scored", type=Path, nargs="*", default=[])
    ap.add_argument("--extra-posts", type=Path, nargs="*", default=[])
    ap.add_argument("--extra-labels", type=Path, nargs="*", default=[])
    ap.add_argument("--extra-second", type=Path, nargs="*", default=[])
    ap.add_argument("--extra-doss", type=Path, nargs="*", default=[])
    ap.add_argument("--source-tag", default="pool_leftfalse")
    ap.add_argument("--out-parquet", type=Path, default=DATA / "survey_pool_us_2026-09-25.parquet")
    ap.add_argument("--out-labels", type=Path, default=DATA / "survey_pool_us_labels_2026-09-25.json")
    ap.add_argument("--out-second", type=Path, default=DATA / "survey_pool_us_second_opinion_2026-09-25.json")
    ap.add_argument("--out-doss", type=Path, default=OUT_DOSS)
    ap.add_argument("--backup-doss", type=Path, default=SRC / "eval/data/urn_runs/cn_survey/results_pool_us_2026-09-24.jsonl")
    a = ap.parse_args()

    if a.base_pool is not None:
        leftfalse(a)
        return

    old = pd.read_parquet(OLD); old["post_id"] = old.post_id.astype(str)
    old_lab = json.load(open(OLD_LABELS))["survivors"]
    country = {}
    for s in old_lab:
        country.setdefault(str(s["post_id"]), set()).add(s["country"])
    us_old = [p for p, c in country.items() if c == {"United States"}]
    old = old[old.post_id.isin(us_old)].copy()
    old["source"] = "pool656"; old["gate_votes"] = 3

    rc = pd.read_parquet(a.refill_scored); rc["post_id"] = rc.post_id.astype(str)
    all_pass = rc.groupby("post_id").flag.agg(lambda s: (~s.astype(bool)).all())
    keep_new = set(all_pass[all_pass].index)
    posts = pd.read_parquet(a.scratch / "refill_core_posts.parquet"); posts["post_id"] = posts.post_id.astype(str)
    posts = posts.drop_duplicates("post_id").set_index("post_id")
    claims = pd.read_parquet(REFILL_CLAIMS); claims["post_id"] = claims.post_id.astype(str)
    claims = claims[claims.post_id.isin(keep_new)]
    meta_cols = [c for c in old.columns if c in posts.columns]
    new = claims[["post_id", "claim", "claim_idx_in_post", "subtopic", "gate_votes"]].join(posts[meta_cols], on="post_id")
    new["source"] = "refill"
    for c in old.columns:
        if c not in new.columns:
            new[c] = None
    pool = pd.concat([old, new[old.columns.tolist() + ["source", "gate_votes"]] if "source" not in old.columns else new[old.columns]], ignore_index=True)

    # rank on recency x log likes over the whole pool, cap the refill at max_claims total
    pp = pool.drop_duplicates("post_id").set_index("post_id")
    ep = pp.created_at.astype("int64"); ll = pp.likes.fillna(0).map(math.log1p)
    score = 0.5 * (ep - ep.min()) / (ep.max() - ep.min() + 1) + 0.5 * (ll - ll.min()) / (ll.max() - ll.min() + 1e-9)
    n_old = len(old)
    budget = max(0, a.max_claims - n_old)
    new_posts = pp[pp.source == "refill"].index
    ranked = score.loc[new_posts].sort_values(ascending=False)
    chosen, n = [], 0
    for pid in ranked.index:
        k = int((pool.post_id == pid).sum())
        if n + k > budget:
            continue
        chosen.append(pid); n += k
    pool = pool[(pool.source == "pool656") | pool.post_id.isin(chosen)].copy()
    pp = pool.drop_duplicates("post_id").set_index("post_id")
    rank = {pid: i + 1 for i, pid in enumerate(score.loc[pp.index].sort_values(ascending=False).index)}
    pool["post_rank"] = pool.post_id.map(rank); pool["claim_rank"] = pool.post_rank
    pool["score"] = pool.post_id.map(score); pool["selected_top500"] = True
    pool = pool.sort_values(["post_rank", "claim_idx_in_post"])
    pool.to_parquet(OUT)

    new_lab = json.load(open(a.scratch / "refill_labels_flash.json"))["survivors"]
    keep_pairs = set(zip(pool.post_id, pool.claim))
    merged = [s for s in old_lab if (str(s["post_id"]), s["claim"]) in keep_pairs] + \
             [s for s in new_lab if (str(s["post_id"]), s["claim"]) in keep_pairs]
    json.dump({"survivors": merged}, open(OUT_LABELS, "w"), ensure_ascii=False)
    with OUT_DOSS.open("w") as f:
        for src in (DOSS_OLD, DOSS_NEW):
            for line in src.open():
                f.write(line)
    print(f"old US posts {old.post_id.nunique()} claims {n_old} | refill all-pass posts {len(keep_new)} of {len(all_pass)}, "
          f"chosen {len(chosen)} posts / {n} claims | pool {pool.post_id.nunique()} posts / {len(pool)} claims | labels {len(merged)}")
    print(f"wrote {OUT}\n      {OUT_LABELS}\n      {OUT_DOSS}")


if __name__ == "__main__":
    main()
