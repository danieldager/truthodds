"""C2 (Log-Odds Sprint) — build the CN FALSE stratum of the tweet fit corpus.

Fork-1 recommendation from docs/tweet_fit_corpus.md, executed: gold tier (CRH,
misleading-class), CLUSTER REPRESENTATIVES only, English, text hydrated, minus
media-only notes, minus satire-only notes, minus notes younger than 30 days at
snapshot. Extraction is the PRODUCTION chain, invoked as subprocesses so nothing
is reimplemented: extract_tweet_claims -> normalize_tweet_claims ->
build_verify_input, all DeepSeek-V4-Flash text-only (the confirmed config; the
chain scripts' own defaults are wrong).

Post-chain screens (CF-probe lessons, clog/220826): attribution-form claims
TAGGED for exclusion from the fit (the tool grades the content Y, the note
grades the tweet — pair design means the bare content assertion survives as its
own row); media-locus regex tag; mixed-signal tag from the note's checkboxes
(missing-context without factual-error — the misleading-framing/true-core class
that sank CF's French stratum); claim-level near-dup dedup (CF shipped dups; we
don't); E1 cross-corpus leakage tag (claim near-dup vs fc-gold claim text).
Note text NEVER enters any pipeline input (leakage guard, Fork 4).

  uv run python -m eval.scripts.build_eval.cn_false_stratum --smoke   # 50 tweets
  uv run python -m eval.scripts.build_eval.cn_false_stratum --n 2000  # full (needs go)

Outputs under eval/data/tweet_corpus/: cn_false_posts.parquet (locked draw),
cn_false_claims.parquet (screened verify input), + the chain's manifest and
provenance files. Smoke uses the cn_false_smoke_* prefix.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
CN = SRC / "eval/data/community_notes"
OUT = SRC / "eval/data/tweet_corpus"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SNAPSHOT = datetime.date(2026, 7, 23)  # CN dump day (raw/ mtime; hydration same day)
E1_RESULTS = SRC / "eval/data/urn_runs/e1_ctx/results-00.jsonl"

MEDIA_RE = re.compile(
    r"\b(video|photo|image|footage|clip|picture|audio|recording|screenshot)s?\b.{0,40}"
    r"\b(show|shows|showing|depict|depicts|depicting|captur|of|from|is real|is fake|"
    r"is authentic|AI-generated|doctored|manipulated|edited)\b", re.I)


def _b(col):
    return pl.col(col).cast(pl.Int64, strict=False).fill_null(0)


def select(n: int, seed: int) -> pl.DataFrame:
    """Selection funnel -> locked posts draw in the extract_tweet_claims schema."""
    reps = pl.read_parquet(CN / "cn_gold_clusters.parquet").filter(pl.col("is_rep"))
    flags = ["misleadingFactualError", "misleadingManipulatedMedia",
             "misleadingOutdatedInformation", "misleadingMissingImportantContext",
             "misleadingUnverifiedClaimAsFact", "misleadingSatire"]
    tot = sum(_b(f) for f in flags)
    reps = reps.with_columns([
        (_b("misleadingManipulatedMedia").eq(1) & tot.eq(1)).alias("media_only"),
        (_b("misleadingSatire").eq(1) & tot.eq(1)).alias("satire_only"),
        (_b("misleadingMissingImportantContext").eq(1)
         & _b("misleadingFactualError").eq(0)
         & _b("misleadingUnverifiedClaimAsFact").eq(0)).alias("mixed_signal"),
        (pl.col("note_date") > SNAPSHOT - datetime.timedelta(days=30)).alias("too_young"),
    ])
    n0 = reps.height
    sel = reps.filter(~pl.col("media_only") & ~pl.col("satire_only") & ~pl.col("too_young"))
    n1 = sel.height

    # hydration store holds repeat attempts per tweetId — keep the last success
    hyd: dict[str, dict] = {}
    with open(CN / "hydrated.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if str(d.get("code")) == "200" and d.get("text"):
                hyd[str(d["tweetId"])] = d
    hdf = pl.DataFrame({"tweetId": list(hyd),
                        "text": [d["text"] for d in hyd.values()],
                        "author": [d.get("author") or "" for d in hyd.values()],
                        "hyd_lang": [d.get("lang") for d in hyd.values()],
                        "x_created_at": [d.get("created_at") for d in hyd.values()]})
    sel = sel.join(hdf, on="tweetId", how="inner")
    n2 = sel.height
    en = sel.filter(pl.col("hyd_lang") == "en")
    n3 = en.height
    print(f"funnel: reps {n0} -> media/satire/recency {n1} -> hydrated {n2} -> EN {n3}",
          flush=True)

    draw = en.sample(n=min(n, en.height), seed=seed)
    posts = draw.select([
        pl.col("tweetId").alias("post_id"),
        pl.lit("cn_false").alias("cell"),
        pl.lit("x.com").alias("domain"),
        pl.col("author").alias("handle"),
        (pl.lit("https://x.com/") + pl.col("author") + pl.lit("/status/") + pl.col("tweetId")).alias("url"),
        pl.col("x_created_at").alias("created_at"),
        pl.col("hyd_lang").alias("lang"),
        pl.col("text"),
        pl.lit(0).alias("n_images"),
        pl.lit([], dtype=pl.List(pl.Utf8)).alias("image_urls"),
        pl.lit(0).alias("n_videos"),
        pl.lit([], dtype=pl.List(pl.Utf8)).alias("video_urls"),
        # provenance for screens + folds (never fed to prompts)
        pl.col("noteId"), pl.col("cluster_id"), pl.col("cluster_size"),
        pl.col("note_date").cast(pl.Utf8), pl.col("mixed_signal"),
    ])
    return posts


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", (s or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def screen(claims_path: Path, posts: pl.DataFrame, out_path: Path):
    """Post-chain screens: tags only — nothing is silently dropped."""
    df = pl.read_parquet(claims_path)
    n0 = df.height
    df = df.with_columns([
        (pl.col("type") == "attribution").alias("attribution_form"),
        pl.col("claim").map_elements(lambda c: bool(MEDIA_RE.search(c or "")),
                                     return_dtype=pl.Boolean).alias("media_locus"),
    ])
    df = df.join(posts.select(["post_id", "noteId", "cluster_id", "cluster_size",
                               "note_date", "mixed_signal"]),
                 on="post_id", how="left")

    # claim-level near-dup dedup (within corpus, first-kept), then E1 leakage tag
    toks = [_tokens(c) for c in df["claim"].to_list()]
    dup = [False] * len(toks)
    kept: list[int] = []
    for i, t in enumerate(toks):
        for j in kept:
            if _jaccard(t, toks[j]) >= 0.85:
                dup[i] = True
                break
        else:
            kept.append(i)
    e1_toks = []
    with open(E1_RESULTS) as f:
        for line in f:
            e1_toks.append(_tokens(json.loads(line).get("claim_text", "")))
    leak = [any(_jaccard(t, e) >= 0.6 for e in e1_toks) if t else False for t in toks]
    df = df.with_columns([pl.Series("claim_dup", dup), pl.Series("e1_leak", leak)])

    fit = df.filter(pl.col("checkworthy") & ~pl.col("attribution_form")
                    & ~pl.col("media_locus") & ~pl.col("claim_dup") & ~pl.col("e1_leak"))
    print(f"screens: {n0} claims | checkworthy {df['checkworthy'].sum()} | "
          f"attribution {df['attribution_form'].sum()} | media_locus {df['media_locus'].sum()} | "
          f"mixed_signal {df['mixed_signal'].sum()} | dup {sum(dup)} | e1_leak {sum(leak)} | "
          f"FIT-ELIGIBLE {fit.height}", flush=True)
    df.write_parquet(out_path)
    return df, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="tweets to draw")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--smoke", action="store_true", help="50-tweet end-to-end smoke")
    ap.add_argument("--voice", default="user", choices=["outlet", "user"],
                    help="extraction/normalization prompt variant (user = ordinary-user "
                         "posts, the C2 default after the 2026-08-24 prompt A/B)")
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()
    n = 50 if args.smoke else args.n
    tag = "cn_false_smoke" if args.smoke else "cn_false"
    OUT.mkdir(parents=True, exist_ok=True)

    posts = select(n, args.seed)
    posts_path = OUT / f"{tag}_posts.parquet"
    posts.write_parquet(posts_path)
    print(f"locked draw: {posts.height} posts -> {posts_path}", flush=True)

    ex_html = OUT / f"{tag}_extracted.html"
    ex_parq = OUT / f"{tag}_extracted.parquet"
    nm_html = OUT / f"{tag}_normalized.html"
    ck_parq = OUT / f"{tag}_verify_input.parquet"
    run = lambda cmd: subprocess.run(cmd, cwd=SRC, check=True)
    run([sys.executable, "eval/scripts/claim_sourcing/extract_tweet_claims.py",
         "-i", str(posts_path), "-o", str(ex_parq), "--html", str(ex_html),
         "--no-images", "--model", MODEL, "--voice", args.voice,
         "--concurrency", str(args.concurrency)])
    run([sys.executable, "eval/scripts/claim_sourcing/normalize_tweet_claims.py",
         "--payload", str(ex_html), "-o", str(nm_html), "--model", MODEL,
         "--voice", args.voice, "--concurrency", str(args.concurrency)])
    run([sys.executable, "eval/scripts/claim_sourcing/build_verify_input.py",
         "--normalized", str(nm_html), "--posts", str(posts_path), "-o", str(ck_parq),
         "--extract-model", MODEL, "--normalize-model", MODEL])

    screen(ck_parq, posts, OUT / f"{tag}_claims.parquet")


if __name__ == "__main__":
    main()
