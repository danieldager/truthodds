"""Timeline capture -> claims parquet: THE ingest path (TH2).

One command takes a zeerover NDJSON export off the browser extension (`capture/`,
ported into this repo 2026-08-25) and carries it to the claims parquet the verify
loop consumes. Owns the upstream half of the general-pool chain; the eval-stratum
build (`build_eval/general_pool_build.py`) imports these stages rather than
duplicating them, so both paths screen and extract identically.

  land   — validate exported x_capture_*.ndjson, dedup against the corpus, copy in
  screen — STRICT qualify screen (public affairs + stakes; anecdote, mockery,
           rhetoric, self-promo out) over every unscreened capture; resumable
           (general_pool_screen.call(strict=True) -> screen_strict.jsonl). The
           eval build passes screen="v1" to keep its original pool.
  chain  — qualifying posts -> extract_tweet_claims -> normalize_tweet_claims ->
           build_verify_input, --voice user except curated outlet handles.
           INCREMENTAL: only posts absent from the cumulative verify_input are
           chained; each batch keeps its own provenance json.

Modes (paid work is opt-in — pre-run checklist: docs/run_checklist.md):
  (default)  plan  — land + counts + cost projection. NO paid calls.
  --smoke    screen a 200-tweet sample + chain 20 posts into *_smoke artifacts,
             print the claims and the measured cost. Bounded, ~$0.01.
  --go       full screen + full incremental chain.

  cd src && uv run python -m eval.scripts.claim_sourcing.ingest_timeline_capture \
      [--from ~/Downloads] [--smoke | --go] [--concurrency 12]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import polars as pl

SRC = Path(__file__).resolve().parents[3]
CORPUS = SRC / "eval/data/tweet_corpus/general_pool"
TC = SRC / "eval/data/tweet_corpus"
RUNS = SRC / "eval/data/urn_runs/general_pool"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SEED = 20260824  # same draw as the general-pool build, so smokes are comparable
SMOKE_SCREEN_N = 200
SMOKE_CHAIN_N = 20
# Which qualify screen defines "qualifying": "strict" (production, 2026-08-26) or
# "v1" (the original public-subject-matter screen the eval pool was built on).
SCREEN = "strict"

# Measured unit costs — eval/data/run_ledger.md (screen $0.03 / 1,053 calls,
# 2026-08-24 build; chain $0.062 / 543 user posts = extract $0.026 + normalize
# $0.036, 2026-08-26 noise-floor runs — the 08-24 figure of $0.037/586 had
# missed the normalize pass). Projection only; the chain scripts report actuals.
COST_SCREEN = 0.03 / 1053
COST_CHAIN = 0.062 / 543

# Established news organizations present in the captures -> --voice outlet.
# Everything else is a general user account (the deployment population).
OUTLET_HANDLES = {
    "BFMTV", "bfmbusiness", "Europe1", "franceinfo", "LCI", "Le_Figaro",
    "afpfr", "ReutersBiz", "AJENews", "RMCsport", "ActuFoot_", "nexta_tv",
    "dohanews",
}

# The simplified-schema fields the ingest requires. Captures from the June 2026
# extension build predate image_urls / retweeted_by / topic, so those are read
# with .get() and never required.
REQUIRED_FIELDS = ("id", "full_text", "screen_name", "lang")


# ------------------------------------------------------------------- stage: land

def _read_ndjson(path: Path) -> tuple[list[dict], int]:
    """Parse one capture file. Returns (valid rows, count of unusable lines)."""
    rows, bad = [], 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if any(d.get(k) is None for k in REQUIRED_FIELDS):
            bad += 1
            continue
        rows.append(d)
    return rows, bad


def stage_land(sources: list[Path], corpus: Path = CORPUS) -> dict:
    """Copy exported captures into the corpus, reporting new vs already-held rows.

    Files are copied WHOLE and unmodified — the export is the provenance artifact.
    A file already in the corpus (by name) is skipped, not re-copied.
    """
    corpus.mkdir(parents=True, exist_ok=True)
    held = {r["id"] for r in load_pool(corpus, filtered=False)}
    landed, stats = [], {"files": 0, "rows": 0, "new": 0, "bad": 0, "skipped": 0}
    for src in sources:
        dst = corpus / src.name
        if dst.exists():
            stats["skipped"] += 1
            print(f"  skip (already landed): {src.name}", flush=True)
            continue
        rows, bad = _read_ndjson(src)
        if not rows:
            print(f"  SKIP (no usable rows): {src.name}", flush=True)
            continue
        new = sum(1 for r in rows if r["id"] not in held)
        held |= {r["id"] for r in rows}
        shutil.copy2(src, dst)
        landed.append(dst)
        stats["files"] += 1
        stats["rows"] += len(rows)
        stats["new"] += new
        stats["bad"] += bad
        print(f"  landed {src.name}: {len(rows)} rows | {new} new | {bad} unusable",
              flush=True)
    if not sources:
        print("  no capture files found to land", flush=True)
    return stats


# ------------------------------------------------------------------ pool + screen

def load_pool(corpus: Path = CORPUS, filtered: bool = True) -> list[dict]:
    """Every unique capture in the corpus, newest file last, first id wins.

    `filtered` applies the extraction-eligibility floor the general-pool build has
    always used: en/fr only, text longer than 15 chars — plus no ads (`promoted`,
    a flag captures before 2026-08-26 lack).
    """
    rows, seen = [], set()
    for f in sorted(glob.glob(str(corpus / "x_capture_*.ndjson"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not d.get("id") or d["id"] in seen:
                continue
            seen.add(d["id"])
            rows.append(d)
    if not filtered:
        return rows
    return [r for r in rows if r.get("lang") in ("en", "fr")
            and len(r.get("full_text") or "") > 15 and not r.get("promoted")]


def _screen_files(runs: Path, screen: str) -> tuple[Path, ...]:
    if screen == "strict":
        return (runs / "screen_strict.jsonl",)
    return (runs / "screen_smoke.jsonl", runs / "screen_all.jsonl")


def screened(runs: Path = RUNS, screen: str = SCREEN) -> list[dict]:
    """Every verdict on record for this screen version, deduped by tweet id."""
    rows: dict[str, dict] = {}
    for p in _screen_files(runs, screen):
        if p.exists():
            for line in open(p):
                d = json.loads(line)
                if d.get("qualifies") is not None:
                    rows[d["id"]] = d
    return list(rows.values())


def stage_screen(concurrency: int, limit: int = 0, corpus: Path = CORPUS,
                 runs: Path = RUNS, screen: str = SCREEN) -> Path:
    """Qualify screen over unscreened captures. Resumable; one short call each.

    `limit` > 0 screens only that many (fixed-seed draw) — the smoke path.
    """
    from eval.scripts.build_eval.general_pool_screen import call  # the same judge
    runs.mkdir(parents=True, exist_ok=True)
    out = _screen_files(runs, screen)[-1]
    done = {d["id"] for d in screened(runs, screen)}
    pool = load_pool(corpus)
    todo = [r for r in pool if r["id"] not in done]
    if limit and len(todo) > limit:
        random.Random(SEED).shuffle(todo)
        todo = todo[:limit]
    print(f"screen: pool {len(pool)} | already {len(done)} | todo {len(todo)}", flush=True)
    if todo:
        lock = threading.Lock()
        fh = open(out, "a")
        n = [0]
        t0 = time.time()

        def work(r):
            # A quote post's own text is often a fragment ("unbelievable") that only
            # qualifies read against the quoted post, so the screen sees both; the
            # quote extraction prompt then decides what the author actually asserts.
            text = r["full_text"]
            if r.get("quoted_text"):
                text += f"\n\n[Quoted post by @{r.get('quoted_handle')}]: {r['quoted_text']}"
            try:
                j = call(text, strict=(screen == "strict"))
            except Exception as e:  # noqa: BLE001
                j = {"qualifies": None, "category": "error", "reason": str(e)[:120]}
            with lock:
                fh.write(json.dumps({"id": r["id"], "lang": r["lang"],
                                     "screen_name": r.get("screen_name"),
                                     "operation": r.get("operation"),
                                     "is_quote": bool(r.get("quoted_text")),
                                     "text": r["full_text"], **j},
                                    ensure_ascii=False) + "\n")
                n[0] += 1
                if n[0] % 100 == 0:
                    rate = n[0] / max((time.time() - t0) / 60, .01)
                    print(f"  {n[0]}/{len(todo)} | {rate:.0f}/min | "
                          f"ETA {(len(todo)-n[0])/max(rate,1):.1f}m", flush=True)

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(ex.map(work, todo))
        fh.close()
    return out


# ------------------------------------------------------------------ stage: chain

def _posts_df(rows: list[dict], raw: dict[str, dict]) -> pl.DataFrame:
    """Screen rows -> the canonical posts parquet the extraction chain reads.

    Media travels even though extraction runs text-only (--no-images): captures
    from the current extension build carry photo URLs, and build_verify_input's
    contract is that media context reaches the claims parquet.
    """
    def img(i):
        return list(raw.get(i, {}).get("image_urls") or [])
    return pl.DataFrame({
        "post_id": [r["id"] for r in rows],
        "cell": ["general_pool"] * len(rows),
        "domain": ["x.com"] * len(rows),
        "handle": [r.get("screen_name") or "" for r in rows],
        "url": [f"https://x.com/{r.get('screen_name')}/status/{r['id']}" for r in rows],
        "created_at": [raw.get(r["id"], {}).get("created_at") for r in rows],
        "lang": [r["lang"] for r in rows],
        "text": [r["text"] for r in rows],
        "n_images": [len(img(r["id"])) for r in rows],
        "image_urls": [img(r["id"]) for r in rows],
        "n_videos": [0] * len(rows),
        "video_urls": [[] for _ in rows],
        "screen_category": [r.get("category") for r in rows],
        # Sampling frame: HomeTimeline = For You, HomeLatestTimeline = Following,
        # SearchTimeline etc. = side pool; `account` = whose feed it came from.
        "operation": [r.get("operation") for r in rows],
        "account": [raw.get(r["id"], {}).get("account") for r in rows],
        "captured_at": [raw.get(r["id"], {}).get("captured_at") for r in rows],
        "topic": [raw.get(r["id"], {}).get("topic") for r in rows],
        "is_reply": [bool(raw.get(r["id"], {}).get("is_reply")) for r in rows],
        "reply_to": [raw.get(r["id"], {}).get("reply_to") for r in rows],
        "conversation_id": [raw.get(r["id"], {}).get("conversation_id") for r in rows],
        # Engagement at capture time + author followers (captures from 2026-08-26
        # 11:00 on; null before). Names match build_verify_input's POST_COLS.
        **{k: [raw.get(r["id"], {}).get(k) for r in rows]
           for k in ("like_count", "retweet_count", "reply_count", "quote_count",
                     "view_count", "followers")},
        # A retweet is exported AS its original post (real author, real id); this
        # is the handle whose retweet surfaced it, null otherwise. Captures before
        # 2026-08-25 lack the field.
        "retweeted_by": [raw.get(r["id"], {}).get("retweeted_by") for r in rows],
        # Quote posts: link + the quoted post as extraction context (--voice quote).
        "quotes": [raw.get(r["id"], {}).get("quotes") for r in rows],
        "quoted_handle": [raw.get(r["id"], {}).get("quoted_handle") for r in rows],
        "quoted_text": [raw.get(r["id"], {}).get("quoted_text") for r in rows],
        "is_quote": [bool(raw.get(r["id"], {}).get("quotes")) for r in rows],
    })


def _manifest_path(ck: Path) -> Path:
    """build_verify_input's default posts-manifest name for a claims parquet."""
    return ck.with_name(ck.stem + "_posts_manifest.parquet")


def chained_posts(ck: Path) -> set[str]:
    """Every post already put through the chain for this voice.

    Read from the POSTS MANIFEST, not the claims parquet: build_verify_input emits
    claim rows only for posts that yielded claims, and ~26% of qualifying posts
    yield none (status skip_no_claims). Keying resume off the claims parquet would
    re-extract — and re-pay for — those posts on every subsequent run.
    """
    man = _manifest_path(ck)
    if man.exists():
        return set(pl.read_parquet(man)["post_id"].to_list())
    if ck.exists():
        return set(pl.read_parquet(ck)["post_id"].to_list())
    return set()


def _merge(path: Path, new: pl.DataFrame, key: str) -> pl.DataFrame:
    """Append a batch to a cumulative parquet, first row per key wins."""
    merged = (pl.concat([pl.read_parquet(path), new], how="diagonal_relaxed")
              if path.exists() else new)
    merged = merged.unique(subset=[key], keep="first")
    merged.write_parquet(path)
    return merged


def _run_chain(tag: str, posts: pl.DataFrame, voice: str, concurrency: int) -> Path:
    """extract -> normalize -> build_verify_input for one batch. Returns its parquet."""
    posts_path = TC / f"{tag}_posts.parquet"
    ck = TC / f"{tag}_verify_input.parquet"
    posts.write_parquet(posts_path)
    ex_parq, ex_html = TC / f"{tag}_extracted.parquet", TC / f"{tag}_extracted.html"
    nm_html = TC / f"{tag}_normalized.html"
    run = lambda cmd: subprocess.run(cmd, cwd=SRC, check=True)
    run([sys.executable, "eval/scripts/claim_sourcing/extract_tweet_claims.py",
         "-i", str(posts_path), "-o", str(ex_parq), "--html", str(ex_html),
         "--no-images", "--model", MODEL, "--voice", voice,
         "--concurrency", str(concurrency)])
    run([sys.executable, "eval/scripts/claim_sourcing/normalize_tweet_claims.py",
         "--payload", str(ex_html), "-o", str(nm_html), "--model", MODEL,
         "--voice", voice, "--concurrency", str(concurrency)])
    run([sys.executable, "eval/scripts/claim_sourcing/build_verify_input.py",
         "--normalized", str(nm_html), "--posts", str(posts_path), "-o", str(ck),
         "--extract-model", MODEL, "--normalize-model", MODEL])
    return ck


# Frame of a capture: the sampling process that put it in front of us. Both frames
# go through the identical gate + extract + normalize and land in ONE file
# (timeline_urn.parquet); `frame` is the covariate that keeps them apart.
#   feed        - an account's For You timeline, tagged with the logged-in account
#   search_june - the June search-derived draw (untagged, ~85 days older)
# Anything else (profile timelines, opened threads, search or Explore DURING a
# tagged session) is captured and screened but never chained: it is what Daniel
# chose to look at, not what a feed served.
def frame_of(op, account) -> str | None:
    if account and op in URN_OPS:
        return "feed"
    if account is None:
        return "search_june"
    return None


# The urn is FEED EXPOSURE. A session that also browsed profiles, opened threads,
# searched or hit Explore lands those posts in the corpus and screens them (kept,
# nothing is lost), but they are not chained: they are not what the account's feed
# served. Widen this set to fold a source back in — resume then picks them up.
URN_OPS = {"HomeTimeline"}


def frames(corpus: Path = CORPUS) -> dict[str, str | None]:
    """post id -> frame, for every capture in the corpus."""
    return {r["id"]: frame_of(r.get("operation"), r.get("account")) for r in load_pool(corpus)}


def qualifying(corpus: Path = CORPUS, runs: Path = RUNS, screen: str = SCREEN) -> list[dict]:
    """Screen rows that qualified and belong to a chained frame, ordered by tweet id."""
    fr = frames(corpus)
    return sorted([dict(r, frame=fr[r["id"]]) for r in screened(runs, screen)
                   if r["qualifies"] and fr.get(r["id"])],
                  key=lambda r: r["id"])


def write_urn(corpus: Path = CORPUS):
    """timeline_urn.parquet — ONE file, every claim off the same gate + chain.

    Sources are the urn_{voice}_verify_input cumulative parquets and nothing else.
    (general_*_verify_input is frozen: it holds the Aug-24 claims the eval build
    reads, produced by an older chain — never mix the two.)
    """
    fr = frames(corpus)
    parts = []
    for voice in ("quote", "user", "outlet"):
        f = TC / f"urn_{voice}_verify_input.parquet"
        if f.exists():
            parts.append(pl.read_parquet(f).with_columns(pl.lit(voice).alias("voice")))
    if not parts:
        return
    urn = (pl.concat(parts, how="diagonal_relaxed")
             .with_columns(pl.col("post_id").map_elements(lambda i: fr.get(i),
                                                          return_dtype=pl.String).alias("frame")))
    urn.write_parquet(TC / "timeline_urn.parquet")
    print(f"urn:   {urn.height} claims ({urn['checkworthy'].sum()} checkworthy) from "
          f"{urn['post_id'].n_unique()} posts -> {TC/'timeline_urn.parquet'}", flush=True)
    for r in (urn.group_by("frame").agg(pl.len().alias("claims"),
                                        pl.col("checkworthy").sum().alias("cw"),
                                        pl.col("post_id").n_unique().alias("posts"))
                 .sort("frame").iter_rows(named=True)):
        print(f"       {str(r['frame']):12} {r['claims']:5d} claims | {r['cw']:5d} cw "
              f"| {r['posts']:4d} posts", flush=True)


def stage_chain(concurrency: int, smoke: int = 0, batch: str = "",
                corpus: Path = CORPUS, runs: Path = RUNS, screen: str = SCREEN):
    """Qualifying posts -> claims parquet, incrementally.

    Posts already present in a voice's cumulative verify_input are skipped, so a
    new capture day costs only its own extraction. `smoke` > 0 chains that many
    posts into throwaway *_smoke artifacts and leaves the cumulative parquets
    untouched. Voices: quote (quote posts, quoted text as context) / user /
    outlet — see extract_tweet_claims.py --voice.
    """
    qual = qualifying(corpus, runs, screen)
    raw = {r["id"]: r for r in load_pool(corpus)}
    print(f"chain: qualifying {len(qual)}/{len(screened(runs, screen))} "
          f"(en {sum(1 for r in qual if r['lang']=='en')} "
          f"fr {sum(1 for r in qual if r['lang']=='fr')})", flush=True)

    batch = batch or date.today().strftime("%Y%m%d")
    # Route by voice: quote posts (quoted text on hand) -> the quote prompt, then the
    # curated outlets, then everyone else. One post, one voice.
    is_q = lambda r: bool(raw.get(r["id"], {}).get("quoted_text"))
    groups = {"quote": [r for r in qual if is_q(r)],
              "user": [r for r in qual if not is_q(r) and r["screen_name"] not in OUTLET_HANDLES],
              "outlet": [r for r in qual if not is_q(r) and r["screen_name"] in OUTLET_HANDLES]}

    if smoke:
        # The checkpoint is on what --go WOULD chain: draw from not-yet-chained
        # posts only, `smoke` user posts plus a few quote posts through their own
        # prompt, so a new capture day is what gets eyeballed, not the old pool.
        for voice, n in (("user", smoke), ("quote", min(10, smoke))):
            done = chained_posts(TC / f"urn_{voice}_verify_input.parquet")
            rows = [r for r in groups[voice] if r["id"] not in done]
            if not rows:
                print(f"\nsmoke[{voice}]: nothing unchained to sample", flush=True)
                continue
            pick = random.Random(SEED).sample(rows, min(n, len(rows)))
            ck = _run_chain(f"urn_smoke_{voice}_{batch}", _posts_df(pick, raw),
                            voice, concurrency)
            df = pl.read_parquet(ck)
            print(f"\nsmoke[{voice}]: {len(pick)} posts -> {df.height} claims from "
                  f"{df['post_id'].n_unique()} posts | checkworthy {df['checkworthy'].sum()}",
                  flush=True)
            for r in df.head(25).iter_rows(named=True):
                mark = "✓" if r["checkworthy"] else "·"
                print(f"  {mark} [{r['type']}] {r['claim'][:120]}")
            print(f"  artifacts: {ck}", flush=True)
        print("\nCheckpoint: review these claims before running --go.", flush=True)
        return

    for voice, rows in groups.items():
        ck = TC / f"urn_{voice}_verify_input.parquet"
        done = chained_posts(ck)
        todo = [r for r in rows if r["id"] not in done]
        print(f"chain[{voice}]: {len(rows)} qualifying | {len(done)} already chained "
              f"| {len(todo)} todo", flush=True)
        if todo:
            batch_ck = _run_chain(f"urn_{voice}_{batch}", _posts_df(todo, raw),
                                  voice, concurrency)
            new = pl.read_parquet(batch_ck)
            merged = _merge(ck, new, "claim_id")
            # the manifest carries the no-claim posts, so it is what resume reads
            _merge(_manifest_path(ck), pl.read_parquet(_manifest_path(batch_ck)), "post_id")
            print(f"chain[{voice}]: +{new.height} claims -> {merged.height} total",
                  flush=True)
    write_urn(corpus)


# ------------------------------------------------------------------- stage: plan

def stage_plan(corpus: Path = CORPUS, runs: Path = RUNS, screen: str = SCREEN):
    """Counts + cost projection from measured unit costs. No paid calls."""
    pool = load_pool(corpus)
    scr = screened(runs, screen)
    done = {d["id"] for d in scr}
    unscreened = [r for r in pool if r["id"] not in done]
    qual = qualifying(corpus, runs, screen)
    qrate = (sum(1 for d in scr if d["qualifies"]) / len(scr)) if scr else 0.44

    chained = set()
    for voice in ("quote", "user", "outlet"):
        chained |= chained_posts(TC / f"urn_{voice}_verify_input.parquet")
    todo_chain = len([r for r in qual if r["id"] not in chained])
    # Unscreened captures that will qualify, at the measured rate. Only the
    # urn-eligible ones get chained, so only those carry a chain cost.
    pending_urn = [r for r in unscreened if r.get("operation") in URN_OPS and r.get("account")]
    proj_chain = todo_chain + int(len(pending_urn) * qrate)

    print(f"\n== ingest plan ==  (screen: {screen})")
    print(f"  corpus            {len(pool)} captures (en/fr, >15 chars)")
    print(f"  screened          {len(scr)} | qualify rate {qrate:.0%}")
    print(f"  to screen         {len(unscreened)}  -> ~${len(unscreened)*COST_SCREEN:.3f}"
          f"  ({len(pending_urn)} of them feed posts)")
    print(f"  to chain          {proj_chain} posts -> ~${proj_chain*COST_CHAIN:.3f}")
    print(f"  TOTAL projection  ~${len(unscreened)*COST_SCREEN + proj_chain*COST_CHAIN:.3f}")
    print(f"  claims already in the urn: "
          f"{pl.read_parquet(TC/'timeline_urn.parquet').height if (TC/'timeline_urn.parquet').exists() else 0}")
    print("  frame (account × operation):",
          Counter((r.get("account"), r.get("operation")) for r in pool).most_common(8))
    if scr:
        print("  screen categories:",
              Counter(d["category"] for d in scr).most_common(8))
    print("\n  --smoke to sample-screen + chain 20 posts; --go for the full run.")
    print("  Pre-run checklist: docs/run_checklist.md · ledger: eval/data/run_ledger.md\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="src_dir", default=str(Path.home() / "Downloads"),
                    help="directory holding exported x_capture_*.ndjson (default: ~/Downloads)")
    ap.add_argument("--no-land", action="store_true", help="skip the land stage")
    ap.add_argument("--smoke", action="store_true",
                    help=f"sample-screen {SMOKE_SCREEN_N} + chain {SMOKE_CHAIN_N} posts")
    ap.add_argument("--go", action="store_true", help="full screen + full incremental chain")
    ap.add_argument("--batch", default="", help="batch tag for this run's artifacts (default: today)")
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()

    if not args.no_land:
        print("== land ==")
        stage_land(sorted(Path(args.src_dir).glob("x_capture_*.ndjson")))

    if args.go:
        stage_screen(args.concurrency)
        stage_chain(args.concurrency, batch=args.batch)
    elif args.smoke:
        stage_screen(args.concurrency, limit=SMOKE_SCREEN_N)
        stage_chain(args.concurrency, smoke=SMOKE_CHAIN_N, batch=args.batch)
    stage_plan()


if __name__ == "__main__":
    main()
